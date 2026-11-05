"""The sheet of one trial, opened alike from every figure and table of the Study page: a dialog
over the page, named by the URL (``&trial=12``), so that a link opens it; closing it drops the
trial from the URL.

It shows what the trial measured and was made of first, then how its constraints, stages and
diagnostics went, and where its record is last, out of the way.
"""

from __future__ import annotations

import streamlit as st

from k4opt.optimization import StudyResults, TrialResult
from k4opt.visualization import names

import labels
import tables
import widgets

SHOWN = "trial"  # the URL's parameter of the trial open
INNER = "inner"  # the URL's parameter of the study of a level below shown: its trials, by "."
VIEW = "view"  # the Study page's tabs, by their key: a comparison opens the Trade-offs tab
# the color of each outcome's badge; every other outcome is a trial without a result
BADGES = {"feasible": "green", "infeasible": "orange", "running": "blue"}


def reference_key(results: StudyResults) -> str:
    """The key of the Trade-offs tab's reference trial, one for each study: another study's
    reference is not this one's."""
    return f"trade-reference-{results.id}"


def show(number: int) -> None:
    """Open trial ``number``: name it in the URL, and run the page again without any selection of
    the figure or table it was opened from."""
    st.query_params[SHOWN] = str(number)
    st.session_state[widgets.PICKS] = widgets.picks() + 1
    st.rerun()


def linked(results: StudyResults) -> int | None:
    """The trial the URL names, if the study has it."""
    held = st.query_params.get(SHOWN, "")
    numbers = {trial.number for trial in results.trials}
    return int(held) if held.isdigit() and int(held) in numbers else None


def sheet(results: StudyResults) -> None:
    """The sheet of the trial the URL names, if it names one of the study's; a name it lacks is
    dropped."""
    number = linked(results)
    if number is None:
        if SHOWN in st.query_params:
            del st.query_params[SHOWN]
        return
    dialog = st.dialog(f"Trial {number}", width="large", on_dismiss=_closed)
    dialog(_sheet)(results, results.trial(number))


def inner_path() -> list[int] | None:
    """The trials, from the top, whose studies of the level below lead to the study shown: none
    for the top study; ``None`` for a URL whose ``inner`` is no such path (a part that is no
    number), which leads nowhere rather than somewhere else."""
    held = st.query_params.get(INNER, "")
    if not held:
        return []
    parts = held.split(".")
    # an ASCII digit each: str.isdigit() takes digits int() does not ("²")
    if not all(part.isascii() and part.isdigit() for part in parts):
        return None
    return [int(part) for part in parts]


def open_study(path: list[int]) -> None:
    """Show the study the trials ``path`` lead to, from its Overview, with no trial open."""
    if path:
        st.query_params[INNER] = ".".join(map(str, path))
    elif INNER in st.query_params:
        del st.query_params[INNER]
    st.session_state[VIEW] = "Overview"
    _closed()


def _closed() -> None:
    if SHOWN in st.query_params:
        del st.query_params[SHOWN]


def _compare(results: StudyResults, number: int) -> None:
    """Make trial ``number`` the reference of the trade-off, and open its tab to show it."""
    st.session_state[reference_key(results)] = number
    st.session_state[VIEW] = "Trade-offs"
    _closed()


def _sheet(results: StudyResults, trial: TrialResult) -> None:
    called = names(results.problem)
    badges, action = st.columns([3, 2], vertical_alignment="center")
    with badges, st.container(horizontal=True, gap="small"):
        st.badge(trial.outcome, color=BADGES.get(trial.outcome, "red"))  # type: ignore[arg-type]
        if trial.number in results.pareto:
            st.badge("Pareto front", icon=":material/star:", color="blue")
        if trial.recovered:
            st.badge("finished by a resume", color="gray")
        if trial.elapsed_seconds is not None:
            st.badge(labels.duration(trial.elapsed_seconds), icon=":material/timer:", color="gray")
    with action:
        if trial.objectives is not None and st.button(
            "Compare with this trial",
            icon=":material/balance:",
            key=f"trial-compare-{trial.number}",
            on_click=_compare,
            args=(results, trial.number),
            help="Make it the reference of a trade-off, in the Trade-offs tab",
            width="stretch",
        ):
            st.rerun()  # from a dialog too: the whole page shows the new reference
    if trial.message:
        st.code(trial.message, language=None, wrap_lines=True)
    assessment = trial.assessment
    measured = () if assessment is None else assessment.objectives or ()
    arrows = {"minimize": "▼", "maximize": "▲"}
    tiles = [
        (
            f"{arrows[direction]} {called.objective(value.label)}",
            labels.number(value.value, value.unit, value.uncertainty),
            f"{value.label}, to {direction}",
        )
        for value, direction in zip(
            measured, (o.direction for o in results.scope.criteria.objectives), strict=False
        )
    ] + [
        (
            called.parameter(p.name),
            labels.number(trial.candidate[p.name], p.unit),
            f"{p.name}, a parameter",
        )
        for p in results.problem.parameters
        if p.name in trial.candidate
    ]
    for start in range(0, len(tiles), 4):
        columns = st.columns(4)
        for column, (label, value, said) in zip(columns, tiles[start : start + 4], strict=False):
            column.metric(labels.plain(label), value, help=labels.plain(said), border=True)
    if trial.inner is not None:
        _tuning(results, trial)
    limits = tables.trial_limits(results, trial)
    if limits:
        st.markdown("**Constraints**")
        widgets.table(limits)
    stages = tables.trial_stages(trial)
    if stages:
        st.markdown("**Stages**")
        widgets.table(
            stages,
            hidden=("Seconds",),
            config={"Time": st.column_config.TextColumn("Time", width="small")},
        )
    diagnostics = tables.trial_diagnostics(trial)
    if diagnostics:
        st.markdown("**Diagnostics that did not succeed**")
        widgets.table(diagnostics)
    with st.expander("Record", icon=":material/description:"):
        widgets.table(tables.trial_record(trial))


def _tuning(results: StudyResults, trial: TrialResult) -> None:
    """What the study of the level below found for ``trial``: its best trial, what that measured
    on the study's events, the values it gives; and a way into that study."""
    inner = trial.inner
    assert inner is not None
    called = names(results.problem)
    searched = ", ".join(called.parameter(name) for name in inner.scope.searched)
    st.markdown("**Tuned for this candidate**")
    found = tables.tuned(trial)
    said = f"Its study of {labels.plain(searched)} ran {labels.count(len(inner.trials), 'trial')}"
    if found is None or trial.selected is None:
        st.caption(f"{said} and found no feasible one, so this trial has no value.")
    else:
        best = int(str(trial.selected["trial"]))
        objective = labels.plain(called.objective(found.label))
        value = labels.number(found.value, found.unit, found.uncertainty)
        st.caption(
            f"{said}; the best, trial {best}, reached {objective} = {value} on that study's "
            "events. Its "
            "values are this trial's, and the values above were measured again, on new events: "
            "a best of many is optimistic on the events it was chosen on."
        )
    st.button(
        "Open its study",
        icon=":material/account_tree:",
        key=f"trial-inner-{trial.number}",
        on_click=open_study,
        args=([*(inner_path() or []), trial.number],),
        help="Show the study of the level below as a study of its own, with every tab",
    )
