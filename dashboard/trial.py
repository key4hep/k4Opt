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
            measured, (o.direction for o in results.problem.objectives), strict=False
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
