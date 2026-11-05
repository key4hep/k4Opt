"""The Study page: one study, chosen in the sidebar and named in the URL (``/study?study=<id>``),
so that a link names it, in tabs across the top, the one open in the URL too (``&view=Trials``):
its *Overview* (how far it got, what it optimizes, how a candidate is evaluated), *Results*,
*Trade-offs*, *Trials* and *Analysis*; a trial opened from any of them is a sheet over the page
(``&trial=12``)."""

from __future__ import annotations

import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import StudyResults
from k4opt.publication import CatalogEntry, PublicationUnavailableError
from k4opt.visualization import names

import data
import labels
import trial
from views import analysis, overview, trade_offs, trials
from views import results as measured

TABS = ("Overview", "Results", "Trade-offs", "Trials", "Analysis")
PAGES = (overview.render, measured.render, trade_offs.render, trials.render, analysis.render)


def main(found: data.Source) -> None:
    catalog = found.catalog
    if catalog is None or not catalog.studies or found.location is None:
        st.title("Study")
        data.show(found)
        return
    entries = {entry.id: entry for entry in data.newest_first(catalog)}
    requested = st.query_params.get("study")
    if requested not in entries:
        st.warning(
            f"The catalog of {labels.plain(found.location)} has no study "
            f"{labels.plain(requested)}; choose one in the sidebar."
        )
        return
    entry = entries[requested]
    try:
        top = data.study(found.location, entry)
    except PublicationUnavailableError as exc:
        st.warning(
            f"The study cannot be read just now ({labels.plain(exc)}); it is read again on your "
            "next interaction."
        )
        return
    except K4OptError as exc:
        st.error(f"The study's snapshot does not hold up: {labels.plain(exc)}")
        return
    results, path = _descend(top)
    # a bar of tabs across the top of the page; only the open one runs, and the URL says which
    view = st.segmented_control(
        "View",
        TABS,
        default=TABS[0],
        required=True,
        key=trial.VIEW,
        bind="query-params",
        label_visibility="collapsed",
        width="stretch",
    )
    _header(entry)
    if path:
        _below(top, path)
    PAGES[TABS.index(view if view in TABS else TABS[0])](results)
    trial.sheet(results)


def _header(entry: CatalogEntry) -> None:
    """One muted line under the tabs: which study is shown, of which problem, from when."""
    st.caption(
        f"**{labels.plain(entry.name)}** · problem {labels.plain(entry.problem or 'unnamed')} · "
        f"{labels.plain(labels.detector(entry.geometry))} · {labels.plain(entry.key4hep)} · "
        f"published {labels.plain(labels.when(entry.published))}"
    )
    running = entry.outcomes["running"]
    if running:
        st.info(
            f"{labels.count(running, 'trial')} were running when this snapshot was published; "
            "publish the study again to see them end.",
            icon=":material/pending:",
        )


def _descend(top: StudyResults) -> tuple[StudyResults, list[int]]:
    """The study the URL leads to from ``top`` (``&inner=2.5``: the study of the level below trial
    2, then that of its trial 5), and the trials that lead there; as far as the trials named
    have studies below them."""
    results, path = top, []
    for number in trial.inner_path():
        held = {each.number: each for each in results.trials}.get(number)
        if held is None or held.inner is None:
            st.warning(f"Trial {number} has no study of a level below; the study above is shown.")
            break
        results, path = held.inner, [*path, number]
    if path != trial.inner_path():
        trial.open_study(path)
    return results, path


def _below(top: StudyResults, path: list[int]) -> None:
    """Which study of a level below is shown: the trials that lead to it, each with the values it
    fixes, and a way back up."""
    called = names(top.problem)
    steps, results = [], top
    for number in path:
        held = results.trial(number)
        fixed = [
            f"{called.parameter(name)} = {labels.number(value, unit(top, name))}"
            for name, value in held.candidate.items()
            if name in results.scope.searched
        ]
        steps.append(f"trial {number} ({', '.join(fixed)})")
        assert held.inner is not None
        results = held.inner
    up, said = st.columns([1, 5], vertical_alignment="center")
    up.button(
        "Up a level",
        icon=":material/arrow_upward:",
        on_click=trial.open_study,
        args=(path[:-1],),
        width="stretch",
    )
    searched = ", ".join(called.parameter(name) for name in results.scope.searched)
    said.caption(
        f"The study of level {results.scope.level} for {labels.plain(', then '.join(steps))}: it "
        f"tunes {labels.plain(searched)} for those values, and its best trial is theirs."
    )


def unit(results: StudyResults, name: str) -> str | None:
    return next(p.unit for p in results.problem.parameters if p.name == name)


main(data.current())
