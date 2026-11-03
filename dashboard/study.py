"""The Study page: one study, chosen in the sidebar and named in the URL (``/study?study=<id>``),
so that a link names it, in tabs across the top, the one open in the URL too (``&view=Trials``):
its *Overview* (how far it got, what it optimizes, how a candidate is evaluated), *Results*,
*Trade-offs*, *Trials* and *Analysis*; a trial opened from any of them is a sheet over the page
(``&trial=12``)."""

from __future__ import annotations

import streamlit as st

from k4opt.errors import K4OptError
from k4opt.publication import CatalogEntry, PublicationUnavailableError

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
        results = data.study(found.location, entry)
    except PublicationUnavailableError as exc:
        st.warning(
            f"The study cannot be read just now ({labels.plain(exc)}); it is read again on your "
            "next interaction."
        )
        return
    except K4OptError as exc:
        st.error(f"The study's snapshot does not hold up: {labels.plain(exc)}")
        return
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


main(data.current())
