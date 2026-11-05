"""The Analysis tab: how the trials went and how the parameters act on an objective, one view at
a time, both kept in the URL (``&objective=...&analysis=Slice``)."""

from __future__ import annotations

from collections.abc import Callable

import plotly.graph_objects as go
import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import StudyResults
from k4opt.visualization import (
    edf_figure,
    history_figure,
    parallel_figure,
    rank_figure,
    slice_figure,
)

import labels
import trial
import widgets

ANALYZED, SHOWN = "objective", "analysis"  # the URL's parameters: the objective and the view
SAID = {
    "History": "The objective in the order the trials were proposed; click a trial to open it.",
    "Parallel coordinates": "Each feasible trial is a line through its parameters, objectives "
    "and constraint margins. Drag along an axis to keep the trials within a range; click the "
    "range to drop it.",
    "Slice": "The objective against each parameter, colored by trial number; click a trial to "
    "open it.",
    "Rank": "Each pair of parameters, the feasible trials colored by how they rank in the "
    "objective; nothing is interpolated between them. Click a trial to open it.",
    "Distribution": "The share of the trials with a value of each objective at most each value.",
}


def render(results: StudyResults) -> None:
    objectives = [f"objective:{o.label}" for o in results.scope.criteria.objectives]
    # ranking the trials over pairs of parameters needs two parameters at least
    views = [view for view in SAID if view != "Rank" or len(results.problem.parameters) > 1]
    pick, show = st.columns([1, 2], vertical_alignment="bottom")
    with pick:
        objective = widgets.choice(
            "Objective",
            objectives,
            ANALYZED,
            objectives[0],
            format_func=lambda key: labels.axis(key, results),
        )
    with show:
        view = st.segmented_control(
            "View",
            views,
            default=views[0],
            required=True,
            key=SHOWN,
            bind="query-params",
        )
    view = view if view in views else views[0]
    dark = st.context.theme.type == "dark"
    draw: dict[str, tuple[Callable[[], go.Figure], str | None]] = {
        "History": (lambda: history_figure(results, objective, dark=dark), "history"),
        "Parallel coordinates": (lambda: parallel_figure(results, objective, dark=dark), None),
        "Slice": (lambda: slice_figure(results, objective, dark=dark), "slice"),
        "Rank": (lambda: rank_figure(results, objective, dark=dark), "rank"),
        "Distribution": (lambda: edf_figure(results, dark=dark), None),
    }
    make, clickable = draw[view]
    try:
        figure = make()
    except K4OptError as exc:
        st.warning(labels.plain(exc))
        return
    clicked = widgets.chart(figure, clickable)
    st.caption(SAID[view])
    if clicked is not None:
        trial.show(clicked)
