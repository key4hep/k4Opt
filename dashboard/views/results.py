"""The Results tab: the trials in the plane of two axes, and the trials on the Pareto front."""

from __future__ import annotations

import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import StudyResults
from k4opt.visualization import trials_figure

import labels
import tables
import trial
import widgets


def render(results: StudyResults) -> None:
    x, y, color = widgets.axes(results, color=True)
    try:
        shade = None if color == widgets.BY_OUTCOME else color
        figure = trials_figure(results, x, y, color=shade, dark=st.context.theme.type == "dark")
    except K4OptError as exc:
        st.warning(labels.plain(exc))
    else:
        figure.update_layout(title=None)  # the page names the study
        clicked = widgets.chart(figure, "trials")
        if clicked is not None:
            trial.show(clicked)
    if not results.pareto:
        return
    st.markdown(f"**Pareto front** · {labels.count(len(results.pareto), 'trial')}")
    st.caption("Select a trial to open it.")
    rows = tables.trials(results, results.pareto)
    picked = widgets.table(
        rows,
        "front",
        helps=tables.helps(results),
        hidden=("Outcome", "Pareto", "Message"),
        height=min(36 + 35 * len(rows), 420),
    )
    if picked is not None:
        trial.show(int(str(rows[picked]["Trial"])))
