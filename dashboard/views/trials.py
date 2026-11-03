"""The Trials tab: every trial of the study, those of the outcomes chosen; a trial selected opens
its sheet."""

from __future__ import annotations

import streamlit as st

from k4opt.optimization import StudyResults

import labels
import tables
import trial
import widgets


def render(results: StudyResults) -> None:
    counts = {outcome: n for outcome, n in results.outcomes.items() if n}
    shown = st.pills(
        "Outcome",
        list(counts),
        selection_mode="multi",
        default=list(counts),
        format_func=lambda outcome: f"{outcome} · {counts[outcome]}",
        key="trials-outcomes",
        label_visibility="collapsed",
    )
    rows = [row for row in tables.trials(results) if row["Outcome"] in shown]
    if not rows:
        st.caption("No trial has the outcomes chosen.")
        return
    st.caption(f"{labels.count(len(rows), 'trial')}; select one to open all it recorded.")
    picked = widgets.table(rows, "trials", helps=tables.helps(results), height=560)
    if picked is not None:
        trial.show(int(str(rows[picked]["Trial"])))
