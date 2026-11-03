"""The Studies page: every study of the publication, the latest published first."""

from __future__ import annotations

import streamlit as st

import data
import tables


def main(found: data.Source) -> None:
    st.title("Studies")
    st.caption(
        "Detector optimization studies published with `k4opt publish`. Select a study to see "
        "its problem, its trials and its Pareto front."
    )
    catalog = found.catalog
    if catalog is None or not catalog.studies:
        data.show(found)
        return
    rows = tables.studies(catalog)
    event = st.dataframe(
        rows,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="studies",
        column_config={
            "Created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Published": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Running at publication": st.column_config.NumberColumn(
                help="Trials that were running when the study was published: publish it again "
                "to see them end."
            ),
            "Id": st.column_config.TextColumn(width="small"),
        },
    )
    selected = event.selection.rows if event is not None else []
    if selected:
        st.switch_page("study.py", query_params={"study": str(rows[selected[0]]["Id"])})


main(data.current())
