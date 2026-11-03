"""The Studies page: the studies of the publication the sidebar keeps (of its detector and Key4hep
stack) in one table, the latest published first, to sort by any column, search by name, problem
or detector, and open by selecting it."""

from __future__ import annotations

import streamlit as st

from k4opt.publication import CatalogEntry

import data
import labels
import tables
import widgets


def main(found: data.Source) -> None:
    st.title("Studies", anchor=False)
    st.caption(
        "Detector optimization studies published with `k4opt publish`. Click a column's heading to "
        "sort by it; select a study to open its trials, its Pareto front and how a candidate is "
        "evaluated."
    )
    catalog = found.catalog
    if catalog is None or not catalog.studies:
        data.show(found)
        return
    kept = set(st.session_state.get(data.KEPT, ()))  # of the detector and stack the sidebar keeps
    entries = [entry for entry in data.newest_first(catalog) if entry.id in kept]
    text = st.text_input(
        "Search",
        placeholder="Search by study, problem or detector",
        icon=":material/search:",
        label_visibility="collapsed",
    ).casefold()
    shown = [entry for entry in entries if _matches(entry, text)]
    if not shown:
        st.caption("No study matches.")
        return
    rows = [_row(entry) for entry in shown]
    picked = widgets.table(
        rows,
        "studies",
        hidden=("Id",),
        config={
            "Published": st.column_config.DatetimeColumn("Published", format="D MMM YYYY, HH:mm"),
            "Created": st.column_config.DatetimeColumn("Created", format="D MMM YYYY, HH:mm"),
        },
    )
    if picked is not None:
        st.switch_page("study.py", query_params={"study": str(rows[picked]["Id"])})


def _matches(entry: CatalogEntry, text: str) -> bool:
    said = f"{entry.name} {entry.problem or ''} {labels.detector(entry.geometry)}"
    return text in said.casefold()


def _row(entry: CatalogEntry) -> tables.Row:
    outcomes = entry.outcomes
    return {
        "Study": entry.name,
        "Detector": labels.detector(entry.geometry),
        "Problem": entry.problem,
        "Key4hep": entry.key4hep,
        "Published": entry.published,
        "Trials": entry.trials,
        "Feasible": outcomes["feasible"],
        "Infeasible": outcomes["infeasible"],
        "Without a result": sum(outcomes[o] for o in tables.WITHOUT_RESULT),
        "Running": outcomes["running"],
        "Parameters": entry.parameters,
        "Objectives": entry.objectives,
        "Constraints": entry.constraints,
        "Created": entry.created,
        "Id": entry.id,
    }


main(data.current())
