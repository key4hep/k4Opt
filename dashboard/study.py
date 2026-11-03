"""The Study page: one study, chosen by its id in the URL (``/study?study=<id>``), so that a link
names it: what it optimizes, how its trials ended, its trials in the plane of two axes, and what
it was made with."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import StudyResults
from k4opt.publication import CatalogEntry, PublicationUnavailableError
from k4opt.visualization import axis_choices, default_axes, trials_figure

import data
import tables

# the URL's parameters: the study by its id, and the figure's axes (a page script shows a bare
# string, so nothing here is a docstring but the module's)
STUDY, X, Y = "study", "x", "y"


def _chosen(
    label: str,
    options: list[str],
    key: str,
    default: str | None,
    format_func: Callable[[str], str],
    placeholder: str | None = None,
) -> str | None:
    """A choice kept in the URL's parameter ``key``, so that a link reopens it; a value the options
    do not hold gives ``default``."""
    held = st.query_params.get(key)
    current = held if held in options else default
    chosen = st.selectbox(
        label,
        options,
        index=None if current is None else options.index(current),
        format_func=format_func,
        placeholder=placeholder,
        key=f"choice-{key}",
    )
    if chosen is not None and chosen != held:
        st.query_params[key] = chosen
    return chosen


def main(found: data.Source) -> None:
    catalog = found.catalog
    if catalog is None or not catalog.studies or found.location is None:
        st.title("Study")
        data.show(found)
        return
    entries = {entry.id: entry for entry in _newest_first(catalog.studies)}
    requested = st.query_params.get(STUDY)
    if requested is not None and requested not in entries:
        st.warning(f"The catalog of {found.location} has no study `{requested}`; choose one.")
    names = {study: entry.name for study, entry in entries.items()}
    chosen = _chosen(
        "Study",
        list(entries),
        STUDY,
        None if requested is not None else next(iter(entries)),
        format_func=lambda study: names[study],
        placeholder="Choose a study",
    )
    if not chosen:
        return
    entry = entries[chosen]
    try:
        results = data.study(found.location, entry)
    except PublicationUnavailableError as exc:
        st.warning(
            f"The study cannot be read just now ({exc}); it is read again on your next interaction."
        )
        return
    except K4OptError as exc:
        st.error(f"The study's snapshot does not hold up: {exc}")
        return
    _render(entry, results)


def _newest_first(entries: tuple[CatalogEntry, ...]) -> list[CatalogEntry]:
    return sorted(entries, key=lambda entry: entry.published, reverse=True)


def _render(entry: CatalogEntry, results: StudyResults) -> None:
    st.title(entry.name)
    st.caption(
        f"Problem **{entry.problem or 'unnamed'}** · created {entry.created} · published "
        f"{entry.published} · study `{entry.id}` · snapshot `{entry.snapshot.sha256[:16]}`"
    )
    outcomes = entry.outcomes
    if outcomes["running"]:
        st.info(
            f"{outcomes['running']} trial(s) were running when this snapshot was published; "
            "publish the study again to see them end."
        )
    columns = st.columns(6)
    shown = [
        ("Trials", entry.trials),
        ("Feasible", outcomes["feasible"]),
        ("Infeasible", outcomes["infeasible"]),
        ("Without a result", sum(outcomes[o] for o in tables.WITHOUT_RESULT)),
        ("Running", outcomes["running"]),
        ("Pareto front", len(results.pareto)),
    ]
    for column, (label, value) in zip(columns, shown, strict=True):
        column.metric(label, value, border=True)

    st.subheader("What it optimizes")
    st.markdown("**Parameters**")
    st.dataframe(tables.parameters(results), hide_index=True)
    st.markdown("**Objectives**")
    st.dataframe(tables.objectives(results), hide_index=True)
    if results.problem.constraints:
        st.markdown("**Constraints**")
        st.dataframe(tables.constraints(results), hide_index=True)

    st.subheader("Trials")
    choices = list(axis_choices(results))
    default_x, default_y = default_axes(results)
    left, right = st.columns(2)
    with left:
        x = _chosen("Horizontal axis", choices, X, default_x, format_func=_axis_name)
    with right:
        y = _chosen("Vertical axis", choices, Y, default_y, format_func=_axis_name)
    try:
        figure = trials_figure(results, x, y, dark=st.context.theme.type == "dark")
    except K4OptError as exc:
        st.warning(str(exc))
    else:
        figure.update_layout(title=None)  # the page names the study
        st.plotly_chart(figure, theme="streamlit", key="trials")
    if results.pareto:
        st.markdown(f"**On the Pareto front** ({len(results.pareto)} trials)")
        st.dataframe(tables.trials(results, results.pareto), hide_index=True)
    with st.expander(f"Every trial ({len(results.trials)})"):
        st.dataframe(tables.trials(results), hide_index=True)
    with st.expander("What it was made with"):
        st.dataframe(tables.provenance(results), hide_index=True)
        st.dataframe(tables.evaluators(results), hide_index=True)
        st.caption(
            "Each trial's record stays in the study directory; the snapshot names it by its id "
            "and SHA-256, and holds no file of it."
        )


def _axis_name(key: str) -> str:
    kind, _, name = key.partition(":")
    return f"{name} ({kind})"


main(data.current())
