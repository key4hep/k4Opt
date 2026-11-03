"""The Study page: one study, chosen by its id in the URL (``/study?study=<id>``), so that a link
names it: how its trials ended, how a candidate is evaluated, what it optimizes, its trials in the
plane of two axes, and what it was made with."""

from __future__ import annotations

from collections.abc import Callable

import plotly.graph_objects as go
import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import StudyResults
from k4opt.publication import CatalogEntry, PublicationUnavailableError
from k4opt.visualization import (
    GraphNode,
    TaskGraph,
    axis_choices,
    default_axes,
    pipeline_figure,
    task_graph,
    trials_figure,
)

import data
import tables

# the URL's parameters: the study by its id, and the figure's axes (a page script shows a bare
# string, so nothing here is a docstring but the module's)
STUDY, X, Y = "study", "x", "y"
# the task selected, the stacks of samples drawn sample by sample (by key; ALL: every one), the
# study they are of, and a counter bumped after each click, so that the chart comes back without
# a selection and the next click on the same node is a click again
SELECTED, EXPANDED, OF, CHART = "tasks-selected", "tasks-expanded", "tasks-of", "tasks-chart"
ALL = "*"
# drag to pan, scroll to zoom, buttons for both and for going back to the whole graph
CHART_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtons": [["zoomIn2d", "zoomOut2d", "resetScale2d"]],
}


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
        st.warning(
            f"The catalog of {data.plain(found.location)} has no study "
            f"{data.plain(requested)}; choose one."
        )
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
            f"The study cannot be read just now ({data.plain(exc)}); it is read again on your "
            "next interaction."
        )
        return
    except K4OptError as exc:
        st.error(f"The study's snapshot does not hold up: {data.plain(exc)}")
        return
    _render(entry, results)


def _newest_first(entries: tuple[CatalogEntry, ...]) -> list[CatalogEntry]:
    return sorted(entries, key=lambda entry: entry.published, reverse=True)


def _render(entry: CatalogEntry, results: StudyResults) -> None:
    st.title(data.plain(entry.name), anchor=False)
    st.caption(  # an id is a UUID and a SHA-256 hex digits, checked: nothing in them formats
        f"Problem **{data.plain(entry.problem or 'unnamed')}** · created "
        f"{data.plain(entry.created)} · published {data.plain(entry.published)} · study "
        f"`{entry.id}` · snapshot `{entry.snapshot.sha256[:16]}`"
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

    _tasks(results)

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
        st.warning(data.plain(exc))
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


def _tasks(results: StudyResults) -> None:
    st.subheader("How a candidate is evaluated")
    state = st.session_state
    if state.get(OF) != results.id:  # another study: nothing of the last one stays selected
        state[OF], state[SELECTED], state[EXPANDED] = results.id, None, set()
    state.setdefault(CHART, 0)
    expanded: set[str] = state[EXPANDED]
    stack = ALL not in expanded
    graph = task_graph(results, stack=stack, expanded=expanded)
    stacks = [key for key, box in graph.samples.items() if box.is_stack]
    caption, each, again = st.columns([4, 1, 1], vertical_alignment="bottom")
    with caption:
        st.caption(
            "Left to right: the parameters (blue) and what each sets (dashed), the geometry, "
            "each sample (violet) with its stages, the evaluations (aqua), and the objectives "
            "and constraints (orange) they measure. Samples alike but for their options are "
            "stacked in one box: click it to draw each. A dotted task is a diagnostic: recorded, "
            "but no objective or constraint needs it. Times are medians over the trials."
        )
    with each:
        if st.button("Draw each sample", disabled=not stacks, width="stretch"):
            state[EXPANDED] = {ALL}
            st.rerun()
    with again:
        if st.button("Stack alike samples", disabled=not expanded, width="stretch"):
            state[EXPANDED] = set()
            st.rerun()
    try:
        figure = pipeline_figure(results, stack=stack, expanded=expanded, selected=state[SELECTED])
    except K4OptError as exc:
        st.warning(data.plain(exc))
        return
    event = st.plotly_chart(
        figure,
        theme="streamlit",
        key=f"tasks-{state[CHART]}",
        on_select="rerun",
        selection_mode="points",
        config=CHART_CONFIG,
    )
    clicked = _clicked(event, figure)
    if clicked is not None:
        node = next((n for n in graph.nodes if n.id == clicked), None)
        box = graph.samples.get(clicked if node is None else node.sample or "")
        if box is not None and box.is_stack and (node is None or node.samples):
            state[EXPANDED] = {*expanded, box.members[0]}  # a stack: draw each of its samples
        else:
            state[SELECTED] = clicked
        state[CHART] += 1
        st.rerun()
    _selected(results, graph)


def _clicked(event: object, figure: go.Figure) -> str | None:
    """The id of the node clicked, if the chart reports one: a point of the figure's last trace,
    the nodes, whose ``customdata[0]`` is the id."""
    nodes = len(figure.data) - 1
    selection = getattr(event, "selection", None) or {}
    for point in selection.get("points", []):
        index = point.get("point_index")
        if point.get("curve_number") == nodes and index is not None:
            return str(figure.data[nodes].customdata[index][0])
    return None


def _selected(results: StudyResults, graph: TaskGraph) -> None:
    """What the problem says of the task clicked, and how it went in each trial."""
    chosen = st.session_state.get(SELECTED)
    node: GraphNode | None = next((n for n in graph.nodes if n.id == chosen), None)
    if node is None:
        st.caption("Click a task for what it is and how it went in each trial.")
        return
    title, clear = st.columns([5, 1], vertical_alignment="bottom")
    with title:
        st.markdown(f"##### {data.plain(node.title)}")
        where = node.kind if node.sample is None else f"{node.kind} of {node.sample}"
        st.caption(data.plain(where + ("" if node.required else ", a diagnostic")))
    with clear:
        if st.button("Clear", width="stretch", key="tasks-clear"):
            st.session_state[SELECTED] = None
            st.session_state[CHART] += 1
            st.rerun()
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(tables.facts(node), hide_index=True)
    rows = tables.task_trials(results, node)
    with right:
        if rows:
            st.dataframe(rows, hide_index=True)
        else:
            st.caption("Its value in each trial is in the table of every trial below.")


def _axis_name(key: str) -> str:
    kind, _, name = key.partition(":")
    return f"{name} ({kind})"


main(data.current())
