"""The Overview tab, read top to bottom: how far the study got (its trials, its Pareto front, the
best value of each objective), its model (how a candidate is evaluated: the task graph, laid out
by Graphviz, and full screen on demand), what it optimizes, and, out of the way, what it was made
with."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import StudyResults
from k4opt.visualization import (
    COLLAPSED,
    GraphNode,
    TaskGraph,
    collapsed,
    names,
    pipeline_figure,
    task_graph,
)

import labels
import tables
import widgets

# the task selected, the stacks of samples drawn sample by sample (by key; ALL: every one), the
# study they are of, and a counter bumped after each click, so that the chart comes back without
# a selection and the next click on the same node is a click again
SELECTED, EXPANDED, OF, CHART = "tasks-selected", "tasks-expanded", "tasks-of", "tasks-chart"
ALL = "*"
FOLDED = "tasks-folded"  # the samples whose box is one node, by key
FULL = "tasks-full"  # whether the model is shown full screen
LARGE = 30  # tasks a graph has at most to be drawn task by task at first
BESTS = 3  # the objectives whose best value the status shows, at most
WIDE, TALL = 1300, 820  # pixels the model is drawn across, and at most high, full screen
# on the page the wheel scrolls the page: drag to pan, buttons to zoom; full screen it zooms
BUTTONS = [["zoomIn2d", "zoomOut2d", "resetScale2d"]]
ON_PAGE = {"scrollZoom": False, "displaylogo": False, "modeBarButtons": BUTTONS}
FULL_SCREEN = {"scrollZoom": True, "displaylogo": False, "modeBarButtons": BUTTONS}


def render(results: StudyResults) -> None:
    _status(results)
    _model(results)
    _problem(results)
    with st.expander("Provenance: what the study was made with", icon=":material/fingerprint:"):
        widgets.table(tables.provenance(results))
        widgets.table(tables.evaluators(results))
        st.caption(
            "Each trial's record stays in the study directory; the snapshot names it by its id "
            "and SHA-256, and holds no file of it."
        )


# --- how far it got ----------------------------------------------------------------------------


def _status(results: StudyResults) -> None:
    """How far the study got, in one strip: its trials and how they ended, its Pareto front, and
    the best value of each objective a feasible trial reached."""
    outcomes = results.outcomes
    ended = [f"{n} {outcome}" for outcome, n in outcomes.items() if n and outcome != "feasible"]
    bests = tables.bests(results)[:BESTS]
    values = [labels.number(b.value, b.unit) for b in bests]  # its uncertainty: in its trial
    with st.container(border=True, key="status"):
        columns = st.columns(2 + len(bests))
        with columns[0]:
            feasible = f"{outcomes['feasible']} feasible"
            _stat(
                "Trials",
                len(results.trials),
                " · ".join([feasible, *ended]) if ended else "all feasible",
            )
        with columns[1]:
            _stat(
                "Pareto front",
                len(results.pareto),
                f"of {outcomes['feasible']} feasible",
                "The study's own Pareto front, of its objectives and constraints",
            )
        for column, best, value in zip(columns[2:], bests, values, strict=True):
            with column:
                _stat(
                    f"Best {best.name}",
                    value,
                    f"trial {best.trial}",
                    f"The best ({best.direction}d) value a feasible trial reached of {best.label}; "
                    "the History view of the Analysis tab shows how the trials got there.",
                )


def _stat(label: str, value: object, note: str, said: str | None = None) -> None:
    """A number of the status: its ``label`` above it whole, however long (a metric's own label
    is cut short), and a ``note`` under it."""
    st.caption(labels.plain(label), help=None if said is None else labels.plain(said))
    st.metric(
        labels.plain(label),
        str(value),
        delta=note,
        delta_color="off",
        delta_arrow="off",
        label_visibility="collapsed",
    )


# --- the model ---------------------------------------------------------------------------------


def _model(results: StudyResults) -> None:
    """The task graph in a card of its own, left to right from what is set to what is measured,
    with what draws it otherwise. Samples alike are stacked, and a graph of more than
    :data:`LARGE` tasks shows each sample's box as one node at first; a click on a stack draws
    each of its samples, on a folded sample its tasks, on a task what it is and how it went."""
    state = st.session_state
    if state.get(OF) != results.id:  # another study: nothing of the last one stays
        state[OF], state[SELECTED], state[EXPANDED] = results.id, None, set()
        state[FOLDED], state[FULL] = None, False  # as large as it is: folded, or not, at first
    state.setdefault(CHART, 0)
    expanded: set[str] = state[EXPANDED]
    stack = ALL not in expanded
    try:
        whole = task_graph(results, stack=stack, expanded=expanded)
    except K4OptError as exc:
        st.warning(labels.plain(exc))
        return
    if state[FOLDED] is None:
        state[FOLDED] = set(whole.samples) if len(whole.nodes) > LARGE else set()
    folded: set[str] = state[FOLDED] & set(whole.samples)
    graph = collapsed(whole, folded)
    with st.container(border=True):
        title, full = st.columns([5, 1], vertical_alignment="center")
        with title:
            st.markdown(":material/account_tree: **Model**")
            st.caption(
                "How a candidate is evaluated, from the parameters to the objectives. Click a "
                "task for what it is and how it went; a stack of samples or a folded sample to "
                "draw what it holds."
            )
        with full:
            if st.button("Full screen", icon=":material/fullscreen:", width="stretch"):
                state[FULL] = True
        try:
            figure = pipeline_figure(  # in the page's colors, not the browser's preference
                results,
                stack=stack,
                expanded=expanded,
                selected=state[SELECTED],
                folded=folded,
                dark=st.context.theme.type == "dark",
            )
        except K4OptError as exc:
            st.warning(labels.plain(exc))
            return
        # no legend, nor its room above the graph: each task's hover says what it is
        figure.update_layout(showlegend=False, margin={"t": 8}, height=figure.layout.height - 28)
        event = st.plotly_chart(
            figure,
            theme="streamlit",
            key=f"tasks-{state[CHART]}",
            on_select="rerun",
            selection_mode="points",
            config=ON_PAGE,
        )
        clicked = _clicked(event, figure)
        if clicked is not None:
            node = next((n for n in graph.nodes if n.id == clicked), None)
            box = whole.samples.get(clicked if node is None else node.sample or "")
            if clicked.startswith(COLLAPSED):  # a folded sample: draw its tasks
                state[FOLDED] = folded - {clicked.removeprefix(COLLAPSED)}
            elif box is not None and box.is_stack and (node is None or node.samples):
                state[EXPANDED] = {*expanded, box.members[0]}  # a stack: draw each sample
            else:
                state[SELECTED] = None if state[SELECTED] == clicked else clicked
            state[CHART] += 1
            st.rerun()
        _selected(results, graph)
    if state[FULL]:
        _full_screen(figure)


def _closed() -> None:
    st.session_state[FULL] = False


def _full_screen(figure: go.Figure) -> None:
    """The model across the screen, taller, the wheel zooming it."""

    @st.dialog("Model", width="large", on_dismiss=_closed)
    def shown() -> None:
        with st.container(key="model-full"):  # the page's style makes this dialog the screen's
            st.caption("Scroll to zoom, drag to pan, double-click for the whole graph.")
            st.plotly_chart(
                go.Figure(figure).update_layout(height=_height(figure)),
                theme="streamlit",
                key="tasks-full-chart",
                config=FULL_SCREEN,
            )

    shown()


def _height(figure: go.Figure) -> int:
    """How high the model is drawn full screen: as its shape is, :data:`WIDE` pixels across, but
    :data:`TALL` at most, so that no band of nothing is drawn above and below it."""
    (x0, x1), (y0, y1) = figure.layout.xaxis.range, figure.layout.yaxis.range
    return int(min(round(WIDE * abs(float(y1) - y0) / abs(float(x1) - x0)) + 16, TALL))


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
        return
    st.divider()
    title, clear = st.columns([5, 1], vertical_alignment="center")
    with title:
        box = None if node.sample is None else graph.samples.get(node.sample)
        on = node.sample if box is None else box.title
        where = node.kind if on is None else f"{node.kind} of {on}"
        st.markdown(
            f"**{labels.plain(node.title)}** · "
            + labels.plain(where + ("" if node.required else ", a diagnostic"))
        )
    with clear:
        if st.button("Clear", width="stretch", key="tasks-clear", icon=":material/close:"):
            st.session_state[SELECTED] = None
            st.session_state[CHART] += 1
            st.rerun()
    left, right = st.columns([2, 3])
    with left:
        widgets.table(tables.facts(node))
    rows = tables.task_trials(results, node)
    with right:
        if rows:
            widgets.table(rows, height=320)
        else:
            st.caption("Its value in each trial is in the table of the Trials tab.")


# --- what it optimizes -------------------------------------------------------------------------


def _problem(results: StudyResults) -> None:
    """The parameters, objectives and constraints in one card, a column each: an item a line, its
    name with what it is in badges, and under it what else the problem says of it."""
    spec = results.problem
    called, shown = names(spec), tables.units(results)
    with st.container(border=True):
        columns = st.columns(3 if spec.constraints else 2, gap="large")
        with columns[0]:
            _heading("Parameters", ":material/tune:", len(spec.parameters))
            for p in spec.parameters:
                unit = labels.unit(p.unit)
                badges = [(p.kind, "blue")] + ([(unit, "gray")] if unit else [])
                sets = "; ".join(destination.label for destination in p.targets)
                said = f"{tables.domain(p)} · sets {sets}"
                titled = p.title is not None  # its name too, as the problem writes it
                _item(called.parameter(p.name), badges, f"{p.name} · {said}" if titled else said)
        with columns[1]:
            _heading("Objectives", ":material/flag:", len(spec.objectives))
            for o in spec.objectives:
                unit = labels.unit(shown.get(o.label))
                arrow = "▲" if o.direction == "maximize" else "▼"
                way = "green" if o.direction == "maximize" else "red"  # as the model says it
                badges = [(f"{arrow} {o.direction}", way)] + ([(unit, "gray")] if unit else [])
                _item(called.metric(o.label), badges, o.label)
        if spec.constraints:
            with columns[2]:
                _heading("Constraints", ":material/rule:", len(spec.constraints))
                for c in spec.constraints:
                    unit = labels.unit(c.unit or shown.get(c.label))
                    bounds = [f"{s} {v}" for s, v in (("≥", c.min), ("≤", c.max)) if v is not None]
                    bound = " and ".join(bounds) + (f" {unit}" if unit else "")
                    scale = "" if c.scale is None else f" · scale {c.scale}"
                    _item(called.metric(c.label), [(bound, "red")], c.label + scale)


def _heading(title: str, icon: str, count: int) -> None:
    st.markdown(f"{icon} **{title}** &nbsp; :gray-badge[{count}]")


def _item(name: str, badges: list[tuple[str, str]], said: str) -> None:
    """One item: its ``name`` and ``badges`` (text, color) on a line, ``said`` under it."""
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.markdown(f"**{labels.plain(name)}**", width="content")
        for text, color in badges:
            st.badge(text, color=color)  # type: ignore[arg-type]
    st.caption(labels.plain(said))
