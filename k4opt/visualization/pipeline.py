"""The task graph of a candidate laid out and drawn by Graphviz: as DOT text, in a Plotly figure.

:func:`pipeline_graph` writes the graph (:func:`~k4opt.visualization.task_graph`) as Graphviz DOT,
read left to right: the parameters, the geometry, each sample as a box of its own holding its
stages and evaluations (samples alike stacked in one box), the evaluations of the candidate, and
the objectives and constraints. Each kind has a color of its own: the parameters blue, the samples
and their stages violet, the evaluations aqua, the objectives and constraints orange, the geometry
neutral. A parameter reaches each task it sets something of by a dashed edge, and the task names
what it varies (``varies ECal_cell_size``); a diagnostic is dotted; of a study, each stage
shows the median of its recorded times, and each diagnostic that did not succeed in some trials in
how many, in red.

:func:`pipeline_figure` has Graphviz (its ``dot`` program) lay the same graph out and draw it as
SVG, and shows that drawing as one image in the plane of a Plotly figure, so that its text zooms
with its boxes: drag to pan, scroll to zoom, hover a task or an edge for what it is, click a task
to select it, or a stack to expand it. The last trace holds what can be clicked, each
``customdata[0]`` the id of a node or the key of a box (:attr:`TaskGraph.samples`).
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from k4opt.optimization import StudyResults
from k4opt.problem import ProblemSpec
from k4opt.visualization import _draw
from k4opt.visualization.errors import VisualizationError
from k4opt.visualization.graph import (
    COLLAPSED,
    GraphEdge,
    GraphNode,
    TaskGraph,
    collapsed,
    status_lines,
    task_graph,
)

_MEASURED = ("objective", "constraint")
# as modern node editors draw one: white cards on the page's own ground, each framed by a
# soft neutral hairline, a small dot before its title in its kind's color its only color (blue
# parameters, violet samples and stages, teal evaluations, orange objectives and constraints,
# slate geometry), its details in a quiet gray, an objective's direction in green (up, to
# maximize) or red (down, to minimize); each sample a slightly darker gray panel, quiet gray
# edges. A diagnostic that did not always succeed is framed in ``failing`` (and says so in
# words). A light color stands for one dark color, so that a drawing in the light ones can be
# told to take the dark ones (:func:`_themed`).
_COLORS = {
    False: {
        "surface": "#ffffff",
        "ink": "#0f172a",
        "secondary": "#64748b",
        "muted": "#b8c2cf",
        "border": "#dfe4ec",
        "failing": "#e11d48",
        "sample-wash": "#ebeff5",
        "sample-frame": "#dde3ec",
        "parameter-accent": "#3b82f6",
        "sample-accent": "#8b5cf6",
        "evaluator-accent": "#14b8a6",
        "measured-accent": "#f97316",
        "muted-accent": "#94a3b8",
        "up": "#16a34a",
        "down": "#dc2626",
    },
    True: {
        "surface": "#161b22",
        "ink": "#e6edf3",
        "secondary": "#8b949e",
        "muted": "#3d4652",
        "border": "#2a313c",
        "failing": "#f87171",
        "sample-wash": "#11161d",
        "sample-frame": "#222a35",
        "parameter-accent": "#58a6ff",
        "sample-accent": "#a78bfa",
        "evaluator-accent": "#2dd4bf",
        "measured-accent": "#fb923c",
        "muted-accent": "#6e7781",
        "up": "#3fb950",
        "down": "#f85149",
    },
}
_FONT = "Helvetica,Arial,sans-serif"  # Graphviz measures text in Helvetica's widths
_POINTS = 72  # Graphviz's points per inch
_TIMEOUT = 30  # seconds Graphviz may take to lay a graph out


def pipeline_graph(
    source: StudyResults | ProblemSpec,
    *,
    stack: bool = True,
    expanded: Collection[str] = (),
    dark: bool = False,
    folded: Collection[str] = (),
) -> str:
    """The tasks of a candidate of ``source`` as Graphviz DOT text; of a study, with the median
    time of each stage and how often each diagnostic did not succeed. Samples alike are stacked
    unless ``stack`` is false or the stack's key is in ``expanded``. ``dark`` takes the colors
    stepped for a dark background; the graph's own background is transparent. The boxes of the
    samples (or stacks) in ``folded``, by key, are each one node (:func:`collapsed`)."""
    graph = collapsed(task_graph(source, stack=stack, expanded=expanded), folded)
    return _dot(graph, _COLORS[dark])


def _ids(graph: TaskGraph) -> dict[str, str]:
    """The DOT id of each node, by node id, and of each box, by key: ``n<index>`` and
    ``cluster_<index>``, so that no name needs quoting."""
    ids = {node.id: f"n{index}" for index, node in enumerate(graph.nodes)}
    return ids | {key: f"cluster_{index}" for index, key in enumerate(graph.samples)}


def _boxed(node: GraphNode) -> bool:
    """Whether ``node`` is drawn in the box of its sample: a task, not what it measures."""
    return node.sample is not None and node.kind not in _MEASURED


def _dot(graph: TaskGraph, colors: Mapping[str, str]) -> str:
    ids = _ids(graph)
    lines = [
        "digraph pipeline {",
        f'  graph [rankdir=LR, newrank=true, bgcolor="transparent", fontname="{_FONT}", '
        f'fontsize=12, fontcolor="{colors["ink"]}", nodesep=0.28, ranksep=0.4, pad=0.15, '
        f"splines=true]",
        f'  node [shape=box, style="rounded,filled", fillcolor="{colors["surface"]}", '
        f'color="{colors["border"]}", fontname="{_FONT}", fontsize=12, penwidth=1.3, '
        f'fontcolor="{colors["ink"]}", margin="0.18,0.08"]',
        f'  edge [color="{colors["muted"]}", penwidth=1.2, arrowhead=normal, arrowsize=0.6, '
        f'fontname="{_FONT}", fontsize=10, fontcolor="{colors["secondary"]}"]',
    ]
    if graph.loops:
        lines += _looped(graph, ids, colors)
    else:
        lines += [
            "  " + _node(node, ids, colors, graph.recorded)
            for node in graph.nodes
            if not _boxed(node)
        ]
        for key in graph.samples:
            inside = [
                "    " + _node(node, ids, colors, graph.recorded)
                for node in graph.nodes
                if _boxed(node) and node.sample == key
            ]
            lines += _sample(graph, key, ids, colors, inside)
    # parameters in the frames of their loops are where those put them, not all at the left
    ranks = (
        (("sink", _MEASURED),) if graph.loops else (("source", ("parameter",)), ("sink", _MEASURED))
    )
    for rank, kinds in ranks:
        members = [ids[node.id] for node in graph.nodes if node.kind in kinds]
        if members:
            lines.append(f"  {{rank={rank}; " + " ".join(f"{name};" for name in members) + "}")
    drawn = set()
    for edge in graph.edges:
        pair = (ids[edge.source], ids[edge.target])
        if pair in drawn:  # a parameter that sets several things of one task: one edge
            continue
        drawn.add(pair)
        if edge.kind == "sets":
            style = " [" + _attributes(style="dashed") + "]"
        elif edge.kind == "measures":  # labeled only where a level below measured it again
            style = "" if edge.label is None else f" [label=<{html.escape(edge.label)}>]"
        elif edge.label is not None:
            style = f" [label=<{html.escape(edge.label)}>]"
        else:
            style = ""
        lines.append(f"  {pair[0]} -> {pair[1]}{style}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _sample(
    graph: TaskGraph,
    key: str,
    ids: Mapping[str, str],
    colors: Mapping[str, str],
    inside: Sequence[str] = (),
    indent: str = "  ",
) -> list[str]:
    """The box of the sample (or stack) ``key`` with its tasks, and the lines ``inside`` it."""
    box = graph.samples[key]
    return [
        f"{indent}subgraph {ids[key]} {{",
        f"{indent}  "
        + _attributes(
            label=_label(box.title, box.lines, colors, dot=colors["sample-accent"], left=True),
            style="rounded,filled" + (",bold" if box.is_stack else ""),
            fillcolor=colors["sample-wash"],
            color=colors["sample-frame"],
            labeljust="l",
            margin="14",
            penwidth="1",
            separator="; ",
        )
        + ";",
        *inside,
        f"{indent}}}",
    ]


def _looped(graph: TaskGraph, ids: Mapping[str, str], colors: Mapping[str, str]) -> list[str]:
    """The nodes of a study with levels below its own, each in the frame of its loop
    (:attr:`~k4opt.visualization.GraphNode.loop`), the frames nested as the levels are: what
    runs once for each trial of a level in that level's frame, inside those of the levels above.
    A sample's box lies in the frame of the outermost loop of its tasks, and the frames of its
    tasks of the loops below lie in it; a parameter of a level below lies in the frame of its
    level in the first sample with tasks of that loop, beside what it sets. Each level's frame
    says what it is once; drawn again, in another sample's box, it is titled only."""
    nodes, said = list(graph.nodes), set()
    first = graph.loops[0].level
    boxes = {key: [n for n in nodes if _boxed(n) and n.sample == key] for key in graph.samples}
    outermost = {
        key: min((n.loop if n.loop is not None else first - 1) for n in members)
        for key, members in boxes.items()
        if members
    }
    homes: dict[str, str | None] = {}  # the sample a parameter of a level below is drawn in
    for node in nodes:
        if node.kind == "parameter" and node.loop is not None and node.loop > first:
            homes[node.id] = next(
                (
                    key
                    for key, members in boxes.items()
                    if any(n.loop == node.loop for n in members)
                ),
                None,
            )

    def frame(
        at: int, members: Sequence[GraphNode], boxed: Sequence[str], where: str, indent: str
    ) -> list[str]:
        """The frame of level ``at`` (and those below it) holding ``members`` and the boxes
        ``boxed``; empty if none of them is in it."""
        if at > graph.loops[-1].level:
            return []
        here = [n for n in members if n.loop == at]
        lower = [n for n in members if n.loop is not None and n.loop > at]
        inner_boxes = [key for key in boxed if outermost[key] > at]
        these = [key for key in boxed if outermost[key] == at]
        if not (here or lower or these or inner_boxes):
            return []
        loop = graph.loops[at - first]
        lines = () if at in said else loop.lines
        said.add(at)
        content = [f"{indent}  " + _node(n, ids, colors, graph.recorded) for n in here]
        for key in these:
            content += _sample(
                graph, key, ids, colors, _inside(key, at, indent + "    "), indent + "  "
            )
        content += frame(at + 1, lower, inner_boxes, where, indent + "  ")
        return [
            f"{indent}subgraph cluster_loop{at}_{where} {{",
            f"{indent}  "
            + _attributes(
                label=_label(loop.title, lines, colors, dot=colors["parameter-accent"], left=True),
                style="rounded,dashed",
                color=colors["muted"],
                labeljust="l",
                margin="12",
                penwidth="1",
                separator="; ",
            )
            + ";",
            *content,
            f"{indent}}}",
        ]

    def _inside(key: str, at: int, indent: str) -> list[str]:
        """What the box of sample ``key``, in the frame of loop ``at``, holds: its tasks of that
        loop, then the frames of the loops below with theirs and the parameters drawn there."""
        members = boxes[key] + [n for n in nodes if homes.get(n.id) == key]
        top = [
            f"{indent}" + _node(n, ids, colors, graph.recorded)
            for n in members
            if n.loop == at or (n.loop is None and at < first)
        ]
        below = [n for n in members if n.loop is not None and n.loop > at]
        return top + frame(at + 1, below, (), ids[key], indent)

    outside = [n for n in nodes if not _boxed(n) and n.loop is None]  # fixed, or reached by none
    lines = ["  " + _node(n, ids, colors, graph.recorded) for n in outside]
    for key in [key for key, at in outermost.items() if at < first]:
        lines += _sample(graph, key, ids, colors, _inside(key, first - 1, "    "))
    framed = [n for n in nodes if not _boxed(n) and n.loop is not None and homes.get(n.id) is None]
    lines += frame(
        first, framed, [key for key, at in outermost.items() if at >= first], "root", "  "
    )
    return lines


def _hue(node: GraphNode) -> str:
    """The color of what ``node`` is: ``parameter``, ``sample`` (a stage), ``evaluator``,
    ``measured``, or ``muted`` for the geometry."""
    if node.kind in _MEASURED:
        return "measured"
    return {"parameter": "parameter", "stage": "sample", "evaluator": "evaluator"}.get(
        node.kind, "muted"
    )


def _node(node: GraphNode, ids: Mapping[str, str], colors: Mapping[str, str], recorded: int) -> str:
    """One node: its title and details, and what a study says of it."""
    details = [*node.details, *status_lines(node, recorded)]
    attributes: dict[str, str | _Html] = {
        "label": _label(
            node.title,
            details,
            colors,
            dot=colors[f"{_hue(node)}-accent"],
            failing=bool(node.failed),
        ),
        "tooltip": "\n".join(f"{name}: {value}" for name, value in node.facts),
        "color": colors["border"],
        "fillcolor": colors["surface"],
        "penwidth": "1",
    }
    if node.given:
        attributes["shape"] = "note"
        attributes["style"] = "filled"
    if not node.required:
        attributes["style"] = "rounded,filled,dotted"
    if node.failed:
        attributes |= {"color": colors["failing"], "penwidth": "1.8"}
    return f"{ids[node.id]} [{_attributes(**attributes)}]"


class _Html(str):
    """An HTML-like label of DOT, written as it is between ``<`` and ``>``."""

    __slots__ = ()


def _label(
    title: str,
    details: Sequence[str],
    colors: Mapping[str, str],
    *,
    dot: str | None = None,
    failing: bool = False,
    left: bool = False,
) -> _Html:
    """An HTML-like label: the title in bold, after a small ``dot`` of that color if given, each
    detail on a line of its own, smaller and quieter; a direction in green (``▲ maximize``) or
    red (``▼ minimize``), a warning (``⚠ ...``) of a ``failing`` node in the failing color; every
    line flush ``left`` if asked, else centered."""
    head = "" if dot is None else f'<font color="{dot}" point-size="10">●</font>  '
    parts = [f'{head}<font color="{colors["ink"]}"><b>{html.escape(title)}</b></font>']
    if details:  # a little air between the title and what is said under it
        parts.append('<font point-size="4"> </font>')
    for line in details:
        color = colors["secondary"]
        if line.startswith("▲"):
            color = colors["up"]
        elif line.startswith("▼"):
            color = colors["down"]
        elif failing and line.startswith("⚠"):
            color = colors["failing"]
        parts.append(f'<font point-size="10" color="{color}">{html.escape(line)}</font>')
    if left:
        return _Html("<" + "".join(f'{part}<br align="left"/>' for part in parts) + ">")
    return _Html("<" + "<br/>".join(parts) + ">")


def _attributes(separator: str = ", ", **attributes: str | _Html) -> str:
    """DOT attributes: an HTML-like label as it is, every other value quoted."""
    return separator.join(
        f"{key}={value}" if isinstance(value, _Html) else f"{key}={_quoted(value)}"
        for key, value in attributes.items()
    )


def _quoted(text: str) -> str:
    """``text`` as a quoted DOT string (a tooltip), shown as it is but for a backslash: Graphviz
    expands one as an escape (``\\G``: the graph's name) however it is escaped, so a backslash is
    shown as its look-alike U+29F5. A quote and an ampersand are entities, which Graphviz keeps."""
    for character, entity in (
        ("&", "&amp;"),
        ('"', "&quot;"),
        ("\\", "\N{REVERSE SOLIDUS OPERATOR}"),
    ):
        text = text.replace(character, entity)
    return '"' + text.replace("\n", "\\n") + '"'


# --- the figure --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Box:
    """Where Graphviz put a node or a sample's box, in points, ``y`` up; ``label`` the position of
    a box's label."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class _Spline:
    """An edge as Graphviz drew it: its cubic Bézier points."""

    tail: str
    head: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _Layout:
    """Where Graphviz put each node, sample and edge, and its drawing: an SVG image of ``extent``,
    the graph with its pad."""

    nodes: Mapping[str, _Box]  # by DOT id
    samples: Mapping[str, _Box]  # by DOT id of the cluster
    edges: tuple[_Spline, ...]
    svg: str
    extent: _Box


# what the legend and the mark of the selected node show each kind with: its color for the
# border, and a translucent wash of it, which reads on a light and a dark background alike
_STYLES = {
    "parameter": (_draw.PARAMETER, _draw.wash(_draw.PARAMETER, 0.16)),
    "sample": (_draw.SAMPLE, _draw.wash(_draw.SAMPLE, 0.16)),
    "evaluator": (_draw.EVALUATOR, _draw.wash(_draw.EVALUATOR, 0.16)),
    "measured": (_draw.MEASURED, _draw.wash(_draw.MEASURED, 0.16)),
    "muted": (_draw.MUTED, _draw.wash(_draw.MUTED, 0.16)),
}
_LEGEND = (
    ("parameter", "parameter"),
    ("muted", "geometry"),
    ("sample", "sample and its stages"),
    ("evaluator", "evaluation"),
    ("measured", "objective or constraint"),
)
_PAGE = 980  # pixels wide the figure is sized for: as wide as the dashboard's page
_DENSEST = 1.6  # pixels per point at most, so that a small graph is not drawn huge
_HEIGHTS = (220, 720)  # pixels the graph is drawn in, at least and at most, before zooming
_STEP = 12.0  # points between the transparent markers that make a task clickable


def pipeline_figure(
    source: StudyResults | ProblemSpec,
    *,
    stack: bool = True,
    expanded: Collection[str] = (),
    selected: str | None = None,
    folded: Collection[str] = (),
    dark: bool | None = None,
) -> go.Figure:
    """The tasks of a candidate of ``source`` laid out and drawn by Graphviz, shown with Plotly,
    samples alike stacked as :func:`pipeline_graph` says, the node ``selected`` (by id) marked; of
    a study, with what :func:`pipeline_graph` shows of it. Graphviz's drawing is one image in the
    plane of the figure, so that its text zooms with its boxes; it takes the dark colors if
    ``dark``, the light ones if not, and, unless told, where the viewer prefers a dark scheme.
    The boxes of the samples in ``folded`` are each one node (:func:`collapsed`). Raises
    :class:`VisualizationError` if Graphviz's ``dot`` cannot be run."""
    graph = collapsed(task_graph(source, stack=stack, expanded=expanded), folded)
    ids = _ids(graph)
    layout = _layout(_dot(graph, _COLORS[bool(dark)]))  # in the colors it is told, if told
    extent = layout.extent
    width, height = extent.x1 - extent.x0, extent.y1 - extent.y0
    scale = min(_PAGE / width, _DENSEST)  # pixels per point, as first drawn
    by_dot = {ids[node.id]: node for node in graph.nodes}
    boxes = {ids[key]: key for key in graph.samples}
    figure = go.Figure()
    clickable: list[tuple[float, float, str, str]] = []  # x, y, id, hover
    for cluster, box in layout.samples.items():
        if cluster not in boxes:  # a level's box: nothing to draw by sample
            continue
        sample = graph.samples[boxes[cluster]]
        if not sample.is_stack or box.label is None:
            continue
        hover = f"<b>{html.escape(sample.title)}</b><br>" + "<br>".join(
            html.escape(line) for line in sample.lines
        )
        title = _Box(box.x0, 2 * box.label[1] - box.y1, box.x1, box.y1)  # its label, at its top
        clickable += [
            (x, y, boxes[cluster], hover + "<br>click to draw each") for x, y in _grid(title)
        ]
    for dot_id, box in layout.nodes.items():
        node = by_dot[dot_id]
        hover = _draw.hover(node, graph)
        if node.id.startswith(COLLAPSED):
            hover += "<br>click to draw its tasks"
        elif node.samples:
            hover += "<br>click to draw each sample"
        clickable += [(x, y, node.id, hover) for x, y in _grid(box)]
    shapes = []
    chosen = next((d for d, node in by_dot.items() if node.id == selected), None)
    if chosen is not None and chosen in layout.nodes:
        color = _STYLES[_hue(by_dot[chosen])][0]
        box = layout.nodes[chosen]
        ring = _Box(box.x0 - 3, box.y0 - 3, box.x1 + 3, box.y1 + 3)
        shapes.append(
            _rounded(
                ring,
                8,
                fillcolor=_draw.wash(color, 0.14),
                line={"color": color, "width": 3},
                layer="above",
            )
        )
    edges: dict[tuple[str, str], list[GraphEdge]] = {}
    for each in graph.edges:
        edges.setdefault((ids[each.source], ids[each.target]), []).append(each)
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_text: list[str] = []
    for spline in layout.edges:
        carried = edges[(spline.tail, spline.head)]
        points = _draw.bezier(spline.points)
        text = _draw.edge_text(carried, by_dot[spline.tail].title, by_dot[spline.head].title)
        inner = points[len(points) // 6 : len(points) - len(points) // 6] or points
        hover_x += [p[0] for p in inner]
        hover_y += [p[1] for p in inner]
        hover_text += [text] * len(inner)
    figure.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            showlegend=False,
            text=hover_text,
            marker={"size": 8, "color": "rgba(0, 0, 0, 0)"},
            hovertemplate="%{text}<extra></extra>",
        )
    )
    for hue, name in _LEGEND:
        color, fill = _STYLES[hue]
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker={
                    "symbol": "square",
                    "size": 12,
                    "color": fill,
                    "line": {"color": color, "width": 2},
                },
                hoverinfo="skip",
            )
        )
    # what can be clicked, last: transparent markers covering each node and each stack's title
    figure.add_trace(
        go.Scatter(
            x=[c[0] for c in clickable],
            y=[c[1] for c in clickable],
            mode="markers",
            showlegend=False,
            customdata=[[c[2]] for c in clickable],
            text=[c[3] for c in clickable],
            hovertemplate="%{text}<extra></extra>",
            marker={"symbol": "square", "size": _STEP * scale, "color": "rgba(0, 0, 0, 0)"},
            selected={"marker": {"color": "rgba(0, 0, 0, 0)"}},
            unselected={"marker": {"color": "rgba(0, 0, 0, 0)"}},
        )
    )
    low, high = _HEIGHTS
    figure.add_layout_image(
        source="data:image/svg+xml;base64,"
        + base64.b64encode(
            (layout.svg if dark is not None else _themed(layout.svg)).encode()
        ).decode("ascii"),
        xref="x",
        yref="y",
        x=extent.x0,
        y=extent.y1,
        sizex=width,
        sizey=height,
        sizing="stretch",
        xanchor="left",
        yanchor="top",
        layer="below",
    )
    figure.update_layout(
        height=round(min(max(height * scale, low), high)) + 44,
        margin={"l": 4, "r": 4, "t": 36, "b": 8},
        xaxis={"visible": False, "range": [extent.x0, extent.x1]},
        yaxis={"visible": False, "range": [extent.y0, extent.y1], "scaleanchor": "x"},
        shapes=shapes,
        showlegend=True,
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.0,
            "yanchor": "bottom",
            "itemclick": False,
            "itemdoubleclick": False,
            "font": {"size": 11},
        },
        hoverlabel={"align": "left"},
        hovermode="closest",
        clickmode="event+select",
        dragmode="pan",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _grid(box: _Box) -> list[tuple[float, float]]:
    """Points covering ``box``, :data:`_STEP` apart at most, a little inside its border."""
    x0, y0, x1, y1 = box.x0 + 3, box.y0 + 3, box.x1 - 3, box.y1 - 3
    columns = max(int((x1 - x0) // _STEP) + 1, 1)
    rows = max(int((y1 - y0) // _STEP) + 1, 1)
    return [
        (
            x0 + (x1 - x0) * (column + 0.5) / columns,
            y0 + (y1 - y0) * (row + 0.5) / rows,
        )
        for row in range(rows)
        for column in range(columns)
    ]


def _themed(svg: str) -> str:
    """``svg``, drawn in the light colors, with a style that takes the dark ones where the viewer
    prefers a dark scheme, its gradients too: an image keeps its colors whatever the page around
    it does, so it follows the preference itself."""
    light, dark = _COLORS[False], _COLORS[True]
    changed = [key for key in light if light[key] != dark[key]]
    rules = " ".join(
        f'[{attribute}="{light[key]}"] {{ {attribute}: {dark[key]} }}'
        for key in changed
        for attribute in ("fill", "stroke")
    ) + " ".join(  # a gradient's stops are colored by their style
        f'[style*="stop-color:{light[key]}"] {{ stop-color: {dark[key]} !important }}'
        for key in changed
    )
    style = f"<style>@media (prefers-color-scheme: dark) {{ {rules} }}</style>"
    opened = svg.index(">", svg.index("<svg")) + 1
    return svg[:opened] + style + svg[opened:]


_VIEW = re.compile(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"')
_SHIFT = re.compile(r'class="graph" transform="[^"]*translate\(([-\d.]+)[ ,]([-\d.]+)\)"')


def _layout(dot: str) -> _Layout:
    """Where Graphviz puts each node, sample and edge of ``dot``, and how it draws them: one run
    of ``dot`` writes both, so that they are of one layout."""
    program = shutil.which("dot")
    if program is None:
        raise VisualizationError(
            "laying the graph out needs Graphviz's dot program, which is not on PATH "
            "(install Graphviz: apt install graphviz, dnf install graphviz, brew install graphviz)"
        )
    with tempfile.TemporaryDirectory(prefix="k4opt-graph-") as scratch:
        placed, drawn = Path(scratch, "graph.json"), Path(scratch, "graph.svg")
        try:
            done = subprocess.run(
                [program, "-Tjson", "-o", str(placed), "-Tsvg", "-o", str(drawn)],
                input=dot,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VisualizationError(f"Graphviz could not lay the graph out: {exc}") from None
        if done.returncode != 0:
            raise VisualizationError(f"Graphviz could not lay the graph out: {done.stderr.strip()}")
        data = json.loads(placed.read_text(encoding="utf-8"))
        svg = drawn.read_text(encoding="utf-8")
    names = {item["_gvid"]: item["name"] for item in data.get("objects", ())}
    nodes, samples = {}, {}
    for item in data.get("objects", ()):
        if "bb" in item:  # a cluster: a sample's box
            x0, y0, x1, y1 = _numbers(item["bb"])
            samples[item["name"]] = _Box(x0, y0, x1, y1, _point(item.get("lp")))
        elif "pos" in item:
            x, y = _numbers(item["pos"])
            w, h = float(item["width"]) * _POINTS / 2, float(item["height"]) * _POINTS / 2
            nodes[item["name"]] = _Box(x - w, y - h, x + w, y + h)
    edges = []
    for item in data.get("edges", ()):
        points: list[tuple[float, float]] = []
        for token in item["pos"].split():
            if not token.startswith(("s,", "e,")):
                found = _point(token)
                assert found is not None
                points.append(found)
        edges.append(_Spline(names[item["tail"]], names[item["head"]], tuple(points)))
    return _Layout(nodes, samples, tuple(edges), svg, _extent(svg))


def _extent(svg: str) -> _Box:
    """What Graphviz's SVG shows, in the points of its layout, ``y`` up: its view box, shifted by
    the translation that takes the layout into it (and flips ``y``)."""
    view, shift = _VIEW.search(svg), _SHIFT.search(svg)
    if view is None or shift is None:
        raise VisualizationError("Graphviz drew the graph in an SVG without its view box")
    left, top, width, height = map(float, view.groups())
    dx, dy = map(float, shift.groups())
    return _Box(left - dx, dy - top - height, left - dx + width, dy - top)


def _numbers(text: str) -> list[float]:
    return [float(part) for part in text.split(",")]


def _point(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    x, y = _numbers(text)
    return x, y


def _rounded(box: _Box, radius: float, **style: Any) -> dict[str, Any]:
    """A rectangle with rounded corners, as a Plotly path shape in data coordinates."""
    x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    path = (
        f"M {x0 + r},{y0} L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + r} "
        f"L {x1},{y1 - r} Q {x1},{y1} {x1 - r},{y1} L {x0 + r},{y1} Q {x0},{y1} {x0},{y1 - r} "
        f"L {x0},{y0 + r} Q {x0},{y0} {x0 + r},{y0} Z"
    )
    return {"type": "path", "path": path, "xref": "x", "yref": "y", "layer": "below", **style}
