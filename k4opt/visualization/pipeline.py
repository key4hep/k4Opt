"""The task graph of a candidate laid out by Graphviz: as DOT text, and drawn with Plotly.

:func:`pipeline_graph` writes the graph (:func:`~k4opt.visualization.task_graph`) as Graphviz DOT,
read left to right: the parameters, the geometry, each sample as a box of its own holding its
stages and evaluations (samples alike stacked in one box), the evaluations of the candidate, and
the objectives and constraints. Each kind has a color of its own: the parameters blue, the samples
and their stages violet, the evaluations aqua, the objectives and constraints orange, the geometry
neutral. A parameter reaches each task it sets something of by a dashed edge, and the task names
what it sets (``ECal_cell_size ← ecal_cell_size``); a diagnostic is dotted; of a study, each stage
shows the median of its recorded times, and each diagnostic that did not succeed in some trials in
how many, in red.

:func:`pipeline_figure` has Graphviz (its ``dot`` program) lay the same graph out, and draws it
with Plotly: drag to pan, scroll to zoom, hover a task or an edge for what it is, click a task to
select it, or a stack to expand it. The last trace holds what can be clicked, each
``customdata[0]`` the id of a node or the key of a box (:attr:`TaskGraph.samples`).
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from k4opt.optimization import StudyResults
from k4opt.problem import ProblemSpec
from k4opt.visualization import _draw
from k4opt.visualization.errors import VisualizationError
from k4opt.visualization.graph import GraphEdge, GraphNode, TaskGraph, status_lines, task_graph

_MEASURED = ("objective", "constraint")
# the reference palette's categorical slots 1 (blue: parameters), 7 (violet: samples), 3 (aqua:
# evaluations) and 2 (orange: objectives and constraints), its inks and hairlines, and status
# "critical" (a diagnostic that did not always succeed, also said in words), light and dark steps
_COLORS = {
    False: {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "border": "#c3c2b7",
        "parameter": "#2a78d6",
        "parameter-wash": "#e6effa",
        "sample": "#6250d6",
        "sample-wash": "#f1effb",
        "stage-wash": "#e4e0f8",
        "evaluator": "#1baf7a",
        "evaluator-wash": "#ddf3ea",
        "measured": "#eb6834",
        "measured-wash": "#fce8df",
        "failing": "#d03b3b",
    },
    True: {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "border": "#52514e",
        "parameter": "#3987e5",
        "parameter-wash": "#1b2a3d",
        "sample": "#9085e9",
        "sample-wash": "#211f2c",
        "stage-wash": "#2b2840",
        "evaluator": "#199e70",
        "evaluator-wash": "#16302a",
        "measured": "#d95926",
        "measured-wash": "#3a2219",
        "failing": "#d03b3b",
    },
}
_FONT = "Helvetica,Arial,sans-serif"  # Graphviz measures text in Helvetica's widths
_POINTS = 72  # Graphviz's points per inch
_WIDEST = 880  # pixels the figure is drawn in at most; a wider graph is drawn smaller
_TIMEOUT = 30  # seconds Graphviz may take to lay a graph out


def pipeline_graph(
    source: StudyResults | ProblemSpec,
    *,
    stack: bool = True,
    expanded: Collection[str] = (),
    dark: bool = False,
) -> str:
    """The tasks of a candidate of ``source`` as Graphviz DOT text; of a study, with the median
    time of each stage and how often each diagnostic did not succeed. Samples alike are stacked
    unless ``stack`` is false or the stack's key is in ``expanded``. ``dark`` takes the colors
    stepped for a dark background; the graph's own background is transparent."""
    return _dot(task_graph(source, stack=stack, expanded=expanded), _COLORS[dark])


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
        f'fontsize=11, fontcolor="{colors["ink"]}", nodesep=0.22, ranksep=0.42, pad=0.1]',
        f'  node [shape=box, style="rounded,filled", fillcolor="{colors["surface"]}", '
        f'color="{colors["border"]}", fontname="{_FONT}", fontsize=11, '
        f'fontcolor="{colors["ink"]}", margin="0.2,0.07"]',
        f'  edge [color="{colors["muted"]}", arrowsize=0.6, fontname="{_FONT}", fontsize=9, '
        f'fontcolor="{colors["secondary"]}"]',
    ]
    lines += [
        "  " + _node(node, ids, colors, graph.recorded) for node in graph.nodes if not _boxed(node)
    ]
    for key, box in graph.samples.items():
        lines += [
            f"  subgraph {ids[key]} {{",
            "    "
            + _attributes(
                label=_label(box.title, box.lines, colors),
                style="rounded,filled" + (",bold" if box.is_stack else ""),
                fillcolor=colors["sample-wash"],
                color=colors["sample"],
                labeljust="l",
                separator="; ",
            )
            + ";",
            *(
                "    " + _node(node, ids, colors, graph.recorded)
                for node in graph.nodes
                if _boxed(node) and node.sample == key
            ),
            "  }",
        ]
    for rank, kinds in (("source", ("parameter",)), ("sink", _MEASURED)):
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
            style = " [" + _attributes(style="dashed", color=colors["parameter"]) + "]"
        elif edge.kind == "measures":
            style = " [" + _attributes(color=colors["measured"]) + "]"
        elif edge.label is not None:
            style = f" [label=<{html.escape(edge.label)}>]"
        else:
            style = ""
        lines.append(f"  {pair[0]} -> {pair[1]}{style}")
    lines.append("}")
    return "\n".join(lines) + "\n"


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
    hue = _hue(node)
    wash = {"sample": "stage-wash"}.get(hue, f"{hue}-wash")
    attributes: dict[str, str | _Html] = {
        "label": _label(node.title, details, colors, muted=not node.required),
        "tooltip": "\n".join(f"{name}: {value}" for name, value in node.facts),
        "color": colors[hue],
        "fillcolor": colors.get(wash, colors["surface"]),
    }
    if node.kind in _MEASURED:
        attributes["penwidth"] = "1.6"
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
    title: str, details: Iterable[str], colors: Mapping[str, str], *, muted: bool = False
) -> _Html:
    """An HTML-like label: the title in bold, each detail on a line of its own, smaller."""
    ink = colors["secondary"] if muted else colors["ink"]
    parts = [f'<font color="{ink}"><b>{html.escape(title)}</b></font>']
    parts += [
        f'<font point-size="9" color="{colors["secondary"]}">{html.escape(detail)}</font>'
        for detail in details
    ]
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
    """An edge as Graphviz drew it: its cubic Bézier points, the tip of its arrow, and where its
    label goes."""

    tail: str
    head: str
    points: tuple[tuple[float, float], ...]
    tip: tuple[float, float] | None
    label: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class _Layout:
    width: float
    height: float
    nodes: Mapping[str, _Box]  # by DOT id
    samples: Mapping[str, _Box]  # by DOT id of the cluster
    edges: tuple[_Spline, ...]


# what each kind is drawn with in the figure: a translucent wash of its color, which reads on a
# light and a dark background alike, and its color for the border
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


def pipeline_figure(
    source: StudyResults | ProblemSpec,
    *,
    stack: bool = True,
    expanded: Collection[str] = (),
    selected: str | None = None,
) -> go.Figure:
    """The tasks of a candidate of ``source`` laid out by Graphviz and drawn with Plotly, samples
    alike stacked as :func:`pipeline_graph` says, the node ``selected`` (by id) marked; of a
    study, with what :func:`pipeline_graph` shows of it. The figure sets no color of text, so that
    it reads on a light and a dark background alike. Raises :class:`VisualizationError` if
    Graphviz's ``dot`` cannot be run."""
    graph = task_graph(source, stack=stack, expanded=expanded)
    ids = _ids(graph)
    layout = _layout(_dot(graph, _COLORS[False]))
    scale = min(96 / _POINTS, _WIDEST / layout.width)  # pixels per point
    font = 10.5 * scale  # Graphviz's 11 points, a little smaller: the page's font is not its
    by_dot = {ids[node.id]: node for node in graph.nodes}
    boxes = {ids[key]: key for key in graph.samples}
    figure = go.Figure()
    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    clickable: list[tuple[float, float, float, str, str]] = []  # x, y, size, id, hover
    for cluster, box in layout.samples.items():
        sample = graph.samples[boxes[cluster]]
        border = {"color": _draw.SAMPLE, "width": 2.2 if sample.is_stack else 1.2}
        if sample.is_stack:
            border["dash"] = "longdash"
        shapes.append(_rounded(box, 8, fillcolor=_draw.wash(_draw.SAMPLE, 0.06), line=border))
        assert box.label is not None
        title = sample.title + (" · click to draw each" if sample.is_stack else "")
        annotations.append(
            {
                "x": box.x0 + 8,
                "y": box.label[1],
                "xanchor": "left",
                "text": _draw.label(title, sample.lines, font),
                "showarrow": False,
                "align": "left",
                "font": {"size": font},
            }
        )
        if sample.is_stack:
            hover = f"<b>{html.escape(sample.title)}</b><br>" + "<br>".join(
                html.escape(line) for line in sample.lines
            )
            clickable.append(
                (box.x0 + 40, box.label[1], 18, boxes[cluster], hover + "<br>click to draw each")
            )
    for dot_id, box in layout.nodes.items():
        node = by_dot[dot_id]
        color, fill = _STYLES[_hue(node)]
        line: dict[str, Any] = {"color": color, "width": 2 if node.kind in _MEASURED else 1.4}
        if not node.required:
            line["dash"] = "dot"
        if node.failed:
            line = {"color": _draw.FAILING, "width": 2.2}
        if node.id == selected:
            line, fill = {"color": color, "width": 3.5}, _draw.wash(color, 0.34)
        shapes.append(_rounded(box, 5, fillcolor=fill, line=line))
        details = [*node.details, *status_lines(node, graph.recorded)]
        annotations.append(
            {
                "x": (box.x0 + box.x1) / 2,
                "y": (box.y0 + box.y1) / 2,
                "text": _draw.label(node.title, details, font, failing=bool(node.failed)),
                "showarrow": False,
                "font": {"size": font},
                "opacity": 1 if node.required else 0.75,
            }
        )
        size = max(min(box.y1 - box.y0, box.x1 - box.x0) * scale, 10)
        hover = _draw.hover(node, graph)
        if node.samples:
            hover += "<br>click to draw each sample"
        clickable.append(((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, size, node.id, hover))
    edges: dict[tuple[str, str], list[GraphEdge]] = {}
    for each in graph.edges:
        edges.setdefault((ids[each.source], ids[each.target]), []).append(each)
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_text: list[str] = []
    for spline in layout.edges:
        carried = edges[(spline.tail, spline.head)]
        kind = carried[0].kind
        color = {"sets": _draw.PARAMETER, "measures": _draw.MEASURED}.get(kind, _draw.EDGE)
        points = _draw.bezier(spline.points)
        if spline.tip is not None:  # the arrow's line, from the spline's end to its tip
            points.append(spline.tip)
        figure.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
                line={"color": color, "width": 1.3, "dash": "dash" if kind == "sets" else "solid"},
            )
        )
        if spline.tip is not None:
            annotations.append(_arrow(points[-2], spline.tip, color))
        label = carried[0].label
        if spline.label is not None and label is not None and kind == "reads":
            annotations.append(
                {
                    "x": spline.label[0],
                    "y": spline.label[1],
                    "text": f"<i>{html.escape(label)}</i>",
                    "showarrow": False,
                    "font": {"size": font * 0.85, "color": _draw.MUTED},
                }
            )
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
    # what can be clicked, last: transparent markers over each node and each stack's title
    figure.add_trace(
        go.Scatter(
            x=[c[0] for c in clickable],
            y=[c[1] for c in clickable],
            mode="markers",
            showlegend=False,
            customdata=[[c[3]] for c in clickable],
            text=[c[4] for c in clickable],
            hovertemplate="%{text}<extra></extra>",
            marker={
                "symbol": "square",
                "size": [c[2] for c in clickable],
                "color": "rgba(0, 0, 0, 0)",
            },
            selected={"marker": {"color": "rgba(0, 0, 0, 0)"}},
            unselected={"marker": {"color": "rgba(0, 0, 0, 0)"}},
        )
    )
    figure.update_layout(
        height=int(layout.height * scale) + 56,
        margin={"l": 4, "r": 4, "t": 40, "b": 8},
        xaxis={"visible": False, "range": [0, layout.width]},
        yaxis={"visible": False, "range": [0, layout.height], "scaleanchor": "x"},
        shapes=shapes,
        annotations=annotations,
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


def _layout(dot: str) -> _Layout:
    """Where Graphviz puts each node, sample and edge of ``dot``."""
    program = shutil.which("dot")
    if program is None:
        raise VisualizationError(
            "laying the graph out needs Graphviz's dot program, which is not on PATH "
            "(install Graphviz: apt install graphviz, dnf install graphviz, brew install graphviz)"
        )
    try:
        done = subprocess.run(
            [program, "-Tjson"],
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
    data = json.loads(done.stdout)
    _, _, width, height = _numbers(data["bb"])
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
        tip = None
        for token in item["pos"].split():
            if token.startswith("e,"):
                tip = _point(token[2:])
            elif not token.startswith("s,"):
                found = _point(token)
                assert found is not None
                points.append(found)
        edges.append(
            _Spline(
                names[item["tail"]], names[item["head"]], tuple(points), tip, _point(item.get("lp"))
            )
        )
    return _Layout(width, height, nodes, samples, tuple(edges))


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


def _arrow(start: tuple[float, float], tip: tuple[float, float], color: str) -> dict[str, Any]:
    """An arrowhead at ``tip``, pointing away from ``start``."""
    return {
        "x": tip[0],
        "y": tip[1],
        "ax": start[0],
        "ay": start[1],
        "xref": "x",
        "yref": "y",
        "axref": "x",
        "ayref": "y",
        "text": "",
        "showarrow": True,
        "arrowhead": 2,
        "arrowsize": 1,
        "arrowwidth": 1.2,
        "arrowcolor": color,
        "standoff": 0,
    }
