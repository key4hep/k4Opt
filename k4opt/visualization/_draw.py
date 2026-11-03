"""What the figures of a task graph share: colors that read on a light and a dark background
alike, the text shown on hover, and the points along an edge that hovering finds.

The figures set no color of text: it is the theme's (Plotly's, or Streamlit's when it draws them),
so that they read on either background without being told which; fills are translucent washes.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

from k4opt.visualization.graph import GraphEdge, GraphNode, TaskGraph, status_lines

# the reference palette's muted ink, categorical slots 1 (blue: parameters), 7 (violet: samples),
# 3 (aqua: evaluations) and 2 (orange: objectives and constraints): the steps for the dark
# surface, which read on the light one as well
MUTED = "#898781"
PARAMETER = "#3987e5"
SAMPLE = "#9085e9"
EVALUATOR = "#199e70"
MEASURED = "#eb6834"


def wash(color: str, alpha: float) -> str:
    """``color`` (``#rrggbb``) as a translucent fill: it takes on the background behind it."""
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def hover(node: GraphNode, graph: TaskGraph) -> str:
    """Everything the graph knows of ``node``, as hover text."""
    lines = [f"<b>{html.escape(node.title)}</b>"]
    lines += [f"{html.escape(name)}: {html.escape(_cut(value))}" for name, value in node.facts]
    lines += [html.escape(line) for line in status_lines(node, graph.recorded)]
    return "<br>".join(lines)


def edge_text(edges: Sequence[GraphEdge], source: str, target: str) -> str:
    """What the edges from ``source`` to ``target`` (all of one kind) carry, as hover text."""
    kind = edges[0].kind
    what = {"sets": "sets", "reads": "is read by", "measures": "measures"}[kind]
    text = f"<b>{html.escape(source)}</b> {what} <b>{html.escape(target)}</b>"
    labels = [edge.label for edge in edges if edge.label]
    if labels:
        joined = ", ".join(html.escape(label) for label in dict.fromkeys(labels))
        text += f"<br>{'sets' if kind == 'sets' else 'as'} {joined}"
    return text


def bezier(points: Sequence[tuple[float, float]], steps: int = 12) -> list[tuple[float, float]]:
    """The cubic Bézier spline through ``points`` (a start, then three points per segment),
    sampled with ``steps`` points per segment."""
    sampled = [points[0]]
    for start in range(0, len(points) - 3, 3):
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points[start : start + 4]
        for i in range(1, steps + 1):
            s = i / steps
            a, b, c, d = (1 - s) ** 3, 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s * s, s**3
            sampled.append((a * x0 + b * x1 + c * x2 + d * x3, a * y0 + b * y1 + c * y2 + d * y3))
    return sampled


def _cut(text: str, longest: int = 90) -> str:
    return text if len(text) <= longest else text[: longest - 1] + "…"
