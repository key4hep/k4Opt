"""How the parameters of a study act on its objectives: the figures of Optuna Dashboard's
analytics, from the study's results alone.

* :func:`parallel_figure`: each feasible trial as a line through one axis per parameter,
  objective and constraint margin, colored by one objective, the better the stronger.
* :func:`slice_figure`: an objective against each parameter, a panel each, colored by the order
  the trials were proposed in.
* :func:`rank_figure`: each pair of parameters, the feasible trials colored by how they rank in
  an objective; with fewer than two parameters there is no pair.
* :func:`edf_figure`: the empirical distribution of each objective over the trials that have one.

An objective is named as :func:`~k4opt.visualization.axis_choices` names it
(``objective:<label>``), the first one by default. Nothing is interpolated between trials (no
contour, which would show values no trial measured), and every trial a figure leaves out is
counted below it, by outcome.
"""

from __future__ import annotations

import html
import math
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from k4opt.optimization import StudyResults, TrialResult
from k4opt.problem import Names, Parameter, names
from k4opt.visualization.errors import VisualizationError
from k4opt.visualization.trials import (
    _COLORS,
    _RAMPS,
    OBJECTIVE,
    _Axis,
    _axis,
    _hover,
    _unit,
)

_COLUMNS = 3  # panels side by side at most
_PANEL = 300  # pixels high each row of panels is


def parallel_figure(
    results: StudyResults, objective: str | None = None, *, dark: bool = False
) -> go.Figure:
    """The feasible trials of ``results``, each a line through an axis per parameter (a
    categorical one with its choices), per objective and per bound of each constraint (the
    margin: zero or below when satisfied); colored by ``objective``, the better the further
    from the background. Raises :class:`VisualizationError` for an objective the study does not
    have, and for values of one recorded in different units."""
    shade = _objective(results, objective)
    spec = results.problem
    called = names(spec)
    drawn = [t for t in results.trials if t.outcome == "feasible"]
    left_out = Counter(t.outcome for t in results.trials if t.outcome != "feasible")
    dimensions: list[dict[str, Any]] = [_dimension(p, drawn, called) for p in spec.parameters]
    for index, o in enumerate(spec.objectives):
        unit = _unit(_axis(results, f"{OBJECTIVE}{o.label}"), drawn)
        dimensions.append(
            {
                "label": _heading(called.objective(o.label), unit),
                "values": [_measured(t, index) for t in drawn],
            }
        )
    for constraint in spec.constraints:
        for bound in ("min", "max"):
            if getattr(constraint, bound) is None:
                continue
            name = f"{constraint.label}:{bound}"
            dimensions.append(
                {
                    "label": _heading(
                        f"margin {called.constraint(constraint.label)} {bound}", constraint.unit
                    ),
                    "values": [_margin(t, name) for t in drawn],
                }
            )
    ramp = _RAMPS[dark]
    assert shade.objective is not None
    figure = go.Figure(
        go.Parcoords(
            dimensions=dimensions,
            line={
                "color": [shade.value(t) for t in drawn],
                "colorscale": [[i / (len(ramp) - 1), step] for i, step in enumerate(ramp)],
                # the better end far from the background, in either direction
                "reversescale": shade.objective.direction == "minimize",
                "showscale": bool(drawn),
                "colorbar": {"title": {"text": html.escape(shade.called)}},
            },
            labelangle=0,
            labelside="top",
        )
    )
    notes = [
        f"Each feasible trial, colored by {html.escape(shade.called)} "
        f"({shade.objective.direction}): the stronger, the better"
    ]
    notes += _left_out(left_out, "Not drawn, not feasible")
    if not drawn:
        notes.insert(0, "No trial is feasible yet")
    figure.update_layout(
        # the notes as the title, at the bottom: Plotly draws no annotation over parallel axes
        title={
            "text": "<br>".join(notes),
            "font": {"size": 12},
            "x": 0,
            "xanchor": "left",
            "xref": "paper",
            "y": 0.02,
            "yanchor": "bottom",
        },
        margin={"l": 70, "r": 70, "t": 70, "b": 90},
    )
    return figure


def slice_figure(
    results: StudyResults, objective: str | None = None, *, dark: bool = False
) -> go.Figure:
    """``objective`` against each parameter of ``results``, a panel each, sharing the vertical
    axis: the feasible trials colored by their number, the infeasible ones marked apart. Raises
    :class:`VisualizationError` for an objective the study does not have, and for values of one
    recorded in different units."""
    axis = _objective(results, objective)
    colors = _COLORS[dark]
    drawn, left_out = _valid(results, axis)
    unit = _unit(axis, drawn)
    parameters = results.problem.parameters
    called = names(results.problem)
    columns, rows = _grid(len(parameters))
    figure = make_subplots(
        rows=rows, cols=columns, shared_yaxes=True, horizontal_spacing=0.03, vertical_spacing=0.18
    )
    front = set(results.pareto)
    for index, parameter in enumerate(parameters):
        row, column = index // columns + 1, index % columns + 1
        for outcome in ("feasible", "infeasible"):
            trials = [t for t in drawn if t.outcome == outcome]
            if not trials:
                continue
            marker: dict[str, Any] = (
                {
                    "color": [t.number for t in trials],
                    "coloraxis": "coloraxis",
                    "size": 8,
                    "line": {"width": 1, "color": colors["ring"]},
                }
                if outcome == "feasible"
                else {"symbol": "x-thin-open", "size": 9, "color": colors["infeasible"]}
            )
            figure.add_trace(
                go.Scatter(
                    x=[t.candidate.get(parameter.name) for t in trials],
                    y=[axis.value(t) for t in trials],
                    mode="markers",
                    name="feasible" if outcome == "feasible" else "infeasible (a limit violated)",
                    legendgroup=outcome,
                    showlegend=index == 0,
                    marker=marker,
                    customdata=[t.number for t in trials],
                    hovertext=[_hover(results, t, front, called) for t in trials],
                    hoverinfo="text",
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(
            title_text=_heading(called.parameter(parameter.name), parameter.unit),
            row=row,
            col=column,
            **_category(parameter),
        )
    for row in range(1, rows + 1):
        figure.update_yaxes(title_text=axis.title(unit), row=row, col=1)
    figure.update_layout(
        coloraxis=_scale(dark, "trial"),
        height=rows * _PANEL + 120,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hovermode="closest",
        margin={"t": 40, "b": 90},
    )
    _annotate(figure, _left_out(left_out), None if drawn else "No trial has a value of it yet")
    return figure


def rank_figure(
    results: StudyResults, objective: str | None = None, *, dark: bool = False
) -> go.Figure:
    """Each pair of parameters of ``results``, a panel each: the feasible trials colored by how
    they rank in ``objective`` (the share of the other feasible trials each is better than), the
    infeasible ones marked apart. Raises :class:`VisualizationError` for fewer than two
    parameters, and for an objective the study does not have."""
    parameters = results.problem.parameters
    if len(parameters) < 2:
        raise VisualizationError(
            "ranking trials over pairs of parameters needs two parameters at least; "
            f"the study has {len(parameters)}"
        )
    axis = _objective(results, objective)
    assert axis.objective is not None
    colors = _COLORS[dark]
    drawn, left_out = _valid(results, axis)
    feasible = [t for t in drawn if t.outcome == "feasible"]
    signed = {t.number: axis.signed(t) for t in feasible}
    others = max(len(feasible) - 1, 1)
    rank = {n: sum(v < s for s in signed.values()) / others for n, v in signed.items()}
    called = names(results.problem)
    pairs = list(combinations(parameters, 2))
    columns, rows = _grid(len(pairs))
    figure = make_subplots(rows=rows, cols=columns, horizontal_spacing=0.08, vertical_spacing=0.18)
    front = set(results.pareto)
    for index, (across, up) in enumerate(pairs):
        row, column = index // columns + 1, index % columns + 1
        for outcome in ("feasible", "infeasible"):
            trials = [t for t in drawn if t.outcome == outcome]
            if not trials:
                continue
            marker: dict[str, Any] = (
                {
                    "color": [rank[t.number] for t in trials],
                    "coloraxis": "coloraxis",
                    "size": 8,
                    "line": {"width": 1, "color": colors["ring"]},
                }
                if outcome == "feasible"
                else {"symbol": "x-thin-open", "size": 9, "color": colors["muted"]}
            )
            figure.add_trace(
                go.Scatter(
                    x=[t.candidate.get(across.name) for t in trials],
                    y=[t.candidate.get(up.name) for t in trials],
                    mode="markers",
                    name="feasible" if outcome == "feasible" else "infeasible (a limit violated)",
                    legendgroup=outcome,
                    showlegend=index == 0,
                    marker=marker,
                    customdata=[t.number for t in trials],
                    hovertext=[_hover(results, t, front, called) for t in trials],
                    hoverinfo="text",
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(
            title_text=_heading(called.parameter(across.name), across.unit),
            row=row,
            col=column,
            **_category(across),
        )
        figure.update_yaxes(
            title_text=_heading(called.parameter(up.name), up.unit),
            row=row,
            col=column,
            **_category(up),
        )
    scale = _scale(dark, f"rank in {html.escape(axis.called)}")
    scale |= {"cmin": 0, "cmax": 1}
    scale["colorbar"] |= {"tickvals": [0, 1], "ticktext": ["worst", "best"]}
    figure.update_layout(
        coloraxis=scale,
        height=rows * _PANEL + 120,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hovermode="closest",
        margin={"t": 40, "b": 90},
    )
    _annotate(figure, _left_out(left_out), None if drawn else "No trial has a value of it yet")
    return figure


def edf_figure(results: StudyResults, *, dark: bool = False) -> go.Figure:
    """The empirical distribution of each objective of ``results``, a panel each: the share of
    the trials with a value of it whose value is at most each value. Raises
    :class:`VisualizationError` for values of an objective recorded in different units."""
    colors = _COLORS[dark]
    objectives = results.problem.objectives
    columns, rows = _grid(len(objectives))
    figure = make_subplots(rows=rows, cols=columns, horizontal_spacing=0.08, vertical_spacing=0.22)
    counted: Counter[str] = Counter()
    for index, o in enumerate(objectives):
        axis = _axis(results, f"{OBJECTIVE}{o.label}")
        drawn, left_out = _valid(results, axis)
        counted |= left_out
        values = sorted(float(v) for t in drawn if isinstance(v := axis.value(t), (int, float)))
        row, column = index // columns + 1, index % columns + 1
        figure.add_trace(
            go.Scatter(
                x=values,
                y=[(k + 1) / len(values) for k in range(len(values))],
                mode="lines",
                line={"shape": "hv", "width": 2, "color": colors["front"]},
                name=html.escape(axis.called),
                showlegend=False,
                hovertemplate="%{x:.6g}: %{y:.0%} of the trials at most<extra></extra>",
            ),
            row=row,
            col=column,
        )
        figure.update_xaxes(title_text=axis.title(_unit(axis, drawn)), row=row, col=column)
        figure.update_yaxes(
            range=[0, 1.02], tickformat=".0%", row=row, col=column, title_text="trials"
        )
    figure.update_layout(height=rows * _PANEL + 60, margin={"t": 30, "b": 90}, hovermode="closest")
    _annotate(figure, _left_out(counted))
    return figure


def _objective(results: StudyResults, key: str | None) -> _Axis:
    """The objective ``key`` names, the first by default; a parameter is refused."""
    objectives = results.problem.objectives
    axis = _axis(results, key or f"{OBJECTIVE}{objectives[0].label}")
    if axis.objective is None:
        raise VisualizationError(f"{key!r} is a parameter, not an objective")
    return axis


def _valid(results: StudyResults, axis: _Axis) -> tuple[list[TrialResult], Counter[str]]:
    """The trials with a value of ``axis``, and how many of each outcome have none."""
    drawn = [t for t in results.trials if axis.value(t) is not None]
    return drawn, Counter(t.outcome for t in results.trials if axis.value(t) is None)


def _measured(trial: TrialResult, index: int) -> float | int | None:
    assessment = trial.assessment
    if assessment is None or assessment.objectives is None:
        return None
    return assessment.objectives[index].value


def _margin(trial: TrialResult, name: str) -> float | None:
    assessment = trial.assessment
    limits = () if assessment is None else assessment.limits
    return next((limit.margin for limit in limits if limit.name == name), None)


def _dimension(
    parameter: Parameter, trials: Sequence[TrialResult], called: Names
) -> dict[str, Any]:
    """The axis of ``parameter``: its bounds, or its choices in their order."""
    values = [t.candidate.get(parameter.name) for t in trials]
    label = _heading(called.parameter(parameter.name), parameter.unit)
    if parameter.choices is not None:
        choices = list(parameter.choices)
        return {
            "label": label,
            "values": [None if v is None else choices.index(v) for v in values],
            "tickvals": list(range(len(choices))),
            "ticktext": [str(choice) for choice in choices],
            "range": [0, max(len(choices) - 1, 1)],
        }
    assert parameter.bounds is not None
    return {"label": label, "values": values, "range": list(parameter.bounds)}


def _heading(name: str, unit: str | None) -> str:
    return html.escape(f"{name} [{unit}]" if unit and unit != "1" else name)


def _category(parameter: Parameter) -> dict[str, str]:
    return {"type": "category"} if parameter.kind == "categorical" else {}


def _grid(panels: int) -> tuple[int, int]:
    """Columns and rows of ``panels``: :data:`_COLUMNS` side by side at most."""
    columns = max(min(panels, _COLUMNS), 1)
    return columns, max(math.ceil(panels / columns), 1)


def _scale(dark: bool, title: str) -> dict[str, Any]:
    ramp = _RAMPS[dark]
    return {
        "colorscale": [[i / (len(ramp) - 1), step] for i, step in enumerate(ramp)],
        "colorbar": {"title": {"text": title, "side": "right"}, "thickness": 12},
    }


def _left_out(counts: Counter[str], why: str = "Not shown, without a value") -> list[str]:
    if not counts:
        return []
    listed = ", ".join(f"{count} {outcome}" for outcome, count in sorted(counts.items()))
    return [f"{why}: {listed}"]


def _annotate(figure: go.Figure, notes: Sequence[str], empty: str | None = None) -> None:
    """Say ``empty`` (that nothing is drawn) in the plot, if given, and the ``notes`` below it."""
    if empty is not None:
        figure.add_annotation(
            text=empty,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 15},
        )
    if notes:
        figure.add_annotation(
            text="<br>".join(notes),
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            yshift=-60,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font={"size": 12},
        )
