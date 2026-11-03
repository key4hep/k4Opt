"""The trials of a study in the plane of two of its objectives or parameters.

An axis is named as the columns of ``k4opt study export --format csv`` name the values:
``objective:<label>`` (``objective:gamma/calo.photon/simulation/ecal_barrel.angular_residual``)
or ``parameter:<name>``. A trial is drawn when it has a value on both axes, as one of

* **on the Pareto front**: the study's own front (:attr:`StudyResults.pareto`), of every
  objective and limit, whichever two axes are shown;
* **feasible**, dominated by a trial on the front;
* **infeasible**: valid, but a limit is violated (the hover says which, and by how much);
* **no result**: invalid, failed, rejected, interrupted, an error or running; such a trial has
  no objective value, so it appears only where both axes are parameters.

Every other trial is counted below the plot, by outcome. When the axes are exactly the two
objectives of a two-objective study, the front is joined by its step line, the boundary of what
it dominates; otherwise the figure says that it shows a projection of the front.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import plotly.graph_objects as go

from k4opt.evaluation import Scalar
from k4opt.optimization import StudyResults, TrialResult
from k4opt.problem import Objective, Parameter
from k4opt.visualization.errors import VisualizationError

OBJECTIVE, PARAMETER = "objective:", "parameter:"

# the reference palette's categorical slots 1 and 2 and its muted ink, light and dark steps; the
# groups also differ by symbol, so that none is told by color alone
_COLORS = {
    False: {"front": "#2a78d6", "infeasible": "#eb6834", "muted": "#898781", "ring": "#fcfcfb"},
    True: {"front": "#3987e5", "infeasible": "#d95926", "muted": "#898781", "ring": "#1a1a19"},
}


@dataclass(frozen=True, slots=True)
class _Axis:
    """What an axis shows: an objective (``index`` in the problem's) or a parameter."""

    key: str
    objective: Objective | None = None
    index: int = -1
    parameter: Parameter | None = None

    def value(self, trial: TrialResult) -> Scalar | None:
        if self.parameter is not None:
            return trial.candidate.get(self.parameter.name)
        assessment = trial.assessment
        if assessment is None or assessment.objectives is None:
            return None
        return assessment.objectives[self.index].value

    def unit(self, trial: TrialResult) -> str | None:
        if self.parameter is not None:
            return self.parameter.unit
        assert trial.assessment is not None
        assert trial.assessment.objectives is not None
        return trial.assessment.objectives[self.index].unit

    def signed(self, trial: TrialResult) -> float:
        """The objective's value of ``trial``, negated if larger is better: smaller is better."""
        value = self.value(trial)
        assert self.objective is not None
        assert isinstance(value, (int, float))
        return -value if self.objective.direction == "maximize" else value

    def title(self, unit: str | None) -> str:
        shown = f" [{html.escape(unit)}]" if unit else ""
        if self.parameter is not None:
            return f"{html.escape(self.parameter.name)}{shown}<br><sup>parameter</sup>"
        objective = self.objective
        assert objective is not None
        where = [objective.sample, objective.evaluator, objective.stage]
        source = " · ".join(html.escape(part) for part in where if part is not None)
        return (
            f"{html.escape(objective.metric)}{shown}, {objective.direction}<br><sup>{source}</sup>"
        )


def axis_choices(results: StudyResults) -> tuple[str, ...]:
    """The axes a figure of ``results`` can have: each objective, then each parameter."""
    spec = results.problem
    return tuple(
        [f"{OBJECTIVE}{o.label}" for o in spec.objectives]
        + [f"{PARAMETER}{p.name}" for p in spec.parameters]
    )


def _axis(results: StudyResults, key: str) -> _Axis:
    spec = results.problem
    for index, objective in enumerate(spec.objectives):
        if key == f"{OBJECTIVE}{objective.label}":
            return _Axis(key, objective=objective, index=index)
    for parameter in spec.parameters:
        if key == f"{PARAMETER}{parameter.name}":
            return _Axis(key, parameter=parameter)
    raise VisualizationError(
        f"the study has no axis {key!r}; it has {', '.join(axis_choices(results))}"
    )


def default_axes(results: StudyResults) -> tuple[str, str]:
    """The axes :func:`trials_figure` shows unless told others: the first two objectives, or, for
    one objective, the first parameter against it."""
    choices = axis_choices(results)
    objectives = len(results.problem.objectives)
    return (choices[0], choices[1]) if objectives >= 2 else (choices[1], choices[0])


def trials_figure(
    results: StudyResults, x: str | None = None, y: str | None = None, *, dark: bool = False
) -> go.Figure:
    """The trials of ``results`` with ``x`` and ``y`` (:func:`axis_choices`; by default the first
    two objectives, or the first parameter and the only objective). ``dark`` takes the colors
    stepped for a dark background. Raises :class:`VisualizationError` for an axis the study does
    not have, and for values of one objective recorded in different units: they are never drawn
    on one scale."""
    default_x, default_y = default_axes(results)
    horizontal, vertical = _axis(results, x or default_x), _axis(results, y or default_y)
    colors = _COLORS[dark]
    front = set(results.pareto)
    groups: dict[str, list[TrialResult]] = {
        "front": [],
        "feasible": [],
        "infeasible": [],
        "none": [],
    }
    left_out: Counter[str] = Counter()
    for trial in results.trials:
        if horizontal.value(trial) is None or vertical.value(trial) is None:
            left_out[trial.outcome] += 1
        elif trial.number in front:
            groups["front"].append(trial)
        elif trial.outcome in ("feasible", "infeasible"):
            groups[trial.outcome].append(trial)
        else:
            groups["none"].append(trial)
    drawn = [trial for group in groups.values() for trial in group]
    units = (_unit(horizontal, drawn), _unit(vertical, drawn))
    figure = go.Figure()
    styles = {
        "front": ("on the Pareto front", "circle", 11, colors["front"]),
        "feasible": ("feasible", "circle", 8, colors["muted"]),
        "infeasible": ("infeasible (a limit violated)", "x-thin-open", 9, colors["infeasible"]),
        "none": ("no result", "circle-open", 8, colors["muted"]),
    }
    for group, (name, symbol, size, color) in styles.items():
        trials = groups[group]
        if not trials:
            continue
        figure.add_trace(
            go.Scatter(
                x=[horizontal.value(t) for t in trials],
                y=[vertical.value(t) for t in trials],
                mode="markers",
                name=name,
                marker={
                    "symbol": symbol,
                    "size": size,
                    "color": color,
                    # a ring of the background around a filled mark, so overlapping ones part
                    "line": {"width": 2, "color": colors["ring"]}
                    if symbol == "circle"
                    else {"width": 1.5, "color": color},
                },
                customdata=[t.number for t in trials],  # which trial a point is
                hovertext=[_hover(results, t, front) for t in trials],
                hoverinfo="text",
            )
        )
    objectives = results.problem.objectives
    exact = (
        len(objectives) == 2
        and horizontal.objective is not None
        and vertical.objective is not None
        and horizontal.index != vertical.index
    )
    if exact and len(groups["front"]) > 1:
        # in the order of the first objective, each to be minimized: the step from one point
        # to the next, along x first, bounds what the front dominates, in either direction
        points = sorted(
            groups["front"],
            key=lambda t: (horizontal.signed(t), vertical.signed(t)),
        )
        figure.add_trace(
            go.Scatter(
                x=[horizontal.value(t) for t in points],
                y=[vertical.value(t) for t in points],
                mode="lines",
                line={"shape": "hv", "width": 2, "color": colors["front"]},
                name="Pareto front",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.data = figure.data[-1:] + figure.data[:-1]  # the line under the markers
    notes = _notes(results, groups["front"], exact, left_out)
    figure.update_layout(
        title={"text": html.escape(results.name)},
        xaxis={"title": {"text": horizontal.title(units[0])}, **_kind(horizontal)},
        yaxis={"title": {"text": vertical.title(units[1])}, **_kind(vertical)},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hovermode="closest",
        margin={"b": 110 if notes else 70},
    )
    if not drawn:
        figure.add_annotation(
            text="No trial has values on these axes yet",
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
            y=-0.24,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font={"size": 12},
        )
    return figure


def _kind(axis: _Axis) -> dict[str, str]:
    """A categorical parameter on a category axis, its choices as written."""
    if axis.parameter is not None and axis.parameter.kind == "categorical":
        return {"type": "category"}
    return {}


def _unit(axis: _Axis, trials: Sequence[TrialResult]) -> str | None:
    units = {axis.unit(trial) for trial in trials}
    if len(units) > 1:
        raise VisualizationError(
            f"{axis.key} is recorded in several units ({', '.join(sorted(map(str, units)))}); "
            "values in different units are not drawn on one scale"
        )
    return next(iter(units), None)


def _notes(
    results: StudyResults, front: list[TrialResult], exact: bool, left_out: Counter[str]
) -> list[str]:
    notes = []
    objectives, limits = len(results.problem.objectives), len(results.problem.constraints)
    if front and not exact:
        judged = f"its {objectives} objective{'s' if objectives > 1 else ''}"
        if limits:
            judged += f" and {limits} constraint{'s' if limits > 1 else ''}"
        notes.append(f"Pareto front of {judged}, projected on these axes")
    if left_out:
        counts = ", ".join(f"{count} {outcome}" for outcome, count in sorted(left_out.items()))
        notes.append(f"Not shown, without a value on these axes: {counts}")
    return notes


def _number(value: Scalar, unit: str | None, uncertainty: float | None = None) -> str:
    text = f"{value:.6g}" if isinstance(value, float) else html.escape(str(value))
    if uncertainty is not None:
        text += f" ± {uncertainty:.3g}"
    return f"{text} {html.escape(unit)}" if unit else text


def _hover(results: StudyResults, trial: TrialResult, front: set[int]) -> str:
    """Everything a trial was measured with and found: its candidate, its objectives with their
    uncertainties, and the limits it violates."""
    state = trial.outcome + (", on the Pareto front" if trial.number in front else "")
    lines = [f"<b>trial {trial.number}</b>: {state}"]
    for parameter in results.problem.parameters:
        if parameter.name in trial.candidate:
            value = _number(trial.candidate[parameter.name], parameter.unit)
            lines.append(f"{html.escape(parameter.name)} = {value}")
    assessment = trial.assessment
    if assessment is not None and assessment.objectives is not None:
        for objective in assessment.objectives:
            value = _number(objective.value, objective.unit, objective.uncertainty)
            lines.append(f"{html.escape(objective.label)} = {value}")
        lines.extend(
            f"violates {html.escape(limit.label)} {limit.bound} "
            f"{_number(limit.limit, limit.unit)} by {_number(limit.margin, limit.unit)}"
            for limit in assessment.limits
            if limit.satisfied is False and limit.margin is not None
        )
    elif trial.message:
        lines.append(html.escape(trial.message[:200]))
    return "<br>".join(lines)
