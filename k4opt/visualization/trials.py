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
it dominates; otherwise the figure says that it shows a projection of the front. Colored by a
value (:func:`color_choices`: the trial number, an objective or a numeric parameter), the front
and the feasible trials take the palette's sequential blue, the front as diamonds.

:func:`history_figure` draws an objective against the trial number, the order the trials were
proposed in, each trial as above; for a study of one objective, with the best feasible value so
far as a step line.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from k4opt.evaluation import Scalar
from k4opt.optimization import StudyResults, TrialResult
from k4opt.problem import Objective, Parameter
from k4opt.visualization.errors import VisualizationError
from k4opt.visualization.names import names

OBJECTIVE, PARAMETER = "objective:", "parameter:"
TRIAL = "trial"
"""What :func:`trials_figure` colors its trials by to show the order they were proposed in."""

# the reference palette's categorical slots 1 and 2 and its muted ink, light and dark steps; the
# groups also differ by symbol, so that none is told by color alone
_COLORS = {
    False: {
        "front": "#2a78d6",
        "infeasible": "#eb6834",
        "muted": "#898781",
        "ring": "#fcfcfb",
        "ink": "#0b0b0b",
    },
    True: {
        "front": "#3987e5",
        "infeasible": "#d95926",
        "muted": "#898781",
        "ring": "#1a1a19",
        "ink": "#f0efec",
    },
}
_FAINT = 0.18  # the opacity of a trial a highlight leaves out
# the reference palette's sequential blue, from near the surface to far from it, for marks: from
# step 250 to 700 on the light surface, from step 600 to 100 on the dark one
_RAMPS = {
    False: ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"),
    True: ("#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
}


@dataclass(frozen=True, slots=True)
class _Axis:
    """What an axis shows: an objective (``index`` in the problem's) or a parameter, ``called``
    as the figures call it (:func:`~k4opt.visualization.names`)."""

    key: str
    called: str
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
        shown = f" [{html.escape(unit)}]" if unit is not None and _shown(unit) else ""
        called = html.escape(self.called)
        if self.parameter is not None:
            # a titled parameter's name under it, as the problem and the command line name it
            name = "" if self.parameter.title is None else f" {html.escape(self.parameter.name)}"
            return f"{called}{shown}<br><sup>parameter{name}</sup>"
        objective = self.objective
        assert objective is not None
        where = [objective.sample, objective.evaluator, objective.stage]
        where += [] if objective.title is None else [objective.metric]
        source = " · ".join(html.escape(part) for part in where if part is not None)
        return f"{called}{shown}, {objective.direction}<br><sup>{source}</sup>"


def axis_choices(results: StudyResults) -> tuple[str, ...]:
    """The axes a figure of ``results`` can have: each objective, then each parameter."""
    spec = results.problem
    return tuple(
        [f"{OBJECTIVE}{o.label}" for o in spec.objectives]
        + [f"{PARAMETER}{p.name}" for p in spec.parameters]
    )


def _axis(results: StudyResults, key: str) -> _Axis:
    spec = results.problem
    called = names(spec)
    for index, objective in enumerate(spec.objectives):
        if key == f"{OBJECTIVE}{objective.label}":
            return _Axis(key, called.objective(objective.label), objective=objective, index=index)
    for parameter in spec.parameters:
        if key == f"{PARAMETER}{parameter.name}":
            return _Axis(key, called.parameter(parameter.name), parameter=parameter)
    raise VisualizationError(
        f"the study has no axis {key!r}; it has {', '.join(axis_choices(results))}"
    )


def default_axes(results: StudyResults) -> tuple[str, str]:
    """The axes :func:`trials_figure` shows unless told others: the first two objectives, or, for
    one objective, the first parameter against it."""
    choices = axis_choices(results)
    objectives = len(results.problem.objectives)
    return (choices[0], choices[1]) if objectives >= 2 else (choices[1], choices[0])


def color_choices(results: StudyResults) -> tuple[str, ...]:
    """What :func:`trials_figure` can color its trials by: the trial number (:data:`TRIAL`), each
    objective, and each parameter that is not categorical."""
    spec = results.problem
    return (
        TRIAL,
        *(f"{OBJECTIVE}{o.label}" for o in spec.objectives),
        *(f"{PARAMETER}{p.name}" for p in spec.parameters if p.kind != "categorical"),
    )


def trials_figure(
    results: StudyResults,
    x: str | None = None,
    y: str | None = None,
    *,
    color: str | None = None,
    highlight: Collection[int] | None = None,
    reference: int | None = None,
    dark: bool = False,
) -> go.Figure:
    """The trials of ``results`` with ``x`` and ``y`` (:func:`axis_choices`; by default the first
    two objectives, or the first parameter and the only objective). ``color`` (one of
    :func:`color_choices`) shades the feasible trials by its value, the front set apart by its
    symbol; by default they are told apart by group alone. ``highlight`` (trial numbers) leaves
    every other trial faint, and the trial numbered ``reference`` is ringed: what a trade-off
    (:func:`~k4opt.optimization.trade_offs`) found, and against what. ``dark`` takes the colors
    stepped for a dark background. Raises :class:`VisualizationError` for an axis the study does
    not have, and for values of one objective recorded in different units: they are never drawn
    on one scale."""
    default_x, default_y = default_axes(results)
    horizontal, vertical = _axis(results, x or default_x), _axis(results, y or default_y)
    shade = None if color is None else _shade(results, color)
    groups, left_out = _groups(
        results, lambda t: horizontal.value(t) is not None and vertical.value(t) is not None
    )
    drawn = [trial for group in groups.values() for trial in group]
    units = (_unit(horizontal, drawn), _unit(vertical, drawn))
    figure = go.Figure()
    _mark(
        figure,
        results,
        groups,
        horizontal.value,
        vertical.value,
        shade,
        dark=dark,
        highlight=None if highlight is None else set(highlight) | {reference},
    )
    if reference is not None and reference in {t.number for t in drawn}:
        chosen = results.trial(reference)
        figure.add_trace(
            go.Scatter(
                x=[horizontal.value(chosen)],
                y=[vertical.value(chosen)],
                mode="markers",
                name=f"reference: trial {reference}",
                marker={  # an open symbol is drawn in the marker's color
                    "symbol": "circle-open",
                    "size": 20,
                    "color": _COLORS[dark]["ink"],
                    "line": {"width": 2.5},
                },
                customdata=[reference],
                hovertext=[_hover(results, chosen, set(results.pareto))],
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
                line={
                    "shape": "hv",
                    "width": 2,
                    "color": _COLORS[dark]["front" if shade is None else "muted"],
                },
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
    _shaded(figure, results, shade, drawn, dark=dark)
    _annotate(figure, drawn, notes)
    return figure


def history_figure(
    results: StudyResults, objective: str | None = None, *, dark: bool = False
) -> go.Figure:
    """The values of ``objective`` (``objective:<label>``; by default the first objective) in the
    order the trials were proposed, each trial drawn as :func:`trials_figure` draws it. For a
    study of one objective, a step line follows the best feasible value so far; the front of
    several objectives is set apart by its color instead. Raises :class:`VisualizationError` for
    an objective the study does not have, and for values recorded in different units."""
    key = objective or f"{OBJECTIVE}{results.problem.objectives[0].label}"
    axis = _axis(results, key)
    if axis.objective is None:
        raise VisualizationError(f"the history is of an objective, and {key!r} is a parameter")
    groups, left_out = _groups(results, lambda t: axis.value(t) is not None)
    drawn = [trial for group in groups.values() for trial in group]
    unit = _unit(axis, drawn)
    single = len(results.problem.objectives) == 1
    figure = go.Figure()
    _mark(
        figure,
        results,
        groups,
        lambda t: t.number,
        axis.value,
        None,
        dark=dark,
        front="best" if single else "on the Pareto front",
    )
    if single:
        steps_x: list[int] = []
        steps_y: list[Scalar | None] = []
        best: TrialResult | None = None
        for trial in sorted(drawn, key=lambda t: t.number):
            if trial.outcome == "feasible" and (
                best is None or axis.signed(trial) < axis.signed(best)
            ):
                best = trial
            steps_x.append(trial.number)
            steps_y.append(None if best is None else axis.value(best))
        figure.add_trace(
            go.Scatter(
                x=steps_x,
                y=steps_y,
                mode="lines",
                line={"shape": "hv", "width": 2, "color": _COLORS[dark]["front"]},
                name="best feasible so far",
                hoverinfo="skip",
            )
        )
        figure.data = figure.data[-1:] + figure.data[:-1]  # the line under the markers
    notes = []
    if left_out:
        counts = ", ".join(f"{count} {outcome}" for outcome, count in sorted(left_out.items()))
        notes.append(f"Not shown, without a value: {counts}")
    figure.update_layout(
        xaxis={"title": {"text": "trial"}, "rangemode": "tozero"},
        yaxis={"title": {"text": axis.title(unit)}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hovermode="closest",
        margin={"t": 40, "b": 110 if notes else 70},
    )
    _annotate(figure, drawn, notes)
    return figure


@dataclass(frozen=True, slots=True)
class _Shade:
    """What the trials are colored by: a value of each, named ``title``."""

    key: str
    value: Callable[[TrialResult], Scalar | None]
    axis: _Axis | None = None  # the objective or parameter, if it is one


def _shade(results: StudyResults, key: str) -> _Shade:
    if key == TRIAL:
        return _Shade(key, lambda t: t.number)
    if key not in color_choices(results):
        raise VisualizationError(
            f"the trials cannot be colored by {key!r}; they can be by "
            f"{', '.join(color_choices(results))}"
        )
    axis = _axis(results, key)
    return _Shade(key, axis.value, axis)


def _groups(
    results: StudyResults, shown: Callable[[TrialResult], bool]
) -> tuple[dict[str, list[TrialResult]], Counter[str]]:
    """The trials ``shown``, by how they are drawn, and how many of each outcome are not."""
    front = set(results.pareto)
    groups: dict[str, list[TrialResult]] = {
        "front": [],
        "feasible": [],
        "infeasible": [],
        "none": [],
    }
    left_out: Counter[str] = Counter()
    for trial in results.trials:
        if not shown(trial):
            left_out[trial.outcome] += 1
        elif trial.number in front:
            groups["front"].append(trial)
        elif trial.outcome in ("feasible", "infeasible"):
            groups[trial.outcome].append(trial)
        else:
            groups["none"].append(trial)
    return groups, left_out


def _mark(
    figure: go.Figure,
    results: StudyResults,
    groups: Mapping[str, Sequence[TrialResult]],
    x: Callable[[TrialResult], Scalar | None],
    y: Callable[[TrialResult], Scalar | None],
    shade: _Shade | None,
    *,
    dark: bool,
    front: str = "on the Pareto front",
    highlight: Collection[int | None] | None = None,
) -> None:
    """A trace of markers for each group of trials, each point's ``customdata`` its trial's
    number; the front and the feasible trials shaded by ``shade``, if any; every trial not in
    ``highlight``, if given, faint."""
    colors = _COLORS[dark]
    numbers = set(results.pareto)
    styles = {
        "front": (front, "diamond" if shade else "circle", 12 if shade else 11, colors["front"]),
        "feasible": ("feasible", "circle", 8, colors["muted"]),
        "infeasible": ("infeasible (a limit violated)", "x-thin-open", 9, colors["infeasible"]),
        "none": ("no result", "circle-open", 8, colors["muted"]),
    }
    for group, (name, symbol, size, color) in styles.items():
        trials = groups[group]
        if not trials:
            continue
        filled = symbol in ("circle", "diamond")
        marker: dict[str, Any] = {
            "symbol": symbol,
            "size": size,
            "color": color,
            # a ring of the background around a filled mark, so overlapping ones part
            "line": {"width": 2, "color": colors["ring"]}
            if filled
            else {"width": 1.5, "color": color},
        }
        if shade is not None and group in ("front", "feasible"):
            marker |= {"color": [shade.value(t) for t in trials], "coloraxis": "coloraxis"}
        if highlight is not None:
            marker["opacity"] = [1.0 if t.number in highlight else _FAINT for t in trials]
        figure.add_trace(
            go.Scatter(
                x=[x(t) for t in trials],
                y=[y(t) for t in trials],
                mode="markers",
                name=name,
                marker=marker,
                customdata=[t.number for t in trials],  # which trial a point is
                hovertext=[_hover(results, t, numbers) for t in trials],
                hoverinfo="text",
            )
        )


def _shaded(
    figure: go.Figure,
    results: StudyResults,
    shade: _Shade | None,
    drawn: Sequence[TrialResult],
    *,
    dark: bool,
) -> None:
    """The scale the trials are shaded on: the reference palette's sequential blue, from near
    the surface to far from it."""
    if shade is None:
        return
    if shade.axis is None:
        title = "trial"
    else:
        shaded = [t for t in drawn if t.outcome == "feasible" or t.number in results.pareto]
        title = shade.axis.title(_unit(shade.axis, shaded)).split("<br>")[0]
    ramp = _RAMPS[dark]
    figure.update_layout(
        coloraxis={
            "colorscale": [[i / (len(ramp) - 1), step] for i, step in enumerate(ramp)],
            "colorbar": {"title": {"text": title, "side": "right"}, "thickness": 12},
        }
    )


def _annotate(figure: go.Figure, drawn: Sequence[TrialResult], notes: Sequence[str]) -> None:
    """Say that no trial is drawn, if none is, and the ``notes`` below the plot."""
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
    return f"{text} {html.escape(unit)}" if unit is not None and _shown(unit) else text


def _shown(unit: str | None) -> bool:
    """Whether ``unit`` is shown: ``1`` is a number of no unit."""
    return unit not in (None, "", "1")


def _hover(results: StudyResults, trial: TrialResult, front: set[int]) -> str:
    """Everything a trial was measured with and found: its candidate, its objectives with their
    uncertainties, and the limits it violates."""
    state = trial.outcome + (", on the Pareto front" if trial.number in front else "")
    lines = [f"<b>trial {trial.number}</b>: {state}"]
    called = names(results.problem)
    for parameter in results.problem.parameters:
        if parameter.name in trial.candidate:
            value = _number(trial.candidate[parameter.name], parameter.unit)
            lines.append(f"{html.escape(called.parameter(parameter.name))} = {value}")
    assessment = trial.assessment
    if assessment is not None and assessment.objectives is not None:
        for objective in assessment.objectives:
            value = _number(objective.value, objective.unit, objective.uncertainty)
            lines.append(f"{html.escape(called.objective(objective.label))} = {value}")
        lines.extend(
            f"violates {html.escape(called.constraint(limit.label))} {limit.bound} "
            f"{_number(limit.limit, limit.unit)} by {_number(limit.margin, limit.unit)}"
            for limit in assessment.limits
            if limit.satisfied is False and limit.margin is not None
        )
    elif trial.message:
        lines.append(html.escape(trial.message[:200]))
    return "<br>".join(lines)
