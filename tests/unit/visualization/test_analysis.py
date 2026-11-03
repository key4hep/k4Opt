"""The analysis figures: parallel coordinates, slices, ranks over pairs of parameters and the
distribution of each objective; each drawn from the study's own values, nothing interpolated,
and what one leaves out counted."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("plotly")

from k4opt.optimization import StudyResults, TrialResult
from k4opt.optimization._results import pareto_front
from k4opt.problem import parse_problem
from k4opt.visualization import (
    VisualizationError,
    edf_figure,
    parallel_figure,
    rank_figure,
    slice_figure,
)
from tests.unit.optimization.conftest import RADIUS, problem_document
from tests.unit.visualization.test_trials import numbers, study_of, traces

OUTCOMES = [
    ("feasible", [1.0, 9.0]),
    ("feasible", [3.0, 4.0]),
    ("infeasible", [9.0, 1.0]),
    ("failed", None),
]


def test_parallel_coordinates_draw_the_feasible_trials_through_every_value() -> None:
    study = study_of(("minimize", "maximize"), OUTCOMES, constraint=5)
    figure = parallel_figure(study)
    (lines,) = figure.data
    labels = [dimension.label for dimension in lines.dimensions]
    assert labels == [RADIUS, "m0 [mm]", "m1 [mm]", "margin m0 max"]
    values = {dimension.label: list(dimension.values) for dimension in lines.dimensions}
    assert values[RADIUS] == [100.0, 101.0] and values["m0 [mm]"] == [1.0, 3.0]
    assert values["margin m0 max"] == [-4.0, -2.0]  # satisfied: zero or below
    assert list(lines.dimensions[0].range) == [100.0, 600.0]  # the parameter's bounds
    assert list(lines.line.color) == [1.0, 3.0] and lines.line.reversescale  # smaller is better
    assert (
        not parallel_figure(study, "objective:test.trade_off/analysis/m1").data[0].line.reversescale
    )
    assert "Not drawn, not feasible: 1 failed, 1 infeasible" in figure.layout.title.text


def test_a_categorical_parameter_is_an_axis_of_its_choices() -> None:
    parameter = {"type": "categorical", "choices": [300, 100]}
    study = study_of(("minimize",), [("feasible", [1.0]), ("feasible", [2.0])], parameter=parameter)
    (lines,) = parallel_figure(study).data
    axis = lines.dimensions[0]
    assert list(axis.ticktext) == ["300", "100"] and list(axis.values) == [0, 1]


def test_a_slice_shares_the_objective_across_the_parameters_and_marks_the_infeasible() -> None:
    study = study_of(("minimize", "minimize"), OUTCOMES, constraint=5)
    figure = slice_figure(study, "objective:test.trade_off/analysis/m1")
    shown = traces(figure)
    assert numbers(shown["feasible"]) == [0, 1] and list(shown["feasible"].y) == [9.0, 4.0]
    assert list(shown["feasible"].marker.color) == [0, 1]  # by trial number
    assert numbers(shown["infeasible (a limit violated)"]) == [2]
    assert figure.layout.xaxis.title.text == RADIUS
    assert figure.layout.yaxis.title.text.startswith("m1 [mm], minimize")
    assert "Not shown, without a value: 1 failed" in figure.layout.annotations[0].text


def two_parameters(trials: list[tuple[str, float, float, float]]) -> StudyResults:
    """A study of the objective ``m0`` (minimized) over ``radius`` and ``length``, each trial
    ``(outcome, radius, length, m0)``; the limit ``m0 <= 5``."""
    base = study_of(("minimize",), [("feasible", [0.0])], constraint=5)
    document = problem_document(
        objectives=[("analysis", "m0", "minimize")],
        constraints=[{"stage": "analysis", "metric": "m0", "max": 5}],
    )
    document["parameters"]["length"] = {
        "targets": {"geometry": "InnerTracker_Barrel_half_length_1"},
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "unit": "m",
    }
    spec = parse_problem(document)
    template = base.trials[0]
    assert template.assessment is not None
    results = []
    for number, (outcome, radius, length, value) in enumerate(trials):
        objective = replace(template.assessment.objectives[0], value=value)
        limit = replace(template.assessment.limits[0], value=value, satisfied=value <= 5)
        assessment = replace(
            template.assessment, objectives=(objective,), limits=(limit,), feasible=value <= 5
        )
        assert outcome == ("feasible" if value <= 5 else "infeasible")
        results.append(TrialResult(number, "valid", {RADIUS: radius, "length": length}, assessment))
    return replace(base, problem=spec, trials=tuple(results), pareto=pareto_front(results, spec))


def test_a_rank_colors_the_feasible_trials_by_how_they_rank_over_each_pair() -> None:
    study = two_parameters(
        [
            ("feasible", 100.0, 1.0, 3.0),
            ("feasible", 200.0, 1.5, 1.0),
            ("feasible", 300.0, 2.0, 2.0),
            ("infeasible", 400.0, 1.2, 9.0),
        ]
    )
    figure = rank_figure(study)
    shown = traces(figure)
    assert list(shown["feasible"].marker.color) == [0.0, 1.0, 0.5]  # worst, best, between
    assert list(shown["feasible"].x) == [100.0, 200.0, 300.0]
    assert list(shown["feasible"].y) == [1.0, 1.5, 2.0]
    assert numbers(shown["infeasible (a limit violated)"]) == [3]
    assert figure.layout.yaxis.title.text == "length [m]"
    assert list(figure.layout.coloraxis.colorbar.ticktext) == ["worst", "best"]


def test_a_rank_needs_two_parameters_and_an_objective() -> None:
    study = study_of(("minimize",), [("feasible", [1.0])])
    with pytest.raises(VisualizationError, match="needs two parameters at least"):
        rank_figure(study)
    with pytest.raises(VisualizationError, match="is a parameter, not an objective"):
        slice_figure(study, "parameter:radius")


def test_the_distribution_of_each_objective_counts_the_trials_with_a_value() -> None:
    study = study_of(("minimize", "maximize"), OUTCOMES, constraint=5)
    figure = edf_figure(study)
    first, second = figure.data
    assert list(first.x) == [1.0, 3.0, 9.0] and list(first.y) == pytest.approx([1 / 3, 2 / 3, 1])
    assert list(second.x) == [1.0, 4.0, 9.0] and first.line.shape == "hv"
    assert "Not shown, without a value: 1 failed" in figure.layout.annotations[-1].text


def test_trials_without_values_leave_the_figures_empty_but_said() -> None:
    study = study_of(("minimize",), [("failed", None)])
    assert "No trial is feasible yet" in parallel_figure(study).layout.title.text
    texts = [a.text for a in slice_figure(study).layout.annotations]
    assert "No trial has a value of it yet" in texts
