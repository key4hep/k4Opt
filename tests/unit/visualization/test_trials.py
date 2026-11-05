"""``trials_figure``: every trial where its values put it, as the study judged it (feasible,
infeasible, on its Pareto front), with its units, and what it cannot show said, not hidden."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("plotly")

from k4opt.frozen import FrozenMap
from k4opt.optimization import StudyResults, TrialResult
from k4opt.optimization._results import pareto_front
from k4opt.problem import parse_problem
from k4opt.record import Assessment, Limit, ObjectiveValue
from k4opt.visualization import (
    TRIAL,
    VisualizationError,
    axis_choices,
    color_choices,
    trials_figure,
)
from tests.unit.optimization.conftest import RADIUS, problem_document

REPOSITORY = Path(__file__).resolve().parents[3]
EVALUATOR = "test.trade_off"


def study_of(
    directions: tuple[str, ...],
    trials: list[tuple[str, list[float] | None]],
    *,
    units: list[str] | None = None,
    constraint: float | None = None,
    parameter: dict[str, Any] | None = None,
) -> StudyResults:
    """A study of objectives ``analysis/m0``, ``m1``, ... in these directions, each trial
    ``(outcome, values)``; with ``constraint``, the limit ``m0 <= constraint``."""
    objectives = [("analysis", f"m{i}", d) for i, d in enumerate(directions)]
    limits = (
        [] if constraint is None else [{"stage": "analysis", "metric": "m0", "max": constraint}]
    )
    spec = parse_problem(
        problem_document(objectives=objectives, constraints=limits, parameter=parameter)
    )
    results = []
    for number, (outcome, values) in enumerate(trials):
        status = "valid" if outcome in ("feasible", "infeasible") else outcome
        candidate = {
            RADIUS: spec.parameters[0].choices[number % 2] if parameter else 100.0 + number
        }
        if values is None:
            results.append(TrialResult(number, status, candidate, message=f"{outcome} here"))
            continue
        measured = tuple(
            ObjectiveValue(EVALUATOR, "analysis", f"m{i}", d, v, (units or ["mm"] * 9)[number])
            for i, (d, v) in enumerate(zip(directions, values, strict=True))
        )
        bound = (
            ()
            if constraint is None
            else (
                Limit(
                    EVALUATOR,
                    "analysis",
                    "m0",
                    "max",
                    constraint,
                    "mm",
                    values[0],
                    None,
                    values[0] - constraint,
                    outcome == "feasible",
                ),
            )
        )
        assessment = Assessment(measured, bound, outcome == "feasible")
        results.append(
            TrialResult(number, status, candidate, assessment, record="r", sha256="0" * 64)
        )
    return StudyResults(
        "trade-0", "id", "now", spec, FrozenMap({}), tuple(results), pareto_front(results, spec)
    )


def traces(figure: Any) -> dict[str, Any]:
    return {trace.name: trace for trace in figure.data}


def numbers(trace: Any) -> list[int]:
    return list(trace.customdata)


def notes(figure: Any) -> str:
    return " ".join(annotation.text for annotation in figure.layout.annotations)


@pytest.mark.parametrize(
    "directions",
    [
        ("minimize", "minimize"),
        ("minimize", "maximize"),
        ("maximize", "minimize"),
        ("maximize", "maximize"),
    ],
)
def test_two_objectives_show_the_front_and_its_step_line_in_their_directions(
    directions: tuple[str, str],
) -> None:
    signs = [1 if d == "minimize" else -1 for d in directions]
    points = [[1, 9], [2, 4], [4, 2], [9, 1], [5, 5], [8, 8]]  # a front of 4, in minimize space
    study = study_of(
        directions, [("feasible", [s * v for s, v in zip(signs, p, strict=True)]) for p in points]
    )
    assert study.pareto == (0, 1, 2, 3)
    figure = trials_figure(study)
    shown = traces(figure)
    assert numbers(shown["on the Pareto front"]) == [0, 1, 2, 3]
    assert numbers(shown["feasible"]) == [4, 5]
    line = shown["Pareto front"]
    assert line.line.shape == "hv" and figure.data[0] is line  # under the markers
    ordered = sorted(points[:4])  # by the first objective, to be minimized
    assert list(line.x) == [signs[0] * p[0] for p in ordered]
    assert list(line.y) == [signs[1] * p[1] for p in ordered]
    assert not notes(figure)  # the whole front, no projection


def test_three_objectives_are_a_projection_of_the_whole_front() -> None:
    study = study_of(
        ("minimize",) * 3,
        [("feasible", [1, 1, 9]), ("feasible", [2, 2, 1]), ("feasible", [3, 3, 0])],
    )
    figure = trials_figure(study)  # m0 against m1: trial 2 is dominated there, not in all three
    assert study.pareto == (0, 1, 2)
    assert numbers(traces(figure)["on the Pareto front"]) == [0, 1, 2]
    assert "Pareto front" not in [t.name for t in figure.data if t.mode == "lines"]
    assert "Pareto front of its 3 objectives, projected on these axes" in notes(figure)


def test_one_objective_is_shown_against_the_first_parameter() -> None:
    study = study_of(("maximize",), [("feasible", [1.0]), ("feasible", [3.0]), ("feasible", [2.0])])
    figure = trials_figure(study)
    assert figure.layout.xaxis.title.text.startswith(f"{RADIUS}<br>")
    assert figure.layout.yaxis.title.text.startswith("m0 [mm], maximize<br>")
    assert numbers(traces(figure)["on the Pareto front"]) == [1]  # the best one


def test_infeasible_trials_and_those_without_a_result_are_told_apart() -> None:
    study = study_of(
        ("minimize", "minimize"),
        [
            ("feasible", [1, 5]),
            ("infeasible", [6, 1]),
            ("feasible", [2, 6]),
            ("invalid", None),
            ("failed", None),
            ("running", None),
        ],
        constraint=4,
    )
    assert study.pareto == (0,)  # the infeasible trial 1 would win in b, but is not feasible
    figure = trials_figure(study)
    shown = traces(figure)
    infeasible = shown["infeasible (a limit violated)"]
    assert numbers(infeasible) == [1] and infeasible.marker.symbol == "x-thin-open"
    assert "violates m0 max 4 mm by 2 mm" in infeasible.hovertext[0]
    assert numbers(shown["feasible"]) == [2]
    assert "Not shown, without a value on these axes: 1 failed, 1 invalid, 1 running" in notes(
        figure
    )
    assert "Pareto front of its 2 objectives and 1 constraint, projected" not in notes(figure)
    parameters = trials_figure(study, f"parameter:{RADIUS}", f"parameter:{RADIUS}")
    assert numbers(traces(parameters)["no result"]) == [3, 4, 5]  # where they were proposed
    assert "invalid here" in traces(parameters)["no result"].hovertext[0]


def test_the_axes_carry_their_units_and_the_hover_every_value() -> None:
    study = study_of(("minimize", "minimize"), [("feasible", [1.5, 2.0])])
    figure = trials_figure(study)
    assert figure.layout.xaxis.title.text == (
        "m0 [mm], minimize<br><sup>test.trade_off · analysis</sup>"
    )
    hover = traces(figure)["on the Pareto front"].hovertext[0]
    assert hover.startswith("<b>trial 0</b>: feasible, on the Pareto front")
    assert f"{RADIUS} = 100" in hover and "m0 = 1.5 mm" in hover


def test_values_in_two_units_are_not_drawn_on_one_axis() -> None:
    study = study_of(
        ("minimize", "minimize"), [("feasible", [1, 2]), ("feasible", [2, 1])], units=["mm", "cm"]
    )
    with pytest.raises(VisualizationError, match=r"several units \(cm, mm\)"):
        trials_figure(study)


def test_an_axis_the_study_does_not_have_is_refused() -> None:
    study = study_of(("minimize", "minimize"), [("feasible", [1, 2])])
    assert axis_choices(study) == (
        "objective:test.trade_off/analysis/m0",
        "objective:test.trade_off/analysis/m1",
        f"parameter:{RADIUS}",
    )
    with pytest.raises(VisualizationError, match="has no axis 'objective:m2'"):
        trials_figure(study, "objective:m2")


def test_a_study_without_trials_says_so() -> None:
    figure = trials_figure(study_of(("minimize", "minimize"), []))
    assert not [trace for trace in figure.data if trace.mode == "markers"]
    assert "No trial has values on these axes yet" in notes(figure)
    one = trials_figure(study_of(("minimize", "minimize"), [("feasible", [1, 2])]))
    assert numbers(traces(one)["on the Pareto front"]) == [0]
    assert "Pareto front" not in traces(one)  # no line for one point


def test_a_categorical_parameter_is_a_category_axis() -> None:
    study = study_of(
        ("minimize",),
        [("feasible", [2]), ("feasible", [1])],
        parameter={
            "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": ["thin", "thick"],
        },
    )
    figure = trials_figure(study)
    assert figure.layout.xaxis.type == "category"
    assert sorted(x for t in figure.data for x in t.x) == ["thick", "thin"]


def test_a_real_study_is_drawn_from_its_published_snapshot_without_optuna(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    from k4opt.optimization import load_study
    from k4opt.publication import publish
    from tests.support.studies import mixed_study

    results = load_study(mixed_study(tmp_path / "s"))
    publish(results, tmp_path / "pub")
    expected = without_template(trials_figure(results).to_json())
    code = f"""
import sys
sys.modules["optuna"] = None  # importing it fails
from k4opt.publication import open_publication
from k4opt.visualization import trials_figure
publication = open_publication({str(tmp_path / "pub")!r})
(entry,) = publication.catalog().studies
print(trials_figure(publication.study(entry)).to_json(), end="")
assert not [m for m, held in sys.modules.items() if m.startswith("optuna") and held]
"""
    path = os.pathsep.join([str(REPOSITORY), os.environ.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"PYTHONPATH": path},
    )
    assert result.returncode == 0, result.stderr
    assert without_template(result.stdout) == expected  # the figure of the study itself


def without_template(figure: str) -> dict[str, Any]:
    """A figure's data and layout, without the template a process makes the default (Streamlit
    sets its own when it is imported)."""
    data: dict[str, Any] = json.loads(figure)
    data["layout"].pop("template", None)
    return data


def test_the_feasible_trials_are_shaded_by_a_value_and_the_front_is_a_diamond() -> None:
    study = study_of(
        ("minimize", "minimize"),
        [("feasible", [1, 9]), ("feasible", [9, 1]), ("feasible", [9, 9]), ("infeasible", [0, 0])],
        constraint=10,
    )
    assert color_choices(study) == (
        TRIAL,
        "objective:test.trade_off/analysis/m0",
        "objective:test.trade_off/analysis/m1",
        "parameter:radius",
    )
    plain, shaded = trials_figure(study), trials_figure(study, color="parameter:radius")
    assert plain.layout.coloraxis.colorscale is None
    front, feasible = traces(shaded)["on the Pareto front"], traces(shaded)["feasible"]
    assert front.marker.symbol == "diamond" and front.marker.coloraxis == "coloraxis"
    assert list(front.marker.color) == [100.0, 101.0] and list(feasible.marker.color) == [102.0]
    infeasible = traces(shaded)["infeasible (a limit violated)"]
    assert infeasible.marker.coloraxis is None  # its own color: it is not to be compared
    assert shaded.layout.coloraxis.colorbar.title.text == RADIUS  # its title, no source line
    assert traces(shaded)["Pareto front"].line.color == "#898781"  # the shade is the points'
    by_number = trials_figure(study, color=TRIAL, dark=True)
    assert list(traces(by_number)["feasible"].marker.color) == [2]
    assert by_number.layout.coloraxis.colorscale[0][1] == "#184f95"  # near the dark surface


def test_a_categorical_parameter_or_an_unknown_value_shades_nothing() -> None:
    parameter = {"type": "categorical", "choices": [100, 200]}
    study = study_of(("minimize",), [("feasible", [1.0]), ("feasible", [2.0])], parameter=parameter)
    assert color_choices(study) == (TRIAL, "objective:test.trade_off/analysis/m0")
    for key in ("parameter:radius", "objective:nothing"):
        with pytest.raises(VisualizationError, match="cannot be colored by"):
            trials_figure(study, color=key)


def test_a_highlight_leaves_the_other_trials_faint_and_rings_the_reference() -> None:
    study = study_of(
        ("minimize", "minimize"),
        [("feasible", [1, 9]), ("feasible", [9, 1]), ("feasible", [9, 9]), ("infeasible", [0, 0])],
        constraint=8,
    )
    figure = trials_figure(study, highlight=[1], reference=0)
    shown = traces(figure)
    assert list(shown["on the Pareto front"].marker.opacity) == [1.0, 1.0]  # 0 the reference
    assert list(shown["feasible"].marker.opacity) == [0.18]
    ring = shown["reference: trial 0"]
    assert numbers(ring) == [0] and ring.marker.symbol == "circle-open"
    assert ring.marker.color == "#0b0b0b"  # the ink: an open symbol takes the marker's color
    assert traces(trials_figure(study))["feasible"].marker.opacity is None


def test_a_study_of_a_level_below_is_drawn_by_the_objective_it_is_judged_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("optuna")
    from k4opt.optimization import load_study
    from k4opt.visualization import axis_choices, default_axes, history_figure
    from tests.support.studies import nested_study

    inner = load_study(nested_study(tmp_path / "s", monkeypatch)).trial(0).inner
    assert inner is not None
    seen = "objective:test.candidate_check/analysis/seen"
    objectives = [key for key in axis_choices(inner) if key.startswith("objective:")]
    assert objectives == [seen]  # not the cells the level above compares
    assert default_axes(inner) == ("parameter:radius", seen)
    trials_figure(inner)
    history_figure(inner)


def test_the_model_of_a_study_of_a_level_below_is_what_that_level_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("optuna")
    from k4opt.optimization import load_study
    from k4opt.visualization import task_graph
    from tests.support.studies import nested_study

    top = load_study(nested_study(tmp_path / "s", monkeypatch))
    inner = top.trial(0).inner
    assert inner is not None
    graph = {node.id: node for node in task_graph(inner, stack=False).nodes}
    measured = [id for id in graph if id.startswith(("objective:", "constraint:"))]
    assert measured == ["objective:test.candidate_check/analysis/seen"]
    # what only the level above reads is not run at this level
    assert "gamma/test.cell_count" not in graph and "gamma/simulation" not in graph
    radius = top.trial(0).candidate["radius"]
    assert graph["parameter:radius"].details == (f"fixed: {radius:.4g}",)  # no unit
    whole = {node.id for node in task_graph(top, stack=False).nodes}
    assert {"gamma/test.cell_count", "gamma/simulation"} <= whole
