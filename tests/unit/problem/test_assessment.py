"""The assessment of a candidate: objectives, limits judged on the central value, feasibility;
and a run record checked against its own measurements (``Assessment.from_summary``)."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.evaluation import EvaluationRequest, EvaluationResult, Measurement
from k4opt.problem import Assessment, EvaluationSession, ProblemSpec, RecordError, parse_problem
from k4opt.problem._assess import assess
from k4opt.problem._evaluate import _evaluate_candidate
from tests.support.geometry import MINI_COMPACT, MINI_GEO

CELL = "constants.ECal_cell_size"


def document(
    constraints: list[dict[str, Any]],
    objectives: list[tuple[str, str]],
    evaluators: list[str],
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": "k4opt.problem/v1alpha1",
        "name": "judged",
        "environment": {"key4hep": {"channel": "nightly", "version": "2026-10-20"}},
        "geometry": "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml",
        "parameters": [
            {
                "geometry": {"constant": "ECal_cell_size"},
                "type": "continuous",
                "bounds": [1.0, 10.0],
                "unit": "mm",
            }
        ],
        "evaluators": [{"name": name, "version": "1"} for name in evaluators],
        "objectives": [{"stage": s, "metric": m, "direction": "minimize"} for s, m in objectives],
    }
    if constraints:
        data["constraints"] = [{"stage": "analysis", "unit": "mrad"} | c for c in constraints]
    return data


def problem(
    *constraints: dict[str, Any],
    objectives: tuple[tuple[str, str], ...] = (("analysis", "a"),),
    evaluators: tuple[str, ...] = ("test.fixed",),
) -> ProblemSpec:
    """A problem measured by evaluators that read no stage, with these constraints on
    ``analysis`` metrics (in ``mrad`` unless they say otherwise)."""
    return parse_problem(document(list(constraints), list(objectives), list(evaluators)))


class Fixed:
    """Returns the result it was made with."""

    name = "test.fixed"
    version = "1"

    def __init__(self, result: EvaluationResult) -> None:
        self.result = result

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return self.result


def a(value: float, uncertainty: float | None = None) -> Measurement:
    return Measurement("analysis", "a", value, "mrad", uncertainty)


def judged(
    spec: ProblemSpec, *measurements: Measurement, status: str = "valid", represents: bool = True
) -> Assessment:
    """The assessment of a candidate whose only evaluator returned these measurements."""
    message = None if status == "valid" else f"the evaluator says {status}"
    result = EvaluationResult(status, measurements, message=message)  # type: ignore[arg-type]
    record = _evaluate_candidate(spec, {CELL: 5.0}, [Fixed(result)])
    if not represents:  # as for a stage supplied with --input
        record = dataclasses.replace(record, represents_candidate=False)
    return assess(spec, record)


# --- limits ----------------------------------------------------------------------------------


def test_a_satisfied_maximum_has_a_margin_of_zero_or_below() -> None:
    assessment = judged(problem({"metric": "a", "max": 1.0}), a(0.9))
    (limit,) = assessment.limits
    assert (limit.name, limit.bound, limit.limit, limit.unit) == (
        "analysis/a:max",
        "max",
        1.0,
        "mrad",
    )
    assert (limit.value, limit.margin, limit.satisfied) == (0.9, 0.9 - 1.0, True)
    assert assessment.feasible is True


def test_a_violated_minimum_makes_the_candidate_infeasible() -> None:
    assessment = judged(problem({"metric": "a", "min": 7}), a(5))
    (limit,) = assessment.limits
    assert (limit.name, limit.margin, limit.satisfied) == ("analysis/a:min", 2.0, False)
    assert assessment.feasible is False
    assert assessment.objectives == (a(5),)  # an infeasible candidate keeps its objectives


def test_both_bounds_of_a_constraint_are_two_limits_minimum_first() -> None:
    assessment = judged(problem({"metric": "a", "min": 0, "max": 2}), a(3))
    assert [(limit.name, limit.margin, limit.satisfied) for limit in assessment.limits] == [
        ("analysis/a:min", -3.0, True),
        ("analysis/a:max", 1.0, False),
    ]
    assert assessment.feasible is False


@pytest.mark.parametrize("bound", ["min", "max"])
def test_equality_satisfies_a_limit_with_margin_zero(bound: str) -> None:
    (limit,) = judged(problem({"metric": "a", bound: 1.5}), a(1.5)).limits
    assert (limit.margin, limit.satisfied) == (0.0, True)
    assert limit.margin is not None and math.copysign(1.0, limit.margin) == 1.0  # not -0.0


def test_the_margin_has_the_sign_of_the_exact_comparison() -> None:
    big = 2**53  # float(big + 1) == float(big): a difference in floats would be 0, satisfied
    (upper,) = judged(problem({"metric": "a", "max": big + 1}), a(float(big))).limits
    assert (upper.margin, upper.satisfied) == (-1.0, True)
    (lower,) = judged(problem({"metric": "a", "min": big + 1}), a(float(big))).limits
    assert (lower.margin, lower.satisfied) == (1.0, False)
    smallest = 5e-324  # the smallest positive float still counts as above a limit of 0
    (limit,) = judged(problem({"metric": "a", "max": 0.0}), a(smallest)).limits
    assert (limit.margin, limit.satisfied) == (smallest, False)


@pytest.mark.parametrize(
    ("constraint", "value"),
    [({"max": 10**400}, 1.0), ({"min": -1.5e308}, 1.5e308)],
)
def test_a_margin_beyond_a_float_is_none_and_the_verdict_stays_exact(
    constraint: dict[str, Any], value: float
) -> None:
    (limit,) = judged(problem({"metric": "a", **constraint}), a(value)).limits
    assert (limit.margin, limit.satisfied) == (None, True)


def test_the_uncertainty_is_recorded_but_does_not_decide() -> None:
    (limit,) = judged(problem({"metric": "a", "max": 1.0}), a(1.01, uncertainty=0.5)).limits
    assert (limit.value, limit.uncertainty, limit.satisfied) == (1.01, 0.5, False)


# --- what is assessed ------------------------------------------------------------------------


def test_without_constraints_a_valid_evaluation_is_feasible() -> None:
    assessment = judged(problem(), a(2.0))
    assert (assessment.objectives, assessment.limits, assessment.feasible) == ((a(2.0),), (), True)


def test_the_objectives_are_in_the_order_of_the_problem() -> None:
    spec = problem(objectives=(("analysis", "b"), ("analysis", "a")))
    assessment = judged(spec, a(1.0), Measurement("analysis", "b", 2.0))
    assert assessment.objectives is not None
    assert [m.metric for m in assessment.objectives] == ["b", "a"]


@pytest.mark.parametrize("status", ["invalid", "failed"])
def test_an_evaluation_that_is_not_valid_is_not_assessed(status: str) -> None:
    # the value of an invalid result is a diagnostic: never an objective, never judged
    assessment = judged(problem({"metric": "a", "max": 1.0}), a(0.5), status=status)
    assert (assessment.objectives, assessment.feasible) == (None, None)
    (limit,) = assessment.limits  # still listed, with what the problem declares
    assert (limit.name, limit.limit, limit.unit) == ("analysis/a:max", 1.0, "mrad")
    assert (limit.value, limit.uncertainty, limit.margin, limit.satisfied) == (None,) * 4


def test_a_record_that_does_not_represent_the_candidate_is_measured_but_not_judged() -> None:
    assessment = judged(problem({"metric": "a", "max": 1.0}), a(1.5), represents=False)
    assert assessment.objectives == (a(1.5),)
    (limit,) = assessment.limits
    assert (limit.margin, limit.satisfied) == (0.5, False)
    assert assessment.feasible is None  # whatever its limits say


def test_a_valid_owner_does_not_make_an_invalid_candidate_assessable() -> None:
    class Owner(Fixed):
        name = "test.owner"
        metrics = {("analysis", "a"): "mrad"}

    class Check(Fixed):
        name = "test.check"
        metrics = {("analysis", "check.ok"): "1"}

    spec = problem({"metric": "a", "max": 1.0}, evaluators=("test.owner", "test.check"))
    owner = Owner(EvaluationResult.valid([a(0.5)]))
    check = Check(EvaluationResult.invalid("the candidate fails a validity check"))
    record = _evaluate_candidate(spec, {CELL: 5.0}, [owner, check])
    assert record.status == "invalid"
    assessment = assess(spec, record)
    assert (assessment.objectives, assessment.feasible) == (None, None)
    assert assessment.limits[0].value is None


def test_the_assessment_is_plain_json() -> None:
    assessment = judged(problem({"metric": "a", "min": 0, "max": 2}), a(1.0, uncertainty=0.1))
    data = assessment.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    measured = {"value": 1.0, "unit": "mrad", "uncertainty": 0.1}
    assert data == {
        "feasible": True,
        "objectives": [{"stage": "analysis", "metric": "a"} | measured],
        "limits": [
            {"stage": "analysis", "metric": "a", "bound": "min", "limit": 0}
            | measured
            | {"margin": -1.0, "satisfied": True},
            {"stage": "analysis", "metric": "a", "bound": "max", "limit": 2}
            | measured
            | {"margin": -1.0, "satisfied": True},
        ],
    }
    assert judged(problem(), a(1.0), status="failed").to_dict() == {
        "feasible": None,
        "objectives": None,
        "limits": [],
    }


# --- records ---------------------------------------------------------------------------------

RADIUS = "constants.InnerTracker_Barrel_radius_1"
ECHO = ("analysis", "candidate.first_value")


class Echo:
    """The candidate's radius as a metric; ``invalid`` from 590 mm on."""

    name = "test.echo"
    version = "1"
    metrics = {ECHO: "1"}

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        (value,) = request.candidate.values()
        measured = [Measurement(*ECHO, value, "1")]
        if value >= 590:
            return EvaluationResult.invalid("too far out", measured)
        return EvaluationResult.valid(measured)


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EvaluationSession:
    """A session on the mini geometry whose evaluator reads no stage, so nothing simulates."""
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    data = document([{"metric": ECHO[1], "max": 500, "unit": "1"}], [ECHO], ["test.echo"])
    data["geometry"] = MINI_COMPACT
    data["parameters"][0] = {
        "geometry": {"constant": "InnerTracker_Barrel_radius_1"},
        "type": "continuous",
        "bounds": [100.0, 600.0],
        "unit": "mm",
    }
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(data))
    return EvaluationSession.open(path, [Echo()])


def summary_of(session: EvaluationSession, value: float, output: Path) -> dict[str, Any]:
    outcome = session.evaluate({RADIUS: value}, output)
    data: dict[str, Any] = json.loads(outcome.summary_path.read_text())
    assert Assessment.from_summary(session.spec, data) == outcome.assessment
    return data


@pytest.mark.parametrize(
    ("value", "feasible", "assessed"),
    [(450.0, True, True), (550.0, False, True), (595.0, None, False)],
)
def test_a_record_read_back_gives_the_assessment_it_holds(
    session: EvaluationSession, tmp_path: Path, value: float, feasible: bool | None, assessed: bool
) -> None:
    data = summary_of(session, value, tmp_path / "run")
    assessment = Assessment.from_summary(session.spec, data)
    assert assessment.feasible is feasible
    assert (assessment.objectives is not None) is assessed


def test_a_record_of_a_problem_that_differs_only_in_its_name_is_accepted(
    session: EvaluationSession, tmp_path: Path
) -> None:
    data = summary_of(session, 450.0, tmp_path / "run")
    data["problem"]["definition"]["name"] = "renamed"  # a label, as the file name is
    assert Assessment.from_summary(session.spec, data).feasible is True


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    """Set the item at a dotted ``path`` (list indices as numbers) of a summary."""

    def change(data: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        node: Any = data
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value

    return change


def _consistent_lie(data: dict[str, Any]) -> None:
    """A measurement and its metric changed together: only the assessment still disagrees."""
    data["evaluators"][0]["measurements"][0]["value"] = 400.0
    data["metrics"][0]["value"] = 400.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set("schema", "k4opt.run/0"), "schema is 'k4opt.run/0'"),
        (_set("status", "invalid"), "status is 'invalid', but its evaluators make it 'valid'"),
        (_set("evaluators.0.status", "failed"), "status is 'valid', but its evaluators make it"),
        (_set("evaluators.0.status", "great"), "status 'great' is none of"),
        (_set("evaluators.0.name", "test.other"), "its evaluators are"),
        (_set("evaluators", {}), "evaluators must be a list"),
        (_set("evaluators.0.measurements", []), "is valid but did not measure"),
        (_set("evaluators.0.measurements.0.value", "450"), "must be a number"),
        (_set("evaluators.0.measurements.0.value", 400.0), "recorded"),
        (_set("metrics.0.value", 1.0), "metric analysis/candidate.first_value: recorded"),
        (_set("metrics.0.evaluator", "nobody"), "no evaluator 'nobody'"),
        (_set("metrics", []), "its metrics are"),
        (_consistent_lie, "its assessment is not the one its measurements give"),
        (_set("assessment.feasible", False), "its assessment is not the one"),
        (_set("assessment.limits.0.margin", -1.0), "its assessment is not the one"),
        (_set("assessment", None), "its assessment is not the one"),
        (_set("represents_candidate", "yes"), "represents_candidate must be true or false"),
        (_set("problem.definition.parameters.0.bounds", [100.0, 700.0]), "parameters differ"),
        (_set("problem.definition.constraints.0.max", 600), "another problem: its constraints"),
    ],
)
def test_a_record_that_disagrees_with_itself_or_the_problem_is_refused(
    session: EvaluationSession,
    tmp_path: Path,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    data = summary_of(session, 450.0, tmp_path / "run")
    changed = copy.deepcopy(data)
    change(changed)
    with pytest.raises(RecordError, match=re.escape(message)):
        Assessment.from_summary(session.spec, changed)


def test_a_record_is_checked_against_the_problem_it_is_read_for(
    session: EvaluationSession, tmp_path: Path
) -> None:
    data = summary_of(session, 450.0, tmp_path / "run")
    looser = parse_problem({**data["problem"]["definition"], "constraints": []})
    with pytest.raises(RecordError, match="its constraints differ"):
        Assessment.from_summary(looser, data)
    with pytest.raises(RecordError, match="record must be a mapping"):
        Assessment.from_summary(session.spec, [])  # type: ignore[arg-type]
