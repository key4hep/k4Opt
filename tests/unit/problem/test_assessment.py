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
from k4opt.problem import (
    Assessment,
    EvaluationSession,
    ObjectiveValue,
    ProblemSpec,
    RecordError,
    parse_problem,
)
from k4opt.problem._assess import assess
from k4opt.problem._evaluate import _evaluate_candidate
from tests.support.evaluators import implementations
from tests.support.geometry import MINI_COMPACT, MINI_GEO

CELL = "constants.ECal_cell_size"


def document(
    constraints: list[dict[str, Any]],
    objectives: list[tuple[str, ...]],
    evaluators: list[str],
) -> dict[str, Any]:
    """A problem document; an objective is ``(stage, metric)``, or ``(evaluator, stage, metric)``
    when the problem has several evaluators."""
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
        "objectives": [
            dict(zip(("evaluator", "stage", "metric")[-len(o) :], o, strict=True))
            | {"direction": "minimize"}
            for o in objectives
        ],
    }
    if constraints:
        data["constraints"] = [{"stage": "analysis", "unit": "mrad"} | c for c in constraints]
    return data


def problem(
    *constraints: dict[str, Any],
    objectives: tuple[tuple[str, ...], ...] = (("analysis", "a"),),
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


def objective(m: Measurement, evaluator: str = "test.fixed") -> ObjectiveValue:
    """The value of an objective to minimize that ``m`` measures."""
    return ObjectiveValue(evaluator, m.stage, m.metric, "minimize", m.value, m.unit, m.uncertainty)


def judged(
    spec: ProblemSpec, *measurements: Measurement, status: str = "valid", represents: bool = True
) -> Assessment:
    """The assessment of a candidate whose only evaluator returned these measurements."""
    message = None if status == "valid" else f"the evaluator says {status}"
    result = EvaluationResult(status, measurements, message=message)  # type: ignore[arg-type]
    record = _evaluate_candidate(spec, {CELL: 5.0}, implementations(Fixed(result)))
    if not represents:  # as for a stage supplied with --input
        record = dataclasses.replace(record, represents_candidate=False)
    return assess(spec, record)


# --- limits ----------------------------------------------------------------------------------


def test_a_satisfied_maximum_has_a_margin_of_zero_or_below() -> None:
    assessment = judged(problem({"metric": "a", "max": 1.0}), a(0.9))
    (limit,) = assessment.limits
    assert (limit.name, limit.bound, limit.limit, limit.unit) == (
        "test.fixed/analysis/a:max",
        "max",
        1.0,
        "mrad",
    )
    assert (limit.value, limit.margin, limit.satisfied) == (0.9, 0.9 - 1.0, True)
    assert assessment.feasible is True


def test_a_violated_minimum_makes_the_candidate_infeasible() -> None:
    assessment = judged(problem({"metric": "a", "min": 7}), a(5))
    (limit,) = assessment.limits
    assert (limit.name, limit.margin, limit.satisfied) == ("test.fixed/analysis/a:min", 2.0, False)
    assert assessment.feasible is False
    assert assessment.objectives == (objective(a(5)),)  # an infeasible one keeps its objectives


def test_both_bounds_of_a_constraint_are_two_limits_minimum_first() -> None:
    assessment = judged(problem({"metric": "a", "min": 0, "max": 2}), a(3))
    assert [(limit.name, limit.margin, limit.satisfied) for limit in assessment.limits] == [
        ("test.fixed/analysis/a:min", -3.0, True),
        ("test.fixed/analysis/a:max", 1.0, False),
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
    assert (assessment.objectives, assessment.limits, assessment.feasible) == (
        (objective(a(2.0)),),
        (),
        True,
    )


def test_the_objectives_are_in_the_order_of_the_problem() -> None:
    spec = problem(objectives=(("analysis", "b"), ("analysis", "a")))
    assessment = judged(spec, a(1.0), Measurement("analysis", "b", 2.0))
    assert assessment.objectives is not None
    assert [o.label for o in assessment.objectives] == [
        "test.fixed/analysis/b",
        "test.fixed/analysis/a",
    ]


@pytest.mark.parametrize("status", ["invalid", "failed"])
def test_an_evaluation_that_is_not_valid_is_not_assessed(status: str) -> None:
    # the value of an invalid result is a diagnostic: never an objective, never judged
    assessment = judged(problem({"metric": "a", "max": 1.0}), a(0.5), status=status)
    assert (assessment.objectives, assessment.feasible) == (None, None)
    (limit,) = assessment.limits  # still listed, with what the problem declares
    assert (limit.name, limit.limit, limit.unit) == ("test.fixed/analysis/a:max", 1.0, "mrad")
    assert (limit.value, limit.uncertainty, limit.margin, limit.satisfied) == (None,) * 4


def test_a_record_that_does_not_represent_the_candidate_is_measured_but_not_judged() -> None:
    assessment = judged(problem({"metric": "a", "max": 1.0}), a(1.5), represents=False)
    assert assessment.objectives == (objective(a(1.5)),)
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

    spec = problem(
        {"evaluator": "test.owner", "metric": "a", "max": 1.0},
        objectives=(("test.owner", "analysis", "a"),),
        evaluators=("test.owner", "test.check"),
    )
    owner = Owner(EvaluationResult.valid([a(0.5)]))
    check = Check(EvaluationResult.invalid("the candidate fails a validity check"))
    record = _evaluate_candidate(spec, {CELL: 5.0}, implementations(owner, check))
    assert record.status == "invalid"
    assessment = assess(spec, record)
    assert (assessment.objectives, assessment.feasible) == (None, None)
    assert assessment.limits[0].value is None


def test_the_assessment_is_plain_json() -> None:
    assessment = judged(problem({"metric": "a", "min": 0, "max": 2}), a(1.0, uncertainty=0.1))
    data = assessment.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    measured = {"value": 1.0, "unit": "mrad", "uncertainty": 0.1}
    metric = {"evaluator": "test.fixed", "stage": "analysis", "metric": "a"}
    assert data == {
        "feasible": True,
        "objectives": [metric | {"direction": "minimize"} | measured],
        "limits": [
            metric | {"bound": "min", "limit": 0} | measured | {"margin": -1.0, "satisfied": True},
            metric | {"bound": "max", "limit": 2} | measured | {"margin": -1.0, "satisfied": True},
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
ECHOED = "evaluations/test.echo"  # where the record holds the work of the test.echo binding
COST = ("analysis", "readout.cost")


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


class Cost:
    """A constant readout cost, a second evaluator that reads no stage."""

    name = "test.cost"
    version = "1"
    metrics = {COST: "1"}

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return EvaluationResult.valid([Measurement(*COST, 7.0, "1")])


def open_session(tmp_path: Path, *evaluators: Any) -> EvaluationSession:
    """A session on the mini geometry whose evaluators read no stage, so nothing simulates; each
    evaluator's metric is an objective, and the first one's is constrained."""
    objectives = [(e.name, *next(iter(e.metrics))) for e in evaluators]
    names = [evaluator.name for evaluator in evaluators]
    limit = {"evaluator": names[0], "metric": ECHO[1], "max": 500, "unit": "1"}
    return open_problem(tmp_path, document([limit], objectives, names), *evaluators)


def open_problem(tmp_path: Path, data: dict[str, Any], *evaluators: Any) -> EvaluationSession:
    """A session of the problem ``data`` on the mini geometry."""
    data["geometry"] = MINI_COMPACT
    data["parameters"][0] = {
        "geometry": {"constant": "InnerTracker_Barrel_radius_1"},
        "type": "continuous",
        "bounds": [100.0, 600.0],
        "unit": "mm",
    }
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(data))
    return EvaluationSession.open(path, implementations(*evaluators))


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def session(tmp_path: Path) -> EvaluationSession:
    return open_session(tmp_path, Echo())


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
    """Set the item at a ``path`` of a summary, its keys separated by ``/`` (list indices as
    numbers): ``evaluations/test.echo/status``."""

    def change(data: dict[str, Any]) -> None:
        *parents, last = path.split("/")
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
    data["evaluations"]["test.echo"]["measurements"][0]["value"] = 400.0
    data["metrics"][0]["value"] = 400.0


def _other_unit(data: dict[str, Any]) -> None:
    """The unit of a measurement changed everywhere the record judges it, but not where its
    evaluator declares the metric."""
    data["evaluations"]["test.echo"]["measurements"][0]["unit"] = "mm"
    data["metrics"][0]["unit"] = "mm"
    data["assessment"]["objectives"][0]["unit"] = "mm"
    data["assessment"]["limits"][0]["unit"] = "mm"


def _other_declared_unit(data: dict[str, Any]) -> None:
    """The unit changed everywhere the record holds it, so the record agrees with itself, but
    not with the unit the constraint declares."""
    _other_unit(data)
    data["implementations"]["test.echo"]["metrics"][0]["unit"] = "mm"


def _undeclared_unit(data: dict[str, Any]) -> None:
    """As :func:`_other_declared_unit`, for an evaluator that declares no metrics."""
    _other_unit(data)
    data["implementations"]["test.echo"]["metrics"] = None


def _failed_with_reason(data: dict[str, Any]) -> None:
    data["evaluations"]["test.echo"]["status"] = "failed"
    data["evaluations"]["test.echo"]["message"] = "ddsim crashed"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set("schema", "k4opt.run/0"), "schema is 'k4opt.run/0'"),
        (_set("candidate", []), "candidate must be a mapping"),
        (_set("candidate", {RADIUS: 700.0}), "its candidate is not one of the problem's"),
        (_set("candidate", {RADIUS: 460.0}), "but its candidate is [['constants.Inner"),
        (_set("candidate_key/0/3", "1e0"), "candidate_key is [['constants.InnerTracker"),
        (_set("status", "invalid"), "status is 'invalid', but its tasks make it 'valid'"),
        (
            _set(f"{ECHOED}/status", "failed"),
            "a task that is failed needs a message saying why",
        ),
        (_failed_with_reason, "status is 'valid', but its tasks make it 'failed'"),
        (_set(f"{ECHOED}/message", 3), "message must be a string, got 3"),
        (_set("message", "test.echo valid: fine"), "message is 'test.echo valid: fine', but"),
        (_set(f"{ECHOED}/status", "great"), "status 'great' is none of"),
        (
            _set(f"{ECHOED}/implementation", "test.other"),
            "implementation 'test.other', but the problem binds 'test.echo'",
        ),
        (_set("evaluations", []), "evaluations must be a mapping"),
        (_set("evaluations", {"echo": {}}), "its evaluations are ['echo'], but the problem's"),
        (_set("implementations", {}), "its implementations are [], but the problem binds"),
        (_set("implementations/test.echo/version", "2"), "version '2', but the problem binds"),
        # what the record says of a binding's code and inputs is the problem's, too
        (
            _set("implementations/test.echo/stage", "simulation"),
            "implementation 'test.echo': stage 'simulation', but the problem has it read None",
        ),
        (
            _set(f"{ECHOED}/settings/declared", {"invalid_below": 999}),
            "evaluator 'test.echo': settings.declared {'invalid_below': 999}, but the problem",
        ),
        (_set(f"{ECHOED}/settings", None), "evaluator 'test.echo': settings must be a mapping"),
        (_set(f"{ECHOED}/measurements", []), "is valid but did not measure"),
        (_set(f"{ECHOED}/measurements/0/value", "450"), "must be a number"),
        (_set(f"{ECHOED}/measurements/0/value", 400.0), "recorded"),
        (_set("metrics/0/value", 1.0), f"metric test.echo/{ECHO[0]}/{ECHO[1]}: recorded"),
        (_set("metrics/0/evaluator", "nobody"), "its metrics are [['nobody'"),
        (_set("metrics/0/label", "analysis/candidate.first_value"), "its metrics are"),
        (_set("metrics", []), "its metrics are"),
        (_consistent_lie, "its assessment is not the one its measurements give"),
        (_other_unit, "analysis/candidate.first_value is in 'mm', but the evaluator declares '1'"),
        (_other_declared_unit, "constraint declares '1', but its evaluator declares 'mm'"),
        (_undeclared_unit, "constraint declares '1', but it is measured in 'mm'"),
        (_set("implementations/test.echo/metrics", []), "its evaluator does not declare it"),
        (
            _set("implementations/test.echo/metrics/0/unit", ""),
            "metrics[0].unit must be a nonblank string",
        ),
        (_set("assessment/feasible", False), "its assessment is not the one"),
        (_set("assessment/limits/0/margin", -1.0), "its assessment is not the one"),
        (_set("assessment", None), "its assessment is not the one"),
        (_set("represents_candidate", "yes"), "represents_candidate is 'yes', but its supplied"),
        (_set("supplied", ["simulation"]), "it supplied simulation, which no evaluator reads"),
        (_set("stages", {"simulation": {}}), "its stages are ['simulation'], but the problem"),
        (_set("represents_candidate", False), "represents_candidate is False, but its supplied"),
        (_set("problem/definition/parameters/0/bounds", [100.0, 700.0]), "parameters differ"),
        (_set("problem/definition/constraints/0/max", 600), "another problem: its constraints"),
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


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set(f"{ECHOED}/message", None), "a task that is invalid needs a message saying why"),
        (_set(f"{ECHOED}/message", "  "), "a task that is invalid needs a message saying why"),
        (_set("message", "test.echo invalid: fine after all"), "but its tasks make it"),
        (_set("message", None), "message is None, but its tasks make it"),
    ],
)
def test_an_unusable_record_must_say_why(
    session: EvaluationSession,
    tmp_path: Path,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    data = summary_of(session, 595.0, tmp_path / "run")  # invalid: too far out
    assert data["message"] == "test.echo invalid: too far out"
    change(data)
    with pytest.raises(RecordError, match=re.escape(message)):
        Assessment.from_summary(session.spec, data)


def _moved(data: dict[str, Any]) -> None:
    """The first evaluator's value moved to the second, which declares it too."""
    moved = data["evaluations"]["test.echo"]["measurements"].pop()
    data["evaluations"]["test.cost"]["measurements"].append(moved)
    data["implementations"]["test.cost"]["metrics"].append(
        {"stage": ECHO[0], "metric": ECHO[1], "unit": "1"}
    )


def _owner_changed(data: dict[str, Any]) -> None:
    """The first metric said to be the second evaluator's, everywhere the record names it."""
    entry = data["metrics"][0]
    entry["evaluator"] = "test.cost"
    entry["label"] = f"test.cost/{ECHO[0]}/{ECHO[1]}"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_moved, "evaluator 'test.echo' is valid but did not measure analysis/candidate.first"),
        (_owner_changed, "its metrics are [['test.cost', 'analysis', 'candidate.first_value'"),
        (_set("implementations/test.cost/metrics", []), "its evaluator does not declare it"),
    ],
)
def test_each_value_of_a_record_is_that_of_the_evaluator_its_metric_names(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    pair = open_session(tmp_path, Echo(), Cost())
    data = summary_of(pair, 450.0, tmp_path / "run")
    assert [m["label"] for m in data["metrics"]] == [
        "test.echo/analysis/candidate.first_value",
        "test.cost/analysis/readout.cost",
    ]
    assert {name: e["metrics"] for name, e in data["implementations"].items()} == {
        "test.echo": [{"stage": ECHO[0], "metric": ECHO[1], "unit": "1"}],
        "test.cost": [{"stage": COST[0], "metric": COST[1], "unit": "1"}],
    }
    change(data)
    with pytest.raises(RecordError, match=re.escape(message)):
        Assessment.from_summary(pair.spec, data)


def _twice(tmp_path: Path) -> EvaluationSession:
    """A session that binds test.echo twice, the second time as ``again``, with an objective on
    each binding's value and a limit on the second's."""
    data = document(
        [{"evaluator": "again", "metric": ECHO[1], "max": 500, "unit": "1"}],
        [("test.echo", *ECHO), ("again", *ECHO)],
        ["test.echo"],
    )
    data["evaluators"].append({"name": "test.echo", "as": "again", "version": "1"})
    return open_problem(tmp_path, data, Echo())


def _other_binding_changed(data: dict[str, Any]) -> None:
    """The second binding's value changed, the first's, of the same metric, left as it is."""
    data["evaluations"]["again"]["measurements"][0]["value"] = 1.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_other_binding_changed, "metric again/analysis/candidate.first_value: recorded"),
        (_set("evaluations/again/implementation", "test.cost"), "but the problem binds"),
        (_set("implementations/again", {}), "its implementations are ['test.echo', 'again']"),
    ],
)
def test_two_bindings_of_one_implementation_are_two_owners_in_a_record(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    session = _twice(tmp_path)
    data = summary_of(session, 450.0, tmp_path / "run")
    assert list(data["implementations"]) == ["test.echo"]  # recorded once
    assert list(data["evaluations"]) == ["test.echo", "again"]
    assert [m["label"] for m in data["metrics"]] == [
        "test.echo/analysis/candidate.first_value",
        "again/analysis/candidate.first_value",
    ]
    assert [o["evaluator"] for o in data["assessment"]["objectives"]] == ["test.echo", "again"]
    assert data["assessment"]["limits"][0]["evaluator"] == "again"
    change(data)
    with pytest.raises(RecordError, match=re.escape(message)):
        Assessment.from_summary(session.spec, data)
