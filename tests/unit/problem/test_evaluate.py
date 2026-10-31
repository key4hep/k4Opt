"""One candidate -> evaluator -> EvaluationRecord, with toy evaluators and no Key4hep.

The problem here has one evaluator; several are tested in ``test_evaluators.py``.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from k4opt.errors import K4OptError
from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
)
from k4opt.problem import CandidateError, ProblemSpec, parse_problem
from k4opt.problem._evaluate import _evaluate_candidate
from tests.support.evaluators import implementations

CELL = "constants.ECal_cell_size"


class Toy:
    name = "calo.single_particle_shower"
    version = "1"

    def __init__(self, result: EvaluationResult | None = None) -> None:
        self.calls: list[EvaluationRequest] = []
        self.result = result or EvaluationResult(
            "valid",
            (
                Measurement("simulation", "a.rms", 1.5, unit="mm", uncertainty=0.1),
                Measurement("simulation", "a.cells", 50, unit="1"),
                Measurement("analysis", "extra", 3),
            ),
            provenance={"who": "toy"},
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.calls.append(request)
        return self.result


def evaluate_one(
    spec: ProblemSpec, candidate: Any, evaluator: Any, **kwargs: Any
) -> EvaluationRecord:
    """The record of ``evaluator``, the instance of the problem's only evaluator."""
    (record,) = _evaluate_candidate(spec, candidate, implementations(evaluator), **kwargs).records
    return record


def make(doc: dict[str, Any], *parameters: dict[str, Any]) -> ProblemSpec:
    """``doc`` with ``parameters``, measured by a toy that reads no stage, so it has none."""
    doc = copy.deepcopy(doc)
    if parameters:
        doc["parameters"] = list(parameters)
    del doc["stages"], doc["evaluators"][0]["stage"]
    return parse_problem(doc)


def param(name: str, kind: str, **extra: Any) -> dict[str, Any]:
    return {"geometry": {"constant": name}, "type": kind, **extra}


@pytest.fixture
def spec(doc: dict[str, Any]) -> ProblemSpec:
    return make(doc)


def test_success_builds_request_and_record(spec: ProblemSpec, tmp_path: Path) -> None:
    toy = Toy()
    geometry = tmp_path / "compact.xml"
    record = evaluate_one(spec, {CELL: 6}, toy, inputs={"geometry": geometry}, workdir=tmp_path)
    assert len(toy.calls) == 1
    request = toy.calls[0]
    assert record.request is request
    assert dict(request.candidate) == {CELL: 6}
    # it reads no stage, so it has no events or seed of one
    assert request.events is None and request.seed is None
    assert dict(request.settings) == dict(spec.evaluators[0].settings)
    assert request.inputs["geometry"] is geometry
    # each evaluator writes in a directory of its own, created for the call
    assert request.workdir == tmp_path / "evaluators" / Toy.name and request.workdir.is_dir()
    assert record.result is toy.result
    assert (record.evaluator_name, record.evaluator_version) == (Toy.name, Toy.version)
    assert record.elapsed_seconds is not None and record.elapsed_seconds >= 0
    assert record.candidate_key == ((CELL, "continuous", "num", "6e0"),)
    with pytest.raises(FrozenInstanceError):
        record.elapsed_seconds = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("candidate", "match"),
    [
        ({}, CELL),
        ({CELL: 5.0, "constants.Other": 1}, "constants.Other"),
        ({CELL: 5.0, 3: 1}, "keys must be strings"),
        ([(CELL, 5.0)], "mapping"),
        ({CELL: "5"}, CELL),
        ({CELL: True}, CELL),
        ({CELL: float("nan")}, "finite"),
        ({CELL: float("inf")}, "finite"),
        ({CELL: 3.99}, "bounds"),
        ({CELL: 8.01}, "bounds"),
    ],
)
def test_bad_candidates_never_reach_the_evaluator(
    spec: ProblemSpec, candidate: Any, match: str, tmp_path: Path
) -> None:
    toy = Toy()
    with pytest.raises(CandidateError, match=match):
        evaluate_one(spec, candidate, toy, workdir=tmp_path / "w")
    assert toy.calls == []
    assert not (tmp_path / "w").exists()


def test_candidate_error_is_k4opt_error() -> None:
    assert issubclass(CandidateError, K4OptError)


def test_continuous_bounds_inclusive_and_numeric_equivalence(spec: ProblemSpec) -> None:
    keys = {evaluate_one(spec, {CELL: v}, Toy()).candidate_key for v in (4, 4.0, 4.00)}
    assert len(keys) == 1
    assert evaluate_one(spec, {CELL: 8.0}, Toy()).candidate_key[0][3] == "8e0"


def test_negative_zero_and_zero(doc: dict[str, Any]) -> None:
    s = make(doc, param("x", "continuous", bounds=[-1, 1]))
    keys = {evaluate_one(s, {"constants.x": v}, Toy()).candidate_key for v in (0, 0.0, -0.0)}
    assert keys == {(("constants.x", "continuous", "num", "0"),)}


def test_integer_parameter(doc: dict[str, Any]) -> None:
    s = make(doc, param("n", "integer", bounds=[1, 10**400]))
    ref = "constants.n"
    for good in (1, 10, 10**400):
        record = evaluate_one(s, {ref: good}, Toy())
        assert record.request.candidate[ref] == good  # no float conversion, no overflow
    assert evaluate_one(s, {ref: 10**400}, Toy()).candidate_key[0][3] == "1e400"
    for bad in (True, 1.0, 2.5, "1", 0, 10**400 + 1):
        with pytest.raises(CandidateError, match=ref):
            evaluate_one(s, {ref: bad}, Toy())


def test_huge_continuous_int_stays_exact(doc: dict[str, Any]) -> None:
    s = make(doc, param("x", "continuous", bounds=[0, 10**400]))
    a = evaluate_one(s, {"constants.x": 10**399 + 1}, Toy()).candidate_key
    b = evaluate_one(s, {"constants.x": 10**399}, Toy()).candidate_key
    assert a != b


def test_categorical_kinds(doc: dict[str, Any]) -> None:
    s = make(
        doc,
        param("s", "categorical", choices=["1", "a"]),
        param("b", "categorical", choices=[True, False]),
        param("n", "categorical", choices=[1, 2.5]),
    )
    base = {"constants.s": "1", "constants.b": True, "constants.n": 1}
    record = evaluate_one(s, base, Toy())
    assert [part[2:] for part in record.candidate_key] == [
        ("str", "1"),
        ("bool", "true"),
        ("num", "1e0"),
    ]
    same = evaluate_one(s, {**base, "constants.n": 1.0}, Toy())
    assert same.candidate_key == record.candidate_key
    for bad in (
        {"constants.s": 1},
        {"constants.s": "b"},
        {"constants.b": 1},
        {"constants.b": "true"},
        {"constants.n": True},
        {"constants.n": "1"},
        {"constants.n": 3},
        {"constants.n": float("nan")},
    ):
        with pytest.raises(CandidateError):
            evaluate_one(s, {**base, **bad}, Toy())


def test_bool_choice_not_confused_with_numeric_one(doc: dict[str, Any]) -> None:
    sb = make(doc, param("p", "categorical", choices=[True, False]))
    sn = make(doc, param("p", "categorical", choices=[0, 1]))
    assert evaluate_one(sb, {"constants.p": True}, Toy()).candidate_key != (
        evaluate_one(sn, {"constants.p": 1}, Toy()).candidate_key
    )
    with pytest.raises(CandidateError):
        evaluate_one(sn, {"constants.p": True}, Toy())


def test_key_and_request_follow_problem_order(doc: dict[str, Any]) -> None:
    s = make(doc, param("b", "integer", bounds=[0, 5]), param("a", "integer", bounds=[0, 5]))
    record = evaluate_one(s, {"constants.a": 1, "constants.b": 2}, Toy())
    assert list(record.request.candidate) == ["constants.b", "constants.a"]
    assert [part[0] for part in record.candidate_key] == ["constants.b", "constants.a"]


def test_kind_is_part_of_the_key(doc: dict[str, Any]) -> None:
    c = make(doc, param("p", "continuous", bounds=[0, 5]))
    i = make(doc, param("p", "integer", bounds=[0, 5]))
    assert evaluate_one(c, {"constants.p": 1}, Toy()).candidate_key != (
        evaluate_one(i, {"constants.p": 1}, Toy()).candidate_key
    )


@pytest.mark.parametrize(("attr", "value"), [("name", "other"), ("version", "2")])
def test_evaluator_mismatch_fails_before_evaluation(
    spec: ProblemSpec, attr: str, value: str
) -> None:
    toy = Toy()
    setattr(toy, attr, value)
    with pytest.raises(EvaluationContractError, match=value):
        evaluate_one(spec, {CELL: 5.0}, toy)
    assert toy.calls == []


def test_valid_result_missing_metric_is_a_contract_error(spec: ProblemSpec) -> None:
    toy = Toy(EvaluationResult("valid", (Measurement("simulation", "a.rms", 1.0),)))
    with pytest.raises(EvaluationContractError, match=r"simulation/a\.cells"):
        evaluate_one(spec, {CELL: 5.0}, toy)
    # same metric name in another stage does not satisfy the requirement
    toy = Toy(
        EvaluationResult(
            "valid",
            (Measurement("simulation", "a.rms", 1.0), Measurement("analysis", "a.cells", 1)),
        )
    )
    with pytest.raises(EvaluationContractError, match=r"simulation/a\.cells"):
        evaluate_one(spec, {CELL: 5.0}, toy)


def test_constraint_unit_must_match_when_declared(doc: dict[str, Any]) -> None:
    doc["constraints"] = [{"stage": "simulation", "metric": "a.cells", "max": 100, "unit": "1"}]
    s = make(doc)
    for unit in ("mm", None):
        toy = Toy(
            EvaluationResult(
                "valid",
                (
                    Measurement("simulation", "a.rms", 1.0),
                    Measurement("simulation", "a.cells", 5, unit=unit),
                ),
            )
        )
        with pytest.raises(EvaluationContractError, match="unit"):
            evaluate_one(s, {CELL: 5.0}, toy)
    evaluate_one(s, {CELL: 5.0}, Toy())


def test_constraint_value_is_not_judged(spec: ProblemSpec) -> None:
    result = EvaluationResult(
        "valid",
        (Measurement("simulation", "a.rms", 1.0), Measurement("simulation", "a.cells", 1e9)),
    )
    assert evaluate_one(spec, {CELL: 5.0}, Toy(result)).result is result


@pytest.mark.parametrize("status", ["invalid", "failed"])
def test_non_valid_results_are_preserved(spec: ProblemSpec, status: Any) -> None:
    result = EvaluationResult(status, message="boom", provenance={"k": 1})
    record = evaluate_one(spec, {CELL: 5.0}, Toy(result))
    assert record.result is result
    assert record.result.status == status and record.result.measurements == ()


def test_evaluator_exceptions_propagate(spec: ProblemSpec) -> None:
    class Broken(Toy):
        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        evaluate_one(spec, {CELL: 5.0}, Broken())


def test_non_result_return_is_rejected(spec: ProblemSpec) -> None:
    class Wrong(Toy):
        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            return None  # type: ignore[return-value]

    with pytest.raises(EvaluationContractError, match="EvaluationResult"):
        evaluate_one(spec, {CELL: 5.0}, Wrong())


def test_record_validates_fields(spec: ProblemSpec) -> None:
    record = evaluate_one(spec, {CELL: 5.0}, Toy())

    def build(**changes: Any) -> EvaluationRecord:
        fields: dict[str, Any] = {
            "request": record.request,
            "result": record.result,
            "evaluator_name": "n",
            "evaluator_version": "1",
            "candidate_key": record.candidate_key,
            "elapsed_seconds": 0.0,
        }
        return EvaluationRecord(**{**fields, **changes})

    build()
    assert build(elapsed_seconds=None).elapsed_seconds is None  # the evaluator was not called
    for bad in (
        {"elapsed_seconds": -1.0},
        {"elapsed_seconds": float("nan")},
        {"elapsed_seconds": True},
        {"candidate_key": [("a", "b", "c", "d")]},
        {"candidate_key": (("a",),)},
        {"evaluator_name": " "},
        {"result": None},
        {"request": None},
    ):
        with pytest.raises(EvaluationContractError):
            build(**bad)


def test_evaluation_package_stays_independent_of_problem() -> None:
    code = (
        "import sys, k4opt.evaluation;"
        "banned = ('k4opt.problem', 'k4opt.geometry', 'lxml');"
        "sys.exit(1 if [m for m in sys.modules if m.startswith(banned)] else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, cwd=Path(__file__).parents[3]
    )
    assert result.returncode == 0


def test_float_and_int_with_different_values_have_different_keys(doc: dict[str, Any]) -> None:
    s = make(doc, param("x", "continuous", bounds=[0, 10**30]))
    ref = "constants.x"
    exact = evaluate_one(s, {ref: 10**23}, Toy()).candidate_key
    rounded = evaluate_one(s, {ref: 1e23}, Toy()).candidate_key
    assert 10**23 != 1e23
    assert exact != rounded
    assert evaluate_one(s, {ref: 2**60}, Toy()).candidate_key == (
        evaluate_one(s, {ref: float(2**60)}, Toy()).candidate_key
    )
    assert evaluate_one(s, {ref: 0.1}, Toy()).candidate_key == (
        evaluate_one(s, {ref: 0.1}, Toy()).candidate_key
    )


def test_record_keeps_the_identity_that_was_validated(spec: ProblemSpec) -> None:
    class Shifty(Toy):
        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            self.name, self.version = "changed", "9"
            return super().evaluate(request)

    record = evaluate_one(spec, {CELL: 5.0}, Shifty())
    assert (record.evaluator_name, record.evaluator_version) == (Toy.name, Toy.version)


# --- units are those the evaluator declares for its metrics (review) ---


class Declaring(Toy):
    """A toy evaluator that advertises the unit of each metric it reports."""

    metrics = {("simulation", "a.rms"): "mm", ("simulation", "a.cells"): "1"}


def returning(*measurements: Measurement, status: str = "valid") -> Declaring:
    if status == "valid":
        return Declaring(EvaluationResult.valid(measurements))
    return Declaring(EvaluationResult.invalid("checks failed", measurements))


RMS_MM = Measurement("simulation", "a.rms", 1.5, unit="mm")
CELLS = Measurement("simulation", "a.cells", 50, unit="1")


def test_a_unit_that_matches_the_declared_one_is_accepted(spec: ProblemSpec) -> None:
    record = evaluate_one(spec, {CELL: 6}, returning(RMS_MM, CELLS))
    assert record.result.status == "valid"


@pytest.mark.parametrize("unit", ["cm", None])
def test_an_objective_in_another_unit_than_declared_is_rejected(
    spec: ProblemSpec, unit: str | None
) -> None:
    # a.rms is only an objective here: no constraint carries a unit to compare it with
    wrong = Measurement("simulation", "a.rms", 0.15, unit=unit)
    with pytest.raises(EvaluationContractError, match=r"simulation/a\.rms.*declares 'mm'"):
        evaluate_one(spec, {CELL: 6}, returning(wrong, CELLS))


def test_diagnostic_measurements_of_an_invalid_result_are_checked_too(spec: ProblemSpec) -> None:
    wrong = Measurement("simulation", "a.cells", 50, unit="mm")
    with pytest.raises(EvaluationContractError, match=r"simulation/a\.cells.*declares '1'"):
        evaluate_one(spec, {CELL: 6}, returning(wrong, status="invalid"))


def test_a_metric_the_evaluator_does_not_declare_is_not_unit_checked(spec: ProblemSpec) -> None:
    extra = Measurement("analysis", "extra", 3, unit="whatever")
    record = evaluate_one(spec, {CELL: 6}, returning(RMS_MM, CELLS, extra))
    assert len(record.result.measurements) == 3


def test_an_evaluator_without_declared_metrics_is_not_unit_checked(spec: ProblemSpec) -> None:
    result = EvaluationResult.valid([Measurement("simulation", "a.rms", 1.0, unit="parsec"), CELLS])
    assert evaluate_one(spec, {CELL: 6}, Toy(result)).result.status == "valid"
