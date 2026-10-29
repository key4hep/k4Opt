"""The Evaluator base class: settings schema, metric shorthand and returning plain values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    Status,
    metric_units,
    result_from,
)
from tests.support import simulation
from tests.support.simulation import COLLECTION, HIT_TYPE, events_with


class Counter(Evaluator):
    name = "demo.counter"
    version = "1"
    requires = "simulation"
    settings = {"collection": str, "n": int, "scale": float, "flag": bool}
    metrics = {"cells": "1", ("analysis", "rms"): "mm"}

    def evaluate(self, request: EvaluationRequest) -> Any:
        return {"cells": [1, 2, 3], ("analysis", "rms"): 0.5}


GOOD: dict[str, Any] = {"collection": "C", "n": 3, "scale": 2, "flag": False}


def check(**settings: Any) -> None:
    Counter().check(EvaluationRequest(settings=settings))


def test_matching_settings_pass_and_an_int_is_a_float() -> None:
    check(**GOOD)
    check(**{**GOOD, "scale": 2.5})


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({}, "missing"),
        ({**GOOD, "extra": 1}, r"unknown \['extra'\]"),
        ({k: v for k, v in GOOD.items() if k != "n"}, r"missing \['n'\]"),
        ({**GOOD, "collection": 3}, "settings.collection: expected str"),
        ({**GOOD, "n": True}, "settings.n: expected int"),  # a bool is not an int
        ({**GOOD, "n": 1.5}, "settings.n: expected int"),
        ({**GOOD, "scale": True}, "settings.scale: expected float"),
        ({**GOOD, "flag": 0}, "settings.flag: expected bool"),
    ],
)
def test_mismatching_settings_are_rejected(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(EvaluationContractError, match=message):
        check(**settings)


def test_without_a_settings_declaration_nothing_is_checked() -> None:
    class Free(Evaluator):
        name, version = "demo.free", "1"

    Free().check(EvaluationRequest(settings={"anything": [1, 2]}))


def test_metrics_default_to_the_required_stage_or_name_their_own() -> None:
    assert metric_units(Counter()) == {("simulation", "cells"): "1", ("analysis", "rms"): "mm"}


def test_no_declared_metrics_is_not_an_empty_declaration() -> None:
    class Free(Evaluator):
        name, version = "demo.free", "1"

    assert metric_units(Free()) is None


def test_a_bare_metric_needs_a_stage() -> None:
    class Stageless(Evaluator):
        name, version = "demo.stageless", "1"
        metrics = {"cells": "1"}

    with pytest.raises(EvaluationContractError, match="requires"):
        metric_units(Stageless())


def test_returned_values_become_measurements_with_the_declared_units() -> None:
    evaluator = Counter()
    result = result_from(evaluator, evaluator.evaluate(EvaluationRequest()))
    assert result.status is Status.VALID
    cells, rms = result.measurements
    assert (cells.stage, cells.metric, cells.unit, cells.value) == ("simulation", "cells", "1", 2)
    assert cells.uncertainty == pytest.approx(1 / 3**0.5)  # a list: mean and standard error
    assert (rms.stage, rms.metric, rms.unit, rms.value) == ("analysis", "rms", "mm", 0.5)
    assert rms.uncertainty is None  # a number: just the value


def test_an_evaluation_result_is_passed_through() -> None:
    result = EvaluationResult.invalid("no")
    assert result_from(Counter(), result) is result


@pytest.mark.parametrize("returned", [None, 3, [1, 2], "x"])
def test_anything_else_is_a_contract_error(returned: Any) -> None:
    with pytest.raises(EvaluationContractError, match="EvaluationResult or a mapping"):
        result_from(Counter(), returned)


@pytest.mark.parametrize("value", ["a", None, float("nan"), [], [1, float("inf")]])
def test_unusable_values_are_contract_errors(value: Any) -> None:
    with pytest.raises(EvaluationContractError):
        result_from(Counter(), {"cells": value})


def test_invalid_and_fail_end_the_evaluation() -> None:
    with pytest.raises(EvaluationInvalid, match="why"):
        Counter().invalid("why")
    with pytest.raises(EvaluationFailed, match="how"):
        Counter().fail("how")


def test_evaluate_must_be_implemented() -> None:
    class Empty(Evaluator):
        name, version = "demo.empty", "1"

    with pytest.raises(NotImplementedError):
        Empty().evaluate(EvaluationRequest())


def test_request_read_reads_the_file_of_a_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    simulation.install_events(monkeypatch, events_with([1, 2], [3]))
    request = EvaluationRequest(inputs={"simulation": Path("events.root")})
    assert [len(hits) for hits in request.read("simulation", COLLECTION, HIT_TYPE)] == [2, 1]


def test_request_read_of_a_stage_without_a_file_is_a_contract_error() -> None:
    with pytest.raises(EvaluationContractError, match=r"inputs\['simulation'\]"):
        EvaluationRequest().read("simulation", COLLECTION, HIT_TYPE)
