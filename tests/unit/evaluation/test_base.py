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
    with_defaults,
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


class Nested(Evaluator):
    name, version = "demo.nested", "1"
    settings = {"selection": dict, "thresholds": list, "rings": [1, 2], "cuts": {"pt": 1.0}}


def test_dict_and_list_settings_accept_the_frozen_form_the_request_holds() -> None:
    request = EvaluationRequest(settings={"selection": {"x": 1}, "thresholds": [1, 2]})
    Nested().check(request)  # the request froze them into a read-only mapping and a tuple
    Nested().check(EvaluationRequest(settings=with_defaults(Nested(), request.settings)))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"selection": [1]}, r"settings\.selection: expected dict"),
        ({"thresholds": {"a": 1}}, r"settings\.thresholds: expected list"),
        ({"rings": "1, 2"}, r"settings\.rings: expected list"),
        ({"cuts": 1.0}, r"settings\.cuts: expected dict"),
    ],
)
def test_dict_and_list_settings_reject_other_values(changes: dict[str, Any], message: str) -> None:
    settings = {"selection": {}, "thresholds": [], **changes}
    with pytest.raises(EvaluationContractError, match=message):
        Nested().check(EvaluationRequest(settings=settings))


@pytest.mark.parametrize("declared", [Path, tuple, None])
def test_a_setting_of_an_unsupported_type_cannot_be_declared(declared: Any) -> None:
    class Odd(Evaluator):
        name, version = "demo.odd", "1"
        settings = {"x": declared}

    with pytest.raises(EvaluationContractError, match=r"settings\.x: cannot declare"):
        Odd().check(EvaluationRequest())


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


def test_a_metric_declared_twice_is_a_contract_error_not_overwritten() -> None:
    class Twice(Evaluator):
        name, version, requires = "demo.twice", "1", "simulation"
        metrics = {"energy": "GeV", ("simulation", "energy"): "MeV"}

    with pytest.raises(EvaluationContractError, match="simulation/energy is declared twice"):
        metric_units(Twice())


@pytest.mark.parametrize("unit", [None, "", "  ", 123])
def test_a_declared_unit_must_be_a_nonblank_string(unit: Any) -> None:
    class Unitless(Evaluator):
        name, version, requires = "demo.unitless", "1", "simulation"
        metrics = {"energy": unit}

    with pytest.raises(EvaluationContractError, match="simulation/energy: unit must be"):
        metric_units(Unitless())


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


def test_an_evaluator_without_a_stage_names_the_stage_of_its_metrics() -> None:
    # the package documentation's example: nothing runs first, so there is no `requires`
    class Constant(Evaluator):
        name, version = "demo.constant", "1"

        def evaluate(self, request: EvaluationRequest) -> Any:
            return {("analysis", "answer"): 42}

    evaluator = Constant()
    (m,) = result_from(evaluator, evaluator.evaluate(EvaluationRequest())).measurements
    assert (m.stage, m.metric, m.value, m.unit) == ("analysis", "answer", 42, None)


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


class Tuned(Evaluator):
    name = "demo.tuned"
    version = "1"
    requires = "simulation"
    settings = {"collection": str, "threshold": 0.5, "rings": 3, "strict": False}

    def evaluate(self, request: EvaluationRequest) -> Any:
        return {"n": 1}


def tuned(**settings: Any) -> EvaluationRequest:
    return EvaluationRequest(settings=settings)


def test_a_default_makes_a_setting_optional() -> None:
    Tuned().check(tuned(collection="C"))
    Tuned().check(tuned(collection="C", threshold=2, rings=4, strict=True))  # an int is a float


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_required_text_cannot_be_blank_but_an_optional_one_can(blank: str) -> None:
    class Named(Evaluator):
        name, version = "demo.named", "1"
        settings = {"collection": str, "label": "x"}

    with pytest.raises(EvaluationContractError, match=r"settings\.collection: must not be blank"):
        Named().check(EvaluationRequest(settings={"collection": blank}))
    Named().check(EvaluationRequest(settings={"collection": "C", "label": blank}))


def test_a_default_still_fixes_the_type_and_a_required_key_stays_required() -> None:
    with pytest.raises(EvaluationContractError, match=r"settings\.rings: expected int"):
        Tuned().check(tuned(collection="C", rings=2.5))
    with pytest.raises(EvaluationContractError, match=r"settings\.strict: expected bool"):
        Tuned().check(tuned(collection="C", strict=1))
    with pytest.raises(EvaluationContractError, match=r"missing \['collection'\]"):
        Tuned().check(tuned(threshold=1.0))
    with pytest.raises(EvaluationContractError, match=r"unknown \['extra'\]"):
        Tuned().check(tuned(collection="C", extra=1))


def test_with_defaults_adds_only_the_missing_optional_settings() -> None:
    merged = with_defaults(Tuned(), {"collection": "C", "rings": 5})
    assert merged == {"collection": "C", "rings": 5, "threshold": 0.5, "strict": False}


def test_with_defaults_leaves_other_evaluators_alone() -> None:
    class Plain:
        name, version = "demo.plain", "1"

    assert with_defaults(Plain(), {"a": 1}) == {"a": 1}
    assert with_defaults(Counter(), {"collection": "C"}) == {"collection": "C"}  # no defaults


def test_self_read_uses_the_required_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    simulation.install_events(monkeypatch, events_with([1, 2], [3]))
    request = EvaluationRequest(inputs={"simulation": Path("events.root")})
    assert [len(h) for h in Counter().read(request, COLLECTION, "SimCalorimeterHit")] == [2, 1]


def test_self_read_needs_requires() -> None:
    class Stageless(Evaluator):
        name, version = "demo.stageless", "1"

    with pytest.raises(EvaluationContractError, match="requires"):
        Stageless().read(EvaluationRequest(), COLLECTION, "SimCalorimeterHit")
