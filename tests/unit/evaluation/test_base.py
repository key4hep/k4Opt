"""The Evaluator base class: settings schema, collections, metric shorthand and returning plain
values."""

from __future__ import annotations

import decimal
from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import (
    Collection,
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    Quantity,
    Status,
    collection_names,
    declared_collections,
    metric_units,
    result_from,
    units,
    with_defaults,
)
from tests.support import simulation
from tests.support.simulation import (
    COLLECTION,
    HIT_TYPE,
    Event,
    Hits,
    Particle,
    Particles,
    events_with,
)


class Counter(Evaluator):
    name = "demo.counter"
    version = "1"
    stage = "simulation"
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
        name, version, stage = "demo.twice", "1", "simulation"
        metrics = {"energy": "GeV", ("simulation", "energy"): "MeV"}

    with pytest.raises(EvaluationContractError, match="simulation/energy is declared twice"):
        metric_units(Twice())


@pytest.mark.parametrize("unit", [None, "", "  ", 123])
def test_a_declared_unit_must_be_a_nonblank_string(unit: Any) -> None:
    class Unitless(Evaluator):
        name, version, stage = "demo.unitless", "1", "simulation"
        metrics = {"energy": unit}

    with pytest.raises(EvaluationContractError, match="simulation/energy: unit must be"):
        metric_units(Unitless())


def test_a_bare_metric_needs_a_stage() -> None:
    class Stageless(Evaluator):
        name, version = "demo.stageless", "1"
        metrics = {"cells": "1"}

    with pytest.raises(EvaluationContractError, match=r"set stage = <stage>"):
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
    # the package documentation's example: nothing runs first, so there is no `stage`
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


@pytest.mark.parametrize("value", ["a", b"12", {"a": 1}, None, float("nan"), [], [1, float("inf")]])
def test_unusable_values_are_contract_errors(value: Any) -> None:
    with pytest.raises(EvaluationContractError):
        result_from(Counter(), {"cells": value})


@pytest.mark.parametrize("cells", [range(1, 4), (n for n in (1, 2, 3)), {1, 2, 3}])
def test_per_event_values_may_be_any_collection_of_numbers(cells: Any) -> None:
    measurement = result_from(Counter(), {"cells": cells}).measurements[0]
    assert (measurement.value, measurement.unit) == (2, "1")
    assert measurement.uncertainty == pytest.approx(1 / 3**0.5)


def test_per_event_values_may_be_a_numpy_array() -> None:
    np = pytest.importorskip("numpy")
    measurement = result_from(Counter(), {"cells": np.array([1, 2, 3])}).measurements[0]
    assert measurement.value == 2 and type(measurement.value) is float


def test_a_value_that_cannot_be_iterated_after_all_is_a_contract_error() -> None:
    class Unrollable:  # iterable in type only, like a zero-dimensional numpy array
        def __iter__(self) -> Any:
            raise TypeError("iteration over a 0-d array")

    with pytest.raises(EvaluationContractError, match="neither a number nor per-event"):
        result_from(Counter(), {"cells": Unrollable()})


def test_a_zero_dimensional_numpy_array_is_a_contract_error() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(EvaluationContractError, match="neither a number nor per-event"):
        result_from(Counter(), {"cells": np.array(3.0)})


@pytest.mark.parametrize("key", [("simulation", "a", "b"), ("simulation",), ("s", 5), 5])
def test_a_metric_is_a_name_or_a_pair_of_names(key: Any) -> None:
    class Odd(Evaluator):
        name, version, stage = "demo.odd", "1", "simulation"
        metrics = {key: "1"}

    with pytest.raises(EvaluationContractError, match="pair of strings"):
        metric_units(Odd())
    with pytest.raises(EvaluationContractError, match="pair of strings"):
        result_from(Counter(), {key: 1})


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
    stage = "simulation"
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


# --- quantities: a setting with a unit ---


class Gated(Evaluator):
    name, version = "demo.gated", "1"
    settings = {"threshold": Quantity("GeV"), "radius": Quantity("mm", default=1800.0)}


def test_a_quantity_arrives_in_its_declared_unit_and_its_default_is_used() -> None:
    received = with_defaults(Gated(), {"threshold": {"value": 70, "unit": "keV"}})
    assert received == {"threshold": 70e-6, "radius": 1800.0}
    Gated().check(EvaluationRequest(settings=received))
    radius = {"value": 1.9, "unit": "m"}
    assert with_defaults(Gated(), {"threshold": {"value": 1, "unit": "GeV"}, "radius": radius}) == {
        "threshold": 1.0,
        "radius": 1900.0,
    }


@pytest.mark.parametrize(
    ("threshold", "message"),
    [
        (70, r"settings\.threshold: needs value and unit"),
        ({"value": 70, "unit": "keV", "extra": 1}, "needs value and unit"),
        ({"value": True, "unit": "keV"}, r"threshold\.value: must be a finite number"),
        ({"value": float("inf"), "unit": "keV"}, "finite number"),
        ({"value": 10**320, "unit": "keV"}, r"threshold\.value: too large to express in GeV"),
        ({"value": 1e308, "unit": "TeV"}, "too large to express in GeV"),
        ({"value": 70, "unit": "mm"}, r"threshold\.unit: must be a unit of energy"),
        ({"value": 70, "unit": "kev"}, "must be a unit of energy"),
    ],
)
def test_a_quantity_needs_a_value_and_a_unit_of_its_dimension(
    threshold: object, message: str
) -> None:
    with pytest.raises(EvaluationContractError, match=message):
        with_defaults(Gated(), {"threshold": threshold})


def test_a_conversion_is_exact_whatever_the_decimal_context() -> None:
    # an evaluator or a library it imports may narrow the context of the process
    with decimal.localcontext(prec=5):
        received = with_defaults(Gated(), {"threshold": {"value": 123.456789, "unit": "GeV"}})
        assert received["threshold"] == 123.456789
        assert units.convert(123.456789, "GeV", "MeV") == 123456.789


def test_an_int_beyond_a_float_is_converted_when_its_result_fits() -> None:
    received = with_defaults(Gated(), {"threshold": {"value": 10**310, "unit": "eV"}})
    assert received["threshold"] == 1e301


def test_a_required_quantity_must_be_given_and_is_a_float() -> None:
    with pytest.raises(EvaluationContractError, match=r"missing \['threshold'\]"):
        Gated().check(EvaluationRequest(settings={"radius": 1800.0}))
    with pytest.raises(EvaluationContractError, match=r"settings\.threshold: expected float"):
        Gated().check(EvaluationRequest(settings={"threshold": "70 keV"}))


def test_a_quantity_needs_a_known_unit() -> None:
    with pytest.raises(EvaluationContractError, match="Quantity: unit must be one of"):
        Quantity("furlong")


@pytest.mark.parametrize("default", ["oops", True, float("nan"), float("inf"), [1.0], 10**400])
def test_a_quantity_default_must_be_a_finite_number(default: Any) -> None:
    with pytest.raises(EvaluationContractError, match="Quantity: default must be a finite number"):
        Quantity("GeV", default=default)
    assert Quantity("GeV", default=1).default == 1


# --- collections and events ---


class Pairing(Evaluator):
    name, version, stage = "demo.pairing", "1", "simulation"
    collections = {
        "hits": Collection("SimCalorimeterHit"),
        "particles": Collection("MCParticle", default="MCParticles"),
    }


def pairing(**collections: str) -> EvaluationRequest:
    return EvaluationRequest(
        collections=collections or {"hits": COLLECTION},
        inputs={"simulation": Path("events.root")},
    )


def test_events_hold_each_declared_collection_of_one_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photon = Particle(momentum=(1.0, 0.0, 0.0))
    simulation.install_events(
        monkeypatch,
        [
            Event(**{COLLECTION: Hits([1, 2]), "MCParticles": Particles([photon])}),
            Event(**{COLLECTION: Hits([3]), "MCParticles": Particles([photon, photon])}),
        ],
    )
    events = list(Pairing().events(pairing()))
    assert [(len(e.hits), len(e.particles)) for e in events] == [(2, 1), (1, 2)]
    with pytest.raises(AttributeError, match="has no collection 'hit'; it has hits, particles"):
        events[0].hit  # noqa: B018


def test_the_problem_may_name_another_collection_for_a_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation.install_events(
        monkeypatch, [Event(Other=Hits([1, 2, 3]), Truth=Particles([Particle((1.0, 0, 0))]))]
    )
    (event,) = Pairing().events(pairing(hits="Other", particles="Truth"))
    assert (len(event.hits), len(event.particles)) == (3, 1)


def test_a_missing_or_mistyped_collection_of_an_event_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation.install_events(monkeypatch, [Event(**{COLLECTION: Hits([1])})])
    with pytest.raises(EvaluationFailed, match="'MCParticles' is missing in event 0"):
        list(Pairing().events(pairing()))
    simulation.install_events(
        monkeypatch,
        [Event(**{COLLECTION: Hits([1]), "MCParticles": Hits([1])})],
    )
    with pytest.raises(EvaluationFailed, match="'MCParticles' holds 'edm4hep::SimCalorimeterHit'"):
        list(Pairing().events(pairing()))


def test_roles_are_resolved_from_the_problem_and_the_defaults() -> None:
    assert collection_names(Pairing(), {"hits": "H"}) == {"hits": "H", "particles": "MCParticles"}
    for given, message in [
        ({}, r"expected the roles \['hits', 'particles'\], missing \['hits'\]"),
        ({"hits": "H", "jets": "J"}, r"unknown \['jets'\]"),
        ({"hits": " "}, r"collections\.hits: must be the name of a collection"),
        ({"hits": 3}, r"collections\.hits: must be the name of a collection"),
    ]:
        with pytest.raises(EvaluationContractError, match=message):
            collection_names(Pairing(), given)


def test_an_evaluator_without_collections_takes_none() -> None:
    assert collection_names(Counter(), {}) == {}
    with pytest.raises(EvaluationContractError, match=r"unknown \['hits'\]"):
        collection_names(Counter(), {"hits": COLLECTION})


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"stage": None}, "read from the file of the evaluator's stage"),
        ({"collections": {"2hits": Collection("SimCalorimeterHit")}}, "Python identifier"),
        ({"collections": {"_hits": Collection("SimCalorimeterHit")}}, "Python identifier"),
        ({"collections": {"class": Collection("SimCalorimeterHit")}}, "Python identifier"),
        ({"collections": {"hits": "SimCalorimeterHit"}}, r"declare it as Collection\(type\)"),
        ({"collections": "hits"}, "declare a mapping of role to Collection"),
        ({"collections": []}, "declare a mapping of role to Collection"),
    ],
)
def test_collection_declarations_that_cannot_be_read_are_refused(
    attributes: dict[str, Any], message: str
) -> None:
    evaluator = type("Broken", (Pairing,), attributes)()
    with pytest.raises(EvaluationContractError, match=message):
        declared_collections(evaluator)


@pytest.mark.parametrize(
    ("type_", "default"),
    [("Sim Hit", None), ("", None), ("edm4hep::", None), ("MCParticle", "")],
)
def test_a_collection_needs_a_type_name_and_a_nonblank_default(
    type_: str, default: str | None
) -> None:
    with pytest.raises(EvaluationContractError, match="Collection: "):
        Collection(type_, default)


def test_a_collection_type_is_qualified_as_podio_names_it() -> None:
    assert Collection("SimCalorimeterHit").podio_type == "edm4hep::SimCalorimeterHit"
    assert Collection("edm4hep::MCParticle").podio_type == "edm4hep::MCParticle"


def test_events_need_a_stage_collections_and_the_stage_file() -> None:
    with pytest.raises(EvaluationContractError, match=r"events\(\) needs stage"):
        Counter().events(EvaluationRequest())
    with pytest.raises(EvaluationContractError, match=r"inputs\['simulation'\] is missing"):
        Pairing().events(EvaluationRequest(collections={"hits": COLLECTION}))


def test_an_empty_per_event_list_points_to_invalid() -> None:
    class Empty(Evaluator):
        name, version, stage = "demo.empty", "1", "simulation"

    with pytest.raises(EvaluationContractError, match=r"no per-event values .* self\.invalid"):
        result_from(Empty(), {"cells": []})
