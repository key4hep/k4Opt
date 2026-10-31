"""The immutable models: freezing, serialization and aliasing."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import unquote

import pytest

from k4opt import frozen as frozen_module
from k4opt.problem import (
    EvaluatorSpec,
    MetricRef,
    ProblemConfigError,
    ProblemSpec,
    Sample,
    Stage,
    parse_problem,
)
from k4opt.problem.models import freeze, thaw


def test_to_dict_is_plain_json_data(doc: dict[str, Any]) -> None:
    data = parse_problem(doc).to_dict()
    # left out where the problem has one sample and one evaluator, always in the normalized form
    doc["evaluators"][0]["samples"] = ["gamma"]
    for index, item in enumerate(doc["objectives"]):
        doc["objectives"][index] = {"sample": "gamma", "evaluator": "calo.single_particle_shower"}
        doc["objectives"][index] |= item
    for index, item in enumerate(doc["constraints"]):
        doc["constraints"][index] = {"sample": "gamma", "evaluator": "calo.single_particle_shower"}
        doc["constraints"][index] |= item
    assert data == doc  # same content, same shape
    assert list(data["objectives"][0]) == list(doc["objectives"][0])  # type: ignore[index]
    assert list(data) == list(doc)  # schema order

    def check(value: object) -> None:
        assert type(value) in {dict, list, str, int, float, bool, type(None)}
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str)
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(data)


def test_round_trip(doc: dict[str, Any]) -> None:
    problem = parse_problem(doc)
    assert parse_problem(problem.to_dict()) == problem
    assert problem.to_dict() == problem.to_dict()


def test_mutating_the_input_does_not_change_the_spec(doc: dict[str, Any]) -> None:
    problem = parse_problem(doc)
    before = problem.to_dict()
    doc["samples"]["gamma"]["generation"]["particle"] = "e-"
    doc["samples"]["gamma"]["generation"]["direction"]["theta"].append(1)
    doc["evaluators"][0]["settings"]["collection"] = "Other"
    doc["parameters"][0]["bounds"][0] = -1
    doc["objectives"].clear()
    assert problem.to_dict() == before


def test_mutating_to_dict_does_not_change_the_spec(doc: dict[str, Any]) -> None:
    problem = parse_problem(doc)
    before = problem.to_dict()
    data = cast(dict[str, Any], problem.to_dict())
    data["samples"]["gamma"]["generation"]["particle"] = "e-"
    data["samples"]["gamma"]["simulation"]["physics.list"] = "QGSP_BERT"
    data["parameters"].clear()
    assert problem.to_dict() == before


def test_stored_values_are_read_only(doc: dict[str, Any]) -> None:
    problem = parse_problem(doc)
    stages = problem.samples["gamma"].stages
    generation = stages["generation"]
    with pytest.raises(TypeError):
        generation.options["particle"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        stages["other"] = generation  # type: ignore[index]
    with pytest.raises(TypeError):
        problem.samples["other"] = problem.samples["gamma"]  # type: ignore[index]
    with pytest.raises(AttributeError):
        stages["simulation"].events = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        problem.evaluators[0].samples = ()  # type: ignore[misc]
    assert isinstance(generation.options["direction"]["theta"], tuple)  # type: ignore[index]


def test_directly_constructed_models_copy_their_input(doc: dict[str, Any]) -> None:
    settings: dict[str, Any] = {"a": {"b": [1, 2]}}
    options: dict[str, Any] = {"action.run": ["A"]}
    measured = ["gamma"]
    evaluators = [
        EvaluatorSpec(
            "e",
            "1",
            settings=settings,
            stage="simulation",
            samples=measured,  # type: ignore[arg-type]
        )
    ]
    stages = {"simulation": Stage("simulation", executor="ddsim", events=1, options=options)}
    samples = {"gamma": Sample("gamma", stages)}
    problem = parse_problem(doc)
    spec = ProblemSpec(
        problem.environment,
        problem.geometry,
        problem.parameters,
        evaluators,  # type: ignore[arg-type]
        problem.objectives,
        samples=samples,
    )
    settings["a"]["b"].append(3)
    options["action.run"].append("B")
    evaluators.append(EvaluatorSpec("f", "1"))
    measured.append("pion")
    stages.clear()
    samples.clear()
    data: Any = spec.to_dict()
    assert data["evaluators"] == [
        {
            "name": "e",
            "version": "1",
            "stage": "simulation",
            "samples": ["gamma"],
            "settings": {"a": {"b": [1, 2]}},
        }
    ]
    assert data["samples"] == {
        "gamma": {"simulation": {"executor": "ddsim", "events": 1, "action.run": ["A"]}}
    }
    assert isinstance(spec.evaluators, tuple)


@pytest.mark.parametrize(
    "fields", [{}, {"executor": "gun", "file": "events.hepmc"}], ids=["neither", "both"]
)
def test_a_stage_is_run_or_given_as_a_file(fields: dict[str, Any]) -> None:
    with pytest.raises(ProblemConfigError, match="needs exactly one of 'executor' and 'file'"):
        Stage("generation", **fields)


def test_spec_is_hashable_and_picklable(doc: dict[str, Any]) -> None:
    problem = parse_problem(doc)
    assert hash(problem) == hash(parse_problem(doc))
    assert pickle.loads(pickle.dumps(problem)) == problem


def test_optional_parts_are_omitted_from_to_dict(doc: dict[str, Any]) -> None:
    for key in ("name", "constraints", "samples"):
        del doc[key]
    del doc["evaluators"][0]["settings"], doc["evaluators"][0]["stage"]
    data = parse_problem(doc).to_dict()
    assert "name" not in data and "constraints" not in data and "samples" not in data
    assert data["evaluators"] == [{"name": "calo.single_particle_shower", "version": "1"}]


def test_a_stage_without_seed_omits_it(doc: dict[str, Any]) -> None:
    del doc["samples"]["gamma"]["simulation"]["seed"]
    simulation = parse_problem(doc).to_dict()["samples"]["gamma"]["simulation"]  # type: ignore[index]
    assert simulation == {"executor": "ddsim", "events": 1000, "physics.list": "FTFP_BERT"}


def test_canonical_target(doc: dict[str, Any]) -> None:
    doc["parameters"].append(
        {"geometry": {"target": "GlobalSolenoid.inner_field"}, "type": "integer", "bounds": [1, 2]}
    )
    first, second = parse_problem(doc).parameters
    assert first.geometry.canonical_target == "constants.ECal_cell_size"
    assert second.geometry.canonical_target == "GlobalSolenoid.inner_field"


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        ({1: "x"}, "keys must be strings"),
        ({"a": {"b": {1, 2}}}, r"p\.a\.b: unsupported value of type set"),
        (b"bytes", "unsupported value of type bytes"),
    ],
)
def test_freeze_rejects_unrepresentable_values(value: object, match: str) -> None:
    with pytest.raises(ProblemConfigError, match=match):
        freeze({"a": {"b": value}} if not isinstance(value, dict) else value, "p")


def test_freeze_rejects_cycles_and_excessive_depth() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ProblemConfigError, match="contains itself"):
        freeze({"a": cycle}, "p")
    deep: object = []
    for _ in range(200):
        deep = [deep]
    with pytest.raises(ProblemConfigError, match="nested deeper"):
        freeze(deep, "p")


def test_shared_but_acyclic_values_are_fine() -> None:
    shared = {"x": 1}
    assert thaw(freeze({"a": shared, "b": shared}, "p")) == {"a": {"x": 1}, "b": {"x": 1}}


def _alias_tower(depth: int) -> list[object]:
    """A list whose two items are the same list, nested ``depth`` times (like YAML aliases)."""
    node: list[object] = ["x", "x"]
    for _ in range(depth):
        node = [node, node]
    return node


def test_shared_values_are_frozen_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = frozen_module._freeze

    def counting(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(frozen_module, "_freeze", counting)
    freeze({"tower": _alias_tower(17)}, "p")
    assert calls < 100  # one visit per distinct container, not 2**17


def test_aliases_that_expand_too_far_are_refused() -> None:
    with pytest.raises(ProblemConfigError, match="aliases expand to more than"):
        freeze({"tower": _alias_tower(40)}, "p")


def test_aliases_below_the_limit_are_kept_and_shared_results_are_equal() -> None:
    frozen = freeze({"tower": _alias_tower(10)}, "p")
    assert isinstance(frozen, Mapping)
    tower = frozen["tower"]
    assert isinstance(tower, tuple)
    assert tower[0] == tower[1]
    assert len(thaw(frozen)["tower"]) == 2  # type: ignore[index]


def test_alias_tower_in_a_document_parses_quickly(doc: dict[str, Any]) -> None:
    doc["evaluators"][0]["settings"] = {"tower": _alias_tower(17)}
    problem = parse_problem(doc)
    assert parse_problem(doc) == problem


def test_equal_specs_hash_equally_whatever_the_key_order(doc: dict[str, Any]) -> None:
    doc["evaluators"][0]["settings"] = {"x": 1, "y": {"a": 1, "b": 2}}
    first = parse_problem(doc)
    doc["evaluators"][0]["settings"] = {"y": {"b": 2, "a": 1}, "x": 1}
    second = parse_problem(doc)
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    # insertion order is still what iteration and serialization show
    assert list(second.evaluators[0].settings) == ["y", "x"]
    evaluators: Any = second.to_dict()["evaluators"]
    assert list(evaluators[0]["settings"]) == ["y", "x"]


def test_different_specs_are_still_different(doc: dict[str, Any]) -> None:
    other = copy.deepcopy(doc)
    other["samples"]["gamma"]["simulation"]["seed"] = 43
    assert parse_problem(doc) != parse_problem(other)
    renamed = copy.deepcopy(doc)  # a name is a label, but objectives and records refer to it
    renamed["samples"] = {"photon": renamed["samples"]["gamma"]}
    assert parse_problem(doc) != parse_problem(renamed)


def _nested(levels: int, leaf: object) -> object:
    node = leaf
    for _ in range(levels):
        node = [node]
    return node


@pytest.mark.parametrize("shallow_first", [True, False])
def test_depth_of_a_shared_value_is_checked_at_every_use(shallow_first: bool) -> None:
    shared = _nested(40, "x")  # 40 levels deep on its own
    near, far = ("near", "far") if shallow_first else ("far", "near")
    document: dict[str, object] = {}
    for name in (near, far):
        document[name] = shared if name == "near" else _nested(30, shared)  # 71 levels
    with pytest.raises(ProblemConfigError, match="nested deeper"):
        freeze(document, "p")


def test_shared_value_within_the_depth_limit_is_fine_in_any_order() -> None:
    shared = _nested(20, "x")
    for order in (("a", "b"), ("b", "a")):
        document = {name: shared if name == "a" else _nested(20, shared) for name in order}
        assert thaw(freeze(document, "p")) == {"a": shared, "b": _nested(20, shared)}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (MetricRef("cost", "analysis/x", "y"), MetricRef("cost", "analysis", "x/y")),
        (MetricRef("cost", "analysis", "x/y"), MetricRef("cost", "analysis", "x%2Fy")),
        # four parts with a sample, three without: the last two are always stage and metric
        (MetricRef("calo", "s", "m", "gamma"), MetricRef("gamma", "calo", "s/m")),
    ],
)
def test_two_metric_references_never_share_a_label(first: MetricRef, second: MetricRef) -> None:
    # a stage or metric name may hold "/": each part is escaped, so the label tells them apart
    assert first.label != second.label
    for ref in (first, second):  # and each part reads back
        parts = ref.label.split("/")
        owner = [ref.evaluator] if ref.sample is None else [ref.sample, ref.evaluator]
        assert [unquote(part) for part in parts] == [*owner, ref.stage, ref.metric]


def test_a_label_of_plain_names_is_the_names_joined() -> None:
    ref = MetricRef("calo.photon", "simulation", "ecal_barrel.fired_cells")
    assert ref.label == "calo.photon/simulation/ecal_barrel.fired_cells"
    assert MetricRef("e", "analysis", "50%").label == "e/analysis/50%25"
    measured = MetricRef("calo.photon", "simulation", "ecal_barrel.fired_cells", "gamma_10gev")
    assert measured.label == "gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells"
    assert (measured.invocation, ref.invocation) == ("gamma_10gev/calo.photon", "calo.photon")
