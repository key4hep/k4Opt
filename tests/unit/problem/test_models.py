"""The immutable models: freezing, serialization and aliasing."""

from __future__ import annotations

import pickle
from typing import Any, cast

import pytest

from k4opt.problem import Evaluation, ProblemConfigError, parse_problem
from k4opt.problem.models import freeze, thaw


def test_to_dict_is_plain_json_data(doc: dict[str, Any]) -> None:
    data = parse_problem(doc).to_dict()
    assert data == doc  # same content, same shape
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
    doc["evaluation"]["settings"]["generator"]["particle"] = "e-"
    doc["evaluation"]["backend"]["ddsim"]["args"].append("--extra")
    doc["parameters"][0]["bounds"][0] = -1
    doc["objectives"].clear()
    assert problem.to_dict() == before


def test_mutating_to_dict_does_not_change_the_spec(doc: dict[str, Any]) -> None:
    problem = parse_problem(doc)
    before = problem.to_dict()
    data = cast(dict[str, Any], problem.to_dict())
    data["evaluation"]["settings"]["generator"]["particle"] = "e-"
    data["evaluation"]["backend"]["ddsim"]["args"].append("--extra")
    data["parameters"].clear()
    assert problem.to_dict() == before


def test_stored_values_are_read_only(doc: dict[str, Any]) -> None:
    evaluation = parse_problem(doc).evaluation
    with pytest.raises(TypeError):
        evaluation.settings["generator"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        evaluation.events = 1  # type: ignore[misc]
    assert isinstance(evaluation.backend["ddsim"], tuple)


def test_directly_constructed_evaluation_copies_its_input() -> None:
    settings: dict[str, Any] = {"a": {"b": [1, 2]}}
    evaluation = Evaluation("e", "1", settings=settings)
    settings["a"]["b"].append(3)
    assert evaluation.to_dict()["settings"] == {"a": {"b": [1, 2]}}


def test_spec_is_hashable_and_picklable(doc: dict[str, Any]) -> None:
    problem = parse_problem(doc)
    assert hash(problem) == hash(parse_problem(doc))
    assert pickle.loads(pickle.dumps(problem)) == problem


def test_optional_parts_are_omitted_from_to_dict(doc: dict[str, Any]) -> None:
    for key in ("name", "constraints"):
        del doc[key]
    for key in ("events", "seed", "settings", "backend"):
        del doc["evaluation"][key]
    data = parse_problem(doc).to_dict()
    assert "name" not in data and "constraints" not in data
    assert data["evaluation"] == {"evaluator": "calo.single_particle_shower", "version": "1"}


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
