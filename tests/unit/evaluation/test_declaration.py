"""What an implementation declares, as plain data: read from an instance, written as a record
holds it and read back the same, with or without metrics, artifacts and roles."""

from __future__ import annotations

import json
from typing import Any

import pytest

from k4opt.errors import K4OptError
from k4opt.evaluation import Collection, Declaration, Evaluator, Quantity, Role
from k4opt.frozen import thaw

SELECTED = ("analysis", "selected.sum_weights")
GENERATED = ("analysis", "generated.sum_weights")


class Plain(Evaluator):
    """Declares nothing but its name and version."""

    name = "test.plain"
    version = "1"


class Reader(Evaluator):
    """Reads the simulation, with metrics, artifacts, settings and collections."""

    name = "test.reader"
    version = "3"
    stage = "simulation"
    metrics = {"ecal.rms": "mm", ("analysis", "cells"): "1"}
    artifacts = {"histogram": "the widths, JSON", "plot": "a PNG"}
    settings = {"threshold": Quantity("GeV", default=0.1)}
    collections = {"hits": Collection("SimCalorimeterHit")}


class Combined(Evaluator):
    """A role of each kind: one item, many, one with artifacts, and a stage's file."""

    name = "test.combined"
    version = "2"
    metrics = {("analysis", "significance"): "1"}
    roles = {
        "signal": Role(metrics=[SELECTED, GENERATED]),
        "backgrounds": Role(many=True, metrics=[SELECTED]),
        "plots": Role(artifacts=["histogram"]),
        "events": Role(stage="simulation"),
    }


class Empty(Evaluator):
    """Declares empty metrics, artifacts and roles, which are not the same as none."""

    name = "test.empty"
    version = "1"
    metrics: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    roles: dict[str, Role] = {}


@pytest.mark.parametrize("cls", [Plain, Reader, Combined, Empty])
def test_a_declaration_reads_back_as_it_was_written(cls: type[Evaluator]) -> None:
    declared = Declaration.of(cls())
    assert Declaration.from_dict(declared.name, declared.to_dict(), K4OptError) == declared
    # and as JSON gives it back from a summary
    data = json.loads(json.dumps(thaw(declared.to_dict())))
    assert Declaration.from_dict(declared.name, data, K4OptError) == declared


def test_a_declaration_holds_what_the_record_needs_and_nothing_python() -> None:
    declared = Declaration.of(Reader())
    assert declared == Declaration(
        "test.reader",
        "3",
        "simulation",
        {("simulation", "ecal.rms"): "mm", ("analysis", "cells"): "1"},
        ("histogram", "plot"),
        None,
    )
    assert thaw(declared.to_dict()) == {
        "version": "3",
        "stage": "simulation",
        "metrics": [
            {"stage": "simulation", "metric": "ecal.rms", "unit": "mm"},
            {"stage": "analysis", "metric": "cells", "unit": "1"},
        ],
        "artifacts": ["histogram", "plot"],
        "roles": None,
    }  # no settings, collections or what each artifact holds: those stay with the instance


def test_none_and_empty_declarations_stay_apart() -> None:
    plain, empty = Declaration.of(Plain()), Declaration.of(Empty())
    assert (plain.metrics, plain.artifacts, plain.roles) == (None, None, None)
    assert (empty.metrics, empty.artifacts, empty.roles) == ({}, (), {})
    assert plain.to_dict()["metrics"] is None and empty.to_dict()["metrics"] == ()


def test_the_roles_of_a_declaration_keep_each_kind() -> None:
    roles = Declaration.of(Combined()).roles
    assert roles is not None
    assert roles["signal"] == Role(metrics=[SELECTED, GENERATED])
    assert roles["backgrounds"].many and roles["plots"].artifacts == ("histogram",)
    assert roles["events"].stage == "simulation"


class RefusedError(K4OptError):
    """The error a reader of declarations asks for."""


def _written() -> dict[str, Any]:
    data = thaw(Declaration.of(Combined()).to_dict())
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(version=""), "version must be a nonblank string"),
        (lambda d: d.update(stage=3), "stage must be null or a nonblank string"),
        (lambda d: d.update(metrics={}), "metrics must be a list"),
        (lambda d: d["metrics"].append(dict(d["metrics"][0])), "is declared twice"),
        (lambda d: d["metrics"][0].update(unit=" "), "metrics[0].unit must be a nonblank"),
        (lambda d: d.update(artifacts=["a#b"]), "'a#b' is no artifact name"),
        (lambda d: d["roles"]["signal"].update(many="no"), "many must be True or False"),
        (lambda d: d["roles"]["signal"].update(metrics=[["analysis"]]), "(stage, metric) pairs"),
        (lambda d: d["roles"]["events"].update(metrics=[list(SELECTED)]), "reads no metrics"),
        (lambda d: d.update(roles=[]), "roles must be a mapping"),
    ],
)
def test_data_that_is_no_declaration_raises_the_error_asked_for(change: Any, message: str) -> None:
    data = _written()
    change(data)
    with pytest.raises(RefusedError, match=r"^implementation 'test.combined'") as raised:
        Declaration.from_dict("test.combined", data, RefusedError)
    assert message in str(raised.value)
