"""The evaluators of a problem document: sample properties and the roles of combined
evaluators (no Key4hep)."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from k4opt.problem import ProblemConfigError, Reference, parse_problem
from tests.support.problems import two_samples

Write = Callable[..., Path]
EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "problem" / "cld-ecal-granularity.yaml"


@pytest.fixture
def write(tmp_path: Path) -> Write:
    """Write text/bytes to a file under ``tmp_path`` and return its path."""

    def _write(content: bytes | str, name: str = "problem.yaml") -> Path:
        path = tmp_path / name
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    return _write


def reject(data: object, message: str) -> None:
    """Assert that ``data`` is refused with ``message`` (a literal fragment)."""
    with pytest.raises(ProblemConfigError, match=re.escape(message)):
        parse_problem(data)


# --- sample properties and combined evaluators ----------------------------------------------


def test_a_sample_keeps_its_properties_as_given(doc: dict[str, Any]) -> None:
    properties = {"process": "e+e- -> ZH", "cross_section": {"value": 200, "unit": "fb"}}
    doc["samples"]["gamma"] = {"properties": properties, **doc["samples"]["gamma"]}
    problem = parse_problem(doc)
    assert problem.samples["gamma"].properties == properties
    assert list(problem.samples["gamma"].stages) == ["generation", "simulation"]
    assert problem.to_dict()["samples"]["gamma"]["properties"] == properties  # type: ignore[index]
    assert parse_problem(problem.to_dict()) == problem
    without = copy.deepcopy(doc)
    del without["samples"]["gamma"]["properties"]
    assert "properties" not in parse_problem(without).to_dict()["samples"]["gamma"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("properties", "message"),
    [
        ("ZH", "samples.gamma.properties: must be a mapping, got str"),
        ({"x": float("nan")}, "samples.gamma.properties.x"),
    ],
)
def test_properties_are_json_mappings(
    doc: dict[str, Any], properties: object, message: str
) -> None:
    doc["samples"]["gamma"]["properties"] = properties
    reject(doc, message)


def test_properties_alone_are_no_sample(doc: dict[str, Any]) -> None:
    doc["samples"]["gamma"] = {"properties": {"process": "ZH"}}
    reject(doc, "samples.gamma: must define its stages")


def combined(doc: dict[str, Any], roles: object, **entry: Any) -> dict[str, Any]:
    """``doc`` with two samples, ``gamma`` and ``pion``, and a combined evaluator ``sep`` with
    ``roles`` (and the keys ``entry``), whose metric is the objective."""
    doc = two_samples(doc)
    doc["evaluators"].append({"name": "calo.separation", "as": "sep", "version": "1"} | entry)
    doc["evaluators"][-1]["roles"] = roles
    doc["objectives"] = [
        {"evaluator": "sep", "stage": "analysis", "metric": "rejection", "direction": "maximize"}
    ]
    doc["constraints"] = []
    return doc


SHOWER = "calo.single_particle_shower"


def test_a_role_reads_evaluations_and_stages_of_samples(doc: dict[str, Any]) -> None:
    doc = combined(
        doc,
        {
            "signal": {"sample": "gamma", "evaluator": SHOWER},
            # one item per sample, and a list of references
            "backgrounds": [
                {"samples": ["pion"], "evaluator": SHOWER},
                {"sample": "pion", "stage": "simulation"},
            ],
            "all": {"samples": ["gamma", "pion"], "evaluator": SHOWER},  # gamma in two roles
        },
    )
    problem = parse_problem(doc)
    (sep,) = [e for e in problem.evaluators if e.binding == "sep"]
    assert (sep.stage, sep.samples) == (None, ())
    assert {role: [i.label for i in items] for role, items in sep.roles.items()} == {
        "signal": [f"gamma/{SHOWER}"],
        "backgrounds": [f"pion/{SHOWER}", "pion/simulation"],
        "all": [f"gamma/{SHOWER}", f"pion/{SHOWER}"],
    }
    assert sep.roles["backgrounds"][1] == Reference("pion", stage="simulation")
    assert [i.label for i in sep.reads] == [f"gamma/{SHOWER}", f"pion/{SHOWER}", "pion/simulation"]
    # normalized: every item on its own
    assert problem.to_dict()["evaluators"][1]["roles"]["signal"] == [  # type: ignore[index]
        {"sample": "gamma", "evaluator": SHOWER}
    ]
    assert parse_problem(problem.to_dict()) == problem
    assert problem.objectives[0].label == "sep/analysis/rejection"


def test_a_sample_only_a_role_reads_is_used(doc: dict[str, Any]) -> None:
    doc = combined(doc, {"events": {"sample": "pion", "stage": "simulation"}})
    doc["evaluators"][0]["samples"] = ["gamma"]
    problem = parse_problem(doc)
    assert [e.samples for e in problem.evaluators] == [("gamma",), ()]


@pytest.mark.parametrize(
    ("roles", "entry", "message"),
    [
        ({}, {}, "evaluators[1].roles: must name at least one role"),
        ([], {}, "evaluators[1].roles: must be a mapping, got list"),
        ({"signal": []}, {}, "evaluators[1].roles.signal: must reference at least one"),
        (
            {"signal": {"sample": "gamma"}},
            {},
            "evaluators[1].roles.signal: needs exactly one of 'evaluator' (its evaluation) and "
            "'stage' (its file)",
        ),
        (
            {"signal": {"evaluator": SHOWER}},
            {},
            "evaluators[1].roles.signal: needs exactly one of 'sample' and 'samples'",
        ),
        (
            {"signal": {"sample": "gamma", "evaluator": SHOWER, "unit": "1"}},
            {},
            "evaluators[1].roles.signal: unknown key(s) ['unit']",
        ),
        (
            {"signal": {"samples": [], "evaluator": SHOWER}},
            {},
            "evaluators[1].roles.signal.samples: must list at least one sample",
        ),
        (
            {"signal": [{"sample": "gamma", "evaluator": SHOWER}, "gamma"]},
            {},
            "evaluators[1].roles.signal[1]: must be a mapping, got str",
        ),
        (
            {"signal": {"samples": ["gamma", "gamma"], "evaluator": SHOWER}},
            {},
            f"evaluators[1].roles.signal: gamma/{SHOWER} is given twice in this role, so the "
            "evaluator would read it twice; give it once",
        ),
        (
            {"signal": {"sample": "neutron", "evaluator": SHOWER}},
            {},
            "evaluators[1].roles.signal: the problem has no sample 'neutron'; its samples are "
            "gamma, pion",
        ),
        (
            {"signal": {"sample": "gamma", "evaluator": "calo.other"}},
            {},
            "evaluators[1].roles.signal: the problem has no evaluator 'calo.other'",
        ),
        (
            {"signal": {"sample": "gamma", "evaluator": "sep"}},
            {},
            "evaluators[1].roles.signal: 'sep' measures the candidate, not a sample; chained "
            "combined evaluators are not supported yet",
        ),
        (
            {"signal": {"sample": "gamma", "stage": "reconstruction"}},
            {},
            "evaluators[1].roles.signal.stage: must be one of ['generation', 'simulation']",
        ),
        (
            {"signal": {"sample": "gamma", "stage": "generation"}},
            {},
            "evaluators[1].roles.signal: gamma/generation is fired inside "
            "samples.gamma.simulation and makes no file of its own; read "
            "samples.gamma.simulation instead",
        ),
        (
            {"signal": {"sample": "gamma", "evaluator": SHOWER}},
            {"stage": "simulation"},
            "evaluators[1]: a combined evaluator (roles:) reads no stage of its own, so it takes "
            "no stage; its roles name what it reads",
        ),
        (
            {"signal": {"sample": "gamma", "evaluator": SHOWER}},
            {"samples": ["gamma"], "collections": {"hits": "X"}},
            "so it takes no samples, collections",
        ),
    ],
)
def test_role_errors(doc: dict[str, Any], roles: object, entry: Any, message: str) -> None:
    reject(combined(doc, roles, **entry), message)


def test_a_role_reads_an_evaluation_of_a_sample_its_evaluator_measures(
    doc: dict[str, Any],
) -> None:
    doc = combined(doc, {"signal": {"sample": "pion", "evaluator": SHOWER}})
    doc["evaluators"][0]["samples"] = ["gamma"]
    doc["evaluators"].append(
        {"name": "calo.other", "version": "1", "stage": "simulation", "samples": ["pion"]}
    )
    reject(
        doc,
        f"evaluators[1].roles.signal: evaluator {SHOWER!r} does not measure 'pion'; it measures "
        "gamma",
    )


def test_a_role_reads_a_stage_its_sample_defines(doc: dict[str, Any]) -> None:
    doc = combined(doc, {"events": {"sample": "pion", "stage": "generation"}})
    doc["samples"]["pion"]["generation"] = {"file": "pion.root"}
    assert parse_problem(doc)  # a file: it is read
    doc["evaluators"][0]["samples"] = ["gamma"]
    del doc["samples"]["pion"]["simulation"]
    doc["evaluators"][1]["roles"] = {"events": {"sample": "pion", "stage": "simulation"}}
    reject(
        doc,
        "evaluators[1].roles.events: sample pion defines no simulation stage; add "
        "samples.pion.simulation",
    )
