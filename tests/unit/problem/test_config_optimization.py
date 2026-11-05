"""The ``optimization`` block of a problem: the levels of a nested optimization, each parameter in
exactly one, each inner level with its trials and the objective that picks its best trial, and the
criteria a candidate is judged by at each level."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from typing import Any

import pytest

from k4opt.problem import Criteria, Level, Objective, ProblemConfigError, parse_problem
from k4opt.problem.models import essential

THRESHOLD = {
    "type": "continuous",
    "bounds": [0.1, 1.0],
    "unit": "GeV",
    "targets": {"evaluators": {"calo.single_particle_shower": "threshold"}},
}


def reject(data: object, message: str) -> None:
    """Assert that ``data`` is refused with ``message`` (a literal fragment)."""
    with pytest.raises(ProblemConfigError, match=re.escape(message)):
        parse_problem(data)


@pytest.fixture
def nested(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` with a second parameter, the shower's threshold, tuned for each cell size."""
    doc["parameters"]["threshold"] = copy.deepcopy(THRESHOLD)
    doc["optimization"] = {
        "parameters": ["ecal_cell_size"],
        "inner": {"parameters": ["threshold"], "trials": 5},
    }
    return doc


def test_a_nested_problem_has_its_levels_and_their_criteria(nested: dict[str, Any]) -> None:
    spec = parse_problem(nested)
    (objective,) = spec.objectives
    assert spec.optimization == Level(
        ("ecal_cell_size",), inner=Level(("threshold",), trials=5, objective=objective)
    )
    assert [level.parameters for level in spec.levels] == [("ecal_cell_size",), ("threshold",)]
    assert spec.depth == 1
    # the problem's one objective picks the best trial of each inner study, by default
    assert spec.criteria(0) == Criteria(spec.objectives, spec.constraints)
    assert spec.criteria(1) == Criteria((objective,), spec.constraints)
    with pytest.raises(ValueError, match="the problem's levels are 0 to 1"):
        spec.criteria(2)
    assert parse_problem(spec.to_dict()) == spec  # written out in full, and read back


def test_an_inner_objective_may_be_any_metric_and_stays_out_of_the_top_criteria(
    nested: dict[str, Any],
) -> None:
    nested["optimization"]["inner"]["objective"] = {
        "stage": "simulation",
        "metric": "a.efficiency",
        "direction": "maximize",
    }
    spec = parse_problem(nested)
    shower = "calo.single_particle_shower"
    inner = Objective(shower, "simulation", "a.efficiency", "maximize", sample="gamma")
    assert spec.criteria(1).objectives == (inner,)
    assert inner.ref not in spec.references  # the top level is judged without it
    # but every level's metrics are measured: a valid evaluation carries each of them
    assert spec.measured == (*spec.references, inner.ref)


def test_an_inner_objective_is_named_by_metric_or_label(nested: dict[str, Any]) -> None:
    nested["objectives"].append(
        {"stage": "simulation", "metric": "a.mean", "direction": "maximize"}
    )
    for name in ("a.mean", "gamma/calo.single_particle_shower/simulation/a.mean"):
        nested["optimization"]["inner"]["objective"] = name
        assert parse_problem(nested).criteria(1).objectives == (
            parse_problem(nested).objectives[1],
        )


def test_an_ordinary_block_is_an_ordinary_problem(doc: dict[str, Any]) -> None:
    plain = parse_problem(doc)
    doc["optimization"] = {"parameters": ["ecal_cell_size"]}
    spec = parse_problem(doc)
    assert spec == plain and spec.optimization is None and spec.depth == 0


def test_a_title_of_an_inner_objective_is_no_part_of_the_problem(nested: dict[str, Any]) -> None:
    nested["optimization"]["inner"]["objective"] = {
        "stage": "simulation",
        "metric": "a.efficiency",
        "direction": "maximize",
    }
    plain = parse_problem(nested).to_dict()
    nested["optimization"]["inner"]["objective"]["title"] = "Efficiency"
    titled = parse_problem(nested).to_dict()
    assert titled != plain and essential(titled) == essential(plain)


def _cut() -> dict[str, Any]:
    """A third parameter: another setting of the evaluator."""
    cut = copy.deepcopy(THRESHOLD)
    cut["targets"] = {"evaluators": {"calo.single_particle_shower": "cut"}}
    return cut


def test_three_levels(nested: dict[str, Any]) -> None:
    nested["parameters"]["cut"] = _cut()
    nested["optimization"]["inner"]["inner"] = {"parameters": ["cut"], "trials": 3}
    spec = parse_problem(nested)
    assert spec.depth == 2
    assert [level.trials for level in spec.levels] == [None, 5, 3]
    assert spec.criteria(2) == spec.criteria(1)


REFUSED: dict[str, tuple[Callable[[dict[str, Any]], object], str]] = {
    "no parameters": (
        lambda d: d["optimization"].pop("parameters"),
        "optimization: missing required key(s) ['parameters']",
    ),
    "empty parameters": (
        lambda d: d["optimization"]["inner"].update(parameters=[]),
        "optimization.inner.parameters: must list at least one parameter",
    ),
    "unknown parameter": (
        lambda d: d["optimization"]["inner"]["parameters"].append("pitch"),
        "optimization.inner.parameters[1]: 'pitch' is no parameter of the problem; its "
        "parameters are ecal_cell_size, threshold",
    ),
    "listed twice": (
        lambda d: d["optimization"]["inner"]["parameters"].append("threshold"),
        "optimization.inner.parameters[1]: 'threshold' is listed twice",
    ),
    "in two levels": (
        lambda d: d["optimization"]["inner"]["parameters"].append("ecal_cell_size"),
        "optimization.inner.parameters: 'ecal_cell_size' is in optimization.parameters too",
    ),
    "a parameter left out": (
        lambda d: d["parameters"].update(cut=_cut()),
        "optimization: the parameter(s) cut are in no level",
    ),
    "trials at the top": (
        lambda d: d["optimization"].update(trials=10),
        "optimization.trials: the top level's trials are how long a run goes on; give them on "
        "the command line (k4opt optimize --trials N)",
    ),
    "an objective at the top": (
        lambda d: d["optimization"].update(objective="a.rms"),
        "optimization.objective: the problem's objectives judge the top level",
    ),
    "no inner trials": (
        lambda d: d["optimization"]["inner"].pop("trials"),
        "optimization.inner: missing trials: how many trials each study of this level runs",
    ),
    "no trial": (
        lambda d: d["optimization"]["inner"].update(trials=0),
        "optimization.inner.trials",
    ),
    "an unknown objective": (
        lambda d: d["optimization"]["inner"].update(objective="a.mean"),
        "optimization.inner.objective: 'a.mean' is no objective of the problem "
        "(gamma/calo.single_particle_shower/simulation/a.rms)",
    ),
    "an unknown key": (
        lambda d: d["optimization"]["inner"].update(sampler="tpe"),
        "optimization.inner",
    ),
    "events from a file": (
        lambda d: d["samples"]["gamma"].update(generation={"file": "zh.hepmc"}),
        "optimization.inner: the best trial of each inner study is evaluated again on new "
        "events, but samples.gamma reads its generation from a file",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_what_an_optimization_block_may_not_say(nested: dict[str, Any], case: str) -> None:
    edit, message = REFUSED[case]
    edit(nested)
    reject(nested, message)


def test_several_objectives_need_an_inner_objective(nested: dict[str, Any]) -> None:
    nested["objectives"].append(
        {"stage": "simulation", "metric": "a.mean", "direction": "maximize"}
    )
    reject(
        nested,
        "optimization.inner: the problem has the objectives "
        "gamma/calo.single_particle_shower/simulation/a.rms, "
        "gamma/calo.single_particle_shower/simulation/a.mean; say which one picks the best trial",
    )


def test_a_metric_of_several_objectives_is_named_by_label(nested: dict[str, Any]) -> None:
    nested["samples"]["pion"] = copy.deepcopy(nested["samples"]["gamma"])
    nested["samples"]["pion"]["simulation"]["seed"] = 43
    nested["evaluators"][0]["samples"] = ["gamma", "pion"]
    nested["objectives"] = [
        {"sample": "gamma", "stage": "simulation", "metric": "a.rms", "direction": "minimize"},
        {"sample": "pion", "stage": "simulation", "metric": "a.rms", "direction": "minimize"},
    ]
    nested["constraints"][0]["sample"] = "gamma"
    nested["optimization"]["inner"]["objective"] = "a.rms"
    reject(nested, "'a.rms' is the metric of the objectives gamma/")
    nested["optimization"]["inner"]["objective"] = (
        "pion/calo.single_particle_shower/simulation/a.rms"
    )
    assert parse_problem(nested).criteria(1).objectives[0].sample == "pion"
