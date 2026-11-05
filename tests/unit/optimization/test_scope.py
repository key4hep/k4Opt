"""The scope of a study: what it searches, what it fixes and how its trials are judged, for an
ordinary study and for a study of each level of a nested optimization, which is started for a
candidate of the levels above and so fixes their values."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from k4opt.optimization import OptimizationError
from k4opt.optimization._scope import StudyScope
from k4opt.problem import ProblemSpec, parse_problem
from tests.support.problems import document

SETTING = {"type": "continuous", "bounds": [0.1, 1.0], "unit": "GeV"}


def nested() -> ProblemSpec:
    """Three levels: the cell size, then a threshold, then a cut, each searched for every
    candidate of the level above; the problem's one objective picks at every level."""
    doc: dict[str, Any] = document()
    for name in ("threshold", "cut"):
        doc["parameters"][name] = copy.deepcopy(SETTING) | {
            "targets": {"evaluators": {"calo.single_particle_shower": name}}
        }
    doc["optimization"] = {
        "parameters": ["ecal_cell_size"],
        "inner": {
            "parameters": ["threshold"],
            "trials": 5,
            "inner": {"parameters": ["cut"], "trials": 3},
        },
    }
    return parse_problem(doc)


def test_an_ordinary_study_searches_every_parameter_but_those_it_fixes() -> None:
    spec = parse_problem(document())
    scope = StudyScope.of(spec)
    assert (scope.level, scope.searched, dict(scope.fixed), scope.replicate) == (
        0,
        ("ecal_cell_size",),
        {},
        0,
    )
    assert scope.criteria == spec.criteria(0)
    assert scope.candidate({"ecal_cell_size": 5.0}) == {"ecal_cell_size": 5.0}


def test_a_study_of_each_level_fixes_the_values_of_the_levels_above() -> None:
    spec = nested()
    top = StudyScope.of(spec)
    assert (top.searched, top.replicate) == (("ecal_cell_size",), 2)
    middle = StudyScope.of(
        spec, {"ecal_cell_size": 5.0}, level=1, parent={"study": "a", "trial": 3}
    )
    assert (middle.searched, middle.replicate, middle.criteria) == (
        ("threshold",),
        1,
        spec.criteria(1),
    )
    assert dict(middle.parent or {}) == {"study": "a", "trial": 3}
    # without the cut, which the studies of the level below choose
    assert middle.candidate({"threshold": 0.5}) == {"ecal_cell_size": 5.0, "threshold": 0.5}
    last = StudyScope.of(spec, {"threshold": 0.5, "ecal_cell_size": 5.0}, level=2)
    assert (last.searched, last.replicate) == (("cut",), 0)
    assert list(last.fixed) == ["ecal_cell_size", "threshold"]  # in problem order
    assert last.candidate({"cut": 0.2}) == {"ecal_cell_size": 5.0, "threshold": 0.5, "cut": 0.2}
    # its own parameters may be fixed as well, as with --fix
    assert StudyScope.of(spec, {"ecal_cell_size": 5.0}, level=1).fixed == {"ecal_cell_size": 5.0}
    assert StudyScope.recorded(spec, {"scope": last.to_dict()}) == last


@pytest.mark.parametrize(
    ("fixed", "level", "message"),
    [
        (
            {},
            1,
            "a study of level 1 is started for a candidate of the levels above, so it needs "
            "the value of ecal_cell_size",
        ),
        ({"ecal_cell_size": 5.0}, 2, "it needs the value of threshold"),
        ({"ecal_cell_size": 5.0, "cut": 0.2}, 1, "cut is searched by a level below this one"),
        ({"threshold": 0.5}, 0, "threshold is searched by a level below this one"),
        ({"nope": 1.0}, 0, "nope is no parameter of the problem"),
        ({"ecal_cell_size": 99.0}, 0, "fixed: ecal_cell_size"),
        ({"ecal_cell_size": 5.0, "threshold": 0.5}, 1, "every parameter the study searches"),
        ({}, 3, "the problem's levels are 0 to 2"),
    ],
)
def test_what_a_scope_may_not_be(fixed: dict[str, Any], level: int, message: str) -> None:
    with pytest.raises(OptimizationError, match=message):
        StudyScope.of(nested(), fixed, level=level)


def test_a_recorded_scope_must_be_the_one_its_problem_makes() -> None:
    spec = nested()
    scope = StudyScope.of(spec, {"ecal_cell_size": 5.0}, level=1)
    for change in (
        lambda d: d.update(replicate=0),
        lambda d: d.update(searched=["threshold", "cut"]),
    ):
        recorded = scope.to_dict()
        change(recorded)
        with pytest.raises(OptimizationError, match="but its problem makes it"):
            StudyScope.recorded(spec, {"scope": recorded})
