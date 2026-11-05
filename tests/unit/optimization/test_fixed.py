"""A study that does not search some parameters: each has one value, ``fixed``, for every trial
(``k4opt optimize --fix``), part of its scope. With the geometry fixed and an evaluator's setting
searched, it simulates once."""

from __future__ import annotations

import copy
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.optimization import OptimizationError, StudyResults, load_study, optimize
from tests.support import simulation
from tests.support.evaluators import candidate_check, cell_count
from tests.unit.optimization.conftest import GUN, RADIUS, summary


@pytest.fixture
def two_parameters(write_problem: Callable[..., Path]) -> Path:
    """The radius sets the geometry, ``x`` a setting of ``test.candidate_check``; the sample's
    simulation reads the geometry alone."""
    path = write_problem(
        evaluators=[
            {
                "name": "test.cell_count",
                "version": "1",
                "stage": "simulation",
                "collections": {"hits": "ECalBarrelCollection"},
            },
            {"name": "test.candidate_check", "version": "1"},
        ],
        objectives=[
            ("test.cell_count", "simulation", "ecal_barrel.mean_hit_cells", "minimize"),
            ("test.candidate_check", "analysis", "seen", "minimize"),
        ],
    )
    document = yaml.safe_load(path.read_text())
    # the radius sets the geometry alone; x the setting of the check
    document["parameters"][RADIUS]["targets"] = {"geometry": "InnerTracker_Barrel_radius_1"}
    document["parameters"]["x"] = {
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "targets": {"evaluators": {"test.candidate_check": "x"}},
    }
    document["samples"] = {
        "gamma": {
            "generation": GUN | {"particle": "gamma"},
            "simulation": {"executor": "ddsim", "events": 2, "seed": 1},
        }
    }
    path.write_text(yaml.safe_dump(document))
    return path


def _optimize(problem: Path, output: Path, trials: int, **kwargs: Any) -> Any:
    return optimize(problem, [cell_count(), candidate_check()], output, trials=trials, **kwargs)


def test_a_study_with_the_geometry_fixed_searches_the_rest_and_simulates_once(
    tmp_path: Path, two_parameters: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    study = _optimize(two_parameters, output, 3, seed=1, fixed={RADIUS: 300.0})
    assert [t.status for t in study.trials] == ["valid"] * 3
    assert all(t.candidate[RADIUS] == 300.0 for t in study.trials)
    assert len({t.candidate["x"] for t in study.trials}) == 3  # searched
    assert list(study.trials[0].candidate) == [RADIUS, "x"]  # in problem order
    assert len(fake.calls) == 1  # one geometry: one simulation, reused by the other trials
    assert summary(output, 2)["candidate"] == study.trials[2].candidate
    document = json.loads((output / "study.json").read_text())
    assert document["scope"] == {
        "level": 0,
        "searched": ["x"],
        "fixed": {RADIUS: 300.0},
        "replicate": 0,
        "parent": None,
    }
    assert list(document["optimizer"]["parameters"]) == ["x"]  # what the optimizer searches
    read = load_study(output)
    assert read.scope.fixed == {RADIUS: 300.0} and read.scope.searched == ("x",)
    assert [t.candidate for t in read.trials] == [t.candidate for t in study.trials]
    assert read.to_dict()["study"]["scope"] == document["scope"]  # it leaves with its study
    # a resume with the same values goes on; with others it is refused
    resumed = _optimize(two_parameters, output, 1, fixed={RADIUS: 300.0}, resume=True)
    assert len(resumed.trials) == 4 and len(fake.calls) == 1
    for other in ({RADIUS: 310.0}, None):  # another value, or none fixed
        with pytest.raises(OptimizationError, match=r"give the same --fix as when it was made"):
            _optimize(two_parameters, output, 1, fixed=other, resume=True)


@pytest.mark.parametrize(
    ("fixed", "message"),
    [
        ({"nope": 1.0}, "fixed: nope is no parameter of the problem; this study searches"),
        ({RADIUS: 9999.0}, "fixed: radius"),
        ({RADIUS: 300.0, "x": 1.5}, "every parameter the study searches is fixed"),
    ],
)
def test_fixed_values_are_checked_before_anything_is_created(
    tmp_path: Path, two_parameters: Path, fixed: dict[str, Any], message: str
) -> None:
    with pytest.raises(OptimizationError, match=message.replace("(", r"\(").replace(")", r"\)")):
        _optimize(two_parameters, tmp_path / "s", 1, seed=1, fixed=fixed)
    assert not (tmp_path / "s").exists()


def test_the_command_line_fixes_a_parameter(
    tmp_path: Path, two_parameters: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.evaluators import CANDIDATE_CHECK, CELL_COUNT

    simulation.install(monkeypatch)
    out, err = io.StringIO(), io.StringIO()
    argv = ["optimize", str(two_parameters), "--evaluator", CELL_COUNT, "--evaluator"]
    argv += [CANDIDATE_CHECK, "--output", str(tmp_path / "s"), "--trials", "2", "--seed", "1"]
    argv += ["--fix", f"{RADIUS}=320", "--workers", "inline"]
    assert main(argv, stdout=out, stderr=err) == 0, err.getvalue()
    assert {t.candidate[RADIUS] for t in load_study(tmp_path / "s").trials} == {320.0}


def test_a_fixed_parameter_needs_no_distribution_the_sampler_could_draw(
    tmp_path: Path, two_parameters: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # integers beyond 2**53 cannot be sampled exactly; one value of them can be fixed all the same
    document = yaml.safe_load(two_parameters.read_text())
    document["parameters"]["huge"] = {
        "type": "integer",
        "bounds": [0, 2**60],
        "targets": {"evaluators": {"test.candidate_check": "refuse"}},
    }
    two_parameters.write_text(yaml.safe_dump(document))
    simulation.install(monkeypatch)
    with pytest.raises(OptimizationError, match="beyond 2\\*\\*53"):
        _optimize(two_parameters, tmp_path / "searched", 1, seed=1, fixed={RADIUS: 300.0})
    study = _optimize(two_parameters, tmp_path / "s", 2, seed=1, fixed={RADIUS: 300.0, "huge": 7})
    assert [t.candidate["huge"] for t in study.trials] == [7, 7]
    assert study.scope.searched == ("x",)


def test_a_json_form_whose_trial_breaks_a_fixed_value_is_refused(
    tmp_path: Path, two_parameters: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    study = _optimize(two_parameters, tmp_path / "s", 2, seed=1, fixed={RADIUS: 300.0})
    data = study.to_dict()
    assert StudyResults.from_dict(data).scope.fixed == {RADIUS: 300.0}
    for change in (lambda c: c.update({RADIUS: 350.0}), lambda c: c.pop(RADIUS)):
        edited = copy.deepcopy(data)
        change(edited["trials"][0]["candidate"])
        with pytest.raises(OptimizationError, match=r"fixes it at 300\.0|not the parameters"):
            StudyResults.from_dict(edited)
