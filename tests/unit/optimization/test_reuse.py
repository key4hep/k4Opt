"""A study whose parameters reach no stage simulates once: every trial reuses the first trial's
simulation, after a resume too, and so does another study given it to reuse."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.errors import K4OptError
from k4opt.optimization import OptimizationError, load_study, optimize
from k4opt.optimization._show import render_trial
from tests.support import simulation, stack
from tests.support.evaluators import candidate_check, cell_count
from tests.unit.optimization.conftest import GUN, summary


@pytest.fixture
def analysis_problem(write_problem: Callable[..., Path]) -> Path:
    """A problem whose one parameter sets a setting of an evaluator alone: its sample's
    simulation is the same for every candidate."""
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
        parameter={"type": "continuous", "bounds": [1.0, 2.0], "targets": {}},
        objectives=[
            ("test.cell_count", "simulation", "ecal_barrel.mean_hit_cells", "minimize"),
            ("test.candidate_check", "analysis", "seen", "minimize"),
        ],
    )
    document = yaml.safe_load(path.read_text())
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


def _reused(study: Path, number: int) -> dict[str, Any] | None:
    stage = summary(study, number)["samples"]["gamma"]["stages"]["simulation"]
    found: dict[str, Any] | None = stage.get("reused")
    return found


def test_the_trials_of_a_study_reuse_its_one_simulation_also_after_a_resume(
    tmp_path: Path, analysis_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    first = _optimize(analysis_problem, output, 3, seed=1)
    assert [t.status for t in first.trials] == ["valid"] * 3
    assert len(fake.calls) == 1
    made = summary(output, 0)["id"]
    assert _reused(output, 0) is None
    assert [_reused(output, n)["record"] for n in (1, 2)] == [made, made]  # type: ignore[index]
    # a resumed session reuses what the study's trials made
    resumed = _optimize(analysis_problem, output, 2, resume=True)
    assert len(fake.calls) == 1
    assert [t.status for t in resumed.trials] == ["valid"] * 5
    assert _reused(output, 4)["record"] == made  # type: ignore[index]
    assert [t.status for t in load_study(output).trials] == ["valid"] * 5


def test_a_stage_that_reused_an_output_ran_nothing_and_has_no_time(
    tmp_path: Path, analysis_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    _optimize(analysis_problem, tmp_path / "s", 3, seed=1)
    results = load_study(tmp_path / "s")
    # its record says 0 s, which a median of the stage's times must not count
    assert summary(tmp_path / "s", 1)["samples"]["gamma"]["stages"]["simulation"]["reused"]
    assert ["gamma/simulation" in t.stage_seconds for t in results.trials] == [True, False, False]


def test_a_study_reuses_what_another_one_it_is_given_made(
    tmp_path: Path, analysis_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    _optimize(analysis_problem, tmp_path / "first", 1, seed=1)
    _optimize(analysis_problem, tmp_path / "second", 2, seed=2, reuse=[tmp_path / "first"])
    assert len(fake.calls) == 1
    record = json.loads((tmp_path / "first" / "trial-000000" / "summary.json").read_text())
    assert _reused(tmp_path / "second", 0)["record"] == record["id"]  # type: ignore[index]


@pytest.fixture
def private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A file the fake ddsim of a study reads, which the problem does not name."""
    path = tmp_path / "private" / "calibration.json"
    path.parent.mkdir()
    path.write_text("{}\n")
    stack.install(monkeypatch, **{"*": {"read": [str(path)]}})
    return path


def test_a_trial_says_what_its_stages_used_and_a_published_study_does_not(
    tmp_path: Path, analysis_problem: Path, private: Path
) -> None:
    output = tmp_path / "s"
    _optimize(analysis_problem, output, 1, seed=1)
    study = load_study(output)
    out = io.StringIO()
    render_trial(study, study.trials[0], out)
    (line,) = [text for text in out.getvalue().splitlines() if "gamma/simulation: " in text]
    assert "undeclared inputs" in line and str(private) in line
    public = json.dumps(study.to_dict())
    assert str(private) not in public and "undeclared" not in public


def test_a_study_refuses_an_output_of_another_whose_input_changed(
    tmp_path: Path, analysis_problem: Path, private: Path
) -> None:
    _optimize(analysis_problem, tmp_path / "first", 1, seed=1)
    private.write_text('{"scale": 2}\n')
    _optimize(analysis_problem, tmp_path / "second", 1, seed=2, reuse=[tmp_path / "first"])
    assert _reused(tmp_path / "second", 0) is None
    stage = summary(tmp_path / "second", 0)["samples"]["gamma"]["stages"]["simulation"]
    assert f"{private} changed" in stage["not_reused"]


def test_a_study_watches_what_its_stages_read_while_it_runs(
    tmp_path: Path, analysis_problem: Path, private: Path
) -> None:
    # the first trial made the simulation; before the next, the file it read changes
    def changed(trial: Any) -> None:
        private.write_text('{"scale": 2}\n')

    with pytest.raises(OptimizationError, match=str(private)):
        _optimize(analysis_problem, tmp_path / "s", 3, seed=1, progress=changed)


def test_a_study_whose_input_changed_is_not_resumed(
    tmp_path: Path, analysis_problem: Path, private: Path
) -> None:
    output = tmp_path / "s"
    _optimize(analysis_problem, output, 1, seed=1)
    resumed = _optimize(analysis_problem, output, 1, resume=True)  # the same inputs: it goes on
    assert len(resumed.trials) == 2
    private.write_text('{"scale": 2}\n')
    with pytest.raises(K4OptError, match=f"{private}, read by the study's trials, changed"):
        _optimize(analysis_problem, output, 1, resume=True)
    assert len(load_study(output).trials) == 2  # nothing added
