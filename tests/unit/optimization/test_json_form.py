"""The JSON form of a study: no location leaves the machine with it, and
``StudyResults.from_dict`` reads it back, checked, as the study it is."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from k4opt.optimization import OptimizationError, StudyResults, load_study, optimize
from k4opt.optimization._identity import digest
from k4opt.optimization._study import PORTABLE, portable
from tests.support.evaluators import trade_off
from tests.unit.optimization.conftest import left_running


def run(problem: Path, output: Path, trials: int) -> None:
    optimize(problem, [trade_off()], output, trials=trials, seed=1)


def strings(value: object) -> list[str]:
    """Every string a JSON value holds, keys included."""
    if isinstance(value, dict):
        return [s for key, item in value.items() for s in [key, *strings(item)]]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    return [value] if isinstance(value, str) else []


def test_the_json_form_names_no_location_and_keeps_the_identity(
    tmp_path: Path, mixed_problem: Path
) -> None:
    run(mixed_problem, tmp_path / "s", 6)
    study = load_study(tmp_path / "s")
    data = json.loads(study.to_json())
    held = strings(data)
    assert not [s for s in held if str(tmp_path) in s]
    # what starts with / is an XPath of the geometry: the stack is named by its build, not its path
    absolute = {s for s in held if s.startswith("/")}
    assert absolute == {target["locator"] for target in data["study"]["geometry"]["targets"]}
    assert "environment" not in data["study"] and "id" in data["study"]["software"]
    document = json.loads((tmp_path / "s" / "study.json").read_text())
    assert any(str(tmp_path) in s for s in strings(document))  # study.json itself has them
    assert data["study"] == portable(document)
    assert digest(data["study"]) == document["identity"]  # locations are no part of it
    assert set(data["study"]) == set(PORTABLE)
    assert set(data["study"]["software"]) == {"id", "build", "system"}  # variables can be private
    assert "storage" not in data["study"]["optimizer"]
    assert all("path" not in entry["source"] for entry in data["study"]["implementations"].values())
    assert all(set(trial["record"] or {}) <= {"id", "sha256"} for trial in data["trials"])


def test_a_study_reads_back_from_its_json_form_as_it_is(
    tmp_path: Path, mixed_problem: Path
) -> None:
    run(mixed_problem, tmp_path / "s", 8)
    study = load_study(tmp_path / "s")
    assert {t.outcome for t in study.trials} >= {"feasible", "infeasible", "invalid", "failed"}
    back = StudyResults.from_dict(json.loads(study.to_json()))
    assert (back.name, back.id, back.created) == (study.name, study.id, study.created)
    assert (back.problem, back.trials, back.pareto) == (study.problem, study.trials, study.pareto)
    assert back.outcomes == study.outcomes
    assert (back.directory, back.warnings, back.study_path) == (None, (), None)
    assert back.record_path(back.trials[0]) is None
    assert back.to_json() == study.to_json()  # the same form again


def test_the_programs_of_the_stack_leave_by_content_and_read_back(
    tmp_path: Path, mixed_problem: Path
) -> None:
    run(mixed_problem, tmp_path / "s", 2)
    document = json.loads((tmp_path / "s" / "study.json").read_text())
    python = {"path": "/opt/stack/venv/bin/python3", "bytes": 20, "sha256": "b" * 64}
    document["programs"] = {
        "ddsim": {
            "path": "/opt/stack/bin/ddsim",
            "bytes": 10,
            "sha256": "a" * 64,
            "interpreter": python,
        },
        "k4run": None,  # not found when the session opened
    }
    kept = portable(document)["programs"]
    # where the program and its interpreter are is left out, what they are is kept
    assert kept == {
        "ddsim": {
            "bytes": 10,
            "sha256": "a" * 64,
            "interpreter": {"bytes": 20, "sha256": "b" * 64},
        },
        "k4run": None,
    }
    data = load_study(tmp_path / "s").to_dict()
    data["study"]["programs"] = kept
    back = StudyResults.from_dict(copy.deepcopy(data))
    assert back.to_dict() == data
    data["study"]["programs"]["ddsim"]["path"] = "/opt/stack/bin/ddsim"
    with pytest.raises(OptimizationError, match=r"holds more of study\.json"):
        StudyResults.from_dict(data)


def test_an_active_study_reads_back_with_what_its_running_trials_hold(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 2)
    left_running(output, problem)  # with the candidate proposed
    study = load_study(output)
    data = json.loads(study.to_json())
    data["trials"][-1]["candidate"] = {}  # nothing proposed yet: a session killed in between
    back = StudyResults.from_dict(data)
    assert [t.status for t in back.trials] == ["valid", "valid", "running"]
    assert back.trials[-1].candidate == {}
    assert back.outcomes["running"] == 1


def _valid(data: dict[str, Any]) -> dict[str, Any]:
    return next(t for t in data["trials"] if t["status"] == "valid")


def _failed(data: dict[str, Any]) -> dict[str, Any]:
    return next(t for t in data["trials"] if t["status"] == "failed")


REFUSED: dict[str, tuple[Callable[[dict[str, Any]], object], str]] = {
    "another format": (lambda d: d.update(format="k4opt.study-results/0"), "its format"),
    "a key more": (lambda d: d.update(directory="/somewhere"), "lacks"),
    "an edited definition": (
        lambda d: d["study"]["problem"]["definition"]["parameters"]["radius"].update(
            choices=[1, 2]
        ),
        "identity digest",
    ),
    "a location in study.json": (
        lambda d: d["study"]["problem"].update(path="/home/me/problem.yaml"),
        "holds more of study.json",
    ),
    "an unknown status": (lambda d: _valid(d).update(status="done"), "not a status"),
    "trials out of order": (lambda d: d["trials"].reverse(), "after trial"),
    "a missing trial key": (lambda d: _valid(d).pop("outcome"), "lacks"),
    "another outcome": (lambda d: _valid(d).update(outcome="infeasible"), "its status gives"),
    "a valid trial without a record": (lambda d: _valid(d).update(record=None), "without a record"),
    "a record of another form": (
        lambda d: _valid(d)["record"].update(sha256="abc"),
        "not a SHA-256",
    ),
    "a partial candidate with a record": (
        lambda d: _valid(d).update(candidate={}),
        "not the parameters",
    ),
    "another objective": (
        lambda d: _valid(d)["assessment"]["objectives"][0].update(direction="maximize"),
        "is not the objective",
    ),
    "an objective value of a failed trial": (
        lambda d: _failed(d)["assessment"].update(objectives=_valid(d)["assessment"]["objectives"]),
        "are given, but the trial is failed",
    ),
    "a limit less": (lambda d: _valid(d)["assessment"]["limits"].clear(), "one of each"),
    "another limit": (
        lambda d: _valid(d)["assessment"]["limits"][0].update(limit=1),
        "has limit 1, but the limit test.trade_off/analysis/a:max judges limit 400",
    ),
    "a verdict its limits do not give": (
        lambda d: _valid(d)["assessment"].update(feasible=not _valid(d)["assessment"]["feasible"]),
        "its limits give",
    ),
    "a text where a number is": (
        lambda d: _valid(d)["assessment"]["objectives"][0].update(value="1"),
        "not a number",
    ),
    "another front": (lambda d: d.update(pareto=[]), "the front of its trials"),
    "a flag off the front": (
        lambda d: _valid(d).update(pareto=not _valid(d)["pareto"]),
        "other than its front",
    ),
    "other counts": (lambda d: d["outcomes"].update(feasible=99), "outcomes"),
    "a negative time": (lambda d: _valid(d).update(elapsed_seconds=-1), "not a time"),
    "a value outside the bounds": (
        lambda d: _valid(d)["candidate"].update(radius=99999),
        "is not one of",
    ),
    "a text for a number": (
        lambda d: _valid(d)["candidate"].update(radius="banana"),
        "expected a number, got 'banana'",
    ),
    "a boolean for a number": (
        lambda d: _valid(d)["candidate"].update(radius=True),
        "expected a number, got True",
    ),
    "a verdict its value does not give": (
        lambda d: _limits(_valid(d)).update(satisfied=not _limits(_valid(d))["satisfied"]),
        "judges satisfied",
    ),
    "a margin its value does not give": (
        lambda d: _limits(_valid(d)).update(margin=-_limits(_valid(d))["margin"] - 1),
        "judges margin",
    ),
    "a value beyond the limit called satisfied": (
        lambda d: _limits(_valid(d)).update(value=1500.0),
        "judges",
    ),
    "a judged value of a failed trial": (
        lambda d: _limits(_failed(d)).update(value=1.0, satisfied=True),
        "judges value None",
    ),
}


def _limits(trial: dict[str, Any]) -> dict[str, Any]:
    limit: dict[str, Any] = trial["assessment"]["limits"][0]
    return limit


@pytest.mark.parametrize("case", list(REFUSED))
def test_a_json_form_that_does_not_hold_up_is_refused(
    tmp_path: Path, mixed_problem: Path, case: str
) -> None:
    run(mixed_problem, tmp_path / "s", 8)
    data = json.loads(load_study(tmp_path / "s").to_json())
    StudyResults.from_dict(copy.deepcopy(data))  # as it is, it holds up
    change, message = REFUSED[case]
    change(data)
    with pytest.raises(OptimizationError, match=message):
        StudyResults.from_dict(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_number_json_does_not_hold_is_refused(
    tmp_path: Path, mixed_problem: Path, value: float
) -> None:
    run(mixed_problem, tmp_path / "s", 4)
    data = json.loads(load_study(tmp_path / "s").to_json())
    _valid(data)["assessment"]["objectives"][0]["value"] = value
    with pytest.raises(OptimizationError, match="finite"):
        StudyResults.from_dict(data)
