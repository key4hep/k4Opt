"""Studies whose trials run at once (``jobs > 1``): each trial's tasks in worker processes, the
optimizer asked only when a job is idle and told each trial as it ends, and every trial settled
however the study stops.

``test.trade_off`` here is ``GATED``: its evaluations mark their start in a directory and, as
the environment says, wait there until enough trials started or for a file that never comes, or
raise; markers, never sleeps, say what runs at once.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from optuna.exceptions import StorageInternalError
from optuna.trial import TrialState

from k4opt.optimization import OptimizationError, load_study, optimize
from k4opt.problem import Evaluations
from tests.support import stack
from tests.support.stack import alive, eventually
from tests.unit.optimization.conftest import evaluator_from, optuna_study

REPOSITORY = Path(__file__).resolve().parents[3]

GATED = """
    import json
    import os
    import time
    from pathlib import Path
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            told = json.loads(os.environ["K4OPT_TEST_GATE"])
            run = request.workdir.parents[1].name  # <trial>/evaluators/test.trade_off
            marks = Path(told["marks"])
            (marks / f"{run}.started").write_text(str(os.getpid()))
            deadline = time.monotonic() + 60
            waiting = told.get("await", {}).get(run)
            while len(list(marks.glob("*.started"))) < told.get("gate", 0) or (
                waiting is not None and not Path(waiting).exists()
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{run} still waits")
                time.sleep(0.01)
            if run in told.get("raise", []):
                raise ValueError(f"{run} is broken")
            x, *_ = request.candidate.values()
            return {("analysis", "a"): x, ("analysis", "b"): 1 / x}
"""


@pytest.fixture
def marks(tmp_path: Path) -> Path:
    path = tmp_path / "marks"
    path.mkdir()
    return path


@pytest.fixture
def gated(tmp_path: Path) -> Any:
    return evaluator_from(tmp_path / "evaluators", "gated", GATED)


def gate(monkeypatch: pytest.MonkeyPatch, marks: Path, **told: Any) -> dict[str, str]:
    """Tell ``GATED`` what to do, here and in every worker started from now on."""
    environment = {"K4OPT_TEST_GATE": json.dumps({"marks": str(marks), **told})}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return environment


def held(marks: Path, *runs: str) -> dict[str, str]:
    """``await`` for ``runs``: each waits for a file that never comes, until it is stopped."""
    return {run: str(marks / "never") for run in runs}


def started(marks: Path) -> dict[str, int]:
    """The pid of the worker each evaluation started in, by trial directory."""
    return {
        path.name.removesuffix(".started"): int(path.read_text())
        for path in marks.glob("*.started")
    }


def states(output: Path) -> dict[int, TrialState]:
    return {trial.number: trial.state for trial in optuna_study(output).trials}


def stopped(output: Path, marks: Path) -> None:
    """Nothing of the study runs any more: no lock, no worker, no evaluation."""
    assert not (output / ".lock").exists()
    assert stack.workers() == []
    assert all(eventually(lambda pid=pid: not alive(pid)) for pid in started(marks).values())


def test_trials_run_at_once_and_the_optimizer_is_asked_only_for_an_idle_job(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import optuna

    ask, tell = optuna.Study.ask, optuna.Study.tell
    asked: list[int] = []
    told: list[int] = []
    in_flight: list[int] = []

    def counted_ask(self: optuna.Study, *args: Any, **kwargs: Any) -> Any:
        trial = ask(self, *args, **kwargs)
        asked.append(trial.number)
        in_flight.append(len(asked) - len(told))
        return trial

    def counted_tell(self: optuna.Study, trial: Any, *args: Any, **kwargs: Any) -> Any:
        told.append(trial.number)
        return tell(self, trial, *args, **kwargs)

    monkeypatch.setattr(optuna.Study, "ask", counted_ask)
    monkeypatch.setattr(optuna.Study, "tell", counted_tell)
    gate(monkeypatch, marks, gate=3)  # the first three trials wait for each other
    output = tmp_path / "s"
    report = optimize(write_problem(), [gated], output, trials=7, seed=1, jobs=3)
    assert [trial.status for trial in report.trials] == ["valid"] * 7
    assert max(in_flight) == 3
    assert sorted(told) == list(range(7))
    assert len(set(started(marks).values())) == 3  # one worker per job, reused
    results = load_study(output)  # every record verifies as the study's
    assert [trial.number for trial in results.trials] == list(range(7))
    for number in range(7):
        record = json.loads((output / f"trial-{number:06d}" / "summary.json").read_text())
        assert record["environment"]["observed"]["execution"] == {"backend": "local", "jobs": 3}
    stopped(output, marks)


def test_a_trial_that_stops_the_study_closes_the_trials_still_running_as_interrupted(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = held(marks, "trial-000000", "trial-000002")
    gate(monkeypatch, marks, gate=3, **{"await": waiting, "raise": ["trial-000001"]})
    output = tmp_path / "s"
    with pytest.raises(OptimizationError, match=r"trial 1 stops the study \(trial-000001 is FAIL"):
        optimize(write_problem(), [gated], output, trials=5, seed=1, jobs=3)
    trials = {trial.number: trial for trial in optuna_study(output).trials}
    assert {number: trial.state for number, trial in trials.items()} == {
        0: TrialState.FAIL,
        1: TrialState.FAIL,
        2: TrialState.FAIL,
    }
    assert "trial-000001 is broken" in trials[1].user_attrs["k4opt.error"]
    for number in (0, 2):  # interrupted: no error of their own, their directories kept
        assert "k4opt.error" not in trials[number].user_attrs
        assert (output / f"trial-{number:06d}").is_dir()
        assert not (output / f"trial-{number:06d}" / "summary.json").exists()
    stopped(output, marks)


def _failing_once(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, error: Any) -> None:
    original = getattr(module, name)
    calls: list[int] = []

    def failing(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, failing)


def _altered_record(monkeypatch: pytest.MonkeyPatch) -> None:
    following = Evaluations.__next__

    def altered(self: Evaluations) -> Any:
        request, outcome = following(self)
        data = json.loads(outcome.summary_path.read_text())
        data["metrics"][0]["value"] = 1.0  # not what its evaluator measured
        outcome.summary_path.write_text(json.dumps(data))
        return request, outcome

    monkeypatch.setattr(Evaluations, "__next__", altered)


def _settlement_failure(monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    import optuna

    from k4opt.optimization._optuna import run

    if how == "check_record":
        _failing_once(monkeypatch, run, how, OptimizationError("trial 0: not the study's"))
    elif how == "check_candidate_files":
        _failing_once(monkeypatch, run, how, OptimizationError("trial 0: other files"))
    elif how == "from_summary":
        _altered_record(monkeypatch)
    elif how == "tell":
        _failing_once(monkeypatch, optuna.Study, how, StorageInternalError("the database is gone"))
    else:
        _failing_once(monkeypatch, optuna.Study, "tell", KeyboardInterrupt())


@pytest.mark.parametrize(
    ("how", "error", "state"),
    [
        ("check_record", "not the study's", TrialState.FAIL),
        ("check_candidate_files", "other files", TrialState.FAIL),
        ("from_summary", "metric test.trade_off/analysis/a", TrialState.FAIL),
        ("tell", "StorageInternalError: the database is gone", TrialState.FAIL),
        ("interrupt", None, TrialState.COMPLETE),  # told its record when it is closed
    ],
)
def test_a_trial_that_cannot_be_told_settles_with_the_trials_still_running(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    error: str | None,
    state: TrialState,
) -> None:
    gate(monkeypatch, marks, gate=3, **{"await": held(marks, "trial-000001", "trial-000002")})
    _settlement_failure(monkeypatch, how)
    output = tmp_path / "s"
    expected = KeyboardInterrupt if error is None else OptimizationError
    with pytest.raises(expected) as info:
        optimize(write_problem(), [gated], output, trials=5, seed=1, jobs=3)
    if error is not None:
        assert str(info.value).startswith("trial 0 stops the study (trial-000000 is FAIL): ")
        assert error in str(info.value)
    assert states(output) == {0: state, 1: TrialState.FAIL, 2: TrialState.FAIL}
    stopped(output, marks)


def test_a_tell_that_commits_and_then_raises_is_not_told_again(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import optuna

    tell, calls = optuna.Study.tell, []

    def committed_then_lost(self: optuna.Study, trial: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(trial.number)
        frozen = tell(self, trial, *args, **kwargs)
        if len(calls) == 1:
            raise StorageInternalError("the answer was lost")
        return frozen

    monkeypatch.setattr(optuna.Study, "tell", committed_then_lost)
    gate(monkeypatch, marks, gate=3, **{"await": held(marks, "trial-000001", "trial-000002")})
    output = tmp_path / "s"
    with pytest.raises(OptimizationError) as info:
        optimize(write_problem(), [gated], output, trials=5, seed=1, jobs=3)
    assert str(info.value).startswith(
        "trial 0 stops the study (trial-000000 is COMPLETE): StorageInternalError: the answer"
    )
    assert calls.count(0) == 1
    assert states(output) == {0: TrialState.COMPLETE, 1: TrialState.FAIL, 2: TrialState.FAIL}
    stopped(output, marks)


def test_a_storage_that_cannot_be_read_leaves_the_trials_to_the_resume(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import optuna

    tell, get_trials, lost = optuna.Study.tell, optuna.Study.get_trials, []

    def gone(self: optuna.Study, *args: Any, **kwargs: Any) -> Any:
        lost.append(1)
        raise StorageInternalError("the database is gone")

    def reading(self: optuna.Study, *args: Any, **kwargs: Any) -> Any:
        if lost:
            raise StorageInternalError("the database is gone")
        return get_trials(self, *args, **kwargs)

    monkeypatch.setattr(optuna.Study, "tell", gone)
    monkeypatch.setattr(optuna.Study, "get_trials", reading)
    gate(monkeypatch, marks, gate=3, **{"await": held(marks, "trial-000001", "trial-000002")})
    output, problem = tmp_path / "s", write_problem()
    with pytest.raises(OptimizationError) as info:
        optimize(problem, [gated], output, trials=5, seed=1, jobs=3)
    assert str(info.value).startswith(
        "trial 0 stops the study (the state of trial-000000 could not be read; k4opt optimize "
        "--resume settles it): StorageInternalError"
    )
    monkeypatch.setattr(optuna.Study, "tell", tell)
    monkeypatch.setattr(optuna.Study, "get_trials", get_trials)
    assert states(output) == dict.fromkeys(range(3), TrialState.RUNNING)
    stopped(output, marks)
    gate(monkeypatch, marks)
    report = optimize(problem, [gated], output, trials=1, resume=True, jobs=2)
    assert [(t.number, t.status, t.recovered) for t in report.trials] == [
        (0, "valid", True),  # its record was published before the storage went away
        (1, "interrupted", True),
        (2, "interrupted", True),
        (3, "valid", False),
    ]


def _optimize(problem: Path, output: Path, evaluator: str, *extra: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "k4opt",
        "optimize",
        str(problem),
        "--evaluator",
        evaluator,
        "--output",
        str(output),
        *extra,
    ]


@pytest.mark.parametrize("how", ["sigterm", "sigkill"])
def test_an_interrupted_parallel_study_stops_everything_and_resumes_with_other_jobs(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    waiting = held(marks, "trial-000000", "trial-000001")
    environment = gate(monkeypatch, marks, gate=2, **{"await": waiting})
    reference = f"{tmp_path / 'evaluators' / 'gated.py'}:Probe"
    command = _optimize(problem, output, reference, "--trials", "4", "--seed", "1", "-j", "2")
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        env=os.environ | environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert eventually(lambda: len(started(marks)) == 2, timeout=60)
        process.send_signal(signal.SIGTERM if how == "sigterm" else signal.SIGKILL)
        _, err = process.communicate(timeout=60)
    finally:
        process.kill()
    workers = started(marks).values()
    assert all(eventually(lambda pid=pid: not alive(pid)) for pid in workers)
    if how == "sigterm":
        assert process.returncode == 1, err
        assert b"interrupted; continue the study with --resume" in err
        assert states(output) == {0: TrialState.FAIL, 1: TrialState.FAIL}
    else:
        assert states(output) == {0: TrialState.RUNNING, 1: TrialState.RUNNING}
        (output / ".lock").unlink()  # nothing works on the study any more
    gate(monkeypatch, marks)
    report = optimize(problem, [gated], output, trials=2, resume=True, jobs=3)
    recovered = how == "sigkill"
    assert [(t.status, t.recovered) for t in report.trials] == [
        ("interrupted", recovered),
        ("interrupted", recovered),
        ("valid", False),
        ("valid", False),
    ]
    assert len(load_study(output).trials) == 4
    stopped(output, marks)
