"""Studies whose trials run at once (``jobs > 1``): each trial's tasks in worker processes, the
strategy asked only when a job is idle and told each trial as it ends, and every trial settled
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

from k4opt.execution import Activity, Evaluations
from k4opt.optimization import OptimizationError, load_study, optimize
from tests.support import stack
from tests.support.stack import alive, eventually
from tests.unit.optimization.conftest import RADIUS, STRATEGY_KWARGS, Told, endings, evaluator_from

GRID = {"strategy": "grid", "grid": {RADIUS: [150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0]}}
"""A grid of seven points: as many trials as the tests here run at most."""


def strategy_kwargs(strategy: str) -> dict[str, Any]:
    return GRID if strategy == "grid" else STRATEGY_KWARGS[strategy]


REPOSITORY = Path(__file__).resolve().parents[3]

GATED = """
    import json
    import os
    import time
    from pathlib import Path
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        settings = {"x": float}  # the candidate's value, which it sets
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            told = json.loads(os.environ["PYTEST_K4OPT_GATE"])
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
            x = request.settings["x"]
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
    environment = {"PYTEST_K4OPT_GATE": json.dumps({"marks": str(marks), **told})}
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


def states(output: Path) -> dict[int, str | None]:
    """How each trial ended, by number, as the ledger holds it (``None``: still running)."""
    return dict(enumerate(endings(output)))


def stopped(output: Path, marks: Path) -> None:
    """Nothing of the study runs any more: no lock, no worker, no evaluation."""
    assert not (output / ".lock").exists()
    assert stack.workers() == []
    assert all(eventually(lambda pid=pid: not alive(pid)) for pid in started(marks).values())


def test_trials_run_at_once_and_the_strategy_is_asked_only_for_an_idle_job(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    told: Told,
) -> None:
    gate(monkeypatch, marks, gate=3)  # the first three trials wait for each other
    output = tmp_path / "s"
    seen: list[Activity] = []
    report = optimize(
        write_problem(),
        [gated],
        output,
        trials=7,
        workers=3,
        activity=seen.append,
        **strategy_kwargs(strategy),
    )
    assert [trial.status for trial in report.trials] == ["valid"] * 7
    # at most three proposed and not yet told; each told once, in the order the trials ended
    in_flight, under_way = [], 0
    for kind, *_ in told.events:
        under_way += 1 if kind == "proposed" else -1
        in_flight.append(under_way)
    assert max(in_flight) == 3 and in_flight[-1] == 0
    told_numbers = [number for kind, _, _, number in told.events if kind == "told"]
    assert {activity.capacity for activity in seen} == {3}
    assert max(activity.candidates for activity in seen) == 3
    assert all(activity.busy <= 3 for activity in seen) and seen[-1].candidates == 0
    assert sorted(told_numbers) == list(range(7))
    assert len(set(started(marks).values())) == 3  # one worker per job, reused
    results = load_study(output)  # every record verifies as the study's
    assert [trial.number for trial in results.trials] == list(range(7))
    for number in range(7):
        record = json.loads((output / f"trial-{number:06d}" / "summary.json").read_text())
        assert record["environment"]["observed"]["execution"] == {
            "backend": "process",
            "workers": 3,
        }
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
    match = r"trial 1 stops the study \(trial-000001 is an error"
    with pytest.raises(OptimizationError, match=match):
        optimize(write_problem(), [gated], output, trials=5, workers=3, **GRID)
    assert states(output) == {0: "interrupted", 1: "error", 2: "interrupted"}
    read = load_study(output)
    assert "trial-000001 is broken" in (read.trial(1).message or "")
    for number in (0, 2):  # interrupted: no error of their own, their directories kept
        assert read.trial(number).message is None
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


def _lost_write(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The ledger's writes of records fail (a disk that fills up, say); the real ``os.write``."""
    import os

    from k4opt.optimization import _ledger

    write = os.write

    def losing(descriptor: int, data: Any) -> int:
        if b'"event":"recorded"' in bytes(data):
            raise OSError(28, "No space left on device")
        return write(descriptor, data)

    monkeypatch.setattr(_ledger.os, "write", losing)
    return write


def _settlement_failure(monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    from k4opt.optimization import _results, _trials

    if how == "check_record":
        _failing_once(monkeypatch, _results, how, OptimizationError("trial 0: not the study's"))
    elif how == "check_candidate_files":
        _failing_once(monkeypatch, _results, how, OptimizationError("trial 0: other files"))
    elif how == "from_summary":
        _altered_record(monkeypatch)
    elif how == "ledger":
        _lost_write(monkeypatch)
    else:  # Ctrl-C as the strategy is told, once the ledger holds the record
        _failing_once(monkeypatch, _trials, "_tell", KeyboardInterrupt())


@pytest.mark.parametrize(
    ("how", "error", "became", "ended"),
    [
        ("check_record", "not the study's", "is an error", "error"),
        ("check_candidate_files", "other files", "is an error", "error"),
        ("from_summary", "metric test.trade_off/analysis/a", "is an error", "error"),
        # nothing is written after a write that failed: every trial is left to the resume
        ("ledger", "trials.jsonl: No space left on device", None, None),
        ("interrupt", None, None, "recorded"),  # its record kept before Ctrl-C came
    ],
)
def test_a_trial_that_cannot_be_kept_settles_with_the_trials_still_running(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    error: str | None,
    became: str | None,
    ended: str | None,
) -> None:
    gate(monkeypatch, marks, gate=3, **{"await": held(marks, "trial-000001", "trial-000002")})
    _settlement_failure(monkeypatch, how)
    output = tmp_path / "s"
    expected = KeyboardInterrupt if error is None else OptimizationError
    with pytest.raises(expected) as info:
        optimize(write_problem(), [gated], output, trials=5, workers=3, **GRID)
    if error is not None:
        head = f"trial-000000 {became}" if became else "the ledger could not be written; "
        assert str(info.value).startswith(f"trial 0 stops the study ({head}")
        assert error in str(info.value)
    others = "interrupted" if how != "ledger" else None
    assert states(output) == {0: ended, 1: others, 2: others}
    stopped(output, marks)


def test_a_strategy_that_fails_once_it_is_told_is_not_told_again(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
    told: Told,
) -> None:
    from k4opt.optimization import _grid

    tell, calls = _grid.Grid.tell, []

    def failing(self: Any, number: int, feedback: Any) -> None:
        calls.append(number)
        tell(self, number, feedback)
        if len(calls) == 1:
            raise RuntimeError("the strategy lost its answer")

    monkeypatch.setattr(_grid.Grid, "tell", failing)
    gate(monkeypatch, marks, gate=3, **{"await": held(marks, "trial-000001", "trial-000002")})
    output, problem = tmp_path / "s", write_problem()
    with pytest.raises(OptimizationError) as info:
        optimize(problem, [gated], output, trials=5, workers=3, **GRID)
    assert str(info.value).startswith(
        "trial 0 stops the study (trial-000000 is recorded): OptimizationError: trial 0: the "
        "strategy could not take how it ended: RuntimeError: the strategy lost its answer"
    )
    assert calls.count(0) == 1  # the others were told as they were interrupted
    assert states(output) == {0: "recorded", 1: "interrupted", 2: "interrupted"}
    stopped(output, marks)
    gate(monkeypatch, marks)
    report = optimize(problem, [gated], output, trials=1, resume=True, workers=2)
    # the resumed strategy was made from the ledger, the trial among its history: never told
    assert (
        calls.count(0) == 1
        and calls[-1] == 3
        and [t.status for t in report.trials]
        == [
            "valid",
            "interrupted",
            "interrupted",
            "valid",
        ]
    )
    (_, _, history) = told.opened[-1]
    assert [past.feedback.status for past in history] == ["valid", "interrupted", "interrupted"]


def test_a_ledger_that_cannot_be_written_leaves_the_trials_to_the_resume(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    marks: Path,
    gated: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from k4opt.optimization import _ledger

    gate(monkeypatch, marks, gate=3, **{"await": held(marks, "trial-000001", "trial-000002")})
    write = _lost_write(monkeypatch)
    output, problem = tmp_path / "s", write_problem()
    with pytest.raises(OptimizationError, match="the ledger could not be written"):
        optimize(problem, [gated], output, trials=5, workers=3, **GRID)
    monkeypatch.setattr(_ledger.os, "write", write)
    assert states(output) == dict.fromkeys(range(3))
    stopped(output, marks)
    gate(monkeypatch, marks)
    report = optimize(problem, [gated], output, trials=1, resume=True, workers=2)
    assert [(t.number, t.status, t.recovered) for t in report.trials] == [
        (0, "valid", True),  # its record was published before the ledger could not be written
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
    grid = ("--strategy", "grid", "--grid", f"{RADIUS}=150,200,250,300")
    command = _optimize(problem, output, reference, "--trials", "4", *grid, "--workers", "2")
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
        assert states(output) == {0: "interrupted", 1: "interrupted"}
    else:
        assert states(output) == {0: None, 1: None}
        (output / ".lock").unlink()  # nothing works on the study any more
    gate(monkeypatch, marks)
    report = optimize(problem, [gated], output, trials=2, resume=True, workers=3)
    recovered = how == "sigkill"
    assert [(t.status, t.recovered) for t in report.trials] == [
        ("interrupted", recovered),
        ("interrupted", recovered),
        ("valid", False),
        ("valid", False),
    ]
    assert len(load_study(output).trials) == 4
    stopped(output, marks)
