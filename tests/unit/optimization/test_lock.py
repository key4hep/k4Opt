"""One process per study: the lock, interruptions within a trial, and processes that are killed."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from optuna.trial import TrialState

from k4opt.optimization import OptimizationError, optimize
from k4opt.optimization._study import Lock
from tests.support.common import tree_digest
from tests.support.evaluators import trade_off
from tests.unit.optimization.conftest import evaluator_from, optuna_study

REPOSITORY = Path(__file__).resolve().parents[3]

BLOCKING = """
    import os
    import time
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            if os.environ.get("K4OPT_TEST_BLOCK"):  # the process that is to be stopped
                (request.workdir / "started").write_text("")
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                self.fail("nobody stopped it")
            x, *_ = request.candidate.values()
            return {("analysis", "a"): x, ("analysis", "b"): 1 / x}
"""

INTERRUPTING = """
    import os
    import signal
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}
        settings = {"how": "sigterm"}

        def evaluate(self, request):
            if request.settings["how"] == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)  # as a batch system ends a job
            raise KeyboardInterrupt  # as Ctrl-C does
"""


def test_a_second_process_is_refused_while_one_works_on_the_study(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    optimize(problem, [trade_off()], output, trials=1, seed=1)
    with Lock(output):
        before = tree_digest(output)
        message = rf"in use: .*\.lock names process {os.getpid()} on .*, which is still running"
        with pytest.raises(OptimizationError, match=message):
            optimize(problem, [trade_off()], output, trials=1, resume=True)
        assert tree_digest(output) == before
    assert not (output / ".lock").exists()


def _dead_pid() -> int:
    finished = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(finished.stdout)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, r"names process \d+ on .*, which no longer runs. Once nothing works on the study"),
        ("{not json", r"names an unknown process \(the file cannot be read\)"),
    ],
)
def test_a_lock_left_behind_is_never_taken_over(
    tmp_path: Path, write_problem: Callable[..., Path], content: str | None, message: str
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    optimize(problem, [trade_off()], output, trials=1, seed=1)
    if content is None:  # what a killed process leaves
        owner = {"host": socket.gethostname(), "pid": _dead_pid(), "started": "2026-10-31T09:00:00"}
        content = json.dumps(owner)
    (output / ".lock").write_text(content)
    with pytest.raises(OptimizationError, match=message):
        optimize(problem, [trade_off()], output, trials=1, resume=True)
    assert (output / ".lock").read_text() == content  # left for a person to remove


def _optimize_command(problem: Path, output: Path, evaluator: str, *extra: str) -> list[str]:
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


def test_of_processes_starting_one_study_at_once_exactly_one_runs_it(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    from tests.support.evaluators import TRADE_OFF

    problem, output = write_problem(), tmp_path / "s"
    command = _optimize_command(problem, output, TRADE_OFF, "--trials", "2", "--seed", "1")
    processes = [
        subprocess.Popen(command, cwd=REPOSITORY, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for _ in range(4)
    ]
    results = [(process.wait(timeout=120), process.communicate()) for process in processes]
    codes = sorted(code for code, _ in results)
    assert codes == [0, 1, 1, 1], results
    for code, (_, err) in results:
        if code:
            assert b"in use" in err
    assert len(optuna_study(output).trials) == 2


@pytest.mark.parametrize("how", ["sigterm", "ctrl-c"])
def test_an_interrupted_trial_is_closed_and_the_lock_removed(
    tmp_path: Path, write_problem: Callable[..., Path], how: str
) -> None:
    problem = write_problem(
        evaluators=[{"name": "test.trade_off", "version": "1", "settings": {"how": how}}]
    )
    interrupting = evaluator_from(tmp_path / "evaluators", f"interrupting_{how}", INTERRUPTING)
    handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        optimize(problem, [interrupting], tmp_path / "s", trials=3, seed=1)
    assert signal.getsignal(signal.SIGTERM) is handler  # put back
    assert not (tmp_path / "s" / ".lock").exists()
    (trial,) = optuna_study(tmp_path / "s").trials
    assert trial.state == TrialState.FAIL and "k4opt.error" not in trial.user_attrs
    assert (tmp_path / "s" / "trial-000000").is_dir()  # what it had made, kept as it is


def test_an_interruption_while_the_optimizer_is_told_keeps_the_record(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import optuna

    constraints = [{"stage": "analysis", "metric": "a", "max": 450, "unit": "mm"}]
    problem, output = write_problem(constraints=constraints), tmp_path / "s"
    tell, calls = optuna.Study.tell, []

    def interrupted_once(self: optuna.Study, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("state"))
        if len(calls) == 1:  # the constraints are set, and then Ctrl-C comes
            raise KeyboardInterrupt
        return tell(self, *args, **kwargs)

    monkeypatch.setattr(optuna.Study, "tell", interrupted_once)
    with pytest.raises(KeyboardInterrupt):
        optimize(problem, [trade_off()], output, trials=2, seed=1)
    (trial,) = optuna_study(output).trials
    assert trial.state == TrialState.COMPLETE and set(trial.constraints) == {"analysis/a:max"}


def _wait_for(path: Path, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.05)


@dataclass
class Blocked:
    """A study whose first trial blocks in another process, and its evaluator here."""

    process: subprocess.Popen[bytes]
    evaluator: Any


@pytest.fixture
def blocked(tmp_path: Path, write_problem: Callable[..., Path]) -> Callable[[], Blocked]:
    """Start a study in another process, and return once its first trial blocks."""
    problem, output = write_problem(), tmp_path / "s"
    evaluator = evaluator_from(tmp_path / "evaluators", "blocking", BLOCKING)
    reference = f"{tmp_path / 'evaluators' / 'blocking.py'}:Probe"

    def start() -> Blocked:
        command = _optimize_command(problem, output, reference, "--trials", "2", "--seed", "1")
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            env=os.environ | {"K4OPT_TEST_BLOCK": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _wait_for(output / "trial-000000" / "evaluators" / "test.trade_off" / "started")
        return Blocked(process, evaluator)

    return start


def test_a_killed_study_is_refused_until_its_lock_is_removed_and_then_recovered(
    tmp_path: Path, write_problem: Callable[..., Path], blocked: Callable[[], Blocked]
) -> None:
    study = blocked()
    study.process.kill()  # SIGKILL: nothing of the process runs any more
    study.process.communicate(timeout=60)
    problem, output = tmp_path / "problem.yaml", tmp_path / "s"
    with pytest.raises(OptimizationError, match="which no longer runs"):
        optimize(problem, [study.evaluator], output, trials=1, resume=True)
    left = tree_digest(output / "trial-000000")
    (output / ".lock").unlink()  # the person confirmed that nothing runs
    report = optimize(problem, [study.evaluator], output, trials=1, resume=True)
    assert [(t.status, t.recovered) for t in report.trials] == [
        ("interrupted", True),
        ("valid", False),
    ]
    assert tree_digest(output / "trial-000000") == left  # never rerun or reused


def test_a_terminated_study_removes_its_lock_and_resumes_at_once(
    tmp_path: Path, blocked: Callable[[], Blocked]
) -> None:
    study = blocked()
    study.process.terminate()  # SIGTERM
    _, err = study.process.communicate(timeout=60)
    assert study.process.returncode == 1, err
    assert b"interrupted; continue the study with --resume" in err
    output = tmp_path / "s"
    assert not (output / ".lock").exists()
    problem = tmp_path / "problem.yaml"
    report = optimize(problem, [study.evaluator], output, trials=1, resume=True)
    assert [(t.status, t.recovered) for t in report.trials] == [
        ("interrupted", False),  # closed by the process itself, not by the resume
        ("valid", False),
    ]
