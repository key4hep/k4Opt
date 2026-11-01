"""The benchmark of ``--jobs`` (``tests/benchmarks/parallel.py``) counts the job slots a task held
as the local backend does."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from k4opt.problem._backends import ProcessBackend
from tests.benchmarks.parallel import held, main, valid

SIMULATION = {"argv": ["/bin/ddsim", "--numberOfEvents=10", "--numberOfThreads=4"]}


@pytest.mark.parametrize("jobs", [1, 2, 4, 8])
def test_a_simulation_holds_a_slot_per_thread_but_never_more_than_there_are(jobs: int) -> None:
    backend = SimpleNamespace(_jobs=jobs)  # all that cost() reads of a backend of ``jobs``
    slots = ProcessBackend.cost(backend, 4)  # type: ignore[arg-type]
    assert slots == min(4, jobs)
    assert held(SIMULATION | {"elapsed_seconds": 10.0}, jobs) == 10.0 * slots
    # alone on every slot for the whole run, it fills them and no more
    assert held(SIMULATION | {"elapsed_seconds": 10.0}, jobs) / (jobs * 10.0) <= 1.0


def test_other_tasks_hold_one_slot_and_a_task_that_did_not_run_none() -> None:
    assert held({"elapsed_seconds": 3.0}, 4) == 3.0
    assert held({"argv": ["/bin/ddsim"], "elapsed_seconds": 3.0}, 4) == 3.0
    assert held({"elapsed_seconds": None}, 4) == 0.0


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--jobs", "2", "4"], "the first of --jobs must be 1"),
        (["--jobs", "1", "0"], "--jobs must be distinct positive numbers"),
        (["--jobs", "1", "2", "2"], "--jobs must be distinct positive numbers"),
        (["--candidates", "20"], "--candidates must be 1 to 8 (the fake grid)"),
        (["--stack", "real", "--candidates", "0"], "--candidates must be 1 to 8 (the real grid)"),
        (["--events", "0"], "--events must be positive"),
        (["--stack", "real", "--threads", "0"], "--threads must be positive"),
        (["--threads", "2"], "--threads is for --stack real"),
        (["--burn", "-1"], "--burn must not be negative"),
    ],
)
def test_what_would_run_otherwise_than_asked_is_refused_before_anything_is_created(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as info:
        main([*argv, "--output", str(tmp_path / "bench")])
    assert info.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "bench").exists()


def test_speedup_is_over_one_job_measured(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--jobs", "2", "4", "--output", str(tmp_path / "bench")])
    assert info.value.code == 2  # before anything runs
    assert "the first of --jobs must be 1" in capsys.readouterr().err
    assert not (tmp_path / "bench").exists()


@pytest.mark.parametrize("status", ["failed", "invalid"])
def test_a_candidate_that_is_not_valid_stops_the_benchmark(status: str) -> None:
    summary = {
        "status": status,
        "message": "gamma/simulation failed: ddsim exited",
        "run_dir": "c4",
    }
    with pytest.raises(SystemExit, match=rf"candidate in c4 is {status}: gamma/simulation failed"):
        valid(summary, 2)
    assert valid({"status": "valid"}, 2) == {"status": "valid"}
