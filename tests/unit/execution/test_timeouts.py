"""Time limits of tasks, as the problem sets them, kept by real worker processes on a fake stack.

What a task does in a worker is told through the environment (``test.worker_probe``, the fake
``ddsim`` and ``podio`` of :mod:`tests.support.stack`); the limits are short, and every wait for
what a test cannot see directly is bounded (:func:`~tests.support.stack.eventually`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.evaluation import Status
from k4opt.execution import EvaluationSession, RunError, RunOutcome, _describe, _process
from k4opt.problem.plan import GEOMETRY
from tests.support import stack
from tests.support.evaluators import cell_count, worker_probe
from tests.support.stack import alive, eventually
from tests.unit.execution.test_parallel import (  # noqa: F401 - mini_k4geo is a fixture
    RADIUS,
    mini_k4geo,
    probe,
    verified,
    write_problem,
)

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="watches processes through /proc"
)


@pytest.fixture
def session(tmp_path: Path, doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Open a session of a problem whose ``gamma`` sample a ``test.worker_probe`` bound as ``p``
    measures, with one worker: ``p`` may take ``evaluator`` seconds, the simulation of ``gamma``
    ``simulation`` seconds, and every stage and evaluation ``default`` seconds unless it says
    otherwise; closed after the test."""
    stack.install(monkeypatch)
    monkeypatch.setattr(_process, "_GRACE", 1.0)  # SIGKILL soon after SIGTERM
    opened: list[EvaluationSession] = []

    def open_(
        *,
        evaluator: float | None = None,
        simulation: float | None = None,
        default: float | None = None,
        workers: int | str = 1,
    ) -> EvaluationSession:
        problem = limited(
            write_problem(tmp_path / "probe.yaml", doc, ("p",)), evaluator, simulation, default
        )
        implementations = [cell_count(), worker_probe()]
        opened.append(EvaluationSession.open(problem, implementations, workers=workers))  # type: ignore[arg-type]
        return opened[-1]

    yield open_
    for each in opened:
        each.close()
    assert stack.workers() == []  # no worker, nor any process of one, outlives its session


def limited(
    problem: Path, evaluator: float | None, simulation: float | None, default: float | None
) -> Path:
    """``problem`` with these time limits (none where ``None``)."""
    doc = yaml.safe_load(problem.read_text())
    if evaluator is not None:
        (entry,) = [e for e in doc["evaluators"] if e.get("as") == "p"]
        entry["timeout"] = evaluator
    if simulation is not None:
        doc["samples"]["gamma"]["simulation"]["timeout"] = simulation
    if default is not None:
        doc["timeout"] = default
    problem.write_text(yaml.safe_dump(doc))
    return problem


def evaluated(session: EvaluationSession, output: Path) -> RunOutcome:
    outcome = session.evaluate({RADIUS: 450.0}, output)
    assert isinstance(outcome, RunOutcome)
    return outcome


def test_an_evaluation_that_runs_too_long_fails_and_its_worker_is_replaced(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    marks = tmp_path / "marks"
    marks.mkdir()
    # each worker keeps the environment it started with: both runs are told at once
    told = {"a.p": {"mark": str(marks / "a"), "sleep": 600}, "b.p": {"mark": str(marks / "b")}}
    probe(monkeypatch, **told)
    opened = session(evaluator=1.0)
    outcome = evaluated(opened, tmp_path / "a")
    # an execution failure of the candidate, recorded: never a framework error
    assert outcome.candidate.status == Status.FAILED
    message = outcome.candidate.message or ""
    assert "gamma/p failed: gamma/p ran longer than the evaluator timeout (1 s)" in message
    summary = verified(outcome)  # the record holds up, read back without the session
    log = tmp_path / "a" / summary["samples"]["gamma"]["evaluations"]["p"]["log"]
    assert log.is_file()  # its log is kept
    stopped = int((marks / "a").read_text())
    assert eventually(lambda: not alive(stopped))  # its worker was stopped, and is gone
    # the next candidate's evaluation runs in a new worker, which takes the slot again
    again = evaluated(opened, tmp_path / "b")
    assert again.candidate.status == Status.VALID, again.candidate.message
    assert int((marks / "b").read_text()) != stopped


def test_an_evaluator_whose_creation_hangs_is_stopped_with_what_it_printed(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe(monkeypatch, init={"print": "loading the model", "sleep": 600})
    outcome = evaluated(session(evaluator=1.0), tmp_path / "a")
    assert outcome.candidate.status == Status.FAILED
    assert "ran longer than the evaluator timeout (1 s)" in (outcome.candidate.message or "")
    log = tmp_path / "a" / verified(outcome)["samples"]["gamma"]["evaluations"]["p"]["log"]
    assert "loading the model" in log.read_text()  # created where its evaluate() prints


def test_a_worker_that_ignores_sigterm_is_killed(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    marks = tmp_path / "marks"
    marks.mkdir()
    probe(monkeypatch, p={"mark": str(marks / "pid"), "ignore_term": True, "sleep": 600})
    outcome = evaluated(session(evaluator=1.0), tmp_path / "a")
    assert outcome.candidate.status == Status.FAILED
    assert eventually(lambda: not alive(int((marks / "pid").read_text())))


def test_what_a_task_started_is_stopped_with_it(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child"
    probe(monkeypatch, p={"child": str(child), "sleep": 600})
    outcome = evaluated(session(evaluator=1.0), tmp_path / "a")
    assert outcome.candidate.status == Status.FAILED
    assert eventually(lambda: not alive(int(child.read_text())))  # no orphan outlives it


def test_tasks_that_run_out_of_time_one_after_another_never_stop_the_session(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # unlike workers that die, a time limit reached is the candidate's: none of these counts
    # toward the deaths in a row that stop a session (workers + 1 = 2 here)
    probe(monkeypatch, p={"sleep": 600})
    opened = session(evaluator=0.5)
    for name in "abcd":
        outcome = evaluated(opened, tmp_path / name)
        assert outcome.candidate.status == Status.FAILED
        assert "the evaluator timeout (0.5 s)" in (outcome.candidate.message or "")


def test_only_the_task_that_ran_out_of_time_fails(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # two workers: the simulation of the other sample, and its evaluation, are not touched
    probe(monkeypatch, p={"sleep": 600})
    outcome = evaluated(session(evaluator=1.0, workers=2), tmp_path / "a")
    results = {r.id.label: r.status for r in outcome.candidate.results.values()}
    assert results["gamma/p"] == Status.FAILED
    assert results["pion/test.cell_count"] == results["gamma/test.cell_count"] == Status.VALID


def test_a_geometry_that_runs_too_long_is_a_recorded_failure(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # no option sets it: k4opt's own work on files, limited so that a stall holds no worker
    monkeypatch.setattr(_describe, "GEOMETRY_TIMEOUT", 1.0)
    probe(monkeypatch, materialize={"sleep": 600})
    outcome = evaluated(session(), tmp_path / "a")
    geometry = outcome.candidate.results[GEOMETRY]
    assert geometry.status == Status.FAILED
    assert "ran longer than the time limit of a geometry (1 s)" in (geometry.message or "")
    # nothing that reads it ran (a particle gun reads no geometry), and the record holds up
    # without changes or a fingerprint
    for task in outcome.candidate.plan.tasks:
        status = outcome.candidate.results[task.id].status
        assert status == (Status.BLOCKED if GEOMETRY in task.needs else status)
        assert status != Status.VALID
    summary = verified(outcome)
    assert summary["status"] == "failed" and outcome.geometry_fingerprint is None
    assert set(summary["geometry"]) == {
        "status",
        "message",
        "varies",
        "elapsed_seconds",
        "peak_memory_bytes",
    }


def test_a_simulation_that_runs_too_long_is_stopped_by_its_stage_with_its_logs(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = {"gamma": {"sleep": 600}}
    monkeypatch.setenv("FAKE_DDSIM", json.dumps(control))
    outcome = evaluated(session(simulation=1.0), tmp_path / "a")
    simulated = verified(outcome)["samples"]["gamma"]["stages"]["simulation"]
    assert "ddsim ran longer than 1 s and was killed" in simulated["message"]
    assert (simulated["timeout"], simulated["returncode"]) == (1.0, None)
    assert (tmp_path / "a" / "samples" / "gamma" / "simulation" / "ddsim.stderr.log").is_file()
    assert outcome.candidate.stage("simulation", "pion").status == "done"  # (not limited)


def test_a_stage_whose_own_work_hangs_after_its_program_is_stopped_by_its_worker(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ddsim ended in time, but reading its output never does: the worker's own deadline stops it
    monkeypatch.setattr(_process, "_BACKSTOP", 0.5)
    monkeypatch.setenv("FAKE_PODIO_SLEEP", "600")  # the output of every sample's simulation
    outcome = evaluated(session(default=1.0, workers=2), tmp_path / "a")
    assert outcome.candidate.status == Status.FAILED
    simulated = verified(outcome)["samples"]["gamma"]["stages"]["simulation"]
    assert "ran longer than the stage timeout (1 s) and was stopped" in simulated["message"]


def test_the_problem_s_timeout_limits_every_stage_and_evaluation_without_one_of_its_own(
    tmp_path: Path, session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe(monkeypatch, p={"sleep": 600})
    outcome = evaluated(session(default=2.0, simulation=60), tmp_path / "a")
    assert "gamma/p ran longer than the evaluator timeout (2 s)" in (
        outcome.candidate.message or ""
    )
    summary = verified(outcome)
    assert summary["problem"]["definition"]["timeout"] == 2.0  # the problem, as the record has it
    stages = summary["samples"]["gamma"]["stages"]
    assert stages["simulation"]["timeout"] == 60.0  # its own, not the problem's


@pytest.mark.parametrize("limit", ["evaluator", "default"])
def test_a_problem_that_limits_its_evaluations_is_refused_inline(session: Any, limit: str) -> None:
    # nothing stops an evaluation in the session's own process: no limit is claimed that is not
    # kept; a stage's limit alone stops the program it runs, inline too
    with pytest.raises(RunError, match=r"timeout: an evaluation .* cannot be stopped.*--workers"):
        session(workers="inline", **{limit: 5.0})
    session(workers="inline", simulation=5.0).close()
