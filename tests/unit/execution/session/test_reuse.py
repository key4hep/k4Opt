"""A stage is not run again when a stage with its signature made its output already, in the session
or in a record it is given: its record then says where the output comes from. A source that
changed or went is not reused; a record that claims a reuse it could not have made is refused."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from k4opt.execution import EvaluationSession, RunRequest
from k4opt.record import SUMMARY_NAME, Assessment, RecordError
from tests.support import simulation
from tests.support.evaluators import cell_count
from tests.unit.execution.session.conftest import RADIUS

SIMULATION = "samples/gamma/stages/simulation"


def summary(run: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run / SUMMARY_NAME).read_text())
    return data


def simulated(record: dict[str, Any]) -> dict[str, Any]:
    stage: dict[str, Any] = record["samples"]["gamma"]["stages"]["simulation"]
    return stage


def test_a_candidate_whose_simulation_was_made_already_reuses_it(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        first = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        second = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
        Assessment.from_summary(session.spec, dict(second.summary))  # the record holds up
    assert len(fake.calls) == 1  # one simulation for both
    assert second.candidate.status == "valid"
    made, reused = simulated(summary(tmp_path / "a")), simulated(summary(tmp_path / "b"))
    events = made["outputs"]["events"]
    assert reused["reused"] == {
        "record": first.summary["id"],
        "task": "gamma/simulation",
        "run_dir": str(tmp_path / "a"),
        "path": events["path"],
        "bytes": events["bytes"],
        "sha256": events["sha256"],
    }
    assert reused["signature"] == made["signature"]
    assert reused["outputs"] == {} and reused["logs"] == {}
    assert reused["events_read"] == made["events_read"]
    assert "argv" not in reused  # it ran nothing: the record of a says how the file was made
    assert not (tmp_path / "b" / "samples" / "gamma" / "simulation").exists()
    # the evaluator read a's file, and measured what it measured for a
    assert second.summary["metrics"] == first.summary["metrics"]


def test_another_geometry_is_simulated_again(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        session.evaluate({RADIUS: 460.0}, tmp_path / "b")
    assert len(fake.calls) == 2
    assert "reused" not in simulated(summary(tmp_path / "b"))


@pytest.mark.parametrize("change", ["rewritten", "removed"])
def test_a_source_that_changed_or_went_is_simulated_again(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, change: str
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        events = tmp_path / "a" / simulated(summary(tmp_path / "a"))["outputs"]["events"]["path"]
        if change == "rewritten":
            events.write_text(events.read_text() + " ")
        else:
            events.unlink()
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert len(fake.calls) == 2
    assert outcome.candidate.status == "valid"
    assert "reused" not in simulated(summary(tmp_path / "b"))


def test_a_session_reuses_what_the_records_it_is_given_made(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        first = session.evaluate({RADIUS: 450.0}, tmp_path / "runs" / "a")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        assert session.reuse([tmp_path / "runs"]) == 1  # a directory of runs, as a study is
        session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert len(fake.calls) == 1
    assert simulated(summary(tmp_path / "b"))["reused"]["record"] == first.summary["id"]


def test_a_session_does_not_reuse_a_record_whose_signature_does_not_hold(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    record = summary(tmp_path / "a")
    simulated(record)["signature"]["document"]["definition"]["seed"] = 99  # its hash no more
    (tmp_path / "a" / SUMMARY_NAME).write_text(json.dumps(record))
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        assert session.reuse([tmp_path / "a"]) == 0
        session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert len(fake.calls) == 2


def test_two_candidates_at_once_with_one_simulation_simulate_it_once(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support import stack

    marks = tmp_path / "marks"
    marks.mkdir()
    stack.install(monkeypatch, marks, **{"*": {"sleep": 1}})
    with (
        EvaluationSession.open(problem_file, [cell_count()], workers=2) as session,
        session.evaluate_each(
            [
                RunRequest({RADIUS: 450.0}, tmp_path / "a"),
                RunRequest({RADIUS: 450.0}, tmp_path / "b"),
            ]
        ) as evaluations,
    ):
        outcomes = {Path(request.output).name: outcome for request, outcome in evaluations}
    assert sorted(stack.started(marks)) == ["a.gamma"]  # b waited for a's, then reused it
    assert all(outcome.candidate.status == "valid" for outcome in outcomes.values())  # type: ignore[union-attr]
    assert simulated(summary(tmp_path / "b"))["reused"]["task"] == "gamma/simulation"


def test_a_stage_that_waited_for_one_that_failed_runs_itself(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support import stack

    marks = tmp_path / "marks"
    marks.mkdir()
    stack.install(monkeypatch, marks, **{"a.gamma": {"sleep": 1, "exit": 3}})
    with (
        EvaluationSession.open(problem_file, [cell_count()], workers=2) as session,
        session.evaluate_each(
            [
                RunRequest({RADIUS: 450.0}, tmp_path / "a"),
                RunRequest({RADIUS: 450.0}, tmp_path / "b"),
            ]
        ) as evaluations,
    ):
        outcomes = {Path(request.output).name: outcome for request, outcome in evaluations}
    assert sorted(stack.started(marks)) == ["a.gamma", "b.gamma"]
    assert outcomes["a"].candidate.status == "failed"  # type: ignore[union-attr]
    assert outcomes["b"].candidate.status == "valid"  # type: ignore[union-attr]
    assert "reused" not in simulated(summary(tmp_path / "b"))


def _reused_record(tmp_path: Path, problem_file: Path) -> tuple[EvaluationSession, dict[str, Any]]:
    session = EvaluationSession.open(problem_file, [cell_count()])
    session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    return session, dict(session.evaluate({RADIUS: 450.0}, tmp_path / "b").summary)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda s: s["reused"].update(task="gamma/reconstruction"), "which is no simulation"),
        (lambda s: s["reused"].update(sha256=None), "reused an output without its SHA-256"),
        (lambda s: s["reused"].update(bytes=-1), "bytes must be a size, got -1"),
        (lambda s: s.update(logs={"stdout": "x"}), "reused, but it names files of its own"),
    ],
)
def test_a_record_that_claims_a_reuse_it_could_not_have_made_is_refused(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, tamper: Any, message: str
) -> None:
    session, data = _reused_record(tmp_path, problem_file)
    with session:
        tamper(simulated(data))
        with pytest.raises(RecordError, match=re.escape(message)):
            Assessment.from_summary(session.spec, data)


def test_k4opt_evaluate_reuses_the_runs_it_is_given(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    from tests.unit.execution.session.conftest import run_candidate

    run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "b", reuse=[tmp_path / "a"])
    assert len(fake.calls) == 1
    assert simulated(summary(tmp_path / "b"))["reused"]["run_dir"] == str(tmp_path / "a")
