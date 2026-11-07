"""A stage is not run again when a stage with its signature made its output already, in the session
or in a record it is given: its record then says where the output comes from. A source that
changed or went is not reused; a record that claims a reuse it could not have made is refused."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.execution import EvaluationSession, RunRequest
from k4opt.problem import load_problem
from k4opt.record import SUMMARY_NAME, Assessment, RecordError
from tests.support import simulation, stack
from tests.support.evaluators import cell_count
from tests.support.stack import alive, eventually
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


def test_a_session_does_not_reuse_a_record_whose_signature_is_not_its_stage(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    from k4opt.record.signature import digest

    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    record = summary(tmp_path / "a")
    signature = simulated(record)["signature"]
    signature["document"]["definition"]["seed"] = 99  # hashed again: it holds by itself
    signature["sha256"] = digest(signature["document"])
    (tmp_path / "a" / SUMMARY_NAME).write_text(json.dumps(record))
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        assert session.reuse([tmp_path / "a"]) == 0  # but its stage was made with another seed
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


def _unpublished(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """The candidate in the run directory ``name`` is not published (its record does not hold
    up when it ends)."""
    from k4opt.execution import ProvenanceError, _session

    publish = _session.publish

    def refusing(run: Any, *args: Any) -> Any:
        if run.workspace.root.name == name:
            raise ProvenanceError(f"{name} is not published")
        return publish(run, *args)

    monkeypatch.setattr(_session, "publish", refusing)


def test_what_a_candidate_that_was_not_published_made_is_not_reused(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.execution import ProvenanceError

    _unpublished(monkeypatch, "a")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        with pytest.raises(ProvenanceError, match="a is not published"):
            session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert len(fake.calls) == 2  # no record would say where a's simulation comes from
    assert "reused" not in simulated(summary(tmp_path / "b"))


def test_a_candidate_that_reused_what_one_not_published_made_is_not_published(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.execution import ProvenanceError
    from tests.support import stack

    marks = tmp_path / "marks"
    marks.mkdir()
    stack.install(monkeypatch, marks, **{"*": {"sleep": 1}})
    _unpublished(monkeypatch, "a")
    with (
        EvaluationSession.open(problem_file, [cell_count()], workers=2) as session,
        session.evaluate_each(
            [
                RunRequest({RADIUS: 450.0}, tmp_path / "a"),
                RunRequest({RADIUS: 450.0}, tmp_path / "b"),
            ]
        ) as evaluations,
    ):
        outcomes = [(Path(request.output).name, outcome) for request, outcome in evaluations]
    assert sorted(stack.started(marks)) == ["a.gamma"]  # b reused a's simulation
    assert [name for name, _ in outcomes] == ["a", "b"]  # b is handed on after a
    assert all(isinstance(outcome, ProvenanceError) for _, outcome in outcomes)
    assert "were not published" in str(outcomes[1][1])
    assert not (tmp_path / "b" / SUMMARY_NAME).exists()  # it would name a record that is not


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


# --- a stage that waits for its twin holds no worker ----------------------------------------------


@pytest.fixture
def marks(tmp_path: Path) -> Path:
    """Where the fake ddsim of the workers marks each simulation's start and end."""
    path = tmp_path / "marks"
    path.mkdir()
    return path


@pytest.fixture
def mixed(tmp_path: Path, problem_file: Path) -> Path:
    """:func:`problem_file` with a second sample, ``pion``, whose gun energy the candidate
    gives: candidates at one radius simulate ``gamma`` alike, and each its own ``pion``."""
    document = yaml.safe_load(problem_file.read_text())
    pion = copy.deepcopy(document["samples"]["gamma"])
    pion["generation"]["particle"] = "pi-"
    del pion["generation"]["energy"]
    pion["simulation"]["seed"] = 6
    document["samples"]["pion"] = pion
    document["parameters"]["energy"] = {
        "targets": {"generation": {"option": "energy", "samples": ["pion"]}},
        "type": "continuous",
        "bounds": [1.0, 100.0],
        "unit": "GeV",
    }
    document["evaluators"][0]["samples"] = ["gamma", "pion"]
    document["objectives"][0]["sample"] = "gamma"
    problem_file.write_text(yaml.safe_dump(document))
    return problem_file


def evaluated(
    problem: Path, requests: list[RunRequest], workers: int = 2, **open_: Any
) -> dict[str, Any]:
    """The outcome of each request, by the name of its run directory, evaluated by ``workers``;
    no worker is left once the session closed."""
    with (
        EvaluationSession.open(problem, [cell_count()], workers=workers, **open_) as session,
        session.evaluate_each(requests) as evaluations,
    ):
        outcomes = {Path(request.output).name: outcome for request, outcome in evaluations}
    assert stack.workers() == []
    return outcomes


def verified(outcome: Any) -> dict[str, Any]:
    """The summary on disk of a valid candidate, which holds up without the session."""
    assert outcome.candidate.status == "valid"
    data: dict[str, Any] = json.loads(outcome.summary_path.read_text())
    Assessment.from_summary(load_problem(Path(data["problem"]["path"])), data)
    return data


def test_a_stage_waiting_for_its_twin_leaves_its_worker_to_another_stage(
    tmp_path: Path, mixed: Path, marks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a's gamma ends only once b's pion started: b's gamma, which waits for a's, must leave b's
    # pion the second worker (it held it before, and a's gamma waited in vain)
    stack.install(monkeypatch, marks, **{"a.gamma": {"await": ["b.pion.started"]}})
    outcomes = evaluated(
        mixed,
        [
            RunRequest({RADIUS: 450.0, "energy": 10.0}, tmp_path / "a"),
            RunRequest({RADIUS: 450.0, "energy": 20.0}, tmp_path / "b"),
        ],
    )
    assert sorted(stack.started(marks)) == ["a.gamma", "a.pion", "b.pion"]  # gamma once
    made, reused = verified(outcomes["a"]), verified(outcomes["b"])
    assert simulated(reused)["reused"]["record"] == made["id"]
    assert simulated(reused)["signature"] == simulated(made)["signature"]
    assert "reused" not in reused["samples"]["pion"]["stages"]["simulation"]


def test_stages_waiting_for_one_that_failed_run_once_more_and_the_rest_reuse_that(
    tmp_path: Path, problem_file: Path, marks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch, marks, **{"a.gamma": {"sleep": 1, "exit": 3}})
    outcomes = evaluated(
        problem_file,
        [RunRequest({RADIUS: 450.0}, tmp_path / name) for name in ("a", "b", "c")],
        workers=3,
    )
    assert outcomes["a"].candidate.status == "failed"
    started = sorted(stack.started(marks))
    assert started[0] == "a.gamma" and len(started) == 2  # one of b and c ran it again
    ran = started[1].partition(".")[0]
    other = ({"b", "c"} - {ran}).pop()
    made = verified(outcomes[ran])
    assert "reused" not in simulated(made)
    assert simulated(verified(outcomes[other]))["reused"]["record"] == made["id"]


def test_a_stage_waiting_for_one_whose_worker_died_runs_itself(
    tmp_path: Path, problem_file: Path, marks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(
        monkeypatch, marks, **{"*": {"sleep": 1}, "a.gamma": {"sleep": 1, "kill": "parent"}}
    )
    outcomes = evaluated(
        problem_file, [RunRequest({RADIUS: 450.0}, tmp_path / name) for name in ("a", "b")]
    )
    assert outcomes["a"].candidate.status == "failed"
    assert sorted(stack.started(marks)) == ["a.gamma", "b.gamma"]
    assert "reused" not in simulated(verified(outcomes["b"]))


class _StoppedError(Exception):
    """What stops the coordinator at a chosen point."""


def test_stopping_while_a_stage_waits_for_its_twin_leaves_nothing_running_nor_waiting(
    tmp_path: Path, problem_file: Path, marks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.execution._schedule import Scheduler

    # neither a nor b simulates until stopped; c, evaluated afterwards, does
    never = {"await": ["never"]}
    stack.install(monkeypatch, marks, **{"a.gamma": never, "b.gamma": never})
    park, seen = Scheduler.park, []

    def stopping(self: Scheduler, *args: Any) -> None:  # as Ctrl-C would, once one waits
        park(self, *args)
        seen.append(self.busy)
        raise _StoppedError

    monkeypatch.setattr(Scheduler, "park", stopping)
    with EvaluationSession.open(problem_file, [cell_count()], workers=2) as session:
        requests = [RunRequest({RADIUS: 450.0}, tmp_path / name) for name in ("a", "b")]
        with pytest.raises(_StoppedError), session.evaluate_each(requests) as each:
            list(each)
        assert seen == [1]  # the one that waits holds no worker
        assert len(stack.started(marks)) <= 1  # the twin never started (the other may have)
        assert all(eventually(lambda p=p: not alive(p)) for p in stack.started(marks).values())
        assert not (tmp_path / "a" / SUMMARY_NAME).exists()
        assert not (tmp_path / "b" / SUMMARY_NAME).exists()
        monkeypatch.setattr(Scheduler, "park", park)
        # what they ran or waited for is forgotten: c simulates, as a stage of its own
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "c")
    assert "reused" not in simulated(verified(outcome))
    assert "c.gamma" in stack.started(marks)
    assert stack.workers() == []


def test_two_samples_alike_in_one_candidate_simulate_once_and_end_once(
    tmp_path: Path, problem_file: Path, marks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = yaml.safe_load(problem_file.read_text())
    document["samples"]["twin"] = copy.deepcopy(document["samples"]["gamma"])
    document["evaluators"][0]["samples"] = ["gamma", "twin"]
    document["objectives"][0]["sample"] = "gamma"
    problem_file.write_text(yaml.safe_dump(document))
    stack.install(monkeypatch, marks, **{"*": {"sleep": 1}})
    outcomes = evaluated(problem_file, [RunRequest({RADIUS: 450.0}, tmp_path / "a")])
    assert len(stack.started(marks)) == 1  # one of the two, the other reused it
    record = verified(outcomes["a"])
    stages = [record["samples"][name]["stages"]["simulation"] for name in ("gamma", "twin")]
    (reused,) = [stage["reused"] for stage in stages if "reused" in stage]
    assert reused["record"] == record["id"] and reused["run_dir"] == str(tmp_path / "a")


def test_a_failure_while_waiting_stages_are_settled_stops_only_its_own_candidate(
    tmp_path: Path, problem_file: Path, marks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.execution import _drive

    original, calls = _drive.reused, []

    def failing_once(*args: Any) -> Any:
        calls.append(args)
        if len(calls) == 1:
            raise _StoppedError("cannot take it")
        return original(*args)

    monkeypatch.setattr(_drive, "reused", failing_once)
    stack.install(monkeypatch, marks, **{"a.gamma": {"sleep": 1}})
    names = ("a", "b", "c")
    outcomes = evaluated(
        problem_file, [RunRequest({RADIUS: 450.0}, tmp_path / name) for name in names], workers=3
    )
    assert sorted(stack.started(marks)) == ["a.gamma"]
    made = verified(outcomes["a"])
    failed = [name for name in ("b", "c") if isinstance(outcomes[name], _StoppedError)]
    (other,) = {"b", "c"} - set(failed)
    assert len(failed) == 1 and not (tmp_path / failed[0] / SUMMARY_NAME).exists()
    assert simulated(verified(outcomes[other]))["reused"]["record"] == made["id"]


# --- recorded outputs: found at any depth, read once, hashed when reused --------------------------


def test_a_directory_is_searched_for_runs_at_any_depth(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        first = session.evaluate({RADIUS: 450.0}, tmp_path / "study" / "inner" / "t" / "a")
        session.evaluate({RADIUS: 460.0}, tmp_path / "study" / "b")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        assert session.reuse([tmp_path / "study"]) == 2
        assert session.reuse([tmp_path / "study", tmp_path / "study" / "b"]) == 0  # read already
        session.evaluate({RADIUS: 450.0}, tmp_path / "c")
    assert len(fake.calls) == 2
    assert simulated(summary(tmp_path / "c"))["reused"]["record"] == first.summary["id"]


def test_a_directory_searched_before_a_run_in_it_ended_is_searched_again(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    with EvaluationSession.open(problem_file, [cell_count()]) as later:
        assert later.reuse([runs]) == 0
        with EvaluationSession.open(problem_file, [cell_count()]) as session:
            first = session.evaluate({RADIUS: 450.0}, runs / "a")
        assert later.reuse([runs]) == 1
        later.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert len(fake.calls) == 1
    assert simulated(summary(tmp_path / "b"))["reused"]["record"] == first.summary["id"]


@pytest.mark.parametrize("change", ["rewritten", "removed"])
def test_a_recorded_output_that_changed_gives_way_to_another_of_its_signature(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, change: str
) -> None:
    for name in ("a", "b"):  # two sessions, each simulating what the other did
        with EvaluationSession.open(problem_file, [cell_count()]) as session:
            session.evaluate({RADIUS: 450.0}, tmp_path / "runs" / name)
    assert (
        simulated(summary(tmp_path / "runs" / "a"))["signature"]
        == simulated(summary(tmp_path / "runs" / "b"))["signature"]
    )
    events = (
        tmp_path
        / "runs"
        / "a"
        / simulated(summary(tmp_path / "runs" / "a"))["outputs"]["events"]["path"]
    )
    if change == "rewritten":
        events.write_text(events.read_text() + " ")
    else:
        events.unlink()
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        assert session.reuse([tmp_path / "runs"]) == 2  # registered; hashed only when reused
        session.evaluate({RADIUS: 450.0}, tmp_path / "c")
        session.evaluate({RADIUS: 450.0}, tmp_path / "d")
    assert len(fake.calls) == 2
    for name in ("c", "d"):
        assert simulated(summary(tmp_path / name))["reused"]["run_dir"] == str(
            tmp_path / "runs" / "b"
        )


def test_a_recorded_output_that_changed_is_never_reused(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    events = tmp_path / "a" / simulated(summary(tmp_path / "a"))["outputs"]["events"]["path"]
    events.write_text(events.read_text() + " ")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        assert session.reuse([tmp_path / "a"]) == 1
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert len(fake.calls) == 2
    assert outcome.candidate.status == "valid"
    assert "reused" not in simulated(summary(tmp_path / "b"))
