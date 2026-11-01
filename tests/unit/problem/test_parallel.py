"""Candidates evaluated with several jobs: their tasks in worker processes, on a fake stack.

The fake ``ddsim`` and ``podio`` (:mod:`tests.support.stack`) and ``test.worker_probe`` do in the
workers what a test tells them through the environment, which the workers inherit; markers in a
directory, never sleeps, say when a simulation started or ended.
"""

from __future__ import annotations

import copy
import errno
import json
import multiprocessing
import os
import pickle
import signal
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.evaluation import EvaluationContractError, load_evaluator
from k4opt.problem import (
    SUMMARY_NAME,
    Assessment,
    CandidateError,
    EvaluationSession,
    RunError,
    RunOutcome,
    RunRequest,
    _backends,
    _run,
    _worker,
    load_problem,
)
from k4opt.problem._tasks import Task, stage_task
from k4opt.problem._work import TaskOutcome, TaskSpec
from tests.support import stack
from tests.support.common import scientific
from tests.support.evaluators import CELL_COUNT, cell_count, implementations, worker_probe
from tests.support.geometry import MINI_COMPACT, MINI_GEO
from tests.support.stack import alive, eventually

RADIUS = "constants.InnerTracker_Barrel_radius_1"
SUPPORT = "InnerTrackerSupportCAD.shape.ref"  # cad/support.stl in the mini geometry
REPO = Path(__file__).resolve().parents[3]

type Opener = Callable[..., EvaluationSession]


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def marks(tmp_path: Path) -> Path:
    """Where the fake ddsim marks each simulation's start and end."""
    path = tmp_path / "marks"
    path.mkdir()
    return path


def write_problem(
    path: Path, doc: dict[str, Any], probes: tuple[str, ...] = (), required: bool = True
) -> Path:
    """The problem ``doc`` on the mini geometry with the samples ``gamma`` and ``pion``, each
    measured by ``test.cell_count``, and ``gamma`` by a ``test.worker_probe`` bound as each of
    ``probes`` (``required`` or a diagnostic); an objective on ``gamma``."""
    doc = copy.deepcopy(doc)
    doc["geometry"] = MINI_COMPACT
    doc["parameters"][0]["geometry"] = {"constant": "InnerTracker_Barrel_radius_1"}
    doc["parameters"][0]["bounds"] = [100.0, 600.0]
    gamma = doc["samples"]["gamma"]
    gamma["simulation"].update(events=3, seed=5)
    pion = copy.deepcopy(gamma)
    pion["generation"]["particle"] = "pi-"
    pion["simulation"]["seed"] = 6
    doc["samples"]["pion"] = pion
    doc["evaluators"] = [
        {
            "name": "test.cell_count",
            "version": "1",
            "stage": "simulation",
            "samples": ["gamma", "pion"],
            "collections": {"hits": "ECalBarrelCollection"},
        },
        *(
            {
                "name": "test.worker_probe",
                "as": name,
                "version": "1",
                "stage": "simulation",
                "samples": ["gamma"],
                "settings": {"binding": name},
            }
            | ({"required": True} if required else {})
            for name in probes
        ),
    ]
    doc["objectives"] = [
        {
            "stage": "simulation",
            "metric": "ecal_barrel.mean_hit_cells",
            "sample": "gamma",
            "evaluator": "test.cell_count",
            "direction": "minimize",
        }
    ]
    del doc["constraints"]
    path.write_text(yaml.safe_dump(doc))
    return path


@pytest.fixture
def problem(tmp_path: Path, doc: dict[str, Any]) -> Path:
    return write_problem(tmp_path / "problem.yaml", doc)


def probe(monkeypatch: pytest.MonkeyPatch, **told: dict[str, Any]) -> None:
    """Tell ``test.worker_probe`` what to do, by binding or ``<run>.<binding>``."""
    monkeypatch.setenv("WORKER_PROBE", json.dumps(told))


@pytest.fixture
def session_of(problem: Path) -> Iterator[Opener]:
    """Open sessions of ``problem`` (or ``path``) with ``test.cell_count``, and
    ``test.worker_probe`` if it binds that, each closed after the test."""
    opened: list[EvaluationSession] = []

    def open_(jobs: int, path: Path = problem) -> EvaluationSession:
        named = {entry["name"] for entry in yaml.safe_load(path.read_text())["evaluators"]}
        probes = [worker_probe()] if "test.worker_probe" in named else []
        session = EvaluationSession.open(path, [cell_count(), *probes], jobs=jobs)
        opened.append(session)
        return session

    yield open_
    for session in opened:
        session.close()
    assert stack.workers() == []


def verified(outcome: RunOutcome) -> dict[str, Any]:
    """The summary on disk, which verifies as a record of its problem without the session."""
    summary = json.loads(outcome.summary_path.read_text())
    spec = load_problem(Path(summary["problem"]["path"]))
    assert Assessment.from_summary(spec, summary).to_dict() == summary["assessment"]
    return summary


def files_of(summary: dict[str, Any]) -> list[str]:
    """Every path relative to the run directory that the summary names."""
    found = []
    for sample in summary["samples"].values():
        for stage in sample["stages"].values():
            found += [output["path"] for output in stage["outputs"].values()]
            found += list(stage["logs"].values())
        for evaluation in sample["evaluations"].values():
            found += [artifact["path"] for artifact in evaluation["artifacts"].values()]
            found += [evaluation["log"]] if evaluation["log"] else []
    return found


# --- what runs at once --------------------------------------------------------------------------


def test_the_simulations_of_two_samples_run_at_once(
    tmp_path: Path, marks: Path, session_of: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch, marks, **{"*": {"await": 2}})  # each waits for the other to start
    outcome = session_of(2).evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == "valid", outcome.candidate.message
    summary = verified(outcome)
    assert summary["environment"]["observed"]["execution"] == {"backend": "local", "jobs": 2}
    ended = {path.name for path in marks.glob("*.ended")}
    assert ended == {"a.gamma.ended", "a.pion.ended"}


def test_the_evaluations_of_one_simulation_run_at_once_each_on_an_instance_it_checked(
    tmp_path: Path,
    doc: dict[str, Any],
    marks: Path,
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = ("p1", "p2", "p3")
    problem = write_problem(tmp_path / "three.yaml", doc, names)
    stack.install(monkeypatch, marks)
    every = [str(marks / name) for name in names]
    probe(monkeypatch, **{name: {"mark": str(marks / name), "await": every} for name in names})
    outcome = session_of(3, problem).evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == "valid", outcome.candidate.message
    for name in names:
        (measured,) = outcome.candidate.record(f"gamma/{name}").result.measurements
        assert (measured.metric, measured.value) == ("checked", 1)
    verified(outcome)


def test_what_an_evaluator_prints_in_a_worker_goes_to_its_log(
    tmp_path: Path,
    doc: dict[str, Any],
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = write_problem(tmp_path / "probe.yaml", doc, ("p",))
    stack.install(monkeypatch)
    probe(monkeypatch, p={"print": "measured in a worker"})
    outcome = session_of(2, problem).evaluate({RADIUS: 450.0}, tmp_path / "a")
    evaluation = verified(outcome)["samples"]["gamma"]["evaluations"]["p"]
    assert evaluation["log"] == "samples/gamma/evaluators/p/evaluator.log"
    log = (tmp_path / "a" / evaluation["log"]).read_text()
    assert log == "measured in a worker\n" * 2
    cells = verified(outcome)["samples"]["gamma"]["evaluations"]["test.cell_count"]
    assert cells["log"] == "samples/gamma/evaluators/test.cell_count/evaluator.log"


def test_a_candidate_gives_the_same_record_with_one_job_and_with_four(
    tmp_path: Path,
    doc: dict[str, Any],
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = write_problem(tmp_path / "probe.yaml", doc, ("p",))
    stack.install(monkeypatch, **{"gamma": {"hits": [[1, 2], [3], [4, 5, 6]]}})
    one = session_of(1, problem).evaluate({RADIUS: 450.0}, tmp_path / "one")
    four = session_of(4, problem).evaluate({RADIUS: 450.0}, tmp_path / "four")
    first, second = verified(one), verified(four)
    assert scientific(first) == scientific(second)
    assert first["environment"]["observed"]["execution"] == {"backend": "local", "jobs": 1}
    assert second["environment"]["observed"]["execution"] == {"backend": "local", "jobs": 4}
    assert first["samples"]["gamma"]["evaluations"]["p"]["log"] is None
    assert second["samples"]["gamma"]["evaluations"]["p"]["log"] is not None
    assert first["status"] == "valid" and first["metrics"][0]["value"] == 2.0
    # both verify without the session, and without an optimizer
    check = (
        "import json, sys\n"
        "sys.modules['optuna'] = None\n"
        "from k4opt.problem import Assessment, load_problem\n"
        "for path in sys.argv[1:]:\n"
        "    summary = json.load(open(path))\n"
        "    spec = load_problem(summary['problem']['path'])\n"
        "    assert Assessment.from_summary(spec, summary).to_dict() == summary['assessment']\n"
    )
    paths = [str(one.summary_path), str(four.summary_path)]
    done = subprocess.run([sys.executable, "-c", check, *paths], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_concurrent_candidates_each_write_and_record_only_their_own_files(
    tmp_path: Path, marks: Path, session_of: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch, marks, **{"*": {"await": 2}})  # simulations of two at once, too
    requests = [
        RunRequest({RADIUS: radius}, tmp_path / name, {"name": name})
        for name, radius in [("a", 300.0), ("b", 400.0), ("c", 500.0)]
    ]
    with session_of(2).evaluate_each(requests) as evaluations:
        ended = list(evaluations)
    assert sorted(request.run["name"] for request, _ in ended) == ["a", "b", "c"]  # type: ignore[index]
    ids = set()
    for request, outcome in ended:
        assert isinstance(outcome, RunOutcome), outcome
        summary = verified(outcome)
        assert outcome.run_dir == tmp_path / request.run["name"]  # type: ignore[index]
        assert summary["run"] == request.run and summary["candidate"] == request.candidate
        for name in files_of(summary):
            assert (outcome.run_dir / name).is_file() and not Path(name).is_absolute()
        ids.add(summary["id"])
    assert len(ids) == 3


def threaded(path: Path, threads: int) -> Path:
    """``path``'s problem with the gamma simulation run on ``threads`` threads."""
    document = yaml.safe_load(path.read_text())
    document["samples"]["gamma"]["simulation"]["numberOfThreads"] = threads
    path.write_text(yaml.safe_dump(document))
    return path


def test_a_simulation_on_several_threads_holds_as_many_jobs(
    tmp_path: Path,
    doc: dict[str, Any],
    marks: Path,
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = threaded(write_problem(tmp_path / "threads.yaml", doc), 2)
    stack.install(monkeypatch, marks, **{"*": {"await": 2}})  # each waits for the other
    outcome = session_of(3, problem).evaluate({RADIUS: 450.0}, tmp_path / "a")  # 2 + 1 slots
    assert outcome.candidate.status == "valid", outcome.candidate.message
    simulated = verified(outcome)["samples"]["gamma"]["stages"]["simulation"]
    assert "--numberOfThreads=2" in simulated["argv"]

    stack.install(monkeypatch, marks)
    session = session_of(2, problem)  # the gamma simulation takes both jobs: it runs alone
    assert not [warning for warning in session.warnings if "threads" in warning]
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert outcome.candidate.status == "valid", outcome.candidate.message
    times = {name: (marks / f"b.{name}.started").stat().st_mtime_ns for name in ("gamma", "pion")}
    ended = {name: (marks / f"b.{name}.ended").stat().st_mtime_ns for name in times}
    first, second = sorted(times, key=times.__getitem__)
    assert ended[first] <= times[second]  # one after the other
    warned = [warning for warning in session_of(1, problem).warnings if "threads" in warning]
    assert warned == [
        "gamma/simulation simulates with 2 threads, more than the 1 job(s) of the session: it "
        "runs alone, on 2 processors"
    ]


# --- how much memory each task took ------------------------------------------------------------

MIB = 1 << 20


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/clear_refs is Linux's")
def test_each_task_records_its_own_peak_memory_in_reused_workers(
    tmp_path: Path,
    doc: dict[str, Any],
    marks: Path,
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = write_problem(tmp_path / "probe.yaml", doc, ("p",))
    stack.install(monkeypatch, marks, **{"a.gamma": {"allocate": 300}})
    probe(monkeypatch, **{"a.p": {"allocate": 200}})
    session = session_of(2, problem)
    big = verified(session.evaluate({RADIUS: 450.0}, tmp_path / "a"))
    small = verified(session.evaluate({RADIUS: 450.0}, tmp_path / "b"))  # the same two workers
    for summary in (big, small):
        for sample in ("gamma", "pion"):
            simulated = summary["samples"][sample]["stages"]["simulation"]["peak_memory_bytes"]
            assert simulated["process"] > 0 and simulated["program"] > 0
        assert summary["geometry"]["peak_memory_bytes"]["program"] is None
    stages = {
        run: summary["samples"]["gamma"]["stages"]["simulation"]["peak_memory_bytes"]["program"]
        for run, summary in (("a", big), ("b", small))
    }
    assert stages["a"] >= 300 * MIB > stages["b"] + 200 * MIB  # ddsim, and only ddsim
    probed = {
        run: summary["samples"]["gamma"]["evaluations"]["p"]["peak_memory_bytes"]
        for run, summary in (("a", big), ("b", small))
    }
    assert probed["a"]["process"] >= 200 * MIB > probed["b"]["process"] + 100 * MIB
    assert probed["a"]["program"] is None
    # what the worker held while ddsim ran is its own, not ddsim's
    simulated = big["samples"]["gamma"]["stages"]["simulation"]["peak_memory_bytes"]
    assert simulated["process"] < 300 * MIB


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/clear_refs is Linux's")
def test_what_making_and_checking_the_instance_took_counts_in_a_worker(
    tmp_path: Path,
    doc: dict[str, Any],
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = write_problem(tmp_path / "probe.yaml", doc, ("p",))
    stack.install(monkeypatch)
    probe(monkeypatch, p={"check_allocate": 300})  # in check(), released before evaluate()
    summary = verified(session_of(2, problem).evaluate({RADIUS: 450.0}, tmp_path / "a"))
    memory = summary["samples"]["gamma"]["evaluations"]["p"]["peak_memory_bytes"]
    assert memory["process"] >= 300 * MIB


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/clear_refs is Linux's")
def test_with_one_job_the_session_records_its_own_peak_memory(
    tmp_path: Path, session_of: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch, **{"gamma": {"allocate": 200}})
    summary = verified(session_of(1).evaluate({RADIUS: 450.0}, tmp_path / "a"))
    memory = {
        sample: summary["samples"][sample]["stages"]["simulation"]["peak_memory_bytes"]
        for sample in ("gamma", "pion")
    }
    assert memory["gamma"]["program"] >= 200 * MIB > memory["pion"]["program"]
    assert summary["samples"]["gamma"]["evaluations"]["test.cell_count"]["peak_memory_bytes"][
        "process"
    ] == pytest.approx(memory["pion"]["process"], rel=0.5)  # this process, not ddsim


# --- what fails where ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("told", "error"), [("raise", ValueError), ("contract", EvaluationContractError)]
)
def test_an_exception_in_a_worker_stops_its_candidate_with_its_traceback(
    tmp_path: Path,
    doc: dict[str, Any],
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
    told: str,
    error: type[Exception],
) -> None:
    problem = write_problem(tmp_path / "probe.yaml", doc, ("p",))
    stack.install(monkeypatch)
    probe(monkeypatch, **{"a.p": {told: "no measurement here"}})
    session = session_of(2, problem)
    with pytest.raises(error, match="no measurement here") as raised:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    cause = raised.value.__cause__
    assert isinstance(cause, _worker.RemoteTraceback)
    assert "worker_probe.py" in str(cause) and error.__name__ in str(cause)
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()
    # the session goes on, with the same worker processes
    assert session.evaluate({RADIUS: 450.0}, tmp_path / "b").candidate.status == "valid"


def test_an_exception_that_cannot_be_rebuilt_is_raised_as_a_task_error() -> None:
    class OddError(Exception):
        def __init__(self, first: str, second: str) -> None:
            super().__init__(f"{first} {second}")

    failure = _worker.Failure.of(OddError("two", "arguments"))
    assert failure.exception is None
    error = failure.error()
    assert isinstance(error, _worker.TaskError)
    assert str(error).endswith("OddError: two arguments")
    assert isinstance(error.__cause__, _worker.RemoteTraceback)


@pytest.mark.parametrize(
    ("killed", "required", "status", "message"),
    [
        ("ddsim", True, "failed", "ddsim exited with status -9"),
        ("the worker during ddsim", True, "failed", "died (killed by SIGKILL)"),
        ("the worker during an evaluation", True, "failed", "died (killed by SIGKILL)"),
        ("the worker during an evaluation", False, "valid", None),
    ],
)
def test_a_process_killed_like_the_oom_killer_does_fails_its_task_and_leaves_nothing_behind(
    tmp_path: Path,
    doc: dict[str, Any],
    marks: Path,
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
    killed: str,
    required: bool,
    status: str,
    message: str | None,
) -> None:
    problem = write_problem(tmp_path / "probe.yaml", doc, ("p",), required=required)
    told: dict[str, Any] = {
        "ddsim": {"a.gamma": {"kill": "self"}},
        "the worker during ddsim": {"a.gamma": {"kill": "parent", "sleep": 600}},
    }.get(killed, {})
    stack.install(monkeypatch, marks, **told)
    probe(monkeypatch, **({"a.p": {"kill": True}} if "evaluation" in killed else {}))
    session = session_of(2, problem)
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == status
    summary = verified(outcome)  # a complete record of what happened, on disk
    failed = (
        summary["samples"]["gamma"]["evaluations"]["p"]
        if "evaluation" in killed
        else summary["samples"]["gamma"]["stages"]["simulation"]
    )
    assert failed["status"] == "failed"
    reason = "died (killed by SIGKILL)" if "worker" in killed else "status -9"
    assert reason in failed["message"]
    if "evaluation" not in killed:  # what ddsim wrote so far stays referenced
        assert {"stdout", "stderr"} <= failed["logs"].keys()
        assert all((outcome.run_dir / log).is_file() for log in failed["logs"].values())
    if message is not None:
        assert message in (outcome.candidate.message or "")
    else:  # a diagnostic: recorded, without deciding the candidate
        assert summary["diagnostics"][0]["task"] == "gamma/p"
    assert eventually(lambda: not any(alive(pid) for pid in stack.started(marks).values()))
    # the next candidate runs as if nothing happened, in a new worker where one died
    again = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert again.candidate.status == "valid", again.candidate.message
    verified(again)


def test_a_worker_that_died_while_idle_is_replaced_without_failing_a_task(
    tmp_path: Path, session_of: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch)
    session = session_of(2)
    (idle,) = stack.workers()  # the one the session started when it opened
    os.kill(idle, signal.SIGKILL)
    assert eventually(lambda: not alive(idle))
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == "valid", outcome.candidate.message


def test_workers_that_keep_dying_on_one_task_stop_its_candidates(
    tmp_path: Path,
    doc: dict[str, Any],
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = write_problem(tmp_path / "probe.yaml", doc, ("p",))
    stack.install(monkeypatch)
    probe(monkeypatch, p={"kill": True})
    requests = [RunRequest({RADIUS: 450.0}, tmp_path / name) for name in "abc"]
    with session_of(2, problem).evaluate_each(requests) as evaluations:
        ended = [outcome for _, outcome in evaluations]
    failed = [o for o in ended if isinstance(o, RunOutcome)]
    stopped = [o for o in ended if isinstance(o, RunError)]
    assert len(failed) == 2 and all(o.candidate.status == "failed" for o in failed)
    (error,) = stopped
    assert "3 times in a row for p; lower --jobs?" in str(error)


# --- stopping ---------------------------------------------------------------------------------


@pytest.mark.parametrize("closed", ["the evaluations", "the session"])
def test_closing_stops_every_worker_and_simulation_still_running(
    tmp_path: Path,
    marks: Path,
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
    closed: str,
) -> None:
    stack.install(monkeypatch, marks, **{"b.gamma": {"sleep": 600}, "b.pion": {"sleep": 600}})
    session = session_of(3)
    evaluations = session.evaluate_each(
        [RunRequest({RADIUS: 450.0}, tmp_path / "a"), RunRequest({RADIUS: 450.0}, tmp_path / "b")]
    )
    _, outcome = next(evaluations)
    assert isinstance(outcome, RunOutcome) and outcome.candidate.status == "valid"
    assert eventually(lambda: any(run.startswith("b.") for run in stack.started(marks)))
    sleeping = [pid for run, pid in stack.started(marks).items() if run.startswith("b.")]
    running = stack.workers()
    assert running
    if closed == "the session":
        session.close()
        assert stack.workers() == [] and not any(alive(pid) for pid in running)
    else:
        evaluations.close()
    assert eventually(lambda: not any(alive(pid) for pid in sleeping))
    assert not (tmp_path / "b" / SUMMARY_NAME).exists()
    assert list(evaluations) == []
    # a closed session starts new workers when it evaluates again
    assert session.evaluate({RADIUS: 450.0}, tmp_path / "c").candidate.status == "valid"


@pytest.mark.parametrize(
    "signum",
    [pytest.param(signal.SIGINT, id="SIGINT"), pytest.param(signal.SIGTERM, id="SIGTERM")],
)
def test_interrupting_the_command_leaves_no_worker_nor_simulation(
    tmp_path: Path,
    problem: Path,
    marks: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: signal.Signals,
) -> None:
    stack.install(monkeypatch, marks, **{"*": {"sleep": 600}})
    path = os.pathsep.join([str(stack.FAKE_STACK / "python"), str(REPO), os.environ["PYTHONPATH"]])
    command = [sys.executable, "-m", "k4opt", "evaluate", str(problem)]
    command += ["--evaluator", CELL_COUNT, "--param", f"{RADIUS}=450", "--jobs", "2"]
    command += ["--output", str(tmp_path / "a")]
    process = subprocess.Popen(
        command, env=os.environ | {"PYTHONPATH": path}, stderr=subprocess.PIPE
    )
    try:
        assert eventually(lambda: len(stack.started(marks)) == 2), process.stderr
        running = stack.workers(process.pid)
        sleeping = list(stack.started(marks).values())
        assert len(running) == 2
        process.send_signal(signum)
        process.wait(timeout=30)
    finally:
        process.kill()
        process.wait()
        assert process.stderr is not None
        process.stderr.close()
    assert eventually(lambda: not any(alive(pid) for pid in [*running, *sleeping]))
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()


def test_the_command_evaluates_with_several_jobs(
    tmp_path: Path,
    problem: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stack.install(monkeypatch, **{"pion": {"allocate": 300}})
    output = tmp_path / "a"
    argv = ["evaluate", str(problem), "--evaluator", CELL_COUNT, "--param", f"{RADIUS}=450"]
    assert main([*argv, "-j", "2", "--output", str(output)]) == 0, capsys.readouterr().err
    if sys.platform.startswith("linux"):  # the task that needs most: the pions' ddsim
        line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("memory:"))
        assert " per job (estimate), for pion/simulation (k4opt " in line and ", ddsim 3" in line
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["environment"]["observed"]["execution"] == {"backend": "local", "jobs": 2}
    assert stack.workers() == []  # the command's session closed its workers
    assert main([*argv, "--jobs", "0", "--output", str(tmp_path / "b")]) == 1
    assert "jobs must be a positive integer or 'auto', got 0" in capsys.readouterr().err


# --- what the workers must share with the session ---------------------------------------------


def test_a_worker_rebuilds_the_session_from_plain_data(
    problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))  # which a worker takes the session's of
    with EvaluationSession.open(problem, [cell_count()]) as session:
        environment = session.identity()["environment"]
        assert isinstance(environment, dict)
        described = _worker.session_spec(
            session._context, session._binding.evaluations, environment
        )
    told: list[int] = []
    rebuilt = _worker.Session.of(pickle.loads(pickle.dumps(described)), started=told.append)
    assert rebuilt.context == session._context
    assert rebuilt.bound == dict(session._binding.evaluations)
    for stages in rebuilt.context.contexts.values():  # what only a worker's stages do
        assert stages.started == told.append and stages.die_with_parent
    assert not any(stages.die_with_parent for stages in session._context.contexts.values())


def test_a_worker_that_cannot_be_made_to_end_with_the_session_refuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_worker, "_PR_SET_PDEATHSIG", 0x7FFF)  # no option of prctl(2)
    refused = _worker._die_with(os.getppid())
    if sys.platform.startswith("linux"):
        assert refused is not None
        assert "cannot be made to end with the session (prctl: Invalid argument)" in refused
    ours, theirs = multiprocessing.Pipe()
    ours.send("the session")
    assert _worker.serve(theirs, "why not") == 1
    assert ours.recv() == ("refused", "why not")


# an evaluator file whose import hangs in a worker of this process once a marker file exists
HANGING = """\
import os, time
from pathlib import Path

if os.getppid() == {parent} and Path({marker!r}).exists():
    time.sleep(600)
"""


@pytest.mark.parametrize("hangs", ["the first worker", "a later worker"])
def test_a_worker_that_hangs_while_it_sets_up_is_stopped(
    tmp_path: Path, problem: Path, monkeypatch: pytest.MonkeyPatch, hangs: str
) -> None:
    stack.install(monkeypatch)
    marker = tmp_path / "hang"
    hanging = tmp_path / "hanging_count.py"
    source = (REPO / "tests" / "support" / "cell_count.py").read_text()
    hanging.write_text(HANGING.format(parent=os.getpid(), marker=str(marker)) + source)
    implementation = load_evaluator(f"{hanging}:CellCount")
    message = r"a worker process did not set up the session within 2 s .* and was stopped"
    if hangs == "the first worker":
        marker.touch()
        monkeypatch.setattr(_backends, "_SETUP", 2.0)
        with pytest.raises(RunError, match=message):
            EvaluationSession.open(problem, [implementation], jobs=2)
        assert stack.workers() == []
        return
    with EvaluationSession.open(problem, [implementation], jobs=2) as session:
        marker.touch()  # the second worker starts once both simulations are ready
        monkeypatch.setattr(_backends, "_SETUP", 2.0)
        with pytest.raises(RunError, match=message):
            session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        assert len(stack.workers()) == 1  # the one that set up; the hung one was stopped
        assert not (tmp_path / "a" / SUMMARY_NAME).exists()
    assert stack.workers() == []


def test_a_task_that_cannot_be_sent_whole_stops_its_candidate_and_its_worker(
    tmp_path: Path, session_of: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch)
    session = session_of(2)
    (ready,) = stack.workers()  # set up when the session opened
    sent = Connection.send

    def stalled(self: Connection, message: Any) -> None:
        if isinstance(message, tuple) and isinstance(message[0], TaskSpec):
            # the header of a message whose rest never comes: the worker waits for it, and
            # the send times out (SO_SNDTIMEO)
            os.write(self.fileno(), struct.pack("!i", 1000))
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        sent(self, message)

    monkeypatch.setattr(Connection, "send", stalled)
    message = r"geometry could not be sent to its worker process \(.*\); the worker was stopped"
    with pytest.raises(RunError, match=message):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert not alive(ready)  # never written to again, and stopped
    monkeypatch.setattr(Connection, "send", sent)
    assert session.evaluate({RADIUS: 450.0}, tmp_path / "b").candidate.status == "valid"


def test_a_worker_that_sends_part_of_an_answer_is_stopped_with_its_candidate(
    tmp_path: Path,
    doc: dict[str, Any],
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = write_problem(tmp_path / "probe.yaml", doc, ("p",))
    stack.install(monkeypatch)
    probe(monkeypatch, **{"a.p": {"stall": True}})
    monkeypatch.setattr(_backends, "_MESSAGE", 1.0)  # taken by each worker as it starts
    session = session_of(2, problem)
    message = r"gamma/p: a worker process sent part of an answer and nothing more for 1 s"
    with pytest.raises(RunError, match=message):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()
    assert session.evaluate({RADIUS: 450.0}, tmp_path / "b").candidate.status == "valid"


def test_a_worker_found_dead_as_its_setup_deadline_passes_is_ended_once(tmp_path: Path) -> None:
    backend = _backends.ProcessBackend(None, 2)  # type: ignore[arg-type]  # starts no worker
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    ours, theirs = socket.socketpair()
    theirs.close()  # what the worker's death leaves: the end of the stream
    worker = _backends._Worker(process, Connection(ours.detach()), setup=time.monotonic() - 1)
    task = Task(stage_task("simulation", "gamma"), directory="samples/gamma/simulation")
    worker.running = (_backends.Handle("w", task.id), TaskSpec("w", task))
    backend._workers.append(worker)
    ((handle, outcome, _),) = backend.wait(0)  # died, read first; the deadline then skips it
    assert handle.task == task.id
    assert isinstance(outcome, TaskOutcome) and outcome.status == "failed"
    assert "died (exit status 0)" in (outcome.message or "")
    assert backend._workers == []


@pytest.mark.parametrize("jobs", [1, 2])
def test_the_session_keeps_the_preflight_instances_only_when_it_evaluates_with_them(
    tmp_path: Path, session_of: Opener, monkeypatch: pytest.MonkeyPatch, jobs: int
) -> None:
    stack.install(monkeypatch)
    run = session_of(jobs)._start(RunRequest({RADIUS: 450.0}, tmp_path / "a"))
    assert run.prepared is not None
    assert len(run.prepared.entries) == len(run.prepared.requests) == 2  # for the records
    assert len(run.prepared.instances) == (2 if jobs == 1 else 0)


def test_an_evaluator_a_worker_cannot_load_again_is_refused_when_the_session_opens(
    problem: Path,
) -> None:
    with pytest.raises(RunError, match=r"was not loaded that way by k4opt\.evaluation"):
        EvaluationSession.open(problem, implementations(cell_count()()), jobs=2)
    assert stack.workers() == []


@pytest.mark.parametrize("differs", ["evaluator file", "k4opt code", "KEY4HEP_STACK"])
def test_a_worker_that_would_not_run_what_the_session_runs_refuses_to_start(
    tmp_path: Path, problem: Path, monkeypatch: pytest.MonkeyPatch, differs: str
) -> None:
    source = REPO / "tests" / "support" / "cell_count.py"
    edited = tmp_path / "edited_count.py"
    edited.write_bytes(source.read_bytes())
    implementation = load_evaluator(f"{edited}:CellCount")
    if differs == "evaluator file":
        edited.write_text(source.read_text() + "# edited after it was loaded\n")
        message = rf"evaluator {edited}:CellCount: {edited} changed after the session loaded it"
    elif differs == "k4opt code":
        monkeypatch.setattr(_worker, "code_fingerprint", lambda: "another checkout")
        message = "a worker process runs other k4opt code than the session"
    else:
        described = _run.session_spec
        monkeypatch.setattr(
            _run,
            "session_spec",
            lambda *args: described(*args[:2], {**args[2], "KEY4HEP_STACK": "/other/stack"}),
        )
        message = "KEY4HEP_STACK is .* in a worker process, not '/other/stack' as in the session"
    with pytest.raises(RunError, match=message):
        EvaluationSession.open(problem, [implementation], jobs=2)
    assert stack.workers() == []


# --- what candidates under way share: the files only their geometry names --------------------


@pytest.mark.parametrize("first", ["a", "b", None])
def test_concurrent_candidates_that_read_one_reference_alike_are_both_recorded_else_one(
    tmp_path: Path,
    doc: dict[str, Any],
    marks: Path,
    session_of: Opener,
    monkeypatch: pytest.MonkeyPatch,
    first: str | None,
) -> None:
    """Candidate ``b`` starts once the simulation of ``a`` did, after ``${CAD}/b.stl`` changed
    (unless ``first`` is ``None``); each simulation then puts back what its candidate read and
    waits as told, so that ``first`` ends first: the other is refused, whichever it is."""
    cad = tmp_path / "cad"
    cad.mkdir()
    model = cad / "b.stl"
    model.write_text("solid one")
    monkeypatch.setenv("CAD", str(cad))
    problem = write_problem(tmp_path / "cad.yaml", doc)
    data = yaml.safe_load(problem.read_text())
    choices = ["cad/support.stl", "${CAD}/b.stl"]
    data["parameters"].append(
        {"geometry": {"target": SUPPORT}, "type": "categorical", "choices": choices}
    )
    problem.write_text(yaml.safe_dump(data))
    summary = {name: str(tmp_path / name / SUMMARY_NAME) for name in "ab"}
    told: dict[str, Any] = {
        "a": {
            "a.gamma": {"await": ["b.gamma.started"], "write": {str(model): "solid one"}},
            "b.gamma": {"await": [summary["a"]], "write": {str(model): "solid two"}},
        },
        "b": {"a.gamma": {"await": [summary["b"]], "write": {str(model): "solid one"}}},
        None: {"a.gamma": {"await": ["b.gamma.started"]}},
    }[first]
    stack.install(monkeypatch, marks, **told)
    candidate = {RADIUS: 450.0, SUPPORT: "${CAD}/b.stl"}

    def requests() -> Iterator[RunRequest]:
        yield RunRequest(candidate, tmp_path / "a")
        # refused at once, so the session asks again: how to wait without holding it up
        while "a.gamma" not in stack.started(marks):
            yield RunRequest({RADIUS: 1000.0}, tmp_path / "never")
        if first is not None:
            model.write_text("solid two")
        yield RunRequest(candidate, tmp_path / "b")

    with session_of(2, problem).evaluate_each(requests()) as evaluations:
        ended = {
            Path(request.output).name: outcome
            for request, outcome in evaluations
            if not isinstance(outcome, CandidateError)
        }
    assert sorted(ended) == ["a", "b"]
    if first is None:
        assert all(isinstance(outcome, RunOutcome) for outcome in ended.values())
        return
    second = "b" if first == "a" else "a"
    assert isinstance(ended[first], RunOutcome)
    assert verified(ended[first])["geometry"]["candidate_externals"]  # what it read
    refused = ended[second]
    assert isinstance(refused, RunError)
    assert "does not read the files it read for an earlier candidate" in str(refused)
    assert not (tmp_path / second / SUMMARY_NAME).exists()
