"""A candidate's tasks and files as its record holds them, and its diagnostics."""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.ddsim import run as ddsim_run
from k4opt.evaluation import (
    Artifact,
    EvaluationRequest,
    result_from,
)
from k4opt.problem import (
    SUMMARY_NAME,
    Assessment,
    EvaluationSession,
    RecordError,
    RunError,
    _files,
    load_problem,
)
from k4opt.problem._summary import assemble
from k4opt.problem._tasks import Kind, Task, TaskId, TaskResult
from tests.support import simulation
from tests.support.common import scientific
from tests.support.evaluators import (
    cell_count,
    parameter_echo,
)
from tests.support.simulation import events_with
from tests.unit.problem.session.conftest import (
    CELLS,
    ECHO,
    RADIUS,
    change,
    declare,
    evaluate,
    particle,
    run_candidate,
    two_samples,
)

# --- a candidate's work is tasks: ordered by what they read, recorded by their results ---


def latest_ready(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    """Make the scheduler start, of the ready tasks, always the last in plan order: another order
    in which every task comes after what it reads. ``order`` receives each task as it starts, or
    as it is found blocked."""
    from k4opt.problem import _schedule

    scheduler = _schedule.Scheduler
    start, end = scheduler.next, scheduler.finished

    def latest_first(self: Any, key: Any, task: Task) -> None:
        index = self._index[key][task.id]
        heapq.heappush(self._ready, (self._candidates[key].order, -index, key, task))

    def starting(self: Any) -> Any:
        item = start(self)
        if item is not None:
            order.append(item[1].id.label)
        return item

    def ending(self: Any, key: Any, result: TaskResult) -> list[TaskResult]:
        decided = end(self, key, result)
        order.extend(held.id.label for held in decided)
        return decided

    monkeypatch.setattr(scheduler, "_push", latest_first)
    monkeypatch.setattr(scheduler, "next", starting)
    monkeypatch.setattr(scheduler, "finished", ending)


@pytest.mark.parametrize("ddsim", [{}, {"returncode": 2, "stderr": "G4Exception: bad volume"}])
def test_the_record_does_not_depend_on_the_order_the_tasks_ran_in(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch, ddsim: dict[str, Any]
) -> None:
    simulation.install(monkeypatch, **ddsim)
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])
    seen: list[bool] = []

    class Echo(parameter_echo()):  # type: ignore[misc]
        def evaluate(self, request: EvaluationRequest) -> Any:
            # every evaluator reads the geometry task, so the geometry is there in any order
            seen.append(request.inputs["geometry"].is_file())
            return super().evaluate(request)

    session = EvaluationSession.open(problem_file, [cell_count(), Echo])
    first = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    order: list[str] = []
    latest_ready(monkeypatch, order)
    second = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    planned = [task.id.label for task in second.candidate.plan.tasks]
    assert order != planned and sorted(order) == sorted(planned)
    assert order[:2] == ["gamma/generation", "geometry"]  # the gun first: it reads no geometry
    assert seen == [True, True]
    assert scientific(dict(second.summary)) == scientific(dict(first.summary))
    assert second.summary["status"] == ("valid" if not ddsim else "failed")


def test_the_record_is_assembled_from_the_task_results_alone(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a", run={"label": "golden"})
    written = json.loads(outcome.summary_path.read_text())
    shutil.rmtree(tmp_path / "a")  # nothing it makes the record from is read from disk
    again = assemble(
        written["id"],
        session.problem_path,
        session.spec,
        {RADIUS: 450.0},
        outcome.candidate,
        outcome.assessment,
        tmp_path / "a",
        session._taken,
        {"label": "golden"},
        session._process,
    )
    assert json.loads(json.dumps(again)) == written
    tasks = [
        written["geometry"],
        *written["samples"]["gamma"]["stages"].values(),
        *written["samples"]["gamma"]["evaluations"].values(),
    ]
    assert len(tasks) == 4 and written["evaluations"] == {}  # no evaluation of the candidate
    assert all(task["status"] in ("done", "valid") for task in tasks)


CELLED = "samples/gamma/evaluations/test.cell_count"


SIMULATED = "samples/gamma/stages/simulation"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        # what is blocked, by what, and why: re-derived from what each task reads
        (
            f"{CELLED}/status",
            "failed",
            "evaluator 'gamma/test.cell_count' is failed (blocked by ['gamma/simulation'], "
            "message 'not run: gamma/simulation failed'), but what it reads makes it blocked by "
            "['gamma/simulation']",
        ),
        (f"{CELLED}/blocked_by", [], "is blocked (blocked by [], message"),
        (
            f"{CELLED}/blocked_by",
            ["gamma/generation"],
            "is blocked (blocked by ['gamma/generation']",
        ),
        (f"{CELLED}/message", "not run: gamma/generation failed", "but what it reads makes it"),
        (
            f"{SIMULATED}/status",
            "done",
            "stage 'gamma/simulation': outputs [], but a stage that is done makes events in its "
            "directory",
        ),
        (
            f"{SIMULATED}/status",
            "blocked",
            "stage 'gamma/simulation' is blocked, but every task it reads succeeded",
        ),
        (f"{SIMULATED}/message", None, "stage 'gamma/simulation': a task that is failed needs a"),
        (f"{SIMULATED}/status", "crashed", "status 'crashed' is none of done, failed"),
        (
            "evaluations/test.parameter_echo/blocked_by",
            ["gamma/simulation"],
            "is blocked, but every task it reads",
        ),
        # what each task reads is the problem's, not the record's to say
        (
            f"{CELLED}/reads",
            ["geometry", "gamma/simulation"],
            "evaluator 'gamma/test.cell_count' reads ['geometry', 'gamma/simulation'], but the "
            "problem has it read ['geometry', 'gamma/generation', 'gamma/simulation']",
        ),
        (f"{SIMULATED}/reads", ["geometry"], "stage 'gamma/simulation' reads ['geometry'], but"),
        ("samples/gamma/stages/generation", None, "samples.gamma.stages.generation must be a map"),
        ("samples/gamma/stages", {}, "samples.gamma.stages are [], but the problem makes"),
        ("samples/gamma", None, "samples.gamma must be a mapping"),
        ("samples", {}, "its samples are [], but the problem's are ['gamma']"),
        ("geometry/status", "failed", "geometry: status 'failed', but it is always done"),
        ("supplied", ["gamma/simulation"], "but the problem makes ['simulation']"),
        ("supplied", ["simulation"], "supplied ['simulation'] must list stages of the problem's"),
        ("supplied", ["pion/simulation"], "must list stages of the problem's samples"),
        ("supplied", ["geometry"], "supplied ['geometry'] must list stages"),
        ("supplied", ["gamma/simulation"] * 2, "must list each stage once"),
        # where an evaluation is recorded says what it measured: a sample, or the candidate
        (
            "samples/gamma/evaluations",
            {},
            "samples.gamma.evaluations are [], but the problem's are ['test.cell_count']",
        ),
        (
            "evaluations/test.cell_count",
            {},
            "its evaluations are ['test.parameter_echo', 'test.cell_count'], but the problem's "
            "are ['test.parameter_echo']",
        ),
        ("metrics/0/sample", None, "its metrics are [['test.cell_count', None, "),
        # what a binding was configured with is the problem's, not the record's to say
        (
            f"{CELLED}/collections/hits",
            "OtherCollection",
            "evaluator 'gamma/test.cell_count': collections {'hits': 'OtherCollection'}, but the "
            "problem gives {'hits': ",
        ),
        (
            f"{CELLED}/settings/declared",
            {"threshold": 1},
            "evaluator 'gamma/test.cell_count': settings.declared {'threshold': 1}, but the "
            "problem",
        ),
        (
            "implementations/test.cell_count/stage",
            "generation",
            "implementation 'test.cell_count': stage 'generation', but the problem has it read "
            "'simulation'",
        ),
        ("message", "gamma/simulation failed: boom", "but its tasks make it"),
        ("status", "valid", "status is 'valid', but its tasks make it 'failed'"),
    ],
)
def test_a_record_whose_tasks_disagree_with_its_plan_is_refused(
    tmp_path: Path,
    problem_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: Any,
    message: str,
) -> None:
    simulation.install(monkeypatch, returncode=2, stderr="G4Exception: bad volume")
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])
    session = EvaluationSession.open(problem_file, [cell_count(), parameter_echo()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    data = json.loads(outcome.summary_path.read_text())
    Assessment.from_summary(session.spec, data)  # as written, it holds up
    # the sample's evaluation, blocked by its simulation, then the candidate's
    assert [e["status"] for e in data["samples"]["gamma"]["evaluations"].values()] == ["blocked"]
    assert [e["status"] for e in data["evaluations"].values()] == ["valid"]
    change(data, path, value)
    with pytest.raises(RecordError, match=re.escape(message)):
        Assessment.from_summary(session.spec, data)


def test_a_blocked_evaluator_that_measured_is_refused(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2)
    session = EvaluationSession.open(problem_file, [cell_count()])
    data = json.loads(session.evaluate({RADIUS: 450.0}, tmp_path / "a").summary_path.read_text())
    data["samples"]["gamma"]["evaluations"]["test.cell_count"]["measurements"] = [
        {"stage": "simulation", "metric": "ecal_barrel.mean_hit_cells", "value": 3.0}
        | {"unit": "1", "uncertainty": None}
    ]
    with pytest.raises(RecordError, match="it is blocked, so it was never called, yet it measured"):
        Assessment.from_summary(session.spec, data)


# --- the files of a candidate: named by reference, relative to the run, hashed where claimed ---


HISTOGRAM = "hit cells per event, JSON"


def plotting(**declared: str) -> type[Any]:
    """``test.cell_count``, writing the cells per event as the artifact ``histogram`` in its own
    directory, and declaring the artifacts ``declared`` (``histogram``, unless given)."""

    class Plotting(cell_count()):  # type: ignore[misc]
        artifacts = declared or {"histogram": HISTOGRAM}

        def evaluate(self, request: EvaluationRequest) -> Any:
            measured = result_from(self, super().evaluate(request))
            assert request.workdir is not None
            histogram = request.workdir / "histogram.json"
            histogram.write_text(json.dumps([3, 2, 4]))
            return dataclasses.replace(measured, artifacts={"histogram": histogram})

    return Plotting


def paths_in(value: Any, at: str = "") -> list[tuple[str, str]]:
    """Every text of a summary, with where it is (``samples/gamma/stages/simulation/argv``)."""
    if isinstance(value, dict):
        return [found for k, v in value.items() for found in paths_in(v, f"{at}/{k}")]
    if isinstance(value, list):
        return [found for item in value for found in paths_in(item, at)]
    return [(at, value)] if isinstance(value, str) else []


def test_a_record_stays_true_wherever_its_run_directory_is_moved(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [plotting()])
    made = tmp_path / "a"
    session.evaluate({RADIUS: 450.0}, made)
    summary = json.loads((made / SUMMARY_NAME).read_text())
    # the run directory once, where it was made; the exact ddsim command as it ran
    named = {at for at, text in paths_in(summary) if str(made) in text}
    assert named == {"/run_dir", "/samples/gamma/stages/simulation/argv"}
    moved = tmp_path / "archive" / "a"
    moved.parent.mkdir()
    shutil.move(made, moved)
    Assessment.from_summary(session.spec, summary)  # holds up as it is
    stages = summary["samples"]["gamma"]["stages"]
    (evaluation,) = summary["samples"]["gamma"]["evaluations"].values()
    files = [
        summary["geometry"]["compact"],
        summary["geometry"]["manifest"],
        stages["simulation"]["outputs"]["events"]["path"],
        *stages["simulation"]["logs"].values(),
        evaluation["artifacts"]["histogram"]["path"],
    ]
    assert all((moved / path).is_file() for path in files)  # each read from where it is now
    histogram = evaluation["artifacts"]["histogram"]
    assert (
        hashlib.sha256((moved / histogram["path"]).read_bytes()).hexdigest() == histogram["sha256"]
    )


def test_an_artifact_is_recorded_by_reference_path_size_and_hash(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[plotting()]
    )
    summary = json.loads(outcome.summary_path.read_text())
    (evaluation,) = summary["samples"]["gamma"]["evaluations"].values()
    path = (
        tmp_path / "a" / "samples" / "gamma" / "evaluators" / "test.cell_count" / "histogram.json"
    )
    assert evaluation["artifacts"] == {
        "histogram": {
            "ref": "gamma/test.cell_count#histogram",
            "path": "samples/gamma/evaluators/test.cell_count/histogram.json",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    # what the implementation declares, as the record of every binding of it is checked against
    assert summary["implementations"]["test.cell_count"]["artifacts"] == ["histogram"]
    (made,) = outcome.candidate.results[
        TaskId(Kind.EVALUATOR, "test.cell_count", "gamma")
    ].made.values()
    assert made.artifact == Artifact(
        "gamma/test.cell_count#histogram",
        path,
        path.stat().st_size,
        evaluation["artifacts"]["histogram"]["sha256"],
    )
    without = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "b")
    assert without.summary["implementations"]["test.cell_count"]["artifacts"] is None


def test_an_artifact_is_hashed_once_and_an_edm4hep_output_never(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed: list[str] = []

    def sha256(path: str, *key: int) -> str:
        hashed.append(path)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    monkeypatch.setattr(_files, "_sha256", sha256)  # uncached: every hash is counted
    run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[plotting()])
    artifact = tmp_path / "a" / "samples" / "gamma" / "evaluators" / "test.cell_count"
    assert hashed == [str(artifact / "histogram.json")]  # a gun reads no file


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        # a file the record names, changed after its task made it: the record would lie
        ("histogram", "gamma/test.cell_count#histogram changed after it was made"),
        ("events", "gamma/simulation#events changed after it was made"),
    ],
)
def test_a_file_that_changes_after_it_was_made_leaves_no_summary(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, changed: str, message: str
) -> None:
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])
    run_dir = tmp_path / "a"

    class Meddling(parameter_echo()):  # type: ignore[misc]
        """Measures the candidate after the sample was measured, and changes one of its files."""

        def evaluate(self, request: EvaluationRequest) -> Any:
            sample = run_dir / "samples" / "gamma"
            files = {
                "histogram": sample / "evaluators" / "test.cell_count" / "histogram.json",
                "events": sample / "simulation" / "events.edm4hep.root",
            }
            with files[changed].open("ab") as handle:
                handle.write(b"more")
            return super().evaluate(request)

    session = EvaluationSession.open(problem_file, [plotting(), Meddling])
    with pytest.raises(RunError, match=re.escape(message)):
        session.evaluate({RADIUS: 450.0}, run_dir)
    assert not (run_dir / SUMMARY_NAME).exists()


def test_a_hashed_file_whose_stat_changed_but_not_its_bytes_is_unchanged(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    run_dir = tmp_path / "a"
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])

    class Touching(parameter_echo()):  # type: ignore[misc]
        def evaluate(self, request: EvaluationRequest) -> Any:
            histogram = run_dir / "samples/gamma/evaluators/test.cell_count/histogram.json"
            histogram.write_bytes(histogram.read_bytes())  # rewritten as it was: hashed again
            return super().evaluate(request)

    outcome = EvaluationSession.open(problem_file, [plotting(), Touching]).evaluate(
        {RADIUS: 450.0}, run_dir
    )
    assert outcome.candidate.status == "valid" and outcome.summary_path.is_file()


VALID = "samples/gamma/evaluations/test.cell_count"


OUTPUT = "samples/gamma/stages/simulation/outputs/events"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        # a content identity only where one was computed
        (f"{VALID}/artifacts/histogram/sha256", None, "sha256 None, but such a file is hashed"),
        (f"{OUTPUT}/sha256", "a" * 64, "but such a file is never hashed"),
        (f"{VALID}/artifacts/histogram/sha256", "A" * 64, "sha256 must be 64 lowercase hex"),
        # each by its reference, its own and none other
        (f"{VALID}/artifacts/histogram/ref", "gamma/simulation#events", "ref 'gamma/simulation#"),
        (f"{OUTPUT}/ref", "gamma/simulation#output", "not 'gamma/simulation#events'"),
        (f"{OUTPUT}/bytes", -1, "bytes must be a size, got -1"),
        # relative to the run directory, in the task's own directory
        (f"{OUTPUT}/path", "/data/events.edm4hep.root", "is not a plain path relative to the run"),
        (f"{VALID}/artifacts/histogram/path", "samples/gamma/../x", "not a plain path relative"),
        (f"{VALID}/artifacts/histogram/path", "samples/gamma/x", "not in the task's own directory"),
        ("samples/gamma/stages/simulation/logs/stdout", "geometry/x.log", "not in the task's own"),
        ("geometry/compact", "/abs/compact.xml", "geometry: compact: '/abs/compact.xml' is not a"),
        ("samples/gamma/stages/simulation/directory", "elsewhere", "directory 'elsewhere', but"),
        (f"{VALID}/workdir", "evaluators/x", "workdir 'evaluators/x', but its directory is"),
        (f"{VALID}/workdir", None, "it is valid, so it was called, yet has no workdir"),
        ("run_dir", None, "run_dir must be a nonblank string"),
        # what a stage that is done made, and what an evaluator declares, the record holds
        ("samples/gamma/stages/simulation/outputs", {}, "outputs [], but a stage that is done"),
        (f"{VALID}/artifacts", {}, "it is valid without its declared artifacts ['histogram']"),
        ("implementations/test.cell_count/artifacts", ["plot"], "artifacts ['histogram'], which"),
        ("implementations/test.cell_count/artifacts", ["plot", "plot"], "names one twice"),
        ("implementations/test.cell_count/artifacts", ["a#b"], "'a#b' is no artifact name"),
    ],
)
def test_a_record_whose_files_are_not_as_made_is_refused(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    path: str,
    value: Any,
    message: str,
) -> None:
    session = EvaluationSession.open(problem_file, [plotting()])
    data = json.loads(session.evaluate({RADIUS: 450.0}, tmp_path / "a").summary_path.read_text())
    Assessment.from_summary(session.spec, data)  # as written, it holds up
    change(data, path, value)
    with pytest.raises(RecordError, match=re.escape(message)):
        Assessment.from_summary(session.spec, data)


def test_a_record_with_an_artifact_name_its_reference_would_split_is_refused(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    # declaring none, the evaluator's names are checked by the record alone
    session = EvaluationSession.open(problem_file, [cell_count()])
    data = json.loads(session.evaluate({RADIUS: 450.0}, tmp_path / "a").summary_path.read_text())
    Assessment.from_summary(session.spec, data)  # as written, it holds up
    plot = {
        "ref": "gamma/test.cell_count#plot#other",
        "path": "samples/gamma/evaluators/test.cell_count/plot.json",
        "bytes": 12,
        "sha256": "0" * 64,
    }
    change(data, f"{VALID}/artifacts", {"plot#other": plot})
    with pytest.raises(RecordError, match="'plot#other' is no artifact name"):
        Assessment.from_summary(session.spec, data)


def test_a_blocked_evaluation_has_no_directory_and_no_artifacts(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2)
    session = EvaluationSession.open(problem_file, [plotting()])
    data = json.loads(session.evaluate({RADIUS: 450.0}, tmp_path / "a").summary_path.read_text())
    Assessment.from_summary(session.spec, data)  # blocked: valid without its artifact is fine
    change(data, f"{VALID}/workdir", "samples/gamma/evaluators/test.cell_count")
    with pytest.raises(RecordError, match="it is blocked, so it was never called, yet has a work"):
        Assessment.from_summary(session.spec, data)


def test_a_geometry_changed_between_two_samples_simulates_no_further_and_leaves_no_summary(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    two_samples(problem_file)
    run_dir = tmp_path / "a"
    copied = run_dir / "geometry" / "compact" / "det_v1" / "tracker.xml"

    class Editing(cell_count()):  # type: ignore[misc]
        """Edits the candidate's geometry while it measures the first sample."""

        def evaluate(self, request: EvaluationRequest) -> Any:
            if not fake.calls[1:]:  # only gamma is simulated yet
                copied.write_text(copied.read_text().replace("450.0*mm", "500.0*mm"))
            return super().evaluate(request)

    session = EvaluationSession.open(problem_file, [Editing])
    message = "geometry/compact/det_v1/tracker.xml changed after it was made"
    with pytest.raises(RunError, match=re.escape(message)):
        session.evaluate({RADIUS: 450.0}, run_dir)
    # pion was not simulated with the edited geometry, and no record says one geometry was
    assert [particle(argv) for argv, _ in fake.calls] == ["gamma"]
    assert not (run_dir / SUMMARY_NAME).exists()


# --- diagnostics: evaluations no objective, constraint or required evaluator reads ----------


def diagnosed(problem_file: Path, **binding: Any) -> dict[str, Any]:
    """:func:`two_samples` with the objective on ``gamma`` alone, so that the evaluation of
    ``pion`` (with the stages only it reads) is a diagnostic unless ``binding`` says otherwise."""
    doc = two_samples(problem_file)
    doc["objectives"] = doc["objectives"][:1]
    doc["evaluators"][0] |= binding
    problem_file.write_text(yaml.safe_dump(doc))
    return doc


def crash_pions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = simulation.install(monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9]))

    def pions_crash(argv: list[str], **kwargs: Any) -> Any:
        code = fake(argv, **kwargs)
        return 2 if particle(argv) == "pi-" else code

    monkeypatch.setattr(ddsim_run, "_execute", pions_crash)


def test_a_diagnostic_that_does_not_succeed_is_recorded_and_leaves_the_candidate_valid(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crash_pions(monkeypatch)
    diagnosed(problem_file)
    session = EvaluationSession.open(problem_file, [cell_count()])
    assert session.diagnostics == ("pion/test.cell_count",)
    assert session.notice == (
        "diagnostics (recorded, do not affect the result): pion/test.cell_count; mark an "
        "evaluator required: true if its result must count"
    )
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    record = outcome.candidate
    assert (record.status, record.message) == ("valid", None)
    assert record.record("pion/test.cell_count").result.status == "blocked"
    assert outcome.assessment.feasible is True
    assert [o.value for o in outcome.assessment.objectives or ()] == [3.0]
    written = json.loads(outcome.summary_path.read_text())
    assert written["required"] == [
        "geometry",
        "gamma/generation",
        "gamma/simulation",
        "gamma/test.cell_count",
    ]
    (crashed, held) = written["diagnostics"]
    assert (crashed["task"], crashed["status"]) == ("pion/simulation", "failed")
    assert crashed["message"].startswith("ddsim exited with status 2")
    assert held == {
        "task": "pion/test.cell_count",
        "status": "blocked",
        "message": "not run: pion/simulation failed",
    }
    assert Assessment.from_summary(session.spec, written) == outcome.assessment


def test_the_command_exits_zero_and_warns_of_each_diagnostic_that_did_not_succeed(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crash_pions(monkeypatch)
    diagnosed(problem_file)
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 0, err
    assert "status:      valid" in out
    assert "evaluator:   test.cell_count 1 on gamma: valid\n" in out
    assert "evaluator:   test.cell_count 1 on pion: blocked (diagnostic)" in out
    # told before anything runs, then what each that did not succeed gave
    notice = err.index("k4opt: warning: diagnostics (recorded, do not affect the result)")
    assert notice < err.index("k4opt: warning: diagnostic pion/simulation failed: ddsim exited")
    assert (
        "k4opt: warning: diagnostic pion/test.cell_count blocked: not run: pion/simulation failed"
        in err
    )
    assert "k4opt: evaluation" not in err


def test_an_evaluator_marked_required_decides_the_status_on_every_sample_it_measures(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crash_pions(monkeypatch)
    diagnosed(problem_file, required=True)
    session = EvaluationSession.open(problem_file, [cell_count()])
    assert (session.diagnostics, session.notice) == ((), None)
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    record = outcome.candidate
    assert record.status == "failed" and record.diagnostics == ()
    assert (record.message or "").endswith("; blocked: pion/test.cell_count")
    assert outcome.assessment.objectives is None
    written = json.loads(outcome.summary_path.read_text())
    assert written["diagnostics"] == [] and "pion/simulation" in written["required"]
    Assessment.from_summary(session.spec, written)
    code, _, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "b")
    )
    assert code == 1 and "k4opt: evaluation failed: pion/simulation failed" in err


def test_a_stage_supplied_for_a_diagnostic_alone_leaves_the_record_representative(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    diagnosed(problem_file)
    events_file = tmp_path / "earlier.edm4hep.root"
    events_file.write_bytes(b"fake")
    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450"], tmp_path / "a", inputs=[f"pion/simulation={events_file}"]
    )
    summary = outcome.summary
    assert (summary["represents_candidate"], summary["supplied"]) == (True, ["pion/simulation"])
    assert outcome.assessment.feasible is True  # judged: the objective's events are its own
    written = json.loads(outcome.summary_path.read_text())
    Assessment.from_summary(load_problem(problem_file), written)
    # the same file read by a required evaluation is no evaluation of the candidate
    diagnosed(problem_file, required=True)
    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450"], tmp_path / "b", inputs=[f"pion/simulation={events_file}"]
    )
    assert outcome.summary["represents_candidate"] is False
    assert outcome.assessment.feasible is None
