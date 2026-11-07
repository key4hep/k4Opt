"""Internal: a candidate's ``summary.json``, made from its task results alone.

:func:`assemble` makes the record: each task's part from that task's result, and what the
candidate's status, message, required tasks, diagnostics and metrics derive from them, as
:meth:`~k4opt.record.Assessment.from_summary` derives them again. Every path the record holds
of a file in the run directory is relative to it, so a record stays true wherever the directory
is moved; the run directory itself is recorded once, as ``run_dir``. :func:`peak_memory` reads
back what a record says of the memory its tasks took, :func:`observed_stages` what its stages used
and :func:`unused_local` the local builds none of them read.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from k4opt import __version__
from k4opt.evaluation import Artifact, Scalar
from k4opt.frozen import Json, thaw
from k4opt.problem.candidate import Configuration
from k4opt.problem.models import EvaluatorSpec, ProblemSpec, Stage
from k4opt.problem.plan import GEOMETRY, Kind, Task, evaluations
from k4opt.record.assess import Assessment
from k4opt.record.errors import RecordError
from k4opt.record.read import SUMMARY_SCHEMA, _measurement, _within
from k4opt.record.results import DONE, CandidateRecord, PeakMemory, TaskResult
from k4opt.software import Software


def _relative(path: Path, run_dir: Path) -> str:
    """How the record names ``path``, a file in the run directory ``run_dir``: relative to it."""
    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def _artifact(artifact: Artifact, run_dir: Path) -> dict[str, object]:
    """A file a task made, as the record names it; ``sha256`` is ``null`` unless it was hashed."""
    return {
        "ref": artifact.ref,
        "path": _relative(artifact.path, run_dir),
        "bytes": artifact.bytes,
        "sha256": artifact.sha256,
    }


def _memory(result: TaskResult) -> dict[str, object]:
    """A task's peak memory as its record says it, in bytes: of the k4opt process that did its
    work (``process``) and of the program a stage ran (``program``); ``null`` for one that did
    not run. How it ran, never what was evaluated."""
    return {"peak_memory_bytes": None if result.memory is None else result.memory.to_dict()}


def _fragment(result: TaskResult) -> dict[str, object]:
    """What every task's part of the summary says first: how it ended, and why."""
    return {"status": result.status, "message": result.message}


def _reads(task: Task, result: TaskResult) -> dict[str, object]:
    """What every task's part of the summary says of the tasks before it: those it reads, those
    of them that kept it from running, and the parameters whose values reach it."""
    return {
        "reads": [need.label for need in task.needs],
        "blocked_by": [cause.label for cause in result.blocked_by],
        "varies": list(task.varies),
    }


def _stage(task: Task, result: TaskResult, stage: Stage, run_dir: Path) -> dict[str, object]:
    """One stage's part of the summary: how it ended, how the candidate configured it
    (``definition``, with what its parameters set), how it ran, where, and what it made."""
    provenance = thaw(result.provenance)
    assert isinstance(provenance, dict)
    return (
        _fragment(result)
        | {"definition": stage.to_dict()}
        | provenance
        | {
            "directory": task.directory,
            "outputs": {name: _artifact(made, run_dir) for name, made in result.made.items()},
            "logs": {role: _relative(path, run_dir) for role, path in result.artifacts.items()},
        }
        | _reads(task, result)
        | {"elapsed_seconds": result.elapsed_seconds}
        | _memory(result)
    )


def _evaluation(
    task: Task, result: TaskResult, entry: EvaluatorSpec, run_dir: Path
) -> dict[str, object]:
    """One evaluation's own part of the summary: nothing of the stages it shares, nor of its
    implementation, which ``implementations`` records once, nor of what its roles read, which is
    recorded by the tasks that made it."""
    record = result.record
    assert record is not None  # every binding has one, a blocked one too
    outcome, request = record.result, record.request
    # what each role of an evaluation of the candidate reads, by the task that made it
    roles = {role: [item.label for item in items] for role, items in entry.roles.items()}
    return {
        "implementation": entry.name,
        **_fragment(result),
        # as the problem writes them for this candidate (with what its parameters set), and as
        # the evaluator received them (defaults filled in, quantities in the declared unit)
        "settings": {"declared": thaw(entry.settings), "used": thaw(request.settings)},
        "collections": dict(request.collections),
        **({} if task.id.sample is not None else {"roles": roles}),
        **_reads(task, result),
        "measurements": [
            {"stage": m.stage, "metric": m.metric} | _measurement(m) for m in outcome.measurements
        ],
        "artifacts": {name: _artifact(made, run_dir) for name, made in result.made.items()},
        "provenance": thaw(outcome.provenance),
        # its own directory, made just before it was called; none if it was not
        "workdir": None if request.workdir is None else task.directory,
        # what it printed while it was called in a worker process; none in the caller's process
        "log": _relative(log, run_dir) if (log := result.artifacts.get("log")) else None,
        "elapsed_seconds": record.elapsed_seconds,
        **_memory(result),
    }


def assemble(
    record_id: str,
    problem_path: Path,
    spec: ProblemSpec,
    values: Mapping[str, Scalar],
    candidate: CandidateRecord,
    assessment: Assessment,
    run_dir: Path,
    taken: Mapping[str, Mapping[str, Json]],
    run: object,
    process: Mapping[str, Json],
    configuration: Configuration,
    execution: Mapping[str, Json],
    replicate: int = 0,
    level: int = 0,
) -> dict[str, object]:
    """The summary of a candidate, made from its task results and nothing else that ran.

    Each task's part (``geometry``; each sample's ``stages`` and ``evaluations``; the
    ``evaluations`` of the candidate) is made from that task's result alone; the candidate's
    ``status``, ``message``, ``required``, ``diagnostics``, ``metrics`` and ``assessment`` derive
    from them and the plan, as
    :meth:`Assessment.from_summary` derives them again.
    ``record_id`` is the record's identity; the rest is what the session holds: the problem,
    what it recorded of each implementation (``taken``: its version, stage, source and metrics)
    and of the process (``process``), the candidate's ``configuration`` (each stage and each
    evaluator's settings as its parameters set them) and how it executed the tasks
    (``execution``: provenance, which no identity or comparison of records reads), and the
    ``replicate`` its seeds are of (:func:`k4opt.problem.configure`); the ``level`` of the
    optimization it was evaluated for decides which metrics it names (its criteria:
    :meth:`~k4opt.problem.ProblemSpec.criteria`), as the ``assessment`` and the plan's required
    tasks were made for it.
    """
    plan, results = candidate.plan, candidate.results
    environment = thaw(process["environment"])
    assert isinstance(environment, dict)
    entries = dict(evaluations(configuration.samples, configuration.evaluators))
    geometry = results[GEOMETRY]

    def measured(sample: str | None) -> dict[str, object]:
        """The evaluations of ``sample``, or of the candidate, by binding."""
        return {
            task.id.name: _evaluation(task, results[task.id], entries[task.id], run_dir)
            for task in plan.of(Kind.EVALUATOR)
            if task.id.sample == sample
        }

    made = thaw(geometry.provenance)
    assert isinstance(made, dict)
    return {
        "schema": SUMMARY_SCHEMA,
        "id": record_id,  # the record's identity, wherever it is copied to
        "run": run,  # what the caller evaluates it for (a study and trial, say), null if nothing
        # where it was made, which the paths of its files are relative to; it may move since
        "run_dir": str(run_dir),
        "problem": {"path": str(problem_path), "name": spec.name, "definition": spec.to_dict()},
        "candidate": dict(values),
        "candidate_key": [list(part) for part in candidate.candidate_key],
        # 0, or the replicate whose seeds its stages ran with: the same candidate on other events
        "replicate": replicate,
        # 0, or the level of a nested optimization it was evaluated for: what it is judged by
        "level": level,
        "status": candidate.status,
        "message": candidate.message,
        # the tasks the status is decided by: what the objectives, constraints and the evaluators
        # marked required read; the others are diagnostics
        "required": [task.id.label for task in plan.tasks if task.required],
        # each diagnostic that did not succeed, which the status leaves out
        "diagnostics": [
            {"task": result.id.label, "status": result.status, "message": result.message}
            for result in candidate.diagnostics
        ],
        # False if a required stage file was supplied that was not made from this candidate
        "represents_candidate": candidate.represents_candidate,
        "supplied": sorted(candidate.supplied),
        "warnings": list(candidate.warnings),
        # the objective and constraint metrics, each from the binding it names
        "metrics": [
            {"evaluator": ref.evaluator, "sample": ref.sample, "stage": ref.stage}
            | {"metric": ref.metric, "label": ref.label}
            | _measurement(candidate.value(ref))
            for ref in spec.criteria(level).references
        ],
        # the objectives and the constraints judged, from those metrics
        "assessment": assessment.to_dict(),
        # each implementation once, however many bindings it has
        "implementations": {name: thaw(entry) for name, entry in taken.items()},
        # each sample's stages, each recorded once, and the evaluations of the sample
        "samples": {
            sample: {
                "stages": {
                    task.id.name: _stage(
                        task,
                        results[task.id],
                        configuration.samples[sample].stages[task.id.name],
                        run_dir,
                    )
                    for task in plan.of(Kind.STAGE)
                    if task.id.sample == sample
                },
                "evaluations": measured(sample),
            }
            for sample in spec.samples
        },
        # the evaluations of the candidate, of no sample
        "evaluations": measured(None),
        "elapsed_seconds": candidate.elapsed_seconds,
        # what a geometry that was made changed and reads; one that was not, why
        "geometry": {"status": geometry.status}
        | (made if geometry.status == DONE else {"message": geometry.message})
        | {"varies": list(plan[GEOMETRY].varies), "elapsed_seconds": geometry.elapsed_seconds}
        | _memory(geometry),
        "environment": {
            # observed in this process; the programs are the executables every stage ran
            "observed": {
                "k4opt": __version__,
                # what the version only labels: the content of its code, and its checkout
                "k4opt_source": thaw(process["k4opt"]),
                # the session's, which the process still had when this candidate started
                **environment,
                "programs": thaw(process["programs"]),
                # one digest of the variables: study.json, or the session, names each
                "software": Software.from_dict(process["software"]).to_dict(variables=False),
                # where and how the tasks ran: provenance, never part of what was evaluated
                "execution": thaw(execution),
            },
        },
    }


def peak_memory(summary: Mapping[str, object]) -> dict[str, PeakMemory]:
    """The peak memory of each task a ``summary.json`` records, by the task's label
    (``geometry``, ``gamma/simulation``, ``gamma/calo.photon``, ``pi0_rejection``), for the tasks
    that ran; read as the record holds it, in bytes. Raises
    :class:`~k4opt.record.RecordError` for one that is not a peak memory."""
    entries: dict[str, object] = {"geometry": summary.get("geometry")}
    samples = summary.get("samples")
    for sample, part in (samples if isinstance(samples, Mapping) else {}).items():
        for kind in ("stages", "evaluations"):
            tasks = part.get(kind) if isinstance(part, Mapping) else None
            for name, entry in (tasks if isinstance(tasks, Mapping) else {}).items():
                entries[f"{sample}/{name}"] = entry
    evaluations = summary.get("evaluations")
    entries |= dict(evaluations) if isinstance(evaluations, Mapping) else {}
    found: dict[str, PeakMemory] = {}
    for label, entry in entries.items():
        value = entry.get("peak_memory_bytes") if isinstance(entry, Mapping) else None
        if value is None:
            continue
        if not isinstance(value, Mapping) or set(value) != {"process", "program"}:
            raise RecordError(f"{label}: peak_memory_bytes is no process and program: {value!r}")
        for key, count in value.items():
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise RecordError(f"{label}: peak_memory_bytes.{key} is no number of bytes")
        found[label] = PeakMemory(value["process"], value["program"])
    return found


def observed_stages(summary: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """What a ``summary.json`` says each stage that ran a program used
    (:meth:`k4opt.software.Footprint.summary`), by the stage's label (``gamma/simulation``); a
    stage whose output was reused ran nothing and has none."""
    return {
        f"{sample}/{name}": observed
        for sample in _within(summary, "samples")
        for name in _within(summary, "samples", sample, "stages")
        if (observed := _within(summary, "samples", sample, "stages", name, "observed"))
    }


def unused_local(summary: Mapping[str, object]) -> list[str]:
    """The local entries of the software that no stage of a ``summary.json`` read from: a local
    build put in front of the stack that did not run. Empty where that is not known: unless every
    stage ran its program itself, observed at the ``files`` level."""
    observed = observed_stages(summary)
    for sample in _within(summary, "samples"):
        for name in _within(summary, "samples", sample, "stages"):
            stage = _within(summary, "samples", sample, "stages", name)
            if "reused" in stage:
                return []
            if "argv" in stage and observed[f"{sample}/{name}"].get("level") != "files":
                return []
    used = {path for entry in observed.values() for path in cast(Any, entry.get("local") or ())}
    software = _within(summary, "environment", "observed", "software")
    return sorted({entry["path"] for entry in cast(Any, software.get("local") or ())} - used)


SUMMARY_NAME = "summary.json"
