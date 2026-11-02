"""Internal: the work of one task, done from a description that holds no local object.

What a task does is the same wherever it runs: in the process that coordinates the candidates,
in a worker process of the same machine, or (later) in a batch job. :func:`execute_task` is that
work, for each kind of task: it materializes the candidate's geometry, makes a stage of a sample
(:func:`~k4opt.problem._stages.make_stage`), or calls an evaluator and checks what it returned.
It reads nothing of the session that coordinates it; whatever it needs comes in three parts:

* a :class:`SessionContext`, the same for every task of a session: the problem, the stage
  conditions of each sample and what each evaluation must measure;
* a :class:`TaskSpec`, the task itself, plain data: the planned task, the files it reads by
  their place in the candidate's workspace, and the request of an evaluation;
* the root of the candidate's workspace (:class:`Workspace`) on the machine that runs it, and
  the :class:`Observations` taken of the files its reads made.

Paths come in two kinds. A file a task made is named relative to its candidate's workspace, so
that a description stays true wherever the workspace is; an input from outside it (a stage file
of the problem, a file the caller supplied, the geometry's source) keeps its absolute path,
which the session identified by its content. The ``stat`` of a file (its size, times and inode)
tells a rewritten file apart only on the filesystem that took it: it is an
:class:`Observations`, handed along with the description by an executor that shares the
filesystem, never part of what a task returns (:class:`TaskOutcome`).

What a task returns is portable in the same way; the coordinator turns it into the
:class:`~k4opt.problem._tasks.TaskResult` its records are assembled from (:func:`to_result`).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path

from k4opt.evaluation import (
    Artifact,
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorProtocol,
    RoleInput,
    Status,
    available_collections,
    collection_types,
    declared_collections,
    result_from,
)
from k4opt.frozen import FrozenMap, Json, thaw
from k4opt.geometry import GeometryError, GeometrySource, Override
from k4opt.problem._files import (
    Made,
    Stat,
    Tree,
    artifacts,
    local,
    local_artifact,
    portable,
    portable_artifact,
    stat_of,
)
from k4opt.problem._stages import StageContext, make_stage, stage_output
from k4opt.problem._tasks import DONE, GEOMETRY, Kind, PeakMemory, Task, TaskId, TaskResult
from k4opt.problem.errors import RunError
from k4opt.problem.models import EvaluatorSpec, ProblemSpec

SUMMARY_NAME = "summary.json"

type MetricKey = tuple[str, str]
"""A metric as an evaluator declares and measures it: ``(stage, metric)``."""

type Output = Callable[[Path], AbstractContextManager[object]]
"""Where an evaluator's output goes while it is called, given its own directory."""


@dataclass(frozen=True, slots=True)
class Workspace:
    """Where the tasks of one candidate write: ``id`` names it wherever it is (the record id of
    the candidate), ``root`` is its directory on the coordinator's machine, ``None`` for a
    candidate none of whose tasks writes a file. A task description names only the ``id``;
    whoever runs the task knows where the workspace is."""

    id: str
    root: Path | None


@dataclass(frozen=True, slots=True)
class Measuring:
    """What the work of an evaluation needs besides its request: the ``entry`` that binds it, the
    ``metrics`` and ``artifacts`` its implementation declared when bound (``None`` where it
    declares none), and the metrics the roles of combined evaluators ``demanded`` of it."""

    entry: EvaluatorSpec
    metrics: Mapping[MetricKey, str] | None
    artifacts: Mapping[str, str] | None
    demanded: tuple[MetricKey, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionContext:
    """What every task of a session needs: the problem ``spec`` (with its geometry source), the
    stage conditions of each sample (``contexts``, the session's time limit included), what each
    evaluation must measure (``evaluations``) and what the external references of the session's
    geometry read (``externals``), which a candidate's own references are compared with."""

    spec: ProblemSpec
    contexts: Mapping[str, StageContext]
    evaluations: Mapping[TaskId, Measuring]
    externals: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    # the collection names of each stage file, by path and stat, so that a file several
    # evaluations read is listed once; a cache of this process, never part of a description,
    # which keeps the files of the last few candidates only
    listed: dict[tuple[Path, Stat | None], tuple[str, ...] | str] = field(
        default_factory=dict, compare=False
    )


@dataclass(frozen=True, slots=True)
class GeometryWork:
    """How the geometry task makes the candidate's geometry: by applying ``overrides`` to the
    session's geometry, or by taking the compact file ``given``, already prepared (named by its
    absolute path: it is an input from outside the workspace)."""

    overrides: tuple[Override, ...] = ()
    given: Path | None = None


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One task of a candidate, as plain data.

    ``workspace`` is the id of the candidate's :class:`Workspace`, ``task`` the planned task.
    ``inputs`` are the files the task is given, by name (``geometry``, a stage); ``watched`` the
    files its reads made, and ``tree`` the directory of the geometry it reads (``None`` if it
    reads none made in the workspace): what it checks before and after it runs. ``geometry`` says
    how the geometry task works (``None``: there is no geometry), ``supplied`` is a file the
    caller gives for a stage, and ``request`` and ``roles`` are what an evaluation is asked, its
    request without inputs or a directory. ``cpus`` is how many processors the task keeps busy.
    """

    workspace: str
    task: Task
    inputs: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    watched: tuple[Artifact, ...] = ()
    tree: Path | None = None
    geometry: GeometryWork | None = None
    supplied: Path | None = None
    request: EvaluationRequest | None = None
    roles: Mapping[str, tuple[RoleInput, ...]] = field(default_factory=lambda: FrozenMap({}))
    cpus: int = 1


@dataclass(frozen=True, slots=True)
class Observations:
    """The ``stat`` of files, valid only on the filesystem that took them: of what a task made, by
    the reference of each file (``made``), and of every file of the geometry it made, by its path
    below the geometry's directory (``tree``)."""

    made: Mapping[str, Stat] = field(default_factory=lambda: FrozenMap({}))
    tree: Mapping[str, Stat] | None = None


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """What a task left behind, each file in it named relative to its workspace (or by its
    absolute path, for an input from outside).

    As :class:`~k4opt.problem._tasks.TaskResult` describes it, without the record of an
    evaluation, whose request the coordinator made: ``result`` is what the evaluator returned (its
    artifacts relative to the workspace), ``workdir`` the evaluation's own directory if it was
    called, and ``call_seconds`` the time of its call. ``own_refs`` says whether the geometry a
    geometry task made has ``ref`` attributes the candidate changed. ``memory`` is its peak memory
    where it ran.

    Only the files are named so that they hold wherever the workspace is. ``provenance`` says how
    the task ran where it ran (the command of a stage, with the absolute paths it was given) and
    is kept as it is: it records what ran, not where to find a file.
    """

    task: TaskId
    status: str
    message: str | None = None
    output: Path | None = None
    artifacts: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    provenance: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    elapsed_seconds: float | None = None
    made: Mapping[str, Artifact] = field(default_factory=lambda: FrozenMap({}))
    result: EvaluationResult | None = None
    workdir: Path | None = None
    call_seconds: float | None = None
    own_refs: bool = False
    memory: PeakMemory | None = None


def candidate_externals(
    compact: str, output_root: Path, own_refs: bool, externals: Mapping[str, Json]
) -> dict[str, dict[str, Json]]:
    """What the external references of a candidate's geometry, made in ``output_root``, read that
    those of the session's (``externals``) do not: a ``ref`` the candidate changes (``own_refs``)
    can name another file (``${CAD}/other.stl``)."""
    if not own_refs:
        return {}
    try:
        now = GeometrySource(compact, root=output_root).external_files()
    except GeometryError as exc:
        raise RunError(f"geometry: {exc}") from exc
    opened = thaw(externals)
    assert isinstance(opened, dict)
    return {ref: entry for ref, entry in now.items() if content(entry) != content(opened.get(ref))}


def content(entry: object) -> Json:
    """What an external reference reads (:func:`~k4opt.geometry.includes.external_files`),
    without where the files are: the same files in another snapshot read alike."""
    if not isinstance(entry, Mapping):
        return None
    return {key: value for key, value in entry.items() if key != "path"}


def execute_task(
    context: SessionContext,
    spec: TaskSpec,
    root: Path | None,
    seen: Observations,
    instance: EvaluatorProtocol | None = None,
    output: Output | None = None,
    *,
    make: Callable[[], EvaluatorProtocol] | None = None,
) -> tuple[TaskOutcome, Observations]:
    """Do the work of the task ``spec`` in the workspace whose directory is ``root`` here
    (``None``: the candidate has no directory, so no task of it writes).

    ``seen`` holds the stats taken of the files the task's reads made, which it checks before and
    after it runs. An evaluation calls ``instance``, within ``output`` of its own directory if
    given (a worker process points the evaluator's output at a log there), or the instance
    ``make`` returns, which the task makes first (a worker: a new instance, checked), so that
    making it counts as the task's work. Returns what the task left behind and the
    stats of what it made. Raises what the work of the task raises: a geometry that cannot be
    made, an :class:`~k4opt.evaluation.EvaluationContractError` for a result that breaks the
    evaluation contract, or any exception of an evaluator other than
    :class:`~k4opt.evaluation.EvaluationFailed` and :class:`~k4opt.evaluation.EvaluationInvalid`.
    The outcome holds the time and the peak memory of this process while it did the task, the
    making of its instance included, and the peak memory of the program a stage ran.
    """
    began, measuring = time.perf_counter(), _reset_peak()
    if make is not None:
        instance = make()
    outcome, observed = _run_task(context, spec, root, seen, instance, output, began)
    program = None if outcome.memory is None else outcome.memory.program
    process = _process_peak() if measuring else None
    return replace(outcome, memory=PeakMemory(process, program)), observed


_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")


def _reset_peak() -> bool:
    """Start this process's peak resident memory afresh, at what it holds now (Linux); whether
    it could."""
    try:
        _CLEAR_REFS.write_text("5")  # 5: reset the peak RSS (proc(5))
    except OSError:
        return False
    return True


def _process_peak() -> int | None:
    """This process's peak resident memory in bytes since :func:`_reset_peak` (``VmHWM``)."""
    try:
        lines = _STATUS.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("VmHWM:"):
            value, unit = line.split()[1:3]
            return int(value) * 1024 if unit == "kB" else None
    return None


def _run_task(
    context: SessionContext,
    spec: TaskSpec,
    root: Path | None,
    seen: Observations,
    instance: EvaluatorProtocol | None,
    output: Output | None,
    began: float,
) -> tuple[TaskOutcome, Observations]:
    kind = spec.task.id.kind
    if kind is Kind.GEOMETRY:
        return _geometry(context, spec, root, began)
    if kind is Kind.STAGE:
        return _stage(context, spec, root, seen, began)
    assert instance is not None
    return _evaluation(context, spec, root, seen, instance, began, output)


def _since(started: float) -> float:
    return time.perf_counter() - started


def _own(spec: TaskSpec, root: Path | None) -> Path | None:
    return None if root is None or spec.task.directory is None else root / spec.task.directory


def _geometry(
    context: SessionContext, spec: TaskSpec, root: Path | None, began: float
) -> tuple[TaskOutcome, Observations]:
    task, work, own = spec.task.id, spec.geometry, _own(spec, root)
    if work is None:
        return TaskOutcome(task, DONE), Observations()
    if work.given is not None:
        # every simulation of every sample reads these files: watched like what a stage made
        tree = Tree.of(own) if own is not None and own.is_dir() else None
        given = portable(work.given, root)
        outcome = TaskOutcome(task, DONE, output=given, elapsed_seconds=_since(began))
        return outcome, Observations(tree=None if tree is None else tree.stats)
    source, catalog = context.spec._source, context.spec._catalog
    assert source is not None
    assert root is not None
    assert own is not None
    # materialize() validates the location (not inside the geometry source) before it creates
    # the workspace's directory, so a rejected location leaves nothing behind
    try:
        snapshot = source.materialize(work.overrides, own, catalog=catalog)
    except BaseException:
        with contextlib.suppress(OSError):
            root.rmdir()  # created by the failed attempt; only removed while empty
        raise
    refs = any(change.attribute == "ref" and change.changed for change in snapshot.changes)
    try:
        externals = candidate_externals(
            source.compact, snapshot.output_root, refs, context.externals
        )
    except RunError as exc:
        raise RunError(f"{exc} (in {root}, which has no {SUMMARY_NAME})") from None
    provenance: dict[str, Json] = {
        "fingerprint": snapshot.fingerprint,
        "compact": portable(snapshot.compact_path, root).as_posix(),
        "manifest": portable(snapshot.manifest_path, root).as_posix(),
        "source_root": str(snapshot.source_root),
        "changes": tuple(
            {"target": c.target, "old_value": c.old_value, "new_value": c.new_value}
            for c in snapshot.changes
        ),
        "externals": context.externals,
        "candidate_externals": externals,
    }
    tree = Tree.of(own) if own.is_dir() else None
    outcome = TaskOutcome(
        task,
        DONE,
        output=portable(snapshot.compact_path, root),
        provenance=provenance,
        elapsed_seconds=_since(began),
        own_refs=refs,
    )
    return outcome, Observations(tree=None if tree is None else tree.stats)


def changed_reads(spec: TaskSpec, root: Path | None, seen: Observations) -> list[str]:
    """The files the task's reads made that changed since: the reference of each, or for the
    geometry its path below the task (``geometry/compact/ECalBarrel.xml``)."""
    changed = []
    if spec.tree is not None and seen.tree is not None:
        tree = Tree(local(spec.tree, root), seen.tree)
        changed += [f"{GEOMETRY.label}/{name}" for name in tree.changed()]
    for artifact in spec.watched:
        made = Made(local_artifact(artifact, root), seen.made[artifact.ref])
        if made.changed():
            changed.append(artifact.ref)
    return changed


def _changed_before(refs: Sequence[str]) -> str:
    return f"not run: {', '.join(refs)} changed after it was made"


def _changed_during(refs: Sequence[str]) -> str:
    return f"{', '.join(refs)} changed while it ran, so the record cannot say what it read"


def _inputs(spec: TaskSpec, root: Path | None) -> dict[str, Path]:
    return {name: local(path, root) for name, path in spec.inputs.items()}


def _stage(
    context: SessionContext, spec: TaskSpec, root: Path | None, seen: Observations, began: float
) -> tuple[TaskOutcome, Observations]:
    task, own = spec.task.id, _own(spec, root)
    assert task.sample is not None
    changed = changed_reads(spec, root, seen)
    if changed:
        return TaskOutcome(task, Status.FAILED, _changed_before(changed)), Observations()
    outcome, main = make_stage(
        context.contexts[task.sample],
        task.name,
        inputs=_inputs(spec, root),
        workdir=own,
        supplied=spec.supplied,
    )
    error = outcome.error
    made: dict[str, Artifact] = {}
    stats: dict[str, Stat] = {}
    if error is None and main is not None and spec.task.given is None:
        name = stage_output(task.name)  # made in its directory: never hashed
        try:
            file = Made.file(task, name, main, hashed=False)
        except OSError as exc:
            error, main = f"cannot read its output: {exc}", None
        else:
            made[name] = portable_artifact(file.artifact, root)
            stats[file.artifact.ref] = file.stat
    changed = changed_reads(spec, root, seen)
    if changed and error is None:
        error, main, made, stats = _changed_during(changed), None, {}, {}
    return TaskOutcome(
        task,
        DONE if error is None else Status.FAILED,
        error,
        None if main is None else portable(main, root),
        {role: portable(path, root) for role, path in outcome.artifacts.items()},
        outcome.provenance,
        _since(began),
        made=made,
        memory=PeakMemory(program=outcome.peak_memory_bytes),
    ), Observations(made=stats)


def _evaluation(
    context: SessionContext,
    spec: TaskSpec,
    root: Path | None,
    seen: Observations,
    evaluator: EvaluatorProtocol,
    began: float,
    output: Output | None = None,
) -> tuple[TaskOutcome, Observations]:
    """An evaluator called once on the files the task reads, which all succeeded, and the
    artifacts it made, hashed once, when it returns.

    An evaluator whose stage's file lacks a collection it declares (or holds it with another
    type), or reads a file that changed since its task made it, gets a ``failed`` result saying
    so and is not called. Its own directory is created just before the call; ``call_seconds``
    is the time of the call alone. A file it reads that changes during the call fails it too:
    the record could not say what it read.
    """
    task = spec.task.id
    measuring = context.evaluations[task]
    request = asked(spec, root)

    def ended(result: EvaluationResult) -> tuple[TaskOutcome, Observations]:
        """A result of an evaluator that was not called."""
        check_result(context, task, measuring, evaluator.name, result)
        return TaskOutcome(
            task, result.status, result.message, elapsed_seconds=_since(began), result=result
        ), Observations()

    changed = changed_reads(spec, root, seen)
    if changed:
        return ended(EvaluationResult.failed(_changed_before(changed)))
    error = collection_error(evaluator, request, context.listed)
    if error is not None:
        return ended(EvaluationResult.failed(error))
    workdir = _own(spec, root)
    if workdir is not None:
        try:
            workdir.mkdir(parents=True)
        except OSError as exc:  # not called, so its request has no directory
            return ended(
                EvaluationResult.failed(f"cannot create evaluator directory {workdir}: {exc}")
            )
    request = replace(request, workdir=workdir)
    started = time.perf_counter()
    surrounding = contextlib.nullcontext() if output is None or workdir is None else output(workdir)
    try:
        with surrounding:
            result = result_from(evaluator, evaluator.evaluate(request))
    except EvaluationFailed as exc:
        result = EvaluationResult.failed(str(exc))
    except EvaluationInvalid as exc:
        result = EvaluationResult.invalid(str(exc))
    elapsed = time.perf_counter() - started
    check_result(context, task, measuring, evaluator.name, result)
    # what it declared when bound, which the record holds, not what it declares now
    made = artifacts(task, measuring.artifacts, result, workdir)
    changed = changed_reads(spec, root, seen)
    if changed:  # what it returned stays, as diagnostics of a failed result
        result = EvaluationResult.failed(
            _changed_during(changed),
            result.measurements,
            artifacts=result.artifacts,
            provenance=result.provenance,
        )
    returned = {name: portable(path, root) for name, path in result.artifacts.items()}
    return TaskOutcome(
        task,
        result.status,
        result.message,
        elapsed_seconds=_since(began),
        made={name: portable_artifact(file.artifact, root) for name, file in made.items()},
        result=replace(result, artifacts=returned),
        workdir=None if workdir is None else portable(workdir, root),
        call_seconds=elapsed,
    ), Observations(made={file.artifact.ref: file.stat for file in made.values()})


def asked(spec: TaskSpec, root: Path | None) -> EvaluationRequest:
    """The request the evaluation ``spec`` gets, its files where the workspace ``root`` is here,
    before its own directory exists."""
    assert spec.request is not None
    return replace(spec.request, inputs=_inputs(spec, root), roles=_roles(spec, root))


def _roles(spec: TaskSpec, root: Path | None) -> dict[str, tuple[RoleInput, ...]]:
    """What each role of a combined evaluator reads, its files where the workspace is here."""
    return {
        role: tuple(
            replace(
                item,
                artifacts={
                    name: local_artifact(artifact, root)
                    for name, artifact in item.artifacts.items()
                },
            )
            for item in items
        )
        for role, items in spec.roles.items()
    }


def show(metrics: Sequence[MetricKey] | Mapping[MetricKey, object]) -> str:
    return ", ".join(f"{stage}/{metric}" for stage, metric in metrics)


def check_units(
    name: str, declared: Mapping[MetricKey, str] | None, result: EvaluationResult
) -> None:
    """Every measurement of a metric the evaluator ``name`` ``declared`` must carry the declared
    unit.

    Whatever the status: diagnostic values of an ``invalid`` result are recorded too. An
    evaluator that declares no ``metrics``, and metrics it does not declare, are not checked.
    """
    if declared is None:
        return
    for m in result.measurements:
        unit = declared.get((m.stage, m.metric))
        if (m.stage, m.metric) in declared and m.unit != unit:
            raise EvaluationContractError(
                f"measurement {m.stage}/{m.metric} has unit {m.unit!r} but evaluator "
                f"{name!r} declares {unit!r}"
            )


def check_result(
    context: SessionContext,
    task: TaskId,
    measuring: Measuring,
    name: str,
    result: EvaluationResult,
) -> None:
    """The result of the evaluation ``task`` carries the declared units (:func:`check_units`),
    and a ``valid`` one every metric an objective or a constraint names of it, in the unit a
    constraint declares, and every metric the roles that read it demand."""
    check_units(name, measuring.metrics, result)
    if result.status is not Status.VALID:
        return
    spec = context.spec
    measured = {(m.stage, m.metric): m for m in result.measurements}
    owned = [(r.stage, r.metric) for r in spec.references if r.invocation == task.label]
    wanted = dict.fromkeys([*owned, *measuring.demanded])
    missing = [key for key in wanted if key not in measured]
    if missing:
        raise EvaluationContractError(
            f"evaluator {task.label!r} returned a valid result without required metric(s): "
            f"{show(missing)}"
        )
    for constraint in spec.constraints:
        if constraint.ref.invocation != task.label:
            continue
        unit = measured[(constraint.stage, constraint.metric)].unit
        if constraint.unit is not None and unit != constraint.unit:
            raise EvaluationContractError(
                f"constraint on {constraint.label} declares unit {constraint.unit!r} but the "
                f"measurement has unit {unit!r}"
            )


def collection_error(
    evaluator: EvaluatorProtocol,
    request: EvaluationRequest,
    listed: dict[tuple[Path, Stat | None], tuple[str, ...] | str],
) -> str | None:
    """Why the file of the evaluator's stage lacks a collection it declares, or holds it with
    another type; ``None`` if every one is there. Only the declared collections are read, so one
    the evaluator does not read cannot fail it. ``listed`` keeps the collection names of each file
    (or why they cannot be listed), so a file read by several evaluators is listed once."""
    declared = declared_collections(evaluator)
    stage: str | None = getattr(evaluator, "stage", None)
    if not declared or stage is None or stage not in request.inputs:
        return None
    path = request.inputs[stage]
    key = (path, stat_of(path))
    if key not in listed:
        if len(listed) >= _LISTED:  # those of earlier candidates' files, named nowhere any more
            del listed[next(iter(listed))]
        try:
            listed[key] = available_collections(path)
        except EvaluationFailed as exc:
            listed[key] = str(exc)
    available = listed[key]
    if isinstance(available, str):
        return available
    names = {role: request.collections[role] for role in declared}
    for role, name in names.items():
        if name not in available:
            listing = ", ".join(available) or "no collections"
            return f"collection {name!r} ({role}) is not in {path.name}, which holds: {listing}"
    try:
        types = collection_types(path, names.values())
    except EvaluationFailed as exc:
        return str(exc)
    for role, collection in declared.items():
        name, held = names[role], types.get(names[role])
        if held != collection.podio_type:
            return (
                f"collection {name!r} ({role}) holds {held!r}, but the evaluator reads "
                f"{collection.podio_type!r}"
            )
    return None


_LISTED = 64  # stage files whose collections are kept listed: those of a few candidates


def to_result(
    task: Task, outcome: TaskOutcome, observed: Observations, root: Path | None
) -> TaskResult:
    """The coordinator's :class:`~k4opt.problem._tasks.TaskResult` of ``outcome``, its files
    where the workspace ``root`` is here, with the stats ``observed`` where they were taken.
    An evaluation's record is the coordinator's to add: it made the request."""
    made = {
        name: Made(local_artifact(artifact, root), observed.made[artifact.ref])
        for name, artifact in outcome.made.items()
    }
    tree = None
    if observed.tree is not None:
        assert root is not None
        assert task.directory is not None
        tree = Tree(root / task.directory, FrozenMap(dict(observed.tree)))
    return TaskResult(
        outcome.task,
        outcome.status,
        outcome.message,
        None if outcome.output is None else local(outcome.output, root),
        {role: local(path, root) for role, path in outcome.artifacts.items()},
        outcome.provenance,
        outcome.elapsed_seconds,
        made=made,
        tree=tree,
        memory=outcome.memory,
    )


def local_result(outcome: TaskOutcome, root: Path | None) -> tuple[EvaluationResult, Path | None]:
    """What an evaluation returned and its own directory (``None`` if it was not called), its
    files where the workspace ``root`` is here."""
    assert outcome.result is not None
    returned = {name: local(path, root) for name, path in outcome.result.artifacts.items()}
    workdir = None if outcome.workdir is None else local(outcome.workdir, root)
    return replace(outcome.result, artifacts=returned), workdir
