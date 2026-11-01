"""Internal: the tasks of candidates, run through a scheduler and a backend.

:func:`_drive` runs the tasks of candidates prepared by :mod:`k4opt.problem._binding` (their
geometry, every stage of each sample that their evaluators read, each once, and each evaluation
once: an evaluator on one of its samples, or on the candidate, with the outputs of the tasks it
reads). Each task that is ready, when the scheduler (:mod:`k4opt.problem._schedule`) gives it a
slot, is described as plain data from the results of the tasks it reads
(:class:`~k4opt.problem._work.TaskSpec`) and handed to a backend
(:mod:`k4opt.problem._backends`); what it left behind becomes its result. A task whose needs did
not all succeed is ``blocked`` and not run. Each ``valid`` result must carry the metrics the
objectives and constraints name of it, and the artifacts its evaluator declares; every artifact
must be a file in the evaluation's own directory, and is hashed once, when the evaluation
returns. A task whose input files changed since they were made fails. Once all its tasks ended,
:func:`_finish_plan` makes the candidate's :class:`~k4opt.problem._candidate.CandidateRecord`.

:func:`_evaluate_candidate` evaluates one candidate without a session, in this thread. It
neither locates evaluators nor materializes geometry: the caller gives the geometry, or none.

Only the required tasks decide the candidate's status: the evaluations its objectives and
constraints name, those of bindings marked ``required: true``, and what they read. Every other
evaluation is a diagnostic: it runs, and its record is kept, but it neither fails the candidate
nor makes it invalid, and a supplied stage only diagnostics read does not keep the record from
representing the candidate.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from k4opt.evaluation import (
    Artifact,
    CandidateKey,
    EvaluationContractError,
    EvaluationRecord,
    EvaluationResult,
    EvaluatorProtocol,
    RoleInput,
    Scalar,
    Status,
)
from k4opt.frozen import FrozenMap
from k4opt.problem._backends import Backend, Handle, InlineBackend
from k4opt.problem._binding import (
    _bind_problem,
    _Binding,
    _context,
    _preflight,
    _Prepared,
    task_cpus,
)
from k4opt.problem._candidate import CandidateRecord, _validate_candidate
from k4opt.problem._files import portable, portable_artifact
from k4opt.problem._schedule import Scheduler
from k4opt.problem._stages import (
    STAGE_TYPES,
    changed_by_geometry,
    check_run_inputs,
    stage_output,
)
from k4opt.problem._tasks import (
    GEOMETRY,
    Kind,
    Task,
    TaskId,
    TaskResult,
    reference_task,
    stage_task,
)
from k4opt.problem._work import (
    GeometryWork,
    Observations,
    TaskOutcome,
    TaskSpec,
    Workspace,
    asked,
    local_result,
    to_result,
)
from k4opt.problem.models import EvaluatorSpec, ProblemSpec, Reference


def _inputs(reads: Mapping[TaskId, TaskResult], root: Path | None) -> dict[str, Path]:
    """What a task is given of the tasks it reads: the main output of each, by its name, named
    in the candidate's workspace (:func:`~k4opt.problem._files.portable`)."""
    return {
        task.name: portable(result.output, root)
        for task, result in reads.items()
        if result.output is not None
    }


def _role_inputs(
    binding: _Binding, entry: EvaluatorSpec, reads: Mapping[TaskId, TaskResult], root: Path | None
) -> dict[str, tuple[RoleInput, ...]]:
    """What each role of a combined evaluator reads, from the results of the tasks it reads, its
    files named in the candidate's workspace."""
    return {
        role: tuple(_role_input(binding, item, reads[reference_task(item)], root) for item in items)
        for role, items in entry.roles.items()
    }


def _role_input(
    binding: _Binding, item: Reference, result: TaskResult, root: Path | None
) -> RoleInput:
    """One item of a role, from the result of its task, which succeeded: an evaluation with what
    it measured and made, or a stage with its file."""
    properties = binding.properties[item.sample]
    if item.evaluator is not None:
        record = result.record
        assert record is not None  # every evaluation has one
        return RoleInput(
            item.sample,
            binding=item.evaluator,
            measurements=record.result.measurements,
            artifacts={
                name: portable_artifact(made.artifact, root) for name, made in result.made.items()
            },
            events=record.request.events,
            seed=record.request.seed,
            properties=properties,
            settings=record.request.settings,
            provenance=record.result.provenance,
        )
    assert item.stage is not None
    assert result.output is not None  # a stage that makes a file
    name = stage_output(item.stage)
    made = result.made.get(name)
    if made is not None:  # made in its directory: never hashed
        artifact = made.artifact
    else:  # a file of the problem or the caller's, hashed when the session opened
        size, sha256 = result.provenance.get("bytes"), result.provenance.get("sha256")
        artifact = Artifact(
            f"{result.id.label}#{name}",
            result.output,
            size if isinstance(size, int) else result.output.stat().st_size,
            sha256 if isinstance(sha256, str) else None,
        )
    events, seed = binding.contexts[item.sample].conditions(item.stage)
    return RoleInput(
        item.sample,
        stage=item.stage,
        artifacts={item.stage: portable_artifact(artifact, root)},
        events=events,
        seed=seed,
        properties=properties,
        provenance=result.provenance,
    )


@dataclass(slots=True)
class _Run:
    """One candidate under way, as its coordinator follows it: its workspace, what its preflight
    prepared, its key, the files the caller supplied for its stages and how its geometry is made.
    ``started`` is when its evaluation began, its checks included; ``specs`` holds the
    description of each task handed out, ``results`` its results once all its tasks ended, and
    ``error`` what stopped it, if anything did. ``request``, ``values`` and ``reference`` (what
    its record says it was made for) are the caller's."""

    workspace: Workspace
    prepared: _Prepared | None
    key: CandidateKey = ()
    supplied: Mapping[TaskId, Path] = field(default_factory=dict)
    geometry: GeometryWork | None = None
    started: float = 0.0
    specs: dict[TaskId, TaskSpec] = field(default_factory=dict)
    results: dict[TaskId, TaskResult] = field(default_factory=dict)
    error: Exception | None = None
    own_refs: bool = False
    request: object = None
    values: Mapping[str, Scalar] = field(default_factory=dict)
    reference: object = None


def _task_spec(run: _Run, task: Task, results: Mapping[TaskId, TaskResult]) -> TaskSpec:
    """The description of ``task`` of ``run``, from the results of the tasks it reads."""
    assert run.prepared is not None
    binding, root = run.prepared.binding, run.workspace.root
    reads = {need: results[need] for need in task.needs}
    watched = tuple(
        portable_artifact(made.artifact, root)
        for result in reads.values()
        for made in result.made.values()
    )
    trees = [result.tree for result in reads.values() if result.tree is not None]
    tree = None if not trees else portable(trees[0].root, root)
    workspace = run.workspace.id
    if task.id.kind is Kind.GEOMETRY:
        return TaskSpec(workspace, task, watched=watched, tree=tree, geometry=run.geometry)
    if task.id.kind is Kind.STAGE:
        given, supplied = _inputs(reads, root), run.supplied.get(task.id)
        cpus = task_cpus(binding, task)
        return TaskSpec(workspace, task, given, watched, tree, supplied=supplied, cpus=cpus)
    entry = run.prepared.entries[task.id]
    if task.id.sample is None:  # the geometry, and what its roles read: no stage of its own
        given = _inputs({need: result for need, result in reads.items() if need == GEOMETRY}, root)
        roles = _role_inputs(binding, entry, reads, root)
    else:
        given, roles = _inputs(reads, root), {}
    request = run.prepared.requests[task.id]
    return TaskSpec(workspace, task, given, watched, tree, request=request, roles=roles)


def _seen(task: Task, results: Mapping[TaskId, TaskResult]) -> Observations:
    """The stats taken of the files the tasks ``task`` reads made, where they were made."""
    reads = [results[need] for need in task.needs]
    made = {m.artifact.ref: m.stat for result in reads for m in result.made.values()}
    trees = [result.tree.stats for result in reads if result.tree is not None]
    return Observations(FrozenMap(made), trees[0] if trees else None)


def _result(run: _Run, task: Task, outcome: TaskOutcome, observed: Observations) -> TaskResult:
    """The coordinator's result of ``task`` of ``run`` from what it left behind; the record of an
    evaluation holds the request it was asked, its files where the workspace is here."""
    assert run.prepared is not None
    root = run.workspace.root
    result = to_result(task, outcome, observed, root)
    if task.id.kind is not Kind.EVALUATOR:
        return result
    entry = run.prepared.entries[task.id]
    returned, workdir = local_result(outcome, root)
    request = asked(run.specs[task.id], root)
    if workdir is not None:  # it was called, in its own directory
        request = replace(request, workdir=workdir)
    record = EvaluationRecord(
        request, returned, entry.name, entry.version, run.key, outcome.call_seconds
    )
    return replace(result, record=record)


@dataclass(frozen=True, slots=True)
class Activity:
    """What runs at a moment of :meth:`~k4opt.problem.EvaluationSession.evaluate_each`: the
    ``candidates`` under way, and the ``busy`` of the ``capacity`` job slots their running tasks
    hold (a simulation on several threads holds as many)."""

    candidates: int
    busy: int
    capacity: int


type Check = Callable[[_Run, TaskOutcome, TaskResult], None]
"""What the coordinator checks of each result as it comes, raising to stop its candidate."""

_WAIT = 0.2  # seconds the coordinator waits for a task to end before it looks around again


def _drive(
    scheduler: Scheduler,
    backend: Backend,
    runs: dict[str, _Run],
    pull: Callable[[], _Run | None],
    check: Check | None = None,
    activity: Callable[[Activity], None] | None = None,
) -> Generator[_Run]:
    """Run the tasks of candidates until there are none left; yield each candidate once all its
    tasks ended (``results``), or once its work raised (``error``: its other tasks are stopped).

    ``pull`` gives the next candidate, or ``None`` once there is none; it is called only when the
    scheduler is idle, after every candidate that ended was yielded, so with one slot the
    candidates run one after another, each in plan order. ``runs`` holds the candidates under
    way, by workspace. Leaving the loop early (``close()``, an exception) stops every task still
    running. ``activity`` is told what runs whenever that changed.
    """
    pulling = True
    told: Activity | None = None

    def tell() -> None:
        nonlocal told
        now = Activity(len(runs), scheduler.busy, scheduler.capacity)
        if activity is not None and now != told:
            told = now
            activity(now)

    def stop(run: _Run, error: Exception) -> None:
        run.error = error
        key = run.workspace.id
        running = scheduler.drop(key)
        backend.cancel([Handle(key, task) for task in running])
        backend.close_workspace(key)
        runs.pop(key, None)

    try:
        while True:
            ended: list[_Run] = []
            while (item := scheduler.next()) is not None:
                key, task = item
                run = runs[str(key)]
                try:
                    results = scheduler.results(key)
                    spec = _task_spec(run, task, results)
                    run.specs[task.id] = spec
                    backend.submit(spec, _seen(task, results))
                except Exception as exc:
                    stop(run, exc)
                    ended.append(run)
            tell()
            for handle, outcome, observed in backend.wait(_WAIT) if scheduler.running else []:
                found = runs.get(handle.workspace)
                if found is None:  # stopped meanwhile
                    continue
                run = found
                assert run.prepared is not None
                task = run.prepared.binding.plan[handle.task]
                try:
                    if isinstance(outcome, Exception):  # raised where the task ran
                        raise outcome
                    result = _result(run, task, outcome, observed)
                    if check is not None:
                        check(run, outcome, result)
                    scheduler.finished(handle.workspace, result)
                except Exception as exc:
                    stop(run, exc)
                    ended.append(run)
                    continue
                if scheduler.done(handle.workspace):
                    run.results = scheduler.remove(handle.workspace)
                    backend.close_workspace(handle.workspace)
                    del runs[handle.workspace]
                    ended.append(run)
            tell()
            yield from ended
            if pulling and scheduler.idle:
                pulled = pull()
                if pulled is None:
                    pulling = False
                elif pulled.prepared is None:  # it failed before anything was created
                    yield pulled
                else:
                    runs[pulled.workspace.id] = pulled
                    backend.open(pulled.workspace)
                    binding = pulled.prepared.binding
                    costs = {t.id: backend.cost(task_cpus(binding, t)) for t in binding.plan.tasks}
                    scheduler.add(pulled.workspace.id, binding.plan, costs)
                continue
            if not runs:  # and nothing more to pull: a free slot would have pulled
                return
            # each candidate under way has a task running or ready (a free slot takes it)
            assert scheduler.running or not scheduler.idle
    finally:
        for run in list(runs.values()):
            key = run.workspace.id
            backend.cancel([Handle(key, task) for task in scheduler.drop(key)])
            backend.close_workspace(key)
        runs.clear()


def _evaluate_candidate(
    spec: ProblemSpec,
    candidate: Mapping[str, Scalar],
    implementations: Sequence[type[EvaluatorProtocol]],
    *,
    inputs: Mapping[str, Path] | None = None,
    workdir: Path | None = None,
) -> CandidateRecord:
    """Evaluate one complete ``candidate`` of ``spec`` with an instance of each implementation.

    The candidate maps each canonical parameter reference to a scalar; ``implementations`` (an
    evaluator class, say) implement the problem's ``evaluators``, in any order. Before any stage
    runs, raises :class:`~k4opt.problem.CandidateError` for a candidate outside the declared
    domains and :class:`EvaluationContractError` for implementations that cannot be created or
    do not match the problem (missing, undeclared, given twice, another version), settings an
    evaluator's ``check()`` rejects, a stage that cannot run, or an objective or constraint
    metric its evaluator does not declare. ``inputs["geometry"]`` is the candidate's geometry,
    already prepared; any other entry of ``inputs`` supplies the file of the stage of a sample
    it is named after (``gamma/simulation``). The stages the evaluators read and ``inputs`` does
    not supply then run once each, in ``workdir``. An evaluator whose stage failed, or whose
    stage output lacks the requested events, is ``blocked`` and not called, as is a combined
    evaluator an item of whose roles did not succeed (failed, invalid or blocked); the others
    are still called. An evaluator that raises :class:`~k4opt.evaluation.EvaluationFailed` or
    :class:`~k4opt.evaluation.EvaluationInvalid` gets that result; any other exception
    propagates, and so does an :class:`EvaluationContractError` for a result that breaks the
    contract.
    """
    started = time.perf_counter()
    values, key = _validate_candidate(spec, candidate)
    given = dict(inputs or {})
    geometry = given.pop(GEOMETRY.name, None)
    if geometry is not None:  # an input from outside the workspace: named absolutely
        geometry = Path(os.path.abspath(geometry))
    supplied = {}
    for label, path in given.items():
        sample, _, stage = label.partition("/")
        if sample not in spec.samples or stage not in STAGE_TYPES:
            raise EvaluationContractError(
                f"input {label!r}: expected SAMPLE/STAGE, a sample of the problem and one of "
                f"{', '.join(STAGE_TYPES)}"
            )
        supplied[stage_task(stage, sample)] = Path(os.path.abspath(path))
    binding = _bind_problem(spec, implementations, supplied)
    planned = {task.id for task in binding.plan.of(Kind.STAGE)}
    for task in supplied:  # a file no task reads would be dropped without a word
        if task not in planned:
            raise EvaluationContractError(
                f"input {task.label!r}: no evaluator of the problem reads that stage of "
                f"{task.sample}"
            )
    prepared = _preflight(binding, values)
    checked = time.perf_counter() - started
    return _execute_plan(
        spec,
        prepared,
        key,
        supplied=supplied,
        geometry=None if geometry is None else GeometryWork(given=geometry),
        workdir=workdir,
        checked_seconds=checked,
    )


def _execute_plan(
    spec: ProblemSpec,
    prepared: _Prepared,
    key: CandidateKey,
    *,
    supplied: Mapping[TaskId, Path],
    geometry: GeometryWork | None,
    workdir: Path | None,
    checked_seconds: float,
    timeout: float | None = None,
) -> CandidateRecord:
    """Run the tasks of a prepared candidate, one after another in plan order, in this thread;
    evaluator checks and defaults are not repeated.

    ``supplied`` holds the files the caller supplies by stage task, the ones the plan was made with;
    ``geometry`` says how the candidate's geometry is made, and ``None`` gives none (to evaluators
    that need none). Each task writes below ``workdir``, in the directory the plan gives it.
    ``checked_seconds``, the time the candidate's validation and preflight took, counts towards
    ``elapsed_seconds``, which thus covers k4opt's checks too (not what the caller does between).
    ``timeout`` limits each stage that runs a program
    (:attr:`~k4opt.problem._stages.StageContext.timeout`).
    """
    started = time.perf_counter() - checked_seconds
    binding = prepared.binding
    stages = binding.plan.of(Kind.STAGE)
    check_run_inputs([t.id.name for t in stages if t.given is None], geometry is not None, workdir)
    run = _Run(Workspace("candidate", workdir), prepared, key, supplied, geometry, started)
    backend = InlineBackend(
        _context(spec, binding, timeout), lambda workspace, task: prepared.instances[task]
    )
    pulled = iter([run])
    driven = _drive(Scheduler(backend.capacity), backend, {}, lambda: next(pulled, None))
    with contextlib.closing(driven):
        for ended in driven:
            if ended.error is not None:
                raise ended.error
            return _finish_plan(ended)
    raise AssertionError("the candidate never ended")  # pragma: no cover


def _finish_plan(run: _Run) -> CandidateRecord:
    """The record of a candidate whose tasks all ended. A blocked evaluator's record holds what it
    would have been asked."""
    prepared = run.prepared
    assert prepared is not None
    plan, results = prepared.binding.plan, dict(run.results)
    for task in plan.of(Kind.EVALUATOR):
        held = results[task.id]
        if held.status == Status.BLOCKED:
            entry = prepared.entries[task.id]
            assert held.message is not None
            outcome = EvaluationResult(Status.BLOCKED, message=held.message)
            request = prepared.requests[task.id]
            record = EvaluationRecord(request, outcome, entry.name, entry.version, run.key, None)
            results[task.id] = replace(held, record=record)
    represents = not any(
        changed_by_geometry(task.id.name)
        for task in plan.of(Kind.STAGE)
        if task.given == "supplied" and task.required
    )
    return CandidateRecord(
        run.key, plan, results, _since(run.started), represents, prepared.binding.warnings
    )


def _since(started: float) -> float:
    return time.perf_counter() - started
