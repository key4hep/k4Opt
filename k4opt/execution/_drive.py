"""Internal: the tasks of candidates, run through a scheduler and a backend.

:func:`_drive` runs the tasks of candidates prepared by :mod:`k4opt.execution._binding` (their
geometry, every stage of each sample that their evaluators read, each once, and each evaluation
once: an evaluator on one of its samples, or on the candidate, with the outputs of the tasks it
reads). Each task that is ready, when the scheduler (:mod:`k4opt.execution._schedule`) gives it a
slot, is described as plain data from the results of the tasks it reads
(:mod:`k4opt.execution._describe`) and handed to a backend
(:mod:`k4opt.execution._backends`); what it left behind becomes its result. A task whose needs did
not all succeed is ``blocked`` and not run. Each ``valid`` result must carry the metrics the
objectives and constraints name of it, and the artifacts its evaluator declares; every artifact
must be a file in the evaluation's own directory, and is hashed once, when the evaluation
returns. A task whose input files changed since they were made fails. Once all its tasks ended,
:func:`_finish_plan` makes the candidate's :class:`~k4opt.record.CandidateRecord`.

Only the required tasks decide the candidate's status: the evaluations its objectives and
constraints name, those of bindings marked ``required: true``, and what they read. Every other
evaluation is a diagnostic: it runs, and its record is kept, but it neither fails the candidate
nor makes it invalid, and a supplied stage only diagnostics read does not keep the record from
representing the candidate.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from k4opt.evaluation import (
    CandidateKey,
    EvaluationRecord,
    EvaluationResult,
    Scalar,
    Status,
)
from k4opt.execution._backends import Backend, Handle
from k4opt.execution._binding import task_cpus
from k4opt.execution._describe import describe, seen
from k4opt.execution._files import Made, Observed, Tree, local, local_artifact
from k4opt.execution._preflight import Rejected, _Prepared
from k4opt.execution._reuse import REUSED, Reuse, made, reused
from k4opt.execution._schedule import Scheduler
from k4opt.execution._tasks import (
    GeometryWork,
    Observations,
    TaskOutcome,
    TaskSpec,
    Workspace,
)
from k4opt.execution._work import asked
from k4opt.frozen import FrozenMap, Json
from k4opt.problem.plan import Kind, Task, TaskId, represents_candidate
from k4opt.record.results import CandidateRecord, TaskResult
from k4opt.record.signature import SIGNATURE


@dataclass(slots=True)
class _Run:
    """One candidate under way, as its coordinator follows it: its workspace, what its preflight
    prepared, its key, the files the caller supplied for its stages and how its geometry is made.
    ``started`` is when its evaluation began, its checks included; ``specs`` holds the
    description of each task handed out, ``results`` its results once all its tasks ended, and
    ``error`` what stopped it, if anything did. ``observed`` holds, by task, what is watched of
    the files each task that ran made, from when its result came until the candidate is
    published: what the tasks that read them check, and the candidate checks at its end; a task
    that never ran (blocked) has none. ``rejected`` says why an evaluator rejected it, in which
    case nothing of it was created and it has no ``prepared``. ``request``, ``values`` and
    ``reference`` (what its record says it was made for) are the caller's. ``signatures`` holds
    the signature of each stage handed out (:mod:`k4opt.record.signature`), which its result
    carries. ``relies`` holds the records of the other candidates of the session whose outputs
    it reused before they were published: it is published only if they were."""

    workspace: Workspace
    prepared: _Prepared | None
    key: CandidateKey = ()
    supplied: Mapping[TaskId, Path] = field(default_factory=dict)
    geometry: GeometryWork | None = None
    started: float = 0.0
    specs: dict[TaskId, TaskSpec] = field(default_factory=dict)
    results: dict[TaskId, TaskResult] = field(default_factory=dict)
    observed: dict[TaskId, Observed] = field(default_factory=dict)
    error: Exception | None = None
    rejected: Rejected | None = None
    own_refs: bool = False
    request: object = None
    values: Mapping[str, Scalar] = field(default_factory=dict)
    reference: object = None
    signatures: dict[TaskId, Mapping[str, Json]] = field(default_factory=dict)
    relies: set[str] = field(default_factory=set)


def _result(run: _Run, task: Task, outcome: TaskOutcome, observed: Observations) -> TaskResult:
    """The coordinator's result of ``task`` of ``run`` from what it left behind, with what is
    watched of the files it made kept in ``run.observed``; the record of an evaluation holds the
    request it was asked, its files where the workspace is here."""
    assert run.prepared is not None
    root = run.workspace.root
    result, run.observed[task.id] = to_result(task, outcome, observed, root)
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
    """What runs at a moment of :meth:`~k4opt.execution.EvaluationSession.evaluate_each`: the
    ``candidates`` under way, and the ``busy`` of the ``capacity`` job slots their running tasks
    hold (a simulation on several threads holds as many)."""

    candidates: int
    busy: int
    capacity: int


type Check = Callable[[_Run, TaskOutcome | None, TaskResult], None]
"""What the coordinator checks of each result as it comes, raising to stop its candidate; a
reused stage has no outcome of its own."""

type Sign = Callable[[_Run, Task, Mapping[TaskId, TaskResult]], Mapping[str, Json] | None]
"""The signature of a task of a candidate, from the results of what it reads, ``None`` for one
that has none (:mod:`k4opt.record.signature`): what the coordinator attaches to its result."""

_WAIT = 0.2  # seconds the coordinator waits for a task to end before it looks around again


def _drive(
    scheduler: Scheduler,
    backend: Backend,
    runs: dict[str, _Run],
    pull: Callable[[], _Run | None],
    check: Check | None = None,
    activity: Callable[[Activity], None] | None = None,
    sign: Sign | None = None,
    reuse: Reuse | None = None,
) -> Generator[_Run]:
    """Run the tasks of candidates until there are none left; yield each candidate once all its
    tasks ended (``results``), or once its work raised (``error``: its other tasks are stopped).

    ``pull`` gives the next candidate, or ``None`` once there is none; it is called only when the
    scheduler is idle, after every candidate that ended was yielded, so with one slot the
    candidates run one after another, each in plan order. ``runs`` holds the candidates under
    way, by workspace. Leaving the loop early (``close()``, an exception) stops every task still
    running. ``activity`` is told what runs whenever that changed. ``sign`` gives the signature
    of a task before it is handed out, which its result then carries in its provenance. With
    ``reuse``, a stage whose signature already made an output reuses it and is not run, and one
    whose signature runs for another stage waits for it, holding its slot
    (:mod:`k4opt.execution._reuse`); a candidate that reused the output of another that was not
    published yet is yielded only after that one, so that the caller publishes it first, or
    not. Candidates never rely on each other: one is added only when no task is ready, and ready
    tasks start in the order of their candidates, so a stage of a later candidate never runs
    before the same stage of an earlier one, which it reuses. Each task may run as long as its
    description says
    (:attr:`~k4opt.execution._tasks.TaskSpec.timeout`), which the backend enforces.
    """
    pulling = True
    told: Activity | None = None
    ended: list[_Run] = []
    held: list[_Run] = []  # ended, but relying on a candidate not yet yielded

    def release() -> None:
        """Yield, after them, the candidates held for others that are no longer under way."""
        waiting = {run.workspace.id for run in (*runs.values(), *held)}
        while True:
            free = [run for run in held if not (run.relies - {run.workspace.id}) & waiting]
            if not free:
                return
            for run in free:
                held.remove(run)
                ended.append(run)
                waiting.discard(run.workspace.id)

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
        ended.append(run)
        release()
        if reuse is not None:  # what waited for its stages runs itself
            for item in reuse.forget(key):
                assert isinstance(item, tuple)
                hand_out(*item)

    def submit(run: _Run, task: Task) -> None:
        """Describe ``task`` of ``run`` and hand it to the backend."""
        key = run.workspace.id
        if key not in runs:  # stopped meanwhile
            return
        try:
            assert run.prepared is not None
            spec = describe(
                task,
                run.prepared,
                run.workspace,
                scheduler.results(key),
                run.observed,
                supplied=run.supplied.get(task.id),
                geometry=run.geometry,
            )
            run.specs[task.id] = spec
            backend.submit(spec, seen(task, run.observed))
        except Exception as exc:
            stop(run, exc)

    def hand_out(run: _Run, task: Task) -> None:
        """Sign ``task`` of ``run``, then reuse what a stage of its signature made, wait for one
        that runs, or hand it to the backend."""
        key = run.workspace.id
        try:
            signature = None if sign is None else sign(run, task, scheduler.results(key))
        except Exception as exc:
            stop(run, exc)
            return
        if signature is None:
            submit(run, task)
            return
        run.signatures[task.id] = signature
        digest = str(signature["sha256"])
        if reuse is None:
            submit(run, task)
        elif (found := reuse.find(digest)) is not None:
            source, stat = found
            if source.record != key and reuse.pending(source.record):
                run.relies.add(source.record)
            result, run.observed[task.id] = reused(task, source, stat)
            complete(run, task, result)
        elif reuse.running(digest):
            reuse.wait(digest, key, task, (run, task))
        else:
            reuse.start(digest, key)
            submit(run, task)

    def complete(
        run: _Run, task: Task, result: TaskResult, outcome: TaskOutcome | None = None
    ) -> None:
        """``task`` of ``run`` ended with ``result`` (from ``outcome``, if it ran): check it and
        take it; the stages that waited for it then reuse what it made, or run themselves."""
        key = run.workspace.id
        signature = run.signatures.get(task.id)
        if signature is not None:
            provenance = {**result.provenance, SIGNATURE: signature}
            result = replace(result, provenance=FrozenMap(provenance))
        try:
            if check is not None:
                check(run, outcome, result)
            scheduler.finished(key, result)
        except Exception as exc:
            stop(run, exc)
            return
        finally:
            if signature is not None and reuse is not None and REUSED not in result.provenance:
                digest = str(signature["sha256"])
                source = made(task, result, run.workspace.id, run.workspace.root)
                for item in reuse.ended(digest, source):
                    assert isinstance(item, tuple)
                    hand_out(*item)
        if scheduler.done(key):
            run.results = scheduler.remove(key)
            backend.close_workspace(key)
            del runs[key]
            held.append(run)
            release()

    try:
        while True:
            while (item := scheduler.next()) is not None:
                key, task = item
                hand_out(runs[str(key)], task)
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
                except Exception as exc:
                    stop(run, exc)
                    continue
                complete(run, task, result, outcome)
            tell()
            yield from ended
            ended.clear()
            if pulling and scheduler.idle:
                pulled = pull()
                if pulled is None:
                    pulling = False
                elif pulled.prepared is None:  # rejected or failed before anything was created
                    yield pulled
                else:
                    runs[pulled.workspace.id] = pulled
                    backend.open(pulled.workspace)
                    plan, contexts = pulled.prepared.binding.plan, pulled.prepared.contexts
                    costs = {t.id: backend.cost(task_cpus(contexts, t)) for t in plan.tasks}
                    scheduler.add(pulled.workspace.id, plan, costs)
                continue
            if not runs:  # and nothing more to pull: a free slot would have pulled
                # those held for each other: none that relies on an unpublished one is published
                yield from held
                return
            # each candidate under way has a task running, ready or waiting for its twin
            assert scheduler.running or not scheduler.idle
    finally:
        for run in list(runs.values()):
            key = run.workspace.id
            backend.cancel([Handle(key, task) for task in scheduler.drop(key)])
            backend.close_workspace(key)
        if reuse is not None:  # none of these is published now: a later evaluation waits for
            # none of their stages, and reuses none of their outputs
            for run in (*runs.values(), *held, *ended):
                reuse.forget(run.workspace.id)
        runs.clear()


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
    return CandidateRecord(
        run.key,
        plan,
        results,
        _since(run.started),
        represents_candidate(plan),
        prepared.binding.warnings,
    )


def _since(started: float) -> float:
    return time.perf_counter() - started


def to_result(
    task: Task, outcome: TaskOutcome, observed: Observations, root: Path | None
) -> tuple[TaskResult, Observed]:
    """The coordinator's :class:`~k4opt.record.TaskResult` of ``outcome``, its files
    where the workspace ``root`` is here, and what it watches of the files the task made: each
    with the stat ``observed`` where it was made, and the geometry it made. An evaluation's
    record is the coordinator's to add: it made the request."""
    made = {name: local_artifact(artifact, root) for name, artifact in outcome.made.items()}
    tree = None
    if observed.tree is not None:
        assert root is not None
        assert task.directory is not None
        tree = Tree(root / task.directory, FrozenMap(dict(observed.tree)))
    watched = Observed(
        FrozenMap({name: Made(made[name], observed.made[made[name].ref]) for name in made}), tree
    )
    result = TaskResult(
        outcome.task,
        outcome.status,
        outcome.message,
        None if outcome.output is None else local(outcome.output, root),
        {role: local(path, root) for role, path in outcome.artifacts.items()},
        outcome.provenance,
        outcome.elapsed_seconds,
        made=made,
        memory=outcome.memory,
    )
    return result, watched


def local_result(outcome: TaskOutcome, root: Path | None) -> tuple[EvaluationResult, Path | None]:
    """What an evaluation returned and its own directory (``None`` if it was not called), its
    files where the workspace ``root`` is here."""
    assert outcome.result is not None
    returned = {name: local(path, root) for name, path in outcome.result.artifacts.items()}
    workdir = None if outcome.workdir is None else local(outcome.workdir, root)
    return replace(outcome.result, artifacts=returned), workdir
