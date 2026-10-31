"""Internal: the work of one candidate as tasks, each run at most once and only after what it reads.

A task is the smallest step with a meaning of its own: the candidate's ``geometry`` (materialized
once, read by every simulation and every evaluator), one pipeline stage (``generation``,
``simulation``) or one evaluator, named by its binding. Each task names the tasks it reads, its
``needs``, and nothing else orders it. Three seams keep what runs apart from how it runs:

* :func:`plan_tasks` resolves the dependencies from the problem alone, once per session, so that
  a record can be checked against the same plan
  (:meth:`~k4opt.problem.Assessment.from_summary`);
* :func:`run_tasks` executes a plan one task after another. A task whose needs did not all
  succeed is ``blocked`` (:func:`blocked`) and is not run; the work of a task is the caller's.
  Another executor (processes, a batch system) would replace only this loop;
* the candidate's record is assembled from the task results alone (:mod:`k4opt.problem._run`),
  and its status and message are derived from them (:func:`candidate_status`,
  :func:`candidate_message`).

A stage given as a file, in the problem or by the caller, is a task too: it reads that file and
no task, so the stages before it are planned only if something else needs them.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from k4opt.evaluation import EvaluationContractError, EvaluationRecord, Status
from k4opt.frozen import FrozenMap, Json, freeze
from k4opt.problem._stages import STAGE_TYPES, graph, reads_geometry, writes_files
from k4opt.problem.models import Stage


class Kind(StrEnum):
    """What a task does: makes the geometry, makes a stage or runs an evaluator."""

    GEOMETRY = "geometry"
    STAGE = "stage"
    EVALUATOR = "evaluator"


@dataclass(frozen=True, slots=True)
class TaskId:
    """A task of a candidate: its kind and its name (``geometry``, a stage type, the binding of an
    evaluator)."""

    kind: Kind
    name: str

    @property
    def label(self) -> str:
        """How records name the task; unambiguous, since no binding may be named after the
        geometry or a stage type (:data:`RESERVED_NAMES`)."""
        return self.name


GEOMETRY = TaskId(Kind.GEOMETRY, "geometry")

RESERVED_NAMES = (GEOMETRY.name, *STAGE_TYPES)
"""Names of tasks k4opt runs itself, which no binding of an evaluator may take."""

DONE = "done"
"""The status of a geometry or a stage that is available; an evaluator's is ``valid`` instead."""

_SUCCEEDED = frozenset({DONE, Status.VALID})


def stage_task(name: str) -> TaskId:
    return TaskId(Kind.STAGE, name)


def evaluator_task(name: str) -> TaskId:
    return TaskId(Kind.EVALUATOR, name)


@dataclass(frozen=True, slots=True)
class Task:
    """One task of a plan: the tasks it reads, and where it writes.

    ``directory`` is relative to the candidate's run directory, ``None`` for a task that writes
    nothing. ``given`` says how a stage that does not run is given: as a ``file`` of the problem,
    or ``supplied`` by the caller.
    """

    id: TaskId
    needs: tuple[TaskId, ...] = ()
    directory: str | None = None
    given: Literal["file", "supplied"] | None = None


@dataclass(frozen=True, slots=True)
class Plan:
    """The tasks of every candidate of a problem, each after the tasks it reads.

    The order of ``tasks`` is the order :func:`run_tasks` runs them in: the geometry, the stages
    in pipeline order, the evaluators in problem order. It is one of the orders in which every
    task comes after its needs, and nothing may depend on which one: another executor may run any
    of them, or several tasks at once.
    """

    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        seen: set[TaskId] = set()
        for task in self.tasks:
            if task.id in seen:
                raise EvaluationContractError(f"task {task.id.label!r} is planned twice")
            later = [need.label for need in task.needs if need not in seen]
            if later:  # also refuses a cycle
                raise EvaluationContractError(
                    f"task {task.id.label!r} reads {', '.join(later)}, which is not planned "
                    "before it"
                )
            seen.add(task.id)

    def __getitem__(self, task: TaskId) -> Task:
        for planned in self.tasks:
            if planned.id == task:
                return planned
        raise KeyError(task)

    def of(self, kind: Kind) -> tuple[Task, ...]:
        """The tasks of one kind, in plan order."""
        return tuple(task for task in self.tasks if task.id.kind is kind)


def plan_tasks(
    stages: Mapping[str, Stage],
    evaluators: Sequence[tuple[str, str | None]],
    supplied: Collection[str] = (),
) -> Plan:
    """The plan of a problem with ``stages`` and ``evaluators`` (``(binding, stage it reads)`` in
    problem order), the stages in ``supplied`` given by the caller.

    The stages planned are those the evaluators read and the stages these read, except that a
    stage given as a file reads nothing. A stage that runs reads the geometry if it is made from
    it; every evaluator reads the geometry, and its stage with every stage that one reads.
    """
    files = {name for name, stage in stages.items() if stage.file is not None}
    given = {*supplied, *files}
    targets = [stage for _, stage in evaluators if stage is not None]
    tasks = [Task(GEOMETRY, directory=GEOMETRY.name)]
    for name, reads in graph(targets, given):
        if name in supplied:
            tasks.append(Task(stage_task(name), given="supplied"))
        elif name in files:
            tasks.append(Task(stage_task(name), given="file"))
        else:
            needs = tuple(map(stage_task, reads))
            tasks.append(
                Task(
                    stage_task(name),
                    (GEOMETRY, *needs) if reads_geometry(name) else needs,
                    name if writes_files(name) else None,
                )
            )
    for name, stage in evaluators:
        chain = [] if stage is None else [node for node, _ in graph([stage], given)]
        needs = (GEOMETRY, *map(stage_task, chain))
        tasks.append(Task(evaluator_task(name), needs, f"evaluators/{name}"))
    return Plan(tuple(tasks))


@dataclass(frozen=True, slots=True)
class TaskResult:
    """What one task left behind.

    ``status`` is ``done``, ``failed`` or ``blocked`` for the geometry and a stage, and the
    status of its result (``valid``, ``invalid``, ``failed``) or ``blocked`` for an evaluator;
    ``message`` says why one did not succeed. ``output`` is what the tasks that read it are given,
    by its name (the compact file of the geometry, the events of a stage), ``None`` if it gives
    nothing (a particle gun); ``artifacts`` are its other files (logs) and ``provenance`` how it
    ran. ``blocked_by`` holds the tasks that did not succeed and kept a ``blocked`` one from
    running; ``record`` the evaluator's record, blocked ones included. ``elapsed_seconds`` is
    the wall time k4opt spent on it, ``None`` if it did not run.
    """

    id: TaskId
    status: str
    message: str | None = None
    output: Path | None = None
    artifacts: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    provenance: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    elapsed_seconds: float | None = None
    blocked_by: tuple[TaskId, ...] = ()
    record: EvaluationRecord | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", FrozenMap(self.artifacts))
        provenance = freeze(self.provenance, f"{self.id.label} provenance", EvaluationContractError)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "blocked_by", tuple(self.blocked_by))

    @property
    def succeeded(self) -> bool:
        """Whether the tasks that read it can run: it is ``done``, or ``valid``."""
        return self.status in _SUCCEEDED


def blocked(task: Task, results: Mapping[TaskId, TaskResult]) -> TaskResult | None:
    """The ``blocked`` result of ``task`` if a task it reads did not succeed, ``None`` if it can
    run. ``results`` holds the result of every task it reads, and of the tasks these read.

    It is blocked by the tasks that did not succeed themselves, not by the blocked ones in
    between, so its message names where the candidate went wrong.
    """
    roots: dict[TaskId, None] = {}
    for need in task.needs:
        result = results[need]
        if result.status == Status.BLOCKED:
            roots |= dict.fromkeys(result.blocked_by)
        elif not result.succeeded:
            roots[need] = None
    if not roots:
        return None
    causes = ", ".join(f"{root.label} {results[root].status}" for root in roots)
    return TaskResult(task.id, Status.BLOCKED, f"not run: {causes}", blocked_by=tuple(roots))


type RunTask = Callable[[Task, Mapping[TaskId, TaskResult]], TaskResult]
"""The work of one task, given the results of the tasks it reads; the only code that does work."""


def run_tasks(plan: Plan, run: RunTask) -> dict[TaskId, TaskResult]:
    """Run every task of ``plan`` in plan order, one at a time, and return their results.

    A task whose needs did not all succeed is not run, and its result is the ``blocked`` one of
    :func:`blocked`. ``run`` gets the results of the tasks it reads, and only those. An exception
    of ``run`` propagates: a task that cannot do its work reports so in its result.
    """
    results: dict[TaskId, TaskResult] = {}
    for task in plan.tasks:
        result = blocked(task, results)
        if result is None:
            result = run(task, {need: results[need] for need in task.needs})
        results[task.id] = result
    return results


def candidate_status(statuses: Iterable[str]) -> Status:
    """The status of a candidate from its tasks': ``failed`` if any failed or was blocked (the
    evaluation is incomplete), otherwise ``invalid`` if any is, otherwise ``valid``."""
    seen = set(statuses)
    if seen & {Status.FAILED, Status.BLOCKED}:
        return Status.FAILED
    return Status.INVALID if Status.INVALID in seen else Status.VALID


def candidate_message(results: Iterable[TaskResult]) -> str | None:
    """The message of a candidate from its task results, in plan order: each that failed or is
    invalid, with its status and message, then the tasks they blocked."""
    results = list(results)
    causes = [
        f"{result.id.label} {result.status}: {result.message}"
        for result in results
        if result.status in (Status.FAILED, Status.INVALID)
    ]
    held = [result.id.label for result in results if result.status == Status.BLOCKED]
    if held:
        causes.append(f"blocked: {', '.join(held)}")
    return "; ".join(causes) or None
