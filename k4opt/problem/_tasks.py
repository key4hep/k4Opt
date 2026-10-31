"""Internal: the work of one candidate as tasks, each run at most once and only after what it reads.

A task is the smallest step with a meaning of its own: the candidate's ``geometry`` (materialized
once, read by every simulation and every evaluator), one pipeline stage of one sample
(``gamma/simulation``), or one evaluation: an evaluator, named by its binding, on one sample
(``gamma/calo.photon``) or on the candidate (``readout``). Each task names the tasks it reads, its
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
no task, so the stages before it are planned only if something else needs them. The samples
share nothing but the geometry: a stage of one sample is never read by a task of another.
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
from k4opt.problem.models import EvaluatorSpec, Sample


class Kind(StrEnum):
    """What a task does: makes the geometry, makes a stage or runs an evaluator."""

    GEOMETRY = "geometry"
    STAGE = "stage"
    EVALUATOR = "evaluator"


@dataclass(frozen=True, slots=True)
class TaskId:
    """A task of a candidate: its kind, its name (``geometry``, a stage type, the binding of an
    evaluator) and the sample it works on, ``None`` for the geometry and an evaluator that
    measures the candidate."""

    kind: Kind
    name: str
    sample: str | None = None

    @property
    def label(self) -> str:
        """How records name the task: ``sample/name``, or ``name`` without a sample. Unambiguous,
        since no binding may be named after the geometry or a stage type
        (:data:`RESERVED_NAMES`), and neither a sample nor a binding name holds ``/``."""
        return self.name if self.sample is None else f"{self.sample}/{self.name}"


GEOMETRY = TaskId(Kind.GEOMETRY, "geometry")

RESERVED_NAMES = (GEOMETRY.name, *STAGE_TYPES)
"""Names of tasks k4opt runs itself, which no binding of an evaluator may take."""

DONE = "done"
"""The status of a geometry or a stage that is available; an evaluator's is ``valid`` instead."""

_SUCCEEDED = frozenset({DONE, Status.VALID})


def stage_task(name: str, sample: str) -> TaskId:
    return TaskId(Kind.STAGE, name, sample)


def evaluator_task(name: str, sample: str | None = None) -> TaskId:
    return TaskId(Kind.EVALUATOR, name, sample)


def evaluations(
    samples: Collection[str], evaluators: Sequence[EvaluatorSpec]
) -> list[tuple[TaskId, EvaluatorSpec]]:
    """Each evaluation of a problem with ``samples`` and ``evaluators``, with the entry that binds
    it, in plan order: those of each sample in the order of ``samples``, each in problem order,
    then those of the candidate."""
    found = [
        (evaluator_task(entry.binding, sample), entry)
        for sample in samples
        for entry in evaluators
        if sample in entry.samples
    ]
    return found + [(evaluator_task(e.binding), e) for e in evaluators if e.stage is None]


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

    The order of ``tasks`` is the order :func:`run_tasks` runs them in: the geometry, then for
    each sample its stages in pipeline order and its evaluations in problem order, then the
    evaluations of the candidate. It is one of the orders in which every task comes after its
    needs, and nothing may depend on which one: another executor may run any of them, or several
    tasks at once.
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
    samples: Mapping[str, Sample],
    evaluators: Sequence[EvaluatorSpec],
    supplied: Collection[TaskId] = (),
) -> Plan:
    """The plan of a problem with ``samples`` and ``evaluators`` (in problem order), the stages
    in ``supplied`` given by the caller.

    The stages planned of a sample are those its evaluators read and the stages these read,
    except that a stage given as a file reads nothing. A stage that runs reads the geometry if it
    is made from it; every evaluation reads the geometry, and one of a sample the stage its
    evaluator reads, with every stage that one reads. A stage of sample ``s`` writes in
    ``samples/s/<stage>``, an evaluation in ``samples/s/evaluators/<binding>``, or in
    ``evaluators/<binding>`` for one of the candidate.
    """
    tasks = [Task(GEOMETRY, directory=GEOMETRY.name)]
    planned = evaluations(samples, evaluators)
    for sample, spec in samples.items():
        files = {name for name, stage in spec.stages.items() if stage.file is not None}
        handed = {task.name for task in supplied if task.sample == sample}
        given = {*handed, *files}
        own = [(task, entry) for task, entry in planned if task.sample == sample]
        targets = [entry.stage for _, entry in own if entry.stage is not None]
        for name, reads in graph(targets, given):
            task = stage_task(name, sample)
            if name in handed:
                tasks.append(Task(task, given="supplied"))
            elif name in files:
                tasks.append(Task(task, given="file"))
            else:
                needs = tuple(stage_task(need, sample) for need in reads)
                tasks.append(
                    Task(
                        task,
                        (GEOMETRY, *needs) if reads_geometry(name) else needs,
                        f"samples/{sample}/{name}" if writes_files(name) else None,
                    )
                )
        for task, entry in own:
            assert entry.stage is not None  # an evaluation of a sample reads one of its stages
            chain = [stage_task(node, sample) for node, _ in graph([entry.stage], given)]
            tasks.append(
                Task(task, (GEOMETRY, *chain), f"samples/{sample}/evaluators/{entry.binding}")
            )
    for task, entry in planned:
        if task.sample is None:
            tasks.append(Task(task, (GEOMETRY,), f"evaluators/{entry.binding}"))
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
