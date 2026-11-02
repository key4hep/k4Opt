"""Internal: the work of one candidate as tasks, each run at most once and only after what it reads.

A task is the smallest step with a meaning of its own: the candidate's ``geometry`` (materialized
once, read by every simulation and every evaluator), one pipeline stage of one sample
(``gamma/simulation``), or one evaluation: an evaluator, named by its binding, on one sample
(``gamma/calo.photon``) or on the candidate (``readout``); a combined evaluator is an evaluation of
the candidate that reads, by role, evaluations or stages of several samples. Each task names the
tasks it reads, its ``needs``, and nothing else orders it. Three seams keep what runs apart
from how it runs:

* :func:`plan_tasks` resolves the dependencies from the problem alone, once per session, so that
  a record can be checked against the same plan
  (:meth:`~k4opt.record.Assessment.from_summary`);
* the scheduler (:mod:`k4opt.execution._schedule`) starts each task once what it reads ended; a
  task whose needs did not all succeed is ``blocked`` (:func:`blocked`) and is not run. Where
  and how a task does its work is a backend's (:mod:`k4opt.execution._backends`);
* the candidate's record is assembled from the task results alone
  (:mod:`k4opt.record.write`), and its status and message are derived from them
  (:func:`candidate_status`, :func:`candidate_message`).

A task is ``required`` if an objective, a constraint or a binding marked ``required: true`` needs
it: the evaluation that measures what one names, and every task it reads, transitively. Only the
required tasks decide the candidate's status; every other task is a diagnostic, run and recorded
like any other, and one that does not succeed is listed (:func:`diagnostics`) but fails nothing.

A stage given as a file, in the problem or by the caller, is a task too: it reads that file and
no task, so the stages before it are planned only if something else needs them. The samples
share nothing but the geometry: a task of one sample never reads a task of another, and only a
combined evaluator reads tasks of several.

A result names the files its task made in its directory (the events of a simulation, an
evaluator's artifacts) as the record does; the coordinator watches them, and the geometry, for
changes until the candidate ends (:class:`~k4opt.execution._files.Observed`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

from k4opt.evaluation import EvaluationContractError
from k4opt.problem.models import EvaluatorSpec, MetricRef, Reference, Sample
from k4opt.problem.stages import STAGE_TYPES, graph, reads_geometry, writes_files


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


def read_stages(sample: str, evaluators: Sequence[EvaluatorSpec]) -> tuple[str, ...]:
    """The stages of ``sample`` that tasks read directly: those its evaluators read, in problem
    order, then those roles read, each once."""
    measured = [entry.stage for entry in evaluators if entry.stage and sample in entry.samples]
    roles = [i.stage for e in evaluators for i in e.reads if i.stage and i.sample == sample]
    return tuple(dict.fromkeys([*measured, *roles]))


def reference_task(item: Reference) -> TaskId:
    """The task a role item reads: an evaluation, or a stage, of its sample."""
    if item.evaluator is not None:
        return evaluator_task(item.evaluator, item.sample)
    assert item.stage is not None
    return stage_task(item.stage, item.sample)


@dataclass(frozen=True, slots=True)
class Task:
    """One task of a plan: the tasks it reads, and where it writes.

    ``directory`` is relative to the candidate's run directory, ``None`` for a task that writes
    nothing. ``given`` says how a stage that does not run is given: as a ``file`` of the problem,
    or ``supplied`` by the caller. ``required`` says whether the candidate's status depends on
    it; a task that is not required is a diagnostic.
    """

    id: TaskId
    needs: tuple[TaskId, ...] = ()
    directory: str | None = None
    given: Literal["file", "supplied"] | None = None
    required: bool = False


@dataclass(frozen=True, slots=True)
class Plan:
    """The tasks of every candidate of a problem, each after the tasks it reads.

    The order of ``tasks`` is the order they run in with one job: the geometry, then for
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

    @property
    def diagnostics(self) -> tuple[Task, ...]:
        """The tasks that are not required, in plan order."""
        return tuple(task for task in self.tasks if not task.required)


def plan_tasks(
    samples: Mapping[str, Sample],
    evaluators: Sequence[EvaluatorSpec],
    supplied: Collection[TaskId] = (),
    *,
    references: Collection[MetricRef],
) -> Plan:
    """The plan of a problem with ``samples`` and ``evaluators`` (in problem order), the stages
    in ``supplied`` given by the caller, and the metrics its objectives and constraints name,
    ``references``.

    The stages planned of a sample are those its evaluators and the roles of combined evaluators
    read (:func:`read_stages`) and the stages these read, except that a stage given as a file
    reads nothing. A stage that runs reads the geometry if it is made from it; every evaluation
    reads the geometry, one of a sample the stage its evaluator reads, with every stage that one
    reads, and a combined evaluator each item of its roles. A stage of sample ``s`` writes in
    ``samples/s/<stage>``, an evaluation in ``samples/s/evaluators/<binding>``, or in
    ``evaluators/<binding>`` for one of the candidate.

    The required tasks are the evaluations that measure a metric of ``references``, those of an
    entry marked ``required``, and every task these read, transitively.
    """
    tasks = [Task(GEOMETRY, directory=GEOMETRY.name)]
    planned = evaluations(samples, evaluators)
    for sample, spec in samples.items():
        files = {name for name, stage in spec.stages.items() if stage.file is not None}
        handed = {task.name for task in supplied if task.sample == sample}
        given = {*handed, *files}
        own = [(task, entry) for task, entry in planned if task.sample == sample]
        for name, reads in graph(read_stages(sample, evaluators), given):
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
            needs = (GEOMETRY, *(reference_task(item) for item in entry.reads))
            tasks.append(Task(task, needs, f"evaluators/{entry.binding}"))
    needed = {evaluator_task(ref.evaluator, ref.sample) for ref in references}
    needed |= {task for task, entry in planned if entry.required}
    for each in reversed(tasks):  # each after its needs, so a need is reached after its readers
        if each.id in needed:
            needed.update(each.needs)
    assert needed <= {each.id for each in tasks}  # the problem names only evaluations it has
    return Plan(tuple(replace(each, required=each.id in needed) for each in tasks))
