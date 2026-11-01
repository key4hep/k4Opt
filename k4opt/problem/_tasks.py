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
  (:meth:`~k4opt.problem.Assessment.from_summary`);
* :func:`run_tasks` executes a plan one task after another. A task whose needs did not all
  succeed is ``blocked`` (:func:`blocked`) and is not run; the work of a task is the caller's.
  Another executor (processes, a batch system) would replace only this loop;
* the candidate's record is assembled from the task results alone (:mod:`k4opt.problem._run`),
  and its status and message are derived from them (:func:`candidate_status`,
  :func:`candidate_message`).

A task is ``required`` if an objective, a constraint or a binding marked ``required: true`` needs
it: the evaluation that measures what one names, and every task it reads, transitively. Only the
required tasks decide the candidate's status; every other task is a diagnostic, run and recorded
like any other, and one that does not succeed is listed (:func:`diagnostics`) but fails nothing.

A stage given as a file, in the problem or by the caller, is a task too: it reads that file and
no task, so the stages before it are planned only if something else needs them. The samples
share nothing but the geometry: a task of one sample never reads a task of another, and only a
combined evaluator reads tasks of several.

The files a task makes in its directory (the events of a simulation, an evaluator's artifacts)
are :class:`Made`: what the record says of each, and the file's ``stat`` when it was made. A file
whose stat changed is no longer what the record says (:meth:`Made.changed`), so the tasks that
read it check it before and after they run (:func:`changed_files`), and the candidate checks all
of them at its end. The files of the geometry, which the record identifies together by their
fingerprint and every simulation of every sample reads, are watched as one :class:`Tree`. The
stats are machine-specific and never recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal

from k4opt.evaluation import Artifact, EvaluationContractError, EvaluationRecord, Status
from k4opt.frozen import FrozenMap, Json, freeze
from k4opt.problem._stages import (
    STAGE_TYPES,
    file_identity,
    graph,
    reads_geometry,
    writes_files,
)
from k4opt.problem.models import EvaluatorSpec, MetricRef, Reference, Sample


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


type Stat = tuple[int, int, int, int]
"""What tells a file apart from a rewritten one: its size, mtime, ctime and inode."""


def _stat(path: Path) -> Stat | None:
    try:
        status = path.stat()
    except OSError:
        return None
    return (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)


@dataclass(frozen=True, slots=True)
class Made:
    """A file a task made in its own directory: its ``artifact`` as the record names it, and its
    ``stat`` when it was made."""

    artifact: Artifact
    stat: Stat

    @classmethod
    def file(cls, task: TaskId, name: str, path: Path, *, hashed: bool) -> Made:
        """The file ``path`` that ``task`` made as ``name``, with its SHA-256 if ``hashed``;
        raises :class:`OSError` if it cannot be read, or if it changes while it is hashed (its
        size and its hash would describe other versions of it)."""
        stat = _stat(path)
        if stat is None or not path.is_file():
            raise OSError(f"{path} is no file")
        sha256 = file_identity(path)["sha256"] if hashed else None
        assert sha256 is None or isinstance(sha256, str)
        if hashed and _stat(path) != stat:
            raise OSError(f"{path} changed while it was hashed")
        return cls(Artifact(f"{task.label}#{name}", path, stat[0], sha256), stat)

    def changed(self) -> bool:
        """Whether the file is no longer what the record says: it is gone, or its stat changed
        and it is not hashed, or hashed again it has other bytes. Equal stats are trusted."""
        now = _stat(self.artifact.path)
        if now == self.stat:
            return False
        if now is None or self.artifact.sha256 is None:
            return True
        try:
            return file_identity(self.artifact.path)["sha256"] != self.artifact.sha256
        except OSError:
            return True


@dataclass(frozen=True, slots=True)
class Tree:
    """Every file below ``root``, a directory a task made, by its path relative to ``root``, with
    its stat when the task was done: a tree the record identifies as a whole (the geometry, by
    its fingerprint), so any file in it that changes, goes or is added changes it."""

    root: Path
    stats: Mapping[str, Stat]

    @classmethod
    def of(cls, root: Path) -> Tree:
        stats = {
            path.relative_to(root).as_posix(): stat
            for path in sorted(root.rglob("*"))
            if not path.is_dir() and (stat := _stat(path)) is not None
        }
        return cls(root, FrozenMap(stats))

    def changed(self) -> list[str]:
        """The files of the tree, by relative path, that are not as they were."""
        now = Tree.of(self.root).stats
        return [
            name
            for name in dict.fromkeys([*self.stats, *now])
            if now.get(name) != self.stats.get(name)
        ]


def changed_files(results: Iterable[TaskResult]) -> list[str]:
    """The files that the tasks of ``results`` made and that changed since: the reference of
    each, or for a tree its path below the task (``geometry/compact/ECalBarrel.xml``)."""
    changed = []
    for result in results:
        if result.tree is not None:
            changed += [f"{result.id.label}/{name}" for name in result.tree.changed()]
        changed += [made.artifact.ref for made in result.made.values() if made.changed()]
    return changed


@dataclass(frozen=True, slots=True)
class PeakMemory:
    """The peak resident memory of a task, in bytes, ``None`` where it is not known.

    ``process`` is that of the k4opt process that did the task's work (a worker process, or the
    session's own with one job) while it did it, all it held included; ``program`` that of the
    program a stage ran (ddsim), the largest of it and of each process it started and waited for.
    """

    process: int | None = None
    program: int | None = None

    @property
    def per_job(self) -> int | None:
        """What one job needs for the task: both peaks added, as if they came at once (they
        may not, so this is at most what it held); ``None`` if neither is known."""
        if self.process is None and self.program is None:
            return None
        return (self.process or 0) + (self.program or 0)

    def to_dict(self) -> dict[str, int | None]:
        return {"process": self.process, "program": self.program}


@dataclass(frozen=True, slots=True)
class TaskResult:
    """What one task left behind.

    ``status`` is ``done``, ``failed`` or ``blocked`` for the geometry and a stage, and the
    status of its result (``valid``, ``invalid``, ``failed``) or ``blocked`` for an evaluator;
    ``message`` says why one did not succeed. ``output`` is what the tasks that read it are given,
    by its name (the compact file of the geometry, the events of a stage), ``None`` if it gives
    nothing (a particle gun); ``artifacts`` are its other files (logs) and ``provenance`` how it
    ran. ``made`` holds, by name, the files it made that its record identifies: the output of a
    stage that ran, the artifacts of an evaluator; ``tree`` the files of the geometry it made,
    which its record identifies together. ``blocked_by`` holds the tasks that did not
    succeed and kept a ``blocked`` one from running; ``record`` the evaluator's record, blocked
    ones included. ``elapsed_seconds`` is
    the wall time k4opt spent on it, ``None`` if it did not run; ``memory`` its peak memory,
    ``None`` if it did not run.
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
    made: Mapping[str, Made] = field(default_factory=lambda: FrozenMap({}))
    tree: Tree | None = None
    memory: PeakMemory | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", FrozenMap(self.artifacts))
        object.__setattr__(self, "made", FrozenMap(self.made))
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


def _required(plan: Plan, results: Mapping[TaskId, TaskResult]) -> list[TaskResult]:
    return [results[task.id] for task in plan.tasks if task.required]


def candidate_status(plan: Plan, results: Mapping[TaskId, TaskResult]) -> Status:
    """The status of a candidate from the ``results`` of the required tasks of its ``plan``:
    ``failed`` if any failed (the evaluation is incomplete), otherwise ``invalid`` if any is
    invalid or blocked (by an invalid one, since a required task reads only required ones),
    otherwise ``valid``. A diagnostic never changes it."""
    seen = {result.status for result in _required(plan, results)}
    if Status.FAILED in seen:
        return Status.FAILED
    return Status.INVALID if seen & {Status.INVALID, Status.BLOCKED} else Status.VALID


def candidate_message(plan: Plan, results: Mapping[TaskId, TaskResult]) -> str | None:
    """The message of a candidate from the ``results`` of the required tasks of its ``plan``, in
    plan order: each that failed or is invalid, with its status and message, then the tasks
    they blocked."""
    return _message(_required(plan, results))


def diagnostics(plan: Plan, results: Mapping[TaskId, TaskResult]) -> list[TaskResult]:
    """The results of the tasks of ``plan`` that are not required and did not succeed, in plan
    order: what the candidate's status leaves out."""
    return [result for task in plan.diagnostics if not (result := results[task.id]).succeeded]


def _message(results: Sequence[TaskResult]) -> str | None:
    causes = [
        f"{result.id.label} {result.status}: {result.message}"
        for result in results
        if result.status in (Status.FAILED, Status.INVALID)
    ]
    held = [result.id.label for result in results if result.status == Status.BLOCKED]
    if held:
        causes.append(f"blocked: {', '.join(held)}")
    return "; ".join(causes) or None
