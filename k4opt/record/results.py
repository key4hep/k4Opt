"""What the evaluation of a candidate found: the result of each task, and the candidate's status.

Each task of a candidate's plan (:mod:`k4opt.problem.plan`) ends with a :class:`TaskResult`: its
status, why it did not succeed, what it gives the tasks that read it, the files it made as the
record names them, how it ran. A task whose needs did not all succeed is ``blocked``
(:func:`blocked`) and never runs. The candidate's status and message derive from the results of
its required tasks alone (:func:`candidate_status`, :func:`candidate_message`); a task that is not
required is a diagnostic, listed when it does not succeed (:func:`diagnostics`) but failing
nothing. :class:`CandidateRecord` holds the result of every task of one candidate, which its
``summary.json`` is assembled from (:mod:`k4opt.record.write`) and checked against when read back
(:mod:`k4opt.record.read`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from k4opt.evaluation import (
    Artifact,
    CandidateKey,
    EvaluationContractError,
    EvaluationRecord,
    Measurement,
    Status,
)
from k4opt.frozen import FrozenMap, Json, freeze
from k4opt.problem.models import MetricRef
from k4opt.problem.plan import Kind, Plan, Task, TaskId, stage_task

DONE = "done"
"""The status of a geometry or a stage that is available; an evaluator's is ``valid`` instead."""


_SUCCEEDED = frozenset({DONE, Status.VALID})


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
    def per_task(self) -> int | None:
        """An estimate of what the task needs while it runs, however many jobs it holds (the
        threads of a simulation share its memory): both peaks added; ``None`` if neither
        is known. Not a bound: the peaks need not come at once (it is then high), and a program
        that runs several processes at once counts by the largest (it is then low)."""
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
    stage that ran, the artifacts of an evaluator (the geometry it made is identified by its
    fingerprint, in ``provenance``). ``blocked_by`` holds the tasks that did not
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
    made: Mapping[str, Artifact] = field(default_factory=lambda: FrozenMap({}))
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


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One candidate, measured by every evaluator of its problem.

    ``plan`` holds its tasks, and ``results`` the result of each, in plan order: its geometry,
    the pipeline stages of each sample that its evaluators read, each recorded once, and every
    evaluation, with only its own work. ``elapsed_seconds`` is the wall time of the whole
    evaluation: the tasks and k4opt's checks. Its mappings are read-only copies.

    ``represents_candidate`` is ``False`` when the caller supplied a required stage that the
    parameters change (``k4opt evaluate --input gamma/simulation=FILE``): that file was not made
    from the candidate, so the record must never count as an evaluation of it, whatever its
    status.
    ``warnings`` are the preflight's: collections the simulation was not expected to write, and
    what is measured or simulated twice.
    """

    candidate_key: CandidateKey
    plan: Plan
    results: Mapping[TaskId, TaskResult]
    elapsed_seconds: float
    represents_candidate: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = [task.id.label for task in self.plan.tasks if task.id not in self.results]
        if missing or len(self.results) != len(self.plan.tasks):
            raise EvaluationContractError(
                f"the results do not match the plan's tasks (missing: {', '.join(missing)})"
            )
        results = {task.id: self.results[task.id] for task in self.plan.tasks}
        object.__setattr__(self, "results", FrozenMap(results))

    @property
    def evaluations(self) -> Mapping[str, EvaluationRecord]:
        """The record of each evaluation, by its label (``gamma/calo.photon``, ``readout``), in
        plan order; a blocked one's evaluator was never called."""
        return FrozenMap(
            {
                task.id.label: record
                for task in self.plan.of(Kind.EVALUATOR)
                if (record := self.results[task.id].record) is not None
            }
        )

    @property
    def records(self) -> tuple[EvaluationRecord, ...]:
        """The record of each evaluation, in plan order."""
        return tuple(self.evaluations.values())

    @property
    def supplied(self) -> frozenset[str]:
        """The stages the caller supplied as files, by label (``gamma/simulation``)."""
        return frozenset(t.id.label for t in self.plan.of(Kind.STAGE) if t.given == "supplied")

    def stage(self, name: str, sample: str) -> TaskResult | None:
        """The result of the stage ``name`` of ``sample``, ``None`` if it was not planned."""
        return self.results.get(stage_task(name, sample))

    @property
    def status(self) -> Status:
        """From the required tasks alone: ``failed`` if one failed (the evaluation is
        incomplete), otherwise ``invalid`` if an evaluator found the result unusable (and the
        tasks that read it were not run), otherwise ``valid``. A diagnostic changes nothing."""
        return candidate_status(self.plan, self.results)

    @property
    def message(self) -> str | None:
        """Every required task that failed or is invalid, with its message, then the required
        ones they blocked."""
        return candidate_message(self.plan, self.results)

    @property
    def diagnostics(self) -> tuple[TaskResult, ...]:
        """The result of each task that is not required and did not succeed, in plan order."""
        return tuple(diagnostics(self.plan, self.results))

    def record(self, label: str) -> EvaluationRecord:
        """The record of the evaluation ``label`` (``gamma/calo.photon``, ``readout``);
        :class:`KeyError` if there is none."""
        return self.evaluations[label]

    def value(self, ref: MetricRef) -> Measurement | None:
        """The measurement of the metric ``ref`` by the evaluation it names.

        ``None`` unless that evaluation's result is ``valid``: the diagnostic values of an
        ``invalid`` or ``failed`` result never stand in for an objective.
        """
        try:
            result = self.record(ref.invocation).result
        except KeyError:
            return None
        if result.status is not Status.VALID:
            return None
        return next(
            (m for m in result.measurements if (m.stage, m.metric) == (ref.stage, ref.metric)),
            None,
        )
