"""The assessment of a candidate: its objective values, its constraints judged, its feasibility.

It depends on no optimizer: an optimizer, a dashboard and a person read the same judgment. It is
written into ``summary.json``, and :meth:`Assessment.from_summary` recomputes it from the
record's raw measurements, so any reader can check a record before trusting it.

A constraint is judged on the central value of its metric: ``max`` is satisfied when the value is
at most the limit, ``min`` when it is at least the limit, both inclusive. The uncertainty is
recorded next to the value but not used. Only a ``valid`` evaluation is judged, and only one
that represents its candidate is called feasible or not.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, NamedTuple, cast

from k4opt.evaluation import EvaluationContractError, Measurement, Scalar, Status
from k4opt.frozen import thaw
from k4opt.problem._evaluate import CandidateRecord, MetricKey, _validate_candidate
from k4opt.problem._stages import STAGE_TYPES, changed_by_geometry
from k4opt.problem._tasks import (
    DONE,
    GEOMETRY,
    Kind,
    TaskId,
    TaskResult,
    blocked,
    candidate_message,
    candidate_status,
    evaluations,
    evaluator_task,
    plan_tasks,
    stage_task,
)
from k4opt.problem.errors import CandidateError, RecordError
from k4opt.problem.models import Constraint, Direction, MetricRef, Objective, ProblemSpec

SUMMARY_SCHEMA = "k4opt.run/1"

type Bound = Literal["min", "max"]
type _Lookup = Callable[[MetricRef], Measurement | None]

_BOUNDS: tuple[Bound, ...] = ("min", "max")


@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    """The value of one objective for a candidate: its metric, its direction and the measurement
    of that metric by the evaluator the objective names, on its ``sample`` (``None`` for an
    evaluator that measures the candidate), with its unit and uncertainty."""

    evaluator: str
    stage: str
    metric: str
    direction: Direction
    value: int | float
    unit: str | None = None
    uncertainty: int | float | None = None
    sample: str | None = None

    @property
    def ref(self) -> MetricRef:
        return MetricRef(self.evaluator, self.stage, self.metric, self.sample)

    @property
    def label(self) -> str:
        """``[sample/]evaluator/stage/metric``, as the optimizer and the exports name the
        objective."""
        return self.ref.label

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluator": self.evaluator,
            "sample": self.sample,
            "stage": self.stage,
            "metric": self.metric,
            "direction": self.direction,
            "value": self.value,
            "unit": self.unit,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class Limit:
    """One bound of a constraint, and how the candidate's measurement compares with it.

    ``evaluator``, ``stage`` and ``metric`` name the metric the constraint limits, and ``sample``
    the sample it is measured on (``None`` for an evaluator that measures the candidate).
    ``limit`` is
    the bound as the problem declares it, and ``unit`` the unit of the measurement (the
    constraint's own when there is no measurement). ``margin`` is ``value - max`` or
    ``min - value`` in that unit, zero or below when the limit is ``satisfied``: equality
    satisfies it, with margin 0. The margin is computed exactly and rounded once, so its sign is
    always that of the exact comparison; it is ``None`` if it does not fit in a float, while
    ``satisfied`` stays exact. ``value``, its ``uncertainty`` (recorded, not used), ``margin``
    and ``satisfied`` are ``None`` when the evaluation is not ``valid``.
    """

    evaluator: str
    stage: str
    metric: str
    bound: Bound
    limit: int | float
    unit: str | None = None
    value: int | float | None = None
    uncertainty: int | float | None = None
    margin: float | None = None
    satisfied: bool | None = None
    sample: str | None = None

    @property
    def ref(self) -> MetricRef:
        return MetricRef(self.evaluator, self.stage, self.metric, self.sample)

    @property
    def label(self) -> str:
        """``[sample/]evaluator/stage/metric`` of the metric it limits."""
        return self.ref.label

    @property
    def name(self) -> str:
        """``[sample/]evaluator/stage/metric:bound``, such as
        ``gamma/calo.photon/simulation/ecal_barrel.angular_residual:max``; unique within a
        problem."""
        return f"{self.label}:{self.bound}"

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluator": self.evaluator,
            "sample": self.sample,
            "stage": self.stage,
            "metric": self.metric,
            "bound": self.bound,
            "limit": self.limit,
            "unit": self.unit,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "margin": self.margin,
            "satisfied": self.satisfied,
        }


@dataclass(frozen=True, slots=True)
class Assessment:
    """An evaluation judged against the problem's objectives and constraints.

    ``objectives`` holds the value of each objective (:class:`ObjectiveValue`), in the problem's
    order, and is ``None`` unless the evaluation is ``valid``: there is never a partial vector.
    ``limits`` holds a :class:`Limit` for each bound of each constraint, in the problem's order
    (``min`` before ``max``). ``feasible`` is ``True`` when every limit is satisfied (so also
    without constraints), ``False`` when one is not, and ``None`` when the candidate is not
    assessed: its evaluation is ``invalid`` or ``failed``, or a supplied stage was not made from it
    (``represents_candidate`` is false), in which case the objectives and limits are still given as
    measured.
    """

    objectives: tuple[ObjectiveValue, ...] | None
    limits: tuple[Limit, ...] = ()
    feasible: bool | None = None

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-compatible data, as ``summary.json`` holds it under ``assessment``."""
        objectives = self.objectives
        return {
            "feasible": self.feasible,
            "objectives": None if objectives is None else [o.to_dict() for o in objectives],
            "limits": [limit.to_dict() for limit in self.limits],
        }

    @classmethod
    def from_summary(cls, spec: ProblemSpec, summary: Mapping[str, object]) -> Assessment:
        """Recompute the assessment of a run record (``summary.json`` as read back) and check
        the record against it.

        The record's ``implementations`` must be those the problem binds, with its versions and
        stages; its ``samples`` the problem's, each with the evaluations of the bindings that
        measure it, and its ``evaluations`` those of the bindings that measure the candidate, each
        of the implementation the problem binds it to, with the settings (``declared``) and
        collections the problem gives it. The candidate's tasks are planned again from the problem
        and the stages the record says were ``supplied``: each must read what the plan says, be
        ``blocked`` exactly when a task
        it reads did not succeed (by the tasks the plan makes it blocked by, with their message)
        and otherwise have run. The candidate's status and message are recomputed from its tasks'
        (``failed`` before ``invalid`` before ``valid``), and so is whether it represents its
        candidate; each objective and constraint value is taken from the own measurements of the
        evaluation it names, if that evaluation is ``valid``. The rules a live evaluation
        enforces are checked again: the candidate is in the problem's domains and
        ``candidate_key`` is its key, an ``invalid`` or ``failed`` task has a message, a blocked
        evaluator measured nothing, a measurement of a metric its implementation declares carries
        the declared unit, a binding whose implementation declares metrics declares those named
        of it, and a constraint's unit is that of its metric. Raises
        :class:`~k4opt.problem.RecordError` if the record is malformed or breaks one of those
        rules, was made for another problem (its definition differs other than in ``name``), or if
        its ``status``, ``message``, ``represents_candidate``, ``metrics`` or ``assessment``
        disagree with the recomputation.
        """
        record = _mapping(summary, "record")
        if record.get("schema") != SUMMARY_SCHEMA:
            raise RecordError(f"schema is {record.get('schema')!r}, not {SUMMARY_SCHEMA!r}")
        _check_problem(spec, record)
        _check_candidate(spec, record)
        declared = _implementations(spec, record)
        samples = _samples(spec, record)
        evaluators = _evaluations(spec, record, samples, declared)
        supplied = _supplied(spec, record)
        results = _results(spec, record, samples, supplied, evaluators)
        status = candidate_status(result.status for result in results)
        if record.get("status") != status:
            raise RecordError(
                f"status is {record.get('status')!r}, but its tasks make it {status.value!r}"
            )
        message = candidate_message(results)
        if record.get("message") != message:
            raise RecordError(
                f"message is {record.get('message')!r}, but its tasks make it {message!r}"
            )
        represents = not any(changed_by_geometry(task.name) for task in supplied)
        if record.get("represents_candidate") is not represents:
            raise RecordError(
                f"represents_candidate is {record.get('represents_candidate')!r}, but its "
                f"supplied stages make it {represents}"
            )
        for ref in spec.references:
            metrics = evaluators[_task(ref)].declared
            if metrics is not None and (ref.stage, ref.metric) not in metrics:
                raise RecordError(f"metric {ref.label}: its evaluator does not declare it")

        def value(ref: MetricRef) -> Measurement | None:
            owner = evaluators[_task(ref)]
            if owner.status is not Status.VALID:
                return None
            found = owner.measured.get((ref.stage, ref.metric))
            if found is None:
                raise RecordError(
                    f"evaluator {ref.invocation!r} is valid but did not measure "
                    f"{ref.stage}/{ref.metric}, which the problem names"
                )
            return found

        _check_metrics(spec, record, value)
        _check_units(spec, evaluators, value)
        assessment = _judge(spec, status, represents, value)
        if record.get("assessment") != _plain(assessment.to_dict()):
            raise RecordError(
                "its assessment is not the one its measurements give: "
                f"recorded {record.get('assessment')!r}, recomputed {assessment.to_dict()!r}"
            )
        return assessment


def assess(spec: ProblemSpec, candidate: CandidateRecord) -> Assessment:
    """The assessment of a candidate just evaluated."""
    return _judge(spec, candidate.status, candidate.represents_candidate, candidate.value)


def _task(ref: MetricRef) -> TaskId:
    """The evaluation that measures the metric ``ref``."""
    return evaluator_task(ref.evaluator, ref.sample)


def _judge(spec: ProblemSpec, status: Status, represents: bool, value: _Lookup) -> Assessment:
    """The one way an assessment is made, live or from a record: ``value(ref)`` is the
    measurement of an objective or constraint metric by the binding that ``ref`` names."""
    bounds = [
        (constraint, bound)
        for constraint in spec.constraints
        for bound in _BOUNDS
        if getattr(constraint, bound) is not None
    ]
    if status is not Status.VALID:
        return Assessment(None, tuple(_limit(c, bound, None) for c, bound in bounds), None)
    objectives = tuple(_objective(o, _required(value, o.ref)) for o in spec.objectives)
    limits = tuple(_limit(c, b, _required(value, c.ref)) for c, b in bounds)
    feasible = all(limit.satisfied for limit in limits) if represents else None
    return Assessment(objectives, limits, feasible)


def _required(value: _Lookup, ref: MetricRef) -> Measurement:
    measurement = value(ref)
    if measurement is None:  # a valid result carries every metric named of it (_check_result)
        raise RecordError(f"a valid evaluation has no value of {ref.label}")
    return measurement


def _objective(objective: Objective, m: Measurement) -> ObjectiveValue:
    return ObjectiveValue(
        objective.evaluator,
        objective.stage,
        objective.metric,
        objective.direction,
        m.value,
        m.unit,
        m.uncertainty,
        objective.sample,
    )


def _limit(constraint: Constraint, bound: Bound, measurement: Measurement | None) -> Limit:
    limit = constraint.min if bound == "min" else constraint.max
    assert limit is not None
    ref = (constraint.evaluator, constraint.stage, constraint.metric)
    if measurement is None:
        return Limit(*ref, bound, limit, constraint.unit, sample=constraint.sample)
    # exact: Fraction holds every float and int, and float() rounds the difference once
    excess = Fraction(measurement.value) - Fraction(limit)
    if bound == "min":
        excess = -excess
    try:
        margin: float | None = float(excess)
    except OverflowError:
        margin = None
    return Limit(
        *ref,
        bound,
        limit,
        measurement.unit,
        measurement.value,
        measurement.uncertainty,
        margin,
        excess <= 0,
        constraint.sample,
    )


def _measurement(m: Measurement | None) -> dict[str, object]:
    return {
        "stage": None if m is None else m.stage,
        "metric": None if m is None else m.metric,
        "value": None if m is None else m.value,
        "unit": None if m is None else m.unit,
        "uncertainty": None if m is None else m.uncertainty,
    }


# --- reading a record ------------------------------------------------------------------------


def _plain(value: object) -> object:
    """``value`` as it reads back from JSON."""
    return json.loads(json.dumps(value, allow_nan=False))


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecordError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise RecordError(f"{where} must be a list, got {type(value).__name__}")
    return value


def _check_problem(spec: ProblemSpec, record: Mapping[str, object]) -> None:
    """The record must have been made for ``spec``; its ``name`` (a label) may differ."""
    definition = _mapping(
        _mapping(record.get("problem"), "problem").get("definition"), "definition"
    )
    expected = _plain(spec.to_dict())
    assert isinstance(expected, dict)
    keys = dict.fromkeys([*expected, *definition])
    differ = [k for k in keys if k != "name" and definition.get(k) != expected.get(k)]
    if differ:
        raise RecordError(f"it was made for another problem: its {', '.join(differ)} differ")


class _Fragment(NamedTuple):
    """A task's part of a record, as far as its status reads it."""

    status: str
    message: str | None
    reads: list[object]
    blocked_by: list[object]


def _fragment(entry: Mapping[str, object], where: str, statuses: Collection[str]) -> _Fragment:
    status = entry.get("status")
    if not isinstance(status, str) or status not in statuses:
        raise RecordError(f"{where}: status {status!r} is none of {', '.join(statuses)}")
    message = entry.get("message")
    if message is not None and not isinstance(message, str):
        raise RecordError(f"{where}: message must be a string, got {message!r}")
    if status not in (DONE, Status.VALID) and not (message and message.strip()):
        raise RecordError(f"{where}: a task that is {status} needs a message saying why")
    reads = _list(entry.get("reads"), f"{where}: reads")
    causes = _list(entry.get("blocked_by"), f"{where}: blocked_by")
    return _Fragment(status, message, reads, causes)


@dataclass(frozen=True, slots=True)
class _Evaluator:
    """One evaluation of a record, as far as the assessment reads it."""

    status: Status
    message: str | None
    measured: Mapping[MetricKey, Measurement]  # its own measurements
    declared: Mapping[MetricKey, str] | None  # the metrics its implementation declares, with units
    fragment: _Fragment  # its status, and the tasks it reads


def _check_candidate(spec: ProblemSpec, record: Mapping[str, object]) -> None:
    """The record's candidate is one of the problem's, and its ``candidate_key`` is that
    candidate's, as a live evaluation validates and keys it."""
    candidate = _mapping(record.get("candidate"), "candidate")
    try:  # the values are checked for their types and domains here
        _, key = _validate_candidate(spec, cast(Mapping[str, Scalar], candidate))
    except CandidateError as exc:
        raise RecordError(f"its candidate is not one of the problem's: {exc}") from None
    expected = [list(part) for part in key]
    if record.get("candidate_key") != expected:
        raise RecordError(
            f"candidate_key is {record.get('candidate_key')!r}, but its candidate is {expected!r}"
        )


def _implementations(
    spec: ProblemSpec, record: Mapping[str, object]
) -> dict[str, dict[MetricKey, str] | None]:
    """The metrics each implementation of the record declares (``None``: none), by name; they
    must be those the problem binds, in the order it first binds them, with its versions and
    reading the stage its entries say."""
    implementations = _mapping(record.get("implementations"), "implementations")
    versions = {entry.name: entry.version for entry in spec.evaluators}
    if list(implementations) != list(versions):
        raise RecordError(
            f"its implementations are {list(implementations)}, but the problem binds "
            f"{list(versions)}"
        )
    declared: dict[str, dict[MetricKey, str] | None] = {}
    for name, entry in implementations.items():
        where = f"implementation {name!r}"
        implementation = _mapping(entry, where)
        if implementation.get("version") != versions[name]:
            raise RecordError(
                f"{where}: version {implementation.get('version')!r}, but the problem binds "
                f"version {versions[name]!r}"
            )
        for entry in spec.evaluators:
            if entry.name == name and implementation.get("stage") != entry.stage:
                raise RecordError(
                    f"{where}: stage {implementation.get('stage')!r}, but the problem has it read "
                    f"{entry.stage!r}"
                )
        declared[name] = _declared(implementation.get("metrics"), where)
    return declared


def _samples(spec: ProblemSpec, record: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """The part of the record of each sample, which must be the problem's, in its order."""
    samples = _mapping(record.get("samples"), "samples")
    if list(samples) != list(spec.samples):
        raise RecordError(
            f"its samples are {list(samples)}, but the problem's are {list(spec.samples)}"
        )
    return {name: _mapping(part, f"samples.{name}") for name, part in samples.items()}


def _evaluations(
    spec: ProblemSpec,
    record: Mapping[str, object],
    samples: Mapping[str, Mapping[str, object]],
    declared: Mapping[str, Mapping[MetricKey, str] | None],
) -> dict[TaskId, _Evaluator]:
    """Each evaluation of the record, in plan order, checked as a live result is: of the
    implementation the problem binds it to, with the settings and collections the problem gives
    it. Those of each sample are in its part of the record, those of the candidate in
    ``evaluations``, each by binding in problem order."""
    held: dict[str | None, Mapping[str, object]] = {
        name: _mapping(part.get("evaluations"), f"samples.{name}.evaluations")
        for name, part in samples.items()
    }
    held[None] = _mapping(record.get("evaluations"), "evaluations")
    planned = evaluations(spec.samples, spec.evaluators)
    for sample, found in held.items():
        expected = [entry.binding for task, entry in planned if task.sample == sample]
        if list(found) != expected:
            where = "its evaluations" if sample is None else f"samples.{sample}.evaluations"
            raise RecordError(f"{where} are {list(found)}, but the problem's are {expected}")
    evaluators: dict[TaskId, _Evaluator] = {}
    for task, spec_entry in planned:
        where = f"evaluator {task.label!r}"
        entry = _mapping(held[task.sample][spec_entry.binding], where)
        if entry.get("implementation") != spec_entry.name:
            raise RecordError(
                f"{where}: implementation {entry.get('implementation')!r}, but the problem binds "
                f"{spec_entry.name!r}"
            )
        # as the problem wrote them; the settings used and the collections of roles the problem
        # leaves to their defaults depend on the evaluator's code, which a record is read without
        given = _plain(thaw(spec_entry.settings))
        recorded = _mapping(entry.get("settings"), f"{where}: settings").get("declared")
        if recorded != given:
            raise RecordError(
                f"{where}: settings.declared {recorded!r}, but the problem gives {given!r}"
            )
        collections = _mapping(entry.get("collections"), f"{where}: collections")
        named = {role: collections.get(role) for role in spec_entry.collections}
        if named != dict(spec_entry.collections):
            raise RecordError(
                f"{where}: collections {named!r}, but the problem gives "
                f"{dict(spec_entry.collections)!r}"
            )
        fragment = _fragment(entry, where, set(Status))
        metrics = declared[spec_entry.name]
        own: dict[MetricKey, Measurement] = {}
        for index, item in enumerate(_list(entry.get("measurements"), f"{where}: measurements")):
            at = f"{where}: measurements[{index}]"
            measurement = _measured(_mapping(item, at), at)
            key = (measurement.stage, measurement.metric)
            if key in own:
                raise RecordError(f"{at}: {key[0]}/{key[1]} is measured twice")
            if metrics is not None and key in metrics and measurement.unit != metrics[key]:
                # whatever the status, as live: one metric identity never carries two units
                raise RecordError(
                    f"{at}: {key[0]}/{key[1]} is in {measurement.unit!r}, but the evaluator "
                    f"declares {metrics[key]!r}"
                )
            own[key] = measurement
        if fragment.status == Status.BLOCKED and own:
            raise RecordError(f"{where}: it is blocked, so it was never called, yet it measured")
        status = Status(fragment.status)
        evaluators[task] = _Evaluator(status, fragment.message, own, metrics, fragment)
    return evaluators


def _supplied(spec: ProblemSpec, record: Mapping[str, object]) -> list[TaskId]:
    """The stages the caller supplied, as the record lists them: distinct stages of the
    problem's samples (``gamma/simulation``), sorted."""
    supplied = _list(record.get("supplied"), "supplied")
    tasks = []
    for label in supplied:
        sample, _, stage = label.partition("/") if isinstance(label, str) else ("", "", "")
        if sample not in spec.samples or stage not in STAGE_TYPES:
            raise RecordError(
                f"supplied {supplied!r} must list stages of the problem's samples, as "
                f"SAMPLE/STAGE with a stage of {', '.join(STAGE_TYPES)}"
            )
        tasks.append(stage_task(stage, sample))
    labels = cast(list[str], supplied)
    if labels != sorted(set(labels)):
        raise RecordError(f"supplied {labels!r} must list each stage once, in sorted order")
    return tasks


_STAGE_STATUSES = (DONE, Status.FAILED, Status.BLOCKED)


def _results(
    spec: ProblemSpec,
    record: Mapping[str, object],
    samples: Mapping[str, Mapping[str, object]],
    supplied: Collection[TaskId],
    evaluators: Mapping[TaskId, _Evaluator],
) -> list[TaskResult]:
    """The status of each task of the record, in plan order, checked against the plan of the
    problem: what each task reads, and which are blocked and by what."""
    plan = plan_tasks(spec.samples, spec.evaluators, supplied)
    planned = [task.id for task in plan.of(Kind.STAGE)]
    unread = [task.label for task in supplied if task not in planned]
    if unread:
        raise RecordError(f"it supplied {', '.join(unread)}, which no evaluator reads")
    fragments: dict[TaskId, _Fragment] = {}
    for sample, part in samples.items():
        where = f"samples.{sample}.stages"
        stages = _mapping(part.get("stages"), where)
        made = [task.name for task in planned if task.sample == sample]
        if list(stages) != made:
            raise RecordError(f"{where} are {list(stages)}, but the problem makes {made}")
        for name, entry in stages.items():
            made_by = stage_task(name, sample)
            fragments[made_by] = _fragment(
                _mapping(entry, f"{where}.{name}"), f"stage {made_by.label!r}", _STAGE_STATUSES
            )
    geometry = _mapping(record.get("geometry"), "geometry")
    if geometry.get("status") != DONE:
        raise RecordError(f"geometry: status {geometry.get('status')!r}, but it is always done")
    fragments |= {task: e.fragment for task, e in evaluators.items()}
    results: dict[TaskId, TaskResult] = {GEOMETRY: TaskResult(GEOMETRY, DONE)}
    for task in plan.tasks:
        if task.id == GEOMETRY:
            continue
        where = f"{task.id.kind} {task.id.label!r}"
        fragment = fragments[task.id]
        needs = [need.label for need in task.needs]
        if fragment.reads != needs:
            raise RecordError(
                f"{where} reads {fragment.reads}, but the problem has it read {needs}"
            )
        expected = blocked(task, results)
        if expected is None:
            if fragment.status == Status.BLOCKED or fragment.blocked_by:
                raise RecordError(f"{where} is blocked, but every task it reads succeeded")
            results[task.id] = TaskResult(task.id, fragment.status, fragment.message)
            continue
        causes = [cause.label for cause in expected.blocked_by]
        if (fragment.status, fragment.blocked_by, fragment.message) != (
            Status.BLOCKED,
            causes,
            expected.message,
        ):
            raise RecordError(
                f"{where} is {fragment.status} (blocked by {fragment.blocked_by}, message "
                f"{fragment.message!r}), but what it reads makes it blocked by {causes} "
                f"({expected.message!r})"
            )
        results[task.id] = expected
    return list(results.values())


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{where} must be a nonblank string, got {value!r}")
    return value


_DECLARED = ("stage", "metric", "unit")


def _declared(value: object, where: str) -> dict[MetricKey, str] | None:
    """The ``metrics`` an evaluator of a record declares, ``None`` if it declares none."""
    if value is None:
        return None
    declared: dict[MetricKey, str] = {}
    for index, item in enumerate(_list(value, f"{where}: metrics")):
        at = f"{where}: metrics[{index}]"
        entry = _mapping(item, at)
        stage, metric, unit = (_text(entry.get(key), f"{at}.{key}") for key in _DECLARED)
        if (stage, metric) in declared:
            raise RecordError(f"{where}: {stage}/{metric} is declared twice")
        declared[(stage, metric)] = unit
    return declared


def _measured(item: Mapping[str, object], where: str) -> Measurement:
    fields: dict[str, Any] = {
        key: item.get(key) for key in ("stage", "metric", "value", "unit", "uncertainty")
    }
    try:
        return Measurement(**fields)
    except EvaluationContractError as exc:
        raise RecordError(f"{where}: {exc}") from None


def _check_units(
    spec: ProblemSpec, evaluations: Mapping[TaskId, _Evaluator], value: _Lookup
) -> None:
    """A constraint that declares a unit is in the unit its evaluator declares for the metric,
    and so is a valid measurement of it, as a live evaluation enforces: a record consistent in
    another unit would compare its value with a limit in the wrong one."""
    for constraint in spec.constraints:
        if constraint.unit is None:
            continue
        where = f"{constraint.label}: its constraint declares {constraint.unit!r}"
        declared = evaluations[_task(constraint.ref)].declared
        key = (constraint.stage, constraint.metric)
        if declared is not None and declared[key] != constraint.unit:
            raise RecordError(f"{where}, but its evaluator declares {declared[key]!r}")
        measurement = value(constraint.ref)  # None unless the evaluator is valid, as live
        if measurement is not None and measurement.unit != constraint.unit:
            raise RecordError(f"{where}, but it is measured in {measurement.unit!r}")


_NAMED = ("evaluator", "sample", "stage", "metric", "label")


def _check_metrics(spec: ProblemSpec, record: Mapping[str, object], value: _Lookup) -> None:
    """Each entry of ``metrics`` names a metric of the objectives and constraints, in order, and
    holds what its evaluator measured, or nothing if that evaluator is not valid."""
    entries = [
        _mapping(entry, f"metrics[{i}]")
        for i, entry in enumerate(_list(record.get("metrics"), "metrics"))
    ]
    found = [[entry.get(key) for key in _NAMED] for entry in entries]
    wanted = [
        [ref.evaluator, ref.sample, ref.stage, ref.metric, ref.label] for ref in spec.references
    ]
    if found != wanted:
        raise RecordError(f"its metrics are {found}, but the problem names {wanted}")
    for ref, entry in zip(spec.references, entries, strict=True):
        measured = _measurement(value(ref))
        expected = {key: measured[key] for key in ("value", "unit", "uncertainty")}
        recorded = {key: entry.get(key) for key in expected}
        if recorded != expected:
            raise RecordError(
                f"metric {ref.label}: recorded {recorded}, but its evaluator measured {expected}"
            )
