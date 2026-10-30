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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, cast

from k4opt.evaluation import EvaluationContractError, Measurement, Scalar, Status
from k4opt.problem._evaluate import (
    CandidateRecord,
    MetricKey,
    _validate_candidate,
    candidate_message,
    candidate_status,
)
from k4opt.problem.errors import CandidateError, RecordError
from k4opt.problem.models import Constraint, ProblemSpec

SUMMARY_SCHEMA = "k4opt.run/1"

type Bound = Literal["min", "max"]
type _Lookup = Callable[[str, str], Measurement | None]

_BOUNDS: tuple[Bound, ...] = ("min", "max")


@dataclass(frozen=True, slots=True)
class Limit:
    """One bound of a constraint, and how the candidate's measurement compares with it.

    ``limit`` is the bound as the problem declares it, and ``unit`` the unit of the measurement
    (the constraint's own when there is no measurement). ``margin`` is ``value - max`` or
    ``min - value`` in that unit, zero or below when the limit is ``satisfied``: equality
    satisfies it, with margin 0. The margin is computed exactly and rounded once, so its sign is
    always that of the exact comparison; it is ``None`` if it does not fit in a float, while
    ``satisfied`` stays exact. ``value``, its ``uncertainty`` (recorded, not used), ``margin``
    and ``satisfied`` are ``None`` when the evaluation is not ``valid``.
    """

    stage: str
    metric: str
    bound: Bound
    limit: int | float
    unit: str | None = None
    value: int | float | None = None
    uncertainty: int | float | None = None
    margin: float | None = None
    satisfied: bool | None = None

    @property
    def name(self) -> str:
        """``stage/metric:bound``, such as ``simulation/ecal_barrel.angular_residual:max``;
        unique within a problem."""
        return f"{self.stage}/{self.metric}:{self.bound}"

    def to_dict(self) -> dict[str, object]:
        return {
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

    ``objectives`` holds the measurement of each objective, in the problem's order, and is
    ``None`` unless the evaluation is ``valid``: there is never a partial vector. ``limits``
    holds a :class:`Limit` for each bound of each constraint, in the problem's order (``min``
    before ``max``). ``feasible`` is ``True`` when every limit is satisfied (so also without
    constraints), ``False`` when one is not, and ``None`` when the candidate is not assessed:
    its evaluation is ``invalid`` or ``failed``, or a supplied stage was not made from it
    (``represents_candidate`` is false), in which case the objectives and limits are still
    given as measured.
    """

    objectives: tuple[Measurement, ...] | None
    limits: tuple[Limit, ...] = ()
    feasible: bool | None = None

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-compatible data, as ``summary.json`` holds it under ``assessment``."""
        objectives = self.objectives
        return {
            "feasible": self.feasible,
            "objectives": None if objectives is None else [_measurement(m) for m in objectives],
            "limits": [limit.to_dict() for limit in self.limits],
        }

    @classmethod
    def from_summary(cls, spec: ProblemSpec, summary: Mapping[str, object]) -> Assessment:
        """Recompute the assessment of a run record (``summary.json`` as read back) and check
        the record against it.

        The candidate's status and message are recomputed from its evaluators' (``failed`` before
        ``invalid`` before ``valid``), and each objective and constraint value is taken from the own
        measurements of the evaluator that ``metrics`` names as its owner, if that evaluator is
        ``valid``. The rules a live evaluation enforces are checked again: the candidate is in
        the problem's domains and ``candidate_key`` is its key, an ``invalid`` or
        ``failed`` evaluator has a message, a measurement of a metric its evaluator declares carries
        the declared unit, each of several evaluators declares its metrics, no metric is declared
        twice or reported by an evaluator that does not declare it, the owner declares the metric it
        owns, and a constraint's unit is that of its metric. Raises
        :class:`~k4opt.problem.RecordError` if the record is malformed or breaks one of those rules,
        was made for another problem (its definition differs other than in ``name``), or if its
        ``status``, ``message``, ``metrics`` or ``assessment`` disagree with the recomputation.
        """
        record = _mapping(summary, "record")
        if record.get("schema") != SUMMARY_SCHEMA:
            raise RecordError(f"schema is {record.get('schema')!r}, not {SUMMARY_SCHEMA!r}")
        _check_problem(spec, record)
        _check_candidate(spec, record)
        evaluators = _evaluators(spec, record)
        status = candidate_status(entry.status for entry in evaluators.values())
        if record.get("status") != status:
            raise RecordError(
                f"status is {record.get('status')!r}, but its evaluators make it {status.value!r}"
            )
        message = candidate_message((n, e.status, e.message) for n, e in evaluators.items())
        if record.get("message") != message:
            raise RecordError(
                f"message is {record.get('message')!r}, but its evaluators make it {message!r}"
            )
        represents = record.get("represents_candidate")
        if not isinstance(represents, bool):
            raise RecordError(f"represents_candidate must be true or false, got {represents!r}")
        owners = _owners(spec, record, evaluators)

        def value(stage: str, metric: str) -> Measurement | None:
            owner = evaluators[owners[(stage, metric)]]
            if owner.status is not Status.VALID:
                return None
            found = owner.measured.get((stage, metric))
            if found is None:
                raise RecordError(
                    f"evaluator {owners[(stage, metric)]!r} is valid but did not measure "
                    f"{stage}/{metric}, which it owns"
                )
            return found

        _check_metrics(record, owners, value)
        _check_units(spec, owners, evaluators, value)
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


def _judge(spec: ProblemSpec, status: Status, represents: bool, value: _Lookup) -> Assessment:
    """The one way an assessment is made, live or from a record: ``value(stage, metric)`` is the
    measurement of an objective or constraint metric by the evaluator that owns it."""
    bounds = [
        (constraint, bound)
        for constraint in spec.constraints
        for bound in _BOUNDS
        if getattr(constraint, bound) is not None
    ]
    if status is not Status.VALID:
        return Assessment(None, tuple(_limit(c, bound, None) for c, bound in bounds), None)
    objectives = tuple(_required(value, o.stage, o.metric) for o in spec.objectives)
    limits = tuple(_limit(c, b, _required(value, c.stage, c.metric)) for c, b in bounds)
    feasible = all(limit.satisfied for limit in limits) if represents else None
    return Assessment(objectives, limits, feasible)


def _required(value: _Lookup, stage: str, metric: str) -> Measurement:
    measurement = value(stage, metric)
    if measurement is None:  # a valid result carries every metric it owns (_check_result)
        raise RecordError(f"a valid evaluation has no value of {stage}/{metric}")
    return measurement


def _limit(constraint: Constraint, bound: Bound, measurement: Measurement | None) -> Limit:
    limit = constraint.min if bound == "min" else constraint.max
    assert limit is not None
    if measurement is None:
        return Limit(constraint.stage, constraint.metric, bound, limit, constraint.unit)
    # exact: Fraction holds every float and int, and float() rounds the difference once
    excess = Fraction(measurement.value) - Fraction(limit)
    if bound == "min":
        excess = -excess
    try:
        margin: float | None = float(excess)
    except OverflowError:
        margin = None
    return Limit(
        constraint.stage,
        constraint.metric,
        bound,
        limit,
        measurement.unit,
        measurement.value,
        measurement.uncertainty,
        margin,
        excess <= 0,
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


@dataclass(frozen=True, slots=True)
class _Evaluator:
    """One evaluator of a record, as far as the assessment reads it."""

    status: Status
    message: str | None
    measured: Mapping[MetricKey, Measurement]  # its own measurements
    declared: Mapping[MetricKey, str] | None  # the metrics it declares, with their units


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


def _evaluators(spec: ProblemSpec, record: Mapping[str, object]) -> dict[str, _Evaluator]:
    """Each evaluator of the record by name, in problem order, checked as a live result is."""
    entries = [
        _mapping(entry, f"evaluators[{i}]")
        for i, entry in enumerate(_list(record.get("evaluators"), "evaluators"))
    ]
    found = [(entry.get("name"), entry.get("version")) for entry in entries]
    expected = [(entry.name, entry.version) for entry in spec.evaluators]
    if found != expected:
        raise RecordError(f"its evaluators are {found}, but the problem declares {expected}")
    evaluators: dict[str, _Evaluator] = {}
    for (name, _), entry in zip(expected, entries, strict=True):
        where = f"evaluator {name!r}"
        status = entry.get("status")
        if not isinstance(status, str) or status not in set(Status):
            raise RecordError(f"{where}: status {status!r} is none of {', '.join(Status)}")
        message = entry.get("message")
        if message is not None and not isinstance(message, str):
            raise RecordError(f"{where}: message must be a string, got {message!r}")
        if status != Status.VALID and not (message and message.strip()):
            raise RecordError(f"{where}: a result that is {status} needs a message saying why")
        declared = _declared(entry.get("metrics"), where)
        own: dict[MetricKey, Measurement] = {}
        for index, item in enumerate(_list(entry.get("measurements"), f"{where}: measurements")):
            at = f"{where}: measurements[{index}]"
            measurement = _measured(_mapping(item, at), at)
            key = (measurement.stage, measurement.metric)
            if key in own:
                raise RecordError(f"{at}: {key[0]}/{key[1]} is measured twice")
            if declared is not None and key in declared and measurement.unit != declared[key]:
                # whatever the status, as live: one metric identity never carries two units
                raise RecordError(
                    f"{at}: {key[0]}/{key[1]} is in {measurement.unit!r}, but the evaluator "
                    f"declares {declared[key]!r}"
                )
            own[key] = measurement
        evaluators[name] = _Evaluator(Status(status), message, own, declared)
    _check_declarations(evaluators)
    return evaluators


def _check_declarations(evaluators: Mapping[str, _Evaluator]) -> None:
    """The metric declarations of several evaluators, as a live evaluation binds them: each
    declares its metrics, no metric has two of them, and none reports another's metric."""
    silent = [name for name, evaluator in evaluators.items() if evaluator.declared is None]
    if len(evaluators) > 1 and silent:
        raise RecordError(
            f"evaluator(s) {', '.join(map(repr, silent))} declare no metrics, but each of several "
            "evaluators declares the metrics it measures"
        )
    declarer: dict[MetricKey, str] = {}
    for name, evaluator in evaluators.items():
        for key in evaluator.declared or {}:
            if key in declarer:
                raise RecordError(
                    f"metric {key[0]}/{key[1]} is declared by evaluators {declarer[key]!r} and "
                    f"{name!r}; a metric has one owner"
                )
            declarer[key] = name
    for name, evaluator in evaluators.items():
        for key in evaluator.measured:
            if declarer.get(key, name) != name:
                raise RecordError(
                    f"evaluator {name!r} reported {key[0]}/{key[1]}, a metric of evaluator "
                    f"{declarer[key]!r}"
                )


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


def _wanted(spec: ProblemSpec) -> list[MetricKey]:
    """The objective and constraint metrics in the order the record lists them."""
    wanted = [(o.stage, o.metric) for o in spec.objectives]
    return list(dict.fromkeys(wanted + [(c.stage, c.metric) for c in spec.constraints]))


def _owners(
    spec: ProblemSpec, record: Mapping[str, object], evaluators: Mapping[str, _Evaluator]
) -> dict[MetricKey, str]:
    """The evaluator the record names as the owner of each objective and constraint metric; one
    that declares metrics must declare the ones it owns."""
    entries = [
        _mapping(entry, f"metrics[{i}]")
        for i, entry in enumerate(_list(record.get("metrics"), "metrics"))
    ]
    found = [(entry.get("stage"), entry.get("metric")) for entry in entries]
    wanted = _wanted(spec)
    if found != wanted:
        raise RecordError(f"its metrics are {found}, but the problem names {wanted}")
    owners: dict[MetricKey, str] = {}
    for key, entry in zip(wanted, entries, strict=True):
        owner = entry.get("evaluator")
        if not isinstance(owner, str) or owner not in evaluators:
            raise RecordError(f"metric {key[0]}/{key[1]}: no evaluator {owner!r} in the record")
        declared = evaluators[owner].declared
        if declared is not None and key not in declared:
            raise RecordError(f"metric {key[0]}/{key[1]}: its owner {owner!r} does not declare it")
        owners[key] = owner
    return owners


def _check_units(
    spec: ProblemSpec,
    owners: Mapping[MetricKey, str],
    evaluators: Mapping[str, _Evaluator],
    value: _Lookup,
) -> None:
    """A constraint that declares a unit is in the unit its owner declares for the metric, and
    so is a valid measurement of it, as a live evaluation enforces: a record consistent in
    another unit would compare its value with a limit in the wrong one."""
    for constraint in spec.constraints:
        if constraint.unit is None:
            continue
        key = (constraint.stage, constraint.metric)
        where = f"{key[0]}/{key[1]}: its constraint declares {constraint.unit!r}"
        declared = evaluators[owners[key]].declared
        if declared is not None and declared[key] != constraint.unit:
            raise RecordError(f"{where}, but its owner {owners[key]!r} declares {declared[key]!r}")
        measurement = value(*key)  # None unless the owner is valid, as live
        if measurement is not None and measurement.unit != constraint.unit:
            raise RecordError(f"{where}, but it is measured in {measurement.unit!r}")


def _check_metrics(
    record: Mapping[str, object], owners: Mapping[MetricKey, str], value: _Lookup
) -> None:
    """Each entry of ``metrics`` (which name the metrics of ``owners``, in order) holds what its
    owner measured, or nothing if that owner is not valid."""
    entries = _list(record.get("metrics"), "metrics")
    for (stage, metric), item in zip(owners, entries, strict=True):
        entry = _mapping(item, "metrics")
        measured = _measurement(value(stage, metric))
        expected = {key: measured[key] for key in ("value", "unit", "uncertainty")}
        recorded = {key: entry.get(key) for key in expected}
        if recorded != expected:
            raise RecordError(
                f"metric {stage}/{metric}: recorded {recorded}, but its owner measured {expected}"
            )
