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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from k4opt.evaluation import EvaluationContractError, Measurement, Status
from k4opt.problem._evaluate import CandidateRecord, MetricKey
from k4opt.problem.errors import RecordError
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

        The candidate's status is recomputed from its evaluators' (``failed`` before ``invalid``
        before ``valid``), and each objective and constraint value is taken from the own
        measurements of the evaluator that ``metrics`` names as its owner, if that evaluator is
        ``valid``. Raises :class:`~k4opt.problem.RecordError` if the record is malformed, was made
        for another problem (its definition differs other than in ``name``), or if its
        ``status``, ``metrics`` or ``assessment`` disagree with that recomputation.
        """
        record = _mapping(summary, "record")
        if record.get("schema") != SUMMARY_SCHEMA:
            raise RecordError(f"schema is {record.get('schema')!r}, not {SUMMARY_SCHEMA!r}")
        _check_problem(spec, record)
        statuses, measured = _evaluators(spec, record)
        status = _status(statuses.values())
        if record.get("status") != status:
            raise RecordError(
                f"status is {record.get('status')!r}, but its evaluators make it {status.value!r}"
            )
        represents = record.get("represents_candidate")
        if not isinstance(represents, bool):
            raise RecordError(f"represents_candidate must be true or false, got {represents!r}")
        owners = _owners(spec, record, statuses)

        def value(stage: str, metric: str) -> Measurement | None:
            owner = owners[(stage, metric)]
            if statuses[owner] is not Status.VALID:
                return None
            found = measured[owner].get((stage, metric))
            if found is None:
                raise RecordError(
                    f"evaluator {owner!r} is valid but did not measure {stage}/{metric}, "
                    "which it owns"
                )
            return found

        _check_metrics(record, owners, value)
        _check_units(spec, value)
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


def _status(statuses: Iterable[Status]) -> Status:
    """``failed`` if any evaluator failed, otherwise ``invalid`` if any is, otherwise ``valid``;
    the rule of :attr:`CandidateRecord.status`."""
    seen = set(statuses)
    return next((s for s in (Status.FAILED, Status.INVALID) if s in seen), Status.VALID)


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


def _evaluators(
    spec: ProblemSpec, record: Mapping[str, object]
) -> tuple[dict[str, Status], dict[str, dict[MetricKey, Measurement]]]:
    """The status of each evaluator of the record, and its own measurements, by name."""
    entries = [
        _mapping(entry, f"evaluators[{i}]")
        for i, entry in enumerate(_list(record.get("evaluators"), "evaluators"))
    ]
    found = [(entry.get("name"), entry.get("version")) for entry in entries]
    declared = [(entry.name, entry.version) for entry in spec.evaluators]
    if found != declared:
        raise RecordError(f"its evaluators are {found}, but the problem declares {declared}")
    statuses: dict[str, Status] = {}
    measured: dict[str, dict[MetricKey, Measurement]] = {}
    for (name, _), entry in zip(declared, entries, strict=True):
        status = entry.get("status")
        if not isinstance(status, str) or status not in set(Status):
            raise RecordError(
                f"evaluator {name!r}: status {status!r} is none of {', '.join(Status)}"
            )
        statuses[name] = Status(status)
        own: dict[MetricKey, Measurement] = {}
        for index, item in enumerate(_list(entry.get("measurements"), f"{name}: measurements")):
            where = f"evaluator {name!r}: measurements[{index}]"
            measurement = _measured(_mapping(item, where), where)
            key = (measurement.stage, measurement.metric)
            if key in own:
                raise RecordError(f"{where}: {key[0]}/{key[1]} is measured twice")
            own[key] = measurement
        measured[name] = own
    return statuses, measured


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
    spec: ProblemSpec, record: Mapping[str, object], statuses: Mapping[str, Status]
) -> dict[MetricKey, str]:
    """The evaluator the record names as the owner of each objective and constraint metric."""
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
        if not isinstance(owner, str) or owner not in statuses:
            raise RecordError(f"metric {key[0]}/{key[1]}: no evaluator {owner!r} in the record")
        owners[key] = owner
    return owners


def _check_units(spec: ProblemSpec, value: _Lookup) -> None:
    """A constraint that declares a unit is measured in it, as a live evaluation enforces: a
    record consistent in another unit would compare its value with a limit in the wrong one."""
    for constraint in spec.constraints:
        measurement = value(constraint.stage, constraint.metric)
        if constraint.unit is None or measurement is None or measurement.unit == constraint.unit:
            continue
        raise RecordError(
            f"{constraint.stage}/{constraint.metric} is measured in {measurement.unit!r}, but its "
            f"constraint declares {constraint.unit!r}"
        )


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
