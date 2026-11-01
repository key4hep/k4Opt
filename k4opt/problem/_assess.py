"""The assessment of a candidate: its objective values, its constraints judged, its feasibility.

It depends on no optimizer: an optimizer, a dashboard and a person read the same judgment. It is
written into ``summary.json``, and :meth:`Assessment.from_summary` recomputes it from the
record's raw measurements once the record itself is checked (:mod:`k4opt.problem._record`), so
any reader can check a record before trusting it.

A constraint is judged on the central value of its metric: ``max`` is satisfied when the value is
at most the limit, ``min`` when it is at least the limit, both inclusive. The uncertainty is
recorded next to the value but not used. Only a ``valid`` evaluation is judged, and only one
that represents its candidate is called feasible or not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from k4opt.evaluation import Measurement, Status
from k4opt.problem._candidate import CandidateRecord
from k4opt.problem._record import _Lookup, _plain, check_summary
from k4opt.problem.errors import RecordError
from k4opt.problem.models import Constraint, Direction, MetricRef, Objective, ProblemSpec

type Bound = Literal["min", "max"]

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
        stages, and roles that fit what the problem gives each combined evaluator (every role and no
        other, one item unless it reads many, of evaluators that declare what it reads); its
        ``samples`` the problem's, each with the evaluations of the bindings that measure it, and
        its ``evaluations`` those of the bindings that measure the candidate, each of the
        implementation the problem binds it to, with the settings (``declared``) and collections the
        problem gives it, and for a combined evaluator the items each of its roles reads; a
        ``valid`` evaluation a role reads carries the metrics that role reads. The candidate's tasks
        are planned again from the problem and the stages the record says were ``supplied``: each
        must read what the plan says, be ``blocked`` exactly when a task it reads did not succeed
        (by the tasks the plan makes it blocked by, with their message) and otherwise have run. Each
        file a task made is named by its reference (``gamma/simulation#events``) and by a path
        relative to the run directory, in the task's own directory: the output of a stage that ran
        and is ``done``, never hashed, and the artifacts of an evaluation, each hashed, those its
        implementation declares and, for a ``valid`` one, every one of them. Which tasks are
        ``required`` is recomputed from the problem's objectives, constraints and required
        evaluators; the candidate's status and message from its required tasks' (``failed`` before
        ``invalid`` before ``valid``), its ``diagnostics`` from the other tasks that did not
        succeed, and whether it represents its candidate from the required stages supplied. Each
        objective and constraint value is taken from the own measurements of the evaluation it
        names, if that evaluation is ``valid``. The rules a live evaluation enforces are checked
        again: the candidate is in the problem's domains and ``candidate_key`` is its key, an
        ``invalid`` or ``failed`` task has a message, a blocked evaluator measured nothing, a
        measurement of a metric its implementation declares carries the declared unit, a binding
        whose implementation declares metrics declares those named of it, and a constraint's unit is
        that of its metric. Raises :class:`~k4opt.problem.RecordError` if the record is malformed or
        breaks one of those rules, was made for another problem (its definition differs other than
        in ``name``), or if its ``status``, ``message``, ``required``, ``diagnostics``,
        ``represents_candidate``, ``metrics`` or ``assessment`` disagree with the recomputation.
        """
        recorded = check_summary(spec, summary)
        assessment = _judge(spec, recorded.status, recorded.represents, recorded.value)
        if summary.get("assessment") != _plain(assessment.to_dict()):
            raise RecordError(
                "its assessment is not the one its measurements give: "
                f"recorded {summary.get('assessment')!r}, recomputed {assessment.to_dict()!r}"
            )
        return assessment


def assess(spec: ProblemSpec, candidate: CandidateRecord) -> Assessment:
    """The assessment of a candidate just evaluated."""
    return _judge(spec, candidate.status, candidate.represents_candidate, candidate.value)


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
