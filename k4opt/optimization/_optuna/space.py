"""Internal: the problem in Optuna's terms, the search space and the values a trial is told."""

from __future__ import annotations

from collections.abc import Collection

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from k4opt.optimization.errors import OptimizationError
from k4opt.problem import ProblemSpec
from k4opt.record import Assessment

_EXACT_INTEGERS = 2**53  # TPE samples integers through floats, exact up to here


def _float_bound(name: str, value: int | float) -> float:
    try:
        bound = float(value)
    except OverflowError:
        bound = None
    if bound is None or bound != value:
        text = str(value)
        shown = text if len(text) <= 24 else f"{text[:12]}... ({len(text)} digits)"
        raise OptimizationError(f"{name}: the bound {shown} is no float, which the sampler draws")
    return bound


def search_space(
    spec: ProblemSpec, searched: Collection[str] | None = None
) -> dict[str, BaseDistribution]:
    """The distribution of each parameter ``searched`` (every one, by default), by name, in
    problem order; a parameter the study does not search needs none, and is not checked here.

    Continuous parameters are sampled uniformly between their bounds, with no log scale and no
    step; their bounds must be floats exactly, and so must their range. Integer bounds must stay
    within 2**53, below which the sampler's floats hold every integer. A categorical parameter
    takes one of its declared choices, as declared.
    """
    space: dict[str, BaseDistribution] = {}
    for parameter in spec.parameters:
        name = parameter.name
        if searched is not None and name not in searched:
            continue
        if parameter.kind == "categorical":
            assert parameter.choices is not None
            space[name] = CategoricalDistribution(list(parameter.choices))
            continue
        assert parameter.bounds is not None
        low, high = parameter.bounds
        if parameter.kind == "integer":
            if max(abs(low), abs(high)) > _EXACT_INTEGERS:
                raise OptimizationError(
                    f"{name}: integer bounds beyond 2**53 in magnitude cannot be sampled exactly"
                )
            space[name] = IntDistribution(int(low), int(high))
            continue
        lower, upper = _float_bound(name, low), _float_bound(name, high)
        if upper - lower == float("inf"):
            raise OptimizationError(f"{name}: the range [{low}, {high}] is too wide to sample")
        space[name] = FloatDistribution(lower, upper)
    return space


def objective_values(assessment: Assessment) -> list[float]:
    """The objective values of a valid candidate, in the problem's order, as the floats the
    optimizer is told; a value no float holds exactly (an integer beyond 2**53) is refused, as
    it would be told as another value."""
    assert assessment.objectives is not None
    values = []
    for objective in assessment.objectives:
        try:
            value = float(objective.value)
        except OverflowError:
            value = None
        if value is None or value != objective.value:
            raise OptimizationError(
                f"{objective.label}: no float holds {objective.value} exactly, so the optimizer "
                "cannot be told it"
            )
        values.append(value)
    return values


def constraint_values(assessment: Assessment) -> dict[str, float]:
    """The value the optimizer is told for each limit of an assessed candidate, by limit name.

    The normalized margin (:attr:`~k4opt.record.Limit.normalized`: the physical margin divided by
    the constraint's scale, ``study.json`` records this as ``margin / scale``), zero or below when
    the limit is satisfied. The sampler ranks infeasible trials by the sum of their positive
    values, which the scales make dimensionless. A value no float holds stops the study rather
    than being told as another one.
    """
    values = {}
    for limit in assessment.limits:
        assert limit.satisfied is not None  # assessed
        if limit.normalized is None:
            raise OptimizationError(
                f"{limit.name}: the margin of {limit.value} to {limit.limit} in units of its "
                f"scale {limit.scale} does not fit in a float, which the optimizer is told"
            )
        values[limit.name] = limit.normalized
    return values
