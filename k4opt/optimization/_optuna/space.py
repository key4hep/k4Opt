"""Internal: the problem in Optuna's terms, the search space and the values a trial is told."""

from __future__ import annotations

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from k4opt.optimization.errors import OptimizationError
from k4opt.problem import Assessment, ProblemSpec

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


def search_space(spec: ProblemSpec) -> dict[str, BaseDistribution]:
    """The distribution of each parameter, by canonical reference, in problem order.

    Continuous parameters are sampled uniformly between their bounds, with no log scale and no
    step; their bounds must be floats exactly, and so must their range. Integer bounds must stay
    within 2**53, below which the sampler's floats hold every integer. A categorical parameter
    takes one of its declared choices, as declared.
    """
    space: dict[str, BaseDistribution] = {}
    for parameter in spec.parameters:
        name = parameter.geometry.canonical_target
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
    for measurement in assessment.objectives:
        try:
            value = float(measurement.value)
        except OverflowError:
            value = None
        if value is None or value != measurement.value:
            raise OptimizationError(
                f"{measurement.stage}/{measurement.metric}: no float holds {measurement.value} "
                "exactly, so the optimizer cannot be told it"
            )
        values.append(value)
    return values


def constraint_values(assessment: Assessment) -> dict[str, float]:
    """The value the optimizer is told for each limit of an assessed candidate, by limit name.

    The physical margin as it is (``study.json`` records this as ``physical margins``), zero or
    below when the limit is satisfied. The sampler ranks infeasible trials by the sum of their
    positive values, so with limits on several metrics it adds values in different units.
    """
    values = {}
    for limit in assessment.limits:
        assert limit.satisfied is not None  # assessed
        if limit.margin is None:
            raise OptimizationError(
                f"{limit.name}: the margin of {limit.value} to {limit.limit} does not fit in a "
                "float, which the optimizer is told"
            )
        values[limit.name] = limit.margin
    return values
