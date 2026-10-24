"""Internal bridge: one candidate of a problem, one evaluator call, one record.

Not a public workflow. It validates a candidate against the declared parameter domains,
checks the evaluator identity, builds the :class:`EvaluationRequest`, calls the evaluator
exactly once and checks that a ``valid`` result carries the metrics the problem asks for.
It neither locates evaluators, materializes geometry nor runs anything itself.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

from k4opt.evaluation import (
    CandidateKey,
    EvaluationContractError,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
)
from k4opt.evaluation.models import Scalar
from k4opt.problem.errors import CandidateError
from k4opt.problem.models import Parameter, ProblemSpec


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _canonical_number(value: int | float) -> str:
    """A stable text for a number, equal for ``1``, ``1.0`` and ``1e0``; exact for any value."""
    # Both conversions are exact, so equal text means equal numerical value (10**23 differs
    # from 1e23, which is rounded). normalize() is avoided because it rounds to the context
    # precision.
    number = Decimal(value) if isinstance(value, int) else Decimal.from_float(value)
    sign, digits, exponent = number.as_tuple()
    assert isinstance(exponent, int)
    text = "".join(map(str, digits)).lstrip("0")
    if not text:
        return "0"  # also -0.0
    stripped = text.rstrip("0")
    exponent += len(text) - len(stripped)
    return f"{'-' if sign else ''}{stripped}e{exponent}"


def _check_number(value: object, reference: str) -> int | float:
    if not _is_number(value):
        raise CandidateError(f"{reference}: expected a number, got {value!r}")
    assert isinstance(value, (int, float))
    if isinstance(value, float) and not math.isfinite(value):
        raise CandidateError(f"{reference}: value must be finite, got {value!r}")
    return value


def _validate_value(parameter: Parameter, value: object) -> tuple[str, str]:
    """Return ``(tag, canonical value)`` of ``value`` for ``parameter`` or raise."""
    reference = parameter.geometry.canonical_target
    if parameter.kind == "continuous":
        number = _check_number(value, reference)
    elif parameter.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CandidateError(f"{reference}: expected an integer, got {value!r}")
        number = value
    else:
        choices = parameter.choices or ()
        if all(isinstance(choice, str) for choice in choices):
            if not isinstance(value, str) or value not in choices:
                raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
            return "str", value
        if all(isinstance(choice, bool) for choice in choices):
            if not isinstance(value, bool) or value not in choices:
                raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
            return "bool", "true" if value else "false"
        number = _check_number(value, reference)
        if not any(number == choice for choice in choices):
            raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
        return "num", _canonical_number(number)
    assert parameter.bounds is not None
    low, high = parameter.bounds
    if not low <= number <= high:
        raise CandidateError(f"{reference}: {value!r} is outside the bounds [{low}, {high}]")
    return ("int" if parameter.kind == "integer" else "num"), _canonical_number(number)


def _validate_candidate(
    spec: ProblemSpec, candidate: Mapping[str, Scalar]
) -> tuple[dict[str, Scalar], CandidateKey]:
    """Check ``candidate`` against every parameter; return it in problem order with its key."""
    if not isinstance(candidate, Mapping):
        raise CandidateError(f"candidate must be a mapping, got {type(candidate).__name__}")
    declared = [parameter.geometry.canonical_target for parameter in spec.parameters]
    bad_keys = [key for key in candidate if not isinstance(key, str)]
    if bad_keys:
        raise CandidateError(f"candidate keys must be strings, got {bad_keys!r}")
    unknown = [key for key in candidate if key not in declared]
    if unknown:
        raise CandidateError(f"candidate has unknown parameter(s): {', '.join(unknown)}")
    missing = [reference for reference in declared if reference not in candidate]
    if missing:
        raise CandidateError(f"candidate is missing parameter(s): {', '.join(missing)}")
    values: dict[str, Scalar] = {}
    key: list[tuple[str, str, str, str]] = []
    for parameter, reference in zip(spec.parameters, declared, strict=True):
        value = candidate[reference]
        tag, canonical = _validate_value(parameter, value)
        values[reference] = value
        key.append((reference, parameter.kind, tag, canonical))
    return values, tuple(key)


def _check_result(spec: ProblemSpec, result: EvaluationResult) -> None:
    """A ``valid`` result must report every metric the problem refers to, in its unit."""
    if result.status != "valid":
        return
    measured = {(m.stage, m.metric): m for m in result.measurements}
    required = [(o.stage, o.metric) for o in spec.objectives]
    required += [(c.stage, c.metric) for c in spec.constraints]
    missing = [
        f"{stage}/{metric}"
        for stage, metric in dict.fromkeys(required)
        if (stage, metric) not in measured
    ]
    if missing:
        raise EvaluationContractError(
            f"evaluator returned a valid result without required metric(s): {', '.join(missing)}"
        )
    for constraint in spec.constraints:
        unit = measured[(constraint.stage, constraint.metric)].unit
        if constraint.unit is not None and unit != constraint.unit:
            raise EvaluationContractError(
                f"constraint on {constraint.stage}/{constraint.metric} declares unit "
                f"{constraint.unit!r} but the measurement has unit {unit!r}"
            )


def _evaluate_candidate(
    spec: ProblemSpec,
    candidate: Mapping[str, Scalar],
    evaluator: Evaluator,
    *,
    inputs: Mapping[str, Path] | None = None,
    workdir: Path | None = None,
) -> EvaluationRecord:
    """Evaluate one complete ``candidate`` of ``spec`` with ``evaluator``, once.

    The candidate maps each canonical parameter reference to a scalar. Raises
    :class:`CandidateError` for a candidate outside the declared domains and
    :class:`EvaluationContractError` for an evaluator that does not match the problem or a
    ``valid`` result that omits a requested metric. ``inputs`` and ``workdir`` are passed
    through unchanged. Exceptions of the evaluator propagate.
    """
    values, key = _validate_candidate(spec, candidate)
    wanted = (spec.evaluation.evaluator, spec.evaluation.version)
    name, version = evaluator.name, evaluator.version  # read once: the record reuses them
    if (name, version) != wanted:
        raise EvaluationContractError(
            f"problem declares evaluator {wanted[0]!r} version {wanted[1]!r}, "
            f"got {name!r} version {version!r}"
        )
    request = EvaluationRequest(
        candidate=values,
        settings=spec.evaluation.settings,
        inputs=inputs or {},
        workdir=workdir,
        events=spec.evaluation.events,
        seed=spec.evaluation.seed,
        backend_args=spec.evaluation.backend,
    )
    started = time.perf_counter()
    result = evaluator.evaluate(request)
    elapsed = time.perf_counter() - started
    if not isinstance(result, EvaluationResult):
        raise EvaluationContractError(
            f"evaluator must return an EvaluationResult, got {type(result).__name__}"
        )
    _check_result(spec, result)
    return EvaluationRecord(request, result, name, version, key, elapsed)
