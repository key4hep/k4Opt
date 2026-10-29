"""The base class for evaluators and the shorthands it enables.

Subclassing :class:`Evaluator` is optional: anything with ``name``, ``version`` and
``evaluate`` is accepted (see :class:`~k4opt.evaluation.protocol.EvaluatorProtocol`). The base
class adds what most evaluators would write anyway:

* ``settings = {"collection": str}`` declares the evaluator's settings; ``check`` rejects
  missing and unknown keys and wrong types before anything runs;
* ``metrics = {"hit_cells": "1"}`` maps each metric to its unit; its stage is ``requires``,
  or the metric is written ``(stage, metric)``;
* ``evaluate`` may return ``{"hit_cells": 3.2}`` or ``{"hit_cells": [3, 4, 2]}`` (a number, or
  per-event values that become their mean with its standard error) instead of an
  :class:`EvaluationResult`;
* ``self.invalid(...)`` and ``self.fail(...)`` end an evaluation with that status;
* ``request.read(stage, collection, type)`` reads a collection of the file a stage produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NoReturn

from k4opt.evaluation.errors import EvaluationContractError, EvaluationFailed, EvaluationInvalid
from k4opt.evaluation.models import EvaluationRequest, EvaluationResult, Measurement

type MetricKey = str | tuple[str, str]
type Values = Mapping[MetricKey, int | float | Sequence[int | float]]
"""What ``evaluate`` may return instead of an :class:`EvaluationResult`."""


def _qualified(evaluator: object, key: MetricKey) -> tuple[str, str]:
    """``(stage, metric)`` of a metric written as ``metric`` (stage: ``requires``) or a pair."""
    if not isinstance(key, str):
        return key
    stage = getattr(evaluator, "requires", None)
    if stage is None:
        raise EvaluationContractError(
            f"metric {key!r}: write it as (stage, metric) or set requires = <stage>"
        )
    return stage, key


def _is(value: object, kind: type) -> bool:
    """``isinstance``, except that a bool is only a bool and an int is also a float."""
    if isinstance(value, bool):
        return kind is bool
    return isinstance(value, kind) or (kind is float and isinstance(value, int))


def metric_units(evaluator: object) -> dict[tuple[str, str], str] | None:
    """The declared ``{(stage, metric): unit}`` of ``evaluator``, ``None`` if it declares none."""
    declared = getattr(evaluator, "metrics", None)
    if declared is None:
        return None
    return {_qualified(evaluator, key): unit for key, unit in declared.items()}


def result_from(evaluator: object, returned: EvaluationResult | Values) -> EvaluationResult:
    """The result that ``evaluate`` returned, or the ``valid`` one its metric values stand for."""
    if isinstance(returned, EvaluationResult):
        return returned
    if not isinstance(returned, Mapping):
        raise EvaluationContractError(
            "evaluator must return an EvaluationResult or a mapping of metric to value, "
            f"got {type(returned).__name__}"
        )
    units = metric_units(evaluator) or {}
    measurements = []
    for key, value in returned.items():
        stage, metric = _qualified(evaluator, key)
        unit = units.get((stage, metric))
        if isinstance(value, (list, tuple)):
            measurements.append(Measurement.from_samples(stage, metric, value, unit))
        else:
            measurements.append(Measurement(stage, metric, value, unit))  # type: ignore[arg-type]
    return EvaluationResult.valid(measurements)


class Evaluator:
    """Base class of evaluators; see the module documentation. ``name`` and ``version`` are
    required, ``requires``, ``settings`` and ``metrics`` optional."""

    name: str
    version: str
    requires: str | None = None
    settings: Mapping[str, type] | None = None
    metrics: Mapping[MetricKey, str] | None = None

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult | Values:
        raise NotImplementedError

    def check(self, request: EvaluationRequest) -> None:
        """Reject settings that do not match ``settings``; nothing is checked if it is unset."""
        if self.settings is None:
            return
        given = request.settings
        unknown = [key for key in given if key not in self.settings]
        missing = [key for key in self.settings if key not in given]
        if unknown or missing:
            raise EvaluationContractError(
                f"settings: expected exactly {sorted(self.settings)}"
                + (f", unknown {unknown}" if unknown else "")
                + (f", missing {missing}" if missing else "")
            )
        for key, kind in self.settings.items():
            if not _is(given[key], kind):
                raise EvaluationContractError(
                    f"settings.{key}: expected {kind.__name__}, got {given[key]!r}"
                )

    def invalid(self, message: str) -> NoReturn:
        """End the evaluation: it ran, but its result is not usable."""
        raise EvaluationInvalid(message)

    def fail(self, message: str) -> NoReturn:
        """End the evaluation: the execution failed."""
        raise EvaluationFailed(message)
