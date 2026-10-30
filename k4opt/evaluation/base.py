"""The base class for evaluators and the shorthands it enables.

Subclassing :class:`Evaluator` is optional: anything with ``name``, ``version`` and
``evaluate`` is accepted (see :class:`~k4opt.evaluation.protocol.EvaluatorProtocol`). The base
class adds what most evaluators would write anyway:

* ``settings = {"collection": str, "threshold": 0.5}`` declares the evaluator's settings: a type
  for a required key, a default value for an optional one. ``check`` rejects missing and
  unknown keys, wrong types and blank required text before anything runs, and the request
  carries the defaults;
* ``metrics = {"hit_cells": "1"}`` maps each metric to its unit; its stage is ``stage``,
  or the metric is written ``(stage, metric)``;
* ``evaluate`` may return ``{"hit_cells": 3.2}`` or ``{"hit_cells": [3, 4, 2]}`` (a number, or
  per-event values, a list or a numpy array, that become their mean with its standard error)
  instead of an :class:`EvaluationResult`;
* ``self.invalid(...)`` and ``self.fail(...)`` end an evaluation with that status;
* ``self.read(request, collection, type)`` reads a collection of the file of the stage in
  ``stage`` (``request.read(stage, ...)`` names any stage).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, NoReturn, cast

from k4opt.evaluation.errors import EvaluationContractError, EvaluationFailed, EvaluationInvalid
from k4opt.evaluation.models import EvaluationRequest, EvaluationResult, Measurement
from k4opt.frozen import Json

type MetricKey = str | tuple[str, str]
type Values = Mapping[MetricKey, int | float | Iterable[int | float]]
"""What ``evaluate`` may return instead of an :class:`EvaluationResult`."""


def _qualified(evaluator: object, key: MetricKey) -> tuple[str, str]:
    """``(stage, metric)`` of a metric written as ``metric`` (stage: ``stage``) or a pair."""
    if isinstance(key, tuple) and len(key) == 2 and all(isinstance(part, str) for part in key):
        return key
    if not isinstance(key, str):
        raise EvaluationContractError(
            f"metric {key!r}: write it as 'metric' or as a pair of strings (stage, metric)"
        )
    stage = getattr(evaluator, "stage", None)
    if stage is None:
        raise EvaluationContractError(
            f"metric {key!r}: write it as (stage, metric) or set stage = <stage>"
        )
    return stage, key


_KINDS: Mapping[type, type | tuple[type, ...]] = {
    str: str,
    int: int,
    float: (int, float),
    bool: bool,
    # the request holds settings frozen: a mapping is read-only, a list a tuple
    dict: Mapping,
    list: (list, tuple),
}


def _is(value: object, kind: type) -> bool:
    """Whether ``value`` is a ``kind`` setting; a bool is only a bool, an int also a float."""
    if isinstance(value, bool):
        return kind is bool
    return isinstance(value, _KINDS[kind])


def _declared(evaluator: object) -> Mapping[str, Any]:
    return getattr(evaluator, "settings", None) or {}


def with_defaults(evaluator: object, settings: Mapping[str, Json]) -> Mapping[str, Json]:
    """``settings`` plus the default of every optional setting that ``evaluator`` declares."""
    merged = dict(settings)
    for key, declared in _declared(evaluator).items():
        if not isinstance(declared, type):
            merged.setdefault(key, declared)
    return merged


def metric_units(evaluator: object) -> dict[tuple[str, str], str] | None:
    """The declared ``{(stage, metric): unit}`` of ``evaluator``, ``None`` if it declares none."""
    declared = getattr(evaluator, "metrics", None)
    if declared is None:
        return None
    units: dict[tuple[str, str], str] = {}
    for key, unit in declared.items():
        qualified = _qualified(evaluator, key)
        if not isinstance(unit, str) or not unit.strip():
            raise EvaluationContractError(
                f"metric {qualified[0]}/{qualified[1]}: unit must be a nonblank string such as "
                f"'mm' or '1' (dimensionless), got {unit!r}"
            )
        if qualified in units:
            raise EvaluationContractError(
                f"metric {qualified[0]}/{qualified[1]} is declared twice (as {key!r} too)"
            )
        units[qualified] = unit
    return units


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
        # per-event values: a list, a tuple or a numpy array, say; text is not a sequence of numbers
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
            try:
                samples = list(value)
            except TypeError:  # iterable in type only, like a zero-dimensional numpy array
                raise EvaluationContractError(
                    f"metric {stage}/{metric}: {value!r} is neither a number nor per-event values"
                ) from None
            measurements.append(Measurement.from_samples(stage, metric, samples, unit))
        else:  # a number; Measurement rejects anything else
            measurements.append(Measurement(stage, metric, cast(float, value), unit))
    return EvaluationResult.valid(measurements)


class Evaluator:
    """Base class of evaluators; see the module documentation. ``name`` and ``version`` are
    required, ``stage`` (the pipeline stage it reads), ``settings`` and ``metrics`` optional."""

    name: str
    version: str
    stage: str | None = None
    settings: Mapping[str, Any] | None = None
    metrics: Mapping[MetricKey, str] | None = None

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult | Values:
        raise NotImplementedError

    def check(self, request: EvaluationRequest) -> None:
        """Reject settings that do not match ``settings``; nothing is checked if it is unset."""
        if self.settings is None:
            return
        given = request.settings
        kinds = {k: d if isinstance(d, type) else type(d) for k, d in self.settings.items()}
        for key, kind in kinds.items():
            if kind not in _KINDS:
                raise EvaluationContractError(
                    f"settings.{key}: cannot declare {kind.__name__}; "
                    "use str, int, float, bool, list or dict (or a default of one of them)"
                )
        unknown = [key for key in given if key not in kinds]
        missing = [k for k, d in self.settings.items() if isinstance(d, type) and k not in given]
        if unknown or missing:
            raise EvaluationContractError(
                f"settings: expected {sorted(kinds)}"
                + (f", unknown {unknown}" if unknown else "")
                + (f", missing {missing}" if missing else "")
            )
        for key, value in given.items():
            if not _is(value, kinds[key]):
                raise EvaluationContractError(
                    f"settings.{key}: expected {kinds[key].__name__}, got {value!r}"
                )
            # a required text (a collection name, say) cannot be blank; an optional one may be
            if isinstance(value, str) and not value.strip() and self.settings[key] is str:
                raise EvaluationContractError(f"settings.{key}: must not be blank")

    def read(self, request: EvaluationRequest, collection: str, value_type: str) -> Iterator[Any]:
        """The collection of each event of the file of the stage in ``stage``."""
        if self.stage is None:
            raise EvaluationContractError("read() needs stage = <stage> (or request.read)")
        return request.read(self.stage, collection, value_type)

    def invalid(self, message: str) -> NoReturn:
        """End the evaluation: it ran, but its result is not usable."""
        raise EvaluationInvalid(message)

    def fail(self, message: str) -> NoReturn:
        """End the evaluation: the execution failed."""
        raise EvaluationFailed(message)
