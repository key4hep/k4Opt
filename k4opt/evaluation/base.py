"""The base class for evaluators and the shorthands it enables.

Subclassing :class:`Evaluator` is optional: anything with ``name``, ``version`` and
``evaluate`` is accepted (see :class:`~k4opt.evaluation.protocol.EvaluatorProtocol`). The base
class adds what most evaluators would write anyway:

* ``collections = {"hits": Collection("SimCalorimeterHit")}`` declares the EDM4hep collections
  the evaluator reads from the file of its stage, by role: the problem names each one (or a
  ``default`` does), and ``self.events(request)`` yields every event with them as attributes
  (``event.hits``), all read from that one event;
* ``settings = {"threshold": Quantity("GeV"), "cut": 0.5}`` declares the evaluator's settings:
  a type for a required key, a default value for an optional one, a :class:`Quantity` for a
  number with a unit. ``check`` rejects missing and unknown keys, wrong types and blank required
  text before anything runs; the request carries the defaults, and each quantity as a float in
  its declared unit;
* ``metrics = {"hit_cells": "1"}`` maps each metric to its unit; its stage is ``stage``,
  or the metric is written ``(stage, metric)``;
* ``evaluate`` may return ``{"hit_cells": 3.2}`` or ``{"hit_cells": [3, 4, 2]}`` (a number, or
  per-event values, a list or a numpy array, that become their unweighted mean with its
  standard error) instead of an :class:`EvaluationResult`;
* ``self.invalid(...)`` and ``self.fail(...)`` end an evaluation with that status.

``request.read(stage, collection, type)`` reads one collection of any stage's file; collections
read that way are not matched up event by event.
"""

from __future__ import annotations

import keyword
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from k4opt.evaluation import units
from k4opt.evaluation.edm4hep import Event, read_events
from k4opt.evaluation.errors import EvaluationContractError, EvaluationFailed, EvaluationInvalid
from k4opt.evaluation.models import EvaluationRequest, EvaluationResult, Measurement
from k4opt.frozen import Json

_TYPE = re.compile(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)?")


@dataclass(frozen=True, slots=True)
class Collection:
    """An EDM4hep collection an evaluator reads from the file of its stage.

    ``type`` is the type of its elements, ``"SimCalorimeterHit"`` (the ``edm4hep::`` prefix is
    implied). The problem names the collection in the evaluator's ``collections``, unless
    ``default`` names it (``"MCParticles"``); the problem may still name another.
    """

    type: str
    default: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not _TYPE.fullmatch(self.type):
            raise EvaluationContractError(
                f"Collection: type must be an EDM4hep type such as 'SimCalorimeterHit', "
                f"got {self.type!r}"
            )
        if self.default is not None and (not isinstance(self.default, str) or not self.default):
            raise EvaluationContractError(
                f"Collection: default must be a collection name, got {self.default!r}"
            )

    @property
    def podio_type(self) -> str:
        """``type`` as podio names it, ``edm4hep::SimCalorimeterHit``."""
        return self.type if "::" in self.type else f"edm4hep::{self.type}"


@dataclass(frozen=True, slots=True)
class Quantity:
    """A setting with a unit. The problem writes ``value`` and ``unit`` (any unit of the same
    dimension, ``70`` and ``keV``); the evaluator receives a float in ``unit``. ``default``, if
    given, is in ``unit`` and makes the setting optional."""

    unit: str
    default: float | None = None

    def __post_init__(self) -> None:
        default = self.default
        if default is not None and (
            isinstance(default, bool)
            or not isinstance(default, (int, float))
            or not _fits_float(default)
        ):
            raise EvaluationContractError(
                f"Quantity: default must be a finite number in its unit, got {default!r}"
            )
        if not isinstance(self.unit, str) or units.dimension(self.unit) is None:
            known = ", ".join(u for name in ("energy", "length") for u in units.units(name))
            raise EvaluationContractError(
                f"Quantity: unit must be one of {known}, got {self.unit!r}"
            )

    def convert(self, value: Json, path: str) -> float:
        """``value``, ``{"value": ..., "unit": ...}`` as the problem writes it, in ``unit``."""
        name = units.dimension(self.unit)
        assert name is not None
        if not isinstance(value, Mapping) or set(value) != {"value", "unit"}:
            raise EvaluationContractError(
                f"{path}: needs value and unit, such as value: 70 and unit: {self.unit}; "
                f"got {value!r}"
            )
        number, unit = value["value"], value["unit"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or (isinstance(number, float) and not math.isfinite(number))  # an int is finite
        ):
            raise EvaluationContractError(f"{path}.value: must be a finite number, got {number!r}")
        if not isinstance(unit, str) or units.dimension(unit) != name:
            raise EvaluationContractError(
                f"{path}.unit: must be a unit of {name} ({', '.join(units.units(name))}), "
                f"got {unit!r}"
            )
        converted = units.convert(number, unit, self.unit)
        if not math.isfinite(converted):
            raise EvaluationContractError(f"{path}.value: too large to express in {self.unit}")
        return converted


def _fits_float(number: int | float) -> bool:
    """Whether ``number`` is finite and within the range of a float, as an int need not be."""
    try:
        return math.isfinite(number)
    except OverflowError:
        return False


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


def _kind(declared: object) -> type:
    """The type a declared setting holds in the request: a quantity is a float."""
    if isinstance(declared, Quantity):
        return float
    return declared if isinstance(declared, type) else type(declared)


def _required(declared: object) -> bool:
    return isinstance(declared, type) or (
        isinstance(declared, Quantity) and declared.default is None
    )


def with_defaults(evaluator: object, settings: Mapping[str, Json]) -> Mapping[str, Json]:
    """``settings`` as ``evaluator`` receives them: each :class:`Quantity` converted to its
    declared unit, plus the default of every optional setting it declares. Raises
    :class:`EvaluationContractError` for a quantity that is not a ``value`` and a ``unit`` of the
    declared dimension."""
    merged = dict(settings)
    for key, declared in _declared(evaluator).items():
        if isinstance(declared, Quantity):
            if key in merged:
                merged[key] = declared.convert(merged[key], f"settings.{key}")
            elif declared.default is not None:
                merged[key] = float(declared.default)
        elif not isinstance(declared, type):
            merged.setdefault(key, declared)
    return merged


def check_settings(evaluator: object, settings: Mapping[str, Json]) -> None:
    """Reject ``settings``, as the evaluator receives them (:func:`with_defaults`), that do not
    match the ``settings`` it declares: unknown or missing keys, wrong types, a blank required
    text. Nothing is checked for an evaluator that declares none. It needs no candidate, so a
    problem's settings are checked once, before any candidate. Raises
    :class:`EvaluationContractError`."""
    declared = getattr(evaluator, "settings", None)
    if declared is None:
        return
    kinds = {key: _kind(entry) for key, entry in declared.items()}
    for key, kind in kinds.items():
        if kind not in _KINDS:
            raise EvaluationContractError(
                f"settings.{key}: cannot declare {kind.__name__}; use str, int, float, bool, "
                "list or dict (or a default of one of them), or Quantity(unit)"
            )
    unknown = [key for key in settings if key not in kinds]
    missing = [k for k, d in declared.items() if _required(d) and k not in settings]
    if unknown or missing:
        raise EvaluationContractError(
            f"settings: expected {sorted(kinds)}"
            + (f", unknown {unknown}" if unknown else "")
            + (f", missing {missing}" if missing else "")
        )
    for key, value in settings.items():
        if not _is(value, kinds[key]):
            raise EvaluationContractError(
                f"settings.{key}: expected {kinds[key].__name__}, got {value!r}"
            )
        # a required text (a collection name, say) cannot be blank; an optional one may be
        if isinstance(value, str) and not value.strip() and declared[key] is str:
            raise EvaluationContractError(f"settings.{key}: must not be blank")


def declared_collections(evaluator: object) -> dict[str, Collection]:
    """The ``collections`` ``evaluator`` declares, by role; raises
    :class:`EvaluationContractError` for a declaration that cannot be read."""
    declared = getattr(evaluator, "collections", None)
    if declared is None:
        return {}
    if not isinstance(declared, Mapping):
        raise EvaluationContractError(
            "collections: declare a mapping of role to Collection(type), such as "
            f"{{'hits': Collection('SimCalorimeterHit')}}; got {declared!r}"
        )
    if declared and getattr(evaluator, "stage", None) is None:
        raise EvaluationContractError(
            "collections are read from the file of the evaluator's stage; set stage = <stage>"
        )
    for role, collection in declared.items():
        if (
            not isinstance(role, str)
            or not role.isidentifier()
            or keyword.iskeyword(role)
            or role.startswith("_")
        ):
            raise EvaluationContractError(
                f"collections: role {role!r} must be a Python identifier such as hits, "
                "not a keyword"
            )
        if not isinstance(collection, Collection):
            raise EvaluationContractError(
                f"collections.{role}: declare it as Collection(type), got {collection!r}"
            )
    return dict(declared)


def collection_names(evaluator: object, given: Mapping[str, Json]) -> dict[str, str]:
    """The collection each role of ``evaluator`` reads: the one the problem names in ``given``,
    or the declared default. Raises :class:`EvaluationContractError` for an unknown or missing
    role or a name that is not text."""
    declared = declared_collections(evaluator)
    unknown = [role for role in given if role not in declared]
    missing = [r for r, c in declared.items() if c.default is None and r not in given]
    if unknown or missing:
        raise EvaluationContractError(
            f"collections: expected the roles {sorted(declared)}"
            + (f", unknown {unknown}" if unknown else "")
            + (f", missing {missing}" if missing else "")
        )
    names: dict[str, str] = {}
    for role, collection in declared.items():
        name = given.get(role, collection.default)
        if not isinstance(name, str) or not name.strip():
            raise EvaluationContractError(
                f"collections.{role}: must be the name of a collection, got {name!r}"
            )
        names[role] = name
    return names


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
                values = list(value)
            except TypeError:  # iterable in type only, like a zero-dimensional numpy array
                raise EvaluationContractError(
                    f"metric {stage}/{metric}: {value!r} is neither a number nor per-event values"
                ) from None
            if not values:  # may be a bug, so judging validity is left to the evaluator
                raise EvaluationContractError(
                    f"metric {stage}/{metric}: no per-event values to average; if nothing could "
                    "be measured, end the evaluation with self.invalid('...')"
                )
            measurements.append(Measurement.from_values(stage, metric, values, unit))
        else:  # a number; Measurement rejects anything else
            measurements.append(Measurement(stage, metric, cast(float, value), unit))
    return EvaluationResult.valid(measurements)


class Evaluator:
    """Base class of evaluators; see the module documentation. ``name`` and ``version`` are
    required; ``stage`` (the pipeline stage it reads), ``collections``, ``settings`` and
    ``metrics`` are optional."""

    name: str
    version: str
    stage: str | None = None
    collections: Mapping[str, Collection] | None = None
    settings: Mapping[str, Any] | None = None
    metrics: Mapping[MetricKey, str] | None = None

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult | Values:
        raise NotImplementedError

    def check(self, request: EvaluationRequest) -> None:
        """Reject settings that do not match ``settings`` (:func:`check_settings`); override it
        for rules beyond that, and call ``super().check(request)``."""
        check_settings(self, request.settings)

    def events(self, request: EvaluationRequest) -> Iterator[Event]:
        """Each event of the file of ``stage``, with the declared ``collections`` as attributes
        (``event.hits``), all read from that one event; see
        :func:`~k4opt.evaluation.edm4hep.read_events`."""
        declared = declared_collections(self)
        if self.stage is None or not declared:
            raise EvaluationContractError(
                "events() needs stage = <stage> and collections = {role: Collection(type)}"
            )
        if self.stage not in request.inputs:
            raise EvaluationContractError(f"inputs[{self.stage!r}] is missing")
        names = collection_names(self, request.collections)
        wanted = {role: (names[role], declared[role].type) for role in declared}
        return read_events(request.inputs[self.stage], wanted)

    def invalid(self, message: str) -> NoReturn:
        """End the evaluation: it ran, but its result is not usable."""
        raise EvaluationInvalid(message)

    def fail(self, message: str) -> NoReturn:
        """End the evaluation: the execution failed."""
        raise EvaluationFailed(message)
