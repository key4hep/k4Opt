"""The settings an evaluator declares, and the settings a problem gives it, checked.

``settings = {"threshold": Quantity("GeV"), "cut": 0.5}`` declares an evaluator's settings: a
type for a required key, a default value for an optional one, a :class:`Quantity` for a number
with a unit. :func:`check_settings` rejects missing and unknown keys, wrong types and blank
required text before anything runs; :func:`with_defaults` gives the settings as the evaluator
receives them, the defaults included and each quantity as a float in its declared unit; and
:func:`check_bound` says whether a parameter can set a setting. None of it needs a candidate or
an instance: each reads what the evaluator's class declares.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from k4opt.evaluation import units
from k4opt.evaluation.errors import EvaluationContractError
from k4opt.frozen import Json


@dataclass(frozen=True, slots=True)
class Quantity:
    """A setting with a unit. The problem writes ``value`` and ``unit`` (any unit of the same
    dimension, ``70`` and ``keV``); the evaluator receives a float in ``unit``: the value scaled
    exactly, then rounded once to the nearest float. A value whose float would be infinite, or
    zero or subnormal although it is not zero, is refused rather than handed on changed.
    ``default``, if given, is in ``unit`` and makes the setting optional."""

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
        if number != 0 and abs(converted) < sys.float_info.min:  # zero, or a subnormal float
            raise EvaluationContractError(
                f"{path}.value: {number!r} {unit} is too small to express in {self.unit}"
            )
        return converted


def _fits_float(number: int | float) -> bool:
    """Whether ``number`` is finite and within the range of a float, as an int need not be."""
    try:
        return math.isfinite(number)
    except OverflowError:
        return False


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


_VALUES = {float: "numbers", int: "integers", str: "strings", bool: "booleans"}


def _values(kind: type, unit: str | None) -> str:
    """What a parameter's values are, for a message: ``numbers in mm``, ``strings``."""
    text = _VALUES.get(kind, kind.__name__)
    if kind in (int, float):
        text += " without a unit" if unit is None else f" in {unit}"
    return text


def check_bound(evaluator: object, key: str, kind: type, unit: str | None) -> None:
    """Raise :class:`EvaluationContractError` unless a parameter whose values are of the type
    ``kind`` (``float``, ``int``, ``str`` or ``bool``), in ``unit``, can set the setting ``key`` of
    ``evaluator``: one it declares in its ``settings``, of a kind those values fit. A
    :class:`Quantity` takes numbers with a unit of its dimension, ``float`` numbers without one,
    ``int`` integers without one, ``str`` and ``bool`` their own; a ``list`` or a ``dict`` none."""
    declared = getattr(evaluator, "settings", None)
    if declared is None:
        raise EvaluationContractError(
            "cannot be set: the evaluator declares no settings, and a parameter sets only a "
            "setting the evaluator declares"
        )
    if key not in declared:
        raise EvaluationContractError(
            f"is not a setting the evaluator declares; it declares {', '.join(sorted(declared))}"
        )
    entry, given = declared[key], _values(kind, unit)
    number = kind in (int, float)
    if isinstance(entry, Quantity):
        dimension = units.dimension(entry.unit)
        assert dimension is not None
        if number and unit is not None and units.dimension(unit) == dimension:
            return
        raise EvaluationContractError(
            f"expects a quantity in {entry.unit}, a unit of {dimension} "
            f"({', '.join(units.units(dimension))}), but the parameter's values are {given}"
        )
    expected = _kind(entry)
    if expected in (list, dict):
        raise EvaluationContractError(f"is a {expected.__name__}, which no parameter can set")
    fits = number if expected is float else kind is expected
    if fits and unit is None:
        return
    if fits:
        raise EvaluationContractError(
            f"expects a number without a unit, but the parameter has unit {unit}, which it would "
            "lose; declare the setting a Quantity, or give the parameter no unit"
        )
    raise EvaluationContractError(
        f"expects {_KIND_NAMES[expected]}, but the parameter's values are {given}"
    )


_KIND_NAMES = {float: "a number", int: "an integer", str: "a string", bool: "a boolean"}


def check_settings(
    evaluator: object,
    settings: Mapping[str, Json],
    pending: Mapping[str, tuple[type, str | None]] | None = None,
) -> None:
    """Reject ``settings``, as the evaluator receives them (:func:`with_defaults`), that do not
    match the ``settings`` it declares: unknown or missing keys, wrong types, a blank required
    text. It needs no candidate, so a problem's settings are checked once, before any candidate.

    ``pending`` holds the settings parameters set, which every candidate gives instead, by key,
    each with the type of its parameter's values (``float``, ``int``, ``str``, ``bool``) and its
    unit: each must be one the evaluator declares, of a kind those values fit (a
    :class:`Quantity` takes a number with a unit of its dimension, ``float`` a number without
    one, ``int`` an integer without one, ``str`` and ``bool`` their own), and is not missing.
    Nothing but ``pending`` is checked for an evaluator that declares no settings, and it can
    take none. Raises :class:`EvaluationContractError`."""
    declared = getattr(evaluator, "settings", None)
    pending = pending or {}
    for key, (values, unit) in pending.items():
        try:
            check_bound(evaluator, key, values, unit)
        except EvaluationContractError as exc:
            raise EvaluationContractError(f"settings.{key}: {exc}") from None
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
    missing = [
        k for k, d in declared.items() if _required(d) and k not in settings and k not in pending
    ]
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
