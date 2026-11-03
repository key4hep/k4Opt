"""A candidate's values, as the problem defines them.

A candidate maps the name of each parameter to a value. :func:`validate_candidate` checks it
against the parameters' domains and keys it (:data:`~k4opt.evaluation.CandidateKey`),
problem-aware, so that ``True``, ``1`` and ``1.0`` are told apart where the parameter does;
:func:`parse_candidate` reads it from ``NAME=VALUE`` strings (``k4opt evaluate --param``)
exactly. :func:`configure` writes each value wherever its parameter sends it, into a
:class:`Configuration`: the geometry's attributes (as overrides, :func:`geometry_overrides`),
the options of stages and the settings of evaluators.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from k4opt.ddsim import Settable
from k4opt.evaluation import (
    CandidateKey,
    Scalar,
)
from k4opt.frozen import FrozenMap, Json
from k4opt.geometry import Override
from k4opt.problem.errors import CandidateError
from k4opt.problem.models import (
    EvaluatorSpec,
    GeometryDestination,
    Parameter,
    ProblemSpec,
    ResolvedProblem,
    Sample,
    StageDestination,
)
from k4opt.problem.stages import executors


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


def _check_number(value: object, name: str) -> int | float:
    if not _is_number(value):
        raise CandidateError(f"{name}: expected a number, got {value!r}")
    assert isinstance(value, (int, float))
    if isinstance(value, float) and not math.isfinite(value):
        raise CandidateError(f"{name}: value must be finite, got {value!r}")
    return value


def validate_value(parameter: Parameter, value: object) -> tuple[str, str]:
    """Return ``(tag, canonical value)`` of ``value`` for ``parameter``: its type and domain
    checked, as :func:`validate_candidate` checks each; :class:`CandidateError` otherwise."""
    name = parameter.name
    if parameter.kind == "continuous":
        number = _check_number(value, name)
    elif parameter.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CandidateError(f"{name}: expected an integer, got {value!r}")
        number = value
    else:
        choices = parameter.choices or ()
        if all(isinstance(choice, str) for choice in choices):
            if not isinstance(value, str) or value not in choices:
                raise CandidateError(f"{name}: {value!r} is not one of {list(choices)!r}")
            return "str", value
        if all(isinstance(choice, bool) for choice in choices):
            if not isinstance(value, bool) or value not in choices:
                raise CandidateError(f"{name}: {value!r} is not one of {list(choices)!r}")
            return "bool", "true" if value else "false"
        number = _check_number(value, name)
        if not any(number == choice for choice in choices):
            raise CandidateError(f"{name}: {value!r} is not one of {list(choices)!r}")
        return "num", _canonical_number(number)
    assert parameter.bounds is not None
    low, high = parameter.bounds
    if not low <= number <= high:
        raise CandidateError(f"{name}: {value!r} is outside the bounds [{low}, {high}]")
    return ("int" if parameter.kind == "integer" else "num"), _canonical_number(number)


def validate_candidate(
    spec: ProblemSpec, candidate: Mapping[str, Scalar]
) -> tuple[dict[str, Scalar], CandidateKey]:
    """Check ``candidate`` against every parameter; return it in problem order with its key."""
    if not isinstance(candidate, Mapping):
        raise CandidateError(f"candidate must be a mapping, got {type(candidate).__name__}")
    declared = [parameter.name for parameter in spec.parameters]
    bad_keys = [key for key in candidate if not isinstance(key, str)]
    if bad_keys:
        raise CandidateError(f"candidate keys must be strings, got {bad_keys!r}")
    unknown = [key for key in candidate if key not in declared]
    if unknown:
        raise CandidateError(f"candidate has unknown parameter(s): {', '.join(unknown)}")
    missing = [name for name in declared if name not in candidate]
    if missing:
        raise CandidateError(f"candidate is missing parameter(s): {', '.join(missing)}")
    values: dict[str, Scalar] = {}
    key: list[tuple[str, str, str, str]] = []
    for parameter, name in zip(spec.parameters, declared, strict=True):
        value = candidate[name]
        tag, canonical = validate_value(parameter, value)
        values[name] = value
        key.append((name, parameter.kind, tag, canonical))
    return values, tuple(key)


_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")


def _int(name: str, text: str) -> int:
    """``int(text)`` for a text that matches ``_INTEGER``."""
    try:
        return int(text)
    except ValueError:  # more digits than Python converts (sys.get_int_max_str_digits())
        raise CandidateError(f"{name}: {text[:20]}... has too many digits") from None


def _parse_value(parameter: Parameter, text: str) -> Scalar:
    """The value ``text`` stands for, read according to the kind of ``parameter``."""
    name = parameter.name
    choices = parameter.choices or ()
    if parameter.kind == "categorical" and all(isinstance(c, str) for c in choices):
        return text
    if parameter.kind == "categorical" and all(isinstance(c, bool) for c in choices):
        if text not in ("true", "false"):
            raise CandidateError(f"{name}: expected true or false, got {text!r}")
        return text == "true"
    if parameter.kind == "integer":
        if not _INTEGER.fullmatch(text):
            raise CandidateError(f"{name}: expected an integer, got {text!r}")
        return _int(name, text)
    if not _DECIMAL.fullmatch(text):
        raise CandidateError(f"{name}: expected a number, got {text!r}")
    if parameter.kind == "categorical":
        # Compare with the declared choices exactly (Decimal vs int or float is exact), and
        # return the choice itself: float(text) would merge 2**53 and 2**53+1.
        exact = Decimal(text)
        numeric = [c for c in choices if not isinstance(c, bool)]
        for choice in numeric:
            if exact == choice:
                return choice
        for choice in numeric:  # a float choice such as 1e23 is not the decimal 10**23 ...
            # ... so also accept its shortest decimal spelling, never a rounded neighbor
            if (
                isinstance(choice, float)
                and math.isfinite(choice)
                and exact == Decimal(repr(choice))
            ):
                return choice
        # never round an unmatched value into a neighboring choice
        raise CandidateError(f"{name}: {text!r} is not one of {list(choices)!r}")
    if _INTEGER.fullmatch(text):
        # an integer literal stays exact where float would change it (above 2**53) or overflow
        number = _int(name, text)
        try:
            return float(number) if float(number) == number else number
        except OverflowError:
            return number
    return float(text)


def parse_candidate(spec: ProblemSpec, assignments: Sequence[str]) -> dict[str, Scalar]:
    """The candidate ``NAME=VALUE`` strings give, each value read as its parameter's kind."""
    by_name = {p.name: p for p in spec.parameters}
    candidate: dict[str, Scalar] = {}
    for assignment in assignments:
        name, equals, text = assignment.partition("=")  # a name holds no "=", a value may
        if not equals:
            raise CandidateError(f"--param {assignment!r}: expected NAME=VALUE")
        if name in candidate:
            raise CandidateError(f"--param {name} is given more than once")
        parameter = by_name.get(name)
        # an unknown name is reported by the candidate validation, with the known ones
        candidate[name] = text if parameter is None else _parse_value(parameter, text)
    return candidate


EXPRESSION_UNITS = ("um", "mm", "cm", "m", "mrad", "rad", "deg", "MeV", "GeV", "TeV")
"""The units a value is written with into a DD4hep expression (``6.0*mm``), in the geometry and
in a ddsim option; a parameter with another unit is refused when the problem is read."""


def expression(value: Scalar, unit: str | None) -> str:
    """``value`` as a DD4hep expression: ``6.0*mm`` with a unit, the number alone without one,
    ``true`` or ``false`` for a boolean, a string as it is."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return repr(value) if unit is None else f"{value!r}*{unit}"


def edges(parameter: Parameter) -> tuple[Scalar, ...]:
    """The ends of the domain of ``parameter``: its two bounds, or each of its choices."""
    if parameter.bounds is not None:
        return parameter.bounds
    assert parameter.choices is not None
    return parameter.choices


def stage_value(settable: Settable, parameter: Parameter, value: Scalar) -> Json:
    """``value`` of ``parameter`` as the option of a stage it sets holds it: ``{value, unit}``
    for a quantity, a DD4hep expression for a number with a unit that ddsim reads as text, the
    value itself otherwise."""
    if settable.form == "quantity":
        return {"value": value, "unit": parameter.unit}
    if settable.form == "expression" and parameter.unit is not None:
        return expression(value, parameter.unit)
    return value


def setting_value(parameter: Parameter, value: Scalar) -> Json:
    """``value`` of ``parameter`` as a setting it sets holds it: ``{value, unit}`` with a unit,
    as a problem writes a quantity, the value itself without one."""
    return value if parameter.unit is None else {"value": value, "unit": parameter.unit}


def _exact(parameter: Parameter, value: Scalar, where: str) -> None:
    """Refuse a number that a destination which reads it as a double would round."""
    if not _is_number(value):
        return
    try:
        exact = float(value) == value
    except OverflowError:
        exact = False
    if not exact:
        raise CandidateError(
            f"{parameter.name}: {value!r} is written into {where} as text that is read as a "
            "double, which cannot hold it exactly"
        )


@dataclass(frozen=True, slots=True)
class Configuration:
    """What one candidate sets wherever its parameters go: the text of each attribute of the
    geometry they set (``geometry``, in the order of
    :attr:`~k4opt.problem.ProblemSpec.geometry_destinations`), every sample with each of its
    stages complete (``samples``) and every evaluator entry with its settings complete
    (``evaluators``, in problem order). Made from the problem and the candidate alone
    (:func:`configure`), so a record is checked against it without anything that ran."""

    geometry: tuple[str, ...]
    samples: Mapping[str, Sample]
    evaluators: tuple[EvaluatorSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry", tuple(self.geometry))
        object.__setattr__(self, "samples", FrozenMap(self.samples))
        object.__setattr__(self, "evaluators", tuple(self.evaluators))


def configure(spec: ProblemSpec, values: Mapping[str, Scalar]) -> Configuration:
    """The configuration of the candidate ``values`` (valid: :func:`validate_candidate`): each
    value written into every destination of its parameter, in the form the destination reads.

    Nothing of ``spec`` changes; the stages and entries the candidate completes are new.
    Raises :class:`CandidateError` for a number no double holds exactly where a destination
    reads it as text that is evaluated as a double (the geometry, a stage option), so that no
    record says a value the tool never used.
    """
    geometry: list[str] = []
    options: dict[tuple[str, str], dict[str, Json]] = {}
    settings: dict[str, dict[str, Json]] = {}
    for parameter, destination in spec.destinations:
        value = values[parameter.name]
        if isinstance(destination, GeometryDestination):
            _exact(parameter, value, destination.label)
            geometry.append(expression(value, parameter.unit))
        elif isinstance(destination, StageDestination):
            for sample in destination.samples:
                stage = spec.samples[sample].stages[destination.stage]
                assert stage.executor is not None  # parse_problem: a stage that runs
                settable = executors(destination.stage)[stage.executor].settable(destination.option)
                if settable.form != "text":
                    _exact(parameter, value, f"samples.{sample}.{destination.label}")
                written = stage_value(settable, parameter, value)
                options.setdefault((sample, destination.stage), {})[destination.option] = written
        else:
            written = setting_value(parameter, value)
            settings.setdefault(destination.evaluator, {})[destination.setting] = written
    samples = {
        name: replace(
            sample,
            stages={
                stage_name: replace(stage, options={**stage.options, **set_by})
                if (set_by := options.get((name, stage_name)))
                else stage
                for stage_name, stage in sample.stages.items()
            },
        )
        for name, sample in spec.samples.items()
    }
    evaluators = tuple(
        replace(entry, settings={**entry.settings, **settings[entry.binding]})
        if entry.binding in settings
        else entry
        for entry in spec.evaluators
    )
    return Configuration(tuple(geometry), samples, evaluators)


def geometry_overrides(problem: ResolvedProblem, configuration: Configuration) -> list[Override]:
    """The override of each attribute of the geometry the candidate of ``configuration`` sets,
    in the order of the problem's geometry destinations."""
    return [
        Override(target=target.target, value=text, file=target.file)
        for target, text in zip(problem.targets, configuration.geometry, strict=True)
    ]
