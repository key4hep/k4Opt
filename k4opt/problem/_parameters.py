"""Internal: the ``parameters`` of a problem document, parsed in two phases.

The first phase reads each parameter on its own: its name, its domain (``type`` with
``bounds`` or ``choices``, ``unit``) and its ``targets``, a mapping from each kind of target to
what the value goes to there: ``geometry`` to attributes of the geometry, a stage type
(``simulation``) to options of that stage, ``evaluators`` to settings of evaluator bindings, each
one target or a list of them. Each is read into one destination
(:data:`~k4opt.problem.models.Destination`), whichever form the document writes it in. The second
phase, once the samples and evaluators are known, resolves each destination against them (the
samples a stage target leaves to the problem, the stage and evaluator binding each names),
refuses a destination set twice or also written by the problem, and checks that the parameter's
values can be written where it sends them. What the executors of the stages say of the values
(a positive energy, a known option) is checked with the samples
(:func:`k4opt.problem._samples._check_options`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from k4opt.geometry import InvalidTargetError, parse_reference
from k4opt.problem._fields import (
    _choice,
    _fail,
    _integer,
    _is_number,
    _mapping,
    _number,
    _optional_text,
    _sequence,
    _show,
    _text,
    _title,
)
from k4opt.problem._samples import _Place
from k4opt.problem.candidate import EXPRESSION_UNITS
from k4opt.problem.models import (
    Destination,
    EvaluatorSpec,
    GeometryDestination,
    Number,
    Parameter,
    ParameterKind,
    Sample,
    SettingDestination,
    StageDestination,
)
from k4opt.problem.stages import STAGE_TYPES, executors
from k4opt.program import ProgramError, Settable

_KINDS = ("continuous", "integer", "categorical")
_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_KEYS = ("title", "type", "bounds", "choices", "unit", "targets")
# the kinds of target, each the key of ``targets`` that names it: the stage types in between
_TARGETS = ("geometry", *STAGE_TYPES, "evaluators")

# the largest integer below which a double holds every integer
_EXACT_INTEGERS = 2**53


@dataclass(frozen=True, slots=True)
class _Stage:
    """A stage destination as the document writes it: ``samples`` is ``None`` where it leaves
    them to the problem."""

    stage: str
    option: str
    samples: tuple[str, ...] | None


type _Draft = GeometryDestination | _Stage | SettingDestination
"""A destination as the first phase reads it."""


@dataclass(frozen=True, slots=True)
class _Declared:
    """A parameter as the first phase reads it: its domain, and each destination with where it
    is written."""

    name: str
    kind: ParameterKind
    bounds: tuple[Number, Number] | None
    choices: tuple[str | bool | Number, ...] | None
    unit: str | None
    targets: tuple[tuple[str, _Draft], ...]
    title: str | None


def _parameters(raw: object) -> list[_Declared]:
    """The ``parameters`` mapping in the document's order, each read on its own."""
    m = _mapping(raw, "parameters", tuple(raw) if isinstance(raw, Mapping) else ())
    if not m:
        raise _fail("parameters", "must name at least one parameter")
    declared = []
    for name, value in m.items():
        if not _NAME.fullmatch(name):
            raise _fail(
                "parameters",
                f"parameter name {_show(name)} must start with a lowercase letter and hold only "
                "lowercase letters, digits and '_', at most 64 characters",
            )
        declared.append(_parameter(name, value, f"parameters.{name}"))
    return declared


def _parameter(name: str, raw: object, path: str) -> _Declared:
    m = _mapping(raw, path, _KEYS, ("type", "targets"))
    kind = cast(ParameterKind, _choice(m["type"], f"{path}.type", _KINDS))
    unit = _optional_text(m, "unit", path)
    title = _title(m, path)
    targets = _targets(m["targets"], f"{path}.targets")
    if kind == "categorical":
        for key in ("bounds", "unit"):
            if key in m:
                raise _fail(path, f"{key!r} is not allowed for a categorical parameter")
        if "choices" not in m:
            raise _fail(path, "categorical parameter needs 'choices'")
        choices = _choices(m["choices"], f"{path}.choices")
        return _Declared(name, kind, None, choices, None, targets, title)
    if "choices" in m:
        raise _fail(path, f"'choices' is only allowed for a categorical parameter, not {kind}")
    if "bounds" not in m:
        raise _fail(path, f"{kind} parameter needs 'bounds'")
    bounds = _bounds(m["bounds"], f"{path}.bounds", kind)
    return _Declared(name, kind, bounds, None, unit, targets, title)


def _bounds(raw: object, path: str, kind: str) -> tuple[Number, Number]:
    items = _sequence(raw, path)
    if len(items) != 2:
        raise _fail(path, f"must be [low, high], got {len(items)} item(s)")
    low: Number
    high: Number
    if kind == "integer":
        low, high = (_integer(item, f"{path}[{i}]") for i, item in enumerate(items))
    else:
        low, high = (_number(item, f"{path}[{i}]") for i, item in enumerate(items))
    if not low < high:
        raise _fail(path, f"low must be smaller than high, got [{low}, {high}]")
    return (low, high)


def _choices(raw: object, path: str) -> tuple[str | bool | Number, ...]:
    items = _sequence(raw, path)
    if not items:
        raise _fail(path, "must list at least one choice")
    classes: set[str] = set()
    for index, item in enumerate(items):
        where = f"{path}[{index}]"
        if isinstance(item, bool):
            classes.add("boolean")
        elif isinstance(item, str):
            classes.add("string")
            _text(item, where)
        elif _is_number(item):
            classes.add("number")
            _number(item, where)
        else:
            raise _fail(where, f"choices must be strings, booleans or numbers, got {_show(item)}")
    if len(classes) > 1:
        raise _fail(path, f"choices must all be of one kind, found {sorted(classes)}")
    for index, item in enumerate(items):
        if item in items[:index]:
            raise _fail(f"{path}[{index}]", f"duplicate choice {_show(item)}")
    return tuple(items)


def _targets(raw: object, path: str) -> tuple[tuple[str, _Draft], ...]:
    """The destinations of ``targets``, each with where it is written: by kind, in the order of
    the kinds (the geometry, the stages in pipeline order, the evaluators), whatever the order of
    the document's keys, then in the order each kind lists them."""
    m = _mapping(raw, path, _TARGETS)
    if not m:
        raise _fail(path, f"must name at least one target: {', '.join(_TARGETS)}")
    drafts: list[tuple[str, _Draft]] = []
    for key in (kind for kind in _TARGETS if kind in m):
        value, where = m[key], f"{path}.{key}"
        if key == "geometry":
            drafts += [
                (at, GeometryDestination(_text(item, at))) for at, item in _each(value, where)
            ]
        elif key == "evaluators":
            bindings = _mapping(value, where, tuple(value) if isinstance(value, Mapping) else ())
            if not bindings:
                raise _fail(where, "must name an evaluator binding and the setting it receives")
            for binding, settings in bindings.items():
                for at, setting in _each(settings, f"{where}.{binding}"):
                    drafts.append((at, SettingDestination(binding, _text(setting, at))))
        else:
            drafts += [(at, _option(key, item, at)) for at, item in _each(value, where)]
    return tuple(drafts)


def _each(value: object, path: str) -> list[tuple[str, object]]:
    """The targets of one kind as ``(where, target)``: one, or each item of a list of them."""
    if not isinstance(value, (list, tuple)):
        return [(path, value)]
    if not value:
        raise _fail(path, "must name at least one target")
    return [(f"{path}[{index}]", item) for index, item in enumerate(value)]


def _option(stage: str, raw: object, path: str) -> _Stage:
    """A stage target: the option alone (``physics.rangecut``), of the problem's only sample, or
    ``{option, samples}``."""
    if not isinstance(raw, Mapping):
        return _Stage(stage, _text(raw, path), None)
    m = _mapping(raw, path, ("option", "samples"), ("option",))
    samples = None
    if "samples" in m:
        listed = _sequence(m["samples"], f"{path}.samples")
        if not listed:
            raise _fail(f"{path}.samples", "must list at least one sample")
        samples = tuple(_text(s, f"{path}.samples[{i}]") for i, s in enumerate(listed))
        for index, sample in enumerate(samples):
            if sample in samples[:index]:
                raise _fail(f"{path}.samples[{index}]", f"{sample!r} is listed twice")
    return _Stage(stage, _text(m["option"], f"{path}.option"), samples)


# --- the second phase: destinations resolved against the samples and evaluators -------------


def _destinations(
    declared: list[_Declared],
    samples: Mapping[str, Sample],
    places: Mapping[tuple[str, str], _Place],
    evaluators: tuple[EvaluatorSpec, ...],
) -> tuple[Parameter, ...]:
    """Each parameter with its destinations resolved: a stage destination's samples, the stage
    and evaluator binding each names. Refused: a destination set twice, by one parameter or
    two; a stage option or a setting the problem also writes, which would leave a value that
    reads as used but never is; a value that cannot be written where it goes."""
    first: dict[tuple[str, ...], str] = {}  # each destination, by what it sets: who set it first
    parameters = []
    for item in declared:
        targets: list[Destination] = []
        for path, draft in item.targets:
            keys: list[tuple[str, ...]]
            if isinstance(draft, GeometryDestination):
                destination: Destination = draft
                keys = [("geometry", _canonical(draft, path))]
                _check_expression(item, path)
            elif isinstance(draft, SettingDestination):
                destination = draft
                keys = [("setting", draft.evaluator, draft.setting)]
                _check_setting(item, draft, path, evaluators)
            else:
                destination = _stage(item, draft, path, samples, places)
                keys = [
                    ("stage", sample, destination.stage, destination.option)
                    for sample in destination.samples
                ]
            for key in keys:
                if key in first:
                    by = first[key]
                    whom = "this parameter" if by == item.name else f"parameters.{by}"
                    raise _fail(path, f"{_label(key)} is set by {whom} already")
                first[key] = item.name
            targets.append(destination)
        parameters.append(
            Parameter(
                item.name,
                item.kind,
                tuple(targets),
                item.bounds,
                item.choices,
                item.unit,
                item.title,
            )
        )
    return tuple(parameters)


def _label(key: tuple[str, ...]) -> str:
    """What a destination sets, for a message: ``constants.X``, ``samples.gamma.simulation.x``,
    ``the setting threshold of evaluator calo.photon``."""
    if key[0] == "geometry":
        return f"the geometry's {key[1]}"
    if key[0] == "stage":
        return f"samples.{key[1]}.{key[2]}.{key[3]}"
    return f"the setting {key[2]} of evaluator {key[1]!r}"


def _canonical(destination: GeometryDestination, path: str) -> str:
    """The canonical form of the reference ``destination`` names, which a geometry looks it up
    by; two spellings of one reference are one destination."""
    try:
        return parse_reference(destination.canonical_target).text
    except InvalidTargetError as exc:
        raise _fail(path, str(exc)) from exc


def _stage(
    item: _Declared,
    draft: _Stage,
    path: str,
    samples: Mapping[str, Sample],
    places: Mapping[tuple[str, str], _Place],
) -> StageDestination:
    """A stage destination with its samples: those it lists, or the problem's only one. Each
    must run the stage by an executor that lets a parameter set the option, and not set the
    option itself."""
    if draft.samples is not None:
        named = draft.samples
        for index, sample in enumerate(named):
            if sample not in samples:
                known = f"its samples are {', '.join(samples)}" if samples else "it has none"
                raise _fail(
                    f"{path}.samples[{index}]", f"the problem has no sample {sample!r}; {known}"
                )
    elif len(samples) == 1:
        named = tuple(samples)
    elif not samples:
        raise _fail(path, f"the problem has no samples whose {draft.stage} it could set")
    else:
        raise _fail(
            path,
            f"the problem has the samples {', '.join(samples)}; say which this parameter sets: "
            f"{{option: {draft.option}, samples: [...]}}",
        )
    for sample in named:
        stage = samples[sample].stages.get(draft.stage)
        where = f"samples.{sample}.{draft.stage}"
        if stage is None:
            raise _fail(path, f"sample {sample} defines no {draft.stage} stage; add {where}")
        if stage.executor is None:
            raise _fail(
                path,
                f"{where} is given as a file, which is not run, so no parameter can set its "
                f"option {draft.option!r}",
            )
        try:
            settable = executors(draft.stage)[stage.executor].settable(draft.option)
        except ProgramError as exc:  # its message names the option: simulation.physics.pdgfile: ...
            reason = str(exc).partition(": ")[2] or str(exc)
            raise _fail(path, f"{stage.executor} option {draft.option!r}: {reason}") from None
        if draft.option in stage.options:
            raise _fail(
                places[sample, draft.stage].key(draft.option),
                f"is set by parameters.{item.name}; leave it out here",
            )
        _check_form(item, settable, draft.option, path)
    return StageDestination(draft.stage, draft.option, named)


def _check_setting(
    item: _Declared,
    destination: SettingDestination,
    path: str,
    evaluators: tuple[EvaluatorSpec, ...],
) -> None:
    """A setting destination names a binding of the problem none of whose entries writes the
    setting itself; whether the implementation declares it, and of what kind, is checked when
    the implementation is bound to the problem."""
    bindings = list(dict.fromkeys(entry.binding for entry in evaluators))
    if destination.evaluator not in bindings:
        raise _fail(
            path,
            f"the problem has no evaluator {destination.evaluator!r} (a binding: its as:, or its "
            f"name); its evaluators are {', '.join(bindings)}",
        )
    for index, entry in enumerate(evaluators):
        if entry.binding == destination.evaluator and destination.setting in entry.settings:
            raise _fail(
                f"evaluators[{index}].settings.{destination.setting}",
                f"is set by parameters.{item.name}; leave it out here",
            )


def _check_form(item: _Declared, settable: Settable, option: str, path: str) -> None:
    """The parameter's values can be written into ``option`` in the form ``settable`` says."""
    if settable.form == "text":
        if item.kind != "categorical" or not all(isinstance(c, str) for c in item.choices or ()):
            raise _fail(
                path,
                f"option {option!r} takes a name: the parameter must be categorical, with string "
                f"choices, not {item.kind}",
            )
        return
    if settable.form == "quantity":
        if item.kind == "categorical" or item.unit not in settable.units:
            given = item.kind if item.unit is None else f"{item.kind} in {item.unit}"
            raise _fail(
                path,
                f"option {option!r} takes a quantity in {', '.join(settable.units)}: the "
                f"parameter must be continuous or integer, with one of these units, not {given}",
            )
        _check_exact(item, path)
        return
    if settable.form == "value":
        if item.unit is not None:
            raise _fail(
                path,
                f"option {option!r} takes a bare value, whose unit the job decides (k4opt cannot "
                f"convert {item.unit!r} into it): the parameter must have no unit",
            )
        _check_exact(item, path)
        return
    _check_expression(item, path)


def _check_expression(item: _Declared, path: str) -> None:
    """The parameter's values can be written as a DD4hep expression, as the geometry and ddsim
    read them: a number, with a unit DD4hep knows if it has one, a string or a boolean, each
    number a double exactly."""
    if item.unit is not None and item.unit not in EXPRESSION_UNITS:
        raise _fail(
            path,
            f"unit {item.unit!r} cannot be written into a DD4hep expression; supported: "
            f"{', '.join(EXPRESSION_UNITS)}",
        )
    _check_exact(item, path)


def _check_exact(item: _Declared, path: str) -> None:
    """Every number of the parameter's domain is a double exactly: a destination reads its value
    as text that DD4hep or Geant4 evaluates as a double, and the record says that text, so a
    value the tool would round is refused rather than recorded as something it never used. A
    continuous parameter's values between exact bounds are checked for each candidate."""
    reason = "is written as text the destination reads as a double, which cannot hold"
    if item.kind == "integer":
        assert item.bounds is not None
        if max(abs(bound) for bound in item.bounds) > _EXACT_INTEGERS:
            raise _fail(path, f"the parameter {reason} every integer beyond 2**53")
        return
    numbers = item.bounds or tuple(c for c in item.choices or () if _is_number(c))
    for number in numbers:
        if not _double(cast(Any, number)):
            raise _fail(path, f"the parameter {reason} {_show(number)} exactly")


def _double(number: int | float) -> bool:
    """Whether a double holds ``number`` exactly."""
    try:
        return float(number) == number
    except OverflowError:
        return False
