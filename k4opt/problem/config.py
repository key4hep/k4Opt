"""Loading and validation of the problem YAML document (``k4opt.problem/v1alpha1``).

:func:`parse_problem` makes the checks that need no Key4hep: shape, types, ranges, syntactic
duplicates, the executors and options of each sample's stages (its own, or the shared ``stages``
it takes or completes) and whether they fit together and fit the evaluators that measure the
sample. :func:`load_problem` adds the resolution of the
geometry targets; that an evaluator or metric is defined, or that the bounds are physically
sensible, is deliberately not decided. See ``docs/problem.md``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from k4opt.geometry import AmbiguousTargetError, GeometrySource, TargetError
from k4opt.problem._evaluators import _evaluators
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
)
from k4opt.problem._samples import _check_pipeline, _samples, _shared_stages
from k4opt.problem.errors import ProblemConfigError
from k4opt.problem.models import (
    SCHEMA_VERSION,
    Channel,
    Constraint,
    Direction,
    Environment,
    EvaluatorSpec,
    GeometrySelector,
    Key4hepStack,
    MetricRef,
    Number,
    Objective,
    Parameter,
    ParameterKind,
    ProblemSpec,
    ResolvedProblem,
)
from k4opt.yamlio import UniqueKeyLoader

_TOP_KEYS = (
    "schema_version",
    "name",
    "environment",
    "geometry",
    "parameters",
    "stages",
    "samples",
    "evaluators",
    "objectives",
    "constraints",
)
_REQUIRED_TOP = (
    "schema_version",
    "environment",
    "geometry",
    "parameters",
    "evaluators",
    "objectives",
)
_CHANNELS = ("release", "nightly")
_KINDS = ("continuous", "integer", "categorical")
_DIRECTIONS = ("minimize", "maximize")
_SCHEMA_FAMILY = "k4opt.problem/"
_PARAMETER_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


def load_problem(path: str | os.PathLike[str]) -> ProblemSpec:
    """Read the UTF-8 YAML file ``path``, validate it and resolve its geometry parameters.

    The geometry is located in the active k4geo tree (``$K4GEO`` for a relative path) and
    discovered once; every ``parameters[*].geometry`` must then select exactly one attribute
    of it. Raises :class:`ProblemConfigError` for an invalid document or an unresolvable
    target and :class:`~k4opt.geometry.GeometryError` when the geometry itself cannot be
    located or read.
    """
    return resolve_problem(path).spec


def resolve_problem(path: str | os.PathLike[str]) -> ResolvedProblem:
    """:func:`load_problem`, keeping what it resolved: the discovered geometry and the attribute
    each parameter sets."""
    problem_path = Path(path).expanduser()
    if not problem_path.is_file():
        raise ProblemConfigError(f"{problem_path}: problem file not found")
    try:
        text = problem_path.read_bytes().decode("utf-8")
        data = yaml.load(text, Loader=UniqueKeyLoader)
    except UnicodeDecodeError as exc:
        raise ProblemConfigError(f"{problem_path}: not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise ProblemConfigError(f"{problem_path}: cannot read: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ProblemConfigError(f"{problem_path}: invalid YAML: {exc}") from exc
    except RecursionError as exc:
        raise ProblemConfigError(f"{problem_path}: YAML is nested too deeply") from exc
    return _resolve_geometry(parse_problem(data, source=problem_path), problem_path)


def parse_problem(data: object, *, source: str | os.PathLike[str] | None = None) -> ProblemSpec:
    """Validate an already-decoded document; no file system or Key4hep is needed.

    ``source`` only adds context to error messages and supplies the default ``name``
    (the file stem) when the document has none.
    """
    try:
        return _parse(data, None if source is None else Path(source))
    except ProblemConfigError as exc:
        if source is None:
            raise
        raise ProblemConfigError(f"{source}: {exc}") from exc
    except RecursionError as exc:
        where = "" if source is None else f"{source}: "
        raise ProblemConfigError(f"{where}document is nested too deeply") from exc


def _resolve_geometry(spec: ProblemSpec, problem_path: Path) -> ResolvedProblem:
    """``spec`` with its geometry discovered once and every parameter target resolved."""
    source = GeometrySource(spec.geometry)
    catalog = source.discover()
    targets = []
    for parameter in spec.parameters:
        reference = parameter.geometry.canonical_target
        where = f"{problem_path}: parameters.{parameter.name}.geometry: target {reference!r}"
        try:
            targets.append(catalog.lookup(reference))
        except AmbiguousTargetError as exc:
            listing = "\n".join(
                f"  - {c.file}:{c.line} (value {c.value!r})" for c in exc.candidates
            )
            raise ProblemConfigError(
                f"{where} matches {len(exc.candidates)} attributes in different files and "
                "cannot currently be selected unambiguously from the problem "
                f"specification:\n{listing}"
            ) from exc
        except TargetError as exc:
            raise ProblemConfigError(
                f"{where} cannot be resolved in {source.compact}: {exc}"
            ) from exc
    return ResolvedProblem(spec, problem_path.absolute(), source, catalog, tuple(targets))


# --- helpers ---------------------------------------------------------------------------


# --- sections --------------------------------------------------------------------------


def _parse(data: object, source: Path | None) -> ProblemSpec:
    if isinstance(data, Mapping) and "schema_version" in data:
        _check_version(data["schema_version"])  # before anything else: other keys may differ
    top = _mapping(data, "", _TOP_KEYS, _REQUIRED_TOP)

    name = _optional_text(top, "name", "")
    if name is None and source is not None:
        name = source.stem
    environment = _environment(top["environment"])
    geometry = _text(top["geometry"], "geometry")
    parameters = _parameters(top["parameters"])
    shared = _shared_stages(top["stages"]) if "stages" in top else {}
    if "samples" in top:
        samples, places = _samples(top["samples"], shared)
    elif shared:
        raise _fail("stages", "there are no samples to take these stages; add samples:")
    else:
        samples, places = {}, {}
    evaluators = _evaluators(top["evaluators"], list(samples))
    objectives = _objectives(top["objectives"], evaluators)
    constraints = _constraints(top["constraints"], evaluators) if "constraints" in top else ()
    _check_pipeline(samples, places, evaluators, parameters)
    return ProblemSpec(
        name=name,
        environment=environment,
        geometry=geometry,
        parameters=parameters,
        samples=samples,
        evaluators=evaluators,
        objectives=objectives,
        constraints=constraints,
    )


def _check_version(value: object) -> None:
    if not isinstance(value, str):
        raise _fail("schema_version", f"must be a string, got {type(value).__name__}")
    if not value.startswith(_SCHEMA_FAMILY):
        raise _fail(
            "schema_version",
            f"{_show(value)} is not a k4opt problem schema; use {SCHEMA_VERSION!r}",
        )
    if value != SCHEMA_VERSION:
        raise _fail(
            "schema_version", f"unsupported version {value!r}; this k4opt reads {SCHEMA_VERSION!r}"
        )


def _environment(raw: object) -> Environment:
    env = _mapping(raw, "environment", ("key4hep",), ("key4hep",))
    stack = _mapping(
        env["key4hep"], "environment.key4hep", ("channel", "version"), ("channel", "version")
    )
    return Environment(
        key4hep=Key4hepStack(
            channel=cast(
                Channel, _choice(stack["channel"], "environment.key4hep.channel", _CHANNELS)
            ),
            version=_text(stack["version"], "environment.key4hep.version"),
        )
    )


def _parameters(raw: object) -> tuple[Parameter, ...]:
    m = _mapping(raw, "parameters", tuple(raw) if isinstance(raw, Mapping) else ())
    if not m:
        raise _fail("parameters", "must name at least one parameter")
    parameters = []
    first: dict[str, str] = {}
    for name, value in m.items():
        if not _PARAMETER_NAME.fullmatch(name):
            raise _fail(
                "parameters",
                f"parameter name {_show(name)} must start with a lowercase letter and hold only "
                "lowercase letters, digits and '_', at most 64 characters",
            )
        parameter = _parameter(name, value, f"parameters.{name}")
        target = parameter.geometry.canonical_target
        if target in first:
            raise _fail(
                f"parameters.{name}.geometry",
                f"sets the target {target!r} of parameters.{first[target]} too",
            )
        first[target] = name
        parameters.append(parameter)
    return tuple(parameters)


def _parameter(name: str, raw: object, path: str) -> Parameter:
    m = _mapping(raw, path, ("geometry", "type", "bounds", "choices", "unit"), ("geometry", "type"))
    selector = _selector(m["geometry"], f"{path}.geometry")
    kind = cast(ParameterKind, _choice(m["type"], f"{path}.type", _KINDS))
    unit = _optional_text(m, "unit", path)

    if kind == "categorical":
        for key in ("bounds", "unit"):
            if key in m:
                raise _fail(path, f"{key!r} is not allowed for a categorical parameter")
        if "choices" not in m:
            raise _fail(path, "categorical parameter needs 'choices'")
        return Parameter(name, selector, kind, choices=_choices(m["choices"], f"{path}.choices"))
    if "choices" in m:
        raise _fail(path, f"'choices' is only allowed for a categorical parameter, not {kind}")
    if "bounds" not in m:
        raise _fail(path, f"{kind} parameter needs 'bounds'")
    bounds = _bounds(m["bounds"], f"{path}.bounds", kind)
    return Parameter(name, selector, kind, bounds=bounds, unit=unit)


def _selector(raw: object, path: str) -> GeometrySelector:
    m = _mapping(raw, path, ("constant", "target"))
    if len(m) != 1:
        raise _fail(path, "needs exactly one of 'constant' or 'target'")
    if "constant" in m:
        return GeometrySelector(constant=_text(m["constant"], f"{path}.constant"))
    return GeometrySelector(target=_text(m["target"], f"{path}.target"))


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


def _owner(
    m: Mapping[str, Any], path: str, evaluators: tuple[EvaluatorSpec, ...]
) -> tuple[str | None, str]:
    """``(sample, evaluator)``: the evaluation an objective or constraint takes its metric from.

    Either may be left out when the problem leaves one choice: the sample of an evaluator that
    measures only one, the evaluator of a sample that only one measures, or both when the
    problem has only one evaluation. ``sample`` is ``None`` for an evaluator that measures the
    candidate."""
    measures: dict[str, list[str]] = {}  # the samples of each binding; none: the candidate
    for entry in evaluators:
        measures.setdefault(entry.binding, []).extend(entry.samples)
    sample = _text(m["sample"], f"{path}.sample") if "sample" in m else None
    if "evaluator" in m:
        binding = _text(m["evaluator"], f"{path}.evaluator")
        if binding not in measures:
            raise _fail(
                f"{path}.evaluator",
                f"the problem has no evaluator {binding!r}; its evaluators are "
                f"{', '.join(measures)}",
            )
        measured = measures[binding]
        if not measured:
            if sample is not None:
                raise _fail(
                    f"{path}.sample",
                    f"evaluator {binding!r} measures the candidate, not a sample; remove sample:",
                )
            return None, binding
        if sample is None:
            if len(measured) > 1:
                raise _fail(
                    path,
                    f"evaluator {binding!r} measures the samples {', '.join(measured)}; add "
                    "sample: with the one this metric is measured on",
                )
            return measured[0], binding
        if sample not in measured:
            raise _fail(
                f"{path}.sample",
                f"evaluator {binding!r} does not measure {sample!r}; it measures "
                f"{', '.join(measured)}",
            )
        return sample, binding
    evaluations: list[tuple[str | None, str]] = [
        (s, b) for b, on in measures.items() for s in ([*on] if on else [None])
    ]
    if sample is not None:
        held = [b for s, b in evaluations if s == sample]
        if not held:
            raise _fail(f"{path}.sample", f"no evaluator of the problem measures {sample!r}")
        if len(held) > 1:
            raise _fail(
                path,
                f"{sample} is measured by the evaluators {', '.join(held)}; add evaluator: with "
                "one of them",
            )
        return sample, held[0]
    if len(evaluations) > 1:
        labels = [b if s is None else f"{s}/{b}" for s, b in evaluations]
        raise _fail(
            path,
            f"the problem has the evaluations {', '.join(labels)}; add evaluator: (and sample:) "
            "to say which one measures this metric",
        )
    return evaluations[0]


def _objectives(raw: object, evaluators: tuple[EvaluatorSpec, ...]) -> tuple[Objective, ...]:
    items = _sequence(raw, "objectives")
    if not items:
        raise _fail("objectives", "must list at least one objective")
    objectives: list[Objective] = []
    first: dict[MetricRef, int] = {}
    for index, item in enumerate(items):
        path = f"objectives[{index}]"
        m = _mapping(
            item,
            path,
            ("sample", "evaluator", "stage", "metric", "direction"),
            ("stage", "metric", "direction"),
        )
        sample, evaluator = _owner(m, path, evaluators)
        objective = Objective(
            evaluator=evaluator,
            stage=_text(m["stage"], f"{path}.stage"),
            metric=_text(m["metric"], f"{path}.metric"),
            direction=cast(Direction, _choice(m["direction"], f"{path}.direction", _DIRECTIONS)),
            sample=sample,
        )
        if objective.ref in first:
            raise _fail(path, f"duplicates objectives[{first[objective.ref]}] ({objective.label})")
        first[objective.ref] = index
        objectives.append(objective)
    return tuple(objectives)


def _constraints(raw: object, evaluators: tuple[EvaluatorSpec, ...]) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []
    first: dict[MetricRef, int] = {}
    for index, item in enumerate(_sequence(raw, "constraints")):
        path = f"constraints[{index}]"
        m = _mapping(
            item,
            path,
            ("sample", "evaluator", "stage", "metric", "min", "max", "unit"),
            ("stage", "metric"),
        )
        if "min" not in m and "max" not in m:
            raise _fail(path, "needs 'min' and/or 'max'")
        low = _number(m["min"], f"{path}.min") if "min" in m else None
        high = _number(m["max"], f"{path}.max") if "max" in m else None
        if low is not None and high is not None and low > high:
            raise _fail(path, f"min must not exceed max, got min={low}, max={high}")
        sample, evaluator = _owner(m, path, evaluators)
        constraint = Constraint(
            evaluator=evaluator,
            stage=_text(m["stage"], f"{path}.stage"),
            metric=_text(m["metric"], f"{path}.metric"),
            min=low,
            max=high,
            unit=_optional_text(m, "unit", path),
            sample=sample,
        )
        if constraint.ref in first:
            raise _fail(
                path, f"duplicates constraints[{first[constraint.ref]}] ({constraint.label})"
            )
        first[constraint.ref] = index
        constraints.append(constraint)
    return tuple(constraints)
