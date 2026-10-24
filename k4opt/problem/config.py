"""Loading and validation of the problem YAML document (``k4opt.problem/v1alpha1``).

The checks are those that need no Key4hep: shape, types, ranges and syntactic duplicates.
Whether a geometry constant exists, an evaluator or metric is defined, or the bounds are
physically sensible is deliberately not decided here. See ``docs/problem.md``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from k4opt.problem.errors import ProblemConfigError
from k4opt.problem.models import (
    SCHEMA_VERSION,
    Channel,
    Constraint,
    Direction,
    Environment,
    Evaluation,
    GeometrySelector,
    GeometrySpec,
    Json,
    Key4hepStack,
    Number,
    Objective,
    Parameter,
    ParameterKind,
    ProblemSpec,
    freeze,
)

_TOP_KEYS = (
    "schema_version",
    "name",
    "environment",
    "geometry",
    "parameters",
    "evaluation",
    "objectives",
    "constraints",
)
_REQUIRED_TOP = (
    "schema_version",
    "environment",
    "geometry",
    "parameters",
    "evaluation",
    "objectives",
)
_CHANNELS = ("release", "nightly")
_KINDS = ("continuous", "integer", "categorical")
_DIRECTIONS = ("minimize", "maximize")
_BACKENDS = ("ddsim", "k4run")
_SCHEMA_FAMILY = "k4opt.problem/"


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.safe_load`` that rejects duplicate mapping keys instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue  # `<<: *anchor` is expanded by the base class; explicit keys win
            key = self.construct_object(key_node, deep=True)
            if isinstance(key, Hashable):
                if key in seen:
                    raise yaml.YAMLError(
                        f"duplicate key {key!r} (line {key_node.start_mark.line + 1})"
                    )
                seen.add(key)
        return super().construct_mapping(node, deep)  # rejects unhashable keys itself


def load_problem(path: str | os.PathLike[str]) -> ProblemSpec:
    """Read the UTF-8 YAML file ``path`` and validate it as a problem definition."""
    problem_path = Path(path).expanduser()
    if not problem_path.is_file():
        raise ProblemConfigError(f"{problem_path}: problem file not found")
    try:
        text = problem_path.read_bytes().decode("utf-8")
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except UnicodeDecodeError as exc:
        raise ProblemConfigError(f"{problem_path}: not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise ProblemConfigError(f"{problem_path}: cannot read: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ProblemConfigError(f"{problem_path}: invalid YAML: {exc}") from exc
    except RecursionError as exc:
        raise ProblemConfigError(f"{problem_path}: YAML is nested too deeply") from exc
    return parse_problem(data, source=problem_path)


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


# --- helpers ---------------------------------------------------------------------------


def _fail(path: str, message: str) -> ProblemConfigError:
    return ProblemConfigError(f"{path or 'top level'}: {message}")


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _show(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _mapping(
    value: object, path: str, allowed: tuple[str, ...], required: tuple[str, ...] = ()
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, f"must be a mapping, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise _fail(path, f"keys must be strings, got {_show(key)}")
    unknown = [key for key in value if key not in allowed]
    if unknown:
        raise _fail(path, f"unknown key(s) {unknown}; allowed: {sorted(allowed)}")
    missing = [key for key in required if key not in value]
    if missing:
        raise _fail(path, f"missing required key(s) {missing}")
    return value


def _sequence(value: object, path: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise _fail(path, f"must be a list, got {type(value).__name__}")
    return list(value)


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        got = "a blank string" if isinstance(value, str) else f"{type(value).__name__}"
        quote = ""
        if value is not None and not isinstance(value, (str, list, tuple, Mapping)):
            got += f" {_show(value)}"
            quote = "; quote it if it is text"
        raise _fail(path, f"must be a non-empty string, got {got}{quote}")
    return value


def _optional_text(m: Mapping[str, Any], key: str, path: str) -> str | None:
    return _text(m[key], _join(path, key)) if key in m else None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: object, path: str) -> Number:
    if not _is_number(value):
        raise _fail(path, f"must be a number, got {type(value).__name__} {_show(value)}")
    assert isinstance(value, (int, float))
    if not math.isfinite(value):
        raise _fail(path, f"must be finite, got {value!r}")
    return value


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(path, f"must be an integer, got {type(value).__name__} {_show(value)}")
    if minimum is not None and value < minimum:
        raise _fail(path, f"must be >= {minimum}, got {value}")
    return value


def _choice(value: object, path: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _fail(path, f"must be one of {list(allowed)}, got {_show(value)}")
    return value


# --- sections --------------------------------------------------------------------------


def _parse(data: object, source: Path | None) -> ProblemSpec:
    if isinstance(data, Mapping) and "schema_version" in data:
        _check_version(data["schema_version"])  # before anything else: other keys may differ
    top = _mapping(data, "", _TOP_KEYS, _REQUIRED_TOP)

    name = _optional_text(top, "name", "")
    if name is None and source is not None:
        name = source.stem
    return ProblemSpec(
        name=name,
        environment=_environment(top["environment"]),
        geometry=GeometrySpec(
            entry=_text(
                _mapping(top["geometry"], "geometry", ("entry",), ("entry",))["entry"],
                "geometry.entry",
            )
        ),
        parameters=_parameters(top["parameters"]),
        evaluation=_evaluation(top["evaluation"]),
        objectives=_objectives(top["objectives"]),
        constraints=_constraints(top["constraints"]) if "constraints" in top else (),
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
    items = _sequence(raw, "parameters")
    if not items:
        raise _fail("parameters", "must list at least one parameter")
    parameters = [_parameter(item, f"parameters[{i}]") for i, item in enumerate(items)]
    first: dict[str, int] = {}
    for index, parameter in enumerate(parameters):
        target = parameter.geometry.canonical_target
        if target in first:
            raise _fail(
                f"parameters[{index}]",
                f"duplicates parameters[{first[target]}] (target {target!r})",
            )
        first[target] = index
    return tuple(parameters)


def _parameter(raw: object, path: str) -> Parameter:
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
        return Parameter(selector, kind, choices=_choices(m["choices"], f"{path}.choices"))
    if "choices" in m:
        raise _fail(path, f"'choices' is only allowed for a categorical parameter, not {kind}")
    if "bounds" not in m:
        raise _fail(path, f"{kind} parameter needs 'bounds'")
    return Parameter(selector, kind, bounds=_bounds(m["bounds"], f"{path}.bounds", kind), unit=unit)


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


def _evaluation(raw: object) -> Evaluation:
    path = "evaluation"
    m = _mapping(
        raw,
        path,
        ("evaluator", "version", "events", "seed", "settings", "backend"),
        ("evaluator", "version"),
    )
    settings: Mapping[str, Json] = {}
    if "settings" in m:
        if not isinstance(m["settings"], Mapping):
            raise _fail(
                "evaluation.settings", f"must be a mapping, got {type(m['settings']).__name__}"
            )
        frozen = freeze(m["settings"], "evaluation.settings")
        assert isinstance(frozen, Mapping)
        settings = frozen
    return Evaluation(
        evaluator=_text(m["evaluator"], "evaluation.evaluator"),
        version=_text(m["version"], "evaluation.version"),
        events=_integer(m["events"], "evaluation.events", minimum=1) if "events" in m else None,
        seed=_integer(m["seed"], "evaluation.seed") if "seed" in m else None,
        settings=settings,
        backend=_backend(m["backend"]) if "backend" in m else {},
    )


def _backend(raw: object) -> dict[str, tuple[str, ...]]:
    backends = _mapping(raw, "evaluation.backend", _BACKENDS)
    result: dict[str, tuple[str, ...]] = {}
    for name, entry in backends.items():
        path = f"evaluation.backend.{name}"
        args = _sequence(_mapping(entry, path, ("args",), ("args",))["args"], f"{path}.args")
        for index, arg in enumerate(args):
            if not isinstance(arg, str):
                raise _fail(
                    f"{path}.args[{index}]",
                    f"must be a string, got {type(arg).__name__} {_show(arg)}; quote it",
                )
        result[name] = tuple(args)
    return result


def _objectives(raw: object) -> tuple[Objective, ...]:
    items = _sequence(raw, "objectives")
    if not items:
        raise _fail("objectives", "must list at least one objective")
    objectives: list[Objective] = []
    first: dict[tuple[str, str], int] = {}
    for index, item in enumerate(items):
        path = f"objectives[{index}]"
        m = _mapping(item, path, ("stage", "metric", "direction"), ("stage", "metric", "direction"))
        objective = Objective(
            stage=_text(m["stage"], f"{path}.stage"),
            metric=_text(m["metric"], f"{path}.metric"),
            direction=cast(Direction, _choice(m["direction"], f"{path}.direction", _DIRECTIONS)),
        )
        key = (objective.stage, objective.metric)
        if key in first:
            raise _fail(path, f"duplicates objectives[{first[key]}] (stage/metric {key})")
        first[key] = index
        objectives.append(objective)
    return tuple(objectives)


def _constraints(raw: object) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []
    first: dict[tuple[str, str], int] = {}
    for index, item in enumerate(_sequence(raw, "constraints")):
        path = f"constraints[{index}]"
        m = _mapping(item, path, ("stage", "metric", "min", "max", "unit"), ("stage", "metric"))
        if "min" not in m and "max" not in m:
            raise _fail(path, "needs 'min' and/or 'max'")
        low = _number(m["min"], f"{path}.min") if "min" in m else None
        high = _number(m["max"], f"{path}.max") if "max" in m else None
        if low is not None and high is not None and low > high:
            raise _fail(path, f"min must not exceed max, got min={low}, max={high}")
        constraint = Constraint(
            stage=_text(m["stage"], f"{path}.stage"),
            metric=_text(m["metric"], f"{path}.metric"),
            min=low,
            max=high,
            unit=_optional_text(m, "unit", path),
        )
        key = (constraint.stage, constraint.metric)
        if key in first:
            raise _fail(path, f"duplicates constraints[{first[key]}] (stage/metric {key})")
        first[key] = index
        constraints.append(constraint)
    return tuple(constraints)
