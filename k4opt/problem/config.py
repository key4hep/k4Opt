"""Loading and validation of the problem YAML document (``k4opt.problem/v1alpha1``).

:func:`parse_problem` makes the checks that need no Key4hep: shape, types, ranges, syntactic
duplicates, the executors and options of the stages and whether the stages fit together and
fit the evaluators. :func:`load_problem` adds the resolution of the geometry targets; that an
evaluator or metric is defined, or that the bounds are physically sensible, is deliberately not
decided. See ``docs/problem.md``.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

from k4opt.ddsim import DdsimError
from k4opt.frozen import Json
from k4opt.geometry import AmbiguousTargetError, GeometrySource, TargetError
from k4opt.problem._stages import STAGE_TYPES, changed_by_geometry, executors, stage_reads
from k4opt.problem.errors import ProblemConfigError
from k4opt.problem.models import (
    SCHEMA_VERSION,
    Channel,
    Constraint,
    Direction,
    Environment,
    EvaluatorSpec,
    GeometrySelector,
    GeometrySpec,
    Key4hepStack,
    Number,
    Objective,
    Parameter,
    ParameterKind,
    ProblemSpec,
    Stage,
    freeze,
)
from k4opt.yamlio import UniqueKeyLoader

_TOP_KEYS = (
    "schema_version",
    "name",
    "environment",
    "geometry",
    "parameters",
    "stages",
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
_STAGE_KEYS = ("executor", "file", "events", "seed")  # the rest of a stage are executor options
_SCHEMA_FAMILY = "k4opt.problem/"
_EVALUATOR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def load_problem(path: str | os.PathLike[str]) -> ProblemSpec:
    """Read the UTF-8 YAML file ``path``, validate it and resolve its geometry parameters.

    The geometry entry is located in the active k4geo tree (``$K4GEO`` for a relative
    entry) and discovered once; every ``parameters[*].geometry`` must then select exactly
    one attribute of it. Raises :class:`ProblemConfigError` for an invalid document or an
    unresolvable target and :class:`~k4opt.geometry.GeometryError` when the geometry itself
    cannot be located or read. The result keeps the discovered geometry for later use.
    """
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


def _resolve_geometry(spec: ProblemSpec, problem_path: Path) -> ProblemSpec:
    """Return ``spec`` with its geometry discovered once and every parameter target resolved."""
    source = GeometrySource(spec.geometry.entry)
    catalog = source.discover()
    targets = []
    for index, parameter in enumerate(spec.parameters):
        reference = parameter.geometry.canonical_target
        where = f"{problem_path}: parameters[{index}].geometry: target {reference!r}"
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
    return replace(
        spec,
        _source=source,
        _catalog=catalog,
        _targets=tuple(targets),
        _directory=problem_path.parent.absolute(),
    )


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
    if isinstance(value, float) and not math.isfinite(value):  # an int is always finite
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
    environment = _environment(top["environment"])
    geometry = GeometrySpec(
        entry=_text(
            _mapping(top["geometry"], "geometry", ("entry",), ("entry",))["entry"],
            "geometry.entry",
        )
    )
    parameters = _parameters(top["parameters"])
    stages = _stages(top["stages"]) if "stages" in top else {}
    evaluators = _evaluators(top["evaluators"])
    objectives = _objectives(top["objectives"])
    constraints = _constraints(top["constraints"]) if "constraints" in top else ()
    _check_pipeline(stages, evaluators, parameters)
    return ProblemSpec(
        name=name,
        environment=environment,
        geometry=geometry,
        parameters=parameters,
        stages=stages,
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


def _stages(raw: object) -> dict[str, Stage]:
    """The ``stages`` mapping, in pipeline order whatever the order of the document."""
    m = _mapping(raw, "stages", STAGE_TYPES)
    return {name: _stage(name, m[name]) for name in STAGE_TYPES if name in m}


def _stage(name: str, raw: object) -> Stage:
    path = f"stages.{name}"
    # any string keys: which ones the stage takes depends on its executor, checked below
    m = _mapping(raw, path, tuple(raw) if isinstance(raw, Mapping) else ())
    if ("executor" in m) == ("file" in m):
        raise _fail(
            path, "needs exactly one of 'executor' (k4opt runs it) and 'file' (its output exists)"
        )
    if "file" in m:
        others = [key for key in m if key != "file"]
        if others:
            raise _fail(path, f"a stage given as a file is not run, so it takes no {others}")
        return Stage(name, file=_text(m["file"], f"{path}.file"))
    allowed = executors(name)
    executor = _choice(m["executor"], f"{path}.executor", tuple(allowed))
    spec = allowed[executor]
    events = seed = None
    if spec.events:
        if "events" not in m:
            raise _fail(path, f"missing required key 'events': how many events {executor} runs")
        events = _integer(m["events"], f"{path}.events", minimum=1)
        seed = _integer(m["seed"], f"{path}.seed") if "seed" in m else None
    else:
        for key in ("events", "seed"):
            if key in m:
                raise _fail(
                    f"{path}.{key}",
                    f"{executor} runs inside stages.{spec.inside}, which sets the {key}; set "
                    f"stages.{spec.inside}.{key} instead",
                )
    options = freeze({key: value for key, value in m.items() if key not in _STAGE_KEYS}, path)
    assert isinstance(options, Mapping)
    try:
        spec.options(options)
    except DdsimError as exc:  # its message starts with the stage: generation.particle: ...
        raise ProblemConfigError(f"stages.{exc}") from exc
    return Stage(name, executor=executor, events=events, seed=seed, options=options)


def _evaluators(raw: object) -> tuple[EvaluatorSpec, ...]:
    items = _sequence(raw, "evaluators")
    if not items:
        raise _fail("evaluators", "must list at least one evaluator")
    evaluators: list[EvaluatorSpec] = []
    first: dict[str, int] = {}
    for index, item in enumerate(items):
        path = f"evaluators[{index}]"
        m = _mapping(
            item, path, ("name", "version", "stage", "collections", "settings"), ("name", "version")
        )
        name = _text(m["name"], f"{path}.name")
        # the name also names the evaluator's directory in a run
        if not _EVALUATOR_NAME.fullmatch(name):
            raise _fail(
                f"{path}.name",
                f"{_show(name)} must start with a letter or digit and hold only letters, "
                "digits, '.', '_' and '-'",
            )
        if name in first:
            raise _fail(path, f"duplicates evaluators[{first[name]}] (name {name!r})")
        first[name] = index
        settings: Mapping[str, Json] = {}
        if "settings" in m:
            if not isinstance(m["settings"], Mapping):
                raise _fail(
                    f"{path}.settings", f"must be a mapping, got {type(m['settings']).__name__}"
                )
            frozen = freeze(m["settings"], f"{path}.settings")
            assert isinstance(frozen, Mapping)
            settings = frozen
        stage = _choice(m["stage"], f"{path}.stage", STAGE_TYPES) if "stage" in m else None
        collections: dict[str, str] = {}
        if "collections" in m:
            where, raw = f"{path}.collections", m["collections"]
            roles = _mapping(raw, where, tuple(raw) if isinstance(raw, Mapping) else ())
            collections = {role: _text(name, f"{where}.{role}") for role, name in roles.items()}
        version = _text(m["version"], f"{path}.version")
        evaluators.append(EvaluatorSpec(name, version, settings, stage, collections))
    return tuple(evaluators)


def _check_pipeline(
    stages: Mapping[str, Stage],
    evaluators: tuple[EvaluatorSpec, ...],
    parameters: tuple[Parameter, ...],
) -> None:
    """The stages fit together and fit the evaluators.

    Every stage an evaluator reads is defined; a stage that runs has the stages it reads; no
    file stands in for a stage the parameters change (it was not made from the candidate); and
    every stage is used, by an evaluator or by a stage that runs.
    """
    for index, evaluator in enumerate(evaluators):
        if evaluator.collections and evaluator.stage is None:
            raise _fail(
                f"evaluators[{index}].collections",
                "collections are read from the file of the stage the evaluator reads; add its "
                "stage",
            )
        if evaluator.stage is not None and evaluator.stage not in stages:
            raise _fail(
                f"evaluators[{index}].stage",
                f"the problem defines no {evaluator.stage} stage; add stages.{evaluator.stage}",
            )
    for name, stage in stages.items():
        if stage.executor is None:
            continue
        for need in stage_reads(name):
            if need not in stages:
                choices = " or ".join(executors(need))
                raise _fail(
                    f"stages.{name}",
                    f"{name} reads {need}: add stages.{need} with executor: {choices}, or give "
                    "its output as file:",
                )
    if parameters:  # every parameter varies the geometry
        for name, stage in stages.items():
            if stage.file is not None and changed_by_geometry(name):
                raise _fail(
                    f"stages.{name}.file",
                    "the parameters vary the geometry, but this file was not made with a "
                    "candidate's geometry, so every candidate would measure the same events; let "
                    f"an executor run the stage (to try an evaluator on an existing file, use "
                    f"k4opt evaluate --input {name}=FILE)",
                )
    used: set[str] = set()

    def use(name: str) -> None:
        if name not in used:
            used.add(name)
            if stages[name].executor is not None:
                for need in stage_reads(name):
                    use(need)

    for evaluator in evaluators:
        if evaluator.stage is not None:
            use(evaluator.stage)
    for name in stages:
        if name not in used:
            raise _fail(
                f"stages.{name}", "is not used: no evaluator reads it, and no stage that runs does"
            )


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
