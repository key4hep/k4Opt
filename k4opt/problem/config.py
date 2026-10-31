"""Loading and validation of the problem YAML document (``k4opt.problem/v1alpha1``).

:func:`parse_problem` makes the checks that need no Key4hep: shape, types, ranges, syntactic
duplicates, the executors and options of each sample's stages and whether they fit together and
fit the evaluators that measure the sample. :func:`load_problem` adds the resolution of the
geometry targets; that an evaluator or metric is defined, or that the bounds are physically
sensible, is deliberately not decided. See ``docs/problem.md``.
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
from k4opt.problem._tasks import RESERVED_NAMES
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
    Reference,
    Sample,
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
_STAGE_KEYS = ("executor", "file", "events", "seed")  # the rest of a stage are executor options
_SCHEMA_FAMILY = "k4opt.problem/"
_EVALUATOR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# lowercase, so that no two names differ only in case (run directories are named after them)
_SAMPLE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


def load_problem(path: str | os.PathLike[str]) -> ProblemSpec:
    """Read the UTF-8 YAML file ``path``, validate it and resolve its geometry parameters.

    The geometry is located in the active k4geo tree (``$K4GEO`` for a relative path) and
    discovered once; every ``parameters[*].geometry`` must then select exactly one attribute
    of it. Raises :class:`ProblemConfigError` for an invalid document or an unresolvable
    target and :class:`~k4opt.geometry.GeometryError` when the geometry itself cannot be
    located or read. The result keeps the discovered geometry for later use.
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
    source = GeometrySource(spec.geometry)
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
    geometry = _text(top["geometry"], "geometry")
    parameters = _parameters(top["parameters"])
    samples = _samples(top["samples"]) if "samples" in top else {}
    evaluators = _evaluators(top["evaluators"], list(samples))
    objectives = _objectives(top["objectives"], evaluators)
    constraints = _constraints(top["constraints"], evaluators) if "constraints" in top else ()
    _check_pipeline(samples, evaluators, parameters)
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


def _samples(raw: object) -> dict[str, Sample]:
    """The ``samples`` mapping in the document's order, each sample's stages in pipeline order
    whatever the order of the document."""
    m = _mapping(raw, "samples", tuple(raw) if isinstance(raw, Mapping) else ())
    if not m:
        raise _fail("samples", "must name at least one sample")
    samples: dict[str, Sample] = {}
    for name, value in m.items():
        if not _SAMPLE_NAME.fullmatch(name):
            raise _fail(
                "samples",
                f"sample name {_show(name)} must start with a lowercase letter and hold only "
                "lowercase letters, digits and '_', at most 64 characters",
            )
        where = f"samples.{name}"
        stages = _mapping(value, where, (*STAGE_TYPES, "properties"))
        if not set(stages) - {"properties"}:
            raise _fail(where, f"must define its stages: {', '.join(STAGE_TYPES)}")
        made = {
            stage: _stage(where, stage, stages[stage]) for stage in STAGE_TYPES if stage in stages
        }
        properties: Mapping[str, Json] = {}
        if "properties" in stages:
            at, raw = f"{where}.properties", stages["properties"]
            # any JSON, which evaluators read and k4opt does not: a mapping with string keys
            frozen = freeze(_mapping(raw, at, tuple(raw) if isinstance(raw, Mapping) else ()), at)
            assert isinstance(frozen, Mapping)
            properties = frozen
        samples[name] = Sample(name, made, properties)
    return samples


def _stage(where: str, name: str, raw: object) -> Stage:
    """The stage ``name`` of the sample at ``where`` (``samples.gamma``)."""
    path = f"{where}.{name}"
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
                    f"{executor} runs inside {where}.{spec.inside}, which sets the {key}; set "
                    f"{where}.{spec.inside}.{key} instead",
                )
    options = freeze({key: value for key, value in m.items() if key not in _STAGE_KEYS}, path)
    assert isinstance(options, Mapping)
    try:
        spec.options(options)
    except DdsimError as exc:  # its message starts with the stage: generation.particle: ...
        raise ProblemConfigError(f"{where}.{exc}") from exc
    return Stage(name, executor=executor, events=events, seed=seed, options=options)


def _name(value: object, path: str) -> str:
    """An evaluator's ``name`` or ``as``: letters, digits, ``.``, ``_`` and ``-``."""
    name = _text(value, path)
    if not _EVALUATOR_NAME.fullmatch(name):
        raise _fail(
            path,
            f"{_show(name)} must start with a letter or digit and hold only letters, digits, '.', "
            "'_' and '-'",
        )
    return name


def _evaluators(raw: object, samples: list[str]) -> tuple[EvaluatorSpec, ...]:
    """The ``evaluators``, each measuring the ``samples`` it lists (or the problem's only sample)
    if it reads a stage, the candidate otherwise.

    A binding may be named by several entries if each binds the same implementation, version and
    stage on other samples (with other settings, say); otherwise it names one entry."""
    items = _sequence(raw, "evaluators")
    if not items:
        raise _fail("evaluators", "must list at least one evaluator")
    evaluators: list[EvaluatorSpec] = []
    bound: dict[str, list[int]] = {}  # the entries of each binding, by binding case folded
    for index, item in enumerate(items):
        path = f"evaluators[{index}]"
        m = _mapping(
            item,
            path,
            (
                "name",
                "as",
                "version",
                "stage",
                "samples",
                "collections",
                "settings",
                "roles",
                "required",
            ),
            ("name", "version"),
        )
        if "roles" in m:
            others = [key for key in ("stage", "samples", "collections") if key in m]
            if others:
                raise _fail(
                    path,
                    f"a combined evaluator (roles:) reads no stage of its own, so it takes no "
                    f"{', '.join(others)}; its roles name what it reads",
                )
        name = _name(m["name"], f"{path}.name")
        # the binding names the evaluator's task, its directory in a run and its metrics
        where = f"{path}.as" if "as" in m else f"{path}.name"
        binding = _name(m["as"], where) if "as" in m else name
        # records name the work of a candidate, its geometry, stages and evaluators, by name alone
        if binding in RESERVED_NAMES:
            reserved = ", ".join(RESERVED_NAMES)
            raise _fail(where, f"{binding!r} names a step k4opt runs itself; reserved: {reserved}")
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
        measured = _measured(m, path, stage, samples)
        roles = _roles(m["roles"], f"{path}.roles") if "roles" in m else {}
        required = _required(m["required"], f"{path}.required") if "required" in m else False
        entry = EvaluatorSpec(
            name, version, settings, stage, collections, binding, measured, roles, required
        )
        folded = binding.casefold()  # run directories may be on a file system that folds case
        for earlier in bound.get(folded, []):
            _check_repeated(entry, evaluators[earlier], f"evaluators[{earlier}]", path)
        bound.setdefault(folded, []).append(index)
        evaluators.append(entry)
    for index, entry in enumerate(evaluators):  # once all are known: a role may read a later one
        _check_roles(entry, f"evaluators[{index}].roles", evaluators, samples)
    return tuple(evaluators)


def _required(value: object, path: str) -> bool:
    """``required: true``, the one value it takes: an evaluator is required already when an
    objective or a constraint, or a required evaluator, reads it, and a diagnostic otherwise."""
    if value is not True:
        raise _fail(
            path,
            f"must be true, got {_show(value)}; an evaluator that no objective, constraint or "
            "required evaluator reads is a diagnostic without it",
        )
    return True


_REFERENCE_KEYS = ("sample", "samples", "evaluator", "stage")


def _roles(raw: object, path: str) -> dict[str, tuple[Reference, ...]]:
    """The items each role of a combined evaluator reads, by role: each role is given one
    reference, ``{sample, evaluator}`` (an evaluation) or ``{sample, stage}`` (a stage's file),
    with ``samples:`` for one such item per sample, or a list of references."""
    m = _mapping(raw, path, tuple(raw) if isinstance(raw, Mapping) else ())
    if not m:
        raise _fail(
            path, "must name at least one role; an evaluator without roles measures the candidate"
        )
    roles: dict[str, tuple[Reference, ...]] = {}
    for role, value in m.items():
        where = f"{path}.{role}"
        _text(role, path)
        given = value if isinstance(value, (list, tuple)) else [value]
        if not given:
            raise _fail(where, "must reference at least one evaluation or stage")
        items: list[Reference] = []
        for index, reference in enumerate(given):
            at = f"{where}[{index}]" if isinstance(value, (list, tuple)) else where
            for item in _references(reference, at):
                if item in items:
                    raise _fail(
                        at,
                        f"{item.label} is given twice in this role, so the evaluator would read "
                        "it twice; give it once",
                    )
                items.append(item)
        roles[role] = tuple(items)
    return roles


def _references(raw: object, path: str) -> list[Reference]:
    """The items one reference of a role stands for: one per sample it names."""
    m = _mapping(raw, path, _REFERENCE_KEYS)
    if ("sample" in m) == ("samples" in m):
        raise _fail(path, "needs exactly one of 'sample' and 'samples'")
    if ("evaluator" in m) == ("stage" in m):
        raise _fail(
            path, "needs exactly one of 'evaluator' (its evaluation) and 'stage' (its file)"
        )
    if "sample" in m:
        samples = [_text(m["sample"], f"{path}.sample")]
    else:
        listed = _sequence(m["samples"], f"{path}.samples")
        if not listed:
            raise _fail(f"{path}.samples", "must list at least one sample")
        samples = [_text(s, f"{path}.samples[{i}]") for i, s in enumerate(listed)]
    if "evaluator" in m:
        evaluator = _text(m["evaluator"], f"{path}.evaluator")
        return [Reference(sample, evaluator=evaluator) for sample in samples]
    stage = _choice(m["stage"], f"{path}.stage", STAGE_TYPES)
    return [Reference(sample, stage=stage) for sample in samples]


def _check_roles(
    entry: EvaluatorSpec, path: str, evaluators: list[EvaluatorSpec], samples: list[str]
) -> None:
    """Each item a role of ``entry`` reads is of a sample of the problem, and an evaluation of it
    by a binding that measures it; the stage of a stage item is checked with the samples
    (:func:`_check_pipeline`)."""
    measures: dict[str, list[str]] = {}
    for other in evaluators:
        measures.setdefault(other.binding, []).extend(other.samples)
    for role, items in entry.roles.items():
        for item in items:
            where = f"{path}.{role}"
            if item.sample not in samples:
                known = f"its samples are {', '.join(samples)}" if samples else "it has none"
                raise _fail(where, f"the problem has no sample {item.sample!r}; {known}")
            if item.evaluator is None:
                continue
            if item.evaluator not in measures:
                raise _fail(
                    where,
                    f"the problem has no evaluator {item.evaluator!r}; its evaluators are "
                    f"{', '.join(measures)}",
                )
            measured = measures[item.evaluator]
            if not measured:
                raise _fail(
                    where,
                    f"{item.evaluator!r} measures the candidate, not a sample; chained combined "
                    "evaluators are not supported yet",
                )
            if item.sample not in measured:
                raise _fail(
                    where,
                    f"evaluator {item.evaluator!r} does not measure {item.sample!r}; it measures "
                    f"{', '.join(measured)}",
                )


def _measured(
    m: Mapping[str, Any], path: str, stage: str | None, samples: list[str]
) -> tuple[str, ...]:
    """The samples an evaluator entry measures: those it lists, the problem's only sample if it
    lists none, and none if it reads no stage (it measures the candidate)."""
    if "samples" in m:
        where = f"{path}.samples"
        if stage is None:
            raise _fail(
                where,
                "an evaluator measures samples at the stage it reads; add that stage, or remove "
                "samples to measure the candidate",
            )
        items = _sequence(m["samples"], where)
        if not items:
            raise _fail(where, "must list at least one sample")
        for index, item in enumerate(items):
            at = f"{where}[{index}]"
            sample = _text(item, at)
            if sample not in samples:
                known = f"its samples are {', '.join(samples)}" if samples else "it has none"
                raise _fail(at, f"the problem has no sample {sample!r}; {known}")
            if sample in items[:index]:
                raise _fail(at, f"{sample!r} is listed twice")
        return tuple(items)
    if stage is None:
        return ()
    if len(samples) == 1:
        return (samples[0],)
    if not samples:
        raise _fail(path, f"it reads the {stage} stage of a sample, but the problem has no samples")
    raise _fail(
        path,
        f"the problem has the samples {', '.join(samples)}; add samples: with those this "
        "evaluator measures",
    )


def _check_repeated(entry: EvaluatorSpec, other: EvaluatorSpec, at: str, path: str) -> None:
    """``entry`` may share its binding name with the earlier entry ``other`` (at ``at``) only to
    bind the same implementation, version and stage on other samples."""
    binding = entry.binding
    if other.binding != binding:
        why = (
            f"{binding!r} differs only in case from {other.binding!r} of {at}, which run "
            "directories may not tell apart"
        )
    elif entry.stage is None or other.stage is None:
        why = f"{binding!r} is already bound by {at}"
    elif shared := [sample for sample in entry.samples if sample in other.samples]:
        why = f"{binding!r} is already bound to {', '.join(shared)} by {at}"
    elif (entry.name, entry.version, entry.stage) != (other.name, other.version, other.stage):
        why = (
            f"{binding!r} binds {other.name} version {other.version!r} at the {other.stage} stage "
            f"in {at}, and one binding name binds one implementation, version and stage"
        )
    else:
        return
    raise _fail(path, f"{why}; name this one with as:")


def _check_pipeline(
    samples: Mapping[str, Sample],
    evaluators: tuple[EvaluatorSpec, ...],
    parameters: tuple[Parameter, ...],
) -> None:
    """Each sample's stages fit together and fit the evaluators that measure it.

    Every stage an evaluator reads is defined in each sample it measures; a stage that runs has
    the stages it reads; no file stands in for a stage the parameters change (it was not made
    from the candidate); every sample is measured by an evaluator, and every stage of a sample
    is used, by an evaluator that measures it or by a stage that runs.
    """
    for index, evaluator in enumerate(evaluators):
        if evaluator.collections and evaluator.stage is None:
            raise _fail(
                f"evaluators[{index}].collections",
                "collections are read from the file of the stage the evaluator reads; add its "
                "stage",
            )
        for sample in evaluator.samples:
            if evaluator.stage not in samples[sample].stages:
                raise _fail(
                    f"evaluators[{index}].stage",
                    f"sample {sample} defines no {evaluator.stage} stage; add "
                    f"samples.{sample}.{evaluator.stage}",
                )
    read: dict[str, list[str]] = {name: [] for name in samples}  # the stages roles read, by sample
    for index, evaluator in enumerate(evaluators):
        for role, items in evaluator.roles.items():
            for item in items:
                if item.stage is None:
                    continue
                stages = samples[item.sample].stages
                where = f"evaluators[{index}].roles.{role}"
                if item.stage not in stages:
                    raise _fail(
                        where,
                        f"sample {item.sample} defines no {item.stage} stage; add "
                        f"samples.{item.sample}.{item.stage}",
                    )
                executor = stages[item.stage].executor
                inside = None if executor is None else executors(item.stage)[executor].inside
                if inside is not None:
                    raise _fail(
                        where,
                        f"{item.label} is fired inside samples.{item.sample}.{inside} and makes "
                        f"no file of its own; read samples.{item.sample}.{inside} instead",
                    )
                read[item.sample].append(item.stage)
    for spec in samples.values():
        measuring = [e for e in evaluators if spec.name in e.samples]
        _check_sample(spec, measuring, read[spec.name], parameters)


def _check_sample(
    sample: Sample,
    evaluators: list[EvaluatorSpec],
    read: list[str],
    parameters: tuple[Parameter, ...],
) -> None:
    """The stages of one sample fit together and fit the ``evaluators`` that measure it and the
    stages of it that roles ``read``."""
    where, stages = f"samples.{sample.name}", sample.stages
    for name, stage in stages.items():
        if stage.executor is None:
            continue
        for need in stage_reads(name):
            if need not in stages:
                choices = " or ".join(executors(need))
                raise _fail(
                    f"{where}.{name}",
                    f"{name} reads {need}: add {where}.{need} with executor: {choices}, or give "
                    "its output as file:",
                )
    if parameters:  # every parameter varies the geometry
        for name, stage in stages.items():
            if stage.file is not None and changed_by_geometry(name):
                raise _fail(
                    f"{where}.{name}.file",
                    "the parameters vary the geometry, but this file was not made with a "
                    "candidate's geometry, so every candidate would measure the same events; let "
                    f"an executor run the stage (to try an evaluator on an existing file, use "
                    f"k4opt evaluate --input {sample.name}/{name}=FILE)",
                )
    if not evaluators and not read:
        raise _fail(
            where,
            "is measured by no evaluator; list it in the samples of one, have a role read one of "
            "its stages, or remove it",
        )
    used: set[str] = set()

    def use(name: str) -> None:
        if name not in used:
            used.add(name)
            if stages[name].executor is not None:
                for need in stage_reads(name):
                    use(need)

    for evaluator in evaluators:
        assert evaluator.stage is not None  # an evaluator without a stage measures no sample
        use(evaluator.stage)
    for name in read:
        use(name)
    for name in stages:
        if name not in used:
            raise _fail(
                f"{where}.{name}",
                "is not used: no evaluator of the sample reads it, no role, and no stage that runs",
            )


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
