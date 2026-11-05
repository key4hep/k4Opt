"""Loading and validation of the problem YAML document (``k4opt.problem/v1alpha1``).

:func:`parse_problem` makes the checks that need no Key4hep, in two phases: each section on its
own (shape, types, ranges, syntactic duplicates), then across sections: where each parameter
sends its value, the executors and options of each sample's stages (its own, or the shared
``stages`` it takes or completes) given the options parameters set, and whether the stages fit
together and fit the evaluators that measure the sample. :func:`load_problem` adds the
resolution of the geometry destinations; that an evaluator or metric is defined, or that the
bounds are physically sensible, is deliberately not decided. See ``docs/problem.md``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from k4opt.geometry import AmbiguousTargetError, GeometrySource, TargetError
from k4opt.geometry import Parameter as GeometryParameter
from k4opt.problem._evaluators import _evaluators
from k4opt.problem._fields import (
    _choice,
    _fail,
    _integer,
    _mapping,
    _number,
    _optional_text,
    _seconds,
    _sequence,
    _show,
    _text,
    _title,
)
from k4opt.problem._parameters import _destinations, _parameters
from k4opt.problem._samples import _check_options, _check_pipeline, _samples, _shared_stages
from k4opt.problem.errors import ProblemConfigError
from k4opt.problem.models import (
    SCHEMA_VERSION,
    Channel,
    Constraint,
    Direction,
    Environment,
    EvaluatorSpec,
    Key4hepStack,
    Level,
    MetricRef,
    Objective,
    Parameter,
    ProblemSpec,
    ResolvedProblem,
    Sample,
    fixed_events,
)
from k4opt.yamlio import UniqueKeyLoader

_TOP_KEYS = (
    "schema_version",
    "name",
    "environment",
    "timeout",
    "geometry",
    "parameters",
    "stages",
    "samples",
    "evaluators",
    "objectives",
    "constraints",
    "optimization",
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
_DIRECTIONS = ("minimize", "maximize")
_SCHEMA_FAMILY = "k4opt.problem/"


def load_problem(path: str | os.PathLike[str]) -> ProblemSpec:
    """Read the UTF-8 YAML file ``path``, validate it and resolve its geometry destinations.

    The geometry is located in the active k4geo tree (``$K4GEO`` for a relative path) and discovered
    once; every ``geometry`` destination of a parameter must then select exactly one attribute of
    it, and no two the same. Raises :class:`ProblemConfigError` for an invalid document or an
    unresolvable target and :class:`~k4opt.geometry.GeometryError` when the geometry itself cannot
    be located or read."""
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
    """``spec`` with its geometry discovered once and every geometry destination resolved, each
    to an attribute of its own."""
    source = GeometrySource(spec.geometry)
    catalog = source.discover()
    targets: list[GeometryParameter] = []
    first: dict[tuple[str, str, str], str] = {}  # each attribute: the parameter that set it first
    for parameter, destination in spec.geometry_destinations:
        reference = destination.canonical_target
        where = f"{problem_path}: parameters.{parameter.name}.targets.geometry: {reference!r}"
        try:
            target = catalog.lookup(reference)
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
        attribute = (target.file, target.locator, target.attribute)
        if attribute in first:
            by = first[attribute]
            whom = "this parameter" if by == parameter.name else f"parameters.{by}"
            raise ProblemConfigError(
                f"{where} selects attribute {target.attribute!r} of {target.locator} in "
                f"{target.file}, which {whom} sets already"
            )
        first[attribute] = parameter.name
        targets.append(target)
    return ResolvedProblem(spec, problem_path.absolute(), source, catalog, tuple(targets))


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
    # each section on its own first, then the parameters' destinations against the samples and
    # evaluators, and the stages' options with what the parameters set
    declared = _parameters(top["parameters"])
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
    parameters = _destinations(declared, samples, places, evaluators)
    _check_options(samples, places, parameters)
    _check_pipeline(samples, places, evaluators, parameters)
    optimization = (
        _optimization(top["optimization"], parameters, objectives, evaluators, samples)
        if "optimization" in top
        else None
    )
    return ProblemSpec(
        name=name,
        environment=environment,
        geometry=geometry,
        parameters=parameters,
        samples=samples,
        evaluators=evaluators,
        objectives=objectives,
        constraints=constraints,
        timeout=_seconds(top["timeout"], "timeout") if "timeout" in top else None,
        optimization=optimization,
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
        objective = _objective(item, path, evaluators)
        if objective.ref in first:
            raise _fail(path, f"duplicates objectives[{first[objective.ref]}] ({objective.label})")
        first[objective.ref] = index
        objectives.append(objective)
    return tuple(objectives)


def _objective(item: object, path: str, evaluators: tuple[EvaluatorSpec, ...]) -> Objective:
    """One objective: a metric of an evaluation of the problem, and its direction."""
    m = _mapping(
        item,
        path,
        ("sample", "evaluator", "stage", "metric", "title", "direction"),
        ("stage", "metric", "direction"),
    )
    sample, evaluator = _owner(m, path, evaluators)
    return Objective(
        evaluator=evaluator,
        stage=_text(m["stage"], f"{path}.stage"),
        metric=_text(m["metric"], f"{path}.metric"),
        direction=cast(Direction, _choice(m["direction"], f"{path}.direction", _DIRECTIONS)),
        sample=sample,
        title=_title(m, path),
    )


def _optimization(
    raw: object,
    parameters: tuple[Parameter, ...],
    objectives: tuple[Objective, ...],
    evaluators: tuple[EvaluatorSpec, ...],
    samples: Mapping[str, Sample],
) -> Level | None:
    """The levels of a nested optimization, the top one first; ``None`` for a block without
    ``inner``, which is an ordinary optimization, as without the block. Every parameter is in
    the ``parameters`` of exactly one level; the best trial of each inner study is evaluated
    again on new events, so the samples must make theirs from seeds."""
    top = _level(raw, "optimization", parameters, objectives, evaluators)
    seen: dict[str, str] = {}
    level: Level | None = top
    path = "optimization"
    while level is not None:
        for name in level.parameters:
            if name in seen:
                raise _fail(
                    f"{path}.parameters",
                    f"{name!r} is in {seen[name]}.parameters too; each parameter is searched at "
                    "one level",
                )
            seen[name] = path
        level, path = level.inner, f"{path}.inner"
    missing = [parameter.name for parameter in parameters if parameter.name not in seen]
    if missing:
        raise _fail(
            "optimization",
            f"the parameter(s) {', '.join(missing)} are in no level; list each parameter in the "
            "parameters of the level that searches it",
        )
    if top.inner is None:
        return None
    reason = fixed_events(samples)
    if reason is not None:
        raise _fail(
            "optimization.inner",
            f"the best trial of each inner study is evaluated again on new events, but {reason}",
        )
    return top


def _level(
    raw: object,
    path: str,
    parameters: tuple[Parameter, ...],
    objectives: tuple[Objective, ...],
    evaluators: tuple[EvaluatorSpec, ...],
) -> Level:
    """One level of ``optimization`` at ``path``, and those below it."""
    m = _mapping(raw, path, ("parameters", "trials", "objective", "inner"), ("parameters",))
    top = path == "optimization"
    if top and "trials" in m:
        raise _fail(
            f"{path}.trials",
            "the top level's trials are how long a run goes on; give them on the command line "
            "(k4opt optimize --trials N), where a resume can add more",
        )
    if top and "objective" in m:
        raise _fail(
            f"{path}.objective",
            "the problem's objectives judge the top level; objective: says how an inner level "
            "picks the best trial of each of its studies",
        )
    if not top and "trials" not in m:
        raise _fail(
            path,
            "missing trials: how many trials each study of this level runs before its best is "
            "chosen",
        )
    names = [parameter.name for parameter in parameters]
    listed = _sequence(m["parameters"], f"{path}.parameters")
    if not listed:
        raise _fail(f"{path}.parameters", "must list at least one parameter")
    for index, name in enumerate(listed):
        where = f"{path}.parameters[{index}]"
        if _text(name, where) not in names:
            raise _fail(
                where,
                f"{name!r} is no parameter of the problem; its parameters are {', '.join(names)}",
            )
        if name in listed[:index]:
            raise _fail(where, f"{name!r} is listed twice")
    return Level(
        tuple(listed),
        None if top else _integer(m["trials"], f"{path}.trials", minimum=1),
        None if top else _level_objective(m.get("objective"), path, objectives, evaluators),
        _level(m["inner"], f"{path}.inner", parameters, objectives, evaluators)
        if "inner" in m
        else None,
    )


def _level_objective(
    raw: object, path: str, objectives: tuple[Objective, ...], evaluators: tuple[EvaluatorSpec, ...]
) -> Objective:
    """What picks the best trial of each study of the level at ``path``: one of the problem's
    objectives, by label or metric, or any metric of an evaluation, with its direction; the
    problem's objective if it has one and none is named."""
    labels = ", ".join(objective.label for objective in objectives)
    if raw is None:
        if len(objectives) == 1:
            return objectives[0]
        raise _fail(
            path,
            f"the problem has the objectives {labels}; say which one picks the best trial of "
            "each study of this level (objective: METRIC or LABEL), or give one of any metric "
            "(objective: {evaluator, sample, stage, metric, direction})",
        )
    if not isinstance(raw, str):
        return _objective(raw, f"{path}.objective", evaluators)
    found = [o for o in objectives if o.label == raw] or [o for o in objectives if o.metric == raw]
    if len(found) == 1:
        return found[0]
    if found:
        raise _fail(
            f"{path}.objective",
            f"{raw!r} is the metric of the objectives {', '.join(o.label for o in found)}; name "
            "one by its label",
        )
    raise _fail(
        f"{path}.objective",
        f"{raw!r} is no objective of the problem ({labels}); name one by its metric or label, or "
        "give one of any metric (objective: {evaluator, sample, stage, metric, direction})",
    )


def _constraints(raw: object, evaluators: tuple[EvaluatorSpec, ...]) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []
    first: dict[MetricRef, int] = {}
    for index, item in enumerate(_sequence(raw, "constraints")):
        path = f"constraints[{index}]"
        m = _mapping(
            item,
            path,
            ("sample", "evaluator", "stage", "metric", "title", "min", "max", "unit", "scale"),
            ("stage", "metric"),
        )
        if "min" not in m and "max" not in m:
            raise _fail(path, "needs 'min' and/or 'max'")
        low = _number(m["min"], f"{path}.min") if "min" in m else None
        high = _number(m["max"], f"{path}.max") if "max" in m else None
        if low is not None and high is not None and low > high:
            raise _fail(path, f"min must not exceed max, got min={low}, max={high}")
        scale = _number(m["scale"], f"{path}.scale") if "scale" in m else None
        if scale is not None and not scale > 0:
            raise _fail(
                f"{path}.scale",
                f"must be positive: a characteristic size of the metric, got {scale!r}",
            )
        sample, evaluator = _owner(m, path, evaluators)
        constraint = Constraint(
            evaluator=evaluator,
            stage=_text(m["stage"], f"{path}.stage"),
            metric=_text(m["metric"], f"{path}.metric"),
            min=low,
            max=high,
            unit=_optional_text(m, "unit", path),
            sample=sample,
            scale=scale,
            title=_title(m, path),
        )
        if constraint.ref in first:
            raise _fail(
                path, f"duplicates constraints[{first[constraint.ref]}] ({constraint.label})"
            )
        first[constraint.ref] = index
        constraints.append(constraint)
    unscaled = [f"constraints[{i}]" for i, c in enumerate(constraints) if c.scale is None]
    if len(constraints) > 1 and unscaled:
        raise _fail(
            ", ".join(unscaled),
            "needs a scale: with constraints on several metrics, an optimizer adds how far a "
            "candidate is beyond each limit, which means something only in units of a "
            "characteristic size of each metric: give each a scale, in the unit of its limits",
        )
    return tuple(constraints)
