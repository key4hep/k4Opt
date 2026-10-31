"""Internal bridge: one candidate of a problem, measured by each of its evaluators.

Not a public workflow. It validates a candidate against the declared parameter domains, binds
the given evaluator implementations to the problem's ``evaluators`` and checks, before anything
runs, their identity, the stage each reads, their settings, the pipeline stages and which
evaluator measures each objective and constraint metric. It then makes every stage the
evaluators read available, running each once (:mod:`k4opt.problem._stages`), calls each
evaluator once with the files of the stages it reads, and checks that each ``valid`` result
carries the metrics its evaluator owns. It neither locates evaluators nor materializes geometry.

An implementation (:data:`~k4opt.evaluation.EvaluatorFactory`, a class say) is bound to its
entry of ``evaluators`` once, through a probe instance that is used for nothing else. Every
candidate gets a new instance of each, whose ``check()`` and ``evaluate()`` it calls, so no
state of an instance reaches another candidate. An entry is identified by its ``name`` today;
records, run directories and metric owners use that identity.
"""

from __future__ import annotations

import difflib
import math
import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple

from k4opt.ddsim import output_collections
from k4opt.evaluation import (
    CandidateKey,
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorFactory,
    EvaluatorProtocol,
    Measurement,
    Scalar,
    Status,
    available_collections,
    check_settings,
    collection_names,
    collection_types,
    declared_collections,
    metric_units,
    result_from,
    with_defaults,
)
from k4opt.frozen import FrozenMap
from k4opt.problem._stages import (
    STAGE_TYPES,
    StageContext,
    StageRecords,
    changed_by_geometry,
    check_stages,
    no_edm4hep_file,
    run_stages,
)
from k4opt.problem.errors import CandidateError
from k4opt.problem.models import EvaluatorSpec, Parameter, ProblemSpec

type MetricKey = tuple[str, str]
"""A metric as objectives, constraints and measurements name it: ``(stage, metric)``."""


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


def _check_number(value: object, reference: str) -> int | float:
    if not _is_number(value):
        raise CandidateError(f"{reference}: expected a number, got {value!r}")
    assert isinstance(value, (int, float))
    if isinstance(value, float) and not math.isfinite(value):
        raise CandidateError(f"{reference}: value must be finite, got {value!r}")
    return value


def _validate_value(parameter: Parameter, value: object) -> tuple[str, str]:
    """Return ``(tag, canonical value)`` of ``value`` for ``parameter`` or raise."""
    reference = parameter.geometry.canonical_target
    if parameter.kind == "continuous":
        number = _check_number(value, reference)
    elif parameter.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CandidateError(f"{reference}: expected an integer, got {value!r}")
        number = value
    else:
        choices = parameter.choices or ()
        if all(isinstance(choice, str) for choice in choices):
            if not isinstance(value, str) or value not in choices:
                raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
            return "str", value
        if all(isinstance(choice, bool) for choice in choices):
            if not isinstance(value, bool) or value not in choices:
                raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
            return "bool", "true" if value else "false"
        number = _check_number(value, reference)
        if not any(number == choice for choice in choices):
            raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
        return "num", _canonical_number(number)
    assert parameter.bounds is not None
    low, high = parameter.bounds
    if not low <= number <= high:
        raise CandidateError(f"{reference}: {value!r} is outside the bounds [{low}, {high}]")
    return ("int" if parameter.kind == "integer" else "num"), _canonical_number(number)


def _validate_candidate(
    spec: ProblemSpec, candidate: Mapping[str, Scalar]
) -> tuple[dict[str, Scalar], CandidateKey]:
    """Check ``candidate`` against every parameter; return it in problem order with its key."""
    if not isinstance(candidate, Mapping):
        raise CandidateError(f"candidate must be a mapping, got {type(candidate).__name__}")
    declared = [parameter.geometry.canonical_target for parameter in spec.parameters]
    bad_keys = [key for key in candidate if not isinstance(key, str)]
    if bad_keys:
        raise CandidateError(f"candidate keys must be strings, got {bad_keys!r}")
    unknown = [key for key in candidate if key not in declared]
    if unknown:
        raise CandidateError(f"candidate has unknown parameter(s): {', '.join(unknown)}")
    missing = [reference for reference in declared if reference not in candidate]
    if missing:
        raise CandidateError(f"candidate is missing parameter(s): {', '.join(missing)}")
    values: dict[str, Scalar] = {}
    key: list[tuple[str, str, str, str]] = []
    for parameter, reference in zip(spec.parameters, declared, strict=True):
        value = candidate[reference]
        tag, canonical = _validate_value(parameter, value)
        values[reference] = value
        key.append((reference, parameter.kind, tag, canonical))
    return values, tuple(key)


def _check_units(evaluator: EvaluatorProtocol, result: EvaluationResult) -> None:
    """Every measurement of a metric the evaluator declares must carry the declared unit.

    Whatever the status: diagnostic values of an ``invalid`` result are recorded too. An
    evaluator that declares no ``metrics``, and metrics it does not declare, are not checked.
    """
    declared = metric_units(evaluator)
    if declared is None:
        return
    for m in result.measurements:
        unit = declared.get((m.stage, m.metric))
        if (m.stage, m.metric) in declared and m.unit != unit:
            raise EvaluationContractError(
                f"measurement {m.stage}/{m.metric} has unit {m.unit!r} but evaluator "
                f"{evaluator.name!r} declares {unit!r}"
            )


def _stage(evaluator: EvaluatorProtocol) -> str | None:
    """The pipeline stage the evaluator reads, ``None`` if it reads none."""
    stage: str | None = getattr(evaluator, "stage", None)
    return stage


def _label(implementation: EvaluatorFactory) -> str:
    """An implementation as an error names it: its class or function."""
    label: str = getattr(implementation, "__qualname__", None) or repr(implementation)
    return label


def _create(implementation: EvaluatorFactory) -> EvaluatorProtocol:
    """A new instance of ``implementation``, which must have ``name``, ``version`` and
    ``evaluate``; :class:`EvaluationContractError` if it cannot be created or lacks one."""
    if not callable(implementation):
        raise EvaluationContractError(
            f"got a {type(implementation).__name__} object: give the evaluator class, of which "
            "k4opt creates a new instance for each candidate"
        )
    label = _label(implementation)
    try:
        evaluator = implementation()
    except Exception as exc:
        raise EvaluationContractError(
            f"creating {label}() failed: {type(exc).__name__}: {exc}"
        ) from exc
    missing = [
        member for member in ("name", "version", "evaluate") if not hasattr(evaluator, member)
    ]
    if missing or not callable(evaluator.evaluate):
        raise EvaluationContractError(
            f"{label} is not an evaluator: needs name, version and an evaluate() method"
            + (f" (missing {', '.join(missing)})" if missing else "")
        )
    return evaluator


class _Declaration(NamedTuple):
    """What every instance of an implementation must say of itself as its probe did: the session
    records the probe's class as the code that runs, and the stage and metrics decided what each
    evaluator reads and measures."""

    cls: type
    name: str
    version: str
    stage: str | None
    metrics: Mapping[MetricKey, str] | None


def _declaration(evaluator: EvaluatorProtocol) -> _Declaration:
    return _Declaration(
        type(evaluator),
        evaluator.name,
        evaluator.version,
        _stage(evaluator),
        metric_units(evaluator),
    )


@dataclass(frozen=True, slots=True)
class _Bound:
    """An evaluator of the problem: its entry, the implementation given for it and what the probe
    instance that bound them declared."""

    entry: EvaluatorSpec
    implementation: EvaluatorFactory
    declaration: _Declaration


def _instance(bound: _Bound) -> EvaluatorProtocol:
    """A new instance of a bound implementation; :class:`EvaluationContractError` unless it
    declares what the probe declared."""
    evaluator = _create(bound.implementation)
    labels = ("class", "name", "version", "stage", "metrics")
    changed = [
        f"{label} {new!r}, not {old!r}"
        for label, new, old in zip(labels, _declaration(evaluator), bound.declaration, strict=True)
        if new != old
    ]
    if changed:
        raise EvaluationContractError(
            f"evaluator {bound.entry.name!r}: a new instance of {_label(bound.implementation)} "
            f"has {'; '.join(changed)}; every instance must declare what the first did"
        )
    return evaluator


def _bind(
    spec: ProblemSpec, implementations: Sequence[EvaluatorFactory]
) -> list[tuple[EvaluatorSpec, EvaluatorFactory, EvaluatorProtocol]]:
    """Pair every evaluator the problem declares with the implementation given for it, in problem
    order, each with the probe instance that decided it.

    A probe of each implementation is matched by its ``name`` and must have the declared
    ``version`` and read the declared ``stage``; an evaluator that is missing, not declared,
    given twice or reads another stage is an :class:`EvaluationContractError`, and so is an
    implementation that cannot be created.
    """
    declared = {entry.name: entry for entry in spec.evaluators}
    given: dict[str, tuple[EvaluatorFactory, EvaluatorProtocol]] = {}
    for implementation in implementations:
        evaluator = _create(implementation)
        name, version = evaluator.name, evaluator.version
        if name in given:
            raise EvaluationContractError(f"evaluator {name!r} is given twice")
        entry = declared.get(name)
        if entry is None:
            raise EvaluationContractError(
                f"problem declares no evaluator {name!r}; declared: {', '.join(declared)}"
            )
        if version != entry.version:
            raise EvaluationContractError(
                f"problem declares evaluator {name!r} version {entry.version!r}, "
                f"got version {version!r}"
            )
        stage = _stage(evaluator)
        if stage != entry.stage:
            if entry.stage is None:
                fix = f"add stage: {stage} to its entry in the problem"
            elif stage is None:
                fix = f"it reads no stage; remove stage: {entry.stage} from its entry"
            else:
                fix = f"it reads {stage!r}, not {entry.stage!r}"
            raise EvaluationContractError(
                f"evaluator {name!r} does not read the stage the problem says: {fix}"
            )
        given[name] = (implementation, evaluator)
    missing = [
        f"{entry.name!r} version {entry.version!r}"
        for entry in spec.evaluators
        if entry.name not in given
    ]
    if missing:
        raise EvaluationContractError(f"no evaluator given for {', '.join(missing)}")
    return [(entry, *given[entry.name]) for entry in spec.evaluators]


def _show(metrics: Collection[MetricKey]) -> str:
    return ", ".join(f"{stage}/{metric}" for stage, metric in metrics)


def _owners(
    spec: ProblemSpec, evaluators: Sequence[tuple[EvaluatorSpec, EvaluatorProtocol]]
) -> tuple[dict[MetricKey, str], dict[MetricKey, str]]:
    """``(owners, declared)``: the evaluator instance that measures each metric an objective or
    a constraint names, and the evaluator that declares each metric any of them declares.

    A single evaluator measures all of them; if it declares ``metrics``, they must be among them.
    Several evaluators must each declare their metrics, and no metric may be declared by two
    implementations. Each objective and constraint metric is then resolved: of the instances
    whose implementation declares it, exactly one must remain (objectives have no filters yet
    that narrow them down; with one instance per implementation, at most one declares it). A
    constraint ``unit`` must be the unit its evaluator declares. Raises
    :class:`EvaluationContractError`.
    """
    wanted = [(o.stage, o.metric) for o in spec.objectives]
    wanted = list(dict.fromkeys(wanted + [(c.stage, c.metric) for c in spec.constraints]))
    units = {entry.name: metric_units(evaluator) for entry, evaluator in evaluators}
    silent = [name for name, metrics in units.items() if metrics is None]
    if len(units) == 1 and silent:
        return dict.fromkeys(wanted, silent[0]), {}
    if silent:
        raise EvaluationContractError(
            f"evaluator(s) {', '.join(map(repr, silent))} declare no metrics; when a problem has "
            "several evaluators, each must declare the metrics it measures"
        )
    declared: dict[MetricKey, str] = {}
    for name, metrics in units.items():
        assert metrics is not None
        for key in metrics:
            if key in declared:
                raise EvaluationContractError(
                    f"metric {_show([key])} is declared by evaluators {declared[key]!r} and "
                    f"{name!r}; a metric must have one owner"
                )
            declared[key] = name
    measured_by = {
        key: [name for name, metrics in units.items() if metrics is not None and key in metrics]
        for key in wanted
    }
    unmeasured = [key for key in wanted if not measured_by[key]]
    if unmeasured:
        if len(units) == 1:
            ((name, metrics),) = units.items()
            assert metrics is not None
            raise EvaluationContractError(
                f"evaluator {name!r} does not measure {_show(unmeasured)}; "
                f"supported: {_show(sorted(metrics))}"
            )
        supported = "; ".join(
            f"{name!r} measures {_show(sorted(metrics or {}))}" for name, metrics in units.items()
        )
        raise EvaluationContractError(f"no evaluator measures {_show(unmeasured)}; {supported}")
    owners = {key: _resolve(key, measured_by[key]) for key in wanted}
    for constraint in spec.constraints:
        key = (constraint.stage, constraint.metric)
        metrics = units[owners[key]]
        assert metrics is not None
        if constraint.unit is not None and constraint.unit != metrics[key]:
            raise EvaluationContractError(
                f"constraint on {_show([key])} declares unit {constraint.unit!r} but evaluator "
                f"{owners[key]!r} measures it in {metrics[key]!r}"
            )
    return owners, declared


def _resolve(key: MetricKey, instances: Sequence[str]) -> str:
    """The one evaluator instance of ``instances`` (those that measure ``key``) left after the
    objective's filters; there are none yet, so exactly one must measure it."""
    if len(instances) != 1:
        raise EvaluationContractError(
            f"metric {_show([key])} is measured by {', '.join(map(repr, instances))}; it must "
            "resolve to one evaluator"
        )
    return instances[0]


def _unexpected_collections(
    spec: ProblemSpec,
    context: StageContext,
    evaluators: Sequence[tuple[EvaluatorSpec, EvaluatorProtocol]],
    provided: Collection[str],
) -> tuple[str, ...]:
    """A warning for each collection an evaluator reads from the simulation that ddsim is not
    expected to write for the problem's geometry, with the closest expected name.

    Only a warning: the prediction comes from the unmodified geometry's readouts, and custom
    sensitive actions can name collections of their own. The simulation's output decides.
    """
    simulation = context.stages.get("simulation")
    if spec._catalog is None or simulation is None or simulation.executor != "ddsim":
        return ()
    if "simulation" in provided:  # a supplied file is checked itself, before it is read
        return ()
    expected = sorted(output_collections(spec._catalog.readouts))
    warnings = []
    for entry, evaluator in evaluators:
        if _stage(evaluator) != "simulation":
            continue
        for role, name in collection_names(evaluator, entry.collections).items():
            if name in expected:
                continue
            close = difflib.get_close_matches(name, expected, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else "."
            warnings.append(
                f"evaluator {entry.name!r}: ddsim is not expected to write the collection "
                f"{name!r} ({role}) for {spec.geometry}{hint} The simulation's output "
                "decides."
            )
    return tuple(warnings)


def _collection_error(
    evaluator: EvaluatorProtocol,
    request: EvaluationRequest,
    seen: dict[Path, tuple[str, ...] | str],
) -> str | None:
    """Why the file of the evaluator's stage lacks a collection it declares, or holds it with
    another type; ``None`` if every one is there. Only the declared collections are read, so one
    the evaluator does not read cannot fail it. ``seen`` keeps the collection names of each file
    (or why they cannot be listed), so a file read by several evaluators is listed once."""
    declared = declared_collections(evaluator)
    stage = _stage(evaluator)
    if not declared or stage is None or stage not in request.inputs:
        return None
    path = request.inputs[stage]
    if path not in seen:
        try:
            seen[path] = available_collections(path)
        except EvaluationFailed as exc:
            seen[path] = str(exc)
    available = seen[path]
    if isinstance(available, str):
        return available
    names = {role: request.collections[role] for role in declared}
    for role, name in names.items():
        if name not in available:
            listing = ", ".join(available) or "no collections"
            return f"collection {name!r} ({role}) is not in {path.name}, which holds: {listing}"
    try:
        types = collection_types(path, names.values())
    except EvaluationFailed as exc:
        return str(exc)
    for role, collection in declared.items():
        name, held = names[role], types.get(names[role])
        if held != collection.podio_type:
            return (
                f"collection {name!r} ({role}) holds {held!r}, but the evaluator reads "
                f"{collection.podio_type!r}"
            )
    return None


@dataclass(frozen=True, slots=True)
class _Binding:
    """What holds for every candidate: the implementations bound to the problem, the stages they
    read and who measures each metric."""

    context: StageContext
    evaluators: tuple[_Bound, ...]  # problem order
    targets: tuple[str, ...]  # the pipeline stages the evaluators read
    owners: Mapping[MetricKey, str]  # objective or constraint metric -> evaluator
    declared: Mapping[MetricKey, str]  # any declared metric -> evaluator
    warnings: tuple[str, ...] = ()  # collections the simulation is not expected to write


@dataclass(frozen=True, slots=True)
class _Plan:
    """What the preflight established before anything runs, with the candidate's own evaluator
    instances."""

    context: StageContext
    evaluators: tuple[tuple[EvaluatorSpec, EvaluatorProtocol], ...]  # problem order
    targets: tuple[str, ...]  # the pipeline stages the evaluators read
    owners: Mapping[MetricKey, str]  # objective or constraint metric -> evaluator
    declared: Mapping[MetricKey, str]  # any declared metric -> evaluator
    requests: Mapping[str, EvaluationRequest]  # normalized and checked before execution
    warnings: tuple[str, ...] = ()  # collections the simulation is not expected to write


def _bind_problem(
    spec: ProblemSpec,
    implementations: Sequence[EvaluatorFactory],
    provided: Mapping[str, Path] | None = None,
) -> _Binding:
    """Check what does not depend on a candidate, before a file is created or a stage runs.

    Binds ``implementations`` to the problem through a probe instance of each, which is then
    dropped: checks the stages they read (those in ``provided``, the files the caller supplies
    by stage, are not run), each evaluator's settings and collections against what it declares,
    and decides which evaluator measures each objective and constraint metric. An evaluator that
    declares collections of a stage without an EDM4hep file (the gun's, or a HepMC file) is
    refused. Raises :class:`EvaluationContractError`. Collections the simulation is not expected
    to write are only warnings.
    """
    provided = provided or {}
    context = StageContext.of(spec)
    probed = _bind(spec, implementations)
    bound = [(entry, probe) for entry, _, probe in probed]
    targets = tuple(stage for _, ev in bound if (stage := _stage(ev)) is not None)
    check_stages(context, *targets, provided=provided)
    for entry, evaluator in bound:
        stage = _stage(evaluator)
        reason = None if stage is None else no_edm4hep_file(context, stage, provided)
        if reason is not None and declared_collections(evaluator):
            raise EvaluationContractError(
                f"evaluator {entry.name!r} reads collections from the {stage!r} stage, but {reason}"
            )
        # what needs no candidate is checked before any: a check() may depend on the candidate
        check_settings(evaluator, with_defaults(evaluator, entry.settings))
        collection_names(evaluator, entry.collections)
    owners, declared = _owners(spec, bound)
    warnings = _unexpected_collections(spec, context, bound, provided)
    evaluators = tuple(
        _Bound(entry, implementation, _declaration(probe))
        for entry, implementation, probe in probed
    )
    return _Binding(context, evaluators, targets, owners, declared, warnings)


def _preflight(binding: _Binding, values: Mapping[str, Scalar]) -> _Plan:
    """Prepare the evaluators of one candidate before a file is created or a stage runs.

    Creates a new instance of each bound implementation, which must declare what its probe
    declared, checks its settings and collections, then calls each one's ``check()`` with the
    request it would get for this candidate (its settings and collections resolved). Raises
    :class:`EvaluationContractError` (or what a ``check()`` raises). The plan holds these
    instances and requests made for these ``values``, so it serves one candidate only.
    """
    evaluators = tuple((bound.entry, _instance(bound)) for bound in binding.evaluators)
    for entry, evaluator in evaluators:  # before any check(), which may depend on the candidate
        check_settings(evaluator, with_defaults(evaluator, entry.settings))
        collection_names(evaluator, entry.collections)
    requests = {}
    for entry, evaluator in evaluators:
        request = _request(binding.context, values, entry, evaluator)
        requests[entry.name] = request
        check = getattr(evaluator, "check", None)
        if check is not None:
            check(request)
    return _Plan(
        binding.context,
        evaluators,
        binding.targets,
        binding.owners,
        binding.declared,
        FrozenMap(requests),
        binding.warnings,
    )


def _check_result(spec: ProblemSpec, plan: _Plan, name: str, result: EvaluationResult) -> None:
    """A result must not report a metric another evaluator declares, and a ``valid`` one must
    report every metric its evaluator ``name`` owns, in the unit a constraint declares."""
    for m in result.measurements:
        owner = plan.declared.get((m.stage, m.metric), name)
        if owner != name:
            raise EvaluationContractError(
                f"evaluator {name!r} reported {m.stage}/{m.metric}, a metric of evaluator {owner!r}"
            )
    if result.status is not Status.VALID:
        return
    measured = {(m.stage, m.metric): m for m in result.measurements}
    owned = [key for key, owner in plan.owners.items() if owner == name]
    missing = [key for key in owned if key not in measured]
    if missing:
        raise EvaluationContractError(
            f"evaluator {name!r} returned a valid result without required metric(s): "
            f"{_show(missing)}"
        )
    for constraint in spec.constraints:
        key = (constraint.stage, constraint.metric)
        if key not in owned:
            continue
        unit = measured[key].unit
        if constraint.unit is not None and unit != constraint.unit:
            raise EvaluationContractError(
                f"constraint on {_show([key])} declares unit {constraint.unit!r} but the "
                f"measurement has unit {unit!r}"
            )


def _request(
    context: StageContext,
    values: Mapping[str, Scalar],
    entry: EvaluatorSpec,
    evaluator: EvaluatorProtocol | None = None,
) -> EvaluationRequest:
    """The request an evaluator receives, for ``check()`` and ``evaluate()`` alike.

    ``settings`` are those of its ``entry`` in the problem plus the defaults the evaluator
    declares, each quantity in its declared unit; ``collections`` the collection of each role it
    declares; ``events`` and ``seed`` are those of the stage it reads.
    """
    events, seed = context.conditions(entry.stage)
    return EvaluationRequest(
        candidate=values,
        settings=with_defaults(evaluator, entry.settings),
        collections=collection_names(evaluator, entry.collections),
        events=events,
        seed=seed,
    )


def _measure(
    spec: ProblemSpec,
    plan: _Plan,
    key: CandidateKey,
    entry: EvaluatorSpec,
    evaluator: EvaluatorProtocol,
    stages: StageRecords,
    inputs: Mapping[str, Path],
    workdir: Path | None,
    seen: dict[Path, tuple[str, ...] | str],
) -> EvaluationRecord:
    """The record of one evaluator, called once on the files of the stages it reads.

    An evaluator whose stage cannot be read, or whose stage's file lacks a collection it
    declares (or holds it with another type), gets a ``failed`` result saying so and is not
    called. Its own directory, ``workdir/evaluators/<name>``, is created just before the
    call; ``elapsed_seconds`` is the time of the call alone.
    """
    stage = _stage(evaluator)
    given = {name: path for name, path in inputs.items() if name not in STAGE_TYPES}
    given.update(stages.inputs(stage))
    request = replace(plan.requests[entry.name], inputs=given)
    error = stages.error(stage) or _collection_error(evaluator, request, seen)
    if error is not None:
        return _record(spec, plan, entry, evaluator, key, request, EvaluationResult.failed(error))
    own = None if workdir is None else workdir / "evaluators" / entry.name
    request = replace(request, workdir=own)
    if own is not None:
        try:
            own.mkdir(parents=True)
        except OSError as exc:
            result = EvaluationResult.failed(f"cannot create evaluator directory {own}: {exc}")
            return _record(spec, plan, entry, evaluator, key, request, result)
    started = time.perf_counter()
    try:
        result = result_from(evaluator, evaluator.evaluate(request))
    except EvaluationFailed as exc:
        result = EvaluationResult.failed(str(exc))
    except EvaluationInvalid as exc:
        result = EvaluationResult.invalid(str(exc))
    elapsed = time.perf_counter() - started
    return _record(spec, plan, entry, evaluator, key, request, result, elapsed)


def _record(
    spec: ProblemSpec,
    plan: _Plan,
    entry: EvaluatorSpec,
    evaluator: EvaluatorProtocol,
    key: CandidateKey,
    request: EvaluationRequest,
    result: EvaluationResult,
    elapsed: float | None = None,
) -> EvaluationRecord:
    _check_units(evaluator, result)
    _check_result(spec, plan, entry.name, result)
    return EvaluationRecord(request, result, entry.name, entry.version, key, elapsed)


def candidate_status(statuses: Iterable[Status]) -> Status:
    """The status of a candidate from its evaluators': ``failed`` if any failed, otherwise
    ``invalid`` if any is, otherwise ``valid``."""
    seen = set(statuses)
    return next((s for s in (Status.FAILED, Status.INVALID) if s in seen), Status.VALID)


def candidate_message(results: Iterable[tuple[str, Status, str | None]]) -> str | None:
    """The message of a candidate from its evaluators' ``(name, status, message)``, in problem
    order: each that is not ``valid``, with its status and message."""
    reasons = [
        f"{name} {status}: {message}"
        for name, status, message in results
        if status is not Status.VALID
    ]
    return "; ".join(reasons) or None


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One candidate, measured by every evaluator of its problem.

    ``records`` holds one :class:`EvaluationRecord` per evaluator, in the problem's order, with
    only that evaluator's own work; ``stages`` holds the pipeline stages they read, each recorded
    once, and ``reads`` the stages each evaluator read. ``owners`` maps every metric an objective
    or constraint names to the evaluator that measures it. ``elapsed_seconds`` is the wall time
    of the whole evaluation: the stages, the evaluator calls and k4opt's checks. Its mappings are
    read-only copies, so a reader cannot change which evaluator owns a metric.

    ``represents_candidate`` is ``False`` when the caller supplied a stage that the parameters
    change (``k4opt evaluate --input simulation=FILE``): that file was not made from the
    candidate, so the record must never count as an evaluation of it, whatever its status.
    ``warnings`` are the preflight's: collections the simulation was not expected to write.
    """

    candidate_key: CandidateKey
    records: tuple[EvaluationRecord, ...]
    stages: StageRecords
    owners: Mapping[MetricKey, str]
    reads: Mapping[str, tuple[str, ...]]
    elapsed_seconds: float
    represents_candidate: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "owners", FrozenMap(self.owners))
        reads = {name: tuple(stages) for name, stages in self.reads.items()}
        object.__setattr__(self, "reads", FrozenMap(reads))

    @property
    def status(self) -> Status:
        """``failed`` if any evaluator failed (the evaluation is incomplete), otherwise
        ``invalid`` if any found the result unusable, otherwise ``valid``.

        A candidate is thus only ever ``invalid`` after a complete evaluation.
        """
        return candidate_status(record.result.status for record in self.records)

    @property
    def message(self) -> str | None:
        """Every evaluator that is not ``valid``, with its status and message, in problem order."""
        return candidate_message(
            (record.evaluator_name, record.result.status, record.result.message)
            for record in self.records
        )

    def record(self, name: str) -> EvaluationRecord:
        """The record of the evaluator called ``name``; :class:`KeyError` if there is none."""
        for record in self.records:
            if record.evaluator_name == name:
                return record
        raise KeyError(name)

    def value(self, stage: str, metric: str) -> Measurement | None:
        """The measurement of an objective or constraint metric by the evaluator that owns it.

        ``None`` unless that evaluator's result is ``valid``: the diagnostic values of an
        ``invalid`` or ``failed`` result never stand in for an objective.
        """
        owner = self.owners.get((stage, metric))
        if owner is None:
            return None
        result = self.record(owner).result
        if result.status is not Status.VALID:
            return None
        return next(
            (m for m in result.measurements if (m.stage, m.metric) == (stage, metric)), None
        )


def _evaluate_candidate(
    spec: ProblemSpec,
    candidate: Mapping[str, Scalar],
    implementations: Sequence[EvaluatorFactory],
    *,
    inputs: Mapping[str, Path] | None = None,
    workdir: Path | None = None,
) -> CandidateRecord:
    """Evaluate one complete ``candidate`` of ``spec`` with an instance of each implementation.

    The candidate maps each canonical parameter reference to a scalar; ``implementations`` (an
    evaluator class, say) implement the problem's ``evaluators``, in any order. Before any stage
    runs, raises :class:`CandidateError` for a candidate outside the declared domains and
    :class:`EvaluationContractError` for implementations that cannot be created or do not match
    the problem (missing, undeclared, given twice, another version), settings an evaluator's
    ``check()`` rejects, a stage that cannot run, or an objective or constraint metric that no
    evaluator, or two, declare. The stages the evaluators read and ``inputs`` does not supply
    then run once each (on ``inputs["geometry"]``, in ``workdir``); any other entry of
    ``inputs`` supplies the file of the stage it is named after. An evaluator whose stage
    failed, or whose stage output lacks the requested events, gets a ``failed`` result without
    being called; the others are still called. An evaluator that raises
    :class:`EvaluationFailed` or :class:`EvaluationInvalid` gets that result; any other
    exception propagates, and so does an :class:`EvaluationContractError` for a result that
    breaks the contract.
    """
    started = time.perf_counter()
    values, key = _validate_candidate(spec, candidate)
    inputs = dict(inputs or {})
    plan = _preflight(_bind_problem(spec, implementations, inputs), values)
    checked = time.perf_counter() - started
    return _execute_plan(spec, plan, key, inputs=inputs, workdir=workdir, checked_seconds=checked)


def _execute_plan(
    spec: ProblemSpec,
    plan: _Plan,
    key: CandidateKey,
    *,
    inputs: Mapping[str, Path],
    workdir: Path | None,
    checked_seconds: float,
    timeout: float | None = None,
) -> CandidateRecord:
    """Execute an already validated plan; evaluator checks and defaults are not repeated.

    ``checked_seconds``, the time the candidate's validation and preflight took, counts towards
    ``elapsed_seconds``, which thus covers k4opt's checks too (not what the caller does between).
    ``timeout`` limits each stage that runs a program (:attr:`StageContext.timeout`).
    """
    started = time.perf_counter() - checked_seconds
    context = replace(plan.context, timeout=timeout)
    stages = run_stages(context, *plan.targets, inputs=inputs, workdir=workdir)
    seen: dict[Path, tuple[str, ...] | str] = {}  # the collection names of each stage file
    records = tuple(
        _measure(spec, plan, key, entry, evaluator, stages, inputs, workdir, seen)
        for entry, evaluator in plan.evaluators
    )
    reads = {
        entry.name: tuple(stages.chain(_stage(evaluator))) for entry, evaluator in plan.evaluators
    }
    represents = not any(changed_by_geometry(name) for name in stages.supplied)
    elapsed = time.perf_counter() - started
    return CandidateRecord(
        key, records, stages, plan.owners, reads, elapsed, represents, plan.warnings
    )
