"""Internal bridge: one candidate of a problem, measured by each of its evaluators.

Not a public workflow. It validates a candidate against the declared parameter domains, binds
the given evaluator implementations to the problem's ``evaluators`` and checks, before anything
runs, their identity, the stage each reads, their settings, each sample's pipeline stages and
that each objective and constraint names a metric its evaluator measures. It then runs the
candidate's tasks (:mod:`k4opt.problem._tasks`): its geometry, every stage of each sample that
its evaluators read, each once (:mod:`k4opt.problem._stages`), and each evaluation once: an
evaluator on one of its samples, or on the candidate, with the outputs of the tasks it reads. A
task whose needs did not all succeed is ``blocked`` and not run. Each ``valid`` result must
carry the metrics the objectives and constraints name of it, and the artifacts its evaluator
declares; every artifact must be a file in the evaluation's own directory, and is hashed once,
when the evaluation returns. A task whose input files changed since they were made fails. It
neither locates evaluators nor materializes geometry: the caller's function does that, as the
work of the geometry task.

An entry of ``evaluators`` is a binding of an implementation, an evaluator class, identified by
its binding name (``as:``, the implementation's ``name`` when not given): tasks, records, run
directories and metric references use it, with the sample for an evaluation of one
(``gamma/calo.photon``). One implementation may be bound several times, with other settings. It
is bound to its entries once, through a probe instance that is used for nothing else. Every
candidate gets a new instance for each evaluation, whose ``check()`` and ``evaluate()`` it calls,
so no state of an instance reaches another sample, binding or candidate.
"""

from __future__ import annotations

import difflib
import json
import math
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from copy import deepcopy
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
    EvaluatorProtocol,
    Measurement,
    Scalar,
    Status,
    available_collections,
    check_settings,
    collection_names,
    collection_types,
    declared_artifacts,
    declared_collections,
    metric_units,
    result_from,
    with_defaults,
)
from k4opt.frozen import FrozenMap, Json, thaw
from k4opt.problem._stages import (
    STAGE_TYPES,
    StageContext,
    changed_by_geometry,
    check_run_inputs,
    check_stages,
    make_stage,
    no_edm4hep_file,
    stage_output,
)
from k4opt.problem._tasks import (
    DONE,
    GEOMETRY,
    Kind,
    Made,
    Plan,
    Task,
    TaskId,
    TaskResult,
    Tree,
    candidate_message,
    candidate_status,
    changed_files,
    evaluations,
    plan_tasks,
    run_tasks,
    stage_task,
)
from k4opt.problem.errors import CandidateError
from k4opt.problem.models import EvaluatorSpec, MetricRef, Parameter, ProblemSpec

type MetricKey = tuple[str, str]
"""A metric as an evaluator declares and measures it: ``(stage, metric)``."""


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


def _create(implementation: type[EvaluatorProtocol]) -> EvaluatorProtocol:
    """A new instance of the class ``implementation``, which must have ``name``, ``version`` and
    ``evaluate``; :class:`EvaluationContractError` if it is no class, cannot be created or lacks
    one."""
    if not isinstance(implementation, type):
        # a function that builds an instance could configure it in ways the class's source,
        # which the records identify as the code that ran, does not say
        raise EvaluationContractError(
            f"got a {type(implementation).__name__} object, not an evaluator class: give the "
            "class, whose source is recorded and of which k4opt creates an instance per candidate"
        )
    label = implementation.__qualname__
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
    records the probe's class as the code that runs, the stage and metrics decided what each
    evaluator reads and measures, the settings and collections (their defaults included)
    make the request it gets, which must not change from one candidate to the next, and the
    artifacts are what its results are checked against."""

    cls: type
    name: str
    version: str
    stage: str | None
    settings: Mapping[str, object]
    collections: Mapping[str, object]
    metrics: Mapping[MetricKey, str] | None
    artifacts: Mapping[str, str] | None


def _declaration(evaluator: EvaluatorProtocol) -> _Declaration:
    return _Declaration(
        type(evaluator),
        evaluator.name,
        evaluator.version,
        _stage(evaluator),
        # a copy all through: an instance that changes a nested default changes its class's, and
        # the next instance would declare the changed one unnoticed
        deepcopy(dict(getattr(evaluator, "settings", None) or {})),
        declared_collections(evaluator),
        metric_units(evaluator),
        declared_artifacts(evaluator),
    )


@dataclass(frozen=True, slots=True)
class _Bound:
    """A binding of the problem: its entry, the implementation given for it and what the probe
    instance that bound them declared."""

    entry: EvaluatorSpec
    implementation: type[EvaluatorProtocol]
    declaration: _Declaration


def _instance(bound: _Bound) -> EvaluatorProtocol:
    """A new instance of a bound implementation; :class:`EvaluationContractError` unless it
    declares what the probe declared."""
    evaluator = _create(bound.implementation)
    labels = (
        "class",
        "name",
        "version",
        "stage",
        "settings",
        "collections",
        "metrics",
        "artifacts",
    )
    changed = [
        f"{label} {new!r}, not {old!r}"
        for label, new, old in zip(labels, _declaration(evaluator), bound.declaration, strict=True)
        if new != old
    ]
    if changed:
        raise EvaluationContractError(
            f"evaluator {bound.entry.binding!r}: a new instance of "
            f"{bound.implementation.__qualname__} has {'; '.join(changed)}; every instance must "
            "declare what the first did"
        )
    return evaluator


def _as(entry: EvaluatorSpec) -> str:
    """`` (as b)`` for an entry whose binding is not its implementation's name, else nothing."""
    return "" if entry.binding == entry.name else f" (as {entry.binding!r})"


def _bind(
    spec: ProblemSpec, implementations: Sequence[type[EvaluatorProtocol]]
) -> list[tuple[EvaluatorSpec, type[EvaluatorProtocol], EvaluatorProtocol]]:
    """Pair every binding the problem declares with the implementation given for it, in problem
    order, each with the probe instance that decided it.

    A probe of each implementation is matched by its ``name`` to the entries of that name, each
    of which must declare its ``version`` and read its ``stage``; an implementation that is
    missing, not declared, given twice or reads another stage is an
    :class:`EvaluationContractError`, and so is one that cannot be created. Several entries may
    bind one implementation; they share its probe.
    """
    declared: dict[str, list[EvaluatorSpec]] = {}
    for entry in spec.evaluators:
        declared.setdefault(entry.name, []).append(entry)
    given: dict[str, tuple[type[EvaluatorProtocol], EvaluatorProtocol]] = {}
    for implementation in implementations:
        evaluator = _create(implementation)
        name, version = evaluator.name, evaluator.version
        if name in given:
            raise EvaluationContractError(f"evaluator {name!r} is given twice")
        entries = declared.get(name)
        if entries is None:
            raise EvaluationContractError(
                f"problem declares no evaluator {name!r}; declared: {', '.join(declared)}"
            )
        stage = _stage(evaluator)
        for entry in entries:
            if version != entry.version:
                raise EvaluationContractError(
                    f"problem declares evaluator {name!r}{_as(entry)} version "
                    f"{entry.version!r}, got version {version!r}"
                )
            if stage != entry.stage:
                if entry.stage is None:
                    fix = f"add stage: {stage} to its entry in the problem"
                elif stage is None:
                    fix = f"it reads no stage; remove stage: {entry.stage} from its entry"
                else:
                    fix = f"it reads {stage!r}, not {entry.stage!r}"
                raise EvaluationContractError(
                    f"evaluator {name!r}{_as(entry)} does not read the stage the problem says: "
                    f"{fix}"
                )
        given[name] = (implementation, evaluator)
    missing = [
        f"{name!r} version {entries[0].version!r}"
        for name, entries in declared.items()
        if name not in given
    ]
    if missing:
        raise EvaluationContractError(f"no evaluator given for {', '.join(missing)}")
    return [(entry, *given[entry.name]) for entry in spec.evaluators]


def _show(metrics: Collection[MetricKey]) -> str:
    return ", ".join(f"{stage}/{metric}" for stage, metric in metrics)


def _check_references(
    spec: ProblemSpec, evaluators: Sequence[tuple[EvaluatorSpec, EvaluatorProtocol]]
) -> None:
    """Each objective and constraint names a metric its evaluator measures, if that one declares
    its ``metrics``, and a constraint ``unit`` is the unit it declares for it; otherwise the
    result decides. Raises :class:`EvaluationContractError`."""
    units = {entry.binding: metric_units(evaluator) for entry, evaluator in evaluators}
    for ref in spec.references:
        metrics = units[ref.evaluator]
        if metrics is not None and (ref.stage, ref.metric) not in metrics:
            raise EvaluationContractError(
                f"evaluator {ref.evaluator!r} does not measure {ref.stage}/{ref.metric}; "
                f"supported: {_show(sorted(metrics))}"
            )
    for constraint in spec.constraints:
        metrics = units[constraint.evaluator]
        if metrics is None or constraint.unit is None:
            continue
        unit = metrics[(constraint.stage, constraint.metric)]
        if constraint.unit != unit:
            raise EvaluationContractError(
                f"constraint on {constraint.label} declares unit {constraint.unit!r} but "
                f"evaluator {constraint.evaluator!r} measures it in {unit!r}"
            )


def _identical(
    spec: ProblemSpec, evaluators: Sequence[tuple[EvaluatorSpec, EvaluatorProtocol]]
) -> tuple[str, ...]:
    """A warning for each binding that binds an implementation as an earlier one does on the same
    sample, or on the candidate: with the same settings (their defaults filled in) and
    collections, it measures the same twice. And one for each sample whose stages are those of an
    earlier one: its events are simulated again."""
    keys = {  # by entry: one binding name may bind its samples with other settings
        id(entry): (
            entry.name,
            _canonical(with_defaults(evaluator, entry.settings)),
            _canonical(collection_names(evaluator, entry.collections)),
        )
        for entry, evaluator in evaluators
    }
    seen: dict[tuple[str | None, tuple[str, str, str]], str] = {}
    warnings: dict[str, None] = {}
    for task, entry in evaluations(spec.samples, [entry for entry, _ in evaluators]):
        key = (task.sample, keys[id(entry)])
        if key not in seen:
            seen[key] = entry.binding
            continue
        where = "" if task.sample is None else f" on {task.sample}"
        warning = (
            f"evaluators {seen[key]!r} and {entry.binding!r} bind {entry.name} with the same "
            f"settings and collections{where}; each is evaluated separately"
        )
        warnings[warning] = None
    first: dict[object, str] = {}
    for name, sample in spec.samples.items():
        stages = tuple(sample.stages.items())
        if stages in first:
            warnings[
                f"samples {first[stages]} and {name} have identical stages; each is simulated "
                "separately"
            ] = None
        else:
            first[stages] = name
    return tuple(warnings)


def _canonical(settings: Mapping[str, Json]) -> str:
    """``settings`` as text that is the same for equal settings, whatever the key order of any
    nested mapping; a default that is not JSON is written as its ``repr``."""
    return json.dumps(thaw(settings), sort_keys=True, default=repr)


def _unexpected_collections(
    spec: ProblemSpec,
    evaluators: Sequence[tuple[EvaluatorSpec, EvaluatorProtocol]],
    provided: Collection[TaskId],
) -> tuple[str, ...]:
    """A warning for each collection an evaluator reads from the simulation of a sample that
    ddsim is not expected to write for the problem's geometry, with the closest expected name;
    once per binding, however many of its samples ddsim simulates.

    Only a warning: the prediction comes from the unmodified geometry's readouts, and custom
    sensitive actions can name collections of their own. The simulation's output decides.
    """
    if spec._catalog is None:
        return ()
    expected = sorted(output_collections(spec._catalog.readouts))
    warnings: dict[str, None] = {}
    for entry, evaluator in evaluators:
        if _stage(evaluator) != "simulation":
            continue
        simulated = [
            sample
            for sample in entry.samples
            if (stage := spec.samples[sample].stages.get("simulation")) is not None
            and stage.executor == "ddsim"
            # a supplied file is checked itself, before it is read
            and stage_task("simulation", sample) not in provided
        ]
        if not simulated:
            continue
        for role, name in collection_names(evaluator, entry.collections).items():
            if name in expected:
                continue
            close = difflib.get_close_matches(name, expected, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else "."
            warning = (
                f"evaluator {entry.binding!r}: ddsim is not expected to write the collection "
                f"{name!r} ({role}) for {spec.geometry}{hint} The simulation's output "
                "decides."
            )
            warnings[warning] = None
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
    """What holds for every candidate: the implementations bound to the problem, the stages of
    each sample they read and the tasks of a candidate."""

    contexts: Mapping[str, StageContext]  # by sample
    evaluators: tuple[_Bound, ...]  # problem order
    evaluations: tuple[tuple[TaskId, _Bound], ...]  # plan order
    targets: Mapping[str, tuple[str, ...]]  # the pipeline stages the evaluators read, by sample
    plan: Plan  # the tasks of every candidate, the stages ``provided`` given
    # collections the simulation is not expected to write, what is measured or simulated twice
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Prepared:
    """What the preflight of one candidate established before anything runs: its own evaluator
    instance for each evaluation and the requests made for it."""

    binding: _Binding
    # by evaluation, in plan order; the requests normalized and checked before execution
    evaluators: Mapping[TaskId, tuple[EvaluatorSpec, EvaluatorProtocol]]
    requests: Mapping[TaskId, EvaluationRequest]


def _bind_problem(
    spec: ProblemSpec,
    implementations: Sequence[type[EvaluatorProtocol]],
    provided: Mapping[TaskId, Path] | None = None,
) -> _Binding:
    """Check what does not depend on a candidate, before a file is created or a stage runs.

    Binds ``implementations`` to the problem through a probe instance of each, which is then
    dropped: checks the stages of each sample they read (those in ``provided``, the files the
    caller supplies by stage task, are not run), each binding's settings and collections against
    what its implementation declares, and the metric each objective and constraint names
    (:func:`_check_references`). An evaluator that declares collections of a stage without an
    EDM4hep file (the gun's, or a HepMC file) is refused. Raises
    :class:`EvaluationContractError`. Collections the simulation is not expected to write, and
    what is measured or simulated twice, are only warnings.
    """
    provided = provided or {}
    contexts = {name: StageContext.of(spec, name) for name in spec.samples}
    targets = {
        sample: tuple(e.stage for e in spec.evaluators if e.stage and sample in e.samples)
        for sample in spec.samples
    }
    probed = _bind(spec, implementations)
    bound = [(entry, probe) for entry, _, probe in probed]
    for sample, context in contexts.items():
        given = {task.name: path for task, path in provided.items() if task.sample == sample}
        check_stages(context, *targets[sample], provided=given)
    for entry, evaluator in bound:
        stage = _stage(evaluator)
        for sample in (
            entry.samples if stage is not None and declared_collections(evaluator) else ()
        ):
            assert stage is not None
            given = {task.name: path for task, path in provided.items() if task.sample == sample}
            reason = no_edm4hep_file(contexts[sample], stage, given)
            if reason is not None:
                raise EvaluationContractError(
                    f"evaluator {entry.binding!r} reads collections from the {stage!r} stage of "
                    f"{sample}, but {reason}"
                )
        # what needs no candidate is checked before any: a check() may depend on the candidate
        check_settings(evaluator, with_defaults(evaluator, entry.settings))
        collection_names(evaluator, entry.collections)
    _check_references(spec, bound)
    warnings = _unexpected_collections(spec, bound, provided) + _identical(spec, bound)
    declarations: dict[type, _Declaration] = {}  # one probe per implementation
    evaluators = tuple(
        _Bound(entry, implementation, declarations.setdefault(implementation, _declaration(probe)))
        for entry, implementation, probe in probed
    )
    of = {id(bound.entry): bound for bound in evaluators}
    planned = tuple(
        (task, of[id(entry)]) for task, entry in evaluations(spec.samples, spec.evaluators)
    )
    plan = plan_tasks(spec.samples, spec.evaluators, provided)
    return _Binding(contexts, evaluators, planned, FrozenMap(targets), plan, warnings)


def _preflight(binding: _Binding, values: Mapping[str, Scalar]) -> _Prepared:
    """Prepare the evaluations of one candidate before a file is created or a stage runs.

    Creates a new instance of the implementation of each evaluation, which must declare what its
    probe declared, checks its settings and collections, then calls each one's ``check()`` with the
    request it would get for this candidate (its settings and collections resolved, the events
    and seed of its sample). Raises :class:`EvaluationContractError` (or what a ``check()``
    raises). What it returns holds these instances and requests made for these ``values``, so it
    serves one candidate only.
    """
    evaluators = {task: (bound.entry, _instance(bound)) for task, bound in binding.evaluations}
    for entry, evaluator in evaluators.values():  # before any check(), which may use the candidate
        check_settings(evaluator, with_defaults(evaluator, entry.settings))
        collection_names(evaluator, entry.collections)
    requests = {}
    for task, (entry, evaluator) in evaluators.items():
        context = None if task.sample is None else binding.contexts[task.sample]
        request = _request(context, values, entry, evaluator)
        requests[task] = request
        check = getattr(evaluator, "check", None)
        if check is not None:
            check(request)
    return _Prepared(binding, FrozenMap(evaluators), FrozenMap(requests))


def _check_result(spec: ProblemSpec, task: TaskId, result: EvaluationResult) -> None:
    """A ``valid`` result of the evaluation ``task`` must report every metric an objective or a
    constraint names of it, in the unit a constraint declares."""
    if result.status is not Status.VALID:
        return
    measured = {(m.stage, m.metric): m for m in result.measurements}
    owned = [ref for ref in spec.references if ref.invocation == task.label]
    missing = [(ref.stage, ref.metric) for ref in owned if (ref.stage, ref.metric) not in measured]
    if missing:
        raise EvaluationContractError(
            f"evaluator {task.label!r} returned a valid result without required metric(s): "
            f"{_show(missing)}"
        )
    for constraint in spec.constraints:
        if constraint.ref.invocation != task.label:
            continue
        unit = measured[(constraint.stage, constraint.metric)].unit
        if constraint.unit is not None and unit != constraint.unit:
            raise EvaluationContractError(
                f"constraint on {constraint.label} declares unit {constraint.unit!r} but the "
                f"measurement has unit {unit!r}"
            )


def _request(
    context: StageContext | None,
    values: Mapping[str, Scalar],
    entry: EvaluatorSpec,
    evaluator: EvaluatorProtocol | None = None,
) -> EvaluationRequest:
    """The request an evaluator receives, for ``check()`` and ``evaluate()`` alike.

    ``settings`` are those of its ``entry`` in the problem plus the defaults the evaluator
    declares, each quantity in its declared unit; ``collections`` the collection of each role it
    declares; ``events`` and ``seed`` are those of the stage it reads in the sample of
    ``context``, ``None`` for an evaluation of the candidate (``context`` is ``None``).
    """
    events, seed = (None, None) if context is None else context.conditions(entry.stage)
    return EvaluationRequest(
        candidate=values,
        settings=with_defaults(evaluator, entry.settings),
        collections=collection_names(evaluator, entry.collections),
        events=events,
        seed=seed,
    )


def _inputs(reads: Mapping[TaskId, TaskResult]) -> dict[str, Path]:
    """What a task is given of the tasks it reads: the main output of each, by its name."""
    return {task.name: result.output for task, result in reads.items() if result.output is not None}


def _measure(
    spec: ProblemSpec,
    prepared: _Prepared,
    key: CandidateKey,
    task: TaskId,
    reads: Mapping[TaskId, TaskResult],
    workdir: Path | None,
    seen: dict[Path, tuple[str, ...] | str],
) -> tuple[EvaluationRecord, dict[str, Made]]:
    """The record of one evaluation, its evaluator called once on the outputs of the tasks it
    ``reads``, which all succeeded, and the artifacts it made (:func:`_artifacts`).

    An evaluator whose stage's file lacks a collection it declares (or holds it with another
    type), or reads a file that changed since its task made it, gets a ``failed`` result saying
    so and is not called. Its own directory ``workdir`` is created just before the call;
    ``elapsed_seconds`` is the time of the call alone. A file it reads that changes during the
    call fails it too: the record could not say what it read.
    """
    entry, evaluator = prepared.evaluators[task]
    request = replace(prepared.requests[task], inputs=_inputs(reads))
    changed = changed_files(reads.values())
    if changed:
        result = EvaluationResult.failed(_changed_before(changed))
        return _record(spec, task, entry, evaluator, key, request, result), {}
    error = _collection_error(evaluator, request, seen)
    if error is not None:
        result = EvaluationResult.failed(error)
        return _record(spec, task, entry, evaluator, key, request, result), {}
    request = replace(request, workdir=workdir)
    if workdir is not None:
        try:
            workdir.mkdir(parents=True)
        except OSError as exc:
            result = EvaluationResult.failed(f"cannot create evaluator directory {workdir}: {exc}")
            return _record(spec, task, entry, evaluator, key, request, result), {}
    started = time.perf_counter()
    try:
        result = result_from(evaluator, evaluator.evaluate(request))
    except EvaluationFailed as exc:
        result = EvaluationResult.failed(str(exc))
    except EvaluationInvalid as exc:
        result = EvaluationResult.invalid(str(exc))
    elapsed = time.perf_counter() - started
    record = _record(spec, task, entry, evaluator, key, request, result, elapsed)
    made = _artifacts(task, evaluator, result, workdir)
    changed = changed_files(reads.values())
    if changed:  # what it returned stays, as diagnostics of a failed result
        result = EvaluationResult.failed(
            _changed_during(changed),
            result.measurements,
            artifacts=result.artifacts,
            provenance=result.provenance,
        )
        record = replace(record, result=result)
    return record, made


def _changed_before(refs: Sequence[str]) -> str:
    return f"not run: {', '.join(refs)} changed after it was made"


def _changed_during(refs: Sequence[str]) -> str:
    return f"{', '.join(refs)} changed while it ran, so the record cannot say what it read"


def _artifacts(
    task: TaskId, evaluator: EvaluatorProtocol, result: EvaluationResult, workdir: Path | None
) -> dict[str, Made]:
    """The artifacts of the result of the evaluation ``task``, each hashed once, now.

    Each must be a file in the evaluation's own directory ``workdir``, its links resolved. An
    evaluator that declares ``artifacts`` returns no other, and a ``valid`` result of it every
    one. Raises :class:`EvaluationContractError`.
    """
    where = f"evaluator {task.label!r}"
    declared = declared_artifacts(evaluator)
    if declared is not None:
        unknown = [name for name in result.artifacts if name not in declared]
        if unknown:
            raise EvaluationContractError(
                f"{where} returned the artifact(s) {', '.join(unknown)}, which it does not "
                f"declare; it declares {', '.join(declared) or 'none'}"
            )
        missing = [name for name in declared if name not in result.artifacts]
        if missing and result.status is Status.VALID:
            raise EvaluationContractError(
                f"{where} returned a valid result without its declared artifact(s) "
                f"{', '.join(missing)}"
            )
    made: dict[str, Made] = {}
    for name, path in result.artifacts.items():
        if workdir is None:
            raise EvaluationContractError(
                f"{where}: artifact {name!r}: the evaluation has no directory of its own to "
                "write it in"
            )
        own, resolved = workdir.resolve(), path.resolve()
        if resolved == own or not resolved.is_relative_to(own):
            raise EvaluationContractError(
                f"{where}: artifact {name!r} ({path}) is not in its own directory {workdir}; "
                "write it in request.workdir"
            )
        try:
            made[name] = Made.file(task, name, resolved, hashed=True)
        except OSError as exc:
            raise EvaluationContractError(
                f"{where}: artifact {name!r} ({path}) is no file it can read: {exc}"
            ) from None
    return made


def _record(
    spec: ProblemSpec,
    task: TaskId,
    entry: EvaluatorSpec,
    evaluator: EvaluatorProtocol,
    key: CandidateKey,
    request: EvaluationRequest,
    result: EvaluationResult,
    elapsed: float | None = None,
) -> EvaluationRecord:
    _check_units(evaluator, result)
    _check_result(spec, task, result)
    return EvaluationRecord(request, result, entry.name, entry.version, key, elapsed)


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One candidate, measured by every evaluator of its problem.

    ``plan`` holds its tasks, and ``results`` the result of each, in plan order: its geometry,
    the pipeline stages of each sample that its evaluators read, each recorded once, and every
    evaluation, with only its own work. ``elapsed_seconds`` is the wall time of the whole
    evaluation: the tasks and k4opt's checks. Its mappings are read-only copies.

    ``represents_candidate`` is ``False`` when the caller supplied a stage that the parameters
    change (``k4opt evaluate --input simulation=FILE``): that file was not made from the
    candidate, so the record must never count as an evaluation of it, whatever its status.
    ``warnings`` are the preflight's: collections the simulation was not expected to write, and
    what is measured or simulated twice.
    """

    candidate_key: CandidateKey
    plan: Plan
    results: Mapping[TaskId, TaskResult]
    elapsed_seconds: float
    represents_candidate: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = [task.id.label for task in self.plan.tasks if task.id not in self.results]
        if missing or len(self.results) != len(self.plan.tasks):
            raise EvaluationContractError(
                f"the results do not match the plan's tasks (missing: {', '.join(missing)})"
            )
        results = {task.id: self.results[task.id] for task in self.plan.tasks}
        object.__setattr__(self, "results", FrozenMap(results))

    @property
    def evaluations(self) -> Mapping[str, EvaluationRecord]:
        """The record of each evaluation, by its label (``gamma/calo.photon``, ``readout``), in
        plan order; a blocked one's evaluator was never called."""
        return FrozenMap(
            {
                task.id.label: record
                for task in self.plan.of(Kind.EVALUATOR)
                if (record := self.results[task.id].record) is not None
            }
        )

    @property
    def records(self) -> tuple[EvaluationRecord, ...]:
        """The record of each evaluation, in plan order."""
        return tuple(self.evaluations.values())

    @property
    def supplied(self) -> frozenset[str]:
        """The stages the caller supplied as files, by label (``gamma/simulation``)."""
        return frozenset(t.id.label for t in self.plan.of(Kind.STAGE) if t.given == "supplied")

    def stage(self, name: str, sample: str) -> TaskResult | None:
        """The result of the stage ``name`` of ``sample``, ``None`` if it was not planned."""
        return self.results.get(stage_task(name, sample))

    @property
    def status(self) -> Status:
        """``failed`` if a task failed or was blocked (the evaluation is incomplete), otherwise
        ``invalid`` if an evaluator found the result unusable, otherwise ``valid``.

        A candidate is thus only ever ``invalid`` after a complete evaluation.
        """
        return candidate_status(result.status for result in self.results.values())

    @property
    def message(self) -> str | None:
        """Every task that failed or is invalid, with its message, then those they blocked."""
        return candidate_message(self.results.values())

    def changed(self) -> list[str]:
        """The references of the files its tasks made that are no longer what its record says
        (:meth:`~k4opt.problem._tasks.Made.changed`)."""
        return changed_files(self.results.values())

    def record(self, label: str) -> EvaluationRecord:
        """The record of the evaluation ``label`` (``gamma/calo.photon``, ``readout``);
        :class:`KeyError` if there is none."""
        return self.evaluations[label]

    def value(self, ref: MetricRef) -> Measurement | None:
        """The measurement of the metric ``ref`` by the evaluation it names.

        ``None`` unless that evaluation's result is ``valid``: the diagnostic values of an
        ``invalid`` or ``failed`` result never stand in for an objective.
        """
        try:
            result = self.record(ref.invocation).result
        except KeyError:
            return None
        if result.status is not Status.VALID:
            return None
        return next(
            (m for m in result.measurements if (m.stage, m.metric) == (ref.stage, ref.metric)),
            None,
        )


type MakeGeometry = Callable[[], tuple[Path, Mapping[str, Json]]]
"""The work of the geometry task: ``(compact file, what to record of it)``; raises if the
geometry cannot be made, which stops the candidate."""


def _evaluate_candidate(
    spec: ProblemSpec,
    candidate: Mapping[str, Scalar],
    implementations: Sequence[type[EvaluatorProtocol]],
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
    ``check()`` rejects, a stage that cannot run, or an objective or constraint metric its
    evaluator does not declare. ``inputs["geometry"]`` is the candidate's geometry, already
    prepared; any other entry of ``inputs`` supplies the file of the stage of a sample it is named
    after (``gamma/simulation``). The stages the evaluators read and ``inputs`` does not supply
    then run once each, in ``workdir``.
    An evaluator whose stage failed, or whose stage output lacks the requested events, is
    ``blocked`` and not called; the others are still called. An evaluator that raises
    :class:`EvaluationFailed` or :class:`EvaluationInvalid` gets that result; any other
    exception propagates, and so does an :class:`EvaluationContractError` for a result that
    breaks the contract.
    """
    started = time.perf_counter()
    values, key = _validate_candidate(spec, candidate)
    given = dict(inputs or {})
    geometry = given.pop(GEOMETRY.name, None)
    supplied = {}
    for label, path in given.items():
        sample, _, stage = label.partition("/")
        if sample not in spec.samples or stage not in STAGE_TYPES:
            raise EvaluationContractError(
                f"input {label!r}: expected SAMPLE/STAGE, a sample of the problem and one of "
                f"{', '.join(STAGE_TYPES)}"
            )
        supplied[stage_task(stage, sample)] = path
    binding = _bind_problem(spec, implementations, supplied)
    planned = {task.id for task in binding.plan.of(Kind.STAGE)}
    for task in supplied:  # a file no task reads would be dropped without a word
        if task not in planned:
            raise EvaluationContractError(
                f"input {task.label!r}: no evaluator of the problem reads that stage of "
                f"{task.sample}"
            )
    prepared = _preflight(binding, values)
    checked = time.perf_counter() - started
    return _execute_plan(
        spec,
        prepared,
        key,
        supplied=supplied,
        geometry=None if geometry is None else lambda: (geometry, {}),
        workdir=workdir,
        checked_seconds=checked,
    )


def _execute_plan(
    spec: ProblemSpec,
    prepared: _Prepared,
    key: CandidateKey,
    *,
    supplied: Mapping[TaskId, Path],
    geometry: MakeGeometry | None,
    workdir: Path | None,
    checked_seconds: float,
    timeout: float | None = None,
) -> CandidateRecord:
    """Run the tasks of a prepared candidate; evaluator checks and defaults are not repeated.

    ``supplied`` holds the files the caller supplies by stage task, the ones the plan was made with;
    ``geometry`` makes the candidate's geometry, and ``None`` gives none (to evaluators that need
    none). Each task writes below ``workdir``, in the directory the plan gives it.
    ``checked_seconds``, the time the candidate's validation and preflight took, counts towards
    ``elapsed_seconds``, which thus covers k4opt's checks too (not what the caller does between).
    ``timeout`` limits each stage that runs a program (:attr:`StageContext.timeout`).
    """
    started = time.perf_counter() - checked_seconds
    binding = prepared.binding
    contexts = {name: replace(c, timeout=timeout) for name, c in binding.contexts.items()}
    plan = binding.plan
    stages = plan.of(Kind.STAGE)
    check_run_inputs([t.id.name for t in stages if t.given is None], geometry is not None, workdir)
    seen: dict[Path, tuple[str, ...] | str] = {}  # the collection names of each stage file

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        own = None if workdir is None or task.directory is None else workdir / task.directory
        began = time.perf_counter()
        if task.id.kind is Kind.GEOMETRY:
            if geometry is None:
                return TaskResult(task.id, DONE)
            compact, recorded = geometry()
            # every simulation of every sample reads these files: watched like what a stage made
            tree = Tree.of(own) if own is not None and own.is_dir() else None
            return TaskResult(
                task.id,
                DONE,
                output=compact,
                provenance=recorded,
                elapsed_seconds=_since(began),
                tree=tree,
            )
        if task.id.kind is Kind.STAGE:
            assert task.id.sample is not None
            changed = changed_files(reads.values())
            if changed:
                return TaskResult(task.id, Status.FAILED, _changed_before(changed))
            outcome, main = make_stage(
                contexts[task.id.sample],
                task.id.name,
                inputs=_inputs(reads),
                workdir=own,
                supplied=supplied.get(task.id),
            )
            error, made = outcome.error, {}
            if error is None and main is not None and task.given is None:
                name = stage_output(task.id.name)  # made in its directory: never hashed
                try:
                    made[name] = Made.file(task.id, name, main, hashed=False)
                except OSError as exc:
                    error, main = f"cannot read its output: {exc}", None
            changed = changed_files(reads.values())
            if changed and error is None:
                error, main, made = _changed_during(changed), None, {}
            return TaskResult(
                task.id,
                DONE if error is None else Status.FAILED,
                error,
                main,
                outcome.artifacts,
                outcome.provenance,
                _since(began),
                made=made,
            )
        record, made = _measure(spec, prepared, key, task.id, reads, own, seen)
        result = record.result
        return TaskResult(
            task.id,
            result.status,
            result.message,
            elapsed_seconds=_since(began),
            record=record,
            made=made,
        )

    results = run_tasks(plan, run)
    for task in plan.of(Kind.EVALUATOR):  # a blocked evaluator's record: what it would have got
        held = results[task.id]
        if held.status == Status.BLOCKED:
            entry, _ = prepared.evaluators[task.id]
            assert held.message is not None
            outcome = EvaluationResult(Status.BLOCKED, message=held.message)
            request = prepared.requests[task.id]
            record = EvaluationRecord(request, outcome, entry.name, entry.version, key, None)
            results[task.id] = replace(held, record=record)
    represents = not any(
        changed_by_geometry(task.id.name) for task in stages if task.given == "supplied"
    )
    return CandidateRecord(key, plan, results, _since(started), represents, binding.warnings)


def _since(started: float) -> float:
    return time.perf_counter() - started
