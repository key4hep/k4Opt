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
(``gamma/calo.photon``). One implementation may be bound several times, with other settings. A
combined evaluator is bound to the candidate and reads, by role, what the problem names: the
evaluations of other samples, given to it as :class:`~k4opt.evaluation.RoleInput`, or their stage
files; it runs only once all of them succeeded (``valid``), and every ``valid`` result it reads
carries the metrics its roles name. An implementation is bound to its entries once, through a
probe instance that is used for nothing else. Every
candidate gets a new instance for each evaluation, whose ``check()`` and ``evaluate()`` it calls,
so no state of an instance reaches another sample, binding or candidate.

Only the required tasks decide the candidate's status: the evaluations its objectives and
constraints name, those of bindings marked ``required: true``, and what they read. Every other
evaluation is a diagnostic: it runs, and its record is kept, but it neither fails the candidate
nor makes it invalid, and a supplied stage only diagnostics read does not keep the record from
representing the candidate.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import math
import os
import time
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path

from k4opt.ddsim import output_collections, threads
from k4opt.evaluation import (
    Artifact,
    CandidateKey,
    EvaluationContractError,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorProtocol,
    Measurement,
    RoleInput,
    Scalar,
    Status,
    check_settings,
    collection_names,
    declared_artifacts,
    declared_collections,
    declared_roles,
    metric_units,
    with_defaults,
)
from k4opt.frozen import FrozenMap, Json, thaw
from k4opt.problem._backends import Backend, Handle, InlineBackend
from k4opt.problem._instances import _Bound, _create, _Declaration, _declaration, _instance, _stage
from k4opt.problem._schedule import Scheduler
from k4opt.problem._stages import (
    STAGE_TYPES,
    StageContext,
    changed_by_geometry,
    check_run_inputs,
    check_stages,
    no_edm4hep_file,
    stage_output,
)
from k4opt.problem._tasks import (
    GEOMETRY,
    Kind,
    Plan,
    Task,
    TaskId,
    TaskResult,
    candidate_message,
    candidate_status,
    changed_files,
    diagnostics,
    evaluations,
    plan_tasks,
    read_stages,
    reference_task,
    stage_task,
)
from k4opt.problem._work import (
    GeometryWork,
    Measuring,
    MetricKey,
    Observations,
    SessionContext,
    TaskOutcome,
    TaskSpec,
    Workspace,
    asked,
    local_result,
    portable,
    portable_artifact,
    show,
    to_result,
)
from k4opt.problem.errors import CandidateError
from k4opt.problem.models import EvaluatorSpec, MetricRef, Parameter, ProblemSpec, Reference


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
        stage, roles = _stage(evaluator), declared_roles(evaluator)
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
            if entry.roles and not roles:
                raise EvaluationContractError(
                    f"evaluator {name!r}{_as(entry)} declares no roles, so it is no combined "
                    "evaluator; remove roles: from its entry"
                )
            if roles and not entry.roles:
                raise EvaluationContractError(
                    f"evaluator {name!r}{_as(entry)} is a combined evaluator with the roles "
                    f"{', '.join(roles)}; say what each reads under roles: in its entry"
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
    return show(list(metrics))


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


def _check_roles(
    spec: ProblemSpec, evaluators: Sequence[tuple[EvaluatorSpec, EvaluatorProtocol]]
) -> dict[TaskId, tuple[MetricKey, ...]]:
    """Each combined evaluator gets for its roles what its implementation declares: every role
    and no other, one item unless the role reads ``many``, evaluations where the role reads them
    (of evaluators that declare the metrics and artifacts it reads) and the files of its stage
    otherwise. Returns the metrics the roles read of each evaluation, which a ``valid`` result of
    that evaluation must carry. Raises :class:`EvaluationContractError`."""
    probes = {
        (entry.binding, sample): (entry, evaluator)
        for entry, evaluator in evaluators
        for sample in entry.samples
    }
    demands: dict[TaskId, dict[MetricKey, None]] = {}
    for entry, evaluator in evaluators:
        declared = declared_roles(evaluator)
        if not entry.roles or declared is None:
            continue  # the form is checked when it is bound (_bind)
        where = f"evaluator {entry.binding!r}"
        unknown = [role for role in entry.roles if role not in declared]
        missing = [role for role in declared if role not in entry.roles]
        if unknown or missing:
            raise EvaluationContractError(
                f"{where}: its roles are {', '.join(declared)}"
                + (f"; unknown {', '.join(unknown)}" if unknown else "")
                + (f"; missing {', '.join(missing)}" if missing else "")
            )
        for name, role in declared.items():
            items, at = entry.roles[name], f"{where}: role {name!r}"
            if not role.many and len(items) > 1:
                raise EvaluationContractError(
                    f"{at} reads one item, but the problem gives {len(items)}: "
                    f"{', '.join(item.label for item in items)}"
                )
            for item in items:
                if role.stage is not None:
                    if item.stage != role.stage:
                        raise EvaluationContractError(
                            f"{at} reads the {role.stage} file of a sample, as "
                            f"{{sample: ..., stage: {role.stage}}}; got {item.label}"
                        )
                    continue
                if item.evaluator is None:
                    raise EvaluationContractError(
                        f"{at} reads evaluations, as {{sample: ..., evaluator: ...}}; got the "
                        f"stage {item.label}"
                    )
                upstream, probe = probes[(item.evaluator, item.sample)]
                metrics = metric_units(probe) or {}
                undeclared = [key for key in role.metrics if key not in metrics]
                if undeclared:
                    raise EvaluationContractError(
                        f"{at} reads {_show(undeclared)} of {item.label}, but {upstream.name} "
                        f"does not declare {'it' if len(undeclared) == 1 else 'them'}"
                    )
                artifacts = declared_artifacts(probe) or {}
                absent = [name for name in role.artifacts if name not in artifacts]
                if absent:
                    raise EvaluationContractError(
                        f"{at} reads the artifact(s) {', '.join(absent)} of {item.label}, but "
                        f"{upstream.name} does not declare {'it' if len(absent) == 1 else 'them'}"
                    )
                demands.setdefault(reference_task(item), {}).update(dict.fromkeys(role.metrics))
    return {task: tuple(keys) for task, keys in demands.items()}


def _identical(
    spec: ProblemSpec, evaluators: Sequence[tuple[EvaluatorSpec, EvaluatorProtocol]]
) -> tuple[str, ...]:
    """A warning for each binding that binds an implementation as an earlier one does on the same
    sample, or on the candidate: with the same settings (their defaults filled in), collections
    and roles, it measures the same twice. And one for each sample whose stages are those of an
    earlier one: its events are simulated again."""
    keys = {  # by entry: one binding name may bind its samples with other settings
        id(entry): (
            entry.name,
            _canonical(with_defaults(evaluator, entry.settings)),
            _canonical(collection_names(evaluator, entry.collections)),
            _canonical(
                {role: tuple(i.label for i in items) for role, items in entry.roles.items()}
            ),
        )
        for entry, evaluator in evaluators
    }
    seen: dict[tuple[str | None, tuple[str, str, str, str]], str] = {}
    warnings: dict[str, None] = {}
    for task, entry in evaluations(spec.samples, [entry for entry, _ in evaluators]):
        key = (task.sample, keys[id(entry)])
        if key not in seen:
            seen[key] = entry.binding
            continue
        where = "" if task.sample is None else f" on {task.sample}"
        warning = (
            f"evaluators {seen[key]!r} and {entry.binding!r} bind {entry.name} with the same "
            f"settings and {'roles' if entry.roles else 'collections'}{where}; each is evaluated "
            "separately"
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


@dataclass(frozen=True, slots=True)
class _Binding:
    """What holds for every candidate: the implementations bound to the problem, the stages of
    each sample they read and the tasks of a candidate."""

    contexts: Mapping[str, StageContext]  # by sample
    evaluators: tuple[_Bound, ...]  # problem order
    evaluations: tuple[tuple[TaskId, _Bound], ...]  # plan order
    targets: Mapping[str, tuple[str, ...]]  # the pipeline stages tasks read, by sample
    plan: Plan  # the tasks of every candidate, the stages ``provided`` given
    # collections the simulation is not expected to write, what is measured or simulated twice
    warnings: tuple[str, ...] = ()
    # the metrics the roles of combined evaluators read, by the evaluation that measures them
    demands: Mapping[TaskId, tuple[MetricKey, ...]] = FrozenMap({})
    properties: Mapping[str, Mapping[str, Json]] = FrozenMap({})  # by sample


@dataclass(frozen=True, slots=True)
class _Prepared:
    """What the preflight of one candidate established before anything runs: its own evaluator
    instance for each evaluation and the requests made for it."""

    binding: _Binding
    # by evaluation, in plan order: the entry that binds it, and the request normalized and
    # checked before execution
    entries: Mapping[TaskId, EvaluatorSpec]
    requests: Mapping[TaskId, EvaluationRequest]
    # the instance whose check() accepted the request; none once released (a worker process
    # evaluates with an instance of its own)
    instances: Mapping[TaskId, EvaluatorProtocol] = FrozenMap({})

    def released(self) -> _Prepared:
        """The same without its evaluator instances, which hold whatever they allocated."""
        return replace(self, instances=FrozenMap({}))


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
    targets = {sample: read_stages(sample, spec.evaluators) for sample in spec.samples}
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
    demands = _check_roles(spec, bound)
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
    plan = plan_tasks(spec.samples, spec.evaluators, provided, references=spec.references)
    return _Binding(
        contexts,
        evaluators,
        planned,
        FrozenMap(targets),
        plan,
        warnings,
        FrozenMap(demands),
        FrozenMap({name: sample.properties for name, sample in spec.samples.items()}),
    )


def _preflight(binding: _Binding, values: Mapping[str, Scalar]) -> _Prepared:
    """Prepare the evaluations of one candidate before a file is created or a stage runs.

    Creates a new instance of the implementation of each evaluation, which must declare what its
    probe declared, checks its settings and collections, then calls each one's ``check()`` with the
    request it would get for this candidate (its settings and collections resolved, the events,
    seed and properties of its sample; no roles, since nothing exists yet for them to read).
    Raises :class:`EvaluationContractError` (or what a ``check()``
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
        properties = {} if task.sample is None else binding.properties[task.sample]
        request = _request(context, values, entry, evaluator, properties)
        requests[task] = request
        check = getattr(evaluator, "check", None)
        if check is not None:
            check(request)
    return _Prepared(
        binding,
        FrozenMap({task: entry for task, (entry, _) in evaluators.items()}),
        FrozenMap(requests),
        FrozenMap({task: instance for task, (_, instance) in evaluators.items()}),
    )


def _request(
    context: StageContext | None,
    values: Mapping[str, Scalar],
    entry: EvaluatorSpec,
    evaluator: EvaluatorProtocol | None = None,
    properties: Mapping[str, Json] | None = None,
) -> EvaluationRequest:
    """The request an evaluator receives, for ``check()`` and ``evaluate()`` alike.

    ``settings`` are those of its ``entry`` in the problem plus the defaults the evaluator
    declares, each quantity in its declared unit; ``collections`` the collection of each role it
    declares; ``events`` and ``seed`` are those of the stage it reads in the sample of
    ``context``, ``None`` for an evaluation of the candidate (``context`` is ``None``), and
    ``properties`` those of that sample.
    """
    events, seed = (None, None) if context is None else context.conditions(entry.stage)
    return EvaluationRequest(
        candidate=values,
        settings=with_defaults(evaluator, entry.settings),
        collections=collection_names(evaluator, entry.collections),
        events=events,
        seed=seed,
        properties=properties or {},
    )


def _inputs(reads: Mapping[TaskId, TaskResult], root: Path | None) -> dict[str, Path]:
    """What a task is given of the tasks it reads: the main output of each, by its name, named
    in the candidate's workspace (:func:`~k4opt.problem._work.portable`)."""
    return {
        task.name: portable(result.output, root)
        for task, result in reads.items()
        if result.output is not None
    }


def _role_inputs(
    binding: _Binding, entry: EvaluatorSpec, reads: Mapping[TaskId, TaskResult], root: Path | None
) -> dict[str, tuple[RoleInput, ...]]:
    """What each role of a combined evaluator reads, from the results of the tasks it reads, its
    files named in the candidate's workspace."""
    return {
        role: tuple(_role_input(binding, item, reads[reference_task(item)], root) for item in items)
        for role, items in entry.roles.items()
    }


def _role_input(
    binding: _Binding, item: Reference, result: TaskResult, root: Path | None
) -> RoleInput:
    """One item of a role, from the result of its task, which succeeded: an evaluation with what
    it measured and made, or a stage with its file."""
    properties = binding.properties[item.sample]
    if item.evaluator is not None:
        record = result.record
        assert record is not None  # every evaluation has one
        return RoleInput(
            item.sample,
            binding=item.evaluator,
            measurements=record.result.measurements,
            artifacts={
                name: portable_artifact(made.artifact, root) for name, made in result.made.items()
            },
            events=record.request.events,
            seed=record.request.seed,
            properties=properties,
            settings=record.request.settings,
            provenance=record.result.provenance,
        )
    assert item.stage is not None
    assert result.output is not None  # a stage that makes a file
    name = stage_output(item.stage)
    made = result.made.get(name)
    if made is not None:  # made in its directory: never hashed
        artifact = made.artifact
    else:  # a file of the problem or the caller's, hashed when the session opened
        size, sha256 = result.provenance.get("bytes"), result.provenance.get("sha256")
        artifact = Artifact(
            f"{result.id.label}#{name}",
            result.output,
            size if isinstance(size, int) else result.output.stat().st_size,
            sha256 if isinstance(sha256, str) else None,
        )
    events, seed = binding.contexts[item.sample].conditions(item.stage)
    return RoleInput(
        item.sample,
        stage=item.stage,
        artifacts={item.stage: portable_artifact(artifact, root)},
        events=events,
        seed=seed,
        properties=properties,
        provenance=result.provenance,
    )


def _context(
    spec: ProblemSpec,
    binding: _Binding,
    timeout: float | None = None,
    externals: Mapping[str, Json] | None = None,
) -> SessionContext:
    """What every task of the candidates of ``binding`` needs (:class:`SessionContext`), with
    ``timeout`` for every stage that runs a program and the session geometry's ``externals``."""
    return SessionContext(
        spec,
        FrozenMap({name: replace(c, timeout=timeout) for name, c in binding.contexts.items()}),
        FrozenMap(
            {
                task: Measuring(
                    bound.entry,
                    bound.declaration.metrics,
                    bound.declaration.artifacts,
                    binding.demands.get(task, ()),
                )
                for task, bound in binding.evaluations
            }
        ),
        FrozenMap(externals or {}),
    )


@dataclass(slots=True)
class _Run:
    """One candidate under way, as its coordinator follows it: its workspace, what its preflight
    prepared, its key, the files the caller supplied for its stages and how its geometry is made.
    ``started`` is when its evaluation began, its checks included; ``specs`` holds the
    description of each task handed out, ``results`` its results once all its tasks ended, and
    ``error`` what stopped it, if anything did. ``request``, ``values`` and ``reference`` (what
    its record says it was made for) are the caller's."""

    workspace: Workspace
    prepared: _Prepared | None
    key: CandidateKey = ()
    supplied: Mapping[TaskId, Path] = field(default_factory=dict)
    geometry: GeometryWork | None = None
    started: float = 0.0
    specs: dict[TaskId, TaskSpec] = field(default_factory=dict)
    results: dict[TaskId, TaskResult] = field(default_factory=dict)
    error: Exception | None = None
    own_refs: bool = False
    request: object = None
    values: Mapping[str, Scalar] = field(default_factory=dict)
    reference: object = None


def task_cpus(binding: _Binding, task: Task) -> int:
    """How many processors ``task`` keeps busy: the threads of a simulation ddsim runs
    (``numberOfThreads``), one for every other task."""
    if task.id.kind is not Kind.STAGE or task.given is not None or task.id.name != "simulation":
        return 1
    assert task.id.sample is not None
    stage = binding.contexts[task.id.sample].stages["simulation"]
    return threads(stage.options) if stage.executor == "ddsim" else 1


def _task_spec(run: _Run, task: Task, results: Mapping[TaskId, TaskResult]) -> TaskSpec:
    """The description of ``task`` of ``run``, from the results of the tasks it reads."""
    assert run.prepared is not None
    binding, root = run.prepared.binding, run.workspace.root
    reads = {need: results[need] for need in task.needs}
    watched = tuple(
        portable_artifact(made.artifact, root)
        for result in reads.values()
        for made in result.made.values()
    )
    trees = [result.tree for result in reads.values() if result.tree is not None]
    tree = None if not trees else portable(trees[0].root, root)
    workspace = run.workspace.id
    if task.id.kind is Kind.GEOMETRY:
        return TaskSpec(workspace, task, watched=watched, tree=tree, geometry=run.geometry)
    if task.id.kind is Kind.STAGE:
        given, supplied = _inputs(reads, root), run.supplied.get(task.id)
        cpus = task_cpus(binding, task)
        return TaskSpec(workspace, task, given, watched, tree, supplied=supplied, cpus=cpus)
    entry = run.prepared.entries[task.id]
    if task.id.sample is None:  # the geometry, and what its roles read: no stage of its own
        given = _inputs({need: result for need, result in reads.items() if need == GEOMETRY}, root)
        roles = _role_inputs(binding, entry, reads, root)
    else:
        given, roles = _inputs(reads, root), {}
    request = run.prepared.requests[task.id]
    return TaskSpec(workspace, task, given, watched, tree, request=request, roles=roles)


def _seen(task: Task, results: Mapping[TaskId, TaskResult]) -> Observations:
    """The stats taken of the files the tasks ``task`` reads made, where they were made."""
    reads = [results[need] for need in task.needs]
    made = {m.artifact.ref: m.stat for result in reads for m in result.made.values()}
    trees = [result.tree.stats for result in reads if result.tree is not None]
    return Observations(FrozenMap(made), trees[0] if trees else None)


def _result(run: _Run, task: Task, outcome: TaskOutcome, observed: Observations) -> TaskResult:
    """The coordinator's result of ``task`` of ``run`` from what it left behind; the record of an
    evaluation holds the request it was asked, its files where the workspace is here."""
    assert run.prepared is not None
    root = run.workspace.root
    result = to_result(task, outcome, observed, root)
    if task.id.kind is not Kind.EVALUATOR:
        return result
    entry = run.prepared.entries[task.id]
    returned, workdir = local_result(outcome, root)
    request = asked(run.specs[task.id], root)
    if workdir is not None:  # it was called, in its own directory
        request = replace(request, workdir=workdir)
    record = EvaluationRecord(
        request, returned, entry.name, entry.version, run.key, outcome.call_seconds
    )
    return replace(result, record=record)


@dataclass(frozen=True, slots=True)
class Activity:
    """What runs at a moment of :meth:`~k4opt.problem.EvaluationSession.evaluate_each`: the
    ``candidates`` under way, and the ``busy`` of the ``capacity`` job slots their running tasks
    hold (a simulation on several threads holds as many)."""

    candidates: int
    busy: int
    capacity: int


type Check = Callable[[_Run, TaskOutcome, TaskResult], None]
"""What the coordinator checks of each result as it comes, raising to stop its candidate."""

_WAIT = 0.2  # seconds the coordinator waits for a task to end before it looks around again


def _drive(
    scheduler: Scheduler,
    backend: Backend,
    runs: dict[str, _Run],
    pull: Callable[[], _Run | None],
    check: Check | None = None,
    activity: Callable[[Activity], None] | None = None,
) -> Generator[_Run]:
    """Run the tasks of candidates until there are none left; yield each candidate once all its
    tasks ended (``results``), or once its work raised (``error``: its other tasks are stopped).

    ``pull`` gives the next candidate, or ``None`` once there is none; it is called only when the
    scheduler is idle, after every candidate that ended was yielded, so with one slot the
    candidates run one after another, each in plan order. ``runs`` holds the candidates under
    way, by workspace. Leaving the loop early (``close()``, an exception) stops every task still
    running. ``activity`` is told what runs whenever that changed.
    """
    pulling = True
    told: Activity | None = None

    def tell() -> None:
        nonlocal told
        now = Activity(len(runs), scheduler.busy, scheduler.capacity)
        if activity is not None and now != told:
            told = now
            activity(now)

    def stop(run: _Run, error: Exception) -> None:
        run.error = error
        key = run.workspace.id
        running = scheduler.drop(key)
        backend.cancel([Handle(key, task) for task in running])
        backend.close_workspace(key)
        runs.pop(key, None)

    try:
        while True:
            ended: list[_Run] = []
            while (item := scheduler.next()) is not None:
                key, task = item
                run = runs[str(key)]
                try:
                    results = scheduler.results(key)
                    spec = _task_spec(run, task, results)
                    run.specs[task.id] = spec
                    backend.submit(spec, _seen(task, results))
                except Exception as exc:
                    stop(run, exc)
                    ended.append(run)
            tell()
            for handle, outcome, observed in backend.wait(_WAIT) if scheduler.running else []:
                found = runs.get(handle.workspace)
                if found is None:  # stopped meanwhile
                    continue
                run = found
                assert run.prepared is not None
                task = run.prepared.binding.plan[handle.task]
                try:
                    if isinstance(outcome, Exception):  # raised where the task ran
                        raise outcome
                    result = _result(run, task, outcome, observed)
                    if check is not None:
                        check(run, outcome, result)
                    scheduler.finished(handle.workspace, result)
                except Exception as exc:
                    stop(run, exc)
                    ended.append(run)
                    continue
                if scheduler.done(handle.workspace):
                    run.results = scheduler.remove(handle.workspace)
                    backend.close_workspace(handle.workspace)
                    del runs[handle.workspace]
                    ended.append(run)
            tell()
            yield from ended
            if pulling and scheduler.idle:
                pulled = pull()
                if pulled is None:
                    pulling = False
                elif pulled.prepared is None:  # it failed before anything was created
                    yield pulled
                else:
                    runs[pulled.workspace.id] = pulled
                    backend.open(pulled.workspace)
                    binding = pulled.prepared.binding
                    costs = {t.id: backend.cost(task_cpus(binding, t)) for t in binding.plan.tasks}
                    scheduler.add(pulled.workspace.id, binding.plan, costs)
                continue
            if not runs:  # and nothing more to pull: a free slot would have pulled
                return
            # each candidate under way has a task running or ready (a free slot takes it)
            assert scheduler.running or not scheduler.idle
    finally:
        for run in list(runs.values()):
            key = run.workspace.id
            backend.cancel([Handle(key, task) for task in scheduler.drop(key)])
            backend.close_workspace(key)
        runs.clear()


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One candidate, measured by every evaluator of its problem.

    ``plan`` holds its tasks, and ``results`` the result of each, in plan order: its geometry,
    the pipeline stages of each sample that its evaluators read, each recorded once, and every
    evaluation, with only its own work. ``elapsed_seconds`` is the wall time of the whole
    evaluation: the tasks and k4opt's checks. Its mappings are read-only copies.

    ``represents_candidate`` is ``False`` when the caller supplied a required stage that the
    parameters change (``k4opt evaluate --input gamma/simulation=FILE``): that file was not made
    from the candidate, so the record must never count as an evaluation of it, whatever its
    status.
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
        """From the required tasks alone: ``failed`` if one failed (the evaluation is
        incomplete), otherwise ``invalid`` if an evaluator found the result unusable (and the
        tasks that read it were not run), otherwise ``valid``. A diagnostic changes nothing."""
        return candidate_status(self.plan, self.results)

    @property
    def message(self) -> str | None:
        """Every required task that failed or is invalid, with its message, then the required
        ones they blocked."""
        return candidate_message(self.plan, self.results)

    @property
    def diagnostics(self) -> tuple[TaskResult, ...]:
        """The result of each task that is not required and did not succeed, in plan order."""
        return tuple(diagnostics(self.plan, self.results))

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
    ``blocked`` and not called, as is a combined evaluator an item of whose roles did not succeed
    (failed, invalid or blocked); the others are still called. An evaluator that raises
    :class:`EvaluationFailed` or :class:`EvaluationInvalid` gets that result; any other
    exception propagates, and so does an :class:`EvaluationContractError` for a result that
    breaks the contract.
    """
    started = time.perf_counter()
    values, key = _validate_candidate(spec, candidate)
    given = dict(inputs or {})
    geometry = given.pop(GEOMETRY.name, None)
    if geometry is not None:  # an input from outside the workspace: named absolutely
        geometry = Path(os.path.abspath(geometry))
    supplied = {}
    for label, path in given.items():
        sample, _, stage = label.partition("/")
        if sample not in spec.samples or stage not in STAGE_TYPES:
            raise EvaluationContractError(
                f"input {label!r}: expected SAMPLE/STAGE, a sample of the problem and one of "
                f"{', '.join(STAGE_TYPES)}"
            )
        supplied[stage_task(stage, sample)] = Path(os.path.abspath(path))
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
        geometry=None if geometry is None else GeometryWork(given=geometry),
        workdir=workdir,
        checked_seconds=checked,
    )


def _execute_plan(
    spec: ProblemSpec,
    prepared: _Prepared,
    key: CandidateKey,
    *,
    supplied: Mapping[TaskId, Path],
    geometry: GeometryWork | None,
    workdir: Path | None,
    checked_seconds: float,
    timeout: float | None = None,
) -> CandidateRecord:
    """Run the tasks of a prepared candidate, one after another in plan order, in this thread;
    evaluator checks and defaults are not repeated.

    ``supplied`` holds the files the caller supplies by stage task, the ones the plan was made with;
    ``geometry`` says how the candidate's geometry is made, and ``None`` gives none (to evaluators
    that need none). Each task writes below ``workdir``, in the directory the plan gives it.
    ``checked_seconds``, the time the candidate's validation and preflight took, counts towards
    ``elapsed_seconds``, which thus covers k4opt's checks too (not what the caller does between).
    ``timeout`` limits each stage that runs a program (:attr:`StageContext.timeout`).
    """
    started = time.perf_counter() - checked_seconds
    binding = prepared.binding
    stages = binding.plan.of(Kind.STAGE)
    check_run_inputs([t.id.name for t in stages if t.given is None], geometry is not None, workdir)
    run = _Run(Workspace("candidate", workdir), prepared, key, supplied, geometry, started)
    backend = InlineBackend(
        _context(spec, binding, timeout), lambda workspace, task: prepared.instances[task]
    )
    pulled = iter([run])
    driven = _drive(Scheduler(backend.capacity), backend, {}, lambda: next(pulled, None))
    with contextlib.closing(driven):
        for ended in driven:
            if ended.error is not None:
                raise ended.error
            return _finish_plan(ended)
    raise AssertionError("the candidate never ended")  # pragma: no cover


def _finish_plan(run: _Run) -> CandidateRecord:
    """The record of a candidate whose tasks all ended. A blocked evaluator's record holds what it
    would have been asked."""
    prepared = run.prepared
    assert prepared is not None
    plan, results = prepared.binding.plan, dict(run.results)
    for task in plan.of(Kind.EVALUATOR):
        held = results[task.id]
        if held.status == Status.BLOCKED:
            entry = prepared.entries[task.id]
            assert held.message is not None
            outcome = EvaluationResult(Status.BLOCKED, message=held.message)
            request = prepared.requests[task.id]
            record = EvaluationRecord(request, outcome, entry.name, entry.version, run.key, None)
            results[task.id] = replace(held, record=record)
    represents = not any(
        changed_by_geometry(task.id.name)
        for task in plan.of(Kind.STAGE)
        if task.given == "supplied" and task.required
    )
    return CandidateRecord(
        run.key, plan, results, _since(run.started), represents, prepared.binding.warnings
    )


def _since(started: float) -> float:
    return time.perf_counter() - started
