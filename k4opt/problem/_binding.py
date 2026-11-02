"""Internal: the evaluator implementations bound to a problem, and each candidate's evaluations
prepared before anything runs.

:func:`_bind_problem` checks what does not depend on a candidate, once per session: it binds the
given evaluator implementations to the problem's ``evaluators`` and checks their identity, the
stage each reads, their settings and collections, each sample's pipeline stages and that each
objective and constraint names a metric its evaluator measures (:class:`_Binding`).
:func:`_preflight` then prepares one candidate: a new instance of each evaluation's
implementation, whose ``check()`` accepted the request it gets for this candidate
(:class:`_Prepared`). :func:`_context` is what every task of the session needs wherever it
runs, and :func:`_implementations` what a record says of each implementation.

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
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from k4opt.ddsim import output_collections, threads
from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationRequest,
    EvaluatorProtocol,
    Role,
    Scalar,
    check_settings,
    collection_names,
    declared_artifacts,
    declared_collections,
    declared_roles,
    evaluator_source,
    metric_units,
    with_defaults,
)
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.problem._instances import _Bound, _create, _Declaration, _declaration, _instance, _stage
from k4opt.problem._stages import StageContext, check_stages, no_edm4hep_file
from k4opt.problem._tasks import (
    Kind,
    Plan,
    Task,
    TaskId,
    evaluations,
    plan_tasks,
    read_stages,
    reference_task,
    stage_task,
)
from k4opt.problem._work import Measuring, MetricKey, SessionContext, show
from k4opt.problem.errors import RunError
from k4opt.problem.models import EvaluatorSpec, ProblemSpec


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


def task_cpus(binding: _Binding, task: Task) -> int:
    """How many processors ``task`` keeps busy: the threads of a simulation ddsim runs
    (``numberOfThreads``), one for every other task."""
    if task.id.kind is not Kind.STAGE or task.given is not None or task.id.name != "simulation":
        return 1
    assert task.id.sample is not None
    stage = binding.contexts[task.id.sample].stages["simulation"]
    return threads(stage.options) if stage.executor == "ddsim" else 1


def _declared(units: Mapping[MetricKey, str] | None) -> list[dict[str, str]] | None:
    """The metrics an evaluator declares (``units``), each with its unit; ``None`` if it declares
    none."""
    if units is None:
        return None
    return [{"stage": s, "metric": m, "unit": unit} for (s, m), unit in units.items()]


def _roles(roles: Mapping[str, Role] | None) -> dict[str, dict[str, Json]] | None:
    """The roles a combined evaluator declares, each with what it reads; ``None`` if it declares
    none."""
    if roles is None:
        return None
    return {
        name: {
            "many": role.many,
            "stage": role.stage,
            "metrics": tuple(role.metrics),
            "artifacts": tuple(role.artifacts),
        }
        for name, role in roles.items()
    }


def _implementations(binding: _Binding) -> dict[str, Mapping[str, Json]]:
    """What a record says of each implementation ``binding`` binds, by name, in first-bound
    order: what the first instance of each declared (its version, stage, metrics, artifacts
    and roles) and the identity of its source."""
    taken: dict[str, Mapping[str, Json]] = {}
    for bound in binding.evaluators:  # what the first instance of each declared
        declaration = bound.declaration
        if declaration.name in taken:
            continue
        data = {
            "version": declaration.version,
            "stage": declaration.stage,
            "source": evaluator_source(declaration.cls),
            # each with the unit every measurement of it carries, whatever the status
            "metrics": _declared(declaration.metrics),
            # the names of the files a valid result of it returns; null if it declares none
            "artifacts": None if declaration.artifacts is None else list(declaration.artifacts),
            # what each role of a combined evaluator reads; null for an evaluator of no roles
            "roles": _roles(declaration.roles),
        }
        frozen = freeze(data, f"evaluator {declaration.name}", RunError)
        assert isinstance(frozen, Mapping)
        taken[declaration.name] = frozen
    return taken
