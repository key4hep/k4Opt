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
    Scalar,
    check_settings,
    collection_names,
    declared_collections,
    evaluator_source,
    with_defaults,
)
from k4opt.execution._instances import _Bound, _create, _instance, _Probe, _probe, _stage
from k4opt.execution._stages import StageContext, check_stages, no_edm4hep_file
from k4opt.execution._work import Measuring, SessionContext
from k4opt.execution.errors import RunError
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.problem.contract import MetricKey, check_bindings, check_entries
from k4opt.problem.models import EvaluatorSpec, ProblemSpec
from k4opt.problem.plan import (
    Kind,
    Plan,
    Task,
    TaskId,
    evaluations,
    plan_tasks,
    read_stages,
    stage_task,
)


def _bind(
    spec: ProblemSpec, implementations: Sequence[type[EvaluatorProtocol]]
) -> list[tuple[EvaluatorSpec, type[EvaluatorProtocol], EvaluatorProtocol]]:
    """Pair every binding the problem declares with the implementation given for it, in problem
    order, each with the probe instance that decided it.

    A probe of each implementation is matched by its ``name`` to the entries of that name; an
    implementation that is missing, not declared or given twice is an
    :class:`EvaluationContractError`, and so is one that cannot be created. Several entries may
    bind one implementation; they share its probe.
    """
    declared: dict[str, list[EvaluatorSpec]] = {}
    for entry in spec.evaluators:
        declared.setdefault(entry.name, []).append(entry)
    given: dict[str, tuple[type[EvaluatorProtocol], EvaluatorProtocol]] = {}
    for implementation in implementations:
        evaluator = _create(implementation)
        name = evaluator.name
        if name in given:
            raise EvaluationContractError(f"evaluator {name!r} is given twice")
        if name not in declared:
            raise EvaluationContractError(
                f"problem declares no evaluator {name!r}; declared: {', '.join(declared)}"
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
    what its implementation declares, and each binding, objective and constraint against what
    the implementations declare (:mod:`k4opt.problem.contract`). An evaluator that declares
    collections of a stage without an EDM4hep file (the gun's, or a HepMC file) is refused. Raises
    :class:`EvaluationContractError`. Collections the simulation is not expected to write, and
    what is measured or simulated twice, are only warnings.
    """
    provided = provided or {}
    contexts = {name: StageContext.of(spec, name) for name in spec.samples}
    targets = {sample: read_stages(sample, spec.evaluators) for sample in spec.samples}
    probed = _bind(spec, implementations)
    probes: dict[type, _Probe] = {}  # one per implementation
    for _, implementation, evaluator in probed:
        probes.setdefault(implementation, _probe(evaluator))
    declarations = {probe.declaration.name: probe.declaration for probe in probes.values()}
    check_entries(spec, declarations, EvaluationContractError)
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
    demands = check_bindings(spec, declarations, EvaluationContractError)
    warnings = _unexpected_collections(spec, bound, provided) + _identical(spec, bound)
    evaluators = tuple(
        _Bound(entry, implementation, probes[implementation]) for entry, implementation, _ in probed
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
                task: Measuring(bound.entry, bound.probe.declaration, binding.demands.get(task, ()))
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


def _implementations(binding: _Binding) -> dict[str, Mapping[str, Json]]:
    """What a record says of each implementation ``binding`` binds, by name, in first-bound
    order: what the first instance of each declared (:meth:`~k4opt.evaluation.Declaration.to_dict`)
    and the identity of its source."""
    taken: dict[str, Mapping[str, Json]] = {}
    for bound in binding.evaluators:  # what the first instance of each declared
        declaration = bound.probe.declaration
        if declaration.name in taken:
            continue
        data = {**declaration.to_dict(), "source": evaluator_source(bound.probe.cls)}
        frozen = freeze(data, f"evaluator {declaration.name}", RunError)
        assert isinstance(frozen, Mapping)
        taken[declaration.name] = frozen
    return taken
