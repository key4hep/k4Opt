"""Internal: the evaluator implementations bound to a problem, once per session.

:func:`_bind_problem` checks what does not depend on a candidate: it binds the given evaluator
classes to the problem's ``evaluators`` by what each declares, read without creating it or running
its code, and checks the stage each reads, their settings and collections, each sample's pipeline
stages and that each objective and constraint names a metric its evaluator measures
(:class:`_Binding`). Each candidate is then prepared from that by
:mod:`k4opt.execution._preflight`. :func:`_context` is what every task of the session needs
wherever it runs, and :func:`_implementations` what a record says of each implementation.

An entry of ``evaluators`` is a binding of an implementation, an evaluator class, identified by
its binding name (``as:``, the implementation's ``name`` when not given): tasks, records, run
directories and metric references use it, with the sample for an evaluation of one
(``gamma/calo.photon``). One implementation may be bound several times, with other settings. A
combined evaluator is bound to the candidate and reads, by role, what the problem names: the
evaluations of other samples, given to it as :class:`~k4opt.evaluation.RoleInput`, or their stage
files; it runs only once all of them succeeded (``valid``), and every ``valid`` result it reads
carries the metrics its roles name. An implementation is bound to its entries once, by what its
class declares; each evaluation of each candidate gets a new instance of it, created where the
evaluation runs, so no state of an instance reaches another sample, binding or candidate.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from k4opt.evaluation import (
    EvaluationContractError,
    EvaluatorProtocol,
    check_bound,
    check_settings,
    collection_names,
    declared_collections,
    evaluator_source,
    with_defaults,
)
from k4opt.execution._instances import _Bound, _Declared, _declared, _stage
from k4opt.execution._stages import (
    StageContext,
    check_stages,
    expected_collections,
    no_edm4hep_file,
    stage_cpus,
)
from k4opt.execution._tasks import Measuring, SessionContext
from k4opt.execution.errors import RunError
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.geometry import Catalog, GeometrySource
from k4opt.problem.contract import MetricKey, check_bindings, check_entries
from k4opt.problem.models import (
    EvaluatorSpec,
    Parameter,
    ProblemSpec,
    SettingDestination,
    StageDestination,
)
from k4opt.problem.plan import (
    Kind,
    Plan,
    Task,
    TaskId,
    evaluations,
    plan_tasks,
    read_stages,
    rejecting,
    stage_task,
)


def _bind(
    spec: ProblemSpec, implementations: Sequence[type[EvaluatorProtocol]]
) -> list[tuple[EvaluatorSpec, _Declared]]:
    """Pair every binding the problem declares with what the class given for it declares, in
    problem order.

    Each class is matched by the ``name`` it declares to the entries of that name; a class that
    is missing, not declared or given twice is an :class:`EvaluationContractError`, and so is one
    that is no evaluator class or declares anything but plain data
    (:func:`~k4opt.evaluation.check_class`). Nothing is created. Several entries may bind one
    class; they share what it declares.
    """
    declared: dict[str, list[EvaluatorSpec]] = {}
    for entry in spec.evaluators:
        declared.setdefault(entry.name, []).append(entry)
    given: dict[str, _Declared] = {}
    for implementation in implementations:
        found = _declared(implementation)
        name = found.declaration.name
        if name in given:
            raise EvaluationContractError(f"evaluator {name!r} is given twice")
        if name not in declared:
            raise EvaluationContractError(
                f"problem declares no evaluator {name!r}; declared: {', '.join(declared)}"
            )
        given[name] = found
    missing = [
        f"{name!r} version {entries[0].version!r}"
        for name, entries in declared.items()
        if name not in given
    ]
    if missing:
        raise EvaluationContractError(f"no evaluator given for {', '.join(missing)}")
    return [(entry, given[entry.name]) for entry in spec.evaluators]


def _identical(
    spec: ProblemSpec, evaluators: Sequence[tuple[EvaluatorSpec, type]]
) -> tuple[str, ...]:
    """A warning for each binding that binds an implementation as an earlier one does on the same
    sample, or on the candidate: with the same settings (their defaults filled in), collections
    and roles, it measures the same twice. And one for each sample whose stages are those of an
    earlier one: its events are simulated again."""
    keys = {  # by entry: one binding name may bind its samples with other settings
        id(entry): (
            entry.name,
            # with the parameters that set its settings: each sets them anew for a candidate
            _canonical(
                {**with_defaults(evaluator, entry.settings), **_set_by(spec, entry.binding)}
            ),
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
        # with the parameter that sets each option: two samples alike but for which parameter
        # sets an option are simulated alike for no candidate
        set_by = sorted(
            (destination.stage, destination.option, parameter.name)
            for parameter, destination in spec.destinations
            if isinstance(destination, StageDestination) and name in destination.samples
        )
        stages = (tuple(sample.stages.items()), tuple(set_by))
        if stages in first:
            warnings[
                f"samples {first[stages]} and {name} have identical stages: they are simulated "
                "once and measure the same events"
            ] = None
        else:
            first[stages] = name
    return tuple(warnings)


def _set_by(spec: ProblemSpec, binding: str) -> dict[str, str]:
    """The settings of ``binding`` that parameters set, each with its parameter, as a setting
    of its own: ``{key: "parameters.<name>"}``."""
    return {
        destination.setting: f"parameters.{parameter.name}"
        for parameter, destination in spec.destinations
        if isinstance(destination, SettingDestination) and destination.evaluator == binding
    }


def _values(parameter: Parameter) -> type:
    """The type of the values of ``parameter``: ``float`` or ``int`` for a number, ``str`` or
    ``bool`` for choices of either."""
    if parameter.kind != "categorical":
        return int if parameter.kind == "integer" else float
    choices = parameter.choices or ()
    if all(isinstance(choice, bool) for choice in choices):
        return bool
    if all(isinstance(choice, str) for choice in choices):
        return str
    return int if all(isinstance(choice, int) for choice in choices) else float


def pending_settings(spec: ProblemSpec, binding: str) -> dict[str, tuple[type, str | None]]:
    """The settings of ``binding`` that parameters set, each with the type of its parameter's
    values and its unit (:func:`~k4opt.evaluation.check_settings`)."""
    return {
        destination.setting: (_values(parameter), parameter.unit)
        for parameter, destination in spec.destinations
        if isinstance(destination, SettingDestination) and destination.evaluator == binding
    }


def _canonical(settings: Mapping[str, Json]) -> str:
    """``settings`` as text that is the same for equal settings, whatever the key order of any
    nested mapping; a default that is not JSON is written as its ``repr``."""
    return json.dumps(thaw(settings), sort_keys=True, default=repr)


def _unexpected_collections(
    spec: ProblemSpec,
    catalog: Catalog | None,
    evaluators: Sequence[tuple[EvaluatorSpec, type]],
    provided: Collection[TaskId],
) -> tuple[str, ...]:
    """A warning for each collection an evaluator reads from a stage of a sample that its
    executor is not expected to write for the problem's geometry (``catalog``, none without
    one; :func:`~k4opt.execution._stages.expected_collections`), with the closest expected
    name; once per binding, however many of its samples run that stage.

    Only a warning: for ddsim, the prediction comes from the unmodified geometry's readouts, and
    custom sensitive actions can name collections of their own. The stage's output decides.
    """
    if catalog is None:
        return ()
    warnings: dict[str, None] = {}
    for entry, evaluator in evaluators:
        read = _stage(evaluator)
        if read is None:
            continue
        for sample in entry.samples:
            stage = spec.samples[sample].stages.get(read)
            # a supplied file is checked itself, before it is read
            if stage is None or stage_task(read, sample) in provided:
                continue
            expected = expected_collections(read, stage, catalog)
            if expected is None:
                continue
            for role, name in collection_names(evaluator, entry.collections).items():
                if name in expected:
                    continue
                close = difflib.get_close_matches(name, expected, n=1)
                hint = f"; did you mean {close[0]!r}?" if close else "."
                warning = (
                    f"evaluator {entry.binding!r}: {stage.executor} is not expected to write the "
                    f"collection {name!r} ({role}) for {spec.geometry}{hint} The {read}'s output "
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
    # seconds a stage that runs or an evaluation may take unless it sets its own: the problem's
    timeout: float | None = None
    # the tasks that may reject a candidate before anything of it is created (plan.rejecting)
    rejecting: tuple[TaskId, ...] = ()


def _bind_problem(
    spec: ProblemSpec,
    implementations: Sequence[type[EvaluatorProtocol]],
    provided: Mapping[TaskId, Path] | None = None,
    *,
    directory: Path | None = None,
    catalog: Catalog | None = None,
) -> _Binding:
    """Check what does not depend on a candidate, before a file is created or a stage runs.

    ``directory`` is the problem file's, against which a relative stage ``file`` is read;
    ``catalog`` the problem's geometry, from which the collections ddsim writes are predicted.

    Binds ``implementations`` to the problem by what each class declares, without creating any:
    checks the stages of each sample they read (those in ``provided``, the files the caller
    supplies by stage task, are not run), each binding's settings and collections against what its
    implementation declares, and each binding, objective and constraint against what the
    implementations declare (:mod:`k4opt.problem.contract`). What the parameters set is pending in
    both: a stage option, as its executor will check it once a candidate gives it, and a setting,
    which the implementation must declare, of a kind the parameter's values fit. An evaluator that
    declares collections of a stage without an EDM4hep file (the gun's, or a HepMC file) is refused.
    Raises :class:`EvaluationContractError`. Collections the simulation is not expected to write,
    and what is measured or simulated twice, are only warnings."""
    provided = provided or {}
    contexts = {name: StageContext.of(spec, name, directory) for name in spec.samples}
    targets = {sample: read_stages(sample, spec.evaluators) for sample in spec.samples}
    probed = _bind(spec, implementations)
    declarations = {found.declaration.name: found.declaration for _, found in probed}
    check_entries(spec, declarations, EvaluationContractError)
    bound: list[tuple[EvaluatorSpec, type]] = [(entry, found.cls) for entry, found in probed]
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
        # what needs no candidate is checked before any: a check() may depend on the candidate;
        # a setting a parameter sets, first, so that a refusal names that parameter
        for parameter, destination in spec.destinations:
            if (
                isinstance(destination, SettingDestination)
                and destination.evaluator == entry.binding
            ):
                try:
                    check_bound(evaluator, destination.setting, _values(parameter), parameter.unit)
                except EvaluationContractError as exc:
                    raise EvaluationContractError(
                        f"parameters.{parameter.name}.targets.evaluators.{entry.binding}: setting "
                        f"{destination.setting!r} {exc}"
                    ) from None
        try:
            check_settings(
                evaluator,
                with_defaults(evaluator, entry.settings),
                pending_settings(spec, entry.binding),
            )
        except EvaluationContractError as exc:
            raise EvaluationContractError(f"evaluator {entry.binding!r}: {exc}") from None
        collection_names(evaluator, entry.collections)
    demands = check_bindings(spec, declarations, EvaluationContractError)
    warnings = _unexpected_collections(spec, catalog, bound, provided) + _identical(spec, bound)
    evaluators = tuple(_Bound(entry, found) for entry, found in probed)
    of = {id(bound.entry): bound for bound in evaluators}
    planned = tuple(
        (task, of[id(entry)]) for task, entry in evaluations(spec.samples, spec.evaluators)
    )
    plan = plan_tasks(
        spec.samples,
        spec.evaluators,
        provided,
        references=spec.references,
        parameters=spec.parameters,
    )
    return _Binding(
        contexts,
        evaluators,
        planned,
        FrozenMap(targets),
        plan,
        warnings,
        FrozenMap(demands),
        FrozenMap({name: sample.properties for name, sample in spec.samples.items()}),
        None if spec.timeout is None else float(spec.timeout),
        rejecting(plan, spec.samples, spec.parameters),
    )


def _context(
    spec: ProblemSpec,
    binding: _Binding,
    externals: Mapping[str, Json] | None = None,
    source: GeometrySource | None = None,
    catalog: Catalog | None = None,
) -> SessionContext:
    """What every task of the candidates of ``binding`` needs (:class:`SessionContext`), with the
    session geometry's ``externals`` and the geometry candidates start from (``source``,
    discovered as ``catalog``). Its stage conditions hold no stages: each stage task brings those
    of its candidate, and its time limit."""
    return SessionContext(
        spec,
        FrozenMap({name: c.configured(FrozenMap({})) for name, c in binding.contexts.items()}),
        FrozenMap(
            {
                task: Measuring(
                    bound.entry, bound.declared.declaration, binding.demands.get(task, ())
                )
                for task, bound in binding.evaluations
            }
        ),
        FrozenMap(externals or {}),
        source,
        catalog,
    )


def task_cpus(contexts: Mapping[str, StageContext], task: Task) -> int:
    """How many processors ``task`` keeps busy, with the stages of ``contexts`` (by sample): a
    stage that runs as many as its runner says (:func:`~k4opt.execution._stages.stage_cpus`),
    one for every other task."""
    if task.id.kind is not Kind.STAGE or task.given is not None:
        return 1
    assert task.id.sample is not None
    return stage_cpus(task.id.name, contexts[task.id.sample].stages[task.id.name])


def _implementations(binding: _Binding) -> dict[str, Mapping[str, Json]]:
    """What a record says of each implementation ``binding`` binds, by name, in first-bound
    order: what its class declared (:meth:`~k4opt.evaluation.Declaration.to_dict`) and the
    identity of its source."""
    taken: dict[str, Mapping[str, Json]] = {}
    for bound in binding.evaluators:
        declaration = bound.declared.declaration
        if declaration.name in taken:
            continue
        data = {**declaration.to_dict(), "source": evaluator_source(bound.implementation)}
        frozen = freeze(data, f"evaluator {declaration.name}", RunError)
        assert isinstance(frozen, Mapping)
        taken[declaration.name] = frozen
    return taken
