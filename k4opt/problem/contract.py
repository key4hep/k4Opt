"""What a problem asks of the implementations that measure it, checked one way live and from a
record.

A session checks each binding of the problem against the
:class:`~k4opt.evaluation.Declaration` of the implementation it is given, once
(:func:`check_entries`, :func:`check_bindings`), and each result of an evaluation as it comes
(:func:`check_result`). Reading a record back checks the declarations and results the record
holds with the same functions, without the implementation. Each caller passes its own error
(:class:`~k4opt.evaluation.EvaluationContractError` live, a record error when read back), so a
record holds up only where a live evaluation would have accepted what it says. What only an
instance has (the kinds of its settings, its collections, that each instance declares what the
first did) is checked live alone, and what only a record has (where its files are, how its tasks
ended) when it is read.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from k4opt.errors import K4OptError
from k4opt.evaluation import Declaration, Measurement, Status
from k4opt.problem._tasks import TaskId, reference_task
from k4opt.problem.models import EvaluatorSpec, ProblemSpec

type MetricKey = tuple[str, str]
"""A metric as an evaluator declares and measures it: ``(stage, metric)``."""

type Demands = Mapping[TaskId, tuple[MetricKey, ...]]
"""The metrics the roles of combined evaluators read, by the evaluation that measures them."""


def show(metrics: Collection[MetricKey]) -> str:
    return ", ".join(f"{stage}/{metric}" for stage, metric in metrics)


def _as(entry: EvaluatorSpec) -> str:
    """`` (as b)`` for an entry whose binding is not its implementation's name, else nothing."""
    return "" if entry.binding == entry.name else f" (as {entry.binding!r})"


def check_entries(
    spec: ProblemSpec, declarations: Mapping[str, Declaration], error: type[K4OptError]
) -> None:
    """Each binding of the problem declares the version of its implementation
    (``declarations``, by name), reads the stage it reads, and gives roles exactly to a combined
    evaluator (one that declares roles). Raises ``error``."""
    for entry in spec.evaluators:
        declaration = declarations[entry.name]
        name, version, stage = entry.name, declaration.version, declaration.stage
        if version != entry.version:
            raise error(
                f"problem declares evaluator {name!r}{_as(entry)} version {entry.version!r}, "
                f"got version {version!r}"
            )
        if stage != entry.stage:
            if entry.stage is None:
                fix = f"add stage: {stage} to its entry in the problem"
            elif stage is None:
                fix = f"it reads no stage; remove stage: {entry.stage} from its entry"
            else:
                fix = f"it reads {stage!r}, not {entry.stage!r}"
            raise error(
                f"evaluator {name!r}{_as(entry)} does not read the stage the problem says: {fix}"
            )
        if entry.roles and not declaration.roles:
            raise error(
                f"evaluator {name!r}{_as(entry)} declares no roles, so it is no combined "
                "evaluator; remove roles: from its entry"
            )
        if declaration.roles and not entry.roles:
            raise error(
                f"evaluator {name!r}{_as(entry)} is a combined evaluator with the roles "
                f"{', '.join(declaration.roles)}; say what each reads under roles: in its entry"
            )


def check_bindings(
    spec: ProblemSpec, declarations: Mapping[str, Declaration], error: type[K4OptError]
) -> dict[TaskId, tuple[MetricKey, ...]]:
    """Each objective and constraint names a metric its evaluator measures, if that one declares
    its ``metrics``, and a constraint ``unit`` is the unit it declares for it; otherwise the
    result decides. Each combined evaluator gets for its roles what its implementation
    declares: every role and no other, one item unless the role reads ``many``, evaluations
    where the role reads them (of evaluators that declare the metrics and artifacts it reads)
    and the files of its stage otherwise. Returns the metrics the roles read of each evaluation,
    which a ``valid`` result of that evaluation must carry. Raises ``error``."""
    of = {entry.binding: declarations[entry.name] for entry in spec.evaluators}
    for ref in spec.references:
        metrics = of[ref.evaluator].metrics
        if metrics is not None and (ref.stage, ref.metric) not in metrics:
            raise error(
                f"evaluator {ref.evaluator!r} does not measure {ref.stage}/{ref.metric}; "
                f"supported: {show(sorted(metrics))}"
            )
    for constraint in spec.constraints:
        metrics = of[constraint.evaluator].metrics
        if metrics is None or constraint.unit is None:
            continue
        unit = metrics[(constraint.stage, constraint.metric)]
        if constraint.unit != unit:
            raise error(
                f"constraint on {constraint.label} declares unit {constraint.unit!r} but "
                f"evaluator {constraint.evaluator!r} measures it in {unit!r}"
            )
    demands: dict[TaskId, dict[MetricKey, None]] = {}
    for entry in spec.evaluators:
        declared = declarations[entry.name].roles
        if not entry.roles or declared is None:
            continue  # check_entries: roles exactly for a combined evaluator
        where = f"evaluator {entry.binding!r}"
        unknown = [role for role in entry.roles if role not in declared]
        missing = [role for role in declared if role not in entry.roles]
        if unknown or missing:
            raise error(
                f"{where}: its roles are {', '.join(declared)}"
                + (f"; unknown {', '.join(unknown)}" if unknown else "")
                + (f"; missing {', '.join(missing)}" if missing else "")
            )
        for name, role in declared.items():
            items, at = entry.roles[name], f"{where}: role {name!r}"
            if not role.many and len(items) > 1:
                raise error(
                    f"{at} reads one item, but the problem gives {len(items)}: "
                    f"{', '.join(item.label for item in items)}"
                )
            for item in items:
                if role.stage is not None:
                    if item.stage != role.stage:
                        raise error(
                            f"{at} reads the {role.stage} file of a sample, as "
                            f"{{sample: ..., stage: {role.stage}}}; got {item.label}"
                        )
                    continue
                if item.evaluator is None:
                    raise error(
                        f"{at} reads evaluations, as {{sample: ..., evaluator: ...}}; got the "
                        f"stage {item.label}"
                    )
                upstream = of[item.evaluator]
                undeclared = [key for key in role.metrics if key not in (upstream.metrics or {})]
                if undeclared:
                    raise error(
                        f"{at} reads {show(undeclared)} of {item.label}, but {upstream.name} "
                        f"does not declare {'it' if len(undeclared) == 1 else 'them'}"
                    )
                absent = [name for name in role.artifacts if name not in (upstream.artifacts or ())]
                if absent:
                    raise error(
                        f"{at} reads the artifact(s) {', '.join(absent)} of {item.label}, but "
                        f"{upstream.name} does not declare {'it' if len(absent) == 1 else 'them'}"
                    )
                demands.setdefault(reference_task(item), {}).update(dict.fromkeys(role.metrics))
    return {task: tuple(keys) for task, keys in demands.items()}


def check_result(
    spec: ProblemSpec,
    task: TaskId,
    declaration: Declaration,
    demanded: Sequence[MetricKey],
    status: Status,
    measurements: Sequence[Measurement],
    artifacts: Collection[str],
    error: type[K4OptError],
) -> None:
    """The result of the evaluation ``task``, by the implementation ``declaration`` says,
    returned with ``status``, ``measurements`` and the ``artifacts`` named: every measurement
    of a metric it declares carries the declared unit, whatever the status (diagnostic values of
    an ``invalid`` result are recorded too), and it returns no artifact it does not declare. A
    ``valid`` one carries every metric an objective or a constraint names of it, in the unit a
    constraint declares, every metric the roles that read it ``demanded`` and every artifact it
    declares. Raises ``error``."""
    declared = declaration.metrics
    if declared is not None:
        for m in measurements:
            unit = declared.get((m.stage, m.metric))
            if (m.stage, m.metric) in declared and m.unit != unit:
                raise error(
                    f"measurement {m.stage}/{m.metric} has unit {m.unit!r} but evaluator "
                    f"{declaration.name!r} declares {unit!r}"
                )
    if status is Status.VALID:
        measured = {(m.stage, m.metric): m for m in measurements}
        owned = [(r.stage, r.metric) for r in spec.references if r.invocation == task.label]
        wanted = dict.fromkeys([*owned, *demanded])
        missing = [key for key in wanted if key not in measured]
        if missing:
            raise error(
                f"evaluator {task.label!r} returned a valid result without required metric(s): "
                f"{show(missing)}"
            )
        for constraint in spec.constraints:
            if constraint.ref.invocation != task.label:
                continue
            unit = measured[(constraint.stage, constraint.metric)].unit
            if constraint.unit is not None and unit != constraint.unit:
                raise error(
                    f"constraint on {constraint.label} declares unit {constraint.unit!r} but the "
                    f"measurement has unit {unit!r}"
                )
    names = declaration.artifacts
    if names is not None:
        where = f"evaluator {task.label!r}"
        unknown = [name for name in artifacts if name not in names]
        if unknown:
            raise error(
                f"{where} returned the artifact(s) {', '.join(unknown)}, which it does not "
                f"declare; it declares {', '.join(names) or 'none'}"
            )
        lacking = [name for name in names if name not in artifacts]
        if lacking and status is Status.VALID:
            raise error(
                f"{where} returned a valid result without its declared artifact(s) "
                f"{', '.join(lacking)}"
            )
