"""Internal bridge: one candidate of a problem, measured by each of its evaluators.

Not a public workflow. It validates a candidate against the declared parameter domains, binds
the given evaluators to the problem's ``evaluators`` and checks, before anything runs, their
identity, the stage each reads, their settings, the pipeline stages and which evaluator measures
each objective and constraint metric. It then makes every stage the evaluators read available,
running each once (:mod:`k4opt.problem._stages`), calls each evaluator once with the files of the
stages it reads, and checks that each ``valid`` result carries the metrics its evaluator owns.
It neither locates evaluators nor materializes geometry.

Each entry of ``evaluators`` becomes one evaluator instance, identified by its ``name`` today;
records, run directories and metric owners use that identity.
"""

from __future__ import annotations

import math
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

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
    metric_units,
    result_from,
    with_defaults,
)
from k4opt.frozen import FrozenMap
from k4opt.problem._stages import (
    StageContext,
    StageRecords,
    changed_by_geometry,
    check_stages,
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


def _bind(
    spec: ProblemSpec, evaluators: Sequence[EvaluatorProtocol]
) -> list[tuple[EvaluatorSpec, EvaluatorProtocol]]:
    """Pair every evaluator the problem declares with the one given for it, in problem order.

    Evaluators are matched by ``name`` and must have the declared ``version`` and read the
    declared ``stage``; one that is missing, not declared, given twice or reads another stage is
    an :class:`EvaluationContractError`.
    """
    declared = {entry.name: entry for entry in spec.evaluators}
    given: dict[str, EvaluatorProtocol] = {}
    for evaluator in evaluators:
        name, version = evaluator.name, evaluator.version  # read once: the record reuses them
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
        given[name] = evaluator
    missing = [
        f"{entry.name!r} version {entry.version!r}"
        for entry in spec.evaluators
        if entry.name not in given
    ]
    if missing:
        raise EvaluationContractError(f"no evaluator given for {', '.join(missing)}")
    return [(entry, given[entry.name]) for entry in spec.evaluators]


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


@dataclass(frozen=True, slots=True)
class _Plan:
    """What the preflight established before anything runs."""

    context: StageContext
    evaluators: tuple[tuple[EvaluatorSpec, EvaluatorProtocol], ...]  # problem order
    targets: tuple[str, ...]  # the pipeline stages the evaluators read
    owners: Mapping[MetricKey, str]  # objective or constraint metric -> evaluator
    declared: Mapping[MetricKey, str]  # any declared metric -> evaluator


def _preflight(
    spec: ProblemSpec,
    values: Mapping[str, Scalar],
    evaluators: Sequence[EvaluatorProtocol],
    provided: Collection[str] = (),
) -> _Plan:
    """Check everything that can be checked before a file is created or a stage runs.

    Binds ``evaluators`` to the problem, checks the stages they read (those in ``provided``, the
    stages the caller supplies, are not run), calls each evaluator's ``check()`` with the request
    it would get and decides which evaluator measures each objective and constraint metric.
    Raises :class:`EvaluationContractError` (or what a ``check()`` raises).
    """
    context = StageContext.of(spec)
    bound = _bind(spec, evaluators)
    targets = tuple(stage for _, ev in bound if (stage := _stage(ev)) is not None)
    check_stages(context, *targets, provided=provided)
    for entry, evaluator in bound:
        check = getattr(evaluator, "check", None)
        if check is not None:
            check(_request(context, values, entry, evaluator))
    owners, declared = _owners(spec, bound)
    return _Plan(context, tuple(bound), targets, owners, declared)


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
    inputs: Mapping[str, Path] | None = None,
    workdir: Path | None = None,
) -> EvaluationRequest:
    """The request an evaluator receives, for ``check()`` and ``evaluate()`` alike.

    ``settings`` are those of its ``entry`` in the problem plus the defaults the evaluator
    declares; ``events`` and ``seed`` are those of the stage it reads.
    """
    events, seed = context.conditions(entry.stage)
    return EvaluationRequest(
        candidate=values,
        settings=with_defaults(evaluator, entry.settings),
        inputs=inputs or {},
        workdir=workdir,
        events=events,
        seed=seed,
    )


def _measure(
    spec: ProblemSpec,
    plan: _Plan,
    values: Mapping[str, Scalar],
    key: CandidateKey,
    entry: EvaluatorSpec,
    evaluator: EvaluatorProtocol,
    stages: StageRecords,
    inputs: Mapping[str, Path],
    workdir: Path | None,
) -> EvaluationRecord:
    """The record of one evaluator, called once on the files of the stages it reads.

    An evaluator whose stage cannot be read gets that stage's error as a ``failed`` result and
    is not called. Its own directory, ``workdir/evaluators/<name>``, is created just before the
    call; ``elapsed_seconds`` is the time of the call alone.
    """
    stage = _stage(evaluator)
    given = {**inputs, **stages.inputs(stage)}
    error = stages.error(stage)
    if error is not None:
        request = _request(plan.context, values, entry, evaluator, given)
        return _record(spec, plan, entry, evaluator, key, request, EvaluationResult.failed(error))
    own = None if workdir is None else workdir / "evaluators" / entry.name
    request = _request(plan.context, values, entry, evaluator, given, own)
    if own is not None:
        own.mkdir(parents=True)
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
    """

    candidate_key: CandidateKey
    records: tuple[EvaluationRecord, ...]
    stages: StageRecords
    owners: Mapping[MetricKey, str]
    reads: Mapping[str, tuple[str, ...]]
    elapsed_seconds: float
    represents_candidate: bool = True

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
        statuses = {record.result.status for record in self.records}
        for status in (Status.FAILED, Status.INVALID):
            if status in statuses:
                return status
        return Status.VALID

    @property
    def message(self) -> str | None:
        """Every evaluator that is not ``valid``, with its status and message, in problem order."""
        reasons = [
            f"{record.evaluator_name} {record.result.status}: {record.result.message}"
            for record in self.records
            if record.result.status is not Status.VALID
        ]
        return "; ".join(reasons) or None

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
    evaluators: Sequence[EvaluatorProtocol],
    *,
    inputs: Mapping[str, Path] | None = None,
    workdir: Path | None = None,
) -> CandidateRecord:
    """Evaluate one complete ``candidate`` of ``spec`` with each of ``evaluators``, once.

    The candidate maps each canonical parameter reference to a scalar; ``evaluators`` implement
    the problem's ``evaluators``, in any order. Before any stage runs, raises
    :class:`CandidateError` for a candidate outside the declared domains and
    :class:`EvaluationContractError` for evaluators that do not match the problem (missing,
    undeclared, given twice, another version), settings an evaluator's ``check()`` rejects, a
    stage that cannot run, or an objective or constraint metric that no evaluator, or two,
    declare. The stages the evaluators read and ``inputs`` does not supply then run once each
    (on ``inputs["geometry"]``, in ``workdir``); any other entry of ``inputs`` supplies the file
    of the stage it is named after. An evaluator whose stage failed, or whose stage
    output lacks the requested events, gets a ``failed`` result without being called; the others
    are still called. An evaluator that raises :class:`EvaluationFailed` or
    :class:`EvaluationInvalid` gets that result; any other exception propagates, and so does an
    :class:`EvaluationContractError` for a result that breaks the contract.
    """
    started = time.perf_counter()
    values, key = _validate_candidate(spec, candidate)
    inputs = dict(inputs or {})
    plan = _preflight(spec, values, evaluators, inputs)
    stages = run_stages(plan.context, *plan.targets, inputs=inputs, workdir=workdir)
    records = tuple(
        _measure(spec, plan, values, key, entry, evaluator, stages, inputs, workdir)
        for entry, evaluator in plan.evaluators
    )
    reads = {
        entry.name: tuple(stages.chain(_stage(evaluator))) for entry, evaluator in plan.evaluators
    }
    represents = not any(changed_by_geometry(name) for name in stages.supplied)
    elapsed = time.perf_counter() - started
    return CandidateRecord(key, records, stages, plan.owners, reads, elapsed, represents)
