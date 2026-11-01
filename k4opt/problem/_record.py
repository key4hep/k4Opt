"""Internal: a candidate's ``summary.json``, read back and checked before anything trusts it.

:func:`check_summary` checks a record against its problem as
:meth:`~k4opt.problem.Assessment.from_summary` documents: its form, that it was made for this
problem and candidate, its implementations and roles, each task's part against the plan the
problem gives (what it reads, whether and by what it is blocked, the files it made, named
relative to the run directory in its own directory), and the candidate's status, message,
required tasks, diagnostics and metrics recomputed from them. It returns what the candidate's
assessment is made from, which the caller judges as a live evaluation is judged. It reads the
record alone: no file the record names is opened, so an old record stays readable without them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple, cast

from k4opt.evaluation import (
    ARTIFACT_NAME,
    Artifact,
    EvaluationContractError,
    Measurement,
    Scalar,
    Status,
)
from k4opt.frozen import thaw
from k4opt.problem._candidate import _validate_candidate
from k4opt.problem._stages import STAGE_TYPES, changed_by_geometry, stage_output
from k4opt.problem._tasks import (
    DONE,
    GEOMETRY,
    Kind,
    Plan,
    TaskId,
    TaskResult,
    blocked,
    candidate_message,
    candidate_status,
    diagnostics,
    evaluations,
    evaluator_task,
    plan_tasks,
    stage_task,
)
from k4opt.problem._work import MetricKey
from k4opt.problem.errors import CandidateError, RecordError
from k4opt.problem.models import MetricRef, ProblemSpec

SUMMARY_SCHEMA = "k4opt.run/1"


type _Lookup = Callable[[MetricRef], Measurement | None]


def _measurement(m: Measurement | None) -> dict[str, object]:
    return {
        "value": None if m is None else m.value,
        "unit": None if m is None else m.unit,
        "uncertainty": None if m is None else m.uncertainty,
    }


def _task(ref: MetricRef) -> TaskId:
    """The evaluation that measures the metric ``ref``."""
    return evaluator_task(ref.evaluator, ref.sample)


def _plain(value: object) -> object:
    """``value`` as it reads back from JSON."""
    return json.loads(json.dumps(value, allow_nan=False))


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecordError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise RecordError(f"{where} must be a list, got {type(value).__name__}")
    return value


def _check_problem(spec: ProblemSpec, record: Mapping[str, object]) -> None:
    """The record must have been made for ``spec``; its ``name`` (a label) may differ."""
    definition = _mapping(
        _mapping(record.get("problem"), "problem").get("definition"), "definition"
    )
    expected = _plain(spec.to_dict())
    assert isinstance(expected, dict)
    keys = dict.fromkeys([*expected, *definition])
    differ = [k for k in keys if k != "name" and definition.get(k) != expected.get(k)]
    if differ:
        raise RecordError(f"it was made for another problem: its {', '.join(differ)} differ")


class _Fragment(NamedTuple):
    """A task's part of a record, as far as its status reads it."""

    status: str
    message: str | None
    reads: list[object]
    blocked_by: list[object]


def _fragment(entry: Mapping[str, object], where: str, statuses: Collection[str]) -> _Fragment:
    status = entry.get("status")
    if not isinstance(status, str) or status not in statuses:
        raise RecordError(f"{where}: status {status!r} is none of {', '.join(statuses)}")
    message = entry.get("message")
    if message is not None and not isinstance(message, str):
        raise RecordError(f"{where}: message must be a string, got {message!r}")
    if status not in (DONE, Status.VALID) and not (message and message.strip()):
        raise RecordError(f"{where}: a task that is {status} needs a message saying why")
    reads = _list(entry.get("reads"), f"{where}: reads")
    causes = _list(entry.get("blocked_by"), f"{where}: blocked_by")
    return _Fragment(status, message, reads, causes)


@dataclass(frozen=True, slots=True)
class _Evaluator:
    """One evaluation of a record, as far as the assessment reads it."""

    status: Status
    message: str | None
    measured: Mapping[MetricKey, Measurement]  # its own measurements
    declared: Mapping[MetricKey, str] | None  # the metrics its implementation declares, with units
    fragment: _Fragment  # its status, and the tasks it reads


def _check_candidate(spec: ProblemSpec, record: Mapping[str, object]) -> None:
    """The record's candidate is one of the problem's, and its ``candidate_key`` is that
    candidate's, as a live evaluation validates and keys it."""
    candidate = _mapping(record.get("candidate"), "candidate")
    try:  # the values are checked for their types and domains here
        _, key = _validate_candidate(spec, cast(Mapping[str, Scalar], candidate))
    except CandidateError as exc:
        raise RecordError(f"its candidate is not one of the problem's: {exc}") from None
    expected = [list(part) for part in key]
    if record.get("candidate_key") != expected:
        raise RecordError(
            f"candidate_key is {record.get('candidate_key')!r}, but its candidate is {expected!r}"
        )


class _Role(NamedTuple):
    """A role a combined evaluator of a record declares."""

    many: bool
    stage: str | None
    metrics: list[MetricKey]
    artifacts: list[str]


class _Implementation(NamedTuple):
    """What an implementation of a record declares: its metrics with their units, the names of
    its artifacts and its roles; ``None`` for none."""

    metrics: dict[MetricKey, str] | None
    artifacts: list[str] | None
    roles: dict[str, _Role] | None


def _implementations(spec: ProblemSpec, record: Mapping[str, object]) -> dict[str, _Implementation]:
    """What each implementation of the record declares, by name; they must be those the problem
    binds, in the order it first binds them, with its versions and reading the stage its entries
    say."""
    implementations = _mapping(record.get("implementations"), "implementations")
    versions = {entry.name: entry.version for entry in spec.evaluators}
    if list(implementations) != list(versions):
        raise RecordError(
            f"its implementations are {list(implementations)}, but the problem binds "
            f"{list(versions)}"
        )
    declared: dict[str, _Implementation] = {}
    for name, entry in implementations.items():
        where = f"implementation {name!r}"
        implementation = _mapping(entry, where)
        if implementation.get("version") != versions[name]:
            raise RecordError(
                f"{where}: version {implementation.get('version')!r}, but the problem binds "
                f"version {versions[name]!r}"
            )
        for entry in spec.evaluators:
            if entry.name == name and implementation.get("stage") != entry.stage:
                raise RecordError(
                    f"{where}: stage {implementation.get('stage')!r}, but the problem has it read "
                    f"{entry.stage!r}"
                )
        declared[name] = _Implementation(
            _declared(implementation.get("metrics"), where),
            _artifact_names(implementation.get("artifacts"), where),
            _roles(implementation.get("roles"), where),
        )
    _check_roles(spec, declared)
    return declared


def _roles(value: object, where: str) -> dict[str, _Role] | None:
    """The roles a combined evaluator of a record declares, ``None`` for none."""
    if value is None:
        return None
    roles: dict[str, _Role] = {}
    for name, entry in _mapping(value, f"{where}: roles").items():
        at = f"{where}: roles.{name}"
        role = _mapping(entry, at)
        many, stage = role.get("many"), role.get("stage")
        if not isinstance(many, bool) or not (stage is None or stage in STAGE_TYPES):
            raise RecordError(f"{at}: many {many!r} or stage {stage!r} is not a role's")
        metrics: list[MetricKey] = []
        for key in _list(role.get("metrics"), f"{at}.metrics"):
            if not (
                isinstance(key, list)
                and len(key) == 2
                and all(isinstance(k, str) and k.strip() for k in key)
            ):
                raise RecordError(f"{at}.metrics: {key!r} is no [stage, metric]")
            if (key[0], key[1]) in metrics:  # as a live Role refuses it
                raise RecordError(f"{at}.metrics: {key[0]}/{key[1]} is named twice")
            metrics.append((key[0], key[1]))
        artifacts = _artifact_names(role.get("artifacts"), at) or []
        if stage is not None and (metrics or artifacts):
            raise RecordError(f"{at}: a role that reads a stage reads no metrics or artifacts")
        roles[name] = _Role(many, stage, metrics, artifacts)
    return roles


def _check_roles(spec: ProblemSpec, declared: Mapping[str, _Implementation]) -> None:
    """Each binding gives the roles of its implementation, as a live session checks them: every
    one and no other, one item unless the role reads many, of the kind it reads, and of
    evaluators that declare the metrics and artifacts it reads."""
    implementation = {(e.binding, s): e.name for e in spec.evaluators for s in e.samples}
    for entry in spec.evaluators:
        where = f"evaluator {entry.binding!r}"
        roles = declared[entry.name].roles
        if not roles:
            if entry.roles:
                raise RecordError(f"{where}: the problem gives it roles, which it does not declare")
            continue
        if set(roles) != set(entry.roles):  # in any order, as a live session checks them
            raise RecordError(
                f"{where}: its implementation declares the roles {list(roles)}, but the problem "
                f"gives {list(entry.roles)}"
            )
        for name, role in roles.items():
            items = entry.roles[name]
            if not role.many and len(items) > 1:
                raise RecordError(f"{where}: role {name!r} reads one item, given {len(items)}")
            for item in items:
                if item.stage != role.stage:  # None for both: an evaluation
                    raise RecordError(f"{where}: role {name!r} does not read {item.label}")
                if item.evaluator is None:
                    continue
                upstream = declared[implementation[(item.evaluator, item.sample)]]
                metrics, artifacts = upstream.metrics or {}, upstream.artifacts or []
                if any(key not in metrics for key in role.metrics) or any(
                    a not in artifacts for a in role.artifacts
                ):
                    raise RecordError(
                        f"{where}: role {name!r} reads what {item.label} does not declare"
                    )


def _artifact_names(value: object, where: str) -> list[str] | None:
    """The names of the artifacts an implementation of a record declares, ``None`` for none."""
    if value is None:
        return None
    names = _list(value, f"{where}: artifacts")
    for name in names:
        if not isinstance(name, str) or not ARTIFACT_NAME.fullmatch(name):
            raise RecordError(f"{where}: artifacts: {name!r} is no artifact name")
    if len(set(names)) != len(names):
        raise RecordError(f"{where}: artifacts: {names} names one twice")
    return cast(list[str], names)


def _in_run(value: object, directory: str | None, where: str) -> str:
    """A path a record names of a file in the run directory: relative to it, as it must be to
    stay true when the directory moves, and in ``directory``, the task's own."""
    text = _text(value, where)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise RecordError(f"{where}: {text!r} is not a plain path relative to the run directory")
    if directory is None or path == PurePosixPath(directory) or not path.is_relative_to(directory):
        raise RecordError(f"{where}: {text!r} is not in the task's own directory {directory!r}")
    return text


def _files(
    value: object, task: TaskId, directory: str | None, where: str, *, hashed: bool
) -> dict[str, Artifact]:
    """The files a task of a record made, by name: each named ``<task>#<name>``, in the task's
    own ``directory``, with its size, and with its SHA-256 exactly if such files are
    ``hashed``."""
    files: dict[str, Artifact] = {}
    for name, entry in _mapping(value, where).items():
        at = f"{where}.{name}"
        if not ARTIFACT_NAME.fullmatch(name):  # the reference alone would split at its last '#'
            raise RecordError(f"{at}: {name!r} is no artifact name")
        item = _mapping(entry, at)
        ref = f"{task.label}#{name}"
        if item.get("ref") != ref:
            raise RecordError(f"{at}: ref {item.get('ref')!r}, not {ref!r}")
        path = _in_run(item.get("path"), directory, f"{at}.path")
        sha256 = item.get("sha256")
        if (sha256 is not None) != hashed:
            claim = "is hashed when it is made" if hashed else "is never hashed"
            raise RecordError(f"{at}: sha256 {sha256!r}, but such a file {claim}")
        try:
            files[name] = Artifact(
                ref, Path(path), cast(int, item.get("bytes")), cast(str | None, sha256)
            )
        except EvaluationContractError as exc:
            raise RecordError(f"{at}: {exc}") from None
    return files


def _samples(spec: ProblemSpec, record: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """The part of the record of each sample, which must be the problem's, in its order."""
    samples = _mapping(record.get("samples"), "samples")
    if list(samples) != list(spec.samples):
        raise RecordError(
            f"its samples are {list(samples)}, but the problem's are {list(spec.samples)}"
        )
    return {name: _mapping(part, f"samples.{name}") for name, part in samples.items()}


def _evaluations(
    spec: ProblemSpec,
    record: Mapping[str, object],
    samples: Mapping[str, Mapping[str, object]],
    declared: Mapping[str, _Implementation],
    plan: Plan,
) -> dict[TaskId, _Evaluator]:
    """Each evaluation of the record, in plan order, checked as a live result is: of the
    implementation the problem binds it to, with the settings and collections the problem gives
    it, and the artifacts it declares in the directory ``plan`` gives it. Those of each sample
    are in its part of the record, those of the candidate in ``evaluations``, each by binding in
    problem order."""
    held: dict[str | None, Mapping[str, object]] = {
        name: _mapping(part.get("evaluations"), f"samples.{name}.evaluations")
        for name, part in samples.items()
    }
    held[None] = _mapping(record.get("evaluations"), "evaluations")
    planned = evaluations(spec.samples, spec.evaluators)
    for sample, found in held.items():
        expected = [entry.binding for task, entry in planned if task.sample == sample]
        if list(found) != expected:
            where = "its evaluations" if sample is None else f"samples.{sample}.evaluations"
            raise RecordError(f"{where} are {list(found)}, but the problem's are {expected}")
    evaluators: dict[TaskId, _Evaluator] = {}
    for task, spec_entry in planned:
        where = f"evaluator {task.label!r}"
        entry = _mapping(held[task.sample][spec_entry.binding], where)
        if entry.get("implementation") != spec_entry.name:
            raise RecordError(
                f"{where}: implementation {entry.get('implementation')!r}, but the problem binds "
                f"{spec_entry.name!r}"
            )
        # as the problem wrote them; the settings used and the collections of roles the problem
        # leaves to their defaults depend on the evaluator's code, which a record is read without
        given = _plain(thaw(spec_entry.settings))
        recorded = _mapping(entry.get("settings"), f"{where}: settings").get("declared")
        if recorded != given:
            raise RecordError(
                f"{where}: settings.declared {recorded!r}, but the problem gives {given!r}"
            )
        collections = _mapping(entry.get("collections"), f"{where}: collections")
        named = {role: collections.get(role) for role in spec_entry.collections}
        if named != dict(spec_entry.collections):
            raise RecordError(
                f"{where}: collections {named!r}, but the problem gives "
                f"{dict(spec_entry.collections)!r}"
            )
        if task.sample is None:
            roles = {role: [i.label for i in items] for role, items in spec_entry.roles.items()}
            if entry.get("roles") != roles:
                raise RecordError(
                    f"{where}: roles {entry.get('roles')!r}, but the problem's are {roles}"
                )
        fragment = _fragment(entry, where, set(Status))
        metrics, names, _ = declared[spec_entry.name]
        directory = plan[task].directory
        workdir = entry.get("workdir")
        if workdir not in (None, directory):
            raise RecordError(f"{where}: workdir {workdir!r}, but its directory is {directory!r}")
        if workdir is None and fragment.status in (Status.VALID, Status.INVALID):
            raise RecordError(
                f"{where}: it is {fragment.status}, so it was called, yet has no workdir"
            )
        files = _files(entry.get("artifacts"), task, workdir, f"{where}: artifacts", hashed=True)
        if names is not None:
            unknown = [name for name in files if name not in names]
            if unknown:
                raise RecordError(f"{where}: artifacts {unknown}, which it does not declare")
            missing = [name for name in names if name not in files]
            if missing and fragment.status == Status.VALID:
                raise RecordError(f"{where}: it is valid without its declared artifacts {missing}")
        if fragment.status == Status.BLOCKED and workdir is not None:
            raise RecordError(f"{where}: it is blocked, so it was never called, yet has a workdir")
        own: dict[MetricKey, Measurement] = {}
        for index, item in enumerate(_list(entry.get("measurements"), f"{where}: measurements")):
            at = f"{where}: measurements[{index}]"
            measurement = _measured(_mapping(item, at), at)
            key = (measurement.stage, measurement.metric)
            if key in own:
                raise RecordError(f"{at}: {key[0]}/{key[1]} is measured twice")
            if metrics is not None and key in metrics and measurement.unit != metrics[key]:
                # whatever the status, as live: one metric identity never carries two units
                raise RecordError(
                    f"{at}: {key[0]}/{key[1]} is in {measurement.unit!r}, but the evaluator "
                    f"declares {metrics[key]!r}"
                )
            own[key] = measurement
        if fragment.status == Status.BLOCKED and own:
            raise RecordError(f"{where}: it is blocked, so it was never called, yet it measured")
        status = Status(fragment.status)
        evaluators[task] = _Evaluator(status, fragment.message, own, metrics, fragment)
    for task, spec_entry in planned:  # a valid evaluation carries the metrics roles read of it
        for name, role in (declared[spec_entry.name].roles or {}).items():
            for item in spec_entry.roles.get(name, ()):
                if item.evaluator is None:
                    continue
                read = evaluators[evaluator_task(item.evaluator, item.sample)]
                if read.status is not Status.VALID:
                    continue
                missing_metrics = [key for key in role.metrics if key not in read.measured]
                if missing_metrics:
                    raise RecordError(
                        f"evaluator {item.label!r} is valid without {missing_metrics}, which role "
                        f"{name!r} of {task.label!r} reads"
                    )
    return evaluators


def _supplied(spec: ProblemSpec, record: Mapping[str, object]) -> list[TaskId]:
    """The stages the caller supplied, as the record lists them: distinct stages of the
    problem's samples (``gamma/simulation``), sorted."""
    supplied = _list(record.get("supplied"), "supplied")
    tasks = []
    for label in supplied:
        sample, _, stage = label.partition("/") if isinstance(label, str) else ("", "", "")
        if sample not in spec.samples or stage not in STAGE_TYPES:
            raise RecordError(
                f"supplied {supplied!r} must list stages of the problem's samples, as "
                f"SAMPLE/STAGE with a stage of {', '.join(STAGE_TYPES)}"
            )
        tasks.append(stage_task(stage, sample))
    labels = cast(list[str], supplied)
    if labels != sorted(set(labels)):
        raise RecordError(f"supplied {labels!r} must list each stage once, in sorted order")
    return tasks


_STAGE_STATUSES = (DONE, Status.FAILED, Status.BLOCKED)


def _results(
    record: Mapping[str, object],
    samples: Mapping[str, Mapping[str, object]],
    supplied: Collection[TaskId],
    evaluators: Mapping[TaskId, _Evaluator],
    plan: Plan,
) -> dict[TaskId, TaskResult]:
    """The status of each task of the record, in plan order, checked against the plan of the
    problem: what each task reads, which are blocked and by what, and the files each stage made
    in its directory."""
    planned = [task.id for task in plan.of(Kind.STAGE)]
    unread = [task.label for task in supplied if task not in planned]
    if unread:
        raise RecordError(f"it supplied {', '.join(unread)}, which no evaluator reads")
    fragments: dict[TaskId, _Fragment] = {}
    for sample, part in samples.items():
        where = f"samples.{sample}.stages"
        stages = _mapping(part.get("stages"), where)
        made = [task.name for task in planned if task.sample == sample]
        if list(stages) != made:
            raise RecordError(f"{where} are {list(stages)}, but the problem makes {made}")
        for name, entry in stages.items():
            made_by = stage_task(name, sample)
            at = f"stage {made_by.label!r}"
            part = _mapping(entry, f"{where}.{name}")
            fragment = _fragment(part, at, _STAGE_STATUSES)
            fragments[made_by] = fragment
            directory = plan[made_by].directory  # None unless it runs and writes files
            if part.get("directory") != directory:
                raise RecordError(
                    f"{at}: directory {part.get('directory')!r}, but its directory is {directory!r}"
                )
            outputs = _files(
                part.get("outputs"), made_by, directory, f"{at}: outputs", hashed=False
            )
            made = [stage_output(name)] if fragment.status == DONE and directory else []
            if list(outputs) != made:
                raise RecordError(
                    f"{at}: outputs {list(outputs)}, but a stage that is {fragment.status} "
                    f"{'makes ' + ', '.join(made) if made else 'makes none'} in its directory"
                )
            for role, path in _mapping(part.get("logs"), f"{at}: logs").items():
                _in_run(path, directory, f"{at}: logs.{role}")
    geometry = _mapping(record.get("geometry"), "geometry")
    if geometry.get("status") != DONE:
        raise RecordError(f"geometry: status {geometry.get('status')!r}, but it is always done")
    for key in ("compact", "manifest"):
        _in_run(geometry.get(key), GEOMETRY.name, f"geometry: {key}")
    fragments |= {task: e.fragment for task, e in evaluators.items()}
    results: dict[TaskId, TaskResult] = {GEOMETRY: TaskResult(GEOMETRY, DONE)}
    for task in plan.tasks:
        if task.id == GEOMETRY:
            continue
        where = f"{task.id.kind} {task.id.label!r}"
        fragment = fragments[task.id]
        needs = [need.label for need in task.needs]
        if fragment.reads != needs:
            raise RecordError(
                f"{where} reads {fragment.reads}, but the problem has it read {needs}"
            )
        expected = blocked(task, results)
        if expected is None:
            if fragment.status == Status.BLOCKED or fragment.blocked_by:
                raise RecordError(f"{where} is blocked, but every task it reads succeeded")
            results[task.id] = TaskResult(task.id, fragment.status, fragment.message)
            continue
        causes = [cause.label for cause in expected.blocked_by]
        if (fragment.status, fragment.blocked_by, fragment.message) != (
            Status.BLOCKED,
            causes,
            expected.message,
        ):
            raise RecordError(
                f"{where} is {fragment.status} (blocked by {fragment.blocked_by}, message "
                f"{fragment.message!r}), but what it reads makes it blocked by {causes} "
                f"({expected.message!r})"
            )
        results[task.id] = expected
    return results


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{where} must be a nonblank string, got {value!r}")
    return value


_DECLARED = ("stage", "metric", "unit")


def _declared(value: object, where: str) -> dict[MetricKey, str] | None:
    """The ``metrics`` an evaluator of a record declares, ``None`` if it declares none."""
    if value is None:
        return None
    declared: dict[MetricKey, str] = {}
    for index, item in enumerate(_list(value, f"{where}: metrics")):
        at = f"{where}: metrics[{index}]"
        entry = _mapping(item, at)
        stage, metric, unit = (_text(entry.get(key), f"{at}.{key}") for key in _DECLARED)
        if (stage, metric) in declared:
            raise RecordError(f"{where}: {stage}/{metric} is declared twice")
        declared[(stage, metric)] = unit
    return declared


def _measured(item: Mapping[str, object], where: str) -> Measurement:
    fields: dict[str, Any] = {
        key: item.get(key) for key in ("stage", "metric", "value", "unit", "uncertainty")
    }
    try:
        return Measurement(**fields)
    except EvaluationContractError as exc:
        raise RecordError(f"{where}: {exc}") from None


def _check_units(
    spec: ProblemSpec, evaluations: Mapping[TaskId, _Evaluator], value: _Lookup
) -> None:
    """A constraint that declares a unit is in the unit its evaluator declares for the metric,
    and so is a valid measurement of it, as a live evaluation enforces: a record consistent in
    another unit would compare its value with a limit in the wrong one."""
    for constraint in spec.constraints:
        if constraint.unit is None:
            continue
        where = f"{constraint.label}: its constraint declares {constraint.unit!r}"
        declared = evaluations[_task(constraint.ref)].declared
        key = (constraint.stage, constraint.metric)
        if declared is not None and declared[key] != constraint.unit:
            raise RecordError(f"{where}, but its evaluator declares {declared[key]!r}")
        measurement = value(constraint.ref)  # None unless the evaluator is valid, as live
        if measurement is not None and measurement.unit != constraint.unit:
            raise RecordError(f"{where}, but it is measured in {measurement.unit!r}")


_NAMED = ("evaluator", "sample", "stage", "metric", "label")


def _check_metrics(spec: ProblemSpec, record: Mapping[str, object], value: _Lookup) -> None:
    """Each entry of ``metrics`` names a metric of the objectives and constraints, in order, and
    holds what its evaluator measured, or nothing if that evaluator is not valid."""
    entries = [
        _mapping(entry, f"metrics[{i}]")
        for i, entry in enumerate(_list(record.get("metrics"), "metrics"))
    ]
    found = [[entry.get(key) for key in _NAMED] for entry in entries]
    wanted = [
        [ref.evaluator, ref.sample, ref.stage, ref.metric, ref.label] for ref in spec.references
    ]
    if found != wanted:
        raise RecordError(f"its metrics are {found}, but the problem names {wanted}")
    for ref, entry in zip(spec.references, entries, strict=True):
        measured = _measurement(value(ref))
        expected = {key: measured[key] for key in ("value", "unit", "uncertainty")}
        recorded = {key: entry.get(key) for key in expected}
        if recorded != expected:
            raise RecordError(
                f"metric {ref.label}: recorded {recorded}, but its evaluator measured {expected}"
            )


class Recorded(NamedTuple):
    """What a record that holds up says its candidate's assessment is made from: the
    candidate's status, whether it represents its candidate, and the measurement of each
    objective and constraint metric by the evaluation that names it (``value``)."""

    status: Status
    represents: bool
    value: _Lookup


def check_summary(spec: ProblemSpec, summary: Mapping[str, object]) -> Recorded:
    """Check the run record ``summary`` (``summary.json`` as read back) against the problem
    ``spec`` as :meth:`~k4opt.problem.Assessment.from_summary` documents, all but its
    ``assessment``, which the caller recomputes from what this returns. Raises
    :class:`~k4opt.problem.RecordError`."""
    record = _mapping(summary, "record")
    if record.get("schema") != SUMMARY_SCHEMA:
        raise RecordError(f"schema is {record.get('schema')!r}, not {SUMMARY_SCHEMA!r}")
    _check_problem(spec, record)
    _check_candidate(spec, record)
    _text(record.get("run_dir"), "run_dir")  # where it was made: provenance only
    declared = _implementations(spec, record)
    samples = _samples(spec, record)
    supplied = _supplied(spec, record)
    plan = plan_tasks(spec.samples, spec.evaluators, supplied, references=spec.references)
    evaluators = _evaluations(spec, record, samples, declared, plan)
    results = _results(record, samples, supplied, evaluators, plan)
    required = [task.id.label for task in plan.tasks if task.required]
    if record.get("required") != required:
        raise RecordError(
            f"required is {record.get('required')!r}, but its objectives, constraints and "
            f"required evaluators make it {required!r}"
        )
    status = candidate_status(plan, results)
    if record.get("status") != status:
        raise RecordError(
            f"status is {record.get('status')!r}, but its tasks make it {status.value!r}"
        )
    message = candidate_message(plan, results)
    if record.get("message") != message:
        raise RecordError(
            f"message is {record.get('message')!r}, but its tasks make it {message!r}"
        )
    listed = [
        {"task": result.id.label, "status": result.status, "message": result.message}
        for result in diagnostics(plan, results)
    ]
    if record.get("diagnostics") != listed:
        raise RecordError(
            f"diagnostics are {record.get('diagnostics')!r}, but its tasks make them {listed!r}"
        )
    represents = not any(changed_by_geometry(task.name) for task in supplied if plan[task].required)
    if record.get("represents_candidate") is not represents:
        raise RecordError(
            f"represents_candidate is {record.get('represents_candidate')!r}, but its "
            f"required supplied stages make it {represents}"
        )
    for ref in spec.references:
        metrics = evaluators[_task(ref)].declared
        if metrics is not None and (ref.stage, ref.metric) not in metrics:
            raise RecordError(f"metric {ref.label}: its evaluator does not declare it")

    def value(ref: MetricRef) -> Measurement | None:
        owner = evaluators[_task(ref)]
        if owner.status is not Status.VALID:
            return None
        found = owner.measured.get((ref.stage, ref.metric))
        if found is None:
            raise RecordError(
                f"evaluator {ref.invocation!r} is valid but did not measure "
                f"{ref.stage}/{ref.metric}, which the problem names"
            )
        return found

    _check_metrics(spec, record, value)
    _check_units(spec, evaluators, value)
    return Recorded(status, represents, value)
