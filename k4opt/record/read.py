"""Internal: a candidate's ``summary.json``, read back and checked before anything trusts it.

:func:`check_summary` checks a record against its problem as
:meth:`~k4opt.record.Assessment.from_summary` documents: its form, that it was made for this
problem and candidate, its implementations and roles, each task's part against the plan the
problem gives (what it reads, whether and by what it is blocked, the parameters that reach it,
the files it made, named relative to the run directory in its own directory), what the
candidate configures (each stage's definition, each evaluator's settings, the attributes of the
geometry it changes, as :func:`~k4opt.problem.candidate.configure` makes them from the problem
and the candidate), and the candidate's status, message, required tasks, diagnostics and metrics
recomputed from them. It returns what the candidate's
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
    Declaration,
    EvaluationContractError,
    Measurement,
    Scalar,
    Status,
)
from k4opt.frozen import thaw
from k4opt.geometry import parse_reference
from k4opt.problem.candidate import Configuration, configure, validate_candidate
from k4opt.problem.contract import Demands, MetricKey, check_bindings, check_entries, check_result
from k4opt.problem.errors import CandidateError
from k4opt.problem.models import MetricRef, ProblemSpec, ordered
from k4opt.problem.plan import (
    GEOMETRY,
    Kind,
    Plan,
    TaskId,
    evaluations,
    evaluator_task,
    plan_tasks,
    represents_candidate,
    stage_task,
)
from k4opt.problem.stages import STAGE_TYPES, stage_output
from k4opt.record.errors import RecordError
from k4opt.record.results import (
    DECLINED,
    DONE,
    TaskResult,
    blocked,
    candidate_message,
    candidate_status,
    diagnostics,
)

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
    plain = _plain(spec.to_dict())
    assert isinstance(plain, dict)
    expected, definition = ordered(plain), ordered(definition)  # in the order the problem defines
    keys = dict.fromkeys([*expected, *definition])
    differ = [k for k in keys if k != "name" and definition.get(k) != expected.get(k)]
    if differ:
        raise RecordError(f"it was made for another problem: its {', '.join(differ)} differ")


class _Fragment(NamedTuple):
    """A task's part of a record, as far as its status and dependencies read it."""

    status: str
    message: str | None
    reads: list[object]
    blocked_by: list[object]
    varies: list[object]


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
    return _Fragment(status, message, reads, causes, _list(entry.get("varies"), f"{where}: varies"))


@dataclass(frozen=True, slots=True)
class _Evaluator:
    """One evaluation of a record, as far as the assessment reads it."""

    status: Status
    message: str | None
    measured: Mapping[MetricKey, Measurement]  # its own measurements
    fragment: _Fragment  # its status, and the tasks it reads


def _check_candidate(spec: ProblemSpec, record: Mapping[str, object]) -> Configuration:
    """The record's candidate is one of the problem's, and its ``candidate_key`` is that
    candidate's, as a live evaluation validates and keys it; its configuration, as a live
    evaluation makes it."""
    candidate = _mapping(record.get("candidate"), "candidate")
    try:  # the values are checked for their types and domains here
        values, key = validate_candidate(spec, cast(Mapping[str, Scalar], candidate))
        configuration = configure(spec, values)
    except CandidateError as exc:
        raise RecordError(f"its candidate is not one of the problem's: {exc}") from None
    expected = [list(part) for part in key]
    if record.get("candidate_key") != expected:
        raise RecordError(
            f"candidate_key is {record.get('candidate_key')!r}, but its candidate is {expected!r}"
        )
    return configuration


def _implementations(spec: ProblemSpec, record: Mapping[str, object]) -> dict[str, Declaration]:
    """What each implementation of the record declares, by name; they must be those the problem
    binds, in the order it first binds them, and declare what each binding of them says
    (:func:`~k4opt.problem.contract.check_entries`)."""
    implementations = _mapping(record.get("implementations"), "implementations")
    names = list(dict.fromkeys(entry.name for entry in spec.evaluators))
    if list(implementations) != names:
        raise RecordError(
            f"its implementations are {list(implementations)}, but the problem binds {names}"
        )
    declared = {
        name: Declaration.from_dict(name, entry, RecordError)
        for name, entry in implementations.items()
    }
    check_entries(spec, declared, RecordError)
    return declared


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
    configuration: Configuration,
    record: Mapping[str, object],
    samples: Mapping[str, Mapping[str, object]],
    declared: Mapping[str, Declaration],
    demands: Demands,
    plan: Plan,
) -> dict[TaskId, _Evaluator]:
    """Each evaluation of the record, in plan order, checked as a live result is: of the
    implementation the problem binds it to, with the settings the problem and its candidate give it
    (``configuration``) and the collections the problem gives it, its artifacts in the directory
    ``plan`` gives it, and what it returned against what the problem asks of it
    (:func:`~k4opt.problem.contract.check_result`, the metrics the roles that read it ``demand``
    included). Those of each sample are in its part of the record, those of the candidate in
    ``evaluations``, each by binding in problem order."""
    held: dict[str | None, Mapping[str, object]] = {
        name: _mapping(part.get("evaluations"), f"samples.{name}.evaluations")
        for name, part in samples.items()
    }
    held[None] = _mapping(record.get("evaluations"), "evaluations")
    planned = evaluations(configuration.samples, configuration.evaluators)
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
                f"{where}: settings.declared {recorded!r}, but the problem and its candidate give "
                f"{given!r}"
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
        directory = plan[task].directory
        workdir = entry.get("workdir")
        if workdir not in (None, directory):
            raise RecordError(f"{where}: workdir {workdir!r}, but its directory is {directory!r}")
        # a valid or invalid evaluation was called, in its directory, but for a diagnostic whose
        # check() rejected the candidate: invalid, and not run (a required one rejects it)
        called = fragment.status == Status.VALID or (
            fragment.status == Status.INVALID and plan[task].required
        )
        if workdir is None and called:
            raise RecordError(
                f"{where}: it is {fragment.status}, so it was called, yet has no workdir"
            )
        if workdir is None and fragment.status == Status.INVALID:
            _check_declined(entry, fragment, where)
        files = _files(entry.get("artifacts"), task, workdir, f"{where}: artifacts", hashed=True)
        if fragment.status == Status.BLOCKED and workdir is not None:
            raise RecordError(f"{where}: it is blocked, so it was never called, yet has a workdir")
        own: dict[MetricKey, Measurement] = {}
        for index, item in enumerate(_list(entry.get("measurements"), f"{where}: measurements")):
            at = f"{where}: measurements[{index}]"
            measurement = _measured(_mapping(item, at), at)
            key = (measurement.stage, measurement.metric)
            if key in own:
                raise RecordError(f"{at}: {key[0]}/{key[1]} is measured twice")
            own[key] = measurement
        status = Status(fragment.status)
        if status is Status.BLOCKED:
            if own:
                raise RecordError(
                    f"{where}: it is blocked, so it was never called, yet it measured"
                )
        else:  # what a live evaluation accepts of what it returned
            check_result(
                spec,
                task,
                declared[spec_entry.name],
                demands.get(task, ()),
                status,
                list(own.values()),
                list(files),
                RecordError,
            )
        evaluators[task] = _Evaluator(status, fragment.message, own, fragment)
    return evaluators


def _check_declined(entry: Mapping[str, object], fragment: _Fragment, where: str) -> None:
    """A diagnostic that is ``invalid`` without a directory is one whose ``check()`` rejected the
    candidate (:data:`~k4opt.record.results.DECLINED`): it was never called, so it holds that
    message and measured, made, printed and took nothing; :class:`RecordError` otherwise."""
    message = fragment.message or ""
    if not message.startswith(f"{DECLINED} (") or not message.endswith(")"):
        raise RecordError(
            f"{where}: it is invalid without a workdir, which only a diagnostic whose check() "
            f"rejected the candidate is, but its message is {message!r}"
        )
    ran = [key for key in ("measurements", "artifacts", "provenance") if entry.get(key)]
    ran += [key for key in ("log", "elapsed_seconds") if entry.get(key) is not None]
    if ran:
        raise RecordError(
            f"{where}: its check() rejected the candidate, so it was never called, yet it holds "
            f"{', '.join(ran)}"
        )


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
    spec: ProblemSpec,
    configuration: Configuration,
    record: Mapping[str, object],
    samples: Mapping[str, Mapping[str, object]],
    supplied: Collection[TaskId],
    evaluators: Mapping[TaskId, _Evaluator],
    plan: Plan,
) -> dict[TaskId, TaskResult]:
    """The status of each task of the record, in plan order, checked against the plan of the
    problem: what each task reads, which are blocked and by what, which parameters reach it, the
    definition of each stage as the candidate configures it, the files each stage made in its
    directory, and the attributes the geometry changed (:func:`_check_changes`)."""
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
            definition = _plain(configuration.samples[sample].stages[name].to_dict())
            if part.get("definition") != definition:
                raise RecordError(
                    f"{at}: definition {part.get('definition')!r}, but the problem and its "
                    f"candidate make it {definition!r}"
                )
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
    if geometry.get("varies") != list(plan[GEOMETRY].varies):
        raise RecordError(
            f"geometry: varies {geometry.get('varies')!r}, but the parameters that set it are "
            f"{list(plan[GEOMETRY].varies)!r}"
        )
    status = geometry.get("status")
    if status == DONE:
        for key in ("compact", "manifest"):
            _in_run(geometry.get(key), GEOMETRY.name, f"geometry: {key}")
        _check_changes(spec, configuration, geometry.get("changes"))
        geometry_result = TaskResult(GEOMETRY, DONE)
    elif status == Status.FAILED:  # it ran out of time, or its worker died: nothing was made
        message = geometry.get("message")
        if not isinstance(message, str) or not message.strip():
            raise RecordError(f"geometry: failed, but its message is {message!r}")
        named = [key for key in _MADE if key in geometry]
        if named:
            raise RecordError(
                f"geometry: failed, but it names {', '.join(named)}, which only a geometry that "
                "was made has"
            )
        geometry_result = TaskResult(GEOMETRY, Status.FAILED, message)
    else:
        raise RecordError(f"geometry: status {status!r}, but a geometry is done or failed")
    fragments |= {task: e.fragment for task, e in evaluators.items()}
    results: dict[TaskId, TaskResult] = {GEOMETRY: geometry_result}
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
        if fragment.varies != list(task.varies):
            raise RecordError(
                f"{where} varies with {fragment.varies}, but the parameters that reach it are "
                f"{list(task.varies)}"
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


_MADE = (
    "compact",
    "manifest",
    "fingerprint",
    "source_root",
    "changes",
    "externals",
    "candidate_externals",
)
"""What the record of a geometry that was made holds, of what it made and reads."""


def _check_changes(spec: ProblemSpec, configuration: Configuration, value: object) -> None:
    """The ``changes`` of a record's geometry: one for each geometry destination of the problem,
    in its order, by the parameter that sets it, of the target the destination names (in the
    canonical form a geometry looks it up by), to the text the candidate gives it, each of an
    attribute of its own. Where the attribute is and its unmodified value need the geometry,
    which a record is read without: they are checked against the session that made it, and
    across the records of a study."""
    changes = [
        _mapping(change, f"geometry: changes[{index}]")
        for index, change in enumerate(_list(value, "geometry: changes"))
    ]
    destinations = spec.geometry_destinations
    if len(changes) != len(destinations):
        raise RecordError(
            f"geometry: it changed {len(changes)} attribute(s), but the problem's parameters set "
            f"{len(destinations)}"
        )
    seen: dict[tuple[object, ...], int] = {}
    for index, (change, (parameter, destination), text) in enumerate(
        zip(changes, destinations, configuration.geometry, strict=True)
    ):
        where = f"geometry: changes[{index}]"
        target = parse_reference(destination.canonical_target).text
        expected = {"parameter": parameter.name, "target": target, "new_value": text}
        found = {key: change.get(key) for key in expected}
        if found != expected:
            raise RecordError(
                f"{where} is {found!r}, but the problem and its candidate give {expected!r}"
            )
        for key in ("file", "locator", "attribute"):
            _text(change.get(key), f"{where}.{key}")
        if not isinstance(change.get("old_value"), str):  # an attribute's text may be blank
            raise RecordError(
                f"{where}.old_value must be a string, got {change.get('old_value')!r}"
            )
        attribute = tuple(change.get(key) for key in ("file", "locator", "attribute"))
        if attribute in seen:
            raise RecordError(f"{where} changes the attribute changes[{seen[attribute]}] does")
        seen[attribute] = index


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{where} must be a nonblank string, got {value!r}")
    return value


def _measured(item: Mapping[str, object], where: str) -> Measurement:
    fields: dict[str, Any] = {
        key: item.get(key) for key in ("stage", "metric", "value", "unit", "uncertainty")
    }
    try:
        return Measurement(**fields)
    except EvaluationContractError as exc:
        raise RecordError(f"{where}: {exc}") from None


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
    ``spec`` as :meth:`~k4opt.record.Assessment.from_summary` documents, all but its
    ``assessment``, which the caller recomputes from what this returns. Raises
    :class:`~k4opt.record.RecordError`."""
    record = _mapping(summary, "record")
    if record.get("schema") != SUMMARY_SCHEMA:
        raise RecordError(f"schema is {record.get('schema')!r}, not {SUMMARY_SCHEMA!r}")
    _check_problem(spec, record)
    configuration = _check_candidate(spec, record)
    _text(record.get("run_dir"), "run_dir")  # where it was made: provenance only
    declared = _implementations(spec, record)
    demands = check_bindings(spec, declared, RecordError)
    samples = _samples(spec, record)
    supplied = _supplied(spec, record)
    plan = plan_tasks(
        spec.samples,
        spec.evaluators,
        supplied,
        references=spec.references,
        parameters=spec.parameters,
    )
    evaluators = _evaluations(spec, configuration, record, samples, declared, demands, plan)
    results = _results(spec, configuration, record, samples, supplied, evaluators, plan)
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
    represents = represents_candidate(plan)
    if record.get("represents_candidate") is not represents:
        raise RecordError(
            f"represents_candidate is {record.get('represents_candidate')!r}, but its "
            f"required supplied stages make it {represents}"
        )

    def value(ref: MetricRef) -> Measurement | None:
        owner = evaluators[_task(ref)]
        if owner.status is not Status.VALID:
            return None
        return owner.measured[(ref.stage, ref.metric)]  # a valid one measured it: check_result

    _check_metrics(spec, record, value)
    return Recorded(status, represents, value)
