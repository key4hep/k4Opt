"""Internal: a study read back by :func:`k4opt.optimization.load_study`, without changing it.

Which trials a study has and how each ended come from its ledger (:mod:`._ledger`), read without
writing; what each measured from its record, read and checked by
:func:`~k4opt.optimization._results.recorded` against the record the ledger names. No strategy is
opened or needed: a study reads back alike whatever proposed its candidates.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from k4opt.frozen import freeze
from k4opt.optimization._ledger import LEDGER_NAME, Entry, Ledger, NotSetUpError, read_ledger
from k4opt.optimization._results import (
    StudyResults,
    TrialResult,
    check_inner,
    pareto_front,
    recorded,
    recorded_scope,
)
from k4opt.optimization._scope import StudyScope
from k4opt.optimization._study import (
    INNER_NAME,
    STUDY_NAME,
    inner_directory,
    read_study,
    trial_directories,
    trial_directory,
)
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import CandidateError
from k4opt.problem.candidate import validate_value
from k4opt.problem.plan import level_plan, rejecting
from k4opt.record import SUMMARY_NAME


def inner_id(study: str, number: int) -> str:
    """The id of the study of the level below that trial ``number`` of the study ``study``
    stands for: decided by the trial, so that making it again is making the same one."""
    return str(uuid.uuid5(uuid.UUID(study), str(number)))


def trial_result(
    directory: Path,
    document: Mapping[str, Any],
    scope: StudyScope,
    entry: Entry,
    read: dict[str, tuple[int, object]],
) -> tuple[TrialResult, list[str]]:
    """Trial ``entry`` as the ledger and its record hold it, checked, with the warnings of its
    record.

    A running trial is ``running``, unchecked and without a record, even if one is already on
    disk: the study has not kept it. A ``recorded`` trial shows the record the ledger names,
    which must have that status. A trial without a record is ``rejected``, ``malformed``,
    ``unscored``, ``interrupted`` or ``error`` as the ledger says, and must be one such a trial
    can be: a rejection by a task that can reject, nothing created of a rejected or malformed
    candidate, ``unscored`` only of a trial that stands for a study of the level below. The
    proposed values must be of the parameters the study searches, each in its domain. A resume
    checks each trial with it, before it recovers or proposes any."""
    number, ending = entry.number, entry.ending
    kind = None if ending is None else ending.kind
    if kind != "malformed":
        _check_values(scope, entry)
    selected = entry.selected
    if selected is not None and not scope.nested:
        raise OptimizationError(f"trial {number} selected a trial of a level below, but none is")
    added = {} if selected is None else selected["values"]
    # with the values the study fixes, and those of the trial of the level below it selected
    candidate = scope.candidate({**entry.values, **added})  # type: ignore[dict-item]
    recovered = entry.recovered
    if ending is None:
        return TrialResult(number, "running", candidate, selected=selected), []
    own = directory / trial_directory(number)
    if kind == "recorded":
        if scope.nested and selected is None:
            raise OptimizationError(
                f"trial {number} has a record, but selected no trial of its study of the level "
                "below"
            )
        assert ending.record is not None
        told = {**ending.record, "path": f"{trial_directory(number)}/{SUMMARY_NAME}"}
        result, warnings = recorded(
            directory, document, scope, number, entry.values, told, read, selected
        )
        if result.status != ending.status:
            raise OptimizationError(
                f"trial {number}: the ledger holds the status {ending.status!r}, but its record "
                f"is {result.status!r}"
            )
        return replace(result, recovered=recovered), warnings
    where = f"trial {number} is {kind}, but"
    if kind in ("rejected", "malformed") and (own.exists() or own.is_symlink()):
        raise OptimizationError(
            f"{where} {own.name} exists, though nothing of its candidate is created"
        )
    if kind == "malformed":
        if selected is not None:
            raise OptimizationError(f"{where} it selected a trial of a level below")
        message = ending.message
        return TrialResult(number, kind, scope.candidate({}), message=message), []
    if kind == "rejected":
        spec = scope.spec
        plan = level_plan(spec, scope.level)
        if ending.by not in [task.label for task in rejecting(plan, spec.samples, spec.parameters)]:
            raise OptimizationError(
                f"{where} by {ending.by!r}, which is neither a required evaluation of the problem "
                "nor a required reconstruction whose arguments a parameter sets, the only kinds "
                "of task that reject a candidate"
            )
        message = f"{ending.by}: {ending.message}"
    elif kind == "unscored":
        if not scope.nested or selected is not None:
            raise OptimizationError(
                f"{where} its study has no level below, or it selected a trial of it"
            )
        message = ending.message
    else:  # interrupted, error
        message = ending.message
    return TrialResult(
        number, str(kind), candidate, message=message, recovered=recovered, selected=selected
    ), []


def _check_values(scope: StudyScope, entry: Entry) -> None:
    """The values proposed for a trial are of the parameters the study searches, each of its
    type and in its domain."""
    if set(entry.values) != set(scope.searched):
        raise OptimizationError(
            f"trial {entry.number} was proposed with values of {sorted(entry.values)}, but the "
            f"study searches {list(scope.searched)}"
        )
    parameters = {parameter.name: parameter for parameter in scope.spec.parameters}
    for name, value in entry.values.items():
        try:
            validate_value(parameters[name], value)
        except CandidateError as exc:
            raise OptimizationError(f"trial {entry.number} was proposed with {exc}") from None


def results_of(
    directory: Path,
    document: Mapping[str, Any],
    scope: StudyScope,
    ledger: Ledger,
    warnings: Sequence[str] = (),
) -> StudyResults:
    """The study ``document`` describes, whose trials ``ledger`` holds, as its records hold it:
    each trial read by :func:`trial_result`, as :func:`k4opt.optimization.load_study` reads it,
    with ``warnings`` first."""
    trials: list[TrialResult] = []
    notes = list(warnings)
    read: dict[str, tuple[int, object]] = {}
    for entry in ledger.entries:
        result, found = trial_result(directory, document, scope, entry, read)
        if scope.nested:
            result = _with_inner(directory, document, scope, entry, result)
        trials.append(result)
        notes += found
    known = {trial_directory(entry.number) for entry in ledger.entries}
    notes += [
        f"{name} in {directory} belongs to no trial of the study, so it is not shown"
        for name in trial_directories(directory)
        if name not in known
    ]
    document_ = freeze(document, STUDY_NAME, OptimizationError)
    assert isinstance(document_, Mapping)
    return StudyResults(
        name=document["name"],
        id=document["id"],
        created=document["created"],
        directory=directory,
        problem=scope.spec,
        document=document_,
        trials=tuple(trials),
        pareto=pareto_front(trials, scope.criteria),
        warnings=tuple(notes),
        exhausted=ledger.exhausted,
    )


def _ran(path: Path) -> bool:
    """Whether anything ran in the study directory ``path``, which may not exist."""
    return path.is_dir() and bool(trial_directories(path) or (path / INNER_NAME).exists())


def _with_inner(
    directory: Path,
    document: Mapping[str, Any],
    scope: StudyScope,
    entry: Entry,
    result: TrialResult,
) -> TrialResult:
    """``result`` with the study of the level below that its trial stands for, read as any study
    and its link checked (:func:`~k4opt.optimization._results.check_inner`): every trial that
    ended with a record, or unscored, or that selected has one; a trial still running, before
    that, may have none yet, or one whose making was cut short (:class:`NotSetUpError`, with
    nothing in its directory that ran). Whatever else is wrong with it is refused."""
    if result.status in ("malformed",):
        return result  # nothing was made of it
    path = inner_directory(directory, entry.number)
    if (path / STUDY_NAME).is_file():
        try:
            inner = load(path)
        except NotSetUpError:
            if entry.running and entry.selected is None and not _ran(path):
                return result
            raise
        expected = inner_id(document["id"], entry.number)
        if inner.id != expected:
            raise OptimizationError(
                f"trial {entry.number} stands for the study {expected!r}, but {path} holds "
                f"{inner.id!r}"
            )
        result = replace(result, inner=inner)
    elif _ran(path) or (path / LEDGER_NAME).exists():
        raise OptimizationError(f"{path} holds a study's files, but no {STUDY_NAME}")
    held = result.inner is not None or result.record is not None or result.selected is not None
    if held or result.status == "unscored":
        check_inner(document, scope, result)
    return result


def load(directory: str | os.PathLike[str]) -> StudyResults:
    """The study in ``directory`` as its ledger and records hold it, read without changing
    anything; see :func:`k4opt.optimization.load_study`."""
    path = Path(os.path.abspath(Path(directory).expanduser()))
    if not (path / STUDY_NAME).is_file():
        raise OptimizationError(f"{path} is not a study directory: it holds no {STUDY_NAME}")
    document = read_study(path)
    scope = recorded_scope(document)
    ledger = read_ledger(path, document["id"])
    return results_of(path, document, scope, ledger)
