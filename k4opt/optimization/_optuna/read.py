"""Internal: a study read back by :func:`k4opt.optimization.load_study`, without changing it.

The storage is opened so that it refuses every write
(:func:`~k4opt.optimization._optuna.storage.existing_storage`), and nothing that makes, resumes or
recovers a study is called. Each trial's state comes from Optuna; its record is read and checked
by :func:`k4opt.optimization._results.recorded`, and must agree with what Optuna was told.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import BaseStorage
from optuna.trial import FrozenTrial, TrialState

from k4opt.frozen import freeze
from k4opt.optimization._optuna import marks
from k4opt.optimization._optuna.optimizer import (
    check_distributions,
    check_optuna_study,
    study_attributes,
)
from k4opt.optimization._optuna.space import constraint_values, objective_values
from k4opt.optimization._optuna.storage import NotSetUpError, close, existing_storage, leaving
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
    STUDY_NAME,
    inner_directory,
    read_study,
    trial_directories,
    trial_directory,
)
from k4opt.optimization.errors import OptimizationError
from k4opt.problem.plan import level_plan, rejecting


def _existing_study(document: Mapping[str, Any], storage: BaseStorage) -> optuna.Study:
    """The Optuna study ``document`` describes, as ``storage`` holds it; one that was never set
    up, or only in part, is refused rather than set up."""
    name = document["optimizer"]["study"]
    try:
        study = optuna.load_study(study_name=name, storage=storage)
    except KeyError:
        raise NotSetUpError(
            f"the optimizer's storage holds no study {name!r}: the process that made the study "
            "stopped before setting it up; k4opt optimize --resume sets it up"
        ) from None
    missing = [key for key in study_attributes(document) if key not in study.user_attrs]
    if missing:
        raise NotSetUpError(
            f"the optimizer's study {name!r} lacks {', '.join(missing)}: the process that made it "
            "stopped while setting it up; k4opt optimize --resume finishes that"
        )
    check_optuna_study(study, document)
    return study


def _check_told_values(frozen: FrozenTrial, result: TrialResult) -> None:
    """A COMPLETE trial holds what its record gives: the objective values, the feasibility and a
    constraint value for each limit."""
    assessment = result.assessment
    assert assessment is not None
    where = f"trial {frozen.number}: the optimizer holds"
    values, constraints = objective_values(assessment), constraint_values(assessment)
    if frozen.values != values:
        raise OptimizationError(f"{where} the values {frozen.values}, its record gives {values}")
    if frozen.user_attrs.get(marks.FEASIBLE) != assessment.feasible:
        raise OptimizationError(
            f"{where} feasible={frozen.user_attrs.get(marks.FEASIBLE)!r}, its record gives "
            f"{assessment.feasible!r}"
        )
    if frozen.constraints != constraints:
        raise OptimizationError(
            f"{where} the constraint values {frozen.constraints}, its record gives {constraints}"
        )


def trial_result(
    directory: Path,
    document: Mapping[str, Any],
    scope: StudyScope,
    frozen: FrozenTrial,
    read: dict[str, tuple[int, object]],
) -> tuple[TrialResult, list[str]]:
    """Trial ``frozen`` as its record holds it, checked against what the optimizer was told, with
    the warnings of its record.

    A RUNNING trial is ``running``, unchecked and without a record, even if one is already on
    disk: the optimizer has not taken it. A trial told a record shows that record, which must
    agree with the trial's state (COMPLETE for ``valid``, FAIL otherwise) and values, except
    that a FAIL trial with a ``valid`` record and ``k4opt.error`` is ``error``, without a
    result: its values never reached the optimizer. A FAIL trial without a record is
    ``rejected`` if k4opt told it so (:func:`_rejected`), ``error`` with ``k4opt.error`` and
    ``interrupted`` without; any other trial was not made by k4opt. A resume checks each trial
    with it, before it recovers or proposes any.
    """
    number, attrs = frozen.number, frozen.user_attrs
    unknown = [key for key in frozen.params if key not in scope.searched]
    if unknown:
        raise OptimizationError(
            f"trial {number} has the parameter(s) {', '.join(unknown)}, which the study does "
            "not search"
        )
    selected = attrs.get(marks.SELECTED)
    added = {} if selected is None else selected["values"]
    # with the values the study fixes, and those of the trial of the level below it selected
    candidate = scope.candidate({**frozen.params, **added})
    recovered = attrs.get(marks.RECOVERED) is True
    if frozen.state == TrialState.RUNNING:
        return TrialResult(number, "running", candidate, recovered=recovered, selected=selected), []
    told, error = attrs.get(marks.RECORD), attrs.get(marks.ERROR)
    if attrs.get(marks.STATUS) == "rejected":
        return _rejected(frozen, scope, candidate, directory / trial_directory(number)), []
    if attrs.get(marks.STATUS) == "unscored":
        return _unscored(frozen, scope, candidate), []
    if told is None:
        # interrupted, also while its rejection was told (a prefix of it, without the status)
        ours = marks.DIR in attrs or recovered
        own = {key for key in attrs if key.startswith(marks.PREFIX)}
        if frozen.state != TrialState.FAIL or own - marks.PARTLY_REJECTED or not ours:
            raise OptimizationError(
                f"trial {number} is {frozen.state.name} without the record k4opt tells a trial "
                "with, or any mark of k4opt, so it was not made by k4opt optimize"
            )
        status, message = ("interrupted", None) if error is None else ("error", str(error))
        return TrialResult(number, status, candidate, message=message, recovered=recovered), []
    if scope.nested and selected is None:
        raise OptimizationError(
            f"trial {number} has a record, but selected no trial of its study of the level below"
        )
    params = frozen.params
    result, warnings = recorded(directory, document, scope, number, params, told, read, selected)
    if attrs.get(marks.STATUS) != result.status:
        raise OptimizationError(
            f"trial {number}: the optimizer was told the status {attrs.get(marks.STATUS)!r}, "
            f"but its record is {result.status!r}"
        )
    if frozen.state == TrialState.COMPLETE and result.status == "valid":
        _check_told_values(frozen, result)
    elif frozen.state == TrialState.FAIL and result.status == "valid" and error is not None:
        result = replace(result, status="error", assessment=None, message=str(error))
    elif frozen.state != TrialState.FAIL or result.status == "valid":
        raise OptimizationError(
            f"trial {number} is {frozen.state.name}, but its record is {result.status!r}"
        )
    return replace(result, recovered=recovered), warnings


def _unscored(frozen: FrozenTrial, scope: StudyScope, candidate: Mapping[str, Any]) -> TrialResult:
    """Trial ``frozen``, whose study of the level below has no feasible trial: FAIL, without
    values or a record, saying why."""
    attrs, where = frozen.user_attrs, f"trial {frozen.number} is unscored, but"
    if not scope.nested:
        raise OptimizationError(f"{where} its study has no level below")
    if frozen.state != TrialState.FAIL or frozen.values is not None or frozen.constraints:
        raise OptimizationError(f"{where} it is {frozen.state.name}, with values or constraints")
    message = attrs.get(marks.MESSAGE)
    if not isinstance(message, str):
        raise OptimizationError(f"{where} its message is {message!r}")
    recovered = attrs.get(marks.RECOVERED) is True
    return TrialResult(frozen.number, "unscored", candidate, message=message, recovered=recovered)


def _rejected(
    frozen: FrozenTrial, scope: StudyScope, candidate: Mapping[str, Any], own: Path
) -> TrialResult:
    """Trial ``frozen``, which k4opt told was rejected, read strictly: a rejection has no record,
    so all there is of it is what the optimizer holds, which must be exactly what
    :func:`~k4opt.optimization._optuna.tell._reject` tells: FAIL, no values and no constraint
    values, the required evaluation of the problem that rejected it (a diagnostic rejects no
    candidate) and its message, and no trial directory, since nothing of its candidate was
    created."""
    number, attrs = frozen.number, frozen.user_attrs
    where = f"trial {number} is rejected, but"
    if frozen.state != TrialState.FAIL or frozen.values is not None or frozen.constraints:
        raise OptimizationError(f"{where} it is {frozen.state.name}, with values or constraints")
    k4opt = {key for key in attrs if key.startswith(marks.PREFIX)}
    if k4opt - marks.REJECTED or not {marks.REJECTED_BY, marks.MESSAGE} <= k4opt:
        raise OptimizationError(
            f"{where} it carries {', '.join(sorted(k4opt))}, not what k4opt tells a rejected trial"
        )
    by, message = attrs[marks.REJECTED_BY], attrs[marks.MESSAGE]
    spec = scope.spec
    plan = level_plan(spec, scope.level)
    if by not in [task.label for task in rejecting(plan, spec.samples, spec.parameters)]:
        raise OptimizationError(
            f"{where} by {by!r}, which is neither a required evaluation of the problem nor a "
            "required reconstruction whose arguments a parameter sets, the only kinds of task "
            "that reject a candidate"
        )
    if not isinstance(message, str):
        raise OptimizationError(f"{where} its message is {message!r}")
    if own.exists() or own.is_symlink():
        raise OptimizationError(
            f"{where} {own.name} exists, though nothing of a rejected candidate is created"
        )
    return TrialResult(number, "rejected", candidate, message=f"{by}: {message}")


def results_of(
    directory: Path,
    document: Mapping[str, Any],
    scope: StudyScope,
    study: optuna.Study,
    warnings: Sequence[str] = (),
    storage: str | None = None,
) -> StudyResults:
    """The study ``document`` describes, whose trials ``study`` holds, as its records hold it:
    each trial read by :func:`trial_result`, as :func:`k4opt.optimization.load_study` reads it,
    with ``warnings`` first."""
    trials: list[TrialResult] = []
    notes = list(warnings)
    read: dict[str, tuple[int, object]] = {}
    held = sorted(study.get_trials(deepcopy=False), key=lambda frozen: frozen.number)
    for frozen in held:
        result, found = trial_result(directory, document, scope, frozen, read)
        if scope.nested:
            result = _with_inner(directory, document, scope, frozen, result, storage)
        trials.append(result)
        notes += found
    known = {frozen.user_attrs.get(marks.DIR, trial_directory(frozen.number)) for frozen in held}
    notes += [
        f"{name} in {directory} belongs to no trial of the study, so it is not shown"
        for name in trial_directories(directory)
        if name not in known
    ]
    document_ = freeze(document, STUDY_NAME, OptimizationError)
    assert isinstance(document_, Mapping)
    return StudyResults(
        name=document["optimizer"]["study"],
        id=document["id"],
        created=document["created"],
        directory=directory,
        problem=scope.spec,
        document=document_,
        trials=tuple(trials),
        pareto=pareto_front(trials, scope.criteria),
        warnings=tuple(notes),
    )


def _with_inner(
    directory: Path,
    document: Mapping[str, Any],
    scope: StudyScope,
    frozen: FrozenTrial,
    result: TrialResult,
    storage: str | None,
) -> TrialResult:
    """``result`` with the study of the level below that its trial stands for, read as any study
    and its link checked (:func:`~k4opt.optimization._results.check_inner`): every trial that
    ended with a record, or unscored, has one; a trial that was stopped may have none yet, or one
    whose making was cut short (:class:`NotSetUpError`). Whatever else is wrong with it is refused,
    of a trial still running too."""
    path = inner_directory(directory, frozen.number)
    if (path / STUDY_NAME).is_file():
        try:
            inner = load(path, storage)
        except NotSetUpError:
            if frozen.state == TrialState.RUNNING:
                return result
            raise
        if inner.id != frozen.user_attrs.get(marks.CHILD):
            raise OptimizationError(
                f"trial {frozen.number} stands for the study "
                f"{frozen.user_attrs.get(marks.CHILD)!r}, but {path} holds {inner.id!r}"
            )
        result = replace(result, inner=inner)
    if result.inner is not None or result.record is not None or result.status == "unscored":
        check_inner(document, scope, result)
    return result


def load(directory: str | os.PathLike[str], storage: str | None) -> StudyResults:
    """The study in ``directory`` as its records hold it, read without changing anything; see
    :func:`k4opt.optimization.load_study`."""
    path = Path(os.path.abspath(Path(directory).expanduser()))
    if not (path / STUDY_NAME).is_file():
        raise OptimizationError(f"{path} is not a study directory: it holds no {STUDY_NAME}")
    document = read_study(path)
    scope = recorded_scope(document)
    check_distributions(document, scope)
    database: BaseStorage | None = None
    try:
        database = existing_storage(path, document, storage)
        study = _existing_study(document, database)
        return results_of(path, document, scope, study, storage=storage)
    except Exception as exc:
        replacement, secret = leaving(exc, storage, what="the study cannot be read")
        if replacement is None:
            raise
        if secret:  # nothing of the original, which names it, is kept for a traceback
            raise replacement from None
        raise replacement from exc
    finally:
        close(database)  # everything is read: the snapshot holds nothing of the storage
