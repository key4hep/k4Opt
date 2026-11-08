"""Internal: the optimizer told what the record of each trial holds, live or when a resume
recovers it.

What the optimizer learns of a trial is always what its ``summary.json`` holds
(:func:`_report`): the record is read back from disk, checked against the study and the earlier
trials, and its assessment recomputed from its measurements, before anything is told. A valid
candidate becomes a COMPLETE trial with its objective values and one constraint value per limit
(:func:`~k4opt.optimization._optuna.space.constraint_values`); an invalid or failed one a FAIL
trial; a candidate an evaluator rejected a FAIL trial marked ``rejected`` (:func:`_reject`), with
no record. :func:`_settle` tells a trial how its evaluation ended, :func:`_close_interrupted` one
that an interruption ended, and :func:`_recover` the trials a session left running, before a
resume asks for any. Each gives the trial as :func:`~k4opt.optimization.load_study` reads it
back (:class:`~k4opt.optimization.TrialResult`). A trial is told once: after an error, its state
is read back from the storage before anything else is told (:func:`_stored`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import FrozenTrial, Trial, TrialState

from k4opt.errors import K4OptError
from k4opt.execution import Rejected, RunOutcome
from k4opt.optimization._keep import Keeper
from k4opt.optimization._optuna import marks
from k4opt.optimization._optuna.read import trial_result
from k4opt.optimization._optuna.space import constraint_values, objective_values
from k4opt.optimization._results import TrialResult, record_of
from k4opt.optimization._scope import StudyScope
from k4opt.optimization._study import trial_directories, trial_directory
from k4opt.optimization.errors import OptimizationError
from k4opt.record import SUMMARY_NAME

_MESSAGE = 1000  # characters of a message kept in the optimizer's storage; the record has all


type Warn = Callable[[str], None]


type Progress = Callable[[TrialResult], None]


@dataclass(frozen=True, slots=True)
class _Study:
    directory: Path
    document: Mapping[str, Any]  # study.json
    scope: StudyScope  # what the study searches and how its trials are judged
    study: optuna.Study
    warn: Warn
    # what each reference only candidates name read, and the first trial that read it
    read: dict[str, tuple[int, object]] = field(default_factory=dict)
    # what keeps, for a rerun, what each trial's stages read (the top study's, at every level)
    keeper: Keeper | None = None


def _report(
    run: _Study, trial: Trial, existing: Mapping[str, float], *, recovered: bool = False
) -> TrialResult:
    """Tell the optimizer the result of ``trial`` as its ``summary.json`` holds it, and give the
    trial as that record holds it.

    The one way a trial is told, live or when a resume recovers it: the record is read back from
    disk and checked against the study and the earlier trials
    (:func:`~k4opt.optimization._results.record_of`), its assessment recomputed from its
    measurements. ``existing`` are the constraint values the trial already has, which must equal
    the recomputed ones. Raises :class:`OptimizationError`, before anything is told, for a record
    that does not hold up.
    """
    number = trial.number
    recovered = recovered or trial.user_attrs.get(marks.RECOVERED) is True  # a parent reopened
    selected = trial.user_attrs.get(marks.SELECTED)  # of a trial that stands for a study below
    result, warnings = record_of(
        run.directory, run.document, run.scope, number, trial.params, run.read, selected
    )
    assessment = result.assessment
    assert assessment is not None
    valid = result.status == "valid"
    values = objective_values(assessment) if valid else None
    constraints = constraint_values(assessment) if valid else {}
    for key, value in constraints.items():
        if key in existing and existing[key] != value:
            raise OptimizationError(
                f"trial {number}: the optimizer holds {existing[key]} for {key}, but the "
                f"record gives {value}"
            )
    for warning in warnings:
        run.warn(warning)
    if run.keeper is not None:  # what its stages read, before anything of it can change
        run.keeper.trial(run.directory / trial_directory(number))
    trial.set_user_attr(marks.STATUS, result.status)
    path = f"{trial_directory(number)}/{SUMMARY_NAME}"
    record = {"id": result.record, "sha256": result.sha256, "path": path}
    trial.set_user_attr(marks.RECORD, record)
    if isinstance(result.message, str):
        trial.set_user_attr(marks.MESSAGE, result.message[:_MESSAGE])
    # the diagnostics that did not succeed, which the status leaves out (checked by from_summary)
    trial.set_user_attr(marks.DIAGNOSTICS, len(result.diagnostics))
    if recovered:
        trial.set_user_attr(marks.RECOVERED, True)
    if values is None:
        run.study.tell(trial, state=TrialState.FAIL)
    else:
        for key, value in constraints.items():
            if key not in existing:
                trial.set_constraint(key, value)
        trial.set_user_attr(marks.FEASIBLE, assessment.feasible)
        run.study.tell(trial, values)
    return replace(result, recovered=recovered)


def _reject(run: _Study, trial: Trial, rejected: Rejected) -> TrialResult:
    """Tell the optimizer that an evaluator rejected the candidate of ``trial``: FAIL, without
    values or constraint values, marked ``rejected`` with the evaluation that rejected it and
    why. Nothing of the candidate was created, so there is no record: this is all there is of
    it, read back strictly (:func:`~k4opt.optimization._optuna.read.trial_result`). The marks are
    written in the order of :data:`~k4opt.optimization._optuna.marks.REJECTION`, its status last,
    so that a trial interrupted before that reads as interrupted."""
    told = {
        marks.REJECTED_BY: rejected.by,
        marks.MESSAGE: rejected.message[:_MESSAGE],
        marks.STATUS: "rejected",
    }
    for key in marks.REJECTION:
        trial.set_user_attr(key, told[key])
    frozen = run.study.tell(trial, state=TrialState.FAIL)
    result, _ = trial_result(run.directory, run.document, run.scope, frozen, run.read)
    return result


def _unscore(run: _Study, trial: Trial, message: str) -> TrialResult:
    """Tell the optimizer that the study of the level below that ``trial`` stands for has no
    feasible trial: FAIL, without values, its status written last; no value is invented."""
    told = {marks.MESSAGE: message[:_MESSAGE], marks.STATUS: "unscored"}
    for key in marks.UNSCORED:
        trial.set_user_attr(key, told[key])
    frozen = run.study.tell(trial, state=TrialState.FAIL)
    result, _ = trial_result(run.directory, run.document, run.scope, frozen, run.read)
    return result


def _fail(run: _Study, trial: Trial, error: str | None = None) -> FrozenTrial:
    if error is not None:
        trial.set_user_attr(marks.ERROR, error[:_MESSAGE])
    return run.study.tell(trial, state=TrialState.FAIL)


def _stored(run: _Study, number: int) -> FrozenTrial | None:
    """Trial ``number`` as the storage holds it now; ``None`` if it cannot be read.

    Read back before a trial is told anything after an error, since a ``tell`` can commit and
    then raise: a trial is told once."""
    try:
        return next(t for t in run.study.get_trials(deepcopy=False) if t.number == number)
    except Exception:
        return None


def _settle(
    run: _Study, trial: Trial, output: Path, result: RunOutcome | Rejected | Exception
) -> TrialResult:
    """Tell the optimizer how the trial's evaluation ended, and give the trial as told; raise to
    stop the study."""
    try:
        if isinstance(result, Exception):
            raise result
        if isinstance(result, Rejected):
            return _reject(run, trial, result)
        return _report(run, trial, {})
    except KeyboardInterrupt:
        with contextlib.suppress(Exception):  # one that cannot be told is left to --resume
            _close_interrupted(run, trial, output)
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise OptimizationError(
            f"trial {trial.number} stops the study ({_failed(run, trial, error)}): {error}"
        ) from exc


def _failed(run: _Study, trial: Trial, error: str) -> str:
    """Tell ``trial`` FAIL with ``error`` unless the storage holds it ended, or it was stopped
    while its record was told; what became of it."""
    name = trial_directory(trial.number)
    stored = _stored(run, trial.number)
    if stored is None:
        return f"the state of {name} could not be read; k4opt optimize --resume settles it"
    if stored.state != TrialState.RUNNING:
        return f"{name} is {stored.state.name}"
    attrs = stored.user_attrs
    if run.scope.nested:  # its study of the level below stays: a resume goes on from it
        return f"{name} was left running; k4opt optimize --resume goes on with it"
    if marks.RECORD in attrs or attrs.get(marks.STATUS) not in (None, "rejected"):
        # stopped while its record was told (_report): FAIL with part of those marks would not
        # read back, so it stays RUNNING, and a resume tells it its record again
        return f"{name} was left running; k4opt optimize --resume settles it"
    try:
        _fail(run, trial, error)
    except Exception:
        return f"{name} could not be told FAIL; k4opt optimize --resume settles it"
    return f"{name} is FAIL"


def _close_interrupted(run: _Study, trial: Trial, output: Path) -> None:
    """A trial that Ctrl-C, SIGTERM or another trial's error ended: told its record if that was
    published just before, FAIL otherwise, its directory left as it is; nothing if the storage
    holds it ended already, or cannot say (a resume settles it). A trial interrupted before it
    was marked as k4opt's (just after it was asked for) is marked now, so that it reads as
    interrupted."""
    stored = _stored(run, trial.number)
    if stored is None or stored.state != TrialState.RUNNING:
        return
    if (output / SUMMARY_NAME).is_file():
        # the interruption may have come within _report, after the constraints were set
        try:
            _report(run, trial, stored.constraints)
            return
        except K4OptError as exc:  # a record that does not hold up is no result
            run.warn(f"trial {trial.number}: its record does not hold up ({exc}); interrupted")
    if run.scope.nested:  # it stands for a study of the level below, which a resume goes on with
        return
    if marks.DIR not in stored.user_attrs:
        trial.set_user_attr(marks.DIR, output.name)
    _fail(run, trial)


def _recover(run: _Study, progress: Progress) -> None:
    """Finish the trials a session left running, before any new one: one without a record is
    FAIL, its directory left as it is, and one with a record is told it; ``progress`` is called
    with each as it is finished. A trial that stands for a study of the level below stays
    RUNNING, without a record: :mod:`~k4opt.optimization._optuna.tree` goes on with that study.
    Before anything changes, the resume refuses if a trial
    directory belongs to no trial, or if a finished trial does not hold what its record gives,
    checked as :func:`~k4opt.optimization.load_study` checks it: its record is gone or was
    changed, its state, status, candidate, values or constraint values are not the record's, or
    k4opt did not make it."""
    trials = run.study.get_trials(deepcopy=False)
    known = {trial.user_attrs.get(marks.DIR, trial_directory(trial.number)) for trial in trials}
    stray = [name for name in trial_directories(run.directory) if name not in known]
    if stray:
        raise OptimizationError(
            f"{', '.join(stray)} in {run.directory} belong(s) to no trial of the study; move "
            "them out of the study directory to resume it"
        )
    for frozen in sorted(trials, key=lambda frozen: frozen.number):
        # the trials still to come must read what the earlier candidates' own references read
        # (``run.read``); the warnings of a record were given when the optimizer was told it
        try:
            trial_result(run.directory, run.document, run.scope, frozen, run.read)
        except OptimizationError as exc:
            raise OptimizationError(
                f"{exc}; the study resumes only with every trial as its record holds it"
            ) from None
    for frozen in trials:
        if frozen.state != TrialState.RUNNING:
            continue
        trial = Trial(run.study, frozen._trial_id)  # the one private access of Optuna
        if (run.directory / trial_directory(frozen.number) / SUMMARY_NAME).is_file():
            progress(_report(run, trial, frozen.constraints, recovered=True))
            continue
        if run.scope.nested:
            continue  # it stands for a study of the level below: the run goes on with that
        trial.set_user_attr(marks.RECOVERED, True)
        finished = run.study.tell(trial, state=TrialState.FAIL)
        progress(trial_result(run.directory, run.document, run.scope, finished, run.read)[0])
