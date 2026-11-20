"""Internal: how each trial of a study ends, live or when a resume recovers it; nothing of any
strategy but what it is told.

What a study holds of a trial is what its ``summary.json`` holds (:func:`report`): the record is
read back from disk, checked against the study and the earlier trials, and its assessment
recomputed from its measurements, before anything is kept. A candidate an evaluator rejected
has no record (:func:`reject`); a trial whose study of the level below has no feasible trial is
``unscored`` (:func:`unscore`); a proposal that is no candidate ``malformed``
(:func:`malform`). :func:`settle` ends a trial as its evaluation ended,
:func:`close_interrupted` one an interruption ended, and :func:`recover` the trials a session
left running, before a resume asks for any.

Each ending is written to the study's ledger first, once checked as a reader checks it, and only
then is the strategy told (:func:`feedback`): a trial the ledger holds ended is never lost or
told twice, since a strategy is made anew from the ledger whenever the study opens, and one the
ledger does not hold ended is settled again by the next session, from its record. Each gives
the trial as :func:`~k4opt.optimization.load_study` reads it back.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from k4opt.errors import K4OptError
from k4opt.execution import Rejected, RunOutcome
from k4opt.frozen import Json
from k4opt.optimization._keep import Keeper
from k4opt.optimization._ledger import Ending, Entry, LedgerWriter
from k4opt.optimization._proposal import Feedback, Strategy
from k4opt.optimization._read import trial_result
from k4opt.optimization._results import TrialResult, record_of
from k4opt.optimization._scope import StudyScope
from k4opt.optimization._study import trial_directories, trial_directory
from k4opt.optimization.errors import OptimizationError
from k4opt.record import SUMMARY_NAME, Assessment

type Warn = Callable[[str], None]

type Progress = Callable[[TrialResult], None]


@dataclass(slots=True)
class OpenStudy:
    """A study opened to run trials: its directory, ``study.json``, scope and ledger; its
    strategy once made; and what the trials' records are checked against as they come."""

    directory: Path
    document: Mapping[str, Any]
    scope: StudyScope
    ledger: LedgerWriter
    warn: Warn
    keeper: Keeper | None = None  # what keeps, for a rerun, what each trial's stages read
    strategy: Strategy | None = None
    # what each reference only candidates name read, and the first trial that read it
    read: dict[str, tuple[int, object]] = field(default_factory=dict)

    def record(self, number: int) -> Path:
        return self.directory / trial_directory(number) / SUMMARY_NAME


# --- what a strategy is told -----------------------------------------------------------------


def feedback(result: TrialResult, record: Path | None) -> Feedback:
    """What a strategy is told of the trial ``result`` reads back as, whose ``summary.json`` is
    ``record`` if it has one. Raises :class:`OptimizationError` for a valid trial a value of
    which no float holds exactly: told as another value, it would not be what was measured."""
    if result.status != "valid":
        return Feedback(result.status, record=record, message=result.message)
    assessment = result.assessment
    assert assessment is not None
    return Feedback(
        "valid",
        objective_values(assessment),
        constraint_values(assessment),
        assessment.feasible,
        record,
        result.message,
    )


def objective_values(assessment: Assessment) -> tuple[float, ...]:
    """The objective values of a valid candidate, in the problem's order, as floats; a value no
    float holds exactly (an integer beyond 2**53) is refused."""
    assert assessment.objectives is not None
    values = []
    for objective in assessment.objectives:
        try:
            value = float(objective.value)
        except OverflowError:
            value = None
        if value is None or value != objective.value:
            raise OptimizationError(
                f"{objective.label}: no float holds {objective.value} exactly, so the strategy "
                "cannot be told it"
            )
        values.append(value)
    return tuple(values)


def constraint_values(assessment: Assessment) -> dict[str, float]:
    """The normalized margin of each limit of an assessed candidate, by limit name
    (:attr:`~k4opt.record.Limit.normalized`: the physical margin divided by the constraint's
    scale, ``margin / scale``), zero or below when the limit is satisfied; one no float holds is
    refused."""
    values = {}
    for limit in assessment.limits:
        assert limit.satisfied is not None  # assessed
        if limit.normalized is None:
            raise OptimizationError(
                f"{limit.name}: the margin of {limit.value} to {limit.limit} in units of its "
                f"scale {limit.scale} does not fit in a float, which the strategy is told"
            )
        values[limit.name] = limit.normalized
    return values


def _tell(study: OpenStudy, result: TrialResult) -> None:
    """Tell the study's strategy how trial ``result.number`` ended, once the ledger holds it."""
    if study.strategy is None:  # recovered before the strategy is made: it is made from them
        return
    record = study.record(result.number) if result.record is not None else None
    told = feedback(result, record)
    try:
        study.strategy.tell(result.number, told)
    except Exception as exc:
        raise OptimizationError(
            f"trial {result.number}: the strategy could not take how it ended: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# --- the endings of a trial --------------------------------------------------------------------


def report(study: OpenStudy, number: int) -> TrialResult:
    """Keep trial ``number`` as its ``summary.json`` holds it, tell the strategy, and give the
    trial as that record holds it.

    The one way a trial gets its record, live or when a resume recovers it: the record is read
    back from disk and checked against the study and the earlier trials
    (:func:`~k4opt.optimization._results.record_of`), its assessment recomputed from its
    measurements, and what the strategy will be told computed, before the ledger holds it.
    Raises :class:`OptimizationError`, before anything is kept, for a record that does not hold
    up."""
    entry = study.ledger.entry(number)
    result, warnings = record_of(
        study.directory,
        study.document,
        study.scope,
        number,
        entry.values,
        study.read,
        entry.selected,
    )
    if study.scope.nested and entry.selected is None:
        raise OptimizationError(
            f"trial {number} has a record, but selected no trial of its study of the level below"
        )
    feedback(result, study.record(number))  # what the strategy is told holds up, as floats
    for warning in warnings:
        study.warn(warning)
    if study.keeper is not None:  # what its stages read, before anything of it can change
        study.keeper.trial(study.directory / trial_directory(number))
    assert result.record is not None
    assert result.sha256 is not None
    study.ledger.recorded(number, result.status, {"id": result.record, "sha256": result.sha256})
    result = replace(result, recovered=study.ledger.entry(number).recovered)
    _tell(study, result)
    return result


def _end(study: OpenStudy, number: int, kind: str, **told: str) -> TrialResult:
    """End trial ``number`` without a record, as ``kind`` (an event of the ledger) with ``told``
    (``by``, ``message``), checked first as a reader reads it back."""
    entry = study.ledger.entry(number)
    session, seq = study.ledger.session_number, study.ledger.next_seq
    ending = Ending(kind, session, seq, by=told.get("by"), message=told.get("message"))
    result, _ = trial_result(
        study.directory, study.document, study.scope, replace(entry, ending=ending), {}
    )
    getattr(study.ledger, kind)(number, **told)
    _tell(study, result)
    return result


def reject(study: OpenStudy, number: int, rejected: Rejected) -> TrialResult:
    """An evaluation rejected the candidate of trial ``number``, by its ``check`` or a
    reconstruction's job: nothing of it was created, so there is no record."""
    return _end(study, number, "rejected", by=rejected.by, message=rejected.message)


def unscore(study: OpenStudy, number: int, message: str) -> TrialResult:
    """The study of the level below that trial ``number`` stands for has no feasible trial."""
    return _end(study, number, "unscored", message=message)


def malform(
    study: OpenStudy,
    number: int,
    values: Mapping[str, Json],
    proposal: Mapping[str, Json],
    message: str,
) -> TrialResult:
    """Trial ``number`` is a proposal of ``values`` that is no candidate of the study (``message``
    says why): nothing of it is evaluated, and the ledger keeps the values as given."""
    session, seq = study.ledger.session_number, study.ledger.next_seq
    ending = Ending("malformed", session, seq, message=message)
    entry = Entry(number, session, values, proposal, seq, None, ending)  # type: ignore[arg-type]
    result, _ = trial_result(study.directory, study.document, study.scope, entry, {})
    study.ledger.malformed(number, values, proposal, message)
    _tell(study, result)
    return result


def settle(
    study: OpenStudy, number: int, output: Path, outcome: RunOutcome | Rejected | Exception
) -> TrialResult:
    """End trial ``number`` as its evaluation into ``output`` ended, and give it as kept; raise
    to stop the study."""
    try:
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, Rejected):
            return reject(study, number, outcome)
        return report(study, number)
    except KeyboardInterrupt:
        with contextlib.suppress(Exception):  # one that cannot be kept is left to --resume
            close_interrupted(study, number, output)
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise OptimizationError(
            f"trial {number} stops the study ({_failed(study, number, error)}): {error}"
        ) from exc


def _failed(study: OpenStudy, number: int, error: str) -> str:
    """End trial ``number`` as an ``error`` unless the ledger holds it ended; what became of it."""
    name = trial_directory(number)
    if not study.ledger.writable:  # a write failed: what the disk holds is the next session's
        return f"the ledger could not be written; k4opt optimize --resume settles {name}"
    entry = study.ledger.entry(number)
    if entry.ending is not None:
        return f"{name} is {entry.ending.kind}"
    if study.scope.nested:  # its study of the level below stays: a resume goes on from it
        return f"{name} was left running; k4opt optimize --resume goes on with it"
    try:
        study.ledger.error(number, error)
    except OptimizationError:
        return f"{name} could not be kept as an error; k4opt optimize --resume settles it"
    return f"{name} is an error"


def close_interrupted(study: OpenStudy, number: int, output: Path) -> None:
    """A trial that Ctrl-C, SIGTERM or another trial's error ended: kept with its record if that
    was published just before, ``interrupted`` otherwise, its directory left as it is; nothing
    if the ledger holds it ended already, or can be written no more (a resume settles it). A
    trial that stands for a study of the level below stays running: a resume goes on with it."""
    if not study.ledger.writable or not study.ledger.entry(number).running:
        return
    if (output / SUMMARY_NAME).is_file():
        try:
            report(study, number)
            return
        except K4OptError as exc:  # a record that does not hold up is no result
            if not study.ledger.writable:
                return
            study.warn(f"trial {number}: its record does not hold up ({exc}); interrupted")
    if not study.scope.nested:
        _end(study, number, "interrupted")


def check_trials(study: OpenStudy) -> list[TrialResult]:
    """Every trial of a study a session opens again, by number, as it reads back, before
    anything changes: refuse if a trial directory belongs to no trial, or if a trial does not
    hold up as :func:`~k4opt.optimization.load_study` checks it (its record is gone or was
    changed, is not the one the ledger names, or is not made for it)."""
    entries = study.ledger.ledger.entries
    known = {trial_directory(entry.number) for entry in entries}
    stray = [name for name in trial_directories(study.directory) if name not in known]
    if stray:
        raise OptimizationError(
            f"{', '.join(stray)} in {study.directory} belong(s) to no trial of the study; move "
            "them out of the study directory to resume it"
        )
    results = []
    for entry in entries:
        # the trials still to come must read what the earlier candidates' own references read
        # (``study.read``); the warnings of a record were given when it was kept
        try:
            result, _ = trial_result(
                study.directory, study.document, study.scope, entry, study.read
            )
        except OptimizationError as exc:
            raise OptimizationError(
                f"{exc}; the study resumes only with every trial as its record holds it"
            ) from None
        results.append(result)
    return results


def recover(study: OpenStudy, results: list[TrialResult], progress: Progress) -> list[TrialResult]:
    """End the trials an earlier session left running, as :func:`check_trials` read them
    (``results``), once a new session of the study began; the trials, by number, as they read
    back then. A trial left running with a record is kept with it, after the same checks as a
    live one, and one without is ``interrupted``, its directory left as it is; ``progress`` is
    called with each. A trial that stands for a study of the level below stays running,
    without a record: the run goes on with that study."""
    results = list(results)
    for entry in study.ledger.ledger.entries:
        if not entry.running:
            continue
        number = entry.number
        if study.record(number).is_file():
            results[number] = report(study, number)
        elif study.scope.nested:
            continue  # it stands for a study of the level below: the run goes on with that
        else:
            results[number] = _end(study, number, "interrupted")
        progress(results[number])
    return results
