"""Internal: a study run by :func:`k4opt.optimization.optimize`.

Optuna's TPE sampler proposes each candidate. k4opt evaluates it with an
:class:`~k4opt.problem.EvaluationSession` into the trial's directory, and Optuna is told the
result as the trial's ``summary.json`` says it, read back from disk and validated against the
study (:func:`_report`): what the optimizer learns is always what the record holds. A valid
candidate becomes a COMPLETE trial with its objective values and one constraint value per limit
(:func:`~k4opt.optimization._optuna.space.constraint_values`); an invalid or failed one a FAIL
trial, and the study goes on. A trial that breaks the evaluation contract, or whose record does
not hold up, is FAIL and stops the study.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import optuna
from optuna.distributions import BaseDistribution
from optuna.storages import BaseStorage
from optuna.trial import FrozenTrial, Trial, TrialState

from k4opt.errors import K4OptError
from k4opt.evaluation import EvaluatorFactory
from k4opt.optimization._optuna.optimizer import (
    STARTUP_TRIALS,
    check_optuna_study,
    check_resume,
    new_document,
    open_optuna,
)
from k4opt.optimization._optuna.space import constraint_values, objective_values, search_space
from k4opt.optimization._optuna.storage import close, leaving, open_storage
from k4opt.optimization._results import limit_names
from k4opt.optimization._study import (
    LOCK_NAME,
    Lock,
    StudyReport,
    TrialReport,
    check_candidate_files,
    check_record,
    dirty_checkouts,
    read_study,
    session_seed,
    trial_directories,
    trial_directory,
    unhashed,
    write_study,
)
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import SUMMARY_NAME, Assessment, EvaluationSession, ProblemSpec, RecordError

_MESSAGE = 1000  # characters of a message kept in the optimizer's storage; the record has all


type Warn = Callable[[str], None]


type Progress = Callable[[TrialReport], None]


@contextlib.contextmanager
def _terminate_as_interrupt() -> Iterator[None]:
    """SIGTERM (a batch system ending the job) stops the study as Ctrl-C does, so the trial is
    closed and the lock removed; in the main thread only, where signals are handled."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Optuna logs every trial at INFO; k4opt reports them itself."""
    level = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        yield
    finally:
        optuna.logging.set_verbosity(level)


@dataclass(frozen=True, slots=True)
class _Study:
    directory: Path
    document: Mapping[str, Any]  # study.json
    spec: ProblemSpec
    study: optuna.Study
    warn: Warn
    # what each reference only candidates name read, and the first trial that read it
    read: dict[str, tuple[int, object]] = field(default_factory=dict)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _report(
    run: _Study, trial: Trial, existing: Mapping[str, float], *, recovered: bool = False
) -> FrozenTrial:
    """Tell the optimizer the result of ``trial`` as its ``summary.json`` holds it.

    The one way a trial is told, live or when a resume recovers it: the record is read back from
    disk, checked against the study (:func:`~k4opt.optimization._study.check_record`) and the
    earlier trials (:func:`~k4opt.optimization._study.check_candidate_files`), and its
    assessment recomputed from its measurements. ``existing`` are the constraint values the
    trial already has, which must equal the recomputed ones. Raises
    :class:`OptimizationError`, before anything is told, for a record that does not hold up.
    """
    number, name = trial.number, trial_directory(trial.number)
    path = run.directory / name / SUMMARY_NAME
    try:
        data = path.read_bytes()
        summary = json.loads(data)
    except (OSError, ValueError) as exc:
        raise OptimizationError(f"trial {number}: cannot read {path}: {exc}") from exc
    warnings = check_record(run.document, summary, number, trial.params)
    check_candidate_files(run.read, summary, number)
    try:
        assessment = Assessment.from_summary(run.spec, summary)
    except RecordError as exc:
        raise OptimizationError(f"trial {number}: {path}: {exc}") from None
    valid = summary["status"] == "valid"
    values = objective_values(assessment) if valid else None
    constraints = constraint_values(assessment) if valid else {}
    for key, value in constraints.items():
        if key in existing and existing[key] != value:
            raise OptimizationError(
                f"trial {number}: the optimizer holds {existing[key]} for {key}, but the "
                f"record gives {value}"
            )
    for warning in warnings:
        run.warn(f"trial {number}: {warning}")
    trial.set_user_attr("k4opt.status", summary["status"])
    record = {"id": summary.get("id"), "sha256": _sha256(data), "path": f"{name}/{SUMMARY_NAME}"}
    trial.set_user_attr("k4opt.record", record)
    geometry = summary.get("geometry")
    if isinstance(geometry, Mapping):
        trial.set_user_attr("k4opt.fingerprint", geometry.get("fingerprint"))
    if isinstance(summary.get("message"), str):
        trial.set_user_attr("k4opt.message", summary["message"][:_MESSAGE])
    if recovered:
        trial.set_user_attr("k4opt.recovered", True)
    if values is None:
        return run.study.tell(trial, state=TrialState.FAIL)
    for key, value in constraints.items():
        if key not in existing:
            trial.set_constraint(key, value)
    trial.set_user_attr("k4opt.feasible", assessment.feasible)
    return run.study.tell(trial, values)


def _fail(run: _Study, trial: Trial, error: str | None = None) -> FrozenTrial:
    if error is not None:
        trial.set_user_attr("k4opt.error", error[:_MESSAGE])
    return run.study.tell(trial, state=TrialState.FAIL)


def _evaluate(
    run: _Study, session: EvaluationSession, space: Mapping[str, BaseDistribution], seed: int
) -> FrozenTrial:
    """Ask for one candidate, evaluate it and tell the optimizer; raise to stop the study."""
    trial = run.study.ask(fixed_distributions=dict(space))
    name = trial_directory(trial.number)
    trial.set_user_attr("k4opt.dir", name)
    trial.set_user_attr("k4opt.sampler_seed", seed)
    candidate = {key: trial.params[key] for key in space}
    output = run.directory / name
    reference = {"study": run.document["id"], "trial": trial.number}
    try:
        session.evaluate(candidate, output, run=reference)
        return _report(run, trial, {})
    except KeyboardInterrupt:
        _close_interrupted(run, trial, output)
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _fail(run, trial, error)
        raise OptimizationError(
            f"trial {trial.number} stops the study ({name} is FAIL): {error}"
        ) from exc


def _close_interrupted(run: _Study, trial: Trial, output: Path) -> None:
    """A trial that Ctrl-C or SIGTERM ended: told its record if that was published just before,
    FAIL otherwise, its directory left as it is."""
    if (output / SUMMARY_NAME).is_file():
        # the interruption may have come within _report, after the constraints were set
        (frozen,) = [t for t in run.study.get_trials(deepcopy=False) if t.number == trial.number]
        try:
            _report(run, trial, frozen.constraints)
            return
        except K4OptError:
            pass  # a record that does not hold up is no result
    _fail(run, trial)


def _recover(run: _Study, progress: Progress) -> None:
    """Finish the trials a session left running, before any new one: one without a record is
    FAIL, its directory left as it is, and one with a record is told it; ``progress`` is called
    with each as it is finished. Before anything changes, the resume refuses if a trial
    directory belongs to no trial, or if the record a finished trial was told is gone or was
    changed (its SHA-256)."""
    trials = run.study.get_trials(deepcopy=False)
    known = {trial.user_attrs.get("k4opt.dir", trial_directory(trial.number)) for trial in trials}
    stray = [name for name in trial_directories(run.directory) if name not in known]
    if stray:
        raise OptimizationError(
            f"{', '.join(stray)} in {run.directory} belong(s) to no trial of the study; move "
            "them out of the study directory to resume it"
        )
    for frozen in trials:  # the evidence of what the optimizer was told is still there
        _check_told_record(run, frozen)
    for frozen in trials:
        if frozen.state != TrialState.RUNNING:
            continue
        trial = Trial(run.study, frozen._trial_id)  # the one private access of Optuna
        if (run.directory / trial_directory(frozen.number) / SUMMARY_NAME).is_file():
            finished = _report(run, trial, frozen.constraints, recovered=True)
        else:
            trial.set_user_attr("k4opt.recovered", True)
            finished = run.study.tell(trial, state=TrialState.FAIL)
        progress(_trial_report(run, finished))


def _check_told_record(run: _Study, frozen: FrozenTrial) -> None:
    """A trial the optimizer was told a record of still has that record, byte for byte, and
    every other finished trial is one k4opt made: interrupted, or one that stopped the study.

    k4opt marks each trial it asks for with ``k4opt.dir`` at once, and recovery marks one that
    a killed session left without it (``k4opt.recovered``). A finished trial with a result but
    no record, or with no mark of k4opt at all, was added by another client of the storage: it
    would count among the study's trials (and so change the next session's seed) without being
    one of its evaluations.
    """
    attrs = frozen.user_attrs
    told = attrs.get("k4opt.record")
    if frozen.state == TrialState.RUNNING:
        return  # recovery decides about it
    if not isinstance(told, Mapping):
        ours = "k4opt.dir" in attrs or attrs.get("k4opt.recovered") is True
        if frozen.state != TrialState.FAIL or "k4opt.status" in attrs or not ours:
            raise OptimizationError(
                f"trial {frozen.number} is {frozen.state.name} without the record k4opt tells a "
                "trial with, or any mark of k4opt, so it was not made by k4opt optimize; resume "
                "needs every trial to be one of the study's own"
            )
        return
    path = run.directory / str(told.get("path"))
    try:
        data: bytes | None = path.read_bytes()
    except OSError:
        data = None
    if data is None or _sha256(data) != told.get("sha256"):
        state = "is gone" if data is None else "is not the record the optimizer was told"
        raise OptimizationError(
            f"trial {frozen.number}: {path} {state}, so nothing shows what its values rest on; "
            "put the record back to resume the study"
        )
    # the trials still to come must read what its candidate's own references read
    check_candidate_files(run.read, json.loads(data), frozen.number)


def _trial_report(run: _Study, frozen: FrozenTrial) -> TrialReport:
    attrs = frozen.user_attrs
    status = attrs.get("k4opt.status")
    if status is None:
        status = "error" if "k4opt.error" in attrs else "interrupted"
        if frozen.state == TrialState.RUNNING:
            status = "running"
    names = [f"{o.stage}/{o.metric}" for o in run.spec.objectives]
    values = frozen.values if frozen.state == TrialState.COMPLETE else None
    return TrialReport(
        number=frozen.number,
        directory=run.directory / trial_directory(frozen.number),
        candidate=dict(frozen.params),
        status=status,
        feasible=attrs.get("k4opt.feasible"),
        objectives=None if values is None else dict(zip(names, values, strict=True)),
        record=(attrs.get("k4opt.record") or {}).get("id"),
        message=attrs.get("k4opt.message", attrs.get("k4opt.error")),
        recovered=bool(attrs.get("k4opt.recovered", False)),
    )


def _study_report(run: _Study) -> StudyReport:
    trials = tuple(_trial_report(run, t) for t in run.study.get_trials(deepcopy=False))
    return StudyReport(
        name=run.study.study_name,
        directory=run.directory,
        id=run.document["id"],
        trials=trials,
        pareto=tuple(sorted(t.number for t in run.study.best_trials)),
        objectives=tuple(f"{o.stage}/{o.metric}" for o in run.spec.objectives),
        constraints=tuple(limit_names(run.spec)),
    )


def optimize(
    problem: str | os.PathLike[str],
    implementations: Sequence[EvaluatorFactory],
    output: str | os.PathLike[str],
    *,
    trials: int,
    seed: int | None,
    resume: bool,
    storage: str | None,
    timeout: float | None,
    warn: Warn,
    progress: Progress,
) -> StudyReport:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise OptimizationError(f"trials must be a positive integer, got {trials!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise OptimizationError(f"seed must be an integer, got {seed!r}")
    if seed is not None and not 0 <= seed < 2**32:
        raise OptimizationError(f"seed must be in [0, 2**32), got {seed}")
    if resume and seed is not None:
        raise OptimizationError("a resumed study keeps the seed it was made with; leave out --seed")
    directory = Path(os.path.abspath(Path(output).expanduser()))

    # everything that can be checked before a file is created
    session = EvaluationSession.open(problem, implementations, timeout=timeout)
    spec = session.spec
    space = search_space(spec)  # the evaluators' settings were checked when the session opened
    identity: dict[str, Any] = session.identity()
    reasons = unhashed(identity)
    if reasons:
        raise OptimizationError(f"the study could not identify what it runs: {'; '.join(reasons)}")
    names = limit_names(spec)
    for warning in session.warnings:
        warn(warning)
    if not resume:
        for warning in dirty_checkouts(identity["evaluators"]):
            warn(warning)
    metrics = {(c.stage, c.metric) for c in spec.constraints}
    if len(metrics) > 1:
        warn(
            f"constraints on {len(metrics)} metrics: while feasible trials are scarce, the sampler "
            "ranks infeasible ones by the sum of their violations, which adds values in different "
            "units (docs/optimization.md)"
        )
    if resume:
        document = read_study(directory)
    else:
        document = new_document(spec, identity, space, names, seed, storage)

    database: BaseStorage | None = None  # made in the directory, or reached once below
    try:
        if not resume:
            # a database is reached, and set up once, before a file is made: a wrong URL makes none
            database = None if storage is None else open_storage(directory, storage)
            _refuse_used(directory)
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OptimizationError(f"cannot create {directory}: {exc}") from exc
        with _quiet(), _terminate_as_interrupt(), Lock(directory):
            if resume:
                check_resume(document, identity, spec, space, names, storage, warn)
            else:
                _refuse_used(directory, LOCK_NAME)  # again, now that no one else can start here
                write_study(directory, document)
            database = database or open_storage(directory, storage)
            study = open_optuna(directory, document, database, resume)
            run = _Study(directory, document, spec, study, warn)
            if resume:
                check_optuna_study(study, document)
                _recover(run, progress)
            seed = document["optimizer"]["sampler"]["seed"]
            seed_now = session_seed(seed, len(study.trials))
            study.sampler = optuna.samplers.TPESampler(
                seed=seed_now, n_startup_trials=STARTUP_TRIALS
            )
            for _ in range(trials):
                progress(_trial_report(run, _evaluate(run, session, space, seed_now)))
            return _study_report(run)
    except Exception as exc:
        replacement, secret = leaving(exc, storage)
        if replacement is None:
            raise
        if secret:  # nothing of the original, which names it, is kept for a traceback
            raise replacement from None
        raise replacement from exc
    finally:
        close(database)


def _refuse_used(directory: Path, *ours: str) -> None:
    """A new study starts in a directory that does not exist or holds nothing (but ``ours``)."""
    held = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    if [name for name in held if name not in ours] or (
        directory.exists() and not directory.is_dir()
    ):
        raise OptimizationError(
            f"{directory} is in use; resume the study in it with --resume, or choose a new "
            "directory"
        )
