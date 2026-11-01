"""Internal: a study run by :func:`k4opt.optimization.optimize`.

Optuna's TPE sampler proposes each candidate. k4opt evaluates it with an
:class:`~k4opt.problem.EvaluationSession` into the trial's directory, and Optuna is told the
result as the trial's ``summary.json`` says it, read back from disk and validated against the
study (:func:`_report`): what the optimizer learns is always what the record holds. A valid
candidate becomes a COMPLETE trial with its objective values and one constraint value per limit
(:func:`~k4opt.optimization._optuna.space.constraint_values`); an invalid or failed one a FAIL
trial, and the study goes on. A trial that breaks the evaluation contract, or whose record does
not hold up, is FAIL and stops the study, and the trials still running are closed as
interrupted. With more than one job the session runs several trials at once: the optimizer is
asked for the next candidate only when a job would be idle, and told each trial as it ends
(:func:`_run_trials`).
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
from k4opt.evaluation import EvaluatorProtocol
from k4opt.optimization._optuna.optimizer import (
    CONSTANT_LIAR,
    STARTUP_TRIALS,
    check_optuna_study,
    check_resume,
    new_document,
    open_optuna,
)
from k4opt.optimization._optuna.read import trial_result
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
from k4opt.problem import (
    SUMMARY_NAME,
    Assessment,
    EvaluationSession,
    ProblemSpec,
    RecordError,
    RunOutcome,
    RunRequest,
)

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
    # the diagnostics that did not succeed, which the status leaves out (checked by from_summary)
    trial.set_user_attr("k4opt.diagnostics", len(summary["diagnostics"]))
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


def _stored(run: _Study, number: int) -> FrozenTrial | None:
    """Trial ``number`` as the storage holds it now; ``None`` if it cannot be read.

    Read back before a trial is told anything after an error, since a ``tell`` can commit and
    then raise: a trial is told once."""
    try:
        return next(t for t in run.study.get_trials(deepcopy=False) if t.number == number)
    except Exception:
        return None


def _requests(
    run: _Study,
    space: Mapping[str, BaseDistribution],
    seed: int,
    trials: int,
    in_flight: dict[int, tuple[Trial, Path]],
) -> Iterator[RunRequest]:
    """Ask for ``trials`` candidates, one each time the session takes the next; each trial is in
    ``in_flight`` from the moment it is asked for until it is settled."""
    for _ in range(trials):
        trial = run.study.ask(fixed_distributions=dict(space))
        name = trial_directory(trial.number)
        output = run.directory / name
        in_flight[trial.number] = (trial, output)
        trial.set_user_attr("k4opt.dir", name)
        trial.set_user_attr("k4opt.sampler_seed", seed)
        candidate = {key: trial.params[key] for key in space}
        yield RunRequest(candidate, output, {"study": run.document["id"], "trial": trial.number})


def _settle(run: _Study, trial: Trial, output: Path, result: RunOutcome | Exception) -> FrozenTrial:
    """Tell the optimizer how the trial's evaluation ended; raise to stop the study."""
    try:
        if isinstance(result, Exception):
            raise result
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
    """Tell ``trial`` FAIL with ``error`` unless the storage holds it ended; what became of it."""
    name = trial_directory(trial.number)
    stored = _stored(run, trial.number)
    if stored is None:
        return f"the state of {name} could not be read; k4opt optimize --resume settles it"
    if stored.state != TrialState.RUNNING:
        return f"{name} is {stored.state.name}"
    try:
        _fail(run, trial, error)
    except Exception:
        return f"{name} could not be told FAIL; k4opt optimize --resume settles it"
    return f"{name} is FAIL"


def _close_interrupted(run: _Study, trial: Trial, output: Path) -> None:
    """A trial that Ctrl-C, SIGTERM or another trial's error ended: told its record if that was
    published just before, FAIL otherwise, its directory left as it is; nothing if the storage
    holds it ended already, or cannot say (a resume settles it)."""
    stored = _stored(run, trial.number)
    if stored is None or stored.state != TrialState.RUNNING:
        return
    if (output / SUMMARY_NAME).is_file():
        # the interruption may have come within _report, after the constraints were set
        try:
            _report(run, trial, stored.constraints)
            return
        except K4OptError:
            pass  # a record that does not hold up is no result
    _fail(run, trial)


def _run_trials(
    run: _Study,
    session: EvaluationSession,
    space: Mapping[str, BaseDistribution],
    seed: int,
    trials: int,
    progress: Progress,
) -> None:
    """Run ``trials`` new trials, as many at once as the session has jobs, each told as it ends.

    The optimizer is asked for a candidate only when a job would otherwise be idle, and after
    every trial that ended was told. Whatever stops the study first stops the evaluations still
    running; their trials are then closed as interrupted, in number order, and one that cannot
    be is left RUNNING for a resume to settle.
    """
    in_flight: dict[int, tuple[Trial, Path]] = {}
    try:
        requests = _requests(run, space, seed, trials, in_flight)
        with session.evaluate_each(requests) as evaluations:
            for request, result in evaluations:
                assert isinstance(request.run, Mapping)
                number = request.run["trial"]
                assert isinstance(number, int)
                trial, output = in_flight.pop(number)  # settled by _settle, whatever it raises
                progress(_trial_report(run, _settle(run, trial, output, result)))
    finally:  # nothing runs any more
        for number in sorted(in_flight):
            with contextlib.suppress(Exception):  # one that cannot be told: --resume
                _close_interrupted(run, *in_flight[number])


def _recover(run: _Study, progress: Progress) -> None:
    """Finish the trials a session left running, before any new one: one without a record is
    FAIL, its directory left as it is, and one with a record is told it; ``progress`` is called
    with each as it is finished. Before anything changes, the resume refuses if a trial
    directory belongs to no trial, or if a finished trial does not hold what its record gives,
    checked as :func:`~k4opt.optimization.load_study` checks it: its record is gone or was
    changed, its state, status, candidate, values or constraint values are not the record's, or
    k4opt did not make it."""
    trials = run.study.get_trials(deepcopy=False)
    known = {trial.user_attrs.get("k4opt.dir", trial_directory(trial.number)) for trial in trials}
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
            trial_result(run.directory, run.document, run.spec, frozen, run.read)
        except OptimizationError as exc:
            raise OptimizationError(
                f"{exc}; the study resumes only with every trial as its record holds it"
            ) from None
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


def _trial_report(run: _Study, frozen: FrozenTrial) -> TrialReport:
    attrs = frozen.user_attrs
    status = attrs.get("k4opt.status")
    if status is None:
        status = "error" if "k4opt.error" in attrs else "interrupted"
        if frozen.state == TrialState.RUNNING:
            status = "running"
    names = [o.label for o in run.spec.objectives]
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
        diagnostics=attrs.get("k4opt.diagnostics", 0),
    )


def _study_report(run: _Study) -> StudyReport:
    trials = tuple(_trial_report(run, t) for t in run.study.get_trials(deepcopy=False))
    return StudyReport(
        name=run.study.study_name,
        directory=run.directory,
        id=run.document["id"],
        trials=trials,
        pareto=tuple(sorted(t.number for t in run.study.best_trials)),
        objectives=tuple(o.label for o in run.spec.objectives),
        constraints=tuple(limit_names(run.spec)),
    )


def optimize(
    problem: str | os.PathLike[str],
    implementations: Sequence[type[EvaluatorProtocol]],
    output: str | os.PathLike[str],
    *,
    trials: int,
    seed: int | None,
    resume: bool,
    storage: str | None,
    timeout: float | None,
    jobs: int,
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
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise OptimizationError(f"jobs must be a positive integer, got {jobs!r}")
    directory = Path(os.path.abspath(Path(output).expanduser()))

    # its workers, with more than one job, stop with the study however it ends
    with EvaluationSession.open(problem, implementations, timeout=timeout, jobs=jobs) as session:
        return _run_study(
            session,
            directory,
            trials=trials,
            seed=seed,
            resume=resume,
            storage=storage,
            warn=warn,
            progress=progress,
        )


def _run_study(
    session: EvaluationSession,
    directory: Path,
    *,
    trials: int,
    seed: int | None,
    resume: bool,
    storage: str | None,
    warn: Warn,
    progress: Progress,
) -> StudyReport:
    # everything that can be checked before a file is created
    spec = session.spec
    space = search_space(spec)  # the evaluators' settings were checked when the session opened
    identity: dict[str, Any] = session.identity()
    reasons = unhashed(identity)
    if reasons:
        raise OptimizationError(f"the study could not identify what it runs: {'; '.join(reasons)}")
    names = limit_names(spec)
    for warning in session.warnings:
        warn(warning)
    if session.notice is not None:
        warn(session.notice)
    if not resume:
        for warning in dirty_checkouts(identity["implementations"]):
            warn(warning)
    metrics = {c.ref for c in spec.constraints}
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
                seed=seed_now, n_startup_trials=STARTUP_TRIALS, constant_liar=CONSTANT_LIAR
            )
            _run_trials(run, session, space, seed_now, trials, progress)
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
