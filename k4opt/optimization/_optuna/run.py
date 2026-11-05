"""Internal: a study run by :func:`k4opt.optimization.optimize`.

Optuna's TPE sampler proposes each candidate. k4opt evaluates it with an
:class:`~k4opt.execution.EvaluationSession` into the trial's directory, and Optuna is told the
result as the trial's ``summary.json`` says it, read back from disk and validated against the
study (:mod:`~k4opt.optimization._optuna.tell`): what the optimizer learns is always what the
record holds. A valid
candidate becomes a COMPLETE trial with its objective values and one constraint value per limit
(:func:`~k4opt.optimization._optuna.space.constraint_values`); an invalid or failed one a FAIL
trial, and the study goes on. A trial that breaks the evaluation contract, or whose record does
not hold up, is FAIL and stops the study, and the trials still running are closed as
interrupted. A candidate an evaluator rejects before anything of it is created is a FAIL trial
marked ``rejected``, and the study goes on. With more than one worker the session runs several
trials at once: the optimizer is asked for the next candidate only when a slot would be idle, and
told each trial as it ends (:func:`_run_trials`).
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any, Literal

import optuna
from optuna.storages import BaseStorage

from k4opt.evaluation import EvaluatorProtocol
from k4opt.execution import Activity, EvaluationSession
from k4opt.optimization._identity import dirty_checkouts, unhashed
from k4opt.optimization._lock import Lock
from k4opt.optimization._optuna.optimizer import (
    check_optuna_study,
    check_resume,
    new_document,
    open_optuna,
)
from k4opt.optimization._optuna.read import results_of
from k4opt.optimization._optuna.space import search_space
from k4opt.optimization._optuna.storage import (
    close,
    leaving,
    open_storage,
    redacted,
    sqlite_file,
    transient,
)
from k4opt.optimization._optuna.tell import (
    Progress,
    Warn,
    _recover,
    _Study,
)
from k4opt.optimization._optuna.tree import Tree
from k4opt.optimization._results import StudyResults, recorded_scope
from k4opt.optimization._scope import StudyScope
from k4opt.optimization._study import (
    JOURNAL_NAME,
    LOCK_NAME,
    read_study,
    write_study,
)
from k4opt.optimization.errors import OptimizationError

type Watch = Callable[[Activity], None]


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


def _run_trials(tree: Tree, session: EvaluationSession, activity: Watch | None = None) -> None:
    """Run the trials of ``tree``, as many at once as the session has slots, each told as it ends.

    The optimizer is asked for a candidate only when a slot would otherwise be idle, and after
    every trial that ended was told. Whatever stops the study first stops the evaluations still
    running; their trials are then closed as interrupted, in number order, and one that cannot
    be is left RUNNING for a resume to settle.
    """
    try:
        with session.evaluate_each(tree.requests(), activity=activity) as evaluations:
            for request, result in evaluations:
                tree.ended(request, result)
    finally:  # nothing runs any more
        tree.close()


def optimize(
    problem: str | os.PathLike[str],
    implementations: Sequence[type[EvaluatorProtocol]],
    output: str | os.PathLike[str],
    *,
    trials: int,
    seed: int | None,
    resume: bool,
    storage: str | None,
    workers: int | Literal["auto", "inline"],
    warn: Warn,
    progress: Progress,
    activity: Watch | None = None,
    reuse: Sequence[str | os.PathLike[str]] = (),
    fixed: Mapping[str, Any] | None = None,
) -> StudyResults:
    # a resume may ask for none: it goes on with the trials an interrupted session left
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < (0 if resume else 1):
        kind = "a non-negative" if resume else "a positive"
        raise OptimizationError(f"trials must be {kind} integer, got {trials!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise OptimizationError(f"seed must be an integer, got {seed!r}")
    if seed is not None and not 0 <= seed < 2**32:
        raise OptimizationError(f"seed must be in [0, 2**32), got {seed}")
    if resume and seed is not None:
        raise OptimizationError("a resumed study keeps the seed it was made with; leave out --seed")
    if storage is not None and transient(storage):
        raise OptimizationError(
            f"--storage {redacted(storage)}: a SQLite database in memory or a temporary one keeps "
            "no trials once k4opt exits, so the study could not be resumed or read; name a "
            "database file, or leave out --storage to keep the trials in the study directory"
        )
    directory = Path(os.path.abspath(Path(output).expanduser()))

    # its workers stop with the study however it ends
    with EvaluationSession.open(problem, implementations, workers=workers) as session:
        session.reuse(reuse)
        return _run_study(
            session,
            directory,
            trials=trials,
            seed=seed,
            resume=resume,
            storage=storage,
            warn=warn,
            progress=progress,
            activity=activity,
            fixed=fixed,
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
    activity: Watch | None,
    fixed: Mapping[str, Any] | None = None,
) -> StudyResults:
    # everything that can be checked before a file is created
    spec = session.spec
    session.check_limits()  # the problem's time limits can be kept where the trials run
    scope = StudyScope.of(spec, fixed)
    # the evaluators' settings were checked when the session opened
    space = search_space(spec, scope.searched)
    identity: dict[str, Any] = session.identity()
    reasons = unhashed(identity)
    if reasons:
        raise OptimizationError(f"the study could not identify what it runs: {'; '.join(reasons)}")
    names = scope.limit_names
    for warning in session.warnings:
        warn(warning)
    if (notice := session.notice()) is not None:
        warn(notice)
    if not resume:
        for warning in dirty_checkouts(identity["implementations"]):
            warn(warning)
    if resume:
        document = read_study(directory)
        if (document.get("scope") or {}).get("parent") is not None:
            raise OptimizationError(
                f"{directory} is a study of a level below the top, which runs with the study it "
                "belongs to; resume that study"
            )
    else:
        document = new_document(scope, identity, space, names, seed, storage)

    database: BaseStorage | None = None  # made in the directory, or reached once below
    try:
        if not resume:
            # the directory is checked before a database is reached, so one in use leaves the
            # database untouched; a database is reached, and set up once, before a file is made,
            # so a wrong URL makes none
            _refuse_used(directory)
            _refuse_inside(directory, storage)
            database = None if storage is None else open_storage(directory, storage)
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OptimizationError(f"cannot create {directory}: {exc}") from exc
        with _quiet(), _terminate_as_interrupt(), Lock(directory):
            if resume:
                check_resume(document, identity, scope, space, names, storage, warn)
            else:
                _refuse_used(directory, LOCK_NAME)  # again, now that no one else can start here
                write_study(directory, document)
            database = database or open_storage(directory, storage)
            study = open_optuna(directory, document, database, resume)
            run = _Study(directory, document, scope, study, warn)
            if resume:
                check_optuna_study(study, document)
                _recover(run, progress)
                session.reuse([directory])  # every trial's record held up: their stages serve
            tree = Tree(session, run, space, trials, identity, storage, database, progress, warn)
            _run_trials(tree, session, activity)
            # read back as load_study reads it, from the study still open: with the problem
            # study.json records, whose titles a resume with other titles leaves as they were
            return results_of(directory, document, recorded_scope(document), study, storage=storage)
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
    """A new study starts in a directory that does not exist or holds nothing (but ``ours``),
    also through a symlink; a symlink to nothing, or to no directory, is in use."""
    held = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    if [name for name in held if name not in ours] or (
        (directory.exists() or directory.is_symlink()) and not directory.is_dir()
    ):
        raise OptimizationError(
            f"{directory} is in use; resume the study in it with --resume, or choose a new "
            "directory"
        )


def _refuse_inside(directory: Path, storage: str | None) -> None:
    """A new study's SQLite database lies outside its directory, which a new study finds new or
    empty and fills with nothing but its own files (also not through a symlink)."""
    file = None if storage is None else sqlite_file(storage)
    if storage is None or file is None:
        return
    for database, study in ((file, directory), (file.resolve(), directory.resolve())):
        if database.is_relative_to(study):
            raise OptimizationError(
                f"--storage {redacted(storage)}: the database {file} would be in the study "
                f"directory {directory}; keep it elsewhere, or leave out --storage to keep the "
                f"trials in {JOURNAL_NAME} there"
            )
