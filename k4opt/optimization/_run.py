"""Internal: a study run by :func:`k4opt.optimization.optimize`, whatever its strategy.

The study's strategy proposes each candidate (:mod:`._strategy`); k4opt checks it, evaluates it
with an :class:`~k4opt.execution.EvaluationSession` into the trial's directory, and keeps the
trial in the study's ledger as its ``summary.json`` says it, read back from disk and validated
against the study (:mod:`._trials`), before the strategy is told: what is kept, and what the
strategy learns, is always what the record holds. An invalid or failed candidate, one an
evaluator rejects and a proposal that is no candidate end their trial, and the study goes on; a
trial that breaks the evaluation contract, or whose record does not hold up, stops it, and the
trials still running are ended as interrupted. With more than one worker the session runs
several trials at once: the strategy is asked for the next candidate only when a slot would be
idle, and told each trial as it ends (:mod:`._tree`).
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

from k4opt.evaluation import EvaluatorProtocol
from k4opt.execution import Activity, EvaluationSession
from k4opt.frozen import same_json
from k4opt.optimization._custom import Custom
from k4opt.optimization._custom import revision_warnings as custom_revision_warnings
from k4opt.optimization._identity import (
    differences,
    digest,
    dirty_checkouts,
    program_differences,
    revision_warnings,
    unhashed,
)
from k4opt.optimization._keep import Keeper
from k4opt.optimization._lock import Lock
from k4opt.optimization._proposal import Strategy, StudyContext
from k4opt.optimization._read import results_of
from k4opt.optimization._results import StudyResults, recorded_scope
from k4opt.optimization._scope import StudyScope, check_trials
from k4opt.optimization._strategy import STRATEGIES, new_strategy
from k4opt.optimization._strategy import differences as strategy_differences
from k4opt.optimization._study import LOCK_NAME, new_document, read_study, write_study
from k4opt.optimization._tree import Tree, open_ledger
from k4opt.optimization._trials import OpenStudy, Progress, Warn
from k4opt.optimization.errors import OptimizationError
from k4opt.program import Observation

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


def _run_trials(tree: Tree, session: EvaluationSession, activity: Watch | None = None) -> None:
    """Run the trials of ``tree``, as many at once as the session has slots, each ended as it ends.

    The strategy is asked for a candidate only when a slot would otherwise be idle, and after
    every trial that ended was kept and told. Whatever stops the study first stops the
    evaluations still running; their trials are then ended as interrupted, in number order, and
    one that cannot be is left running for a resume to settle.
    """
    try:
        with session.evaluate_each(tree.requests(), activity=activity) as evaluations:
            for request, result in evaluations:
                tree.ended(request, result)
    finally:  # nothing runs any more
        tree.close()


def check_resume(
    document: Mapping[str, Any],
    identity: Mapping[str, Any],
    scope: StudyScope,
    warn: Warn,
    custom: Custom | None = None,
) -> None:
    """Refuse to resume a study with anything it compares otherwise than it was made with, and
    one whose recorded strategy is not the one this k4opt would run it with, or for a strategy of
    one's own, not ``custom``, the one the session is given."""
    found = differences(document, identity) + program_differences(document, identity)
    if not same_json(document.get("scope"), scope.to_dict()):
        found.append(
            f"it searches with {document.get('scope')!r}, not {scope.to_dict()!r} (give the same "
            "--fix as when it was made)"
        )
    found += strategy_differences(document.get("strategy"), scope, custom)
    if digest(document) != document.get("identity"):
        found.append("its study.json does not hold together (its identity digest differs)")
    if found:
        raise OptimizationError(
            f"cannot resume {document.get('name', document['id'])}: {'; '.join(found)}. Start a "
            "new study for this"
        )
    for warning in revision_warnings(document["implementations"], identity["implementations"]):
        warn(warning)
    if custom is not None:
        for warning in custom_revision_warnings(document["strategy"], custom):
            warn(warning)


def optimize(
    problem: str | os.PathLike[str],
    implementations: Sequence[type[EvaluatorProtocol]],
    output: str | os.PathLike[str],
    *,
    trials: int,
    strategy: str | type | None,
    grid: Mapping[str, Sequence[object]] | None,
    seed: int | None,
    resume: bool,
    workers: int | Literal["auto", "inline"],
    warn: Warn,
    progress: Progress,
    activity: Watch | None = None,
    reuse: Sequence[str | os.PathLike[str]] = (),
    fixed: Mapping[str, Any] | None = None,
    observe: Observation = "files",
    strategy_settings: Mapping[str, Any] | None = None,
) -> StudyResults:
    check_trials(trials, resume)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise OptimizationError(f"seed must be an integer, got {seed!r}")
    if seed is not None and not 0 <= seed < 2**32:
        raise OptimizationError(f"seed must be in [0, 2**32), got {seed}")
    if resume and (seed is not None or isinstance(strategy, str) or grid is not None):
        raise OptimizationError(
            "a resumed study keeps the strategy it was made with, its seed and its grid; leave "
            "out --strategy, --grid and --seed (but give a strategy of its own again)"
        )
    custom = _custom(strategy, strategy_settings, grid, seed)
    directory = Path(os.path.abspath(Path(output).expanduser()))

    # its workers stop with the study however it ends
    with EvaluationSession.open(
        problem, implementations, workers=workers, observe=observe
    ) as session:
        session.reuse(reuse)
        return _run_study(
            session,
            directory,
            trials=trials,
            strategy=strategy if isinstance(strategy, str) else None,
            custom=custom,
            grid=grid,
            seed=seed,
            resume=resume,
            warn=warn,
            progress=progress,
            activity=activity,
            fixed=fixed,
        )


def _custom(
    strategy: object,
    settings: Mapping[str, Any] | None,
    grid: Mapping[str, Sequence[object]] | None,
    seed: int | None,
) -> Custom | None:
    """The strategy of one's own a session runs, if ``strategy`` is one (a class
    :func:`~k4opt.optimization.load_strategy` loaded), with ``settings``; refuse a name of no
    strategy, and what is given for another kind of strategy than the one named."""
    if strategy is not None and not isinstance(strategy, (str, type)):
        raise OptimizationError(
            f"strategy must be one of {', '.join(STRATEGIES)} or a class load_strategy loaded, "
            f"not {strategy!r}"
        )
    if isinstance(strategy, str) and strategy not in STRATEGIES:
        raise OptimizationError(
            f"strategy {strategy!r} is none of {', '.join(STRATEGIES)}, nor FILE.py:Class"
        )
    if not isinstance(strategy, type):
        if settings is not None:
            raise OptimizationError(
                "--strategy-settings gives the settings of a strategy of one's own "
                "(--strategy FILE.py:Class)"
            )
        return None
    if seed is not None:
        raise OptimizationError(
            "--seed seeds Optuna's sampler; a strategy of one's own takes a seed in its settings"
        )
    if grid is not None:
        raise OptimizationError("--grid gives the values of the grid strategy: add --strategy grid")
    return Custom.of(strategy, settings)


def _run_study(
    session: EvaluationSession,
    directory: Path,
    *,
    trials: int,
    strategy: str | None,
    custom: Custom | None,
    grid: Mapping[str, Sequence[object]] | None,
    seed: int | None,
    resume: bool,
    warn: Warn,
    progress: Progress,
    activity: Watch | None,
    fixed: Mapping[str, Any] | None = None,
) -> StudyResults:
    # everything that can be checked before a file is created
    spec = session.spec
    session.check_limits()  # the problem's time limits can be kept where the trials run
    scope = StudyScope.of(spec, fixed)
    if custom is not None and scope.nested:
        raise OptimizationError(
            f"strategy {custom.name}: a strategy of one's own runs a problem without inner levels, "
            "for now; a strategy per level is still to come (docs/optimization.md)"
        )
    # the evaluators' settings were checked when the session opened
    identity: dict[str, Any] = session.identity()
    reasons = unhashed(identity)
    if reasons:
        raise OptimizationError(f"the study could not identify what it runs: {'; '.join(reasons)}")
    for warning in session.warnings:
        warn(warning)
    if (notice := session.notice()) is not None:
        warn(notice)
    made: Strategy | None = None
    if resume:
        document = read_study(directory)
        if (document.get("scope") or {}).get("parent") is not None:
            raise OptimizationError(
                f"{directory} is a study of a level below the top, which runs with the study it "
                "belongs to; resume that study"
            )
    else:
        for warning in dirty_checkouts(identity["implementations"]):
            warn(warning)
        if custom is None:
            chosen = new_strategy(strategy or STRATEGIES[0], scope, seed=seed, grid=grid)
        else:
            for warning in custom.dirty():
                warn(warning)
            chosen = custom.record()
        document = new_document(scope, identity, chosen)
        _refuse_used(directory)
        if custom is not None:  # made before anything is created: a bad setting creates nothing
            made = custom.make(StudyContext.of(scope, document["id"]))
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OptimizationError(f"cannot create {directory}: {exc}") from exc

    def check(document: Mapping[str, Any], identity: Mapping[str, Any], scope: StudyScope) -> None:
        check_resume(document, identity, scope, warn, custom)

    with _terminate_as_interrupt(), Lock(directory):
        if resume:
            check_resume(document, identity, scope, warn, custom)
            recorded_scope(document)  # it holds together, and names its study and strategy
        else:
            _refuse_used(directory, LOCK_NAME)  # again, now that no one else can start here
            write_study(directory, document)
        ledger, cut = open_ledger(directory, document, resume)
        try:
            if cut:
                warn(f"{ledger.path}: a line a killed process cut short was removed")
            keeper = Keeper(directory, warn)
            # before any trial runs, or a resume recovers one
            keeper.open(identity, None if custom is None else custom.kept())
            study = OpenStudy(directory, document, scope, ledger, warn, keeper=keeper)
            tree = Tree(
                session,
                study,
                trials,
                identity,
                check,
                progress,
                warn,
                resume,
                custom=custom,
                strategy=made,
            )
            if resume:  # every record of every level held up: their stages serve, wherever
                session.reuse([directory], resumed=True)
            _run_trials(tree, session, activity)
            # read back as load_study reads it: with the problem study.json records, whose titles
            # a resume with other titles leaves as they were
            return results_of(directory, document, recorded_scope(document), ledger.ledger)
        finally:
            ledger.close()


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
