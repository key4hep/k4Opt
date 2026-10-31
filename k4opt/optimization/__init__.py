"""Optimization component: search a problem's parameters for its best candidates.

:func:`optimize` runs a study: an optimizer proposes candidates, each is evaluated as
``k4opt evaluate`` evaluates one, into a trial directory of the study, and the optimizer learns
each result from the trial's ``summary.json``. The optimizer is Optuna's TPE sampler; Optuna is
imported only when a study runs or is read, so the rest of k4opt works without it. A study keeps
everything it is made with in ``study.json`` and can be resumed with exactly that.
:func:`load_study` reads a study back, checked and without changing it, for inspection and
export. See ``docs/optimization.md``.

Typical use::

    from k4opt.evaluation import load_evaluator
    from k4opt.optimization import load_study, optimize

    report = optimize(
        "examples/problem/cld-ecal-granularity.yaml",
        [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],
        "runs/cld-ecal-study",
        trials=30,
        seed=42,
    )
    report.pareto  # the trials on the Pareto front of the feasible valid ones

    study = load_study("runs/cld-ecal-study")  # later, anywhere the directory is
    study.trial(12).assessment.objectives  # each with its value, unit and uncertainty
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator, Sequence

from k4opt.evaluation import EvaluatorProtocol
from k4opt.optimization._results import StudyResults, TrialResult
from k4opt.optimization._study import STUDY_NAME, StudyReport, TrialReport
from k4opt.optimization.errors import OptimizationError

__all__ = [
    "STUDY_NAME",
    "OptimizationError",
    "StudyReport",
    "StudyResults",
    "TrialReport",
    "TrialResult",
    "load_study",
    "optimize",
]


@contextlib.contextmanager
def _needs_optuna(what: str) -> Iterator[None]:
    """Optuna missing, while the adapter is imported, becomes an error that says how to install
    it; any other missing module is not dressed up."""
    try:
        yield
    except ModuleNotFoundError as exc:
        if exc.name != "optuna" and not (exc.name or "").startswith("optuna."):
            raise
        raise OptimizationError(
            f"{what} needs Optuna, which is not installed: pip install 'optuna>=5.0,<6' "
            "(or k4opt[optimize])"
        ) from exc


def optimize(
    problem: str | os.PathLike[str],
    evaluators: Sequence[EvaluatorProtocol],
    output: str | os.PathLike[str],
    *,
    trials: int,
    seed: int | None = None,
    resume: bool = False,
    storage: str | None = None,
    timeout: float | None = None,
    warn: Callable[[str], None] | None = None,
    progress: Callable[[TrialReport], None] | None = None,
) -> StudyReport:
    """Run ``trials`` new trials of a study of the problem file ``problem`` in ``output``.

    ``evaluators`` implement the problem's ``evaluators``. A new study needs an ``output`` that
    does not exist or is empty; ``seed`` (in [0, 2**32)) seeds the sampler, one is drawn and
    recorded without it. ``resume`` continues the study in ``output``: it must have been made
    with the same problem, files, evaluators, geometry, Key4hep stack, k4opt code and Optuna
    version, and is refused otherwise; trials an interrupted session left running are finished
    first and do not count among ``trials``. ``storage`` is an Optuna database URL in place of
    the journal file in ``output``, given again on every resume. ``timeout`` is how many
    seconds a trial's simulation may take before ddsim is killed and the trial fails; it is not
    part of the study, so a resume may set another one.

    ``warn`` is called with each warning, ``progress`` with each trial as it ends, the trials a
    resume finishes first (``recovered``). Invalid and failed candidates are FAIL trials and the
    study goes on. Raises :class:`OptimizationError` for a study that cannot start or resume,
    and when a trial stops it (a broken evaluation contract, a record that does not hold up); a
    :class:`~k4opt.errors.K4OptError` for an invalid problem or evaluator, before anything is
    created. Ctrl-C and SIGTERM end the running trial (FAIL, or told its record), remove the
    lock and raise :class:`KeyboardInterrupt`.
    """
    with _needs_optuna("k4opt optimize"):
        from k4opt.optimization import _optuna
    return _optuna.optimize(
        problem,
        evaluators,
        output,
        trials=trials,
        seed=seed,
        resume=resume,
        storage=storage,
        timeout=timeout,
        warn=warn or (lambda message: None),
        progress=progress or (lambda trial: None),
    )


def load_study(directory: str | os.PathLike[str], *, storage: str | None = None) -> StudyResults:
    """The study in ``directory``, as its records hold it, for inspection and export.

    Nothing is evaluated, created, locked, recovered or written: the optimizer's storage is
    opened so that it refuses every write, and the problem, geometry and evaluators the study
    was made with need not exist any more. ``storage`` is the database URL of a study made with
    one (``study.json`` keeps it without its password); a study in its journal file needs none.
    Every trial the optimizer was told a record of is read back and checked: the record has the
    bytes it was told, belongs to this trial of this study, and its assessment recomputes from
    its measurements. A study in progress is read as it is at that moment.

    Raises :class:`OptimizationError` for a directory that is no study, a storage that is
    missing, not set up or not the study's, and any record that does not hold up; a record is
    never skipped. Needs Optuna, which holds the trials.
    """
    with _needs_optuna("reading a study"):
        from k4opt.optimization import _optuna
    return _optuna.load(directory, storage)
