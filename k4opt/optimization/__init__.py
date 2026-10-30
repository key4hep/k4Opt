"""Optimization component: search a problem's parameters for its best candidates.

:func:`optimize` runs a study: an optimizer proposes candidates, each is evaluated as
``k4opt evaluate`` evaluates one, into a trial directory of the study, and the optimizer learns
each result from the trial's ``summary.json``. The optimizer is Optuna's TPE sampler; Optuna is
imported only when a study runs, so the rest of k4opt works without it. A study keeps
everything it is made with in ``study.json`` and can be resumed with exactly that. See
``docs/optimization.md``.

Typical use::

    from k4opt.evaluation import load_evaluator
    from k4opt.optimization import optimize

    report = optimize(
        "examples/problem/cld-ecal-granularity.yaml",
        [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],
        "runs/cld-ecal-study",
        trials=30,
        seed=42,
    )
    report.pareto  # the trials on the Pareto front of the feasible valid ones
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from k4opt.evaluation import EvaluatorProtocol
from k4opt.optimization._study import STUDY_NAME, StudyReport, TrialReport
from k4opt.optimization.errors import OptimizationError

__all__ = ["STUDY_NAME", "OptimizationError", "StudyReport", "TrialReport", "optimize"]


def optimize(
    problem: str | os.PathLike[str],
    evaluators: Sequence[EvaluatorProtocol],
    output: str | os.PathLike[str],
    *,
    trials: int,
    seed: int | None = None,
    resume: bool = False,
    storage: str | None = None,
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
    the journal file in ``output``, given again on every resume.

    ``warn`` is called with each warning, ``progress`` with each trial as it ends. Invalid and
    failed candidates are FAIL trials and the study goes on. Raises
    :class:`OptimizationError` for a study that cannot start or resume, and when a trial stops
    it (a broken evaluation contract, a record that does not hold up); a
    :class:`~k4opt.errors.K4OptError` for an invalid problem or evaluator, before anything is
    created. Ctrl-C and SIGTERM end the running trial (FAIL, or told its record), remove the
    lock and raise :class:`KeyboardInterrupt`.
    """
    try:
        from k4opt.optimization import _optuna
    except ModuleNotFoundError as exc:
        if exc.name != "optuna" and not (exc.name or "").startswith("optuna."):
            raise
        raise OptimizationError(
            "k4opt optimize needs Optuna, which is not installed: pip install 'optuna>=5.0,<6' "
            "(or k4opt[optimize])"
        ) from exc
    return _optuna.optimize(
        problem,
        evaluators,
        output,
        trials=trials,
        seed=seed,
        resume=resume,
        storage=storage,
        warn=warn or (lambda message: None),
        progress=progress or (lambda trial: None),
    )
