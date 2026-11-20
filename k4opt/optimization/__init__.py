"""Optimization component: search a problem's parameters for its best candidates.

:func:`optimize` runs a study: a *strategy* proposes candidates, each is evaluated as
``k4opt evaluate`` evaluates one, into a trial directory of the study, and the study keeps each
trial in its ledger as the trial's ``summary.json`` holds it, then tells the strategy. The
strategy is Optuna's TPE sampler by default (``strategy="optuna"``, imported only when a study
runs with it), or a grid of given values (``strategy="grid"``), which needs nothing besides
k4opt. A study keeps everything it is made with in ``study.json``, its trials in its own ledger
whatever the strategy, and can be resumed with exactly that. :func:`load_study` reads a study
back, checked and without changing it, for inspection and export, and needs no strategy;
:func:`trade_offs` asks it which feasible trials give up at most so much of some objectives
against a reference trial for at least so much of others. See ``docs/optimization.md``.

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
    trade_offs(study, 12, [Rule("gamma/calo.photon/simulation/res", "worsen", 0.05)]).matches
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from k4opt.evaluation import EvaluatorProtocol, Scalar
from k4opt.execution import Activity
from k4opt.optimization import _run
from k4opt.optimization._read import load
from k4opt.optimization._results import OUTCOMES, StudyResults, TrialResult
from k4opt.optimization._strategy import STRATEGIES
from k4opt.optimization._study import STUDY_NAME
from k4opt.optimization._tradeoffs import RULE_KINDS, Rule, TradeOffs, trade_offs
from k4opt.optimization.errors import OptimizationError
from k4opt.program import Observation

__all__ = [
    "OUTCOMES",
    "RULE_KINDS",
    "STRATEGIES",
    "STUDY_NAME",
    "OptimizationError",
    "Rule",
    "StudyResults",
    "TradeOffs",
    "TrialResult",
    "load_study",
    "optimize",
    "trade_offs",
]


def optimize(
    problem: str | os.PathLike[str],
    implementations: Sequence[type[EvaluatorProtocol]],
    output: str | os.PathLike[str],
    *,
    trials: int,
    strategy: str | None = None,
    grid: Mapping[str, Sequence[Scalar]] | None = None,
    seed: int | None = None,
    resume: bool = False,
    workers: int | Literal["auto", "inline"] = "inline",
    reuse: Sequence[str | os.PathLike[str]] = (),
    fixed: Mapping[str, Scalar] | None = None,
    observe: Observation = "files",
    warn: Callable[[str], None] | None = None,
    progress: Callable[[TrialResult], None] | None = None,
    activity: Callable[[Activity], None] | None = None,
) -> StudyResults:
    """Run ``trials`` new trials of a study of the problem file ``problem`` in ``output``.

    ``implementations`` (evaluator classes, as :func:`~k4opt.evaluation.load_evaluator` returns
    them) implement the problem's ``evaluators``; every trial gets new instances. A new study
    needs an ``output`` that does not exist or is empty. ``strategy`` proposes its candidates
    (one of :data:`STRATEGIES`): ``"optuna"``, the default, Optuna's TPE sampler, seeded with
    ``seed`` (in [0, 2**32); one is drawn and recorded without it); or ``"grid"``, every
    combination of the values ``grid`` gives each searched parameter (by name; a categorical
    parameter without them takes each choice, an integer one each integer), one after another,
    the last parameter varying fastest, at most ``trials`` of them; a study whose grid is
    exhausted ends early. ``resume`` continues the study in ``output`` with the strategy it was
    made with (give none of ``strategy``, ``grid`` and ``seed``): it must have been made with the
    same problem, files, evaluators, geometry, Key4hep stack and k4opt code (and Optuna version,
    for its strategy), and is refused otherwise; trials an interrupted session left running are
    finished first and do not count among ``trials``, which may then be 0 (only those are
    finished). For a problem with inner levels, each trial stands for a study of the level below
    (``docs/optimization.md``), in ``output/inner``, with the same strategy. ``workers`` says
    where the tasks run, as for :meth:`~k4opt.execution.EvaluationSession.open`: ``"inline"``,
    one after another in this process (an evaluator that crashes the process ends the study,
    which ``resume`` continues), or that many worker processes, of one trial and of several
    (``k4opt optimize`` runs one by default); the strategy is asked for a candidate only when a
    slot would otherwise be idle, and told each trial as it ends. It is not part of the study. A
    trial a stage or an evaluation of which runs longer than the problem's ``timeout`` for it
    fails, and the study goes on. A stage of a trial is not run again when a stage with its
    signature made its output already (the same definition, inputs, geometry, files, programs,
    environment and k4opt code: see ``docs/evaluation.md``): in this session, in an earlier trial
    of the study (on a resume), or in the runs and studies ``reuse`` names; its record says where
    the output comes from. Neither is part of the study, nor is ``observe``: how closely each
    stage's program is watched, as for :meth:`~k4opt.execution.EvaluationSession.open`; another
    study reuses an output only if its stage was watched at the ``files`` level.

    ``fixed`` gives some parameters one value for every trial (``k4opt optimize --fix``): the
    optimizer proposes only the others, every record holds the complete candidate, and a resume
    needs the same values. A study of the reconstruction with the geometry fixed simulates once,
    and its trials reuse that simulation.

    ``warn`` is called with each warning, ``progress`` with each trial as it ends, the trials a
    resume finishes first (``recovered``), and ``activity`` with what runs whenever that changes
    (:class:`~k4opt.execution.Activity`: trials under way, busy slots); each trial is given as
    :func:`load_study` reads it back, and so is the study once the trials ended. Invalid and failed
    candidates, those an evaluator rejects and proposals that are no candidate end their trial,
    and the study goes on. Raises
    :class:`OptimizationError` for a study that cannot start or resume, and when a trial stops it (a
    broken evaluation contract, a record that does not hold up); a :class:`~k4opt.errors.K4OptError`
    for an invalid problem or evaluator, before anything is created. Ctrl-C and SIGTERM end the
    running trials (interrupted, or kept with their records), remove the lock and raise
    :class:`KeyboardInterrupt`; a trial that stops the study ends the others the same way.
    """
    return _run.optimize(
        problem,
        implementations,
        output,
        trials=trials,
        strategy=strategy,
        grid=grid,
        seed=seed,
        resume=resume,
        workers=workers,
        reuse=reuse,
        fixed=fixed,
        observe=observe,
        warn=warn or (lambda message: None),
        progress=progress or (lambda trial: None),
        activity=activity,
    )


def load_study(directory: str | os.PathLike[str]) -> StudyResults:
    """The study in ``directory``, as its ledger and records hold it, for inspection and export.

    Nothing is evaluated, created, locked, recovered or written, no strategy is needed (nor
    Optuna, for a study it ran), and the problem, geometry and evaluators the study was made
    with need not exist any more. Every trial the ledger holds is read back and checked: a trial
    with a record has the bytes its ledger names, belongs to this trial of this study, and its
    assessment recomputes from its measurements. A study in progress is read as it is at that
    moment.

    Raises :class:`OptimizationError` for a directory that is no study, a ledger that is missing
    or does not hold up, and any record that does not hold up; a record is never skipped.
    """
    return load(directory)
