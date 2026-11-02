"""One candidate evaluated without a session, in this thread: the semantics of stages and
evaluators, tested on a problem document, without a problem file or a materialized geometry.

:func:`evaluate_candidate` binds the implementations to the problem, prepares the candidate and
runs its tasks through the scheduler and the inline backend, as a session with one worker-less
job does; the caller gives the geometry, or none.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from k4opt.evaluation import CandidateKey, EvaluationContractError, EvaluatorProtocol, Scalar
from k4opt.execution._backends import InlineBackend
from k4opt.execution._binding import _bind_problem, _context, _preflight, _Prepared
from k4opt.execution._drive import _drive, _finish_plan, _Run
from k4opt.execution._schedule import Scheduler
from k4opt.execution._stages import check_run_inputs
from k4opt.execution._tasks import GeometryWork, Workspace
from k4opt.problem.candidate import configure, validate_candidate
from k4opt.problem.models import ProblemSpec
from k4opt.problem.plan import GEOMETRY, Kind, TaskId, stage_task
from k4opt.problem.stages import STAGE_TYPES
from k4opt.record.results import CandidateRecord


def evaluate_candidate(
    spec: ProblemSpec,
    candidate: Mapping[str, Scalar],
    implementations: Sequence[type[EvaluatorProtocol]],
    *,
    inputs: Mapping[str, Path] | None = None,
    workdir: Path | None = None,
    directory: Path | None = None,
) -> CandidateRecord:
    """Evaluate one complete ``candidate`` of ``spec``, whose file is in ``directory``, with an
    instance of each implementation.

    The candidate maps each parameter's name to a scalar; ``implementations`` (an
    evaluator class, say) implement the problem's ``evaluators``, in any order. Before any stage
    runs, raises :class:`~k4opt.problem.CandidateError` for a candidate outside the declared
    domains and :class:`EvaluationContractError` for implementations that cannot be created or
    do not match the problem (missing, undeclared, given twice, another version), settings an
    evaluator's ``check()`` rejects, a stage that cannot run, or an objective or constraint
    metric its evaluator does not declare. ``inputs["geometry"]`` is the candidate's geometry,
    already prepared; any other entry of ``inputs`` supplies the file of the stage of a sample
    it is named after (``gamma/simulation``). The stages the evaluators read and ``inputs`` does
    not supply then run once each, in ``workdir``. An evaluator whose stage failed, or whose
    stage output lacks the requested events, is ``blocked`` and not called, as is a combined
    evaluator an item of whose roles did not succeed (failed, invalid or blocked); the others
    are still called. An evaluator that raises :class:`~k4opt.evaluation.EvaluationFailed` or
    :class:`~k4opt.evaluation.EvaluationInvalid` gets that result; any other exception
    propagates, and so does an :class:`EvaluationContractError` for a result that breaks the
    contract.
    """
    started = time.perf_counter()
    values, key = validate_candidate(spec, candidate)
    given = dict(inputs or {})
    geometry = given.pop(GEOMETRY.name, None)
    if geometry is not None:  # an input from outside the workspace: named absolutely
        geometry = Path(os.path.abspath(geometry))
    supplied = {}
    for label, path in given.items():
        sample, _, stage = label.partition("/")
        if sample not in spec.samples or stage not in STAGE_TYPES:
            raise EvaluationContractError(
                f"input {label!r}: expected SAMPLE/STAGE, a sample of the problem and one of "
                f"{', '.join(STAGE_TYPES)}"
            )
        supplied[stage_task(stage, sample)] = Path(os.path.abspath(path))
    binding = _bind_problem(spec, implementations, supplied, directory=directory)
    planned = {task.id for task in binding.plan.of(Kind.STAGE)}
    for task in supplied:  # a file no task reads would be dropped without a word
        if task not in planned:
            raise EvaluationContractError(
                f"input {task.label!r}: no evaluator of the problem reads that stage of "
                f"{task.sample}"
            )
    prepared = _preflight(binding, configure(spec, values))
    checked = time.perf_counter() - started
    return _execute_plan(
        spec,
        prepared,
        key,
        supplied=supplied,
        geometry=None if geometry is None else GeometryWork(given=geometry),
        workdir=workdir,
        checked_seconds=checked,
    )


def _execute_plan(
    spec: ProblemSpec,
    prepared: _Prepared,
    key: CandidateKey,
    *,
    supplied: Mapping[TaskId, Path],
    geometry: GeometryWork | None,
    workdir: Path | None,
    checked_seconds: float,
    timeout: float | None = None,
) -> CandidateRecord:
    """Run the tasks of a prepared candidate, one after another in plan order, in this thread;
    evaluator checks and defaults are not repeated.

    ``supplied`` holds the files the caller supplies by stage task, the ones the plan was made with;
    ``geometry`` says how the candidate's geometry is made, and ``None`` gives none (to evaluators
    that need none). Each task writes below ``workdir``, in the directory the plan gives it.
    ``checked_seconds``, the time the candidate's validation and preflight took, counts towards
    ``elapsed_seconds``, which thus covers k4opt's checks too (not what the caller does between).
    ``timeout`` limits each stage that runs a program
    (:attr:`~k4opt.execution._stages.StageContext.timeout`).
    """
    started = time.perf_counter() - checked_seconds
    binding = prepared.binding
    stages = binding.plan.of(Kind.STAGE)
    check_run_inputs([t.id.name for t in stages if t.given is None], geometry is not None, workdir)
    run = _Run(Workspace("candidate", workdir), prepared, key, supplied, geometry, started)
    backend = InlineBackend(
        _context(spec, binding, timeout), lambda workspace, task: prepared.instances[task]
    )
    pulled = iter([run])
    driven = _drive(Scheduler(backend.capacity), backend, {}, lambda: next(pulled, None))
    with contextlib.closing(driven):
        for ended in driven:
            if ended.error is not None:
                raise ended.error
            return _finish_plan(ended)
    raise AssertionError("the candidate never ended")  # pragma: no cover
