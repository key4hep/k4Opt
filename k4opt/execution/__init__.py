"""Execution component: candidates of a problem, evaluated end to end.

An :class:`EvaluationSession` loads a problem file once and binds the evaluator implementations
to it, given as classes of which every candidate gets new instances. It evaluates each candidate
(:meth:`~EvaluationSession.evaluate`, or several at once with
:meth:`~EvaluationSession.evaluate_each` and :class:`RunRequest`) into a new run directory: the
candidate's geometry is materialized, each pipeline stage of each sample runs once (ddsim) and
each evaluation measures its sample, or the candidate. The tasks of candidates run in this
process or in worker processes, as the scheduler gives them slots; what each candidate's
evaluation found is published as its ``summary.json`` (:mod:`k4opt.record`), judged against the
objectives and constraints (:class:`RunOutcome`). It knows no optimizer: whatever proposes
candidates, a person or an optimization library, calls it the same way. :func:`run_candidate`
is ``k4opt evaluate``. See ``docs/execution.md``.

Typical use::

    from k4opt.execution import EvaluationSession

    session = EvaluationSession.open("problem.yaml", [MyEvaluator])  # the class
    outcome = session.evaluate({"ecal_cell_size": 7.0}, "runs/7mm")
    outcome.assessment.feasible  # True, False or None (not assessed)
"""

from k4opt.execution._drive import Activity
from k4opt.execution._publish import RunOutcome
from k4opt.execution._session import Evaluations, EvaluationSession, RunRequest, run_candidate
from k4opt.execution.errors import RunError

__all__ = [
    "Activity",
    "EvaluationSession",
    "Evaluations",
    "RunError",
    "RunOutcome",
    "RunRequest",
    "run_candidate",
]
