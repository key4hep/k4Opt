"""Record component: what the evaluation of a candidate found, as its ``summary.json`` holds it.

A candidate's evaluation ends with a :class:`CandidateRecord`: the result of each task of its
plan (:class:`TaskResult`), the candidate's status and message derived from the required ones.
It is written as the run record ``summary.json`` (:data:`SUMMARY_NAME`, schema ``k4opt.run/1``),
with an :class:`Assessment` of the candidate against the problem's objectives and constraints
(:class:`ObjectiveValue`, :class:`Limit`). :meth:`Assessment.from_summary` reads a record back
and checks it against its problem before anything trusts it, without the evaluators or any file
it names: a record holds up only where a live evaluation would have accepted what it says.
:func:`peak_memory` reads back the memory its tasks took. See ``docs/record.md``.

Typical use::

    from k4opt.record import Assessment

    summary = json.loads(Path("runs/7mm/summary.json").read_text())
    assessment = Assessment.from_summary(load_problem("problem.yaml"), summary)
    assessment.feasible  # True, False or None (not assessed)
"""

from k4opt.record.assess import Assessment, Limit, ObjectiveValue
from k4opt.record.errors import RecordError
from k4opt.record.results import CandidateRecord, PeakMemory, TaskResult
from k4opt.record.write import SUMMARY_NAME, peak_memory

__all__ = [
    "SUMMARY_NAME",
    "Assessment",
    "CandidateRecord",
    "Limit",
    "ObjectiveValue",
    "PeakMemory",
    "RecordError",
    "TaskResult",
    "peak_memory",
]
