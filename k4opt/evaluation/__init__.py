"""Evaluation component: the contract between k4opt and whatever measures a candidate.

An evaluator (a subclass of :class:`Evaluator`, or anything with ``name``, ``version`` and
``evaluate``) receives an :class:`EvaluationRequest` (candidate values, settings, prepared
input files) and returns an :class:`EvaluationResult` (a status, zero or more
:class:`Measurement` objects, named artifacts), or just ``{metric: value}``. The contract does
not depend on a problem file, a geometry or the Key4hep stack, and does not prescribe stages.
Running evaluators, looking them up by the name in ``problem.yaml`` and persisting results are
not part of it;
:func:`load_evaluator` only turns ``FILE.py:Class`` into an evaluator object.
An :class:`EvaluationRecord` pairs one request and result with what the framework observed.
See ``docs/evaluation.md``.

Typical use::

    from k4opt.evaluation import Evaluator

    class Constant(Evaluator):
        name = "demo.constant"
        version = "1"
        requires = "analysis"  # the stage of the metrics below

        def evaluate(self, request):
            return {"answer": 42}
"""

from k4opt.evaluation.base import Evaluator, metric_units, result_from
from k4opt.evaluation.edm4hep import count_events, read_collection
from k4opt.evaluation.errors import (
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluatorLoadError,
)
from k4opt.evaluation.loader import load_evaluator
from k4opt.evaluation.models import (
    CandidateKey,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
    Status,
)
from k4opt.evaluation.protocol import EvaluatorProtocol

__all__ = [
    "CandidateKey",
    "EvaluationContractError",
    "EvaluationFailed",
    "EvaluationInvalid",
    "EvaluationRecord",
    "EvaluationRequest",
    "EvaluationResult",
    "Evaluator",
    "EvaluatorLoadError",
    "EvaluatorProtocol",
    "Measurement",
    "Status",
    "count_events",
    "load_evaluator",
    "metric_units",
    "read_collection",
    "result_from",
]
