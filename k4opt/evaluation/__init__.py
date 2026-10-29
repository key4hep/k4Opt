"""Evaluation component: the contract between k4opt and whatever measures a candidate.

An :class:`Evaluator` receives an :class:`EvaluationRequest` (candidate values, settings,
prepared input files) and returns an :class:`EvaluationResult` (a status, zero or more
:class:`Measurement` objects, named artifacts). The contract does not depend on a problem
file, a geometry or the Key4hep stack, and does not prescribe stages. Running evaluators,
looking them up by the name in ``problem.yaml`` and persisting results are not part of it;
:func:`load_evaluator` only turns ``FILE.py:Class`` into an evaluator object.
An :class:`EvaluationRecord` pairs one request and result with what the framework observed.
See ``docs/evaluation.md``.

Typical use::

    from k4opt.evaluation import (
    CandidateKey,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
)

    class Constant:
        name = "demo.constant"
        version = "1"

        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            return EvaluationResult.valid([Measurement("analysis", "answer", 42, unit="1")])
"""

from k4opt.evaluation.errors import EvaluationContractError, EvaluatorLoadError
from k4opt.evaluation.loader import load_evaluator
from k4opt.evaluation.models import (
    CandidateKey,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
    Status,
)
from k4opt.evaluation.protocol import Evaluator

__all__ = [
    "CandidateKey",
    "EvaluationContractError",
    "EvaluationRecord",
    "EvaluationRequest",
    "EvaluationResult",
    "Evaluator",
    "EvaluatorLoadError",
    "Measurement",
    "Status",
    "load_evaluator",
]
