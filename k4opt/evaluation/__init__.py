"""Evaluation component: the contract between k4opt and whatever measures a candidate.

An evaluator (a subclass of :class:`Evaluator`, or anything with ``name``, ``version`` and
``evaluate``) receives an :class:`EvaluationRequest` (candidate values, settings, the
collections it reads, prepared input files) and returns an :class:`EvaluationResult` (a
status, zero or more :class:`Measurement` objects, named artifacts), or just
``{metric: value}``. The contract does
not depend on a problem file, a geometry or the Key4hep stack, and does not prescribe stages.
Running evaluators, looking them up by the name in ``problem.yaml`` and persisting results are
not part of it;
:func:`load_evaluator` only turns ``FILE.py:Class`` into an evaluator class, of which k4opt
creates an instance for each candidate (an :data:`EvaluatorFactory`).
An :class:`EvaluationRecord` pairs one request and result with what the framework observed.
See ``docs/evaluation.md``.

Typical use::

    from k4opt.evaluation import Evaluator

    class Constant(Evaluator):
        name = "demo.constant"
        version = "1"

        def evaluate(self, request):
            # no `stage`: nothing runs first, so the metric names its stage itself
            return {("analysis", "answer"): 42}
"""

from k4opt.evaluation.base import (
    Collection,
    Evaluator,
    Quantity,
    check_settings,
    collection_names,
    declared_collections,
    metric_units,
    result_from,
    with_defaults,
)
from k4opt.evaluation.edm4hep import (
    Event,
    available_collections,
    collection_types,
    count_events,
    read_collection,
    read_events,
)
from k4opt.evaluation.errors import (
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluatorLoadError,
)
from k4opt.evaluation.loader import evaluator_source, load_evaluator
from k4opt.evaluation.models import (
    CandidateKey,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
    Scalar,
    Status,
)
from k4opt.evaluation.protocol import EvaluatorFactory, EvaluatorProtocol

__all__ = [
    "CandidateKey",
    "Collection",
    "EvaluationContractError",
    "EvaluationFailed",
    "EvaluationInvalid",
    "EvaluationRecord",
    "EvaluationRequest",
    "EvaluationResult",
    "Evaluator",
    "EvaluatorFactory",
    "EvaluatorLoadError",
    "EvaluatorProtocol",
    "Event",
    "Measurement",
    "Quantity",
    "Scalar",
    "Status",
    "available_collections",
    "check_settings",
    "collection_names",
    "collection_types",
    "count_events",
    "declared_collections",
    "evaluator_source",
    "load_evaluator",
    "metric_units",
    "read_collection",
    "read_events",
    "result_from",
    "with_defaults",
]
