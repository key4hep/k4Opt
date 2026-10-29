"""The evaluator protocol: what k4opt needs of an evaluator, whether or not it subclasses
:class:`~k4opt.evaluation.base.Evaluator`."""

from __future__ import annotations

from typing import Protocol

from k4opt.evaluation.base import Values
from k4opt.evaluation.models import EvaluationRequest, EvaluationResult


class EvaluatorProtocol(Protocol):
    """Anything that turns an :class:`EvaluationRequest` into an :class:`EvaluationResult`.

    There is one calling convention for every scientific procedure: a simulation, a
    reconstruction, an analysis or a cost model are all evaluators. No base class is
    needed; a class with these members, or an object built by a function, is one.

    ``name`` and ``version`` identify the evaluator's measurement contract, the
    counterparts of ``name`` and ``version`` of an entry of ``evaluation.evaluators`` in the
    problem file. They are plain labels: the protocol does not look evaluators up by them.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult | Values:
        """Evaluate one candidate.

        Report a scientific or execution problem through ``status`` (``invalid`` or
        ``failed``) and a ``message``, not by returning made-up measurements; raising
        :class:`EvaluationInvalid` or :class:`EvaluationFailed` does the same. Bugs and
        misuse (a missing required input, say) may raise anything else.
        """
        ...
