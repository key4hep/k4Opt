"""Exceptions raised by the evaluation component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class EvaluationContractError(K4OptError):
    """A request, measurement or result violates the evaluation contract.

    Raised when such an object is constructed with an invalid value (a NaN measurement,
    a duplicate metric, a blank name, a wrong type). The message names the offending field.
    """


class EvaluatorLoadError(K4OptError):
    """An evaluator named as ``FILE.py:Class`` or ``module:Class`` cannot be loaded.

    The file or module is missing or fails to import, the class is not there, it cannot be
    created without arguments, or the object lacks ``name``, ``version`` or ``evaluate``.
    """


class EvaluationFailed(K4OptError):  # noqa: N818 - names the status ``failed``
    """Raised by an evaluator (or a reading helper) when the execution failed.

    ``k4opt`` turns it into a ``failed`` result whose message is the text of the exception, so
    an evaluator can stop at the first problem instead of building the result by hand.
    """


class EvaluationInvalid(K4OptError):  # noqa: N818 - names the status ``invalid``
    """Raised by an evaluator when the evaluation ran but its result is not usable.

    ``k4opt`` turns it into an ``invalid`` result whose message is the text of the exception.
    """
