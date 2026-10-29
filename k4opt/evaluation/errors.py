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
