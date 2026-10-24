"""Exceptions raised by the evaluation component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class EvaluationContractError(K4OptError):
    """A request, measurement or result violates the evaluation contract.

    Raised when such an object is constructed with an invalid value (a NaN measurement,
    a duplicate metric, a blank name, a wrong type). The message names the offending field.
    """
