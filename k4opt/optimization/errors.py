"""Exceptions raised by the optimization component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class OptimizationError(K4OptError):
    """A study cannot start, resume or go on.

    Its directory is in use or is not a study, a resumed study no longer matches what it was
    made with, a trial broke the evaluation contract, or its record failed validation. The
    message says what to do; the study directory is left as it was.
    """
