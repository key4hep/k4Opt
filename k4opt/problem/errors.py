"""Exceptions raised by the problem component."""

from __future__ import annotations

from k4opt.errors import ConfigError, K4OptError


class ProblemConfigError(ConfigError):
    """A problem file or document is unreadable, malformed or violates the schema.

    The message starts with the location of the offending value inside the document
    (``parameters[0].bounds``), preceded by the file path when the document was loaded
    from a file.
    """


class RunError(K4OptError):
    """``k4opt evaluate`` cannot start: its run directory exists, or a stage file given with
    ``--input`` is unusable."""


class CandidateError(K4OptError):
    """A candidate does not fit the problem it is evaluated for.

    The message names the offending parameter reference(s).
    """
