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
    """A candidate cannot be evaluated: its run directory exists, a stage file supplied for it
    is unusable, or a file the problem names changed after the problem was loaded."""


class RecordError(K4OptError):
    """A run record (``summary.json``) is malformed, was made for another problem, or
    disagrees with its own measurements.

    The message says which part of the record is at fault.
    """


class CandidateError(K4OptError):
    """A candidate does not fit the problem it is evaluated for.

    The message names the offending parameter reference(s).
    """
