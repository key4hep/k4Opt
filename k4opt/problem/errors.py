"""Exceptions raised by the problem component."""

from __future__ import annotations

from k4opt.errors import ConfigError


class ProblemConfigError(ConfigError):
    """A problem file or document is unreadable, malformed or violates the schema.

    The message starts with the location of the offending value inside the document
    (``parameters[0].bounds``), preceded by the file path when the document was loaded
    from a file.
    """
