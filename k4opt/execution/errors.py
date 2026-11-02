"""Exceptions raised by the execution component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class RunError(K4OptError):
    """A candidate cannot be evaluated: its run directory exists, a stage file supplied for it
    is unusable, or a file the stages read changed during an evaluation session."""
