"""Exceptions raised by the k4run component."""

from __future__ import annotations

from k4opt.program import ProgramError


class K4runError(ProgramError):
    """The options of a reconstruction cannot be turned into a ``k4run`` command, or set what
    k4opt manages.

    Raised before anything is run. A ``k4run`` that fails is reported by
    :attr:`k4opt.k4run.ReconstructionRun.error`, not by an exception.
    """
