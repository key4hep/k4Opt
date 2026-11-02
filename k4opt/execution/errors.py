"""Exceptions raised by the execution component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class RunError(K4OptError):
    """A candidate cannot be evaluated: its run directory exists, a stage file supplied for it
    is unusable, a worker process cannot be set up, or its geometry cannot be made."""


class ProvenanceError(RunError):
    """What a session evaluates with changed, so a record could not say what was evaluated: a
    file the stages read, what the geometry's external references read, the environment, k4opt's
    code or an evaluator's, or a file a task made, changed since it was identified; or a
    candidate's geometry is not the one the session resolved. Nothing is recorded for the
    candidate, and an optimizer stops: no later record could be trusted either."""
