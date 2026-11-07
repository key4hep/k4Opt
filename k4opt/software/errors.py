"""Exceptions raised by the software component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class SoftwareError(K4OptError):
    """A record's account of the software is malformed, or is not what its ``id`` says.

    The message says which part of it is at fault.
    """
