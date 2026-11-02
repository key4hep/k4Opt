"""Exceptions raised by the record component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class RecordError(K4OptError):
    """A run record (``summary.json``) is malformed, was made for another problem, or
    disagrees with its own measurements.

    The message says which part of the record is at fault.
    """
