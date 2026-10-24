"""Exceptions raised by the geometry component.

All derive from :class:`GeometryError`, itself a :class:`~k4opt.errors.K4OptError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from k4opt.errors import K4OptError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from k4opt.geometry.discovery import Parameter


class GeometryError(K4OptError):
    """Base class for all errors raised by the geometry component."""


class SourceError(GeometryError):
    """The geometry root or compact entry point is invalid."""


class IncludeError(GeometryError):
    """A local XML include cannot be resolved safely (missing, escaping, unreadable)."""


class TargetError(GeometryError):
    """Base class for problems with override targets.

    When several overrides fail at once, a plain :class:`TargetError` is raised
    whose ``errors`` attribute lists the individual failures.
    """

    def __init__(self, message: str, errors: Sequence[TargetError] = ()) -> None:
        super().__init__(message)
        self.errors: tuple[TargetError, ...] = tuple(errors)


class InvalidTargetError(TargetError):
    """A target reference does not follow the reference grammar."""


class UnknownTargetError(TargetError):
    """A syntactically valid target does not exist in the discovered catalog."""


class AmbiguousTargetError(TargetError):
    """A target matches more than one attribute and needs a ``file`` discriminator."""

    def __init__(self, message: str, candidates: Sequence[Parameter] = ()) -> None:
        super().__init__(message)
        self.candidates: tuple[Parameter, ...] = tuple(candidates)


class DuplicateTargetError(TargetError):
    """Two overrides select the same underlying XML attribute."""


class MaterializationError(GeometryError):
    """Creating the modified geometry snapshot failed; nothing was written."""
