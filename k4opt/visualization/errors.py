"""Exceptions raised by the visualization component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class VisualizationError(K4OptError):
    """A figure cannot show what it is asked to: an axis the study does not have, or values of one
    axis recorded in different units."""
