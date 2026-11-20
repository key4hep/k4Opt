"""Exceptions shared by every k4opt component.

Every error raised on purpose by k4opt derives from :class:`K4OptError`, so
callers (and the CLI) can catch one type to report expected failures while
letting genuine bugs propagate. Components define their own subclasses next to
their code (see :mod:`k4opt.geometry.errors`).
"""

from __future__ import annotations


class K4OptError(Exception):
    """Base class for all k4opt errors."""


class ConfigError(K4OptError):
    """A YAML configuration file is malformed or references missing paths."""


class CodeLoadError(K4OptError):
    """A class of the user's code named as ``FILE.py:Class`` or ``package.module:Class`` cannot be
    loaded, or is not known to run the source it is identified by (:mod:`k4opt.usercode`)."""
