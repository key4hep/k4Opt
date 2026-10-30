"""Terminal output, shared by the components' command lines.

Color, shortening to the terminal's width and redrawn lines are for an interactive terminal
only: piped or written to a file, output stays plain text. ``NO_COLOR`` turns color off.
"""

from __future__ import annotations

import os
import shutil
from typing import TextIO


class Style:
    """Tiny ANSI helper; every method returns the text unchanged when disabled."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled and text else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def gray(self, text: str) -> str:
        return self._wrap("90", text)


def is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def interactive(stream: TextIO) -> bool:
    """Whether ``stream`` is a terminal that can redraw a line."""
    return is_tty(stream) and os.environ.get("TERM") != "dumb"


def width(stream: TextIO) -> int | None:
    """Terminal width for an interactive stream; ``None`` (no shortening) when piped."""
    return shutil.get_terminal_size((100, 24)).columns if is_tty(stream) else None


def color(stream: TextIO) -> bool:
    return interactive(stream) and "NO_COLOR" not in os.environ
