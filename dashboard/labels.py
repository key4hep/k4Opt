"""How the dashboard words what it shows: names, numbers, units, times and durations; no Streamlit
here.

A value is shown as the study records it, only worded for a reader: four significant digits, a
large number of no unit with its SI prefix (``16.83 M``), a unit only where it means one (``1`` is
a number of no unit), a time in UTC as a date reads (``3 Nov 2026, 15:29 UTC``). What a value
means stays the study's to say.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import PurePosixPath

from k4opt.optimization import StudyResults
from k4opt.visualization import TRIAL, names

_MARKDOWN = re.compile(r"([\\`*_\[\]<>~$|])")  # what formats, links or embeds within a line


def plain(text: object) -> str:
    """``text`` as Streamlit's Markdown shows it, as it is: no character of it formats or links
    anything (a study's name, a message, a parameter of the URL)."""
    return _MARKDOWN.sub(r"\\\1", str(text))


def unit(name: str | None) -> str | None:
    """``name`` as a unit is shown: none for a number of no unit (``1``) or no unit at all."""
    return None if name in (None, "", "1") else name


_PREFIXES = ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"))
_EXPONENT = re.compile(r"e([+-])0*(\d)")


def _digits(value: float, digits: int) -> str:
    """``value`` to ``digits`` significant digits, an exponent without its padding (``1.5e7``)."""
    return _EXPONENT.sub(lambda m: f"e{'-' if m[1] == '-' else ''}{m[2]}", f"{value:.{digits}g}")


def number(value: object, of: str | None = None, uncertainty: float | None = None) -> str:
    """``value`` in the unit ``of``, with its ``uncertainty`` if it has one: four significant
    digits (an integer below a million exactly), and a value of no unit from 10 000 on by its SI
    prefix, as a count reads (``16.83 M``)."""
    shown = unit(of)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        text = str(value)
    elif shown is None and abs(value) >= 1e4:
        factor, prefix = next((f, p) for f, p in _PREFIXES if abs(value) >= f)
        text = _digits(value / factor, 4)
        if uncertainty is not None:
            text += f" ± {_digits(uncertainty / factor, 2)}"
        return f"{text} {prefix}"
    elif isinstance(value, int) and abs(value) < 1e6:
        text = str(value)
    else:
        text = _digits(value, 4)
    if uncertainty is not None:
        text += f" ± {_digits(uncertainty, 2)}"
    return f"{text} {shown}" if shown else text


def duration(seconds: float | None) -> str:
    """``seconds`` as a duration reads: ``0.2 ms``, ``4.1 s``, ``29 min 16 s``, ``2 h 05 min``."""
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{_digits(seconds * 1000, 2)} ms"
    if seconds < 60:
        return f"{_digits(seconds, 2)} s"
    whole = round(seconds)
    if whole < 3600:
        return f"{whole // 60} min {whole % 60} s"
    return f"{whole // 3600} h {whole % 3600 // 60:02d} min"


def _parsed(iso: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return moment.astimezone(UTC) if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def when(iso: str) -> str:
    """The time ``iso`` names, in UTC as a date reads (``3 Nov 2026, 15:29 UTC``); as it is if it
    reads as no time."""
    moment = _parsed(iso)
    return iso if moment is None else f"{moment.day} {moment:%b %Y, %H:%M} UTC"


def ago(iso: str, now: datetime | None = None) -> str:
    """How long ago ``iso`` was (``5 minutes ago``), or :func:`when` once it is a week or more."""
    moment = _parsed(iso)
    if moment is None:
        return iso
    seconds = ((now or datetime.now(UTC)) - moment).total_seconds()
    if seconds < 0 or seconds >= 7 * 86400:
        return when(iso)
    for size, word in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            return f"{count(int(seconds // size), word)} ago"
    return "just now"


def count(n: int, word: str, plural: str | None = None) -> str:
    """``n`` of ``word``, its ``plural`` (``word`` and an s) unless one: ``1 parameter``,
    ``2 objectives``."""
    return f"{n} {word if n == 1 else plural or word + 's'}"


def axis(key: str, results: StudyResults) -> str:
    """An axis or a color of a figure as a choice shows it, by what the figures call it
    (:func:`~k4opt.problem.names`): ``background_rejection · objective``, ``ECAL cell size
    · parameter``, ``trial number``."""
    if key == TRIAL:
        return "trial number"
    kind, _, name = key.partition(":")
    if not name:
        return key
    called = names(results.problem)
    return f"{called.objective(name) if kind == 'objective' else called.parameter(name)} · {kind}"


def size(count: int) -> str:
    """A size in decimal units, as the budget of a publication is said (``4.2 kB``, ``48 MB``)."""
    value, unit = float(count), "B"
    for unit in ("B", "kB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            break
        value /= 1000
    return f"{count} B" if unit == "B" else f"{_digits(value, 3)} {unit}"


def text(content: bytes) -> str | None:
    """``content`` as text, ``None`` if it is binary data."""
    if b"\0" in content[:8000]:
        return None
    try:
        return content.decode()
    except UnicodeDecodeError:
        return None


ACCESS = {"public": "Public", "on request": "On request: ask the author", "none": "Not available"}
"""Who can get a requirement, as the Software tab says it."""
STATE = {
    "available": "Available",
    "gone": "Gone",
    "altered": "Altered",
    "not-checked": "Not checked",
}
"""What a requirement's check found, as the Software tab says it."""
LINK = {
    "verified": "Verified",
    "likely": "Likely",
    "adjacent": "Adjacent checkout",
    "none": "None",
}
"""How sure a local build's link to its checkout is, as the Software tab says it."""


def detector(geometry: str) -> str:
    """The detector a compact file describes, by the file's name (``CLD_o2_v09``), as k4Bench
    names it."""
    return PurePosixPath(geometry).stem
