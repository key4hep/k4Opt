"""The resolved options of a k4run job, as Gaudi's ``JobOptionsSvc`` writes them (its
``DUMPFILE``): what the job ran with, the command line included, and the check that it read the
geometry, the input and the output k4opt gave it.

Each line of the dump is ``<component>.<property> = <value>;``, the value written as Python
writes it (``[ '/path/CLD_o2_v09.xml' ]``, ``'events.edm4hep.root'``, ``-1``). Pure: the text is
read by the caller.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Sequence
from pathlib import Path

_LINE = re.compile(r"(?P<key>[^\s=]+)\s*=\s*(?P<value>.*?);?\s*")


def properties(text: str) -> dict[str, object]:
    """Every property of the dump ``text`` by ``<component>.<property>``: its value as Python
    reads it (``ast.literal_eval``), or its text where it is no literal (a tool reference)."""
    found: dict[str, object] = {}
    for line in text.splitlines():
        match = _LINE.fullmatch(line.strip())
        if match is None:
            continue
        raw = match["value"]
        try:
            found[match["key"]] = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            found[match["key"]] = raw
    return found


def _same(value: object, path: Path) -> bool:
    return isinstance(value, str) and os.path.realpath(value) == os.path.realpath(path)


def _list_of(value: object, paths: Sequence[Path]) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == len(paths)
        and all(_same(item, path) for item, path in zip(value, paths, strict=True))
    )


def _shown(value: object) -> str:
    return "nothing" if value is None else repr(value)


def verify(
    text: str,
    *,
    compact: Path,
    input_file: Path,
    output: Path,
    geometry: str | None = None,
    input_argument: str | None = None,
) -> str | None:
    """Why the job whose resolved options are ``text`` did not read the candidate's geometry
    ``compact`` (``GeoSvc.detectors``), the events ``input_file`` (``IOSvc.Input``) or write
    ``output`` (``IOSvc.Output``); ``None`` if it did. ``geometry`` and ``input_argument`` name the
    job's own arguments k4opt gave them to, for the message."""
    found = properties(text)
    problems = []
    detectors = found.get("GeoSvc.detectors")
    if not _list_of(detectors, [compact]):
        given = f"--{geometry}" if geometry else "--GeoSvc.detectors"
        problems.append(
            f"GeoSvc.detectors is {_shown(detectors)}, not the candidate's geometry {compact} "
            f"that k4opt passed as {given}"
            + (
                ""
                if geometry
                else "; if the job takes the geometry as an argument of its own, "
                "name it with geometry:"
            )
        )
    inputs = found.get("IOSvc.Input")
    if not _list_of(inputs, [input_file]):
        given = f"--{input_argument}" if input_argument else "--IOSvc.Input"
        problems.append(
            f"IOSvc.Input is {_shown(inputs)}, not the events {input_file} that k4opt passed as "
            f"{given}"
        )
    written = found.get("IOSvc.Output")
    if not _same(written, output):
        problems.append(f"IOSvc.Output is {_shown(written)}, not {output}")
    if not problems:
        return None
    return "the job did not run on what k4opt gave it: " + "; ".join(problems)
