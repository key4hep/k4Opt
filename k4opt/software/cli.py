"""Command-line interface of the software component: ``k4opt software``.

    k4opt software show

A thin wrapper over :class:`k4opt.software.Software` and :func:`k4opt.software.source.origins`;
:func:`register` plugs it into the top-level ``k4opt`` command (:mod:`k4opt.cli`). Nothing runs
and nothing is written.
"""

from __future__ import annotations

import argparse
import os
from typing import TextIO

from k4opt.software.identity import Software
from k4opt.software.source import Origin, origins
from k4opt.software.stack import Release
from k4opt.terminal import printable


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``software`` command to the top-level parser."""
    group = commands.add_parser(
        "software",
        help="say what candidates would run on in this shell",
        description="Say what k4opt finds in this shell. Nothing runs or is written.",
    )
    actions = group.add_subparsers(dest="action", required=True, metavar="ACTION")
    show = actions.add_parser(
        "show",
        help="the build of the stack, the system and the identity of this shell",
        description=(
            "Print the build of the Key4hep stack this shell runs on, the system, how many "
            "environment variables are compared, the id a study made here would record (two "
            "shells with one id run on the same software), and where each local build in front "
            "of the stack comes from: its checkout and its changes against the revision the "
            "nightly built."
        ),
    )
    show.set_defaults(func=cmd_show)


def cmd_show(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    software = Software.now()
    build = software.build
    rows = {
        "build": build.label,
        "platform": build.platform,
        "system": software.system.label,
        "variables": f"{len(software.variables)} compared",
        "id": software.id,
    }
    for name, value in rows.items():
        if value is not None:
            print(printable(f"{name + ':':<11}{value}"), file=out)
    found = origins(software.local, Release.of(os.environ.get("KEY4HEP_STACK")))
    for index, origin in enumerate(found):
        head = "local:" if index == 0 else ""
        print(printable(f"{head:<11}{_origin(origin)}"), file=out)
    return 0


def _origin(origin: Origin) -> str:
    """Where a local build or directory of local code comes from, on one line."""
    if origin.kind == "environment":
        return f"{origin.prefix}: {len(origin.distributions)} installed distributions"
    parts = [origin.prefix]
    if origin.package:
        parts.append(f"stands in for {origin.package} {origin.revision or ''}".rstrip())
    if origin.kind == "build":
        parts.append(f"{origin.link} link to its source")
    git = origin.git
    if git is not None:
        base = (git.base or "")[:7] or "no public commit"
        parts.append(f"{git.checkout} at {base}: {len(git.changed)} files changed")
    elif origin.kind == "directory":
        parts.append(f"{len(origin.files)} files in no checkout")
    if origin.stale:
        parts.append("source changed after the build")
    return "; ".join(parts)
