"""Command-line interface of the software component: ``k4opt software``.

    k4opt software show

A thin wrapper over :class:`k4opt.software.Software`; :func:`register` plugs it into the top-level
``k4opt`` command (:mod:`k4opt.cli`). Nothing runs and nothing is written.
"""

from __future__ import annotations

import argparse
from typing import TextIO

from k4opt.software.identity import Software
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
            "environment variables are compared, and the id a study made here would record: "
            "two shells with one id run on the same software."
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
    return 0
