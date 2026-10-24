"""The ``k4opt`` command: dispatches to the command groups of each component.

Each component contributes one command group by exposing ``register(commands)``
in its own ``cli`` module (``k4opt geometry ...`` today). Exit codes: 0 success,
1 expected k4opt error (message on stderr), 2 usage error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from k4opt import __version__
from k4opt.errors import K4OptError
from k4opt.geometry import cli as geometry_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k4opt",
        description="Detector optimization framework for Key4hep.",
    )
    parser.add_argument("--version", action="version", version=f"k4opt {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    geometry_cli.register(commands)
    return parser


def main(
    argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args, out, err))
    except K4OptError as exc:
        print(f"k4opt: error: {exc}", file=err)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
