"""Command-line interface of the geometry component (``k4opt geometry ...``).

    k4opt geometry inspect GEOMETRY [--root DIR] [--filter TEXT] [--format text|json]

A thin wrapper over the library; :func:`register` plugs it into the top-level
``k4opt`` command (:mod:`k4opt.cli`).
"""

from __future__ import annotations

import argparse
import json
from typing import TextIO

from k4opt.geometry.render import render_parameters
from k4opt.geometry.source import GeometrySource
from k4opt.terminal import color, width

INSPECT_SCHEMA = "k4opt.geometry.inspect/1"


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``geometry`` command and its actions to the top-level parser."""
    geometry = commands.add_parser("geometry", help="inspect a detector geometry (DD4hep XML)")
    actions = geometry.add_subparsers(dest="action", required=True)

    inspect = actions.add_parser("inspect", help="list the addressable parameters of a geometry")
    inspect.add_argument(
        "geometry",
        help="compact file, relative to $K4GEO (or --root), absolute, or $VAR/...",
    )
    inspect.add_argument(
        "--root",
        metavar="DIR",
        default=None,
        help="tree the geometry is relative to, e.g. a k4geo checkout",
    )
    inspect.add_argument(
        "--filter",
        metavar="TEXT",
        default=None,
        help="case-insensitive substring over target, file and value",
    )
    inspect.add_argument("--format", choices=("text", "json"), default="text")
    inspect.set_defaults(func=cmd_inspect)


# ---------------------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    catalog = GeometrySource(args.geometry, root=args.root).discover()
    parameters = catalog.filter(args.filter) if args.filter else tuple(catalog)

    if args.format == "json":
        payload = {
            "schema": INSPECT_SCHEMA,
            "geometry": {"root": str(catalog.root), "compact": catalog.compact},
            "files": list(catalog.files),
            "resources": list(catalog.resources),
            "external_includes": [
                {"file": e.file, "line": e.line, "ref": e.ref, "reason": e.reason}
                for e in catalog.externals
            ],
            "parameters": [p.to_dict() for p in parameters],
        }
        json.dump(payload, out, indent=2)
        out.write("\n")
        return 0

    render_parameters(
        parameters,
        catalog,
        out,
        filter_text=args.filter,
        width=width(out),
        color=color(out),
    )
    return 0
