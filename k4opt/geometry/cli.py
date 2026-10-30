"""Command-line interface of the geometry component (``k4opt geometry ...``).

    k4opt geometry inspect GEOMETRY [--root DIR] [--filter TEXT] [--format text|json]
    k4opt geometry apply GEOMETRY [--root DIR] [--set [FILE:]TARGET=VALUE ...] --output DIR
        [--dry-run] [--format text|json]

A thin wrapper over the library; :func:`register` plugs it into the top-level
``k4opt`` command (:mod:`k4opt.cli`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import TextIO

from k4opt.geometry.errors import InvalidTargetError
from k4opt.geometry.overrides import Override
from k4opt.geometry.render import render_changes, render_parameters
from k4opt.geometry.source import GeometrySource

INSPECT_SCHEMA = "k4opt.geometry.inspect/1"
APPLY_SCHEMA = "k4opt.geometry.apply/1"


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``geometry`` command and its actions to the top-level parser."""
    geometry = commands.add_parser(
        "geometry", help="inspect or modify a detector geometry (DD4hep XML)"
    )
    actions = geometry.add_subparsers(dest="action", required=True)

    inspect = actions.add_parser("inspect", help="list the addressable parameters of a geometry")
    _add_geometry(inspect)
    inspect.add_argument(
        "--filter",
        metavar="TEXT",
        default=None,
        help="case-insensitive substring over target, file and value",
    )
    inspect.add_argument("--format", choices=("text", "json"), default="text")
    inspect.set_defaults(func=cmd_inspect)

    apply = actions.add_parser("apply", help="materialize a modified copy of a geometry")
    _add_geometry(apply)
    apply.add_argument(
        "--set",
        action="append",
        default=[],
        dest="sets",
        metavar="[FILE:]TARGET=VALUE",
        help=(
            "set a target to a literal value, e.g. 'constants.ECal_cell_size=10*mm' (repeatable); "
            "FILE: selects a target defined in several files"
        ),
    )
    apply.add_argument(
        "--output", required=True, help="new directory to create for the modified geometry"
    )
    apply.add_argument(
        "--dry-run", action="store_true", help="resolve and report changes, write nothing"
    )
    apply.add_argument("--format", choices=("text", "json"), default="text")
    apply.set_defaults(func=cmd_apply)


def _add_geometry(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "geometry",
        help="compact file, relative to $K4GEO (or --root), absolute, or $VAR/...",
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        default=None,
        help="tree the geometry is relative to, e.g. a k4geo checkout",
    )


# ---------------------------------------------------------------------------


def _is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _width(stream: TextIO) -> int | None:
    """Terminal width for an interactive stream; ``None`` (no shortening) when piped."""
    return shutil.get_terminal_size((100, 24)).columns if _is_tty(stream) else None


def _color(stream: TextIO) -> bool:
    return _is_tty(stream) and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def _override(assignment: str) -> Override:
    """``[FILE:]TARGET=VALUE``; the target ends at the first ``=`` outside ``[...]``."""
    depth = 0
    for index, char in enumerate(assignment):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(depth - 1, 0)
        elif char == "=" and depth == 0:
            file, _, target = assignment[:index].rpartition(":")
            try:
                return Override(target=target, value=assignment[index + 1 :], file=file or None)
            except InvalidTargetError as exc:
                raise InvalidTargetError(f"--set {assignment!r}: {exc}") from exc
    raise InvalidTargetError(f"--set {assignment!r}: expected [FILE:]TARGET=VALUE")


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
        width=_width(out),
        color=_color(out),
    )
    return 0


def cmd_apply(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    overrides = [_override(assignment) for assignment in args.sets]
    source = GeometrySource(args.geometry, root=args.root)
    catalog = source.discover()
    resolved = source.resolve(overrides, catalog)

    if args.dry_run:
        from k4opt.geometry.materialize import check_output_location

        output = check_output_location(source.root, args.output)
        if args.format == "json":
            payload = {
                "schema": APPLY_SCHEMA,
                "dry_run": True,
                "source_root": str(source.root),
                "output_root": str(output),
                "compact_path": str(output / source.compact),
                "changes": [c.to_dict() for c in resolved],
            }
            json.dump(payload, out, indent=2)
            out.write("\n")
        else:
            render_changes(resolved, out, color=_color(out))
            print(file=out)
            print(f"geometry root: {source.root}", file=out)
            print(f"would write:   {output}", file=out)
            print(f"dry run: {len(resolved)} change(s) resolved; nothing written", file=out)
        return 0

    result = source.materialize(overrides, args.output, catalog)
    if args.format == "json":
        payload = {"schema": APPLY_SCHEMA, "dry_run": False, **result.to_dict()}
        json.dump(payload, out, indent=2)
        out.write("\n")
    else:
        render_changes(result.changes, out, color=_color(out))
        print(file=out)
        print(f"geometry root:  {source.root}", file=out)
        print(f"wrote snapshot: {result.output_root}", file=out)
        print(f"compact:        {result.compact_path}", file=out)
        print(f"manifest:       {result.manifest_path}", file=out)
        print(f"fingerprint:    {result.fingerprint}", file=out)
    return 0
