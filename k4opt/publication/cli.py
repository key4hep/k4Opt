"""Command-line interface of the publication component: ``k4opt publish``.

    k4opt publish DIR --output PUBLICATION [--storage URL] [--yes] [--include-inputs NAME ...]
                      [--include NAME ...] [--exclude NAME ...] [--include-large]

A thin wrapper over :func:`k4opt.optimization.load_study`, :func:`k4opt.publication.offer` and
:func:`k4opt.publication.publish`; :func:`register` plugs it into the top-level ``k4opt`` command
(:mod:`k4opt.cli`). It lists what would be published, asks about the code and then, separately,
about the inputs, and publishes exactly what was approved. It exits 0 once the study's snapshot is
in the publication's catalog, also when it was already, and 1 when the study cannot be read or
published, a secret was found, or there is no one to ask; the study directory is only read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from k4opt.optimization import load_study
from k4opt.publication import BUDGET, Offer, Piece, PublicationError, offer, publish
from k4opt.publication.exposure import said_of_path, said_of_withheld
from k4opt.terminal import Style, color, is_tty, printable

_LABEL = 12  # the column of what a line says, after its label
_OWNER = 18  # the column of a file, after what it belongs to
_MINUS = "\N{MINUS SIGN}"


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``publish`` command to the top-level parser."""
    parser = commands.add_parser(
        "publish",
        help="publish a study as a portable snapshot, for the dashboard and for others",
        description=(
            "Read a study as k4opt study reads it, checked and unchanged, and publish it into a "
            "publication directory: its JSON form, without any location, as a snapshot named by "
            "its content, what a rerun needs in public form, and its entry in the directory's "
            "catalog.json. Local code that ran and input files the stages read are listed and "
            "published only if you approve them, in two separate questions. Serve the directory "
            "(WebEOS, any web server) or copy it anywhere; publishing again adds the study's "
            "newer state. Everything published is readable by whoever can read the directory. "
            "Needs Optuna, which holds the trials."
        ),
    )
    parser.add_argument("study", metavar="DIR", help="the study directory (--output of optimize)")
    parser.add_argument(
        "--output",
        required=True,
        metavar="PUBLICATION",
        help="the publication directory, made if it does not exist; not inside the study",
    )
    parser.add_argument(
        "--storage",
        metavar="URL",
        help="the database URL of a study made with --storage, as for k4opt study",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="approve the code listed without asking; inputs still need --include-inputs",
    )
    parser.add_argument(
        "--include-inputs",
        action="append",
        default=[],
        metavar="NAME",
        help="publish this input file, also one held back, without asking; may be repeated",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="NAME",
        help="offer code that is held back (untracked and not used, binary); may be repeated",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="leave out this file or package: the dashboard names it, on request; may be repeated",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help=f"publish also what passes the budget of {BUDGET / 1e6:g} MB per group",
    )
    parser.set_defaults(func=cmd_publish)


def cmd_publish(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    results = load_study(args.study, storage=args.storage)
    for warning in results.warnings:
        print(f"k4opt: warning: {printable(warning)}", file=err)

    def warn(message: str) -> None:
        print(f"k4opt: warning: {printable(message)}", file=err)

    style = Style(color(out))
    found = offer(
        results,
        include=args.include,
        include_inputs=args.include_inputs,
        exclude=args.exclude,
        include_large=args.include_large,
    )
    _list(found, args.output, out, style)
    if any(finding.snapshot for finding in found.findings):
        print(
            style.red("nothing was published: a study is published as it is, so this one is not"),
            file=out,
        )
        return 1
    if found.findings:
        print(style.red("nothing was published: remove or --exclude those files"), file=out)
        return 1
    code = _code(found, args.yes, out, style)
    inputs = _inputs(found, args.include_inputs, args.output, out, style)
    release = found.release(code=code, inputs=inputs)
    entry = publish(results, args.output, warn=warn, release=release)
    owners = {piece.owner or piece.name for piece in release.pieces if piece.group == "code"}
    sent = sum(piece.group == "inputs" for piece in release.pieces)
    large = sum(piece in found.large for piece in release.pieces)
    what = f"code of {len(owners)}, {sent} input{'' if sent == 1 else 's'}, {large} large"
    _line(
        out,
        style,
        "published",
        printable(f"{entry.name} (snapshot {entry.snapshot.sha256[:8]}): {what}"),
    )
    counts = ", ".join(f"{count} {outcome}" for outcome, count in entry.outcomes.items() if count)
    _line(
        out,
        style,
        "",
        printable(
            f"{entry.trials} trials ({counts or 'none yet'}), published {entry.published} as "
            f"{args.output}/{entry.snapshot.path}"
        ),
    )
    return 0


def _code(found: Offer, yes: bool, out: TextIO, style: Style) -> bool:
    """Whether the code listed is approved; refuses when there is no one to ask."""
    if not _all(found, "code"):
        return False
    if yes:
        return True
    if not is_tty(sys.stdin):
        raise PublicationError(
            "there is code to approve and no terminal to ask on: say --yes to publish the code "
            "listed, or --exclude what should stay"
        )
    return _ask("Publish the code above?", out, style)


def _inputs(
    found: Offer, named: Sequence[str], where: str, out: TextIO, style: Style
) -> bool | Sequence[str]:
    """The inputs approved: those ``named``, or all of them after the question; none when there
    is no one to ask, since a script never publishes data it does not name."""
    if named:
        return named
    if not _all(found, "inputs") or not is_tty(sys.stdin):
        return False
    question = (
        "Also publish the contents of these input files? Anyone who can read\n"
        f"{printable(where)} can read them."
    )
    return _ask(question, out, style)


def _all(found: Offer, group: str) -> list[Piece]:
    large = [piece for piece in found.large if found.include_large]
    pieces = found.code if group == "code" else found.inputs
    return [*pieces, *(piece for piece in large if piece.group == group)]


def _ask(question: str, out: TextIO, style: Style) -> bool:
    print(f"{style.bold(question)} [y/N] ", end="", file=out, flush=True)
    return sys.stdin.readline().strip().lower() in ("y", "yes")


def _line(out: TextIO, style: Style, label: str, text: str) -> None:
    painted = style.green if label == "published" else style.dim
    print(f"{painted(label.ljust(_LABEL))}{text}", file=out)


def _size(count: int) -> str:
    """A size as the budget is said: in decimal units."""
    value, unit = float(count), "B"
    for unit in ("B", "kB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            break
        value /= 1000
    return f"{count} B" if unit == "B" else f"{value:.3g} {unit}"


def _change(piece: Piece, style: Style) -> str:
    """How much a piece changes: its lines, added and removed; its size, for a whole file."""
    if piece.added is None or piece.removed is None:
        return _size(piece.size)
    return f"{style.green(f'+{piece.added}')} {style.red(f'{_MINUS}{piece.removed}')}"


def _used(piece: Piece) -> str:
    return f"read by {', '.join(piece.stages)}" if piece.stages else ""


def _list(found: Offer, where: str, out: TextIO, style: Style) -> None:
    """What would be published, unit by unit, and what would not, with why (see
    ``docs/publication.md``)."""
    label = "code"
    code = _all(found, "code")  # what the question will be about, over the budget or not
    for part in _by_unit(code):
        owner = printable(part[0].owner)
        size = sum(piece.size for piece in part)
        over = (
            style.yellow(f"   {_size(size)}, over {_size(BUDGET)}")
            if part[0] in found.large
            else ""
        )
        _line(out, style, label, f"{style.bold(owner)}   {_size(size)}{over}")
        label = ""
        for piece in part:
            _piece(piece, out, style)
    if not code:
        _line(out, style, label, "nothing local to publish")
        label = ""
    for h in (h for h in found.held if h.group == "code"):
        files = ", ".join(printable(p.name) for p in h.pieces)
        _line(
            out,
            style,
            label,
            style.gray(f"not published  {printable(h.label)}: {files}   ({h.why})"),
        )
        label = ""
    _input_lines(found, out, style)
    for piece in found.large:
        if not found.include_large:
            _line(
                out,
                style,
                "warning:",
                style.yellow(
                    f"{printable(piece.label)}, {_size(piece.size)}, over "
                    f"{_size(BUDGET)}: not published (--include-large to publish it)"
                ),
            )
    for part in _by_unit(found.excluded):
        said = printable(part[0].owner or part[0].name)
        _line(out, style, "", style.gray(f"not published  {said}   (excluded, with all of it)"))
    if found.readiness is not None:
        for gap in found.readiness.public_gaps:
            _line(out, style, "warning:", style.yellow(printable(gap)))
    for where, path in found.paths:
        _line(out, style, "warning:", style.yellow(printable(said_of_path(where, path))))
    for withheld in found.withheld:
        _line(out, style, "withheld", style.yellow(printable(said_of_withheld(withheld))))
    if found.findings:
        for finding in found.findings:
            _line(
                out,
                style,
                "blocked:",
                style.red(f"{printable(finding.label)}:{finding.line}: {finding.what}"),
            )
    else:
        _line(out, style, "scan", style.green("nothing found"))
    _line(out, style, "to", style.cyan(printable(where)))
    print(file=out)


def _by_unit(pieces: Sequence[Piece]) -> list[list[Piece]]:
    found: dict[str, list[Piece]] = {}
    for piece in pieces:
        found.setdefault(piece.unit, []).append(piece)
    return list(found.values())


def _piece(piece: Piece, out: TextIO, style: Style) -> None:
    """One file of a unit: a patch with each file it changes, any other with its change."""
    if piece.is_patch:
        line = f"patch against {printable(piece.base)}   {len(piece.files)} files"
        _line(out, style, "", f"  {line}  {_change(piece, style)}")
        for name, added, removed in piece.files:
            counted = (
                "binary"
                if added is None or removed is None
                else f"{style.green(f'+{added}')} {style.red(f'{_MINUS}{removed}')}"
            )
            _line(out, style, "", f"    {style.bold(printable(name))}  {counted}")
        return
    parts = [f"  {style.bold(printable(piece.name))}", _change(piece, style)]
    if piece.stages:
        parts.append(printable(_used(piece)))
    _line(out, style, "", "  ".join(parts))


def _input_lines(found: Offer, out: TextIO, style: Style) -> None:
    label = "inputs"
    inputs = _all(found, "inputs")  # what the question will be about, over the budget or not
    used = sum(piece.size for piece in inputs)
    for index, piece in enumerate(inputs):
        total = (
            f"   {style.bold(_size(used))} of {_size(BUDGET)}" if index == len(inputs) - 1 else ""
        )
        if piece in found.large:
            total += "   " + style.yellow(f"over {_size(BUDGET)}")
        shared = sum(other.name == piece.name for other in inputs) > 1
        where = f" ({piece.requirements[0]}: {piece.place})" if shared else ""
        _line(
            out,
            style,
            label,
            f"{style.bold(printable(piece.name + where)).ljust(_OWNER)}{_size(piece.size)}   "
            f"{printable(_used(piece))}{total}",
        )
        label = ""
    held = [h for h in found.held if h.group == "inputs"]
    for why in sorted({h.why for h in held}):
        names = ", ".join(printable(h.name) for h in held if h.why == why)
        _line(out, style, label, style.gray(f"not published  {names}   ({why})"))
        label = ""
