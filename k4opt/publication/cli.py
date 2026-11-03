"""Command-line interface of the publication component: ``k4opt publish``.

    k4opt publish DIR --output PUBLICATION [--storage URL]

A thin wrapper over :func:`k4opt.optimization.load_study` and :func:`k4opt.publication.publish`;
:func:`register` plugs it into the top-level ``k4opt`` command (:mod:`k4opt.cli`). It exits 0
once the study's snapshot is in the publication's catalog, also when it was already, and 1 when
the study cannot be read or published; the study directory is only read.
"""

from __future__ import annotations

import argparse
from typing import TextIO

from k4opt.optimization import load_study
from k4opt.publication import publish
from k4opt.terminal import printable


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``publish`` command to the top-level parser."""
    parser = commands.add_parser(
        "publish",
        help="publish a study as a portable snapshot, for the dashboard and for others",
        description=(
            "Read a study as k4opt study reads it, checked and unchanged, and publish it into a "
            "publication directory: its JSON form, without any location, as a snapshot named by "
            "its content, and its entry in the directory's catalog.json. Serve the directory "
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
    parser.set_defaults(func=cmd_publish)


def cmd_publish(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    results = load_study(args.study, storage=args.storage)
    for warning in results.warnings:
        print(f"k4opt: warning: {printable(warning)}", file=err)

    def warn(message: str) -> None:
        print(f"k4opt: warning: {printable(message)}", file=err)

    entry = publish(results, args.output, warn=warn)
    counts = ", ".join(f"{count} {outcome}" for outcome, count in entry.outcomes.items() if count)
    print(
        printable(
            f"{entry.name}: {entry.trials} trials ({counts or 'none yet'}), published "
            f"{entry.published} as {args.output}/{entry.snapshot.path}"
        ),
        file=out,
    )
    return 0
