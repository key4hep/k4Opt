"""Command-line interface of the problem component: ``k4opt problem``.

    k4opt problem explain PROBLEM --trials N

A thin wrapper over :func:`k4opt.problem.outline.render_outline`; :func:`register` plugs it into the
top-level ``k4opt`` command (:mod:`k4opt.cli`). It reads the problem alone: nothing runs and nothing
is written. Exits 0 once it has explained the problem, 1 for a problem that does not load.
"""

from __future__ import annotations

import argparse
from typing import TextIO

from k4opt.problem.config import load_problem
from k4opt.problem.outline import render_outline
from k4opt.terminal import color


def _budget(text: str) -> int:
    """``--trials``: a positive integer."""
    trials = int(text)
    if trials < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return trials


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``problem`` command to the top-level parser."""
    group = commands.add_parser(
        "problem",
        help="say what a problem means, before anything of it runs",
        description="Read a problem and say what it asks for. Nothing runs or is written.",
    )
    actions = group.add_subparsers(dest="action", required=True, metavar="ACTION")
    explain = actions.add_parser(
        "explain",
        help="what a study of a problem does, and at most what it runs",
        description=(
            "Print what a new study of the problem does, as k4opt optimize does when it starts: "
            "its levels as loops, each with its budget and what picks its best trial, what "
            "judges the top level, and at most how many times each stage and evaluation runs."
        ),
    )
    explain.add_argument("problem", help="path to a k4opt problem YAML file")
    explain.add_argument(
        "--trials",
        type=_budget,
        required=True,
        metavar="N",
        help="how many trials the top level would run, as for k4opt optimize",
    )
    explain.set_defaults(func=cmd_explain)


def cmd_explain(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    render_outline(load_problem(args.problem), args.trials, out, color=color(out))
    return 0
