"""Command-line interface of the optimization component: ``k4opt optimize`` and ``k4opt study``.

    k4opt optimize PROBLEM --evaluator FILE.py:Class [--evaluator ...] --trials N --output DIR
        [--seed S] [--resume] [--storage URL] [--timeout SECONDS]
    k4opt study show DIR [--storage URL]
    k4opt study trial DIR NUMBER [--storage URL]
    k4opt study export DIR --format {csv,json} [--output FILE] [--storage URL]

Thin wrappers over :func:`k4opt.optimization.optimize` and :func:`k4opt.optimization.load_study`;
:func:`register` plugs them into the top-level ``k4opt`` command (:mod:`k4opt.cli`). ``optimize``
exits 0 when every trial asked for has ended, whatever its candidate's status, and 1 when the
study cannot start or resume, a trial stops it, or it is interrupted. ``study`` exits 0 once it
has printed or exported the study, and 1 when the study cannot be read; it changes nothing in the
study, and an export writes only the data to its output, any warning to stderr.
"""

from __future__ import annotations

import argparse
from typing import TextIO

from k4opt.evaluation import load_evaluator
from k4opt.optimization import StudyResults, load_study, optimize
from k4opt.optimization._export import FORMATS, export
from k4opt.optimization._render import Progress, render_study
from k4opt.optimization._show import printable, render_results, render_trial
from k4opt.terminal import color, interactive


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``optimize`` and ``study`` commands to the top-level parser."""
    parser = commands.add_parser(
        "optimize",
        help="search the parameters of a problem for its best candidates",
        description=(
            "Run a study of a problem: the optimizer (Optuna's TPE sampler) proposes candidates, "
            "and each is evaluated as k4opt evaluate evaluates one, into a trial directory of "
            "the study. Needs Optuna and the Key4hep environment."
        ),
    )
    parser.add_argument("problem", help="path to a k4opt problem YAML file")
    parser.add_argument(
        "--evaluator",
        action="append",
        required=True,
        metavar="FILE.py:Class",
        help="an evaluator of the problem, as for k4opt evaluate (repeatable)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        required=True,
        metavar="N",
        help="how many new trials to run; trials a resume finishes do not count",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="the study directory: new or empty for a new study, the study's own with --resume",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="S",
        help="seed of the sampler, in [0, 2**32); drawn and recorded without it",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the study in DIR, made with exactly the same problem, files and code",
    )
    parser.add_argument(
        "--storage",
        metavar="URL",
        help="an Optuna database URL in place of DIR/study.journal (needs SQLAlchemy)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help=(
            "kill ddsim if a trial's simulation runs longer than this; the trial is FAIL and the "
            "study goes on (no limit without it)"
        ),
    )
    parser.set_defaults(func=cmd_optimize)
    _register_study(commands)


def _register_study(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = commands.add_parser(
        "study",
        help="inspect or export the results of a study, without changing it",
        description=(
            "Read a study that k4opt optimize made, as its records hold it: nothing is evaluated "
            "or written, and the problem, geometry and evaluators need not exist any more. Each "
            "record is checked against the study and against what the optimizer was told. Needs "
            "Optuna."
        ),
    )
    actions = group.add_subparsers(dest="action", required=True, metavar="ACTION")

    def action(name: str, help: str, description: str) -> argparse.ArgumentParser:
        parser = actions.add_parser(name, help=help, description=description)
        parser.add_argument(
            "study", metavar="DIR", help="the study directory (--output of optimize)"
        )
        parser.add_argument(
            "--storage",
            metavar="URL",
            help=(
                "the database URL of a study made with --storage; study.json names it without its "
                "password. A SQLite or PostgreSQL database is opened read only"
            ),
        )
        return parser

    show = action(
        "show",
        "what a study optimizes, how its trials ended and its Pareto front",
        "Print the study's parameters, objectives and limits, how many trials ended how, and "
        "the trials on the Pareto front of the feasible valid ones, with their values.",
    )
    show.set_defaults(func=cmd_show)
    trial = action(
        "trial",
        "one trial of a study, with every value as recorded",
        "Print one trial: its status, candidate, objectives with units and uncertainties, "
        "limits with their margins, the times its record holds, its record and its message.",
    )
    trial.add_argument("number", type=int, metavar="NUMBER", help="the trial's number")
    trial.set_defaults(func=cmd_trial)
    exported = action(
        "export",
        "the results of a study as CSV or JSON, for plotting and analysis",
        "Write one CSV row per trial, or the whole study as JSON, every value as recorded "
        "(docs/optimization.md describes both).",
    )
    exported.add_argument("--format", required=True, choices=FORMATS, help="csv or json")
    exported.add_argument(
        "--output",
        metavar="FILE",
        help="a new file outside the study directory; standard output without it",
    )
    exported.set_defaults(func=cmd_export)


def cmd_optimize(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    evaluators = [load_evaluator(reference) for reference in args.evaluator]
    try:
        with Progress(args.trials, out, err, bar=interactive(out), color=color(out)) as progress:
            report = optimize(
                args.problem,
                evaluators,
                args.output,
                trials=args.trials,
                seed=args.seed,
                resume=args.resume,
                storage=args.storage,
                timeout=args.timeout,
                warn=progress.warning,
                progress=progress.trial,
            )
    except KeyboardInterrupt:
        print(
            f"k4opt: interrupted; continue the study with --resume --output {args.output}", file=err
        )
        return 1
    render_study(report, out, color=color(out))
    return 0


def _load(args: argparse.Namespace, err: TextIO) -> StudyResults:
    results = load_study(args.study, storage=args.storage)
    for warning in results.warnings:  # it can name a path or what a record holds
        print(f"k4opt: warning: {printable(warning)}", file=err)
    return results


def cmd_show(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    render_results(_load(args, err), out, color=color(out))
    return 0


def cmd_trial(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    results = _load(args, err)
    render_trial(results, results.trial(args.number), out, color=color(out))
    return 0


def cmd_export(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    export(_load(args, err), args.format, out, args.output)
    return 0
