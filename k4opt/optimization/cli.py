"""Command-line interface of the optimization component: ``k4opt optimize`` and ``k4opt study``.

    k4opt optimize PROBLEM --evaluator FILE.py:Class [--evaluator ...] --trials N --output DIR
        [--strategy optuna|grid] [--grid NAME=V1,V2,...] [--seed S] [--resume] [--fix NAME=VALUE]
        [--reuse DIR] [--workers N|auto|inline] [--observe files|libraries|none]
    k4opt study show DIR
    k4opt study trial DIR NUMBER
    k4opt study export DIR --format {csv,json} [--output FILE]

Thin wrappers over :func:`k4opt.optimization.optimize` and :func:`k4opt.optimization.load_study`;
:func:`register` plugs them into the top-level ``k4opt`` command (:mod:`k4opt.cli`). ``optimize``
exits 0 when every trial asked for has ended, whatever its candidate's status, and 1 when the
study cannot start or resume, a trial stops it, or it is interrupted; a new study first prints
what it will do, as ``k4opt problem explain`` does, and the grid of a grid study. ``study`` exits
0 once it
has printed or exported the study, and 1 when the study cannot be read; it changes nothing in the
study, and an export writes only the data to its output, any warning to stderr.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import TextIO

from k4opt.evaluation import Scalar, load_evaluator
from k4opt.execution.cli import workers
from k4opt.optimization import STRATEGIES, StudyResults, load_study, optimize
from k4opt.optimization._export import FORMATS, export
from k4opt.optimization._grid import grid_values, size
from k4opt.optimization._render import Progress, render_study
from k4opt.optimization._scope import StudyScope, check_trials
from k4opt.optimization._show import render_results, render_trial
from k4opt.problem import CandidateError, ProblemSpec, load_problem
from k4opt.problem.candidate import parse_candidate
from k4opt.problem.outline import render_outline
from k4opt.program import OBSERVATIONS
from k4opt.terminal import color, interactive, printable


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``optimize`` and ``study`` commands to the top-level parser."""
    parser = commands.add_parser(
        "optimize",
        help="search the parameters of a problem for its best candidates",
        description=(
            "Run a study of a problem: a strategy (Optuna's TPE sampler, or a grid) proposes "
            "candidates, and each is evaluated as k4opt evaluate evaluates one, into a trial "
            "directory of the study. Needs the Key4hep environment, and Optuna for its strategy."
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
        "--strategy",
        choices=STRATEGIES,
        help=(
            "what proposes the candidates: 'optuna' (default), Optuna's TPE sampler; 'grid', every "
            "combination of the values --grid gives (each choice of a categorical parameter and "
            "each integer of an integer one without), in order. A resume keeps the study's"
        ),
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="NAME=V1,V2,...",
        help=(
            "the values the grid takes for a parameter, e.g. ecal_cell_size=2.5,5,10 "
            "(repeatable); a continuous parameter needs them. With --strategy grid"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="S",
        help="seed of Optuna's sampler, in [0, 2**32); drawn and recorded without it",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the study in DIR, made with exactly the same problem, files and code",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "give a parameter one value for every trial, which the optimizer then does not "
            "search, e.g. ecal_cell_size=6.0 (repeatable); part of the study: a resume needs the "
            "same"
        ),
    )
    parser.add_argument(
        "--reuse",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "a run or a study whose stages the trials reuse when a stage would be made from the "
            "same things (the same signature) and its output is still there, unchanged "
            "(repeatable); a study reuses its own trials' stages anyway. Not part of the study"
        ),
    )
    parser.add_argument(
        "--workers",
        type=workers,
        default=1,
        metavar="N|auto|inline",
        help=(
            "run the tasks in N worker processes, at most N at once, of one trial and of several; "
            "a simulation with numberOfThreads holds as many. 'auto': as many as the processors "
            "this process may use; 'inline': one task after another in this process, to debug, "
            "where a crash of an evaluator ends the study. It bounds what k4opt starts, not the "
            "memory of each task, and is not part of the study. Default 1: one trial after "
            "another, in one worker"
        ),
    )
    parser.add_argument(
        "--observe",
        choices=OBSERVATIONS,
        default="files",
        help=(
            "how closely the program of each stage is watched as it runs: 'files' (default) "
            "records every file it read, under strace, which another study needs to reuse its "
            "output; 'libraries' only the libraries it loaded; 'none' nothing. Where strace "
            "cannot run, the next level that works is taken, with a warning. It changes no "
            "output and is not part of the study"
        ),
    )
    parser.set_defaults(func=cmd_optimize)
    _register_study(commands)


def _register_study(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = commands.add_parser(
        "study",
        help="inspect or export the results of a study, without changing it",
        description=(
            "Read a study that k4opt optimize made, as its ledger and records hold it: nothing is "
            "evaluated or written, and the problem, geometry, evaluators and strategy need not "
            "exist any more. Each record is checked against the study and its ledger."
        ),
    )
    actions = group.add_subparsers(dest="action", required=True, metavar="ACTION")

    def action(name: str, help: str, description: str) -> argparse.ArgumentParser:
        parser = actions.add_parser(name, help=help, description=description)
        parser.add_argument(
            "study", metavar="DIR", help="the study directory (--output of optimize)"
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
    implementations = [load_evaluator(reference) for reference in args.evaluator]
    spec = load_problem(args.problem)
    fixed = parse_candidate(spec, args.fix, "--fix") if args.fix else None
    grid = _grid(spec, args.grid) if args.grid else None
    expected = args.trials  # the trials the progress bar counts toward
    if not args.resume:
        check_trials(args.trials)
        render_outline(spec, args.trials, out, fixed=fixed, color=color(out))
        if args.strategy == "grid":
            expected = min(expected, _preview(spec, fixed, grid, out))
    try:
        with Progress(expected, out, err, bar=interactive(out), color=color(out)) as progress:
            report = optimize(
                args.problem,
                implementations,
                args.output,
                trials=args.trials,
                strategy=args.strategy,
                grid=grid,
                seed=args.seed,
                resume=args.resume,
                workers=args.workers,
                reuse=args.reuse,
                observe=args.observe,
                fixed=fixed,
                warn=progress.warning,
                progress=progress.trial,
                activity=progress.activity,
            )
    except KeyboardInterrupt:
        print(
            f"k4opt: interrupted; continue the study with --resume --output {args.output}", file=err
        )
        return 1
    render_study(report, out, color=color(out))
    return 0


def _grid(spec: ProblemSpec, assignments: list[str]) -> dict[str, list[Scalar]]:
    """The values ``--grid NAME=V1,V2,...`` gives, each read as its parameter's kind."""
    grid: dict[str, list[Scalar]] = {}
    for assignment in assignments:
        name, equals, text = assignment.partition("=")
        if not equals or not text:
            raise CandidateError(f"--grid {assignment!r}: expected NAME=V1,V2,...")
        if name in grid:
            raise CandidateError(f"--grid {name} is given more than once")
        items = [parse_candidate(spec, [f"{name}={item}"], "--grid") for item in text.split(",")]
        grid[name] = [item[name] for item in items]
    return grid


def _preview(
    spec: ProblemSpec,
    fixed: Mapping[str, Scalar] | None,
    grid: Mapping[str, Sequence[Scalar]] | None,
    out: TextIO,
) -> int:
    """Print the grid of each level, before anything is made: how many points, of which values;
    refuse one that is not a grid of the study. How many points the top level has."""
    values = grid_values(StudyScope.of(spec, fixed), grid)
    for level in range(spec.depth + 1):
        own = [name for name in spec.parameters_of(level) if name in values]
        axes = "; ".join(f"{name} {', '.join(map(repr, values[name]))}" for name in own)
        where = "grid" if spec.depth == 0 else f"grid of level {level}"
        print(printable(f"{where}: {size(spec, values, level)} points: {axes}"), file=out)
    return size(spec, values, 0)


def _load(args: argparse.Namespace, err: TextIO) -> StudyResults:
    results = load_study(args.study)
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
