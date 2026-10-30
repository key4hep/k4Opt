"""Command-line interface of the optimization component: ``k4opt optimize``.

    k4opt optimize PROBLEM --evaluator FILE.py:Class [--evaluator ...] --trials N --output DIR
        [--seed S] [--resume] [--storage URL]

A thin wrapper over :func:`k4opt.optimization.optimize`; :func:`register` plugs it into the
top-level ``k4opt`` command (:mod:`k4opt.cli`). Exit code 0 when every trial asked for has
ended, whatever its candidate's status; 1 when the study cannot start or resume, a trial stops
it, or it is interrupted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import TextIO

from k4opt.evaluation import load_evaluator
from k4opt.optimization import StudyReport, TrialReport, optimize


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``optimize`` command to the top-level parser."""
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
    parser.set_defaults(func=cmd_optimize)


def _trial(trial: TrialReport) -> str:
    head = f"trial {trial.number}:".ljust(11) + trial.status.ljust(9)
    if trial.objectives is not None:
        feasible = "feasible  " if trial.feasible else "infeasible"
        values = "  ".join(f"{name}={value:.6g}" for name, value in trial.objectives.items())
        candidate = "  ".join(f"{name}={value}" for name, value in trial.candidate.items())
        return f"{head}{feasible}  {candidate}  {values}"
    where = f"{trial.directory.name}/summary.json" if trial.record else trial.directory.name
    return f"{head}{trial.message or ''}  {where}".rstrip()


def _report(report: StudyReport, out: TextIO) -> None:
    counts = Counter(trial.status for trial in report.trials)
    feasible = sum(trial.feasible is True for trial in report.trials)
    parts = [
        f"{counts['valid']} valid ({feasible} feasible, {counts['valid'] - feasible} infeasible)"
    ]
    parts += [
        f"{counts[status]} {status}"
        for status in ("invalid", "failed", "interrupted", "error")
        if counts[status]
    ]
    print(
        f"study:    {report.name}, {len(report.objectives)} objective(s), "
        f"{len(report.constraints)} limit(s), {report.study_path}",
        file=out,
    )
    print(f"trials:   {len(report.trials)}: {', '.join(parts)}", file=out)
    pareto = ", ".join(map(str, report.pareto)) if report.pareto else "none feasible yet"
    print(f"pareto:   {pareto}", file=out)


def cmd_optimize(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    evaluators = [load_evaluator(reference) for reference in args.evaluator]
    try:
        report = optimize(
            args.problem,
            evaluators,
            args.output,
            trials=args.trials,
            seed=args.seed,
            resume=args.resume,
            storage=args.storage,
            warn=lambda message: print(f"k4opt: warning: {message}", file=err, flush=True),
            progress=lambda trial: print(_trial(trial), file=out, flush=True),
        )
    except KeyboardInterrupt:
        print(
            f"k4opt: interrupted; continue the study with --resume --output {args.output}", file=err
        )
        return 1
    _report(report, out)
    return 0
