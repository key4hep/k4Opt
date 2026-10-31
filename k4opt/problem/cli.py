"""Command-line interface of the problem component: ``k4opt evaluate``.

    k4opt evaluate PROBLEM --evaluator FILE.py:Class [--evaluator ...] [--param REFERENCE=VALUE ...]
        [--input STAGE=FILE ...] [--timeout SECONDS] --output DIR

A thin wrapper over :func:`k4opt.problem._run.run_candidate`; :func:`register` plugs it into
the top-level ``k4opt`` command (:mod:`k4opt.cli`). Exit code 0 only for a ``valid`` candidate,
feasible or not.
"""

from __future__ import annotations

import argparse
from typing import TextIO

from k4opt.evaluation import Status, load_evaluator
from k4opt.problem._assess import Limit
from k4opt.problem._run import RunOutcome, run_candidate

_FEASIBLE = {True: "yes", False: "no", None: "not assessed"}


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``evaluate`` command to the top-level parser."""
    evaluate = commands.add_parser(
        "evaluate",
        help="simulate one candidate of a problem and measure it",
        description=(
            "Generate the geometry of one candidate, run the stages of the problem on it once "
            "(ddsim) and measure the result with every evaluator of the problem. Needs the "
            "Key4hep environment."
        ),
    )
    evaluate.add_argument("problem", help="path to a k4opt problem YAML file")
    evaluate.add_argument(
        "--evaluator",
        action="append",
        required=True,
        metavar="FILE.py:Class",
        help=(
            "an evaluator to measure with, a file and class or package.module:Class; repeat it "
            "for each evaluator of the problem, whose name and version it must have. The file "
            "is executed, e.g. examples/evaluators/calo_photon.py:CaloPhoton"
        ),
    )
    evaluate.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="REFERENCE=VALUE",
        help="value of a declared parameter, e.g. constants.ECal_cell_size=6.0 (repeatable)",
    )
    evaluate.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="STAGE=FILE",
        help=(
            "use FILE as the output of STAGE instead of running it, e.g. simulation=events.root "
            "to try an evaluator on an existing file (repeatable). A simulation supplied this way "
            "was not made with this candidate's geometry, and the summary says so"
        ),
    )
    evaluate.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="kill ddsim if it runs longer than this; the candidate fails (no limit without it)",
    )
    evaluate.add_argument(
        "--output", required=True, help="new directory for the geometry, simulation and summary"
    )
    evaluate.set_defaults(func=cmd_evaluate)


def _limit(limit: Limit) -> str:
    """``simulation/x.rms <= 1.0: violated, margin 0.25 [mrad]``."""
    unit = f" [{limit.unit}]" if limit.unit else ""
    text = f"{limit.stage}/{limit.metric} {'<=' if limit.bound == 'max' else '>='} {limit.limit}"
    if limit.satisfied is None:
        return f"{text}: not assessed{unit}"
    margin = "" if limit.margin is None else f", margin {limit.margin:.6g}"
    return f"{text}: {'satisfied' if limit.satisfied else 'violated'}{margin}{unit}"


def _report(outcome: RunOutcome, out: TextIO) -> None:
    candidate = outcome.candidate
    print(f"status:      {candidate.status}", file=out)
    for record in candidate.records:
        result = record.result
        name = f"{record.evaluator_name} {record.evaluator_version}"
        print(f"evaluator:   {name}: {result.status}", file=out)
        for m in result.measurements:
            error = "" if m.uncertainty is None else f" +- {m.uncertainty:.6g}"
            unit = f" [{m.unit}]" if m.unit else ""
            print(f"metric:      {m.stage}/{m.metric} = {m.value:.6g}{error}{unit}", file=out)
    for limit in outcome.assessment.limits:
        print(f"limit:       {_limit(limit)}", file=out)
    print(f"feasible:    {_FEASIBLE[outcome.assessment.feasible]}", file=out)
    if outcome.events_read is not None:  # None: no simulation output was inspected
        print(f"events:      {outcome.events_read} of {outcome.events_requested}", file=out)
    for stage in sorted(candidate.supplied):
        given = candidate.stage(stage)
        assert given is not None  # a supplied stage is planned
        print(f"supplied:    {stage} from {given.provenance['provided']}", file=out)
    if not candidate.represents_candidate:
        print(
            "note:        a supplied stage was not made with this candidate's geometry; "
            "this is no evaluation of the candidate",
            file=out,
        )
    print(f"fingerprint: {outcome.fingerprint}", file=out)
    print(f"summary:     {outcome.summary_path}", file=out)
    simulation = candidate.stage("simulation")
    files = {} if simulation is None else simulation.artifacts
    for role in ("stdout", "stderr", "edm4hep"):
        if role in files:
            print(f"{role + ':':12} {files[role]}", file=out)


def cmd_evaluate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    implementations = [load_evaluator(reference) for reference in args.evaluator]
    outcome = run_candidate(
        args.problem,
        args.param,
        args.output,
        implementations=implementations,
        inputs=args.input,
        timeout=args.timeout,
        warn=lambda message: print(f"k4opt: warning: {message}", file=err, flush=True),
    )
    _report(outcome, out)
    candidate = outcome.candidate
    if candidate.status is not Status.VALID:
        print(f"k4opt: evaluation {candidate.status}: {candidate.message}", file=err)
        return 1
    return 0
