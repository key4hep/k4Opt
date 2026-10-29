"""Command-line interface of the problem component: ``k4opt evaluate``.

    k4opt evaluate PROBLEM --evaluator FILE.py:Class [--evaluator ...] [--param REFERENCE=VALUE ...]
        --output DIR

A thin wrapper over :func:`k4opt.problem._run.run_candidate`; :func:`register` plugs it into
the top-level ``k4opt`` command (:mod:`k4opt.cli`). Exit code 0 only for a ``valid`` candidate.
"""

from __future__ import annotations

import argparse
from typing import TextIO

from k4opt.evaluation import Status, load_evaluator
from k4opt.problem._run import RunOutcome, run_candidate


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``evaluate`` command to the top-level parser."""
    evaluate = commands.add_parser(
        "evaluate",
        help="simulate one candidate of a problem and measure it",
        description=(
            "Generate the geometry of one candidate, run ddsim on it once and measure the "
            "simulated events with every evaluator of the problem. Needs the Key4hep "
            "environment."
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
        "--output", required=True, help="new directory for the geometry, simulation and summary"
    )
    evaluate.set_defaults(func=cmd_evaluate)


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
    if outcome.events_read is not None:  # None: no simulation output was inspected
        requested = candidate.records[0].request.events
        print(f"events:      {outcome.events_read} of {requested}", file=out)
    print(f"fingerprint: {outcome.fingerprint}", file=out)
    print(f"summary:     {outcome.summary_path}", file=out)
    files = candidate.stages.artifacts.get("simulation", {})
    for role in ("stdout", "stderr", "edm4hep"):
        if role in files:
            print(f"{role + ':':12} {files[role]}", file=out)


def cmd_evaluate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    evaluators = [load_evaluator(reference) for reference in args.evaluator]
    outcome = run_candidate(args.problem, args.param, args.output, evaluators=evaluators)
    _report(outcome, out)
    candidate = outcome.candidate
    if candidate.status is not Status.VALID:
        print(f"k4opt: evaluation {candidate.status}: {candidate.message}", file=err)
        return 1
    return 0
