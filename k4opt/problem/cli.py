"""Command-line interface of the problem component: ``k4opt evaluate``.

    k4opt evaluate PROBLEM --evaluator FILE.py:Class [--param REFERENCE=VALUE ...] --output DIR

A thin wrapper over :func:`k4opt.problem._run.run_candidate`; :func:`register` plugs it into
the top-level ``k4opt`` command (:mod:`k4opt.cli`). Exit code 0 only for a ``valid`` result.
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
            "Generate the geometry of one candidate, run ddsim on it and measure the "
            "simulated events. Needs the Key4hep environment."
        ),
    )
    evaluate.add_argument("problem", help="path to a k4opt problem YAML file")
    evaluate.add_argument(
        "--evaluator",
        required=True,
        metavar="FILE.py:Class",
        help=(
            "the evaluator to measure with, a file and class or package.module:Class; "
            "its name and version must be those of the problem. The file is executed, "
            "e.g. examples/evaluators/calo_photon.py:CaloPhoton"
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
    result = outcome.record.result
    print(f"status:      {result.status}", file=out)
    for m in result.measurements:
        error = "" if m.uncertainty is None else f" +- {m.uncertainty:.6g}"
        unit = f" [{m.unit}]" if m.unit else ""
        print(f"metric:      {m.stage}/{m.metric} = {m.value:.6g}{error}{unit}", file=out)
    if outcome.events_read is not None:  # None: no simulation output was inspected
        print(f"events:      {outcome.events_read} of {outcome.record.request.events}", file=out)
    print(f"fingerprint: {outcome.fingerprint}", file=out)
    print(f"summary:     {outcome.summary_path}", file=out)
    for role in ("stdout", "stderr", "edm4hep"):
        if f"simulation.{role}" in result.artifacts:
            print(f"{role + ':':12} {result.artifacts[f'simulation.{role}']}", file=out)


def cmd_evaluate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    outcome = run_candidate(
        args.problem, args.param, args.output, evaluator=load_evaluator(args.evaluator)
    )
    _report(outcome, out)
    result = outcome.record.result
    if result.status is not Status.VALID:
        print(f"k4opt: evaluation {result.status}: {result.message}", file=err)
        return 1
    return 0
