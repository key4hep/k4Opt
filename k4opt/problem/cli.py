"""Command-line interface of the problem component: ``k4opt evaluate``.

    k4opt evaluate PROBLEM --evaluator FILE.py:Class [--evaluator ...] [--param REFERENCE=VALUE ...]
        [--input SAMPLE/STAGE=FILE ...] [--timeout SECONDS] --output DIR

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
from k4opt.problem._tasks import Kind

_FEASIBLE = {True: "yes", False: "no", None: "not assessed"}


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the ``evaluate`` command to the top-level parser."""
    evaluate = commands.add_parser(
        "evaluate",
        help="simulate one candidate of a problem and measure it",
        description=(
            "Generate the geometry of one candidate, run the stages of each sample of the problem "
            "on it once (ddsim) and measure the result with every evaluator of the problem. "
            "Needs the Key4hep environment."
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
            "for each evaluator the problem names, whose name and version it must have, once "
            "however often the problem binds it. The file is executed, e.g. "
            "examples/evaluators/calo_photon.py:CaloPhoton"
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
        metavar="SAMPLE/STAGE=FILE",
        help=(
            "use FILE as the output of STAGE of SAMPLE instead of running it, e.g. "
            "gamma/simulation=events.root to try an evaluator on an existing file (repeatable). "
            "A simulation supplied this way was not made with this candidate's geometry, and the "
            "summary says so"
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
    """``calo.photon/simulation/x.rms <= 1.0: violated, margin 0.25 [mrad]``."""
    unit = f" [{limit.unit}]" if limit.unit else ""
    text = f"{limit.label} {'<=' if limit.bound == 'max' else '>='} {limit.limit}"
    if limit.satisfied is None:
        return f"{text}: not assessed{unit}"
    margin = "" if limit.margin is None else f", margin {limit.margin:.6g}"
    return f"{text}: {'satisfied' if limit.satisfied else 'violated'}{margin}{unit}"


def _report(outcome: RunOutcome, out: TextIO) -> None:
    candidate = outcome.candidate
    print(f"status:      {candidate.status}", file=out)
    for task in candidate.plan.of(Kind.EVALUATOR):
        record = candidate.results[task.id].record
        assert record is not None  # every evaluation has one, a blocked one too
        result = record.result
        name = f"{record.evaluator_name} {record.evaluator_version}"
        if task.id.name != record.evaluator_name:
            name = f"{task.id.name} ({name})"
        if task.id.sample is not None:
            name = f"{name} on {task.id.sample}"
        print(f"evaluator:   {name}: {result.status}", file=out)
        for m in result.measurements:
            error = "" if m.uncertainty is None else f" +- {m.uncertainty:.6g}"
            unit = f" [{m.unit}]" if m.unit else ""
            print(f"metric:      {m.stage}/{m.metric} = {m.value:.6g}{error}{unit}", file=out)
    for limit in outcome.assessment.limits:
        print(f"limit:       {_limit(limit)}", file=out)
    print(f"feasible:    {_FEASIBLE[outcome.assessment.feasible]}", file=out)
    for sample, (read, requested) in outcome.events.items():
        print(f"events:      {sample}: {read} of {requested}", file=out)
    stages = candidate.plan.of(Kind.STAGE)
    for task in stages:
        if task.given == "supplied":
            given = candidate.results[task.id].provenance["provided"]
            print(f"supplied:    {task.id.label} from {given}", file=out)
    if not candidate.represents_candidate:
        print(
            "note:        a supplied stage was not made with this candidate's geometry; "
            "this is no evaluation of the candidate",
            file=out,
        )
    print(f"fingerprint: {outcome.fingerprint}", file=out)
    print(f"summary:     {outcome.summary_path}", file=out)
    for simulation in (candidate.results[t.id] for t in stages if t.id.name == "simulation"):
        files, at = simulation.artifacts, simulation.id.sample
        for role in ("stdout", "stderr", "edm4hep"):
            if role in files:
                print(f"{role + ':':12} {at}: {files[role]}", file=out)


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
