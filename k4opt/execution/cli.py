"""Command-line interface of the problem component: ``k4opt evaluate``.

    k4opt evaluate PROBLEM --evaluator FILE.py:Class [--evaluator ...] [--param NAME=VALUE ...]
        [--input SAMPLE/STAGE=FILE ...] [--replicate N] [--workers N|auto|inline]
        (--output DIR | --dry-run)

A thin wrapper over :func:`k4opt.execution._command.run_candidate`, or with ``--dry-run``
:func:`~k4opt.execution._command.preview_candidate`; :func:`register` plugs it into the top-level
``k4opt`` command (:mod:`k4opt.cli`). Exit code 0 only for a ``valid`` candidate, feasible or not,
whatever its diagnostics gave, or one ``--dry-run`` checked; 1 for a candidate that is not valid
or that an evaluator rejected (nothing is created for it).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from k4opt.evaluation import CandidateRejected, Status, load_evaluator
from k4opt.execution._command import Preview, preview_candidate, run_candidate
from k4opt.execution._preflight import Rejected
from k4opt.execution._publish import RunOutcome
from k4opt.frozen import Json
from k4opt.problem.models import GeometryDestination, StageDestination
from k4opt.problem.plan import Kind
from k4opt.program import OBSERVATIONS
from k4opt.record.assess import Limit
from k4opt.record.write import peak_memory
from k4opt.terminal import size

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
        metavar="NAME=VALUE",
        help="value of a parameter of the problem, e.g. ecal_cell_size=6.0 (repeatable)",
    )
    evaluate.add_argument(
        "--replicate",
        type=int,
        default=0,
        metavar="N",
        help=(
            "evaluate the candidate on other events: every seed of the problem replaced by one "
            "derived from it and N, the same on every machine (default 0: the problem's seeds); "
            "for samples that make their events from their seeds, without --input"
        ),
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
        "--reuse",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "a run or a study whose stages a stage of this candidate reuses when it would be "
            "made from the same things (the same signature: definition, inputs, geometry, files, "
            "programs, environment and k4opt code) and its output is still there, unchanged; "
            "the summary says where it comes from (repeatable)"
        ),
    )
    evaluate.add_argument(
        "--workers",
        type=workers,
        metavar="N|auto|inline",
        help=(
            "run the tasks (simulations of different samples, evaluators, the geometry) in N "
            "worker processes, at most N at once; a simulation with numberOfThreads holds as "
            "many. 'auto': as many as the processors this process may use. It bounds what k4opt "
            "starts, not the memory of each task. Default inline: one after another, in this "
            "process"
        ),
    )
    evaluate.add_argument(
        "--observe",
        choices=OBSERVATIONS,
        help=(
            "how closely the program of each stage is watched as it runs: 'files' (default) "
            "records every file it read, under strace, which another study needs to reuse its "
            "output; 'libraries' only the libraries it loaded; 'none' nothing. Where strace "
            "cannot run, the next level that works is taken, with a warning. It changes no "
            "output and the summary says how each stage was watched"
        ),
    )
    evaluate.add_argument("--output", help="new directory for the geometry, simulation and summary")
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "check the candidate as an evaluation would before it simulates (its values, the "
            "options of its stages, every evaluator's settings and check()) and show what each "
            "parameter sets: in the geometry, the stages and the evaluators. Nothing is "
            "simulated and no output is written, so no --output is needed; it does not show that "
            "the simulation or the evaluation will succeed"
        ),
    )
    evaluate.set_defaults(func=cmd_evaluate)


def workers(text: str) -> int | str:
    """``--workers``: a number (checked by the session), ``auto`` or ``inline``; the type of the
    option of every command that evaluates candidates."""
    if text in ("auto", "inline"):
        return text
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number of workers, auto or inline, got {text!r}"
        ) from None


def _limit(limit: Limit) -> str:
    """``calo.photon/simulation/x.rms <= 1.0: violated, margin 0.25 [mrad]``."""
    unit = f" [{limit.unit}]" if limit.unit else ""
    text = f"{limit.label} {'<=' if limit.bound == 'max' else '>='} {limit.limit}"
    if limit.satisfied is None:
        return f"{text}: not assessed{unit}"
    margin = "" if limit.margin is None else f", margin {limit.margin:.6g}"
    return f"{text}: {'satisfied' if limit.satisfied else 'violated'}{margin}{unit}"


def _value(value: object, unit: object) -> str:
    """A parameter's value as the candidate gives it, with the parameter's unit."""
    return f"{value!r}" + (f" {unit}" if isinstance(unit, str) else "")


def _written(value: Json) -> str:
    """A value as a destination holds it: a quantity as ``70 keV``, a text as it is."""
    if isinstance(value, Mapping) and set(value) == {"value", "unit"}:
        return _value(value["value"], value["unit"])
    return value if isinstance(value, str) else repr(value)


def _candidate(values: Mapping[str, object], units: Mapping[str, object], out: TextIO) -> None:
    """``candidate:   name = value unit``, one line per parameter, in the problem's order."""
    for index, (name, value) in enumerate(values.items()):
        label = "candidate:" if index == 0 else ""
        print(f"{label:12} {name} = {_value(value, units.get(name))}", file=out)


def _preview(preview: Preview, out: TextIO) -> None:
    """What each parameter of the candidate sets, below its value: each attribute of the geometry
    from its value now to the candidate's, each stage option in each sample, each evaluator
    setting in each entry of its binding (the samples the entry measures, or the candidate)."""
    spec, configuration = preview.spec, preview.configuration
    geometry = iter(zip(preview.targets, configuration.geometry, strict=True))
    lines = ["Candidate", ""]
    for parameter in spec.parameters:
        lines.append(f"{parameter.name} = {_value(preview.values[parameter.name], parameter.unit)}")
        for destination in parameter.targets:
            if isinstance(destination, GeometryDestination):
                target, text = next(geometry)
                lines.append(f"  geometry: {target.target} ({target.file})")
                lines.append(f"    {target.value} -> {text}")
            elif isinstance(destination, StageDestination):
                lines.append(f"  {destination.stage}: {destination.option}")
                for sample in destination.samples:
                    stage = configuration.samples[sample].stages[destination.stage]
                    lines.append(f"    {sample}: {_written(stage.options[destination.option])}")
            else:
                lines.append(f"  evaluator {destination.evaluator}: {destination.setting}")
                for entry in configuration.evaluators:
                    if entry.binding == destination.evaluator:
                        where = ", ".join(entry.samples) or "the candidate"
                        lines.append(
                            f"    {where}: {_written(entry.settings[destination.setting])}"
                        )
    lines += [
        "",
        "Configuration validated: the candidate's values, the options of every stage it runs, and "
        "every evaluator's settings and check().",
        "No simulation executed; no evaluation output created.",
    ]
    print("\n".join(lines), file=out)


def _report(outcome: RunOutcome, out: TextIO) -> None:
    candidate = outcome.candidate
    evaluations = outcome.summary["evaluations"]  # of the candidate, with what their roles read
    assert isinstance(evaluations, Mapping)
    values, problem = outcome.summary["candidate"], outcome.summary["problem"]
    assert isinstance(values, Mapping)
    assert isinstance(problem, Mapping)
    parameters = problem["definition"]["parameters"]  # the problem as the record holds it
    _candidate(values, {name: p.get("unit") for name, p in parameters.items()}, out)
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
        diagnostic = "" if task.required else " (diagnostic)"
        print(f"evaluator:   {name}: {result.status}{diagnostic}", file=out)
        if task.id.sample is None and (roles := evaluations[task.id.name]["roles"]):
            reads = "; ".join(f"{role} {', '.join(items)}" for role, items in roles.items())
            print(f"roles:       {reads}", file=out)
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
            "note:        a supplied stage that the candidate's values reach was not made "
            "from them; this is no evaluation of the candidate",
            file=out,
        )
    print(f"geometry:    {outcome.geometry_fingerprint or 'none (not made)'}", file=out)
    measured = [(m.per_task, label, m) for label, m in peak_memory(outcome.summary).items()]
    if needed := [entry for entry in measured if entry[0] is not None]:
        most, label, memory = max(needed, key=lambda entry: entry[0] or 0)
        parts = [f"k4opt {size(memory.process)}"] if memory.process is not None else []
        if memory.program is not None:
            ran = [r for r in candidate.results.values() if r.id.label == label]
            argv = ran[0].provenance.get("argv") if ran else None
            program = Path(str(argv[0])).name if isinstance(argv, (list, tuple)) and argv else ""
            parts += [f"{program or 'program'} {size(memory.program)}"]
        print(
            f"memory:      {size(most or 0)} per task (estimate), for {label} ({', '.join(parts)})",
            file=out,
        )
    print(f"summary:     {outcome.summary_path}", file=out)
    for made in (candidate.results[t.id] for t in stages if t.given is None):
        files = dict(made.artifacts)
        if "events" in made.made:
            files["edm4hep"] = made.made["events"].path
        for role in ("stdout", "stderr", "edm4hep"):
            if role in files:
                print(f"{role + ':':12} {made.id.label}: {files[role]}", file=out)


def cmd_evaluate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if (args.output is None) != args.dry_run:
        print("k4opt evaluate: give exactly one of --output DIR and --dry-run", file=err)
        return 2
    told = [name for name in ("workers", "observe") if getattr(args, name) is not None]
    if args.dry_run and told:
        print(
            f"k4opt evaluate: --{told[0]} says how a candidate runs, and --dry-run runs none; "
            "leave it out",
            file=err,
        )
        return 2
    implementations = [load_evaluator(reference) for reference in args.evaluator]
    if args.dry_run:
        try:
            preview = preview_candidate(
                args.problem,
                args.param,
                implementations=implementations,
                inputs=args.input,
                warn=lambda message: print(f"k4opt: warning: {message}", file=err, flush=True),
                replicate=args.replicate,
            )
        except CandidateRejected as exc:
            print(f"k4opt: candidate {exc}; nothing would be created", file=err)
            return 1
        _preview(preview, out)
        return 0
    outcome = run_candidate(
        args.problem,
        args.param,
        args.output,
        implementations=implementations,
        inputs=args.input,
        workers="inline" if args.workers is None else args.workers,
        reuse=args.reuse,
        observe=args.observe or "files",
        warn=lambda message: print(f"k4opt: warning: {message}", file=err, flush=True),
        replicate=args.replicate,
    )
    if isinstance(outcome, Rejected):
        print(f"k4opt: candidate {outcome}; nothing was created", file=err)
        return 1
    _report(outcome, out)
    candidate = outcome.candidate
    for result in candidate.diagnostics:  # the candidate's status leaves them out
        print(
            f"k4opt: warning: diagnostic {result.id.label} {result.status}: {result.message}",
            file=err,
        )
    if candidate.status is not Status.VALID:
        print(f"k4opt: evaluation {candidate.status}: {candidate.message}", file=err)
        return 1
    return 0
