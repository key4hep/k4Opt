"""Internal: the pipeline stages evaluators can read, each run at most once per candidate.

Each sample of a problem says how each of its stages is made: run by an executor (``gun`` for
``generation``, ``ddsim`` for ``simulation``) or given as a file. An evaluator names the stage it
reads with ``stage`` (``"simulation"``); the framework makes that stage of each sample it
measures, and the stages it depends on, available and hands the evaluator the files of all of
them as ``request.inputs[stage]``. However many evaluators read a stage of a sample, it runs
once. An evaluator therefore never builds a command, starts a tool or handles that tool's
failures.

* ``generation``: where the primary particles come from. The ``gun`` executor has no file of
  its own (``ddsim`` fires it); a generation given as a file (an event file such as HepMC) is
  checked, hashed and handed on as ``inputs["generation"]``.
* ``simulation``: ``ddsim`` on the prepared geometry; reads ``generation``. Its output, run or
  given, is checked to hold its events (the requested number, when it ran), so no evaluator has
  to.

The stage types are described once, in ``_STAGES``; :mod:`k4opt.problem.config` checks a problem
against the same table. Each stage of a candidate is one task (:mod:`k4opt.problem._tasks`),
whose work :func:`make_stage` does in the task's own directory: a stage that fails blocks only
the tasks that read it. Each stage has named outputs (``events``); ``request.inputs[stage]`` is
the main one.

The stages of a sample take their conditions from its :class:`StageContext`, never from the
problem itself.

Pipeline stages are steps that run. The ``stage`` of a measurement, objective or constraint
only classifies a metric: an evaluator may report ``analysis`` metrics without an ``analysis``
pipeline stage, and nothing here reads metric labels.

A file the caller supplies as ``inputs[stage]`` (``k4opt evaluate --input``) is authoritative:
that stage is not run, the stages that only it needs are not run either, and the file replaces
what the problem says. Unlike a ``file`` in the problem, it is recorded as ``supplied``: a
supplied stage that the parameters change was not made from the candidate.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from k4opt.ddsim import (
    FILE_OPTIONS,
    LOG_FILES,
    check_arguments,
    check_options,
    check_particle_gun,
    simulate,
)
from k4opt.evaluation import EvaluationContractError, EvaluationFailed, count_events
from k4opt.frozen import Json
from k4opt.problem._files import file_identity, readable_identity
from k4opt.problem.models import ProblemSpec, Stage


@dataclass(frozen=True, slots=True)
class StageContext:
    """The conditions the pipeline stages of one sample run under: its ``stages``.

    ``directory`` is where a relative stage ``file`` (or a file named in an option, such as
    ``physics.pdgfile``) is read from: the directory of the problem file, the current one if it
    is ``None``. ``timeout`` is how many seconds a stage that runs a program (ddsim) may take
    before the program is killed and the stage fails; ``None`` for no limit. It is the caller's,
    not the problem's. ``sample`` names the sample in messages and labels.

    ``started`` and ``die_with_parent`` say how the process that runs a stage watches the
    program it starts (:func:`~k4opt.ddsim.simulate`): a worker process reports the program's
    process group, and has it die with the worker. They belong to that process alone and are
    never part of a description sent elsewhere.
    """

    stages: Mapping[str, Stage] = field(default_factory=dict)
    directory: Path | None = None
    timeout: float | None = None
    sample: str | None = None
    started: Callable[[int], None] | None = field(default=None, compare=False, repr=False)
    die_with_parent: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def of(cls, spec: ProblemSpec, sample: str) -> StageContext:
        """The stages of the sample ``sample`` of ``spec``."""
        return cls(spec.samples[sample].stages, spec._directory, sample=sample)

    def where(self, stage: str) -> str:
        """Where the problem defines ``stage``: ``samples.<sample>.<stage>``."""
        return stage if self.sample is None else f"samples.{self.sample}.{stage}"

    def path(self, value: str) -> Path:
        """The absolute path ``value`` stands for, a relative one taken from ``directory``."""
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(os.path.abspath((self.directory or Path.cwd()) / path))

    def files(self) -> dict[str, Path]:
        """The absolute path of every stage the problem gives as a file, by stage."""
        return {
            name: self.path(stage.file)
            for name, stage in self.stages.items()
            if stage.file is not None
        }

    def conditions(self, name: str | None) -> tuple[int | None, int | None]:
        """``(events, seed)`` of the stage ``name``, ``None`` where it has none."""
        stage = None if name is None else self.stages.get(name)
        return (None, None) if stage is None else (stage.events, stage.seed)


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What a stage left behind: its named outputs, its files and how it ran.

    ``outputs`` holds what later stages and evaluators read (``events``); ``artifacts`` the other
    files (logs, the command), never an output. ``error`` is ``None`` only if the stage did what
    it should; otherwise it says what failed. ``peak_memory_bytes`` is that of the program it ran
    (:attr:`~k4opt.ddsim.SimulationRun.peak_memory_bytes`), ``None`` if it ran none or it is not
    known.
    """

    outputs: Mapping[str, Path] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    provenance: Mapping[str, Json] = field(default_factory=dict)
    error: str | None = None
    peak_memory_bytes: int | None = None


def provided_provenance(path: Path) -> dict[str, Json]:
    """What is recorded for an input the caller supplied instead of running its stage."""
    provenance: dict[str, Json] = {"provided": str(path)}
    # an unusable file is the evaluator's to report; the record still names it
    with contextlib.suppress(OSError):
        provenance |= file_identity(path)
    return provenance


def _primaries(context: StageContext, given: Path | None) -> tuple[Json, Path | None]:
    """``(particle_gun, input_file)`` for ddsim: a generation file, given or supplied, wins."""
    if given is not None:
        return None, given
    generation = context.stages.get("generation")
    return (None if generation is None else generation.options), None


def _check_generation(context: StageContext, provided: Collection[str]) -> None:
    stage = context.stages.get("generation")
    if stage is None:
        raise EvaluationContractError(
            f"{context.where('generation')}: the sample has no generation stage to say where the "
            "particles come from"
        )
    if stage.executor is not None:
        check_particle_gun(stage.options)


def _run_generation(
    context: StageContext, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    # only a gun runs: a generation given as a file is not run
    return StageOutcome(provenance={"particle_gun": context.stages["generation"].options})


def _option_files(context: StageContext, options: Mapping[str, Json]) -> dict[str, Path]:
    """The files that ddsim options read, by option, as absolute paths."""
    return {
        key: context.path(value)
        for key, value in options.items()
        if key in FILE_OPTIONS and isinstance(value, str)
    }


def _check_simulation(context: StageContext, provided: Collection[str]) -> None:
    stage = context.stages.get("simulation")
    if stage is None or stage.executor is None:
        raise EvaluationContractError(
            f"{context.where('simulation')}: the sample has no simulation to run"
        )
    if stage.events is None:
        raise EvaluationContractError(
            f"{context.where('simulation')}.events: ddsim needs the events to simulate"
        )
    given = "generation" in provided or "generation" in context.files()
    particle_gun, input_file = _primaries(context, Path("generation") if given else None)
    check_arguments(
        events=stage.events,
        seed=stage.seed,
        particle_gun=particle_gun,
        input_file=input_file,
        options=stage.options,
    )
    for key, path in _option_files(context, stage.options).items():
        if not path.is_file():
            raise EvaluationContractError(
                f"{context.where('simulation')}.{key}: {path} is not a file"
            )


def _run_simulation(
    context: StageContext, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    stage = context.stages["simulation"]
    assert stage.events is not None  # checked by _check_simulation
    assert workdir is not None  # checked by check_run_inputs
    particle_gun, input_file = _primaries(context, inputs.get("generation"))
    files = _option_files(context, stage.options)
    read = {key: readable_identity(path) for key, path in files.items()}  # before ddsim does
    run = simulate(
        compact=inputs["geometry"],
        workdir=workdir,
        events=stage.events,
        seed=stage.seed,
        particle_gun=particle_gun,
        input_file=input_file,
        options={key: str(files[key]) if key in files else v for key, v in stage.options.items()},
        timeout=context.timeout,
        started=context.started,
        die_with_parent=context.die_with_parent,
    )
    # it ran in workdir, the stage's own directory, which the record names relative to the run
    provenance: dict[str, Json] = {
        "argv": run.argv,
        "returncode": run.returncode,
        "timeout": context.timeout,
    }
    error = run.error
    if files:
        recorded: dict[str, Json] = {}
        for key, path in files.items():
            identity = read[key]
            recorded[key] = {"path": str(path), **(identity or {})}
            if identity is not None and readable_identity(path) != identity:
                changed = f"{key}: {path} changed while ddsim ran; the record cannot say which"
                error = changed if error is None else f"{error}\n{changed}"
        provenance["files"] = recorded
    logs = {role: path for role, path in run.files.items() if role != "edm4hep"}
    peak = run.peak_memory_bytes
    if error is not None:
        return StageOutcome(
            artifacts=logs, provenance=provenance, error=error, peak_memory_bytes=peak
        )
    return StageOutcome({"events": run.output}, logs, provenance, peak_memory_bytes=peak)


def _inspect_simulation(context: StageContext, path: Path) -> tuple[dict[str, Json], str | None]:
    """``(observations, error)`` for the EDM4hep file of the simulation stage."""
    try:
        count = count_events(path)
    except EvaluationFailed as exc:
        return {}, str(exc)
    observed: dict[str, Json] = {"events_read": count}
    expected, _ = context.conditions("simulation")
    if expected is not None and count != expected:
        return observed, f"incomplete simulation: {count} of {expected} events in {path.name}"
    if count == 0:
        return observed, f"{path.name} contains no events"
    return observed, None


@dataclass(frozen=True, slots=True)
class _Executor:
    events: bool  # takes `events` (required) and `seed`
    options: Callable[[Mapping[str, Json]], None]  # checks its own keys; raises DdsimError
    inside: str | None = None  # the stage whose process runs it, which sets its events and seed


@dataclass(frozen=True, slots=True)
class _Stage:
    needs: tuple[str, ...]  # the stage types it reads
    executors: Mapping[str, _Executor]
    check: Callable[[StageContext, Collection[str]], None]
    run: Callable[[StageContext, Mapping[str, Path], Path | None], StageOutcome]
    geometry: bool  # reads the prepared geometry, inputs["geometry"]
    writes: bool  # writes files, so it needs a workdir
    output: str = "events"  # the name of its main output
    logs: Mapping[str, str] = field(default_factory=dict)  # its other files' names, by role
    # checks the output file, whether the stage wrote it or it was given
    inspect: Callable[[StageContext, Path], tuple[dict[str, Json], str | None]] | None = None


_STAGES: Mapping[str, _Stage] = {
    "generation": _Stage(
        (),
        {"gun": _Executor(events=False, options=check_particle_gun, inside="simulation")},
        _check_generation,
        _run_generation,
        geometry=False,
        writes=False,
    ),
    "simulation": _Stage(
        ("generation",),
        {"ddsim": _Executor(events=True, options=check_options)},
        _check_simulation,
        _run_simulation,
        geometry=True,
        writes=True,
        inspect=_inspect_simulation,
        logs=LOG_FILES,
    ),
}

STAGE_TYPES = tuple(_STAGES)
"""The stage types, in pipeline order."""


def stage_reads(stage: str) -> tuple[str, ...]:
    """The stage types that ``stage`` reads."""
    return _STAGES[stage].needs


def reads_geometry(stage: str) -> bool:
    """Whether ``stage``, when it runs, reads the prepared geometry."""
    return _STAGES[stage].geometry


def writes_files(stage: str) -> bool:
    """Whether ``stage``, when it runs, writes files, and so needs a directory of its own."""
    return _STAGES[stage].writes


def stage_logs(stage: str, directory: Path) -> dict[str, Path]:
    """The files besides its outputs (logs, the command) that ``stage`` left in its
    ``directory``, by role, as far as they exist: what a stage whose process died left."""
    files = {role: directory / name for role, name in _STAGES[stage].logs.items()}
    return {role: path for role, path in files.items() if path.is_file()}


def stage_output(stage: str) -> str:
    """The name of the main output of ``stage`` (``events``)."""
    return _STAGES[stage].output


def executors(stage: str) -> Mapping[str, _Executor]:
    """The executors that can run ``stage``, by name."""
    return _STAGES[stage].executors


# podio reads EDM4hep written with ROOT (TTree or RNTuple) or SIO; other event files (HepMC, ...)
# are inputs of ddsim, not files an evaluator can read collections from
_PODIO_SUFFIXES = (".root", ".sio")


def no_edm4hep_file(
    context: StageContext, stage: str, supplied: Mapping[str, Path] | None = None
) -> str | None:
    """Why ``stage`` makes no EDM4hep file that an evaluator could read collections from,
    ``None`` if it does. A file the caller supplies or the problem gives must be one podio reads
    (``.root``, ``.sio``), not an event file such as HepMC; an executor that runs inside another
    stage (the gun, fired by ddsim) writes no file of its own."""
    path = (supplied or {}).get(stage) or context.files().get(stage)
    if path is not None:
        if path.suffix in _PODIO_SUFFIXES:
            return None
        readable = " or ".join(_PODIO_SUFFIXES)
        return (
            f"its file {path.name} is no EDM4hep file that podio reads ({readable}); read the "
            f"collections from a stage that writes one, such as {context.where('simulation')}"
        )
    spec = context.stages.get(stage)
    if spec is None or spec.executor is None:
        return None  # not defined: reported elsewhere
    inside = _STAGES[stage].executors[spec.executor].inside
    if inside is None:
        return None
    return (
        f"executor {spec.executor!r} runs inside {context.where(inside)} and writes no file of "
        f"its own; read the collections from {context.where(inside)}, or give "
        f"{context.where(stage)} as a file"
    )


def changed_by_geometry(stage: str) -> bool:
    """Whether ``stage`` depends on the candidate geometry: it reads it, or reads a stage that
    does. A file given for such a stage was not made from the candidate."""
    return _STAGES[stage].geometry or any(changed_by_geometry(need) for need in stage_reads(stage))


def graph(targets: Collection[str], given: Collection[str]) -> list[tuple[str, tuple[str, ...]]]:
    """The nodes that make ``targets`` available, each with the nodes it reads, dependencies
    first. A node in ``given`` is a leaf: it is not run, so it reads nothing."""
    order: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        reads = () if name in given else _STAGES[name].needs
        for need in reads:
            visit(need)
        order.append((name, reads))

    for target in targets:
        if target not in _STAGES:
            raise EvaluationContractError(
                f"unknown stage {target!r}; available: {', '.join(sorted(_STAGES))}"
            )
        visit(target)
    return order


def plan_stages(*targets: str, provided: Collection[str] = ()) -> list[str]:
    """The stages to run, dependencies first, to make every stage of ``targets`` available.

    Each stage appears once, however many targets need it. A stage whose name is in
    ``provided`` (a given file) is not run, and neither are the stages only it would need.
    """
    return [name for name, _ in graph(targets, provided) if name not in provided]


def check_stages(context: StageContext, *targets: str, provided: Collection[str] = ()) -> list[str]:
    """Raise if a stage of ``targets`` or one they need is unknown or cannot run in
    ``context``; return the plan.

    ``provided`` names the stages the caller supplies. A stage the problem gives as a file must
    exist. Needs no geometry and starts nothing, so it can run before any file is prepared.
    """
    files = {name: path for name, path in context.files().items() if name not in provided}
    given = {*provided, *files}
    plan = plan_stages(*targets, provided=given)
    for name in plan:
        _STAGES[name].check(context, given)
    for name, _ in graph(targets, given):
        if name in files and not files[name].is_file():
            raise EvaluationContractError(
                f"{context.where(name)}.file: {files[name]} is not a file"
            )
    return plan


def input_files(
    context: StageContext, *targets: str, provided: Collection[str] = ()
) -> dict[str, Path]:
    """Every file the problem names that the stages of ``targets`` read, by where it is named.

    ``samples.<sample>.<stage>.file`` for a stage given as a file,
    ``samples.<sample>.<stage>.<option>`` for a file an executor option names
    (``samples.gamma.simulation.physics.pdgfile``). The stages in ``provided``, and those only
    they need, read none: the caller supplies their files.
    """
    files = {name: path for name, path in context.files().items() if name not in provided}
    named: dict[str, Path] = {}
    for name, _ in graph(targets, {*provided, *files}):
        stage = context.stages.get(name)  # each one is defined once check_stages() passed
        if name in files:
            named[f"{context.where(name)}.file"] = files[name]
        elif name not in provided and stage is not None:
            options = _option_files(context, stage.options)
            named |= {f"{context.where(name)}.{key}": path for key, path in options.items()}
    return named


def check_run_inputs(run: Collection[str], geometry: bool, workdir: Path | None) -> None:
    """Raise if a stage that is to ``run`` needs a geometry and there is none, or a directory and
    there is no ``workdir`` to make it in; stages given as files need neither."""
    for name in run:
        stage = _STAGES[name]
        if stage.geometry and not geometry:
            raise EvaluationContractError(f"the {name!r} stage needs inputs['geometry']")
        if stage.writes and workdir is None:
            raise EvaluationContractError(f"the {name!r} stage needs a workdir")


def run_stage(
    context: StageContext, stage: str, *, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    """Run one ``stage`` in its own new directory ``workdir``; ``inputs`` holds the geometry and
    the main output of each stage it reads, by name."""
    return _STAGES[stage].run(context, inputs, workdir)


def make_stage(
    context: StageContext,
    stage: str,
    *,
    inputs: Mapping[str, Path],
    workdir: Path | None,
    supplied: Path | None = None,
) -> tuple[StageOutcome, Path | None]:
    """Make ``stage`` available, the work of its task: ``(outcome, main output)``.

    A file the caller ``supplied`` stands in for the stage, then a ``file`` the problem gives
    (hashed: its size and SHA-256 are recorded); otherwise the stage runs (:func:`run_stage`).
    The main output, run or given, is then checked to hold what the stage promises (the requested
    events of a simulation). ``outcome.error`` says why the stage cannot be read; the main output
    is ``None`` then, and for a stage that gives no file (a particle gun).

    ``context`` must already have passed :func:`check_stages`; the preflight of a candidate does
    that before any file is prepared. It is not checked again here: a missing file is then a
    failed stage, not a contract error.
    """
    spec = _STAGES[stage]
    path = context.files().get(stage)
    if supplied is not None:
        outcome = StageOutcome({spec.output: supplied}, provenance=provided_provenance(supplied))
    elif path is not None:
        try:
            outcome = StageOutcome(
                {spec.output: path}, provenance={"file": str(path)} | file_identity(path)
            )
        except OSError as exc:
            outcome = StageOutcome(
                provenance={"file": str(path)}, error=f"cannot read {path}: {exc}"
            )
    else:
        outcome = run_stage(context, stage, inputs=inputs, workdir=workdir)
    main = outcome.outputs.get(spec.output)
    if outcome.error is not None or main is None or spec.inspect is None:
        return outcome, None if outcome.error is not None else main
    observed, error = spec.inspect(context, main)
    provenance = {**outcome.provenance, **observed}
    outcome = replace(outcome, provenance=provenance, error=error)
    return outcome, None if error is not None else main
