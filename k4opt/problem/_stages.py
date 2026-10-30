"""Internal: the pipeline stages evaluators can read, each run at most once per candidate.

A problem's ``stages`` say how each stage is made: run by an executor (``gun`` for
``generation``, ``ddsim`` for ``simulation``) or given as a file. An evaluator names the stage it
reads with ``stage`` (``"simulation"``); the framework makes that stage and the stages it
depends on available and hands the evaluator the files of all of them as
``request.inputs[stage]``. However many evaluators read a stage, it runs once. An evaluator
therefore never builds a command, starts a tool or handles that tool's failures.

* ``generation``: where the primary particles come from. The ``gun`` executor has no file of
  its own (``ddsim`` fires it); a generation given as a file (an event file such as HepMC) is
  checked, hashed and handed on as ``inputs["generation"]``.
* ``simulation``: ``ddsim`` on the prepared geometry; reads ``generation``. Its output, run or
  given, is checked to hold its events (the requested number, when it ran), so no evaluator has
  to.

The stage types are described once, in ``_STAGES``; :mod:`k4opt.problem.config` checks a problem
against the same table. A candidate's plan is a list of stage nodes with the nodes each reads,
and :func:`run_stages` walks that list: a stage that fails blocks only the stages that depend on
it. Each stage has named outputs (``events``); ``request.inputs[stage]`` is the main one.

The stages take their conditions from a :class:`StageContext`, never from the problem itself.

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
import functools
import hashlib
import os
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from k4opt.ddsim import (
    FILE_OPTIONS,
    check_arguments,
    check_options,
    check_particle_gun,
    simulate,
)
from k4opt.evaluation import EvaluationContractError, EvaluationFailed, count_events
from k4opt.frozen import FrozenMap, Json, freeze
from k4opt.problem.models import ProblemSpec, Stage


@dataclass(frozen=True, slots=True)
class StageContext:
    """The conditions the pipeline stages run under: the problem's ``stages``.

    ``directory`` is where a relative stage ``file`` (or a file named in an option, such as
    ``physics.pdgfile``) is read from: the directory of the problem file, the current one if it
    is ``None``.
    """

    stages: Mapping[str, Stage] = field(default_factory=dict)
    directory: Path | None = None

    @classmethod
    def of(cls, spec: ProblemSpec) -> StageContext:
        """The stages of ``spec``."""
        return cls(spec.stages, spec._directory)

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
    files (logs, the command). ``error`` is ``None`` only if the stage did what it should;
    otherwise it says what failed.
    """

    outputs: Mapping[str, Path] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    provenance: Mapping[str, Json] = field(default_factory=dict)
    error: str | None = None


@functools.lru_cache(maxsize=32)
def _sha256(path: str, size: int, mtime_ns: int, ctime_ns: int, inode: int) -> str:
    """SHA-256 of a file. The other arguments only key the cache, so the same file is read once
    per run of candidates but a changed one is hashed again: writing to a file or replacing it
    changes its ctime or inode even if the content keeps its size and its mtime is restored.
    The kernel stamps ctime with a coarse clock (milliseconds), so a file rewritten within the
    same tick as it was hashed is not told apart; samples are assumed not to change while a
    campaign hashes them."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Json]:
    """``bytes`` and ``sha256`` of the file at ``path``; raises :class:`OSError`."""
    status = path.stat()
    return {
        "bytes": status.st_size,
        "sha256": _sha256(
            str(path), status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino
        ),
    }


def provided_provenance(path: Path) -> dict[str, Json]:
    """What is recorded for an input the caller supplied instead of running its stage."""
    provenance: dict[str, Json] = {"provided": str(path)}
    # an unusable file is the evaluator's to report; the record still names it
    with contextlib.suppress(OSError):
        provenance |= file_identity(path)
    return provenance


def _primaries(context: StageContext, given: Path | None) -> tuple[Json, Path | None]:
    """``(particle_gun, sample)`` for ddsim: a generation file, given or supplied, wins."""
    if given is not None:
        return None, given
    generation = context.stages.get("generation")
    return (None if generation is None else generation.options), None


def _check_generation(context: StageContext, provided: Collection[str]) -> None:
    stage = context.stages.get("generation")
    if stage is None:
        raise EvaluationContractError(
            "stages.generation: the problem has no generation stage to say where the particles "
            "come from"
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
        raise EvaluationContractError("stages.simulation: the problem has no simulation to run")
    if stage.events is None:
        raise EvaluationContractError(
            "stages.simulation.events: ddsim needs the events to simulate"
        )
    given = "generation" in provided or "generation" in context.files()
    particle_gun, sample = _primaries(context, Path("generation") if given else None)
    check_arguments(
        events=stage.events,
        seed=stage.seed,
        particle_gun=particle_gun,
        sample=sample,
        options=stage.options,
    )
    for key, path in _option_files(context, stage.options).items():
        if not path.is_file():
            raise EvaluationContractError(f"stages.simulation.{key}: {path} is not a file")


def _run_simulation(
    context: StageContext, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    stage = context.stages["simulation"]
    assert stage.events is not None  # checked by _check_simulation
    assert workdir is not None  # checked by check_run_inputs
    particle_gun, sample = _primaries(context, inputs.get("generation"))
    files = _option_files(context, stage.options)
    run = simulate(
        compact=inputs["geometry"],
        workdir=workdir / "simulation",
        events=stage.events,
        seed=stage.seed,
        particle_gun=particle_gun,
        sample=sample,
        options={key: str(files[key]) if key in files else v for key, v in stage.options.items()},
    )
    provenance: dict[str, Json] = {
        "argv": run.argv,
        "cwd": str(run.cwd),
        "returncode": run.returncode,
    }
    if files:
        recorded: dict[str, Json] = {}
        for key, path in files.items():
            recorded[key] = {"path": str(path)}
            with contextlib.suppress(OSError):  # a missing file is ddsim's error to report
                recorded[key] = {"path": str(path), **file_identity(path)}
        provenance["files"] = recorded
    if run.error is not None:
        return StageOutcome(artifacts=run.files, provenance=provenance, error=run.error)
    return StageOutcome({"events": run.output}, run.files, provenance)


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
    ),
}

STAGE_TYPES = tuple(_STAGES)
"""The stage types, in pipeline order."""


def stage_reads(stage: str) -> tuple[str, ...]:
    """The stage types that ``stage`` reads."""
    return _STAGES[stage].needs


def executors(stage: str) -> Mapping[str, _Executor]:
    """The executors that can run ``stage``, by name."""
    return _STAGES[stage].executors


def without_file(context: StageContext, stage: str, provided: Collection[str] = ()) -> str | None:
    """Why ``stage`` makes no file that an evaluator could read collections from, ``None`` if it
    does: it is supplied (``provided``) or given as a file, or its executor writes one. An
    executor that runs inside another stage (the gun, fired by ddsim) writes none of its own."""
    spec = context.stages.get(stage)
    if stage in provided or spec is None or spec.executor is None:
        return None  # supplied, given as a file, or not defined (reported elsewhere)
    inside = _STAGES[stage].executors[spec.executor].inside
    if inside is None:
        return None
    return (
        f"executor {spec.executor!r} runs inside stages.{inside} and writes no file of its own; "
        f"read the collections from stages.{inside}, or give stages.{stage} as a file"
    )


def changed_by_geometry(stage: str) -> bool:
    """Whether ``stage`` depends on the candidate geometry: it reads it, or reads a stage that
    does. A file given for such a stage was not made from the candidate."""
    return _STAGES[stage].geometry or any(changed_by_geometry(need) for need in stage_reads(stage))


def _graph(targets: Collection[str], given: Collection[str]) -> list[tuple[str, tuple[str, ...]]]:
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
    return [name for name, _ in _graph(targets, provided) if name not in provided]


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
    for name, _ in _graph(targets, given):
        if name in files and not files[name].is_file():
            raise EvaluationContractError(f"stages.{name}.file: {files[name]} is not a file")
    return plan


def check_run_inputs(plan: Collection[str], inputs: Collection[str], workdir: Path | None) -> None:
    """Raise if a stage of ``plan`` lacks the geometry or workdir it needs; others need none."""
    for name in plan:
        stage = _STAGES[name]
        if stage.geometry and "geometry" not in inputs:
            raise EvaluationContractError(f"the {name!r} stage needs inputs['geometry']")
        if stage.writes and workdir is None:
            raise EvaluationContractError(f"the {name!r} stage needs a workdir")


def run_stage(
    context: StageContext, stage: str, *, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    """Run one ``stage`` in ``workdir``; ``inputs`` holds the geometry and earlier outputs."""
    return _STAGES[stage].run(context, inputs, workdir)


@dataclass(frozen=True, slots=True)
class StageRecords:
    """What the pipeline stages of one candidate left behind.

    Each stage appears at most once, however many evaluators read it. ``provenance`` has an entry,
    in pipeline order, for every stage that ran, that was given as a file (its path, size and
    SHA-256 under ``file``, or under ``provided`` if the caller supplied it) or that could not
    run, plus what its output shows (``events_read``). ``outputs`` maps a stage to its named
    outputs (a particle gun has none), ``artifacts`` to its other files by role, ``errors`` to why
    it cannot be read (it failed, its output is incomplete, or it did not run because a stage it
    reads failed) and ``seconds`` to the wall time k4opt spent on it: running it and checking its
    output. ``provided`` names the stages given as files, which do not depend on the stages before
    them; ``supplied`` those of them the caller supplied rather than the problem. All of it is
    stored as read-only copies.
    """

    provenance: Mapping[str, Mapping[str, Json]] = field(default_factory=lambda: FrozenMap({}))
    outputs: Mapping[str, Mapping[str, Path]] = field(default_factory=lambda: FrozenMap({}))
    artifacts: Mapping[str, Mapping[str, Path]] = field(default_factory=lambda: FrozenMap({}))
    errors: Mapping[str, str] = field(default_factory=lambda: FrozenMap({}))
    seconds: Mapping[str, float] = field(default_factory=lambda: FrozenMap({}))
    provided: frozenset[str] = frozenset()
    supplied: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        provenance = freeze(self.provenance, "stage provenance", EvaluationContractError)
        outputs = {name: FrozenMap(files) for name, files in self.outputs.items()}
        artifacts = {name: FrozenMap(files) for name, files in self.artifacts.items()}
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "outputs", FrozenMap(outputs))
        object.__setattr__(self, "artifacts", FrozenMap(artifacts))
        object.__setattr__(self, "errors", FrozenMap(self.errors))
        object.__setattr__(self, "seconds", FrozenMap(self.seconds))
        object.__setattr__(self, "provided", frozenset(self.provided))
        object.__setattr__(self, "supplied", frozenset(self.supplied))

    def chain(self, stage: str | None) -> list[str]:
        """``stage`` and the recorded stages it needs, in pipeline order; empty for ``None``.

        A given stage needs nothing: the stages before it are not in its chain, even if another
        evaluator had them run.
        """
        if stage is None:
            return []
        return [name for name, _ in _graph([stage], self.provided) if name in self.provenance]

    def error(self, stage: str | None) -> str | None:
        """Why ``stage`` cannot be read: the first error along its chain, ``None`` if it can."""
        return next((self.errors[name] for name in self.chain(stage) if name in self.errors), None)

    def inputs(self, stage: str | None) -> dict[str, Path]:
        """The main output of ``stage`` and of every stage it needs, by stage name.

        An evaluator that reads a later stage (reconstruction, say) thus reads the earlier ones
        (simulation) too.
        """
        main = {name: _STAGES[name].output for name in self.chain(stage)}
        return {
            name: self.outputs[name][output]
            for name, output in main.items()
            if output in self.outputs.get(name, {})
        }


def run_stages(
    context: StageContext, *targets: str, inputs: Mapping[str, Path], workdir: Path | None
) -> StageRecords:
    """Make every stage of ``targets`` available, running each stage they need once.

    ``inputs`` holds the prepared geometry and the stage files the caller supplies; stages write
    below ``workdir`` (``workdir/simulation``). The stages run in dependency order; one whose
    stage it reads failed or turned out unusable does not run, and the others still do. Raises
    :class:`EvaluationContractError` before anything runs if a stage lacks the geometry or the
    workdir it needs.

    ``context`` must already have passed :func:`check_stages` for the same ``targets``, with the
    stages ``inputs`` supplies as ``provided``; the preflight of a candidate does that before any
    file is prepared. It is not checked again here: a missing file is then a failed stage, not
    a contract error.
    """
    supplied = {name: path for name, path in inputs.items() if name in _STAGES}
    files = {name: path for name, path in context.files().items() if name not in supplied}
    graph = _graph(targets, {*supplied, *files})
    check_run_inputs([name for name, _ in graph if name not in supplied | files], inputs, workdir)
    available = dict(inputs)  # the geometry and the main output of each stage so far
    provenance: dict[str, dict[str, Json]] = {}
    outputs: dict[str, dict[str, Path]] = {}
    artifacts: dict[str, dict[str, Path]] = {}
    errors: dict[str, str] = {}
    seconds: dict[str, float] = {}
    blocked: dict[str, str] = {}  # stage -> the failed stage that keeps it from running
    for name, reads in graph:
        stage = _STAGES[name]
        started = time.perf_counter()
        root = next((blocked.get(need, need) for need in reads if need in errors), None)
        if name in supplied:
            provenance[name] = provided_provenance(supplied[name])
            outputs[name] = {stage.output: supplied[name]}
        elif name in files:
            path = files[name]
            try:
                provenance[name] = {"file": str(path), **file_identity(path)}
                outputs[name] = {stage.output: path}
            except OSError as exc:
                provenance[name] = {"file": str(path)}
                errors[name] = f"cannot read {path}: {exc}"
        elif root is not None:
            provenance[name] = {}
            errors[name] = f"not run because the {root!r} stage failed"
            blocked[name] = root
            continue
        else:
            outcome = stage.run(context, available, workdir)
            outputs[name] = dict(outcome.outputs)
            artifacts[name] = dict(outcome.artifacts)
            provenance[name] = dict(outcome.provenance)
            if outcome.error is not None:
                errors[name] = outcome.error
        main = outputs.get(name, {}).get(stage.output)
        if main is not None:
            available[name] = main
        if name not in errors and main is not None and stage.inspect is not None:
            observed, error = stage.inspect(context, main)
            provenance[name] |= observed
            if error is not None:
                errors[name] = error
        seconds[name] = time.perf_counter() - started
    outputs = {name: files for name, files in outputs.items() if files}
    return StageRecords(
        provenance,
        outputs,
        artifacts,
        errors,
        seconds,
        frozenset({*supplied, *files}),
        frozenset(supplied),
    )
