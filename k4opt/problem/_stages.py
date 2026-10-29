"""Internal: the pipeline stages evaluators can require, each run at most once per candidate.

An evaluator names the last pipeline stage it reads with ``requires`` (``"simulation"``); the
framework runs that stage and the stages it depends on and hands the evaluator the files of all
of them as ``request.inputs[stage]``. However many evaluators read a stage, it runs once. An
evaluator therefore never builds a command, starts a tool or handles that tool's failures.

* ``generation``: where the primary particles come from, ``evaluation.generation``. A
  ``sample`` file is checked and hashed and handed on as ``inputs["generation"]``; a particle
  gun has no file (``ddsim`` generates it).
* ``simulation``: ``ddsim`` on the prepared geometry; needs ``generation``. Its output, run
  or supplied, is checked to hold the requested number of events, so no evaluator has to.

The stages take their conditions (events, seed, generation, backend arguments) from a
:class:`StageContext`, never from the problem itself. A problem has one context today. A later
problem with several named contexts (one per physics sample, say) can run the same stages once
per context, each in its own directory, with one :class:`StageRecords` per context.

Pipeline stages are steps that run. The ``stage`` of a measurement, objective or constraint
only classifies a metric: an evaluator may report ``analysis`` metrics without an ``analysis``
pipeline stage, and nothing here reads metric labels.

A file the caller supplies as ``inputs[stage]`` is authoritative: that stage is not run, the
stages that only it needs are not run either, and the file replaces what the problem says (a
supplied ``generation`` file is simulated even if the problem describes a particle gun).
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

from k4opt.ddsim import check_arguments, simulate
from k4opt.evaluation import EvaluationContractError, EvaluationFailed, count_events
from k4opt.frozen import Json
from k4opt.problem.models import ProblemSpec


@dataclass(frozen=True, slots=True)
class StageContext:
    """The conditions the pipeline stages run under: one evaluation context.

    ``events``, ``seed``, ``generation`` and ``backend`` are those of the problem's
    ``evaluation``; ``directory`` is where a relative ``generation.sample`` is read from (the
    directory of the problem file, the current one if it is ``None``).
    """

    events: int | None = None
    seed: int | None = None
    generation: Mapping[str, Json] = field(default_factory=dict)
    backend: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    directory: Path | None = None

    @classmethod
    def of(cls, spec: ProblemSpec) -> StageContext:
        """The one context of ``spec``."""
        evaluation = spec.evaluation
        return cls(
            evaluation.events,
            evaluation.seed,
            evaluation.generation,
            evaluation.backend,
            spec._directory,
        )


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What a stage left behind: the inputs it provides, its files and how it ran.

    ``error`` is ``None`` only if the stage did what it should; otherwise it says what failed.
    """

    inputs: Mapping[str, Path] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    provenance: Mapping[str, Json] = field(default_factory=dict)
    error: str | None = None


def sample_path(context: StageContext) -> Path | None:
    """The absolute path of ``generation.sample``, relative ones taken from the problem file."""
    sample = context.generation.get("sample")
    if sample is None:
        return None
    assert isinstance(sample, str)
    path = Path(sample).expanduser()
    if path.is_absolute():
        return path
    return Path(os.path.abspath((context.directory or Path.cwd()) / path))


def _particle_gun(context: StageContext) -> Json:
    """``generation`` if it is a particle gun, ``None`` if it is a sample."""
    generation = context.generation
    return None if "sample" in generation or not generation else generation


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


def _primaries(context: StageContext, supplied: Path | None) -> tuple[Json, Path | None]:
    """``(particle_gun, sample)`` for ddsim; a supplied generation file wins over the problem."""
    if supplied is not None:
        return None, supplied
    return _particle_gun(context), sample_path(context)


def _check_generation(context: StageContext, provided: Collection[str]) -> None:
    if not context.generation:
        raise EvaluationContractError(
            "evaluation.generation: set a particle gun or a sample to say where the particles "
            "come from"
        )
    sample = sample_path(context)
    if sample is not None and not sample.is_file():
        raise EvaluationContractError(f"evaluation.generation.sample: {sample} is not a file")


def _run_generation(
    context: StageContext, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    sample = sample_path(context)
    if sample is None:
        gun = _particle_gun(context)
        assert isinstance(gun, Mapping)
        return StageOutcome(provenance={"particle_gun": gun})
    try:
        provenance = {"sample": str(sample), **file_identity(sample)}
    except OSError as exc:
        return StageOutcome(error=f"cannot read the sample {sample}: {exc}")
    return StageOutcome({"generation": sample}, provenance=provenance)


def _check_simulation(context: StageContext, provided: Collection[str]) -> None:
    if context.events is None:
        raise EvaluationContractError("events is required to simulate")
    unknown = [tool for tool in context.backend if tool != "ddsim"]
    if unknown:
        raise EvaluationContractError(f"backend: unsupported tool(s) {unknown}; only 'ddsim'")
    particle_gun, sample = _primaries(
        context, Path("generation") if "generation" in provided else None
    )
    check_arguments(
        events=context.events,
        seed=context.seed,
        particle_gun=particle_gun,
        sample=sample,
        args=context.backend.get("ddsim", ()),
    )


def _run_simulation(
    context: StageContext, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    assert context.events is not None  # checked by _check_simulation
    assert workdir is not None  # checked by check_run_inputs
    particle_gun, sample = _primaries(context, inputs.get("generation"))
    run = simulate(
        compact=inputs["geometry"],
        workdir=workdir / "simulation",
        events=context.events,
        seed=context.seed,
        particle_gun=particle_gun,
        sample=sample,
        args=context.backend.get("ddsim", ()),
    )
    provenance: dict[str, Json] = {
        "argv": run.argv,
        "cwd": str(run.cwd),
        "returncode": run.returncode,
    }
    if run.error is not None:
        return StageOutcome(artifacts=run.files, provenance=provenance, error=run.error)
    return StageOutcome({"simulation": run.output}, run.files, provenance)


def _inspect_simulation(context: StageContext, path: Path) -> tuple[dict[str, Json], str | None]:
    """``(observations, error)`` for the EDM4hep file of the simulation stage."""
    try:
        count = count_events(path)
    except EvaluationFailed as exc:
        return {}, str(exc)
    observed: dict[str, Json] = {"events_read": count}
    expected = context.events
    if expected is not None and count != expected:
        return observed, f"incomplete simulation: {count} of {expected} events in {path.name}"
    if count == 0:
        return observed, f"{path.name} contains no events"
    return observed, None


@dataclass(frozen=True, slots=True)
class _Stage:
    needs: tuple[str, ...]
    check: Callable[[StageContext, Collection[str]], None]
    run: Callable[[StageContext, Mapping[str, Path], Path | None], StageOutcome]
    geometry: bool  # reads the prepared geometry, inputs["geometry"]
    writes: bool  # writes files, so it needs a workdir
    # checks the output file, whether the stage wrote it or the caller supplied it
    inspect: Callable[[StageContext, Path], tuple[dict[str, Json], str | None]] | None = None


_STAGES: Mapping[str, _Stage] = {
    "generation": _Stage((), _check_generation, _run_generation, geometry=False, writes=False),
    "simulation": _Stage(
        ("generation",),
        _check_simulation,
        _run_simulation,
        geometry=True,
        writes=True,
        inspect=_inspect_simulation,
    ),
}


def plan_stages(*targets: str, provided: Collection[str] = ()) -> list[str]:
    """The stages to run, dependencies first, to make every stage of ``targets`` available.

    Each stage appears once, however many targets need it. A stage whose name is in
    ``provided`` (a caller-supplied input) is not run, and neither are the stages only it would
    need.
    """
    order: list[str] = []

    def visit(name: str) -> None:
        if name in provided or name in order:
            return
        for need in _STAGES[name].needs:
            visit(need)
        order.append(name)

    for target in targets:
        if target not in _STAGES:
            raise EvaluationContractError(
                f"unknown stage {target!r}; available: {', '.join(sorted(_STAGES))}"
            )
        visit(target)
    return order


def check_stages(context: StageContext, *targets: str, provided: Collection[str] = ()) -> list[str]:
    """Raise if a stage of ``targets`` or one they need is unknown or cannot run in
    ``context``; return the plan.

    Needs no geometry and starts nothing, so it can run before any file is prepared.
    """
    plan = plan_stages(*targets, provided=provided)
    for name in plan:
        _STAGES[name].check(context, provided)
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
    """What the pipeline stages of one candidate, in one context, left behind.

    Each stage appears at most once, however many evaluators read it. ``provenance`` has an entry,
    in pipeline order, for every stage that ran, that the caller supplied (its path, size and
    SHA-256 under ``provided``) or that could not run, plus what its output shows
    (``events_read``). ``outputs`` maps a stage to the file it provides (a particle gun provides
    none), ``artifacts`` to its files by role, ``errors`` to why it cannot be read (it failed, its
    output is incomplete, or it did not run because an earlier stage failed) and ``seconds`` to
    the wall time k4opt spent on it: running it and checking its output. ``provided`` names the
    stages the caller supplied; their files do not depend on the stages before them.
    """

    provenance: Mapping[str, Mapping[str, Json]] = field(default_factory=dict)
    outputs: Mapping[str, Path] = field(default_factory=dict)
    artifacts: Mapping[str, Mapping[str, Path]] = field(default_factory=dict)
    errors: Mapping[str, str] = field(default_factory=dict)
    seconds: Mapping[str, float] = field(default_factory=dict)
    provided: frozenset[str] = frozenset()

    def chain(self, stage: str | None) -> list[str]:
        """``stage`` and the recorded stages it needs, in pipeline order; empty for ``None``.

        A supplied stage needs nothing: the stages before it are not in its chain, even if
        another evaluator had them run.
        """
        if stage is None:
            return []
        order: list[str] = []

        def visit(name: str) -> None:
            if name in order:
                return
            if name not in self.provided:
                for need in _STAGES[name].needs:
                    visit(need)
            order.append(name)

        visit(stage)
        return [name for name in order if name in self.provenance]

    def error(self, stage: str | None) -> str | None:
        """Why ``stage`` cannot be read: the first error along its chain, ``None`` if it can."""
        return next((self.errors[name] for name in self.chain(stage) if name in self.errors), None)

    def inputs(self, stage: str | None) -> dict[str, Path]:
        """The files of ``stage`` and of every stage it needs, by stage name.

        An evaluator that requires a later stage (reconstruction, say) thus reads the earlier
        ones (simulation) too.
        """
        return {name: self.outputs[name] for name in self.chain(stage) if name in self.outputs}


def run_stages(
    context: StageContext, *targets: str, inputs: Mapping[str, Path], workdir: Path | None
) -> StageRecords:
    """Make every stage of ``targets`` available, running each stage they need once.

    ``inputs`` holds the prepared geometry and the stage files the caller supplies; stages write
    below ``workdir`` (``workdir/simulation``). The stages run in pipeline order; once one fails
    or its output turns out unusable, the stages after it do not run. Raises
    :class:`EvaluationContractError` before anything runs if a stage lacks the geometry or the
    workdir it needs.
    """
    plan = plan_stages(*targets, provided=inputs)
    check_run_inputs(plan, inputs, workdir)
    available = dict(inputs)
    provenance: dict[str, dict[str, Json]] = {}
    outputs: dict[str, Path] = {}
    artifacts: dict[str, dict[str, Path]] = {}
    errors: dict[str, str] = {}
    seconds: dict[str, float] = {}
    failed: str | None = None
    for name in plan_stages(*targets):  # every stage the targets need, supplied or not
        stage = _STAGES[name]
        started = time.perf_counter()
        if name in inputs:
            provenance[name] = provided_provenance(inputs[name])
            outputs[name] = inputs[name]
        elif name not in plan:
            continue  # only a supplied stage needs it
        elif failed is not None:
            provenance[name] = {}
            errors[name] = f"not run because the {failed!r} stage failed"
            continue
        else:
            outcome = stage.run(context, available, workdir)
            available |= outcome.inputs
            outputs |= outcome.inputs
            artifacts[name] = dict(outcome.artifacts)
            provenance[name] = dict(outcome.provenance)
            if outcome.error is not None:
                errors[name] = outcome.error
        if name not in errors and name in outputs and stage.inspect is not None:
            observed, error = stage.inspect(context, outputs[name])
            provenance[name] |= observed
            if error is not None:
                errors[name] = error
        seconds[name] = time.perf_counter() - started
        if name in errors:
            failed = name
    provided = frozenset(name for name in inputs if name in _STAGES)
    return StageRecords(provenance, outputs, artifacts, errors, seconds, provided)
