"""Internal: the pipeline stages an evaluator can require, run once per candidate.

An evaluator names the last stage it reads with ``requires`` (``"simulation"``); the framework
runs that stage and the stages it depends on from the problem and hands the evaluator the file
the stage produced as ``request.inputs[stage]``. An evaluator therefore never builds a command,
starts a tool or handles that tool's failures.

* ``generation``: where the primary particles come from, ``evaluation.generation``. A
  ``sample`` file is checked and hashed and handed on as ``inputs["generation"]``; a
  ``particle_gun`` has no file (``ddsim`` generates it).
* ``simulation``: ``ddsim`` on the prepared geometry; needs ``generation``.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from k4opt.ddsim import check_arguments, simulate
from k4opt.evaluation import EvaluationContractError
from k4opt.frozen import Json
from k4opt.problem.models import ProblemSpec


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What a stage left behind: the inputs it provides, its files and how it ran.

    ``error`` is ``None`` only if the stage did what it should; otherwise it says what failed.
    """

    inputs: Mapping[str, Path] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    provenance: Mapping[str, Json] = field(default_factory=dict)
    error: str | None = None


def sample_path(spec: ProblemSpec) -> Path | None:
    """The absolute path of ``generation.sample``, relative ones taken from the problem file."""
    sample = spec.evaluation.generation.get("sample")
    if sample is None:
        return None
    assert isinstance(sample, str)
    path = Path(sample).expanduser()
    if path.is_absolute():
        return path
    return Path(os.path.abspath((spec._directory or Path.cwd()) / path))


def _particle_gun(spec: ProblemSpec) -> Json:
    """``evaluation.generation`` if it is a particle gun, ``None`` if it is a sample."""
    generation = spec.evaluation.generation
    return None if "sample" in generation or not generation else generation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_generation(spec: ProblemSpec) -> None:
    if not spec.evaluation.generation:
        raise EvaluationContractError(
            "evaluation.generation: set a particle gun or a sample to say where the particles "
            "come from"
        )
    sample = sample_path(spec)
    if sample is not None and not sample.is_file():
        raise EvaluationContractError(f"evaluation.generation.sample: {sample} is not a file")


def _run_generation(spec: ProblemSpec, inputs: Mapping[str, Path], workdir: Path) -> StageOutcome:
    sample = sample_path(spec)
    if sample is None:
        gun = _particle_gun(spec)
        assert isinstance(gun, Mapping)
        return StageOutcome(provenance={"particle_gun": gun})
    try:
        provenance: dict[str, Json] = {
            "sample": str(sample),
            "bytes": sample.stat().st_size,
            "sha256": _sha256(sample),
        }
    except OSError as exc:
        return StageOutcome(error=f"cannot read the sample {sample}: {exc}")
    return StageOutcome({"generation": sample}, provenance=provenance)


def _check_simulation(spec: ProblemSpec) -> None:
    evaluation = spec.evaluation
    if evaluation.events is None:
        raise EvaluationContractError("events is required to simulate")
    unknown = [tool for tool in evaluation.backend if tool != "ddsim"]
    if unknown:
        raise EvaluationContractError(f"backend: unsupported tool(s) {unknown}; only 'ddsim'")
    check_arguments(
        events=evaluation.events,
        seed=evaluation.seed,
        particle_gun=_particle_gun(spec),
        sample=sample_path(spec),
        args=evaluation.backend.get("ddsim", ()),
    )


def _run_simulation(spec: ProblemSpec, inputs: Mapping[str, Path], workdir: Path) -> StageOutcome:
    evaluation = spec.evaluation
    assert evaluation.events is not None  # checked by _check_simulation
    run = simulate(
        compact=inputs["geometry"],
        workdir=workdir / "simulation",
        events=evaluation.events,
        seed=evaluation.seed,
        particle_gun=_particle_gun(spec),
        sample=inputs.get("generation"),
        args=evaluation.backend.get("ddsim", ()),
    )
    provenance: dict[str, Json] = {
        "argv": run.argv,
        "cwd": str(run.cwd),
        "returncode": run.returncode,
    }
    if run.error is not None:
        return StageOutcome(artifacts=run.files, provenance=provenance, error=run.error)
    return StageOutcome({"simulation": run.output}, run.files, provenance)


@dataclass(frozen=True, slots=True)
class _Stage:
    needs: tuple[str, ...]
    check: Callable[[ProblemSpec], None]
    run: Callable[[ProblemSpec, Mapping[str, Path], Path], StageOutcome]


_STAGES: Mapping[str, _Stage] = {
    "generation": _Stage((), _check_generation, _run_generation),
    "simulation": _Stage(("generation",), _check_simulation, _run_simulation),
}


def plan_stages(stage: str, provided: Collection[str] = ()) -> list[str]:
    """The stages to run, dependencies first, to make ``stage`` available.

    A stage whose name is in ``provided`` (a caller-supplied input) is not run, and neither
    are the stages only it would need.
    """
    if stage not in _STAGES:
        raise EvaluationContractError(
            f"unknown stage {stage!r}; available: {', '.join(sorted(_STAGES))}"
        )
    order: list[str] = []

    def visit(name: str) -> None:
        if name in provided or name in order:
            return
        for need in _STAGES[name].needs:
            visit(need)
        order.append(name)

    visit(stage)
    return order


def check_stage(spec: ProblemSpec, stage: str, provided: Collection[str] = ()) -> list[str]:
    """Raise if ``stage`` or one it needs is unknown or ``spec`` cannot run it; return the plan.

    Needs no geometry and starts nothing, so it can run before any file is prepared.
    """
    plan = plan_stages(stage, provided)
    for name in plan:
        _STAGES[name].check(spec)
    return plan


def run_stage(
    spec: ProblemSpec, stage: str, *, inputs: Mapping[str, Path], workdir: Path
) -> StageOutcome:
    """Run one ``stage`` in ``workdir``; ``inputs`` holds the geometry and earlier outputs."""
    return _STAGES[stage].run(spec, inputs, workdir)
