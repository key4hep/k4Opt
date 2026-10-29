"""The generation and simulation stages and their use by the evaluation bridge (fake ddsim)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import DdsimError
from k4opt.ddsim import run as ddsim_run
from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRecord,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    Measurement,
    Status,
    edm4hep,
)
from k4opt.frozen import thaw
from k4opt.problem import ProblemSpec, parse_problem
from k4opt.problem._evaluate import CandidateRecord, _evaluate_candidate, _request
from k4opt.problem._stages import (
    StageContext,
    _sha256,
    check_stages,
    file_identity,
    plan_stages,
    run_stage,
    run_stages,
)
from tests.support import simulation
from tests.support.simulation import events_with

CELL = "constants.ECal_cell_size"


GENERATOR = {
    "particle": "gamma",
    "energy": {"value": 10, "unit": "GeV"},
    "direction": {"distribution": "uniform", "theta": [60, 120], "phi": [0, 360], "unit": "deg"},
}


@pytest.fixture(autouse=True)
def full_particle_gun(doc: dict[str, Any]) -> None:
    """ddsim needs a complete particle gun; the shared problem document has only a stub."""
    doc["evaluation"]["generation"] = GENERATOR  # type: ignore[index]


@pytest.fixture
def spec(doc: dict[str, Any]) -> ProblemSpec:
    return parse_problem(doc)


def changed(doc: dict[str, Any], path: str, value: Any) -> ProblemSpec:
    doc = copy.deepcopy(doc)
    node = doc
    *parents, last = path.split(".")
    for key in parents:
        node = node[key]
    if value is None:
        del node[last]
    else:
        node[last] = value
    return parse_problem(doc)


def context(spec: ProblemSpec) -> StageContext:
    return StageContext.of(spec)


def evaluate(
    spec: ProblemSpec, candidate: dict[str, Any], evaluator: Any, **kwargs: Any
) -> CandidateRecord:
    """The bridge with ``evaluator`` as the problem's only evaluator."""
    return _evaluate_candidate(spec, candidate, [evaluator], **kwargs)


def evaluate_one(
    spec: ProblemSpec, candidate: dict[str, Any], evaluator: Any, **kwargs: Any
) -> EvaluationRecord:
    """The record of ``evaluator``, the problem's only evaluator."""
    (record,) = evaluate(spec, candidate, evaluator, **kwargs).records
    return record


@pytest.fixture
def geometry(tmp_path: Path) -> Path:
    path = tmp_path / "compact.xml"
    path.write_text("<lccdd/>", encoding="utf-8")
    return path


class Reader:
    """Evaluator that only reads the file of the stage it requires."""

    name = "calo.single_particle_shower"
    version = "1"
    requires = "simulation"

    def __init__(self) -> None:
        self.calls: list[EvaluationRequest] = []

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.calls.append(request)
        return EvaluationResult.valid(
            [
                Measurement("simulation", "a.rms", 1.0, unit="mm"),
                Measurement("simulation", "a.cells", 5),
            ],
            artifacts={"mine": request.inputs["simulation"]},
            provenance={"read": request.inputs["simulation"].name},
        )


def test_the_simulation_stage_runs_ddsim_from_the_problem(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    outcome = run_stage(
        context(spec), "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )

    assert outcome.error is None
    assert outcome.inputs == {"simulation": tmp_path / "run/simulation/events.edm4hep.root"}
    assert set(outcome.artifacts) == {"stdout", "stderr", "command", "edm4hep"}
    assert outcome.provenance["returncode"] == 0
    ((argv, kwargs),) = fake.calls
    assert argv[0] == simulation.DDSIM
    assert argv[-2:] == ["--physics.list", "FTFP_BERT"]  # the problem's backend arguments
    assert f"--compactFile={geometry}" in argv
    assert "--numberOfEvents=1000" in argv and "--random.seed=42" in argv
    assert "--gun.particle=gamma" in argv  # the problem's generation.particle_gun
    assert kwargs["shell"] is False and kwargs["cwd"] == tmp_path / "run" / "simulation"
    assert outcome.provenance["argv"] == tuple(argv)


@pytest.mark.parametrize(
    ("ddsim", "message"),
    [
        ({"returncode": 3, "stderr": "boom: bad geometry"}, "status 3"),
        ({"returncode": 3, "stderr": "boom: bad geometry"}, "boom"),
        ({"write_output": False}, "wrote no"),
    ],
)
def test_a_failed_ddsim_is_an_error_outcome_that_keeps_the_logs(
    spec: ProblemSpec,
    geometry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ddsim: dict[str, Any],
    message: str,
) -> None:
    simulation.install(monkeypatch, **ddsim)
    outcome = run_stage(
        context(spec), "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert message in (outcome.error or "")
    assert outcome.inputs == {}
    assert {"stdout", "stderr"} <= set(outcome.artifacts)


def test_ddsim_missing_from_the_environment_is_an_error_outcome(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    monkeypatch.setattr(ddsim_run.shutil, "which", lambda name: None)
    outcome = run_stage(
        context(spec), "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert "ddsim not found" in (outcome.error or "")


def test_ddsim_that_cannot_start_is_an_error_outcome(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(ddsim_run.subprocess, "run", refuse)
    outcome = run_stage(
        context(spec), "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert "denied" in (outcome.error or "")


def test_an_existing_simulation_directory_is_never_reused(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "run" / "simulation").mkdir(parents=True)
    with pytest.raises(DdsimError, match="already exists"):
        run_stage(
            context(spec), "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run"
        )
    assert fake.calls == []


def test_a_valid_problem_passes_the_check(spec: ProblemSpec) -> None:
    assert check_stages(context(spec), "simulation") == ["generation", "simulation"]


def test_an_unknown_stage_is_rejected(spec: ProblemSpec) -> None:
    with pytest.raises(EvaluationContractError, match=r"unknown stage 'reconstruction'"):
        check_stages(context(spec), "reconstruction")


def test_simulation_needs_generation_and_provided_inputs_are_not_rerun() -> None:
    assert plan_stages("generation") == ["generation"]
    assert plan_stages("simulation") == ["generation", "simulation"]
    assert plan_stages("simulation", provided={"generation"}) == ["simulation"]
    assert plan_stages("simulation", provided={"simulation", "geometry"}) == []


def test_several_targets_plan_each_stage_once_in_pipeline_order() -> None:
    assert plan_stages("simulation", "generation", "simulation") == ["generation", "simulation"]
    assert plan_stages("generation", "simulation") == ["generation", "simulation"]
    assert plan_stages("generation", "simulation", provided={"generation"}) == ["simulation"]
    assert plan_stages() == []


def test_the_stages_of_two_contexts_run_independently(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stages take everything from their context, so the same stage can run once per
    # context (a photon and a pion sample, say), each in its own directory.
    fake = simulation.install(monkeypatch)
    photons = context(spec)
    pions = dataclasses.replace(
        photons, events=7, seed=3, generation={**GENERATOR, "particle": "pi-"}
    )
    first = run_stages(photons, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "a")
    second = run_stages(pions, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "b")
    (photon_argv, _), (pion_argv, _) = fake.calls
    assert "--gun.particle=gamma" in photon_argv and "--numberOfEvents=1000" in photon_argv
    assert "--gun.particle=pi-" in pion_argv and "--numberOfEvents=7" in pion_argv
    assert "--random.seed=3" in pion_argv
    assert first.outputs["simulation"] == tmp_path / "a" / "simulation" / "events.edm4hep.root"
    assert second.outputs["simulation"] == tmp_path / "b" / "simulation" / "events.edm4hep.root"
    assert (
        first.provenance["simulation"]["events_read"],
        second.provenance["simulation"]["events_read"],
    ) == (1000, 7)


def test_a_failed_stage_leaves_the_stages_after_it_unrun_with_the_reason(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    # a sample that vanished after the preflight: generation fails, simulation never runs
    missing = tmp_path / "gone.hepmc"
    sample = dataclasses.replace(context(spec), generation={"sample": str(missing)})
    records = run_stages(
        sample, "simulation", inputs={"geometry": tmp_path / "c.xml"}, workdir=tmp_path / "run"
    )
    assert fake.calls == []
    assert "cannot read the sample" in records.errors["generation"]
    assert records.errors["simulation"] == "not run because the 'generation' stage failed"
    assert records.error("simulation") == records.errors["generation"]  # the first along the chain
    assert records.inputs("simulation") == {} and "simulation" not in records.seconds


def test_a_supplied_stage_does_not_inherit_the_failure_of_a_stage_before_it(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = spec.evaluation.events
    assert requested is not None
    simulation.install(monkeypatch, events_with(*[[1]] * requested))
    events_file = tmp_path / "mine.root"
    events_file.write_bytes(b"fake")
    # another evaluator requires generation, which fails; the supplied simulation never needed it
    missing = tmp_path / "gone.hepmc"
    sample = dataclasses.replace(context(spec), generation={"sample": str(missing)})
    records = run_stages(
        sample, "generation", "simulation", inputs={"simulation": events_file}, workdir=None
    )
    assert "cannot read the sample" in (records.error("generation") or "")
    assert records.chain("simulation") == ["simulation"]
    assert records.error("simulation") is None
    assert records.inputs("simulation") == {"simulation": events_file}


# --- generation: a particle gun or a sample file ---


def test_a_particle_gun_provides_no_file_but_is_recorded(spec: ProblemSpec, tmp_path: Path) -> None:
    outcome = run_stage(context(spec), "generation", inputs={}, workdir=tmp_path)
    assert outcome.error is None and outcome.inputs == {}
    assert thaw(outcome.provenance["particle_gun"]) == GENERATOR


@pytest.fixture
def sample_spec(doc: dict[str, Any], tmp_path: Path) -> ProblemSpec:
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "zh.hepmc").write_bytes(b"HepMC::Version 3\nevents\n")
    doc["evaluation"]["generation"] = {"sample": "samples/zh.hepmc"}
    # a loaded problem remembers the directory of its file; relative samples are read from there
    return dataclasses.replace(parse_problem(doc), _directory=tmp_path)


def test_a_sample_is_provided_hashed_and_resolved_from_the_problem_directory(
    sample_spec: ProblemSpec, tmp_path: Path
) -> None:
    outcome = run_stage(context(sample_spec), "generation", inputs={}, workdir=tmp_path / "run")
    path = tmp_path / "samples" / "zh.hepmc"
    assert outcome.error is None
    assert outcome.inputs == {"generation": path}
    assert outcome.provenance == {
        "sample": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_a_missing_sample_is_refused_before_anything_runs(
    sample_spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "samples" / "zh.hepmc").unlink()
    with pytest.raises(EvaluationContractError, match=r"generation\.sample.*is not a file"):
        check_stages(context(sample_spec), "simulation")
    assert fake.calls == []


def test_a_problem_without_generation_is_refused(doc: dict[str, Any]) -> None:
    del doc["evaluation"]["generation"]
    with pytest.raises(EvaluationContractError, match=r"evaluation\.generation: set"):
        check_stages(context(parse_problem(doc)), "simulation")


def test_simulation_reads_the_sample_instead_of_a_gun(
    sample_spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    sample = tmp_path / "samples" / "zh.hepmc"
    outcome = run_stage(
        context(sample_spec),
        "simulation",
        inputs={"geometry": geometry, "generation": sample},
        workdir=tmp_path / "run",
    )
    assert outcome.error is None
    ((argv, _),) = fake.calls
    assert f"--inputFiles={sample}" in argv
    assert not any(a.startswith(("--gun", "--enableGun")) for a in argv)


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        ("evaluation.events", None, EvaluationContractError),
        ("evaluation.generation", None, EvaluationContractError),  # nothing to generate
        ("evaluation.generation", {**GENERATOR, "particle": ""}, DdsimError),
        ("evaluation.backend", {"ddsim": {"args": ["--numberOfEvents", "5"]}}, DdsimError),
        ("evaluation.backend", {"k4run": {"args": ["--x"]}}, EvaluationContractError),
    ],
)
def test_misuse_raises_before_anything_is_created(
    doc: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: Any,
    error: type[Exception],
) -> None:
    fake = simulation.install(monkeypatch)
    with pytest.raises(error):
        check_stages(context(changed(doc, path, value)), "simulation")
    assert fake.calls == [] and not (tmp_path / "run").exists()


# --- the bridge runs the required stage once and hands the file to the evaluator ---


def test_the_bridge_runs_the_stage_and_gives_the_evaluator_its_file(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    reader = Reader()
    candidate = evaluate(
        spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert len(fake.calls) == 1 and len(reader.calls) == 1
    events_file = tmp_path / "run/simulation/events.edm4hep.root"
    assert reader.calls[0].inputs["simulation"] == events_file
    (record,) = candidate.records
    assert record.request is reader.calls[0]
    result = record.result
    assert result.status is Status.VALID
    # the result is the evaluator's own; the stage's files and record are the candidate's
    assert result.artifacts == {"mine": events_file}
    assert result.provenance == {"read": events_file.name}
    stages = candidate.stages
    assert set(stages.artifacts["simulation"]) == {"stdout", "stderr", "command", "edm4hep"}
    stage = stages.provenance["simulation"]
    assert stage["returncode"] == 0 and stage["argv"][0] == simulation.DDSIM  # type: ignore[index]
    assert candidate.reads == {"calo.single_particle_shower": ("generation", "simulation")}


def test_a_failed_stage_is_a_failed_record_and_the_evaluator_never_runs(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=3, stderr="boom")
    reader = Reader()
    candidate = evaluate(
        spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    (record,) = candidate.records
    assert reader.calls == []
    assert record.result.status is Status.FAILED
    assert "status 3" in (record.result.message or "")
    assert record.result.measurements == ()
    assert record.elapsed_seconds is None  # it was not called
    assert record.request.workdir is None and not (tmp_path / "run" / "evaluators").exists()
    assert {"stdout", "stderr"} <= set(candidate.stages.artifacts["simulation"])
    assert candidate.stages.provenance["simulation"]["returncode"] == 3
    assert "status 3" in candidate.stages.errors["simulation"]


def test_an_input_supplied_by_the_caller_skips_the_stage(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = spec.evaluation.events
    assert requested is not None
    fake = simulation.install(monkeypatch, events_with(*[[1]] * requested))
    events_file = tmp_path / "mine.root"
    events_file.write_bytes(b"fake")
    reader = Reader()
    candidate = evaluate(spec, {CELL: 6}, reader, inputs={"simulation": events_file})
    assert fake.calls == []
    assert reader.calls[0].inputs["simulation"] == events_file
    # not run, but identified: the file is recorded with its content hash, and generation,
    # which only the supplied simulation would have needed, does not appear
    assert candidate.stages.provenance == {
        "simulation": {
            "provided": str(events_file),
            "bytes": 4,
            "sha256": hashlib.sha256(b"fake").hexdigest(),
            "events_read": requested,
        }
    }


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize(
    ("found", "message"),
    [(2, "incomplete simulation: 2 of 1000 events"), (0, "incomplete simulation: 0 of 1000")],
)
def test_a_simulation_output_without_the_requested_events_fails_before_the_evaluator(
    spec: ProblemSpec,
    geometry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supplied: bool,
    found: int,
    message: str,
) -> None:
    simulation.install(monkeypatch, events_with(*[[1]] * found))
    reader = Reader()
    inputs = {"geometry": geometry}
    if supplied:
        inputs["simulation"] = tmp_path / "mine.root"
        inputs["simulation"].write_bytes(b"fake")
    candidate = evaluate(spec, {CELL: 6}, reader, inputs=inputs, workdir=tmp_path / "run")
    (record,) = candidate.records
    assert reader.calls == []
    assert record.result.status is Status.FAILED
    assert message in (record.result.message or "")
    assert candidate.stages.provenance["simulation"]["events_read"] == found


def test_an_unreadable_simulation_output_fails_before_the_evaluator(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> Any:
        raise RuntimeError("truncated file")

    simulation.install(monkeypatch)
    monkeypatch.setattr(edm4hep, "_frames", broken)
    reader = Reader()
    record = evaluate_one(
        spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert reader.calls == []
    assert record.result.status is Status.FAILED
    assert "cannot read events.edm4hep.root: RuntimeError: truncated file" in (
        record.result.message or ""
    )


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (EvaluationFailed("disk on fire"), Status.FAILED),
        (EvaluationInvalid("no hits"), Status.INVALID),
    ],
)
def test_an_evaluator_that_raises_an_outcome_gets_that_result(
    spec: ProblemSpec,
    geometry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: Status,
) -> None:
    simulation.install(monkeypatch)

    class Raising(Reader):
        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            raise error

    candidate = evaluate(
        spec, {CELL: 6}, Raising(), inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    (record,) = candidate.records
    assert record.result.status is status
    assert record.result.message == str(error)
    assert record.result.measurements == ()
    assert record.elapsed_seconds is not None  # it was called
    assert "simulation" in candidate.stages.provenance


def test_an_evaluator_may_return_plain_values(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)

    class Plain(Evaluator):
        name = "calo.single_particle_shower"
        version = "1"
        requires = "simulation"
        metrics = {"a.rms": "mm", "a.cells": "1"}

        def evaluate(self, request: EvaluationRequest) -> Any:
            return {"a.rms": 1.5, "a.cells": [4, 6]}

    record = evaluate_one(
        spec, {CELL: 6}, Plain(), inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert record.result.status is Status.VALID
    rms, cells = record.result.measurements
    assert (rms.stage, rms.metric, rms.value, rms.unit) == ("simulation", "a.rms", 1.5, "mm")
    assert (cells.value, cells.uncertainty) == (5, 1)
    assert record.result.artifacts == {}  # the stage's files are the candidate's, not its own


def test_the_declared_defaults_reach_check_and_evaluate(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    seen: list[Any] = []

    class Tuned(Evaluator):
        name = "calo.single_particle_shower"
        version = "1"
        requires = "simulation"
        settings = {"collection": str, "threshold": 0.5}
        metrics = {"a.rms": "mm", "a.cells": "1"}

        def check(self, request: EvaluationRequest) -> None:
            seen.append(request.settings)
            super().check(request)

        def evaluate(self, request: EvaluationRequest) -> Any:
            seen.append(request.settings)
            return {"a.rms": 1.0, "a.cells": 5}

    record = evaluate_one(
        spec, {CELL: 6}, Tuned(), inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert record.result.status is Status.VALID
    wanted = {"collection": "ECalBarrelCollection", "threshold": 0.5}
    assert seen == [wanted, wanted]
    assert record.request.settings == wanted  # the record shows the settings that were used


def test_the_bridge_runs_check_before_any_stage(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)

    class Picky(Reader):
        def check(self, request: EvaluationRequest) -> None:
            raise EvaluationContractError("unsupported settings")

    reader = Picky()
    with pytest.raises(EvaluationContractError, match="unsupported settings"):
        evaluate_one(
            spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
        )
    assert fake.calls == [] and reader.calls == []


def test_a_bad_metrics_declaration_fails_before_any_stage(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)

    class Unitless(Reader):
        metrics = {("simulation", "a.rms"): None}

    with pytest.raises(EvaluationContractError, match="unit must be"):
        evaluate_one(
            spec, {CELL: 6}, Unitless(), inputs={"geometry": geometry}, workdir=tmp_path / "run"
        )
    assert fake.calls == []


def test_other_exceptions_of_the_evaluator_still_propagate(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)

    class Buggy(Reader):
        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        evaluate_one(
            spec, {CELL: 6}, Buggy(), inputs={"geometry": geometry}, workdir=tmp_path / "run"
        )


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("geometry", r"'simulation' stage needs inputs\['geometry'\]"),
        ("workdir", r"'simulation' stage needs a workdir"),
    ],
)
def test_the_simulation_stage_needs_a_geometry_and_a_workdir(
    spec: ProblemSpec,
    geometry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    message: str,
) -> None:
    fake = simulation.install(monkeypatch)
    inputs = {} if missing == "geometry" else {"geometry": geometry}
    workdir = None if missing == "workdir" else tmp_path / "run"
    with pytest.raises(EvaluationContractError, match=message):
        evaluate_one(spec, {CELL: 6}, Reader(), inputs=inputs, workdir=workdir)
    assert fake.calls == []


def test_a_sample_problem_runs_generation_then_simulation_and_records_the_hash(
    sample_spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    reader = Reader()
    candidate = evaluate(
        sample_spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    sample = tmp_path / "samples" / "zh.hepmc"
    ((argv, _),) = fake.calls
    assert f"--inputFiles={sample}" in argv
    # an evaluator reads the files of every stage it needs, not only of the one it requires
    assert candidate.records[0].request.inputs["generation"] == sample
    assert reader.calls[0].inputs["simulation"].name == "events.edm4hep.root"
    stages = candidate.stages.provenance
    assert list(stages) == ["generation", "simulation"]
    assert stages["generation"]["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()


def test_a_failed_generation_stops_before_simulation(
    sample_spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    sample = tmp_path / "samples" / "zh.hepmc"
    real_open = Path.open

    def unreadable(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == sample:
            raise PermissionError("denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    reader = Reader()
    candidate = evaluate(
        sample_spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    (record,) = candidate.records
    assert record.result.status is Status.FAILED
    assert "cannot read the sample" in (record.result.message or "")
    assert fake.calls == [] and reader.calls == []
    # the simulation stage is recorded as not run, with the reason
    assert candidate.stages.errors["simulation"] == "not run because the 'generation' stage failed"


# --- a supplied generation file is authoritative (review finding) ---


def test_a_supplied_generation_file_replaces_the_particle_gun(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    sample = tmp_path / "mine.hepmc"
    sample.write_bytes(b"HepMC::Version 3\n")
    # the problem describes a gun, yet the caller's file is what is simulated; preflight agrees
    assert check_stages(context(spec), "simulation", provided={"generation"}) == ["simulation"]
    reader = Reader()
    candidate = evaluate(
        spec,
        {CELL: 6},
        reader,
        inputs={"geometry": geometry, "generation": sample},
        workdir=tmp_path / "run",
    )
    ((argv, _),) = fake.calls
    assert f"--inputFiles={sample}" in argv
    assert not any(a.startswith(("--gun", "--enableGun")) for a in argv)
    assert candidate.records[0].result.status is Status.VALID
    stages = candidate.stages.provenance
    assert list(stages) == ["generation", "simulation"]
    assert stages["generation"]["provided"] == str(sample)
    assert stages["generation"]["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert "argv" in stages["simulation"]


def test_a_supplied_generation_file_needs_no_generation_in_the_problem(
    doc: dict[str, Any],
) -> None:
    del doc["evaluation"]["generation"]
    plan = check_stages(context(parse_problem(doc)), "simulation", provided={"generation"})
    assert plan == ["simulation"]


def test_the_request_carries_the_generation_context(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    reader = Reader()
    evaluate_one(spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run")
    assert thaw(reader.calls[0].generation) == GENERATOR


class OwnWorkflow:
    """An evaluator with no ``requires``: it runs nothing from the framework."""

    name = "calo.single_particle_shower"
    version = "1"

    def __init__(self) -> None:
        self.seen: EvaluationRequest | None = None

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.seen = request
        return EvaluationResult.valid(
            [
                Measurement("simulation", "a.rms", 1.0, unit="mm"),
                Measurement("simulation", "a.cells", 5),
            ]
        )


def test_an_evaluator_without_requires_runs_no_stage_but_still_sees_the_generation(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    own = OwnWorkflow()
    candidate = evaluate(spec, {CELL: 6}, own, workdir=tmp_path)
    assert fake.calls == [] and candidate.stages.provenance == {}
    assert own.seen is not None and thaw(own.seen.generation) == GENERATOR


def test_an_evaluator_may_require_only_generation(
    sample_spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)

    class NeedsSample(OwnWorkflow):
        requires = "generation"

    own = NeedsSample()
    # a sample only has to be found and hashed: no geometry and no workdir are needed
    candidate = evaluate(sample_spec, {CELL: 6}, own)
    assert fake.calls == []  # no simulation
    assert list(candidate.stages.provenance) == ["generation"]
    assert own.seen is not None
    assert own.seen.inputs["generation"] == tmp_path / "samples" / "zh.hepmc"  # resolved


# --- hashing a sample once per file version, not once per candidate ---


def test_a_sample_is_hashed_once_until_it_changes(tmp_path: Path) -> None:
    sample = tmp_path / "zh.hepmc"
    sample.write_bytes(b"one")
    _sha256.cache_clear()
    first = file_identity(sample)
    for _ in range(3):
        assert file_identity(sample) == first
    assert _sha256.cache_info().misses == 1 and _sha256.cache_info().hits == 3
    sample.write_bytes(b"three")  # another size: a different file version
    changed = file_identity(sample)
    assert changed["sha256"] == hashlib.sha256(b"three").hexdigest() != first["sha256"]


# --- the request is the same for check() and evaluate(); results stay the evaluator's ---


class Checking(Reader):
    """Records what ``check()`` and ``evaluate()`` each receive."""

    def __init__(self) -> None:
        super().__init__()
        self.checked: list[EvaluationRequest] = []

    def check(self, request: EvaluationRequest) -> None:
        self.checked.append(request)


def test_check_and_evaluate_receive_the_same_request_fields(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    checking = Checking()
    evaluate_one(spec, {CELL: 6}, checking, inputs={"geometry": geometry}, workdir=tmp_path / "run")
    # what the preflight builds (no inputs yet) against what evaluate() was given
    preflight = _request(context(spec), {CELL: 6}, spec.evaluation.evaluators[0])
    (evaluated,) = checking.calls
    assert thaw(preflight.generation) == GENERATOR == thaw(evaluated.generation)
    for field in ("settings", "events", "seed", "backend_args", "candidate", "generation"):
        assert getattr(preflight, field) == getattr(evaluated, field)


class Colliding(Reader):
    def __init__(self, artifacts: dict[str, Path], provenance: dict[str, Any]) -> None:
        super().__init__()
        self.own_artifacts, self.own_provenance = artifacts, provenance

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return EvaluationResult.valid(
            [
                Measurement("simulation", "a.rms", 1.0, unit="mm"),
                Measurement("simulation", "a.cells", 5),
            ],
            artifacts=self.own_artifacts,
            provenance=self.own_provenance,
        )


def test_an_evaluator_keeps_its_own_artifacts_and_provenance_whatever_their_names(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # stages are recorded apart from the results, so no name of an evaluator can clash with them
    simulation.install(monkeypatch)
    mine = tmp_path / "other.root"
    claimed = {"stages": {"simulation": {"returncode": 0}}}
    candidate = evaluate(
        spec,
        {CELL: 6},
        Colliding({"simulation.edm4hep": mine}, claimed),
        inputs={"geometry": geometry},
        workdir=tmp_path / "run",
    )
    (record,) = candidate.records
    assert record.result.artifacts == {"simulation.edm4hep": mine}
    assert thaw(record.result.provenance) == claimed
    events_file = tmp_path / "run/simulation/events.edm4hep.root"
    assert candidate.stages.artifacts["simulation"]["edm4hep"] == events_file
    assert candidate.stages.provenance["simulation"]["argv"][0] == simulation.DDSIM  # type: ignore[index]


def test_an_evaluator_without_stages_keeps_its_own_provenance_and_artifacts(
    spec: ProblemSpec, tmp_path: Path
) -> None:
    evaluator = OwnWorkflow()
    candidate = evaluate(spec, {CELL: 6}, evaluator, workdir=tmp_path)
    assert candidate.records[0].result.provenance == {} and candidate.stages.provenance == {}


def test_a_particle_gun_generation_needs_nothing_either(spec: ProblemSpec) -> None:
    class NeedsGeneration(OwnWorkflow):
        requires = "generation"

    own = NeedsGeneration()
    candidate = evaluate(spec, {CELL: 6}, own)
    assert "generation" not in own.seen.inputs  # type: ignore[union-attr]
    gun = candidate.stages.provenance["generation"]["particle_gun"]
    assert gun["particle"] == "gamma"  # type: ignore[index]


def test_a_sample_replaced_with_the_same_size_and_mtime_is_hashed_again(tmp_path: Path) -> None:
    sample = tmp_path / "zh.hepmc"
    sample.write_bytes(b"AAAA")
    before = sample.stat()
    _sha256.cache_clear()
    first = file_identity(sample)
    time.sleep(0.05)  # ctime comes from a coarse kernel clock; see _sha256
    sample.write_bytes(b"BBBB")  # same size ...
    os.utime(sample, ns=(before.st_atime_ns, before.st_mtime_ns))  # ... and the old mtime back
    second = file_identity(sample)
    assert first["sha256"] == hashlib.sha256(b"AAAA").hexdigest()
    assert second["sha256"] == hashlib.sha256(b"BBBB").hexdigest()
