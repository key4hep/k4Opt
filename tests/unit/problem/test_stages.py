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
from k4opt.problem import ProblemSpec, Stage, parse_problem
from k4opt.problem import _stages as stages_module
from k4opt.problem._evaluate import CandidateRecord, _evaluate_candidate, _request
from k4opt.problem._stages import (
    StageContext,
    StageOutcome,
    _sha256,
    changed_by_geometry,
    check_stages,
    file_identity,
    input_files,
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
def particle_gun(doc: dict[str, Any]) -> None:
    """The gun of these tests, whose provenance they compare."""
    doc["stages"]["generation"] = {"executor": "gun", **copy.deepcopy(GENERATOR)}


@pytest.fixture
def spec(doc: dict[str, Any]) -> ProblemSpec:
    return parse_problem(doc)


def context(spec: ProblemSpec) -> StageContext:
    return StageContext.of(spec)


def with_stages(spec: ProblemSpec, **stages: Stage) -> StageContext:
    """The context of ``spec`` with some of its stages replaced."""
    return dataclasses.replace(context(spec), stages={**spec.stages, **stages})


def without_stages(doc: dict[str, Any], *, reads: str | None = None) -> ProblemSpec:
    """``doc`` whose evaluator reads ``reads`` (or nothing), with only the stages it needs."""
    doc = copy.deepcopy(doc)
    if reads is None:
        del doc["stages"], doc["evaluators"][0]["stage"]
    else:
        doc["evaluators"][0]["stage"] = reads
        doc["stages"] = {name: doc["stages"][name] for name in ("generation", reads)}
    return parse_problem(doc)


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
    """Evaluator that only reads the file of the stage it reads."""

    name = "calo.single_particle_shower"
    version = "1"
    stage = "simulation"

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
    assert outcome.outputs == {"events": tmp_path / "run/simulation/events.edm4hep.root"}
    assert set(outcome.artifacts) == {"stdout", "stderr", "command", "edm4hep"}
    assert outcome.provenance["returncode"] == 0
    ((argv, kwargs),) = fake.calls
    assert argv[0] == simulation.DDSIM
    assert argv[-1] == "--physics.list=FTFP_BERT"  # the problem's ddsim option
    assert f"--compactFile={geometry}" in argv
    assert "--numberOfEvents=1000" in argv and "--random.seed=42" in argv
    assert "--gun.particle=gamma" in argv  # the problem's gun
    assert kwargs["cwd"] == tmp_path / "run" / "simulation"
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
    assert outcome.outputs == {}
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

    monkeypatch.setattr(ddsim_run, "_execute", refuse)
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


def test_the_stages_take_everything_from_their_context(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same stage can run for another context (a photon and a pion sample, say), each in
    # its own directory.
    fake = simulation.install(monkeypatch)
    photons = context(spec)
    pions = with_stages(
        spec,
        generation=Stage("generation", executor="gun", options={**GENERATOR, "particle": "pi-"}),
        simulation=Stage("simulation", executor="ddsim", events=7, seed=3),
    )
    first = run_stages(photons, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "a")
    second = run_stages(pions, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "b")
    (photon_argv, _), (pion_argv, _) = fake.calls
    assert "--gun.particle=gamma" in photon_argv and "--numberOfEvents=1000" in photon_argv
    assert "--gun.particle=pi-" in pion_argv and "--numberOfEvents=7" in pion_argv
    assert "--random.seed=3" in pion_argv
    assert first.inputs("simulation")["simulation"] == tmp_path / "a/simulation/events.edm4hep.root"
    assert second.outputs["simulation"] == {"events": tmp_path / "b/simulation/events.edm4hep.root"}
    assert (
        first.provenance["simulation"]["events_read"],
        second.provenance["simulation"]["events_read"],
    ) == (1000, 7)


def test_a_failed_stage_leaves_the_stages_after_it_unrun_with_the_reason(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    # a generation file that vanished after the preflight: it cannot be read, simulation never
    # runs
    missing = tmp_path / "gone.hepmc"
    broken = with_stages(spec, generation=Stage("generation", file=str(missing)))
    records = run_stages(
        broken, "simulation", inputs={"geometry": tmp_path / "c.xml"}, workdir=tmp_path / "run"
    )
    assert fake.calls == []
    assert f"cannot read {missing}" in records.errors["generation"]
    assert records.errors["simulation"] == "not run because the 'generation' stage failed"
    assert records.error("simulation") == records.errors["generation"]  # the first along the chain
    assert records.inputs("simulation") == {} and "simulation" not in records.seconds


def test_a_supplied_stage_does_not_inherit_the_failure_of_a_stage_before_it(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = spec.stages["simulation"].events
    assert requested is not None
    simulation.install(monkeypatch, events_with(*[[1]] * requested))
    events_file = tmp_path / "mine.root"
    events_file.write_bytes(b"fake")
    # another evaluator reads generation, which fails; the supplied simulation never needed it
    missing = tmp_path / "gone.hepmc"
    broken = with_stages(spec, generation=Stage("generation", file=str(missing)))
    records = run_stages(
        broken, "generation", "simulation", inputs={"simulation": events_file}, workdir=None
    )
    assert "cannot read" in (records.error("generation") or "")
    assert records.chain("simulation") == ["simulation"]
    assert records.error("simulation") is None
    assert records.inputs("simulation") == {"simulation": events_file}
    assert records.supplied == {"simulation"} and records.provided == {"generation", "simulation"}


def test_a_failed_stage_blocks_only_the_stages_that_read_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # stub stage types: two independent branches, the left one fails
    ran: list[str] = []

    def stub(name: str, needs: tuple[str, ...] = (), error: str | None = None) -> Any:
        def run(context: StageContext, inputs: Any, workdir: Any) -> StageOutcome:
            ran.append(name)
            return StageOutcome(error=error)

        return stages_module._Stage(
            needs, {}, lambda context, provided: None, run, geometry=False, writes=False
        )

    monkeypatch.setattr(
        stages_module,
        "_STAGES",
        {
            "left": stub("left", error="boom"),
            "after_left": stub("after_left", ("left",)),
            "last": stub("last", ("after_left",)),
            "right": stub("right"),
            "after_right": stub("after_right", ("right",)),
        },
    )
    records = run_stages(StageContext(), "last", "after_right", inputs={}, workdir=None)
    assert ran == ["left", "right", "after_right"]
    assert records.errors == {
        "left": "boom",
        "after_left": "not run because the 'left' stage failed",
        "last": "not run because the 'left' stage failed",  # the stage that failed, not the next
    }
    assert records.error("after_right") is None and records.error("last") == "boom"


# --- generation: a particle gun or an event file ---


def test_a_particle_gun_provides_no_file_but_is_recorded(spec: ProblemSpec, tmp_path: Path) -> None:
    outcome = run_stage(context(spec), "generation", inputs={}, workdir=tmp_path)
    assert outcome.error is None and outcome.outputs == {}
    assert thaw(outcome.provenance["particle_gun"]) == GENERATOR


@pytest.fixture
def file_spec(doc: dict[str, Any], tmp_path: Path) -> ProblemSpec:
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "zh.hepmc").write_bytes(b"HepMC::Version 3\nevents\n")
    doc["stages"]["generation"] = {"file": "events/zh.hepmc"}
    # a loaded problem remembers the directory of its file; relative files are read from there
    return dataclasses.replace(parse_problem(doc), _directory=tmp_path)


def test_a_generation_file_is_given_hashed_and_resolved_from_the_problem_directory(
    file_spec: ProblemSpec, tmp_path: Path
) -> None:
    records = run_stages(context(file_spec), "generation", inputs={}, workdir=None)
    path = tmp_path / "events" / "zh.hepmc"
    assert records.errors == {}
    assert records.outputs == {"generation": {"events": path}}
    assert records.provenance == {
        "generation": {
            "file": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    # given by the problem, not supplied by the caller
    assert records.provided == {"generation"} and records.supplied == frozenset()


def test_a_missing_generation_file_is_refused_before_anything_runs(
    file_spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "events" / "zh.hepmc").unlink()
    with pytest.raises(
        EvaluationContractError, match=r"stages\.generation\.file: .* is not a file"
    ):
        check_stages(context(file_spec), "simulation")
    assert fake.calls == []


def test_a_context_without_generation_is_refused(spec: ProblemSpec) -> None:
    alone = StageContext({"simulation": spec.stages["simulation"]})
    with pytest.raises(EvaluationContractError, match=r"stages\.generation: the problem has no"):
        check_stages(alone, "simulation")


def test_simulation_reads_the_generation_file_instead_of_a_gun(
    file_spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    hepmc = tmp_path / "events" / "zh.hepmc"
    outcome = run_stage(
        context(file_spec),
        "simulation",
        inputs={"geometry": geometry, "generation": hepmc},
        workdir=tmp_path / "run",
    )
    assert outcome.error is None
    ((argv, _),) = fake.calls
    assert f"--inputFiles={hepmc}" in argv
    assert not any(a.startswith(("--gun", "--enableGun")) for a in argv)


# stages that a parsed problem never has, but a context built directly may
@pytest.mark.parametrize(
    ("stage", "error", "match"),
    [
        (Stage("simulation", executor="ddsim"), EvaluationContractError, "needs the events"),
        (
            Stage("simulation", file="events.root"),
            EvaluationContractError,
            r"stages\.simulation\.file: .*events\.root is not a file",
        ),
        (
            Stage("generation", executor="gun", options={**GENERATOR, "particle": ""}),
            DdsimError,
            "particle",
        ),
        (
            Stage("simulation", executor="ddsim", events=5, options={"numberOfEvents": 5}),
            DdsimError,
            "set events instead",
        ),
        (
            Stage("simulation", executor="ddsim", events=5, options={"physics.pdgfile": "no.tbl"}),
            EvaluationContractError,
            r"stages\.simulation\.physics\.pdgfile: .*no\.tbl is not a file",
        ),
    ],
)
def test_misuse_raises_before_anything_is_created(
    spec: ProblemSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: Stage,
    error: type[Exception],
    match: str,
) -> None:
    fake = simulation.install(monkeypatch)
    with pytest.raises(error, match=match):
        check_stages(with_stages(spec, **{stage.type: stage}), "generation", "simulation")
    assert fake.calls == [] and not (tmp_path / "run").exists()


def test_a_file_named_in_a_ddsim_option_is_passed_absolute_and_recorded(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "particles.tbl").write_bytes(b"pdg table")
    options = {"physics.pdgfile": "particles.tbl"}
    sim = Stage("simulation", executor="ddsim", events=3, options=options)
    ctx = dataclasses.replace(with_stages(spec, simulation=sim), directory=tmp_path)
    check_stages(ctx, "simulation")
    outcome = run_stage(ctx, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run")
    ((argv, _),) = fake.calls
    assert argv[-1] == f"--physics.pdgfile={tmp_path / 'particles.tbl'}"
    assert outcome.provenance["files"] == {
        "physics.pdgfile": {
            "path": str(tmp_path / "particles.tbl"),
            "bytes": 9,
            "sha256": hashlib.sha256(b"pdg table").hexdigest(),
        }
    }


def test_a_ddsim_option_file_that_changes_during_the_run_fails_the_stage(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    table = tmp_path / "particles.tbl"
    table.write_bytes(b"pdg table")
    simulate = stages_module.simulate

    def replacing(**kwargs: Any) -> Any:
        run = simulate(**kwargs)
        table.write_bytes(b"another table")
        return run

    monkeypatch.setattr(stages_module, "simulate", replacing)
    sim = Stage("simulation", executor="ddsim", events=3, options={"physics.pdgfile": table.name})
    ctx = dataclasses.replace(with_stages(spec, simulation=sim), directory=tmp_path)
    outcome = run_stage(ctx, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run")
    assert outcome.error is not None and "physics.pdgfile" in outcome.error
    assert "changed while ddsim ran" in outcome.error
    # the record keeps the content that was there when ddsim started
    recorded = outcome.provenance["files"]["physics.pdgfile"]  # type: ignore[index]
    assert recorded["sha256"] == hashlib.sha256(b"pdg table").hexdigest()


def test_the_files_a_problem_names_are_those_its_stages_read(
    spec: ProblemSpec, tmp_path: Path
) -> None:
    options = {"physics.pdgfile": "particles.tbl"}
    sim = Stage("simulation", executor="ddsim", events=3, options=options)
    generation = Stage("generation", file="zh.hepmc")
    ctx = dataclasses.replace(
        with_stages(spec, generation=generation, simulation=sim), directory=tmp_path
    )
    assert input_files(ctx, "simulation") == {
        "stages.generation.file": tmp_path / "zh.hepmc",
        "stages.simulation.physics.pdgfile": tmp_path / "particles.tbl",
    }
    assert input_files(ctx, "generation") == {"stages.generation.file": tmp_path / "zh.hepmc"}
    # a supplied simulation reads neither: it does not run, and nothing else needs the event file
    assert input_files(ctx, "simulation", provided={"simulation"}) == {}
    assert input_files(ctx, "simulation", provided={"generation"}) == {
        "stages.simulation.physics.pdgfile": tmp_path / "particles.tbl"
    }


def test_only_the_stages_made_from_the_geometry_depend_on_it() -> None:
    assert not changed_by_geometry("generation")
    assert changed_by_geometry("simulation")


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


def test_a_reader_cannot_change_the_candidate_or_its_stages(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an optimizer reads these records, so which evaluator owns a metric, or what a stage did,
    # must stay as recorded
    simulation.install(monkeypatch)
    candidate = evaluate(
        spec, {CELL: 6}, Reader(), inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    stages = candidate.stages
    assert candidate.owners
    mappings: list[Any] = [
        candidate.owners,
        candidate.reads,
        stages.provenance,
        stages.provenance["simulation"],
        stages.outputs,
        stages.artifacts,
        stages.artifacts["simulation"],
        stages.errors,
        stages.seconds,
    ]
    for mapping in mappings:
        with pytest.raises(TypeError):
            mapping[next(iter(mapping), "new")] = "changed"
    assert isinstance(stages.provenance["simulation"]["argv"], tuple)  # type: ignore[index]


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
    requested = spec.stages["simulation"].events
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
    # a simulation the caller supplied was not made with the candidate's geometry
    assert candidate.stages.supplied == {"simulation"}
    assert not candidate.represents_candidate


def test_a_supplied_generation_still_represents_the_candidate(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    hepmc = tmp_path / "mine.hepmc"
    hepmc.write_bytes(b"HepMC::Version 3\n")
    candidate = evaluate(
        spec,
        {CELL: 6},
        Reader(),
        inputs={"geometry": geometry, "generation": hepmc},
        workdir=tmp_path / "run",
    )
    # the candidate's geometry was simulated; the events do not depend on it
    assert candidate.stages.supplied == {"generation"} and candidate.represents_candidate


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
        stage = "simulation"
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
        stage = "simulation"
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


def test_a_generation_file_problem_runs_generation_then_simulation_and_records_the_hash(
    file_spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    reader = Reader()
    candidate = evaluate(
        file_spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    hepmc = tmp_path / "events" / "zh.hepmc"
    ((argv, _),) = fake.calls
    assert f"--inputFiles={hepmc}" in argv
    # an evaluator reads the files of every stage it needs, not only of the one it reads
    assert candidate.records[0].request.inputs["generation"] == hepmc
    assert reader.calls[0].inputs["simulation"].name == "events.edm4hep.root"
    stages = candidate.stages.provenance
    assert list(stages) == ["generation", "simulation"]
    assert stages["generation"]["sha256"] == hashlib.sha256(hepmc.read_bytes()).hexdigest()


def test_a_failed_generation_stops_before_simulation(
    file_spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    hepmc = tmp_path / "events" / "zh.hepmc"
    real_open = Path.open

    def unreadable(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == hepmc:
            raise PermissionError("denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    reader = Reader()
    candidate = evaluate(
        file_spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    (record,) = candidate.records
    assert record.result.status is Status.FAILED
    assert f"cannot read {hepmc}: denied" in (record.result.message or "")
    assert fake.calls == [] and reader.calls == []
    # the simulation stage is recorded as not run, with the reason
    assert candidate.stages.errors["simulation"] == "not run because the 'generation' stage failed"


# --- a supplied generation file is authoritative (review finding) ---


def test_a_supplied_generation_file_replaces_the_particle_gun(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    hepmc = tmp_path / "mine.hepmc"
    hepmc.write_bytes(b"HepMC::Version 3\n")
    # the problem describes a gun, yet the caller's file is what is simulated; preflight agrees
    assert check_stages(context(spec), "simulation", provided={"generation"}) == ["simulation"]
    reader = Reader()
    candidate = evaluate(
        spec,
        {CELL: 6},
        reader,
        inputs={"geometry": geometry, "generation": hepmc},
        workdir=tmp_path / "run",
    )
    ((argv, _),) = fake.calls
    assert f"--inputFiles={hepmc}" in argv
    assert not any(a.startswith(("--gun", "--enableGun")) for a in argv)
    assert candidate.records[0].result.status is Status.VALID
    stages = candidate.stages.provenance
    assert list(stages) == ["generation", "simulation"]
    assert stages["generation"]["provided"] == str(hepmc)
    assert stages["generation"]["sha256"] == hashlib.sha256(hepmc.read_bytes()).hexdigest()
    assert "argv" in stages["simulation"]


def test_a_supplied_generation_file_needs_no_generation_in_the_problem(spec: ProblemSpec) -> None:
    alone = StageContext({"simulation": spec.stages["simulation"]})
    assert check_stages(alone, "simulation", provided={"generation"}) == ["simulation"]


def test_the_request_carries_the_events_and_seed_of_its_stage(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    reader = Reader()
    evaluate_one(spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run")
    assert (reader.calls[0].events, reader.calls[0].seed) == (1000, 42)


class OwnWorkflow:
    """An evaluator with no ``stage``: it runs nothing from the framework."""

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


def test_an_evaluator_without_a_stage_runs_none(
    doc: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    own = OwnWorkflow()
    candidate = evaluate(without_stages(doc), {CELL: 6}, own, workdir=tmp_path)
    assert fake.calls == [] and candidate.stages.provenance == {}
    assert own.seen is not None and (own.seen.events, own.seen.inputs) == (None, {})


def test_an_evaluator_may_read_only_generation(
    doc: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "zh.hepmc").write_bytes(b"HepMC::Version 3\nevents\n")
    doc["stages"]["generation"] = {"file": "events/zh.hepmc"}
    file_spec = dataclasses.replace(without_stages(doc, reads="generation"), _directory=tmp_path)

    class NeedsGeneration(OwnWorkflow):
        stage = "generation"

    own = NeedsGeneration()
    # a file only has to be found and hashed: no geometry and no workdir are needed
    candidate = evaluate(file_spec, {CELL: 6}, own)
    assert fake.calls == []  # no simulation
    assert list(candidate.stages.provenance) == ["generation"]
    assert own.seen is not None
    assert own.seen.inputs["generation"] == tmp_path / "events" / "zh.hepmc"  # resolved


# --- hashing an input file once per file version, not once per candidate ---


def test_a_file_is_hashed_once_until_it_changes(tmp_path: Path) -> None:
    hepmc = tmp_path / "zh.hepmc"
    hepmc.write_bytes(b"one")
    _sha256.cache_clear()
    first = file_identity(hepmc)
    for _ in range(3):
        assert file_identity(hepmc) == first
    assert _sha256.cache_info().misses == 1 and _sha256.cache_info().hits == 3
    hepmc.write_bytes(b"three")  # another size: a different file version
    changed = file_identity(hepmc)
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
    preflight = _request(context(spec), {CELL: 6}, spec.evaluators[0])
    (evaluated,) = checking.calls
    assert (preflight.events, preflight.seed) == (1000, 42)
    for field in ("settings", "events", "seed", "candidate"):
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
    doc: dict[str, Any], tmp_path: Path
) -> None:
    evaluator = OwnWorkflow()
    candidate = evaluate(without_stages(doc), {CELL: 6}, evaluator, workdir=tmp_path)
    assert candidate.records[0].result.provenance == {} and candidate.stages.provenance == {}


def test_a_particle_gun_generation_needs_nothing_either(doc: dict[str, Any]) -> None:
    class NeedsGeneration(OwnWorkflow):
        stage = "generation"

    own = NeedsGeneration()
    candidate = evaluate(without_stages(doc, reads="generation"), {CELL: 6}, own)
    assert "generation" not in own.seen.inputs  # type: ignore[union-attr]
    gun = candidate.stages.provenance["generation"]["particle_gun"]
    assert gun["particle"] == "gamma"  # type: ignore[index]


def test_a_file_replaced_with_the_same_size_and_mtime_is_hashed_again(tmp_path: Path) -> None:
    hepmc = tmp_path / "zh.hepmc"
    hepmc.write_bytes(b"AAAA")
    before = hepmc.stat()
    _sha256.cache_clear()
    first = file_identity(hepmc)
    time.sleep(0.05)  # ctime comes from a coarse kernel clock; see _sha256
    hepmc.write_bytes(b"BBBB")  # same size ...
    os.utime(hepmc, ns=(before.st_atime_ns, before.st_mtime_ns))  # ... and the old mtime back
    second = file_identity(hepmc)
    assert first["sha256"] == hashlib.sha256(b"AAAA").hexdigest()
    assert second["sha256"] == hashlib.sha256(b"BBBB").hexdigest()
