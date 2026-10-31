"""The generation and simulation stages and their use by the evaluation bridge (fake ddsim)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import re
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
from k4opt.problem import EvaluatorSpec, ProblemSpec, Sample, Stage, parse_problem
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
    make_stage,
    plan_stages,
    run_stage,
)
from k4opt.problem._tasks import (
    GEOMETRY,
    Kind,
    Task,
    TaskId,
    TaskResult,
    plan_tasks,
    run_tasks,
    stage_task,
)
from tests.support import simulation
from tests.support.evaluators import implementations
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
    doc["samples"]["gamma"]["generation"] = {"executor": "gun", **copy.deepcopy(GENERATOR)}


@pytest.fixture
def spec(doc: dict[str, Any]) -> ProblemSpec:
    return parse_problem(doc)


def context(spec: ProblemSpec) -> StageContext:
    return StageContext.of(spec, "gamma")


def with_stages(spec: ProblemSpec, **stages: Stage) -> StageContext:
    """The context of ``spec`` with some of its stages replaced."""
    return dataclasses.replace(context(spec), stages={**spec.samples["gamma"].stages, **stages})


def without_stages(doc: dict[str, Any], *, reads: str | None = None) -> ProblemSpec:
    """``doc`` whose evaluator reads ``reads`` (or nothing), with only the stages it needs."""
    doc = copy.deepcopy(doc)
    if reads is None:
        del doc["samples"], doc["evaluators"][0]["stage"]
    else:
        doc["evaluators"][0]["stage"] = reads
        doc["samples"]["gamma"] = {
            name: doc["samples"]["gamma"][name] for name in ("generation", reads)
        }
    return parse_problem(doc)


def evaluate(
    spec: ProblemSpec, candidate: dict[str, Any], evaluator: Any, **kwargs: Any
) -> CandidateRecord:
    """The bridge with ``evaluator``, a class or a watched instance, as the problem's only
    evaluator."""
    return _evaluate_candidate(spec, candidate, implementations(evaluator), **kwargs)


def evaluate_one(
    spec: ProblemSpec, candidate: dict[str, Any], evaluator: Any, **kwargs: Any
) -> EvaluationRecord:
    """The record of ``evaluator``, the problem's only evaluator."""
    (record,) = evaluate(spec, candidate, evaluator, **kwargs).records
    return record


def record_task(record: EvaluationRecord) -> TaskId:
    """The evaluation of ``record``, of the evaluator of that name on ``gamma``."""
    return TaskId(Kind.EVALUATOR, record.evaluator_name, "gamma")


def provenance(candidate: CandidateRecord) -> dict[str, Any]:
    """What each stage of the candidate recorded of how it ran, by stage, in plan order."""
    return {
        task.id.name: candidate.results[task.id].provenance
        for task in candidate.plan.of(Kind.STAGE)
    }


def stage(candidate: CandidateRecord, name: str) -> TaskResult:
    """The result of the stage ``name`` of ``gamma``, which the candidate planned."""
    result = candidate.stage(name, "gamma")
    assert result is not None
    return result


@pytest.fixture
def geometry(tmp_path: Path) -> Path:
    path = tmp_path / "compact.xml"
    path.write_text("<lccdd/>", encoding="utf-8")
    return path


class Reader:
    """Evaluator that only reads the file of the stage it reads, and notes its name in a file of
    its own, ``mine``, when it has a directory."""

    name = "calo.single_particle_shower"
    version = "1"
    stage = "simulation"

    def __init__(self) -> None:
        self.calls: list[EvaluationRequest] = []

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.calls.append(request)
        read = request.inputs["simulation"].name
        artifacts = {}
        if request.workdir is not None:
            artifacts["mine"] = request.workdir / "read.txt"
            artifacts["mine"].write_text(read)
        return EvaluationResult.valid(
            [
                Measurement("simulation", "a.rms", 1.0, unit="mm"),
                Measurement("simulation", "a.cells", 5),
            ],
            artifacts=artifacts,
            provenance={"read": read},
        )


def test_the_simulation_stage_runs_ddsim_from_the_problem(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    outcome = run_stage(
        context(spec),
        "simulation",
        inputs={"geometry": geometry},
        workdir=tmp_path / "run" / "simulation",
    )

    assert outcome.error is None
    assert outcome.outputs == {"events": tmp_path / "run/simulation/events.edm4hep.root"}
    assert set(outcome.artifacts) == {"stdout", "stderr", "command"}  # the logs, not the output
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
        context(spec),
        "simulation",
        inputs={"geometry": geometry},
        workdir=tmp_path / "run" / "simulation",
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
        context(spec),
        "simulation",
        inputs={"geometry": geometry},
        workdir=tmp_path / "run" / "simulation",
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
        context(spec),
        "simulation",
        inputs={"geometry": geometry},
        workdir=tmp_path / "run" / "simulation",
    )
    assert "denied" in (outcome.error or "")


def test_an_existing_simulation_directory_is_never_reused(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "run" / "simulation").mkdir(parents=True)
    with pytest.raises(DdsimError, match="already exists"):
        run_stage(
            context(spec),
            "simulation",
            inputs={"geometry": geometry},
            workdir=tmp_path / "run" / "simulation",
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
    first, first_events = make_stage(
        photons, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "a"
    )
    second, second_events = make_stage(
        pions, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "b"
    )
    (photon_argv, _), (pion_argv, _) = fake.calls
    assert "--gun.particle=gamma" in photon_argv and "--numberOfEvents=1000" in photon_argv
    assert "--gun.particle=pi-" in pion_argv and "--numberOfEvents=7" in pion_argv
    assert "--random.seed=3" in pion_argv
    assert first_events == tmp_path / "a/events.edm4hep.root"
    assert second.outputs == {"events": second_events} and second.error is None
    assert (first.provenance["events_read"], second.provenance["events_read"]) == (1000, 7)


def test_a_stage_whose_file_cannot_be_read_fails_with_the_reason(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    # a generation file that vanished after the preflight: the stage fails, ddsim never runs
    missing = tmp_path / "gone.hepmc"
    broken = with_stages(spec, generation=Stage("generation", file=str(missing)))
    outcome, events = make_stage(broken, "generation", inputs={}, workdir=None)
    assert fake.calls == [] and events is None
    assert f"cannot read {missing}" in (outcome.error or "")
    assert outcome.provenance == {"file": str(missing)}


def test_a_supplied_stage_reads_nothing_even_where_the_problem_gives_the_stage_before_it(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = spec.samples["gamma"].stages["simulation"].events
    assert requested is not None
    simulation.install(monkeypatch, events_with(*[[1]] * requested))
    events_file = tmp_path / "mine.root"
    events_file.write_bytes(b"fake")
    # another evaluator reads generation, a file; the supplied simulation never needs it
    stages = {**spec.samples["gamma"].stages, "generation": Stage("generation", file="gone.hepmc")}
    readers = [
        EvaluatorSpec("a", "1", stage="generation", samples=("gamma",)),
        EvaluatorSpec("b", "1", stage="simulation", samples=("gamma",)),
    ]
    plan = plan_tasks(
        {"gamma": Sample("gamma", stages)}, readers, {stage_task("simulation", "gamma")}
    )
    simulated = plan[stage_task("simulation", "gamma")]
    assert (simulated.needs, simulated.given) == ((), "supplied")
    assert plan[stage_task("generation", "gamma")] == Task(
        stage_task("generation", "gamma"), given="file"
    )
    outcome, events = make_stage(
        context(spec), "simulation", inputs={}, workdir=None, supplied=events_file
    )
    assert outcome.error is None and events == events_file
    assert outcome.provenance["provided"] == str(events_file)


def test_a_failed_stage_blocks_only_the_tasks_that_read_it(
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

    def run(task: Task, reads: Any) -> TaskResult:
        if task.id.kind is not Kind.STAGE:
            return TaskResult(task.id, "done" if task.id == GEOMETRY else "valid")
        outcome, _ = make_stage(StageContext(), task.id.name, inputs={}, workdir=None)
        return TaskResult(task.id, "failed" if outcome.error else "done", outcome.error)

    readers = [
        EvaluatorSpec("on_left", "1", stage="last", samples=("s",)),
        EvaluatorSpec("on_right", "1", stage="after_right", samples=("s",)),
    ]
    plan = plan_tasks({"s": Sample("s")}, readers)
    results = run_tasks(plan, run)
    assert ran == ["left", "right", "after_right"]
    status = {task.label: (r.status, r.message) for task, r in results.items()}
    assert status == {
        "geometry": ("done", None),
        "s/left": ("failed", "boom"),
        "s/after_left": ("blocked", "not run: s/left failed"),
        "s/last": ("blocked", "not run: s/left failed"),  # the stage that failed, not the next
        "s/right": ("done", None),
        "s/after_right": ("done", None),
        "s/on_left": ("blocked", "not run: s/left failed"),
        "s/on_right": ("valid", None),
    }
    left = TaskId(Kind.STAGE, "left", "s")
    assert {task.label for task, r in results.items() if r.blocked_by == (left,)} == {
        "s/after_left",
        "s/last",
        "s/on_left",
    }


# --- generation: a particle gun or an event file ---


def test_a_particle_gun_provides_no_file_but_is_recorded(spec: ProblemSpec, tmp_path: Path) -> None:
    outcome = run_stage(context(spec), "generation", inputs={}, workdir=tmp_path)
    assert outcome.error is None and outcome.outputs == {}
    assert thaw(outcome.provenance["particle_gun"]) == GENERATOR


@pytest.fixture
def file_spec(doc: dict[str, Any], tmp_path: Path) -> ProblemSpec:
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "zh.hepmc").write_bytes(b"HepMC::Version 3\nevents\n")
    doc["samples"]["gamma"]["generation"] = {"file": "events/zh.hepmc"}
    # a loaded problem remembers the directory of its file; relative files are read from there
    return dataclasses.replace(parse_problem(doc), _directory=tmp_path)


def test_a_generation_file_is_given_hashed_and_resolved_from_the_problem_directory(
    file_spec: ProblemSpec, tmp_path: Path
) -> None:
    outcome, events = make_stage(context(file_spec), "generation", inputs={}, workdir=None)
    path = tmp_path / "events" / "zh.hepmc"
    assert outcome.error is None and events == path
    assert outcome.outputs == {"events": path}
    assert outcome.provenance == {
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    # given by the problem, not supplied by the caller: it reads nothing
    reader = EvaluatorSpec("reader", "1", stage="generation", samples=("gamma",))
    plan = plan_tasks(file_spec.samples, [reader])
    assert plan[stage_task("generation", "gamma")].given == "file"


def test_a_missing_generation_file_is_refused_before_anything_runs(
    file_spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "events" / "zh.hepmc").unlink()
    with pytest.raises(
        EvaluationContractError, match=r"samples\.gamma\.generation\.file: .* is not a file"
    ):
        check_stages(context(file_spec), "simulation")
    assert fake.calls == []


def test_a_context_without_generation_is_refused(spec: ProblemSpec) -> None:
    alone = StageContext({"simulation": spec.samples["gamma"].stages["simulation"]}, sample="gamma")
    with pytest.raises(
        EvaluationContractError, match=r"samples\.gamma\.generation: the sample has no"
    ):
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
        workdir=tmp_path / "run" / "simulation",
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
            r"samples\.gamma\.simulation\.file: .*events\.root is not a file",
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
            r"samples\.gamma\.simulation\.physics\.pdgfile: .*no\.tbl is not a file",
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
    outcome = run_stage(
        ctx, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run" / "simulation"
    )
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
    outcome = run_stage(
        ctx, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run" / "simulation"
    )
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
        "samples.gamma.generation.file": tmp_path / "zh.hepmc",
        "samples.gamma.simulation.physics.pdgfile": tmp_path / "particles.tbl",
    }
    assert input_files(ctx, "generation") == {
        "samples.gamma.generation.file": tmp_path / "zh.hepmc"
    }
    # a supplied simulation reads neither: it does not run, and nothing else needs the event file
    assert input_files(ctx, "simulation", provided={"simulation"}) == {}
    assert input_files(ctx, "simulation", provided={"generation"}) == {
        "samples.gamma.simulation.physics.pdgfile": tmp_path / "particles.tbl"
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
    events_file = tmp_path / "run/samples/gamma/simulation/events.edm4hep.root"
    assert reader.calls[0].inputs["simulation"] == events_file
    (record,) = candidate.records
    assert record.request is reader.calls[0]
    result = record.result
    assert result.status is Status.VALID
    # the result is the evaluator's own; the stage's files and record are the candidate's
    own = tmp_path / "run/samples/gamma/evaluators/calo.single_particle_shower"
    assert result.artifacts == {"mine": own / "read.txt"}
    assert result.provenance == {"read": events_file.name}
    simulated = stage(candidate, "simulation")
    assert set(simulated.artifacts) == {"stdout", "stderr", "command"}  # its logs
    (events,) = simulated.made.values()
    assert (events.artifact.ref, events.artifact.path) == ("gamma/simulation#events", events_file)
    ran = simulated.provenance
    assert ran["returncode"] == 0 and ran["argv"][0] == simulation.DDSIM  # type: ignore[index]
    (read,) = candidate.plan.of(Kind.EVALUATOR)
    assert [need.label for need in read.needs] == [
        "geometry",
        "gamma/generation",
        "gamma/simulation",
    ]
    directory = "samples/gamma/evaluators/calo.single_particle_shower"
    assert (read.id.sample, read.directory) == ("gamma", directory)


def test_a_reader_cannot_change_the_candidate_or_its_stages(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an optimizer reads these records, so what each task and stage did must stay as recorded
    simulation.install(monkeypatch)
    candidate = evaluate(
        spec, {CELL: 6}, Reader(), inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    simulated = stage(candidate, "simulation")
    mappings: list[Any] = [
        candidate.results,
        candidate.evaluations,
        simulated.provenance,
        simulated.artifacts,
    ]
    for mapping in mappings:
        with pytest.raises(TypeError):
            mapping[next(iter(mapping), "new")] = "changed"
    assert isinstance(simulated.provenance["argv"], tuple)


def test_a_failed_stage_blocks_the_evaluator_which_never_runs(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=3, stderr="boom")
    reader = Reader()
    candidate = evaluate(
        spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    (record,) = candidate.records
    assert reader.calls == []
    assert record.result.status is Status.BLOCKED
    assert record.result.message == "not run: gamma/simulation failed"
    assert record.result.measurements == ()
    assert record.elapsed_seconds is None  # it was not called
    assert record.request.workdir is None and not (tmp_path / "run" / "evaluators").exists()
    # the stage carries the failure, with its logs
    simulated = stage(candidate, "simulation")
    assert {"stdout", "stderr"} <= set(simulated.artifacts)
    assert simulated.provenance["returncode"] == 3
    assert simulated.status == "failed" and "status 3" in (simulated.message or "")
    assert candidate.results[record_task(record)].blocked_by == (stage_task("simulation", "gamma"),)


@pytest.mark.parametrize(
    ("label", "message"),
    [
        # a misspelled stage would name a task the plan never makes: the file would be dropped
        ("gamma/simulaton", "input 'gamma/simulaton': expected SAMPLE/STAGE"),
        ("pion/simulation", "input 'pion/simulation': expected SAMPLE/STAGE"),
        ("simulation", "input 'simulation': expected SAMPLE/STAGE"),
    ],
)
def test_an_input_that_names_no_stage_of_a_sample_is_refused(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, message: str
) -> None:
    fake = simulation.install(monkeypatch)
    events_file = tmp_path / "mine.root"
    events_file.write_bytes(b"fake")
    with pytest.raises(EvaluationContractError, match=re.escape(message)):
        evaluate(spec, {CELL: 6}, Reader(), inputs={label: events_file}, workdir=tmp_path / "run")
    assert fake.calls == [] and not (tmp_path / "run").exists()


def test_an_input_of_a_stage_no_evaluator_reads_is_refused(
    spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    events_file = tmp_path / "mine.root"
    events_file.write_bytes(b"fake")
    # the simulation, supplied too, reads no generation: a generation file would go unread
    inputs = {"gamma/simulation": events_file, "gamma/generation": events_file}
    with pytest.raises(
        EvaluationContractError,
        match="input 'gamma/generation': no evaluator of the problem reads that stage of gamma",
    ):
        evaluate(spec, {CELL: 6}, Reader(), inputs=inputs, workdir=tmp_path / "run")
    assert fake.calls == [] and not (tmp_path / "run").exists()


def test_an_input_supplied_by_the_caller_skips_the_stage(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = spec.samples["gamma"].stages["simulation"].events
    assert requested is not None
    fake = simulation.install(monkeypatch, events_with(*[[1]] * requested))
    events_file = tmp_path / "mine.root"
    events_file.write_bytes(b"fake")
    reader = Reader()
    candidate = evaluate(spec, {CELL: 6}, reader, inputs={"gamma/simulation": events_file})
    assert fake.calls == []
    assert reader.calls[0].inputs["simulation"] == events_file
    # not run, but identified: the file is recorded with its content hash, and generation,
    # which only the supplied simulation would have needed, does not appear
    assert provenance(candidate) == {
        "simulation": {
            "provided": str(events_file),
            "bytes": 4,
            "sha256": hashlib.sha256(b"fake").hexdigest(),
            "events_read": requested,
        }
    }
    # a simulation the caller supplied was not made with the candidate's geometry
    assert candidate.supplied == {"gamma/simulation"}
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
        inputs={"geometry": geometry, "gamma/generation": hepmc},
        workdir=tmp_path / "run",
    )
    # the candidate's geometry was simulated; the events do not depend on it
    assert candidate.supplied == {"gamma/generation"} and candidate.represents_candidate


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
        inputs["gamma/simulation"] = tmp_path / "mine.root"
        inputs["gamma/simulation"].write_bytes(b"fake")
    candidate = evaluate(spec, {CELL: 6}, reader, inputs=inputs, workdir=tmp_path / "run")
    (record,) = candidate.records
    assert reader.calls == []
    assert record.result.status is Status.BLOCKED
    simulated = stage(candidate, "simulation")
    assert simulated.status == "failed" and message in (simulated.message or "")
    assert simulated.provenance["events_read"] == found


def test_an_unreadable_simulation_output_fails_before_the_evaluator(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> Any:
        raise RuntimeError("truncated file")

    simulation.install(monkeypatch)
    monkeypatch.setattr(edm4hep, "_frames", broken)
    reader = Reader()
    candidate = evaluate(
        spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert reader.calls == []
    assert candidate.records[0].result.status is Status.BLOCKED
    assert "cannot read events.edm4hep.root: RuntimeError: truncated file" in (
        stage(candidate, "simulation").message or ""
    )


def test_an_evaluator_never_returns_blocked(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)

    class Claiming(Reader):
        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            return EvaluationResult(Status.BLOCKED, message="not my fault")

    with pytest.raises(EvaluationContractError, match="never returns a blocked result"):
        evaluate(spec, {CELL: 6}, Claiming(), inputs={"geometry": geometry}, workdir=tmp_path)


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
    assert stage(candidate, "simulation").status == "done"


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
    stages = provenance(candidate)
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
    assert record.result.status is Status.BLOCKED
    assert record.result.message == "not run: gamma/generation failed"
    assert f"cannot read {hepmc}: denied" in (stage(candidate, "generation").message or "")
    assert fake.calls == [] and reader.calls == []
    # the simulation stage is recorded as not run, with the reason
    simulated = stage(candidate, "simulation")
    assert (simulated.status, simulated.message) == ("blocked", "not run: gamma/generation failed")
    assert simulated.elapsed_seconds is None
    assert (candidate.message or "").endswith(
        "; blocked: gamma/simulation, gamma/calo.single_particle_shower"
    )


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
        inputs={"geometry": geometry, "gamma/generation": hepmc},
        workdir=tmp_path / "run",
    )
    ((argv, _),) = fake.calls
    assert f"--inputFiles={hepmc}" in argv
    assert not any(a.startswith(("--gun", "--enableGun")) for a in argv)
    assert candidate.records[0].result.status is Status.VALID
    stages = provenance(candidate)
    assert list(stages) == ["generation", "simulation"]
    assert stages["generation"]["provided"] == str(hepmc)
    assert stages["generation"]["sha256"] == hashlib.sha256(hepmc.read_bytes()).hexdigest()
    assert "argv" in stages["simulation"]


def test_a_supplied_generation_file_needs_no_generation_in_the_problem(spec: ProblemSpec) -> None:
    alone = StageContext({"simulation": spec.samples["gamma"].stages["simulation"]})
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
    assert fake.calls == [] and provenance(candidate) == {}
    assert own.seen is not None and (own.seen.events, own.seen.inputs) == (None, {})


def test_an_evaluator_may_read_only_generation(
    doc: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "zh.hepmc").write_bytes(b"HepMC::Version 3\nevents\n")
    doc["samples"]["gamma"]["generation"] = {"file": "events/zh.hepmc"}
    file_spec = dataclasses.replace(without_stages(doc, reads="generation"), _directory=tmp_path)

    class NeedsGeneration(OwnWorkflow):
        stage = "generation"

    own = NeedsGeneration()
    # a file only has to be found and hashed: no geometry and no workdir are needed
    candidate = evaluate(file_spec, {CELL: 6}, own)
    assert fake.calls == []  # no simulation
    assert list(provenance(candidate)) == ["generation"]
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
        for path in self.own_artifacts.values():
            path.write_text("its own")
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
    own = tmp_path / "run/samples/gamma/evaluators/calo.single_particle_shower"
    mine = own / "other.root"
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
    events_file = tmp_path / "run/samples/gamma/simulation/events.edm4hep.root"
    simulated = stage(candidate, "simulation")
    assert simulated.made["events"].artifact.path == events_file
    assert simulated.provenance["argv"][0] == simulation.DDSIM  # type: ignore[index]


class Returning(Reader):
    """A reader that returns the artifacts ``write`` makes in its directory, declaring
    ``declared``."""

    write: Any = None
    artifacts: Any = None

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        result = super().evaluate(request)
        assert request.workdir is not None
        return dataclasses.replace(result, artifacts=self.write(request.workdir))


def returning(write: Any, declared: dict[str, str] | None = None) -> type[Returning]:
    return type("Returning", (Returning,), {"write": staticmethod(write), "artifacts": declared})


def written(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def outside(workdir: Path) -> dict[str, Path]:
    return {"plot": written(workdir.parent / "plot.txt")}


def linked(workdir: Path) -> dict[str, Path]:
    link = workdir / "plot.txt"
    link.symlink_to(written(workdir.parent / "elsewhere.txt"))
    return {"plot": link}


@pytest.mark.parametrize(
    ("write", "declared", "message"),
    [
        (outside, None, "artifact 'plot' (.*) is not in its own directory"),
        (linked, None, "artifact 'plot' (.*) is not in its own directory"),  # nor a link out
        (lambda w: {"plot": w}, None, "artifact 'plot' (.*) is not in its own directory"),
        (lambda w: {"plot": w / "gone.txt"}, None, "artifact 'plot' (.*) is no file it can read"),
        (lambda w: {"plot": w / "a"}, None, "no file it can read"),  # w / "a" is a directory
        (
            lambda w: {"plot": written(w / "plot.txt")},
            {"histogram": "counts"},
            "returned the artifact\\(s\\) plot, which it does not declare; it declares histogram",
        ),
        (
            lambda w: {},
            {"histogram": "counts"},
            "returned a valid result without its declared artifact\\(s\\) histogram",
        ),
    ],
)
def test_an_artifact_must_be_a_file_of_the_evaluation_and_as_declared(
    spec: ProblemSpec,
    geometry: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write: Any,
    declared: dict[str, str] | None,
    message: str,
) -> None:
    simulation.install(monkeypatch)

    def making(workdir: Path) -> dict[str, Path]:
        (workdir / "a").mkdir()
        return dict(write(workdir))

    with pytest.raises(EvaluationContractError, match=message):
        evaluate(
            spec,
            {CELL: 6},
            returning(making, declared),
            inputs={"geometry": geometry},
            workdir=tmp_path / "run",
        )


def test_an_artifact_needs_a_directory_of_the_evaluation(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = spec.samples["gamma"].stages["simulation"].events
    assert requested is not None
    simulation.install(monkeypatch, events_with(*[[1]] * requested))

    class Without(Reader):
        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            result = super().evaluate(request)
            return dataclasses.replace(result, artifacts={"plot": written(tmp_path / "p.txt")})

    events = tmp_path / "events.root"
    events.write_bytes(b"x")
    with pytest.raises(EvaluationContractError, match="has no directory of its own to write it"):
        evaluate(spec, {CELL: 6}, Without, inputs={"gamma/simulation": events})


def test_artifacts_are_checked_against_what_the_evaluator_declared_when_bound(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the record holds the declaration of the binding, so a later one cannot pass for it
    simulation.install(monkeypatch)

    def redeclaring(workdir: Path) -> dict[str, Path]:
        return {"histogram": written(workdir / "histogram.txt")}

    class Redeclaring(returning(redeclaring, {"plot": "a plot"})):  # type: ignore[misc]
        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            self.artifacts = {"histogram": "counts"}
            return super().evaluate(request)

    with pytest.raises(EvaluationContractError, match="histogram, which it does not declare"):
        evaluate(
            spec, {CELL: 6}, Redeclaring, inputs={"geometry": geometry}, workdir=tmp_path / "run"
        )


def test_units_are_checked_against_what_the_evaluator_declared_when_bound(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the record holds the declaration of the binding, so a later one cannot pass for it
    simulation.install(monkeypatch)

    class Redeclaring(Reader):
        metrics = {"a.rms": "mm", "a.cells": "1"}

        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            self.metrics = {"a.rms": "cm", "a.cells": "1"}
            return EvaluationResult.valid(
                [
                    Measurement("simulation", "a.rms", 0.1, unit="cm"),
                    Measurement("simulation", "a.cells", 5),
                ]
            )

    with pytest.raises(EvaluationContractError, match=r"has unit 'cm' but (.*) declares 'mm'"):
        evaluate(
            spec, {CELL: 6}, Redeclaring, inputs={"geometry": geometry}, workdir=tmp_path / "run"
        )


def test_a_failed_result_need_not_return_its_declared_artifacts(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)

    class Failing(Reader):
        artifacts = {"histogram": "counts"}

        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            return EvaluationResult.failed("nothing to count")

    candidate = evaluate(
        spec, {CELL: 6}, Failing, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert candidate.records[0].result.status is Status.FAILED


def test_a_file_that_changes_fails_the_tasks_that_read_it(
    doc: dict[str, Any], geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    entry = doc["evaluators"][0]
    doc["evaluators"].append(entry | {"as": "later"})
    for target in (*doc["objectives"], *doc["constraints"]):
        target["evaluator"] = entry["name"]
    calls: list[str] = []

    class Appending(Reader):
        """Changes the file it reads, the first time it is called."""

        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            result = super().evaluate(request)
            calls.append("called")
            if len(calls) == 1:
                with request.inputs["simulation"].open("ab") as handle:
                    handle.write(b"more")
            return result

    candidate = evaluate(
        parse_problem(doc), {CELL: 6}, Appending, inputs={"geometry": geometry}, workdir=tmp_path
    )
    first, later = candidate.records
    # the record could not say which bytes the first read, nor the later one what it would read
    assert first.result.status is Status.FAILED
    assert first.result.message == (
        "gamma/simulation#events changed while it ran, so the record cannot say what it read"
    )
    assert [m.metric for m in first.result.measurements] == ["a.rms", "a.cells"]  # diagnostics
    assert (later.result.status, later.result.message) == (
        Status.FAILED,
        "not run: gamma/simulation#events changed after it was made",
    )
    assert calls == ["called"] and later.elapsed_seconds is None
    assert candidate.changed() == ["gamma/simulation#events"]


def test_an_evaluator_without_stages_keeps_its_own_provenance_and_artifacts(
    doc: dict[str, Any], tmp_path: Path
) -> None:
    evaluator = OwnWorkflow()
    candidate = evaluate(without_stages(doc), {CELL: 6}, evaluator, workdir=tmp_path)
    assert candidate.records[0].result.provenance == {} and provenance(candidate) == {}


def test_a_particle_gun_generation_needs_nothing_either(doc: dict[str, Any]) -> None:
    class NeedsGeneration(OwnWorkflow):
        stage = "generation"

    own = NeedsGeneration()
    candidate = evaluate(without_stages(doc, reads="generation"), {CELL: 6}, own)
    assert "generation" not in own.seen.inputs  # type: ignore[union-attr]
    gun = stage(candidate, "generation").provenance["particle_gun"]
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
