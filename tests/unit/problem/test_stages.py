"""The generation and simulation stages and their use by the evaluation bridge (fake ddsim)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import DdsimError
from k4opt.ddsim import run as ddsim_run
from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
    Status,
)
from k4opt.frozen import thaw
from k4opt.problem import ProblemSpec, parse_problem
from k4opt.problem._evaluate import _evaluate_candidate, _request
from k4opt.problem._stages import (
    _sha256,
    check_stage,
    file_identity,
    plan_stages,
    run_stage,
)
from tests.support import simulation

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
    outcome = run_stage(spec, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run")

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
    outcome = run_stage(spec, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run")
    assert message in (outcome.error or "")
    assert outcome.inputs == {}
    assert {"stdout", "stderr"} <= set(outcome.artifacts)


def test_ddsim_missing_from_the_environment_is_an_error_outcome(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    monkeypatch.setattr(ddsim_run.shutil, "which", lambda name: None)
    outcome = run_stage(spec, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run")
    assert "ddsim not found" in (outcome.error or "")


def test_ddsim_that_cannot_start_is_an_error_outcome(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(ddsim_run.subprocess, "run", refuse)
    outcome = run_stage(spec, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run")
    assert "denied" in (outcome.error or "")


def test_an_existing_simulation_directory_is_never_reused(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    (tmp_path / "run" / "simulation").mkdir(parents=True)
    with pytest.raises(DdsimError, match="already exists"):
        run_stage(spec, "simulation", inputs={"geometry": geometry}, workdir=tmp_path / "run")
    assert fake.calls == []


def test_a_valid_problem_passes_the_check(spec: ProblemSpec) -> None:
    assert check_stage(spec, "simulation") == ["generation", "simulation"]


def test_an_unknown_stage_is_rejected(spec: ProblemSpec) -> None:
    with pytest.raises(EvaluationContractError, match=r"unknown stage 'reconstruction'"):
        check_stage(spec, "reconstruction")


def test_simulation_needs_generation_and_provided_inputs_are_not_rerun() -> None:
    assert plan_stages("generation") == ["generation"]
    assert plan_stages("simulation") == ["generation", "simulation"]
    assert plan_stages("simulation", provided={"generation"}) == ["simulation"]
    assert plan_stages("simulation", provided={"simulation", "geometry"}) == []


# --- generation: a particle gun or a sample file ---


def test_a_particle_gun_provides_no_file_but_is_recorded(spec: ProblemSpec, tmp_path: Path) -> None:
    outcome = run_stage(spec, "generation", inputs={}, workdir=tmp_path)
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
    outcome = run_stage(sample_spec, "generation", inputs={}, workdir=tmp_path / "run")
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
        check_stage(sample_spec, "simulation")
    assert fake.calls == []


def test_a_problem_without_generation_is_refused(doc: dict[str, Any]) -> None:
    del doc["evaluation"]["generation"]
    with pytest.raises(EvaluationContractError, match=r"evaluation\.generation: set"):
        check_stage(parse_problem(doc), "simulation")


def test_simulation_reads_the_sample_instead_of_a_gun(
    sample_spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    sample = tmp_path / "samples" / "zh.hepmc"
    outcome = run_stage(
        sample_spec,
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
        check_stage(changed(doc, path, value), "simulation")
    assert fake.calls == [] and not (tmp_path / "run").exists()


# --- the bridge runs the required stage once and hands the file to the evaluator ---


def test_the_bridge_runs_the_stage_and_gives_the_evaluator_its_file(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    reader = Reader()
    record = _evaluate_candidate(
        spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert len(fake.calls) == 1 and len(reader.calls) == 1
    events_file = tmp_path / "run/simulation/events.edm4hep.root"
    assert reader.calls[0].inputs["simulation"] == events_file
    assert record.request is reader.calls[0]
    result = record.result
    assert result.status is Status.VALID
    # the stage's files are named <stage>.<role>; the evaluator's own role stays as it chose it
    assert {
        "simulation.stdout",
        "simulation.stderr",
        "simulation.command",
        "simulation.edm4hep",
        "mine",
    } == set(result.artifacts)
    assert result.provenance["read"] == events_file.name  # the evaluator's own provenance stays
    stage = result.provenance["stages"]["simulation"]  # type: ignore[index]
    assert stage["returncode"] == 0 and stage["argv"][0] == simulation.DDSIM


def test_a_failed_stage_is_a_failed_record_and_the_evaluator_never_runs(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=3, stderr="boom")
    reader = Reader()
    record = _evaluate_candidate(
        spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert reader.calls == []
    assert record.result.status is Status.FAILED
    assert "status 3" in (record.result.message or "")
    assert record.result.measurements == ()
    assert {"simulation.stdout", "simulation.stderr"} <= set(record.result.artifacts)
    assert record.result.provenance["stages"]["simulation"]["returncode"] == 3  # type: ignore[index]


def test_an_input_supplied_by_the_caller_skips_the_stage(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    events_file = tmp_path / "mine.root"
    events_file.write_bytes(b"fake")
    reader = Reader()
    record = _evaluate_candidate(spec, {CELL: 6}, reader, inputs={"simulation": events_file})
    assert fake.calls == []
    assert reader.calls[0].inputs["simulation"] == events_file
    # not run, but identified: the file is recorded with its content hash, and generation,
    # which only the supplied simulation would have needed, does not appear
    assert record.result.provenance["stages"] == {
        "simulation": {
            "provided": str(events_file),
            "bytes": 4,
            "sha256": hashlib.sha256(b"fake").hexdigest(),
        }
    }


@pytest.mark.parametrize("missing", ["geometry", "workdir"])
def test_the_stage_needs_a_geometry_and_a_workdir(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    fake = simulation.install(monkeypatch)
    inputs = {} if missing == "geometry" else {"geometry": geometry}
    workdir = None if missing == "workdir" else tmp_path / "run"
    with pytest.raises(EvaluationContractError, match=r"geometry.*workdir"):
        _evaluate_candidate(spec, {CELL: 6}, Reader(), inputs=inputs, workdir=workdir)
    assert fake.calls == []


def test_a_sample_problem_runs_generation_then_simulation_and_records_the_hash(
    sample_spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    reader = Reader()
    record = _evaluate_candidate(
        sample_spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    sample = tmp_path / "samples" / "zh.hepmc"
    ((argv, _),) = fake.calls
    assert f"--inputFiles={sample}" in argv
    assert record.request.inputs["generation"] == sample
    assert reader.calls[0].inputs["simulation"].name == "events.edm4hep.root"
    stages = record.result.provenance["stages"]
    assert list(stages) == ["generation", "simulation"]  # type: ignore[arg-type]
    assert stages["generation"]["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()  # type: ignore[index]


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
    record = _evaluate_candidate(
        sample_spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    assert record.result.status is Status.FAILED
    assert "cannot read the sample" in (record.result.message or "")
    assert fake.calls == [] and reader.calls == []


# --- a supplied generation file is authoritative (review finding) ---


def test_a_supplied_generation_file_replaces_the_particle_gun(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    sample = tmp_path / "mine.hepmc"
    sample.write_bytes(b"HepMC::Version 3\n")
    # the problem describes a gun, yet the caller's file is what is simulated; preflight agrees
    assert check_stage(spec, "simulation", {"generation"}) == ["simulation"]
    reader = Reader()
    record = _evaluate_candidate(
        spec,
        {CELL: 6},
        reader,
        inputs={"geometry": geometry, "generation": sample},
        workdir=tmp_path / "run",
    )
    ((argv, _),) = fake.calls
    assert f"--inputFiles={sample}" in argv
    assert not any(a.startswith(("--gun", "--enableGun")) for a in argv)
    assert record.result.status is Status.VALID
    stages = record.result.provenance["stages"]
    assert list(stages) == ["generation", "simulation"]  # type: ignore[arg-type]
    assert stages["generation"]["provided"] == str(sample)  # type: ignore[index]
    assert stages["generation"]["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()  # type: ignore[index]
    assert "argv" in stages["simulation"]  # type: ignore[index, operator]


def test_a_supplied_generation_file_needs_no_generation_in_the_problem(
    doc: dict[str, Any],
) -> None:
    del doc["evaluation"]["generation"]
    assert check_stage(parse_problem(doc), "simulation", {"generation"}) == ["simulation"]


def test_the_request_carries_the_generation_context(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    reader = Reader()
    _evaluate_candidate(
        spec, {CELL: 6}, reader, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
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
    record = _evaluate_candidate(spec, {CELL: 6}, own, workdir=tmp_path)
    assert fake.calls == [] and "stages" not in record.result.provenance
    assert own.seen is not None and thaw(own.seen.generation) == GENERATOR


def test_an_evaluator_may_require_only_generation(
    sample_spec: ProblemSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)

    class NeedsSample(OwnWorkflow):
        requires = "generation"

    own = NeedsSample()
    _evaluate_candidate(
        sample_spec, {CELL: 6}, own, inputs={"geometry": tmp_path / "c.xml"}, workdir=tmp_path
    )
    assert fake.calls == []  # no simulation
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


# --- the request is the same for check() and evaluate(); roles never collide (review) ---


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
    _evaluate_candidate(
        spec, {CELL: 6}, checking, inputs={"geometry": geometry}, workdir=tmp_path / "run"
    )
    # what the preflight builds (no inputs yet) against what evaluate() was given
    preflight = _request(spec, {CELL: 6})
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


@pytest.mark.parametrize("role", ["simulation.edm4hep", "simulation.stderr"])
def test_an_evaluator_artifact_cannot_replace_a_stage_artifact(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    simulation.install(monkeypatch)
    evaluator = Colliding({role: tmp_path / "other.root"}, {})
    with pytest.raises(EvaluationContractError, match=rf"{role}.*collide"):
        _evaluate_candidate(
            spec, {CELL: 6}, evaluator, inputs={"geometry": geometry}, workdir=tmp_path / "run"
        )


def test_an_evaluator_cannot_overwrite_the_stage_provenance(
    spec: ProblemSpec, geometry: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    evaluator = Colliding({}, {"stages": {"simulation": {"returncode": 0}}})
    with pytest.raises(EvaluationContractError, match="'stages' is reserved"):
        _evaluate_candidate(
            spec, {CELL: 6}, evaluator, inputs={"geometry": geometry}, workdir=tmp_path / "run"
        )


def test_an_evaluator_without_stages_keeps_its_own_provenance_and_artifacts(
    spec: ProblemSpec, tmp_path: Path
) -> None:
    evaluator = OwnWorkflow()
    record = _evaluate_candidate(spec, {CELL: 6}, evaluator, workdir=tmp_path)
    assert "stages" not in record.result.provenance
