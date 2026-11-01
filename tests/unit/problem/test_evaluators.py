"""Several evaluators of one candidate: each pipeline stage runs once, each evaluator measures on
its own, and the candidate's status and values follow from theirs (fake ddsim, no Key4hep)."""

from __future__ import annotations

import copy
import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import run as ddsim_run
from k4opt.evaluation import (
    Collection,
    EvaluationContractError,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    Measurement,
    Role,
    Status,
)
from k4opt.problem import MetricRef, ProblemSpec, _evaluate, _work, parse_problem
from k4opt.problem._evaluate import CandidateRecord, _evaluate_candidate
from k4opt.problem._tasks import Kind, stage_task
from tests.support import simulation
from tests.support.evaluators import (
    cell_count,
    hit_energy,
    implementations,
    parameter_echo,
    toy_significance,
    yields,
)
from tests.support.simulation import COLLECTION, Event, Hits, Particle, Particles, events_with

CELL = "constants.ECal_cell_size"
GUN = {
    "particle": "gamma",
    "energy": {"value": 10, "unit": "GeV"},
    "direction": {"distribution": "uniform", "theta": [60, 120], "phi": [0, 360], "unit": "deg"},
}
CELLS = ("simulation", "ecal_barrel.mean_hit_cells")  # test.cell_count
ENERGY = ("simulation", "ecal_barrel.mean_hit_energy")  # test.hit_energy
ECHO = ("analysis", "candidate.first_value")  # test.parameter_echo
TRUTH = ("analysis", "truth.particles")  # test.truth
SIZE = ("analysis", "generation.bytes")  # test.file_size
# the evaluator that measures each metric of these tests
OWNERS = {
    CELLS: "test.cell_count",
    ENERGY: "test.hit_energy",
    ECHO: "test.parameter_echo",
    TRUTH: "test.truth",
    SIZE: "test.file_size",
    ("analysis", "a.value"): "a",
    ("analysis", "b.value"): "b",
}
COLLECTIONS = {
    name: {"hits": COLLECTION}
    for name in ("test.cell_count", "test.hit_energy", "test.busiest", "test.rival")
}


def stages(candidate: CandidateRecord) -> list[str]:
    """The stages the candidate made, in the order they were planned."""
    return [task.id.name for task in candidate.plan.of(Kind.STAGE)]


def reads(candidate: CandidateRecord) -> dict[str, tuple[str, ...]]:
    """The tasks each evaluator of the candidate reads, by evaluator."""
    return {
        task.id.name: tuple(need.label for need in task.needs)
        for task in candidate.plan.of(Kind.EVALUATOR)
    }


# the stage each evaluator of these tests reads; the others read none
STAGES = {
    "test.cell_count": "simulation",
    "test.hit_energy": "simulation",
    "test.busiest": "simulation",
    "test.rival": "simulation",
    "test.file_size": "generation",
    "test.truth": "generation",
}


def measured(candidate: CandidateRecord, metric: tuple[str, str]) -> Any:
    """The value of ``metric`` by the evaluator that measures it (:data:`OWNERS`), on ``gamma``
    if it reads a stage."""
    owner = OWNERS[metric]
    return candidate.value(MetricRef(owner, *metric, "gamma" if owner in STAGES else None))


def problem(
    doc: dict[str, Any],
    *names: str,
    objectives: tuple[tuple[str, ...], ...] = (CELLS,),
    constraints: tuple[dict[str, Any], ...] = (),
    generation_file: str | None = None,
) -> ProblemSpec:
    """The shared problem with evaluators ``names`` (version 1), each reading its stage, the
    stages they need (the gun, or ``generation_file``, and 3 simulated events) and
    these objectives. An objective ``(stage, metric)``, and a constraint without an evaluator,
    is on the metric of the evaluator that measures it (:data:`OWNERS`), named only when the
    problem has several evaluators; ``(evaluator, stage, metric)`` names another."""
    doc = copy.deepcopy(doc)
    doc["evaluators"] = [
        {"name": name, "version": "1"}
        | ({"stage": STAGES[name]} if name in STAGES else {})
        | ({"collections": COLLECTIONS[name]} if name in COLLECTIONS else {})
        for name in names
    ]
    read = {STAGES[name] for name in names if name in STAGES}
    del doc["samples"]  # one sample, gamma, if an evaluator reads a stage
    if read:
        gun = {"executor": "gun", **GUN}
        doc["samples"] = {
            "gamma": {"generation": gun if generation_file is None else {"file": generation_file}}
        }
    if "simulation" in read:
        doc["samples"]["gamma"]["simulation"] = {"executor": "ddsim", "events": 3}

    def owned(metric: tuple[str, ...]) -> dict[str, str]:
        if len(metric) == 3:
            return dict(zip(("evaluator", "stage", "metric"), metric, strict=True))
        stage, name = metric
        owner = {"evaluator": OWNERS[(stage, name)]} if len(names) > 1 else {}
        return owner | {"stage": stage, "metric": name}

    doc["objectives"] = [owned(o) | {"direction": "minimize"} for o in objectives]
    doc["constraints"] = [
        c if "evaluator" in c else owned((c["stage"], c["metric"])) | c for c in constraints
    ]
    return parse_problem(doc)


@pytest.fixture
def geometry(tmp_path: Path) -> Path:
    path = tmp_path / "compact.xml"
    path.write_text("<lccdd/>", encoding="utf-8")
    return path


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """What happened, in order: ``check``/``evaluate`` of a :class:`Spy` and each ddsim run."""
    events: list[str] = []
    simulation.install(monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9]))
    fake = ddsim_run._execute

    def logged(argv: list[str], **kwargs: Any) -> Any:
        events.append("ddsim")
        return fake(argv, **kwargs)

    monkeypatch.setattr(ddsim_run, "_execute", logged)
    return events


class Spy:
    """Passes everything on to an instance of the class ``inner`` and logs each ``check`` and
    ``evaluate``."""

    def __init__(self, inner: type[Any], log: list[str]) -> None:
        self.inner, self.log = inner(), log
        self.name, self.version = inner.name, inner.version
        self.stage, self.metrics = inner.stage, inner.metrics
        self.collections = inner.collections
        self.evaluated: list[EvaluationRequest] = []

    def check(self, request: EvaluationRequest) -> None:
        self.log.append(f"check {self.name}")
        self.inner.check(request)

    def evaluate(self, request: EvaluationRequest) -> Any:
        self.log.append(f"evaluate {self.name}")
        self.evaluated.append(request)
        return self.inner.evaluate(request)


def evaluate(spec: ProblemSpec, evaluators: list[Any], tmp_path: Path, geometry: Path) -> Any:
    return _evaluate_candidate(
        spec,
        {CELL: 6},
        implementations(*evaluators),
        inputs={"geometry": geometry},
        workdir=tmp_path / "run",
    )


# --- the stages run once, whatever number of evaluators read them ---


def test_two_simulation_readers_share_one_ddsim_run(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = problem(doc, "test.cell_count", "test.hit_energy", objectives=(CELLS, ENERGY))
    cells, energy = Spy(cell_count(), log), Spy(hit_energy(), log)
    candidate = evaluate(spec, [cells, energy], tmp_path, geometry)

    # both are checked before anything runs, ddsim runs once, then each evaluator once
    assert log == [
        "check test.cell_count",
        "check test.hit_energy",
        "ddsim",
        "evaluate test.cell_count",
        "evaluate test.hit_energy",
    ]
    events_file = tmp_path / "run" / "samples" / "gamma" / "simulation" / "events.edm4hep.root"
    assert cells.evaluated[0].inputs["simulation"] == events_file
    assert energy.evaluated[0].inputs["simulation"] == events_file
    assert candidate.status is Status.VALID
    first, second = candidate.records
    assert [(m.stage, m.metric) for m in first.result.measurements] == [CELLS]
    assert [(m.stage, m.metric) for m in second.result.measurements] == [ENERGY]
    assert measured(candidate, CELLS).value == 3.0  # distinct cells per event: 3, 2, 4
    assert measured(candidate, ENERGY).value == pytest.approx(10 / 3)  # 1 GeV per hit: 4, 2, 4
    assert [ref.label for ref in spec.references] == [
        "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
        "gamma/test.hit_energy/simulation/ecal_barrel.mean_hit_energy",
    ]
    assert stages(candidate) == ["generation", "simulation"]
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["samples"]
    assert sorted(p.name for p in (tmp_path / "run" / "samples" / "gamma").iterdir()) == [
        "evaluators",
        "simulation",
    ]


def test_evaluators_are_matched_by_name_and_kept_in_problem_order(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = problem(doc, "test.cell_count", "test.hit_energy", objectives=(CELLS, ENERGY))
    candidate = evaluate(spec, [Spy(hit_energy(), log), Spy(cell_count(), log)], tmp_path, geometry)
    names = [record.evaluator_name for record in candidate.records]
    assert names == ["test.cell_count", "test.hit_energy"]
    assert [entry for entry in log if entry.startswith("evaluate")] == [
        "evaluate test.cell_count",
        "evaluate test.hit_energy",
    ]


def test_an_evaluator_that_reads_no_stage_runs_no_simulation(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    # without a simulation, the problem needs no stages at all
    spec = problem(doc, "test.parameter_echo", objectives=(ECHO,))
    assert spec.samples == {}
    candidate = evaluate(spec, [parameter_echo()], tmp_path, geometry)
    assert log == [] and not (tmp_path / "run" / "simulation").exists()
    assert stages(candidate) == [] and reads(candidate) == {"test.parameter_echo": ("geometry",)}
    assert candidate.status is Status.VALID and measured(candidate, ECHO).value == 6


class FileSize(Evaluator):
    """Reads only the generation stage: the size of the generation file."""

    name = "test.file_size"
    version = "1"
    stage = "generation"
    metrics = {("analysis", "generation.bytes"): "B"}

    def __init__(self) -> None:
        self.seen: list[EvaluationRequest] = []

    def evaluate(self, request: EvaluationRequest) -> Any:
        self.seen.append(request)
        return {("analysis", "generation.bytes"): request.inputs["generation"].stat().st_size}


@pytest.fixture
def hepmc_doc(doc: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "zh.hepmc").write_bytes(b"HepMC::Version 3\n")
    return doc


def in_directory(spec: ProblemSpec, tmp_path: Path) -> ProblemSpec:
    return dataclasses.replace(spec, _directory=tmp_path)


def test_each_evaluator_reads_the_files_of_the_stages_it_needs_and_no_others(
    hepmc_doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = problem(
        hepmc_doc,
        "test.cell_count",
        "test.file_size",
        objectives=(CELLS, SIZE),
        generation_file="zh.hepmc",
    )
    cells, size = Spy(cell_count(), log), FileSize()
    candidate = evaluate(in_directory(spec, tmp_path), [cells, size], tmp_path, geometry)
    assert log.count("ddsim") == 1
    # simulation needs generation, so its reader gets both files; the generation reader only its own
    assert set(cells.evaluated[0].inputs) == {"geometry", "generation", "simulation"}
    assert set(size.seen[0].inputs) == {"geometry", "generation"}
    assert reads(candidate) == {
        "test.cell_count": ("geometry", "gamma/generation", "gamma/simulation"),
        "test.file_size": ("geometry", "gamma/generation"),
    }


def test_an_evaluator_is_given_the_same_request_alone_and_among_others(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    alone = Spy(cell_count(), log)
    evaluate(problem(doc, "test.cell_count"), [alone], tmp_path / "alone", geometry)
    among = Spy(cell_count(), log)
    others = [among, parameter_echo(), hit_energy()]
    spec = problem(
        doc, "test.cell_count", "test.parameter_echo", "test.hit_energy", objectives=(CELLS,)
    )
    evaluate(spec, others, tmp_path / "among", geometry)
    first, second = alone.evaluated[0], among.evaluated[0]
    for field in ("candidate", "settings", "collections", "events", "seed"):
        assert getattr(first, field) == getattr(second, field)
    assert set(first.inputs) == set(second.inputs)


def test_supplied_stage_files_only_reach_the_evaluators_that_read_them(
    hepmc_doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = problem(
        hepmc_doc,
        "test.cell_count",
        "test.file_size",
        "test.parameter_echo",
        objectives=(CELLS, SIZE, ECHO),
        generation_file="zh.hepmc",
    )
    events_file = tmp_path / "provided.root"
    events_file.write_bytes(b"fake")
    cells, size, echo = Spy(cell_count(), log), FileSize(), Spy(parameter_echo(), log)
    result = _evaluate_candidate(
        in_directory(spec, tmp_path),
        {CELL: 6},
        implementations(cells, size, echo),
        inputs={"geometry": geometry, "gamma/simulation": events_file},
        workdir=tmp_path / "run",
    )
    assert result.status is Status.VALID and "ddsim" not in log
    assert set(cells.evaluated[0].inputs) == {"geometry", "simulation"}
    assert set(size.seen[0].inputs) == {"geometry", "generation"}
    assert set(echo.evaluated[0].inputs) == {"geometry"}
    assert reads(result)[echo.name] == ("geometry",)


# --- a failure only concerns the evaluators that read the failed stage ---


def test_an_independent_evaluator_keeps_its_result_when_the_simulation_fails(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, returncode=2, stderr="G4Exception: bad volume")
    spec = problem(doc, "test.cell_count", "test.parameter_echo", objectives=(CELLS, ECHO))
    candidate = evaluate(spec, [cell_count(), parameter_echo()], tmp_path, geometry)
    assert len(fake.calls) == 1
    cells, echo = candidate.records
    # the stage carries the failure; its reader is blocked, never called
    assert cells.result.status is Status.BLOCKED and cells.elapsed_seconds is None
    assert cells.result.message == "not run: gamma/simulation failed"
    simulated = candidate.stage("simulation", "gamma")
    assert simulated is not None and simulated.status == "failed"
    assert "ddsim exited with status 2" in (simulated.message or "")
    assert echo.result.status is Status.VALID and echo.elapsed_seconds is not None
    # the evaluation is incomplete, but the measurement that was made is kept
    assert candidate.status is Status.FAILED
    assert measured(candidate, CELLS) is None
    assert measured(candidate, ECHO).value == 6
    message = candidate.message or ""
    assert message.startswith("gamma/simulation failed: ddsim exited")
    assert message.endswith("; blocked: gamma/test.cell_count")
    assert not (tmp_path / "run" / "samples" / "gamma" / "evaluators" / "test.cell_count").exists()


def test_a_generation_reader_keeps_its_result_when_the_simulation_fails(
    hepmc_doc: dict[str, Any], tmp_path: Path, geometry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2)
    spec = problem(
        hepmc_doc,
        "test.cell_count",
        "test.file_size",
        objectives=(CELLS, SIZE),
        generation_file="zh.hepmc",
    )
    size = FileSize()
    candidate = evaluate(in_directory(spec, tmp_path), [cell_count(), size], tmp_path, geometry)
    cells, sized = candidate.records
    assert cells.result.status is Status.BLOCKED
    assert sized.result.status is Status.VALID and set(size.seen[0].inputs) == {
        "geometry",
        "generation",
    }


@pytest.mark.parametrize("readable", [True, False])
def test_a_supplied_simulation_does_not_depend_on_the_generation_another_evaluator_runs(
    hepmc_doc: dict[str, Any],
    tmp_path: Path,
    log: list[str],
    monkeypatch: pytest.MonkeyPatch,
    readable: bool,
) -> None:
    # the caller's simulation file bypasses generation, which runs only for the generation reader:
    # its file never reaches the simulation reader, and its failure does not fail that reader
    spec = problem(
        hepmc_doc,
        "test.cell_count",
        "test.file_size",
        objectives=(CELLS, SIZE),
        generation_file="zh.hepmc",
    )
    hepmc, events_file = tmp_path / "zh.hepmc", tmp_path / "mine.edm4hep.root"
    events_file.write_bytes(b"fake")
    if not readable:
        real_open = Path.open

        def unreadable(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self == hepmc:
                raise PermissionError("denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", unreadable)
    cells = Spy(cell_count(), log)
    candidate = _evaluate_candidate(
        in_directory(spec, tmp_path),
        {CELL: 6},
        implementations(cells, FileSize()),
        inputs={"gamma/simulation": events_file},
    )
    assert "ddsim" not in log
    assert set(cells.evaluated[0].inputs) == {"simulation"}
    # it reads the geometry task, which has no file to give without a geometry
    assert reads(candidate)["test.cell_count"] == ("geometry", "gamma/simulation")
    counted, sized = candidate.records
    assert counted.result.status is Status.VALID
    assert measured(candidate, CELLS).value == 3.0
    if readable:
        assert sized.result.status is Status.VALID and candidate.status is Status.VALID
    else:
        assert sized.result.status is Status.BLOCKED
        assert sized.result.message == "not run: gamma/generation failed"
        assert f"gamma/generation failed: cannot read {hepmc}: denied" in (candidate.message or "")


# --- the candidate's status: incomplete is never reported as scientifically invalid ---


def toy(name: str, status: str) -> Any:
    """An evaluator of ``analysis/<name>.value`` that ends with ``status``."""

    class Toy:
        version = "1"
        metrics = {("analysis", f"{name}.value"): "1"}

        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            measured = [Measurement("analysis", f"{name}.value", 1.0, unit="1")]
            if status == "valid":
                return EvaluationResult.valid(measured)
            if status == "invalid":
                return EvaluationResult.invalid("unusable", measured)  # keeps a diagnostic
            return EvaluationResult.failed("crashed")

    evaluator = Toy()
    evaluator.name = name  # type: ignore[attr-defined]
    return evaluator


@pytest.mark.parametrize(
    ("first", "second", "status", "message"),
    [
        ("valid", "valid", "valid", None),
        ("valid", "invalid", "invalid", "b invalid: unusable"),
        ("invalid", "invalid", "invalid", "a invalid: unusable; b invalid: unusable"),
        ("valid", "failed", "failed", "b failed: crashed"),
        ("invalid", "failed", "failed", "a invalid: unusable; b failed: crashed"),
        ("failed", "invalid", "failed", "a failed: crashed; b invalid: unusable"),
    ],
)
def test_the_candidate_status_follows_its_evaluators(
    doc: dict[str, Any], first: str, second: str, status: str, message: str | None
) -> None:
    spec = problem(doc, "a", "b", objectives=(("analysis", "a.value"), ("analysis", "b.value")))
    candidate = _evaluate_candidate(
        spec, {CELL: 6}, implementations(toy("a", first), toy("b", second))
    )
    assert candidate.status == status
    assert candidate.message == message
    for name, own in (("a", first), ("b", second)):
        record = candidate.record(name)
        assert record.result.status == own
        # a valid result gives the objective; an invalid one keeps its diagnostic in its record
        ref = MetricRef(name, "analysis", f"{name}.value")
        assert (candidate.value(ref) is not None) == (own == "valid")
        assert len(record.result.measurements) == (0 if own == "failed" else 1)


# --- everything that can be checked is checked before anything runs ---


class Ahead(Evaluator):
    """Requires a stage that does not exist yet."""

    name = "test.ahead"
    version = "1"
    stage = "reconstruction"
    metrics = {("reconstruction", "pfo.count"): "1"}


@pytest.mark.parametrize(
    ("declared", "given", "options", "message"),
    [
        (
            ("test.cell_count", "test.hit_energy"),
            (cell_count,),
            {},
            r"no evaluator given for 'test\.hit_energy' version '1'",
        ),
        (
            ("test.cell_count",),
            (cell_count, hit_energy),
            {},
            r"problem declares no evaluator 'test\.hit_energy'; declared: test\.cell_count",
        ),
        (("test.cell_count",), (cell_count, cell_count), {}, r"'test\.cell_count' is given twice"),
        (
            ("test.cell_count", "test.hit_energy"),
            (cell_count, hit_energy),
            {"objectives": (CELLS, ("test.hit_energy", "simulation", "nothing"))},
            r"evaluator 'test\.hit_energy' does not measure simulation/nothing; supported: "
            r"simulation/ecal_barrel\.mean_hit_energy$",
        ),
        (
            ("test.cell_count", "test.hit_energy"),
            (cell_count, hit_energy),
            {"objectives": (("test.hit_energy", *CELLS),)},  # the metric of the other one
            r"evaluator 'test\.hit_energy' does not measure simulation/ecal_barrel\.mean_hit_cells",
        ),
        (
            ("test.cell_count", "test.hit_energy"),
            (cell_count, hit_energy),
            {"constraints": ({"stage": ENERGY[0], "metric": ENERGY[1], "max": 1, "unit": "MeV"},)},
            r"constraint on gamma/test\.hit_energy/simulation/ecal_barrel\.mean_hit_energy "
            r"declares unit 'MeV' but evaluator 'test\.hit_energy' measures it in 'GeV'",
        ),
        (
            ("test.cell_count", "test.ahead"),
            (cell_count, Ahead),
            {},
            r"evaluator 'test\.ahead' does not read the stage the problem says: add stage: "
            r"reconstruction to its entry",
        ),
    ],
)
def test_evaluators_that_do_not_fit_the_problem_are_refused_before_anything_runs(
    doc: dict[str, Any],
    tmp_path: Path,
    geometry: Path,
    log: list[str],
    declared: tuple[str, ...],
    given: tuple[Any, ...],
    options: dict[str, Any],
    message: str,
) -> None:
    spec = problem(doc, *declared, **options)
    with pytest.raises(EvaluationContractError, match=message):
        evaluate(spec, [make() for make in given], tmp_path, geometry)
    assert log == [] and not (tmp_path / "run").exists()


def test_another_version_is_refused(doc: dict[str, Any], tmp_path: Path, log: list[str]) -> None:
    spec = problem(doc, "test.cell_count", "test.hit_energy")
    entries = (spec.evaluators[0], dataclasses.replace(spec.evaluators[1], version="2"))
    spec = dataclasses.replace(spec, evaluators=entries)
    with pytest.raises(
        EvaluationContractError, match=r"'test\.hit_energy' version '2', got version '1'"
    ):
        _evaluate_candidate(spec, {CELL: 6}, implementations(cell_count(), hit_energy()))
    assert log == []


@pytest.mark.parametrize(
    ("name", "stage", "message"),
    [
        ("test.cell_count", None, r"add stage: simulation to its entry in the problem"),
        ("test.cell_count", "generation", r"it reads 'simulation', not 'generation'"),
        ("test.parameter_echo", "simulation", r"it reads no stage; remove stage: simulation"),
    ],
)
def test_the_stage_in_the_problem_must_be_the_one_the_evaluator_reads(
    doc: dict[str, Any],
    tmp_path: Path,
    geometry: Path,
    log: list[str],
    name: str,
    stage: str | None,
    message: str,
) -> None:
    spec = problem(doc, name, objectives=(CELLS if name in STAGES else ECHO,))
    spec = dataclasses.replace(
        spec, evaluators=(dataclasses.replace(spec.evaluators[0], stage=stage),)
    )
    given = cell_count() if name in STAGES else parameter_echo()
    with pytest.raises(
        EvaluationContractError, match=rf"evaluator '{name}' does not read .*{message}"
    ):
        evaluate(spec, [given], tmp_path, geometry)
    assert log == [] and not (tmp_path / "run").exists()


def test_invalid_collections_of_the_second_evaluator_are_refused_before_anything_runs(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = problem(doc, "test.cell_count", "test.hit_energy")
    entries = (
        spec.evaluators[0],
        dataclasses.replace(spec.evaluators[1], collections={"hit": COLLECTION}),
    )
    spec = dataclasses.replace(spec, evaluators=entries)
    with pytest.raises(EvaluationContractError, match=r"collections: .*unknown \['hit'\]"):
        evaluate(spec, [cell_count(), hit_energy()], tmp_path, geometry)
    assert log == [] and not (tmp_path / "run").exists()


class Rival(Evaluator):
    """Also measures the metric of ``test.cell_count``, by name: 99 cells."""

    name = "test.rival"
    version = "1"
    stage = "simulation"
    collections = {"hits": Collection("SimCalorimeterHit")}
    metrics = {"ecal_barrel.mean_hit_cells": "1"}

    def evaluate(self, request: EvaluationRequest) -> Any:
        return {"ecal_barrel.mean_hit_cells": 99}


class Silent(Evaluator):
    """Declares no metrics, and reports one of another evaluator's, by name."""

    name = "test.silent"
    version = "1"

    def evaluate(self, request: EvaluationRequest) -> Any:
        return {CELLS: 7}


@pytest.mark.parametrize(("other", "value"), [(Rival, 99), (Silent, 7)])
def test_a_metric_two_evaluators_report_is_two_values_each_of_its_own_evaluator(
    doc: dict[str, Any],
    tmp_path: Path,
    geometry: Path,
    log: list[str],
    other: type[Any],
    value: int,
) -> None:
    # an objective names the evaluator of its value: a metric of the same name is no rival
    spec = problem(
        doc,
        "test.cell_count",
        other.name,
        objectives=(("test.cell_count", *CELLS), (other.name, *CELLS)),
    )
    candidate = evaluate(spec, [cell_count(), other], tmp_path, geometry)
    assert candidate.status is Status.VALID
    assert measured(candidate, CELLS).value == 3.0  # distinct cells per event: 3, 2, 4
    assert (
        candidate.value(
            MetricRef(other.name, *CELLS, "gamma" if other.name in STAGES else None)
        ).value
        == value
    )


# --- workdirs, artifacts and time stay with whoever made them ---


def plotter(name: str) -> Any:
    """Writes ``plot.txt`` into its own directory and returns it as the artifact ``plot``."""

    class Plotter:
        version = "1"
        metrics: dict[Any, str] = {}

        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            assert request.workdir is not None
            plot = request.workdir / "plot.txt"
            plot.write_text(name)
            return EvaluationResult.valid(artifacts={"plot": plot})

    evaluator = Plotter()
    evaluator.name = name  # type: ignore[attr-defined]
    return evaluator


def test_each_evaluator_writes_and_names_its_files_on_its_own(
    doc: dict[str, Any], tmp_path: Path
) -> None:
    spec = problem(doc, "a", "b", "test.parameter_echo", objectives=(ECHO,))
    candidate = _evaluate_candidate(
        spec,
        {CELL: 6},
        implementations(plotter("a"), plotter("b"), parameter_echo()),
        workdir=tmp_path,
    )
    plots = {r.evaluator_name: r.result.artifacts.get("plot") for r in candidate.records}
    assert plots["a"] == tmp_path / "evaluators" / "a" / "plot.txt"
    assert plots["b"] == tmp_path / "evaluators" / "b" / "plot.txt"
    assert plots["a"].read_text() == "a" and plots["b"].read_text() == "b"  # type: ignore[union-attr]


class Clock:
    """A ``time`` module whose ``perf_counter`` only moves when told to."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


def test_every_second_is_counted_once(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(_evaluate, "time", clock)
    monkeypatch.setattr(_work, "time", clock)
    simulation.install(monkeypatch)
    fake = ddsim_run._execute

    def slow_ddsim(argv: list[str], **kwargs: Any) -> Any:
        clock.now += 10.0
        return fake(argv, **kwargs)

    monkeypatch.setattr(ddsim_run, "_execute", slow_ddsim)

    class Slow(Spy):
        def check(self, request: EvaluationRequest) -> None:
            clock.now += 0.5
            super().check(request)

        def evaluate(self, request: EvaluationRequest) -> Any:
            clock.now += 1.0
            return super().evaluate(request)

    spec = problem(doc, "test.cell_count", "test.hit_energy", objectives=(CELLS, ENERGY))
    candidate: CandidateRecord = evaluate(
        spec, [Slow(cell_count(), []), Slow(hit_energy(), [])], tmp_path, geometry
    )
    seconds = {
        name: candidate.results[stage_task(name, "gamma")].elapsed_seconds
        for name in stages(candidate)
    }
    assert seconds == {"generation": 0.0, "simulation": 10.0}
    assert [record.elapsed_seconds for record in candidate.records] == [1.0, 1.0]
    assert candidate.elapsed_seconds == 13.0  # with the two checks before anything ran


# --- pipeline stages and metric labels are different things ---


def test_analysis_metrics_need_no_analysis_stage(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    class Busiest(Evaluator):
        name = "test.busiest"
        version = "1"
        stage = "simulation"
        collections = {"hits": Collection("SimCalorimeterHit")}
        metrics = {("analysis", "ecal.max_hits"): "1"}

        def evaluate(self, request: EvaluationRequest) -> Any:
            busiest = max(len(event.hits) for event in self.events(request))
            return {("analysis", "ecal.max_hits"): busiest}

    spec = problem(doc, "test.busiest", objectives=(("analysis", "ecal.max_hits"),))
    candidate = evaluate(spec, [Busiest()], tmp_path, geometry)
    assert stages(candidate) == ["generation", "simulation"]
    assert (
        candidate.value(MetricRef("test.busiest", "analysis", "ecal.max_hits", "gamma")).value == 4
    )


# --- the collections an evaluator declares are checked in the file before it is called ---


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            [Event(Other=Hits([1]))],
            f"collection {COLLECTION!r} (hits) is not in events.edm4hep.root, which holds: Other",
        ),
        (
            [Event(**{COLLECTION: Hits([1], "edm4hep::SimTrackerHit")})],
            f"collection {COLLECTION!r} (hits) holds 'edm4hep::SimTrackerHit', but the evaluator "
            "reads 'edm4hep::SimCalorimeterHit'",
        ),
    ],
)
def test_a_collection_missing_from_the_output_fails_the_evaluator_before_it_is_called(
    doc: dict[str, Any],
    tmp_path: Path,
    geometry: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: list[Any],
    message: str,
) -> None:
    simulation.install(monkeypatch, events * 3)  # the 3 events the problem asks for
    spec = problem(doc, "test.cell_count", "test.parameter_echo", objectives=(CELLS, ECHO))
    cells = Spy(cell_count(), [])
    candidate = evaluate(spec, [cells, parameter_echo()], tmp_path, geometry)
    counted, echo = candidate.records
    assert counted.result.status is Status.FAILED and counted.result.message == message
    assert cells.evaluated == [] and counted.elapsed_seconds is None
    # a scientific verdict is the evaluator's to give; the evaluator that needs no file still ran
    assert echo.result.status is Status.VALID and candidate.status is Status.FAILED


def test_the_collections_of_a_file_are_listed_once_for_all_its_readers(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    listed: list[Path] = []
    real = _work.available_collections

    def counting(path: Path) -> tuple[str, ...]:
        listed.append(path)
        return real(path)

    monkeypatch.setattr(_work, "available_collections", counting)
    spec = problem(doc, "test.cell_count", "test.hit_energy", objectives=(CELLS, ENERGY))
    candidate = evaluate(spec, [cell_count(), hit_energy()], tmp_path, geometry)
    assert candidate.status is Status.VALID
    assert listed == [tmp_path / "run" / "samples" / "gamma" / "simulation" / "events.edm4hep.root"]


class Truth(Evaluator):
    """Reads the generated particles from the generation stage's file."""

    name = "test.truth"
    version = "1"
    stage = "generation"
    collections = {"particles": Collection("MCParticle", default="MCParticles")}
    metrics = {("analysis", "truth.particles"): "1"}

    def evaluate(self, request: EvaluationRequest) -> Any:
        return {("analysis", "truth.particles"): [len(e.particles) for e in self.events(request)]}


def test_collections_of_a_stage_that_makes_no_file_are_refused_before_anything_runs(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    # the gun is fired inside ddsim: there is no generation file to read, and the simulation
    # another evaluator needs must not run first only for this one to fail
    spec = problem(doc, "test.cell_count", "test.truth", objectives=(CELLS, TRUTH))
    with pytest.raises(
        EvaluationContractError,
        match=r"'test\.truth' reads collections from the 'generation' stage of gamma, but executor "
        r"'gun' "
        r"runs inside samples\.gamma\.simulation and writes no file of its own",
    ):
        evaluate(spec, [cell_count(), Truth()], tmp_path, geometry)
    assert log == [] and not (tmp_path / "run").exists()


def test_collections_of_an_edm4hep_generation_file_are_read(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ddsim also reads generator events written as EDM4hep, which podio reads too
    (tmp_path / "zh.edm4hep.root").write_bytes(b"fake")
    photon = Particle(momentum=(1.0, 0.0, 0.0))
    events = [Event(**{COLLECTION: Hits([1]), "MCParticles": Particles([photon])})] * 3
    simulation.install(monkeypatch, events)
    spec = problem(
        doc,
        "test.cell_count",
        "test.truth",
        objectives=(CELLS, TRUTH),
        generation_file="zh.edm4hep.root",
    )
    candidate = evaluate(in_directory(spec, tmp_path), [cell_count(), Truth()], tmp_path, geometry)
    assert candidate.status is Status.VALID and measured(candidate, TRUTH).value == 1


@pytest.mark.parametrize("supplied", [False, True])
def test_collections_of_a_file_podio_cannot_read_are_refused_before_anything_runs(
    hepmc_doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str], supplied: bool
) -> None:
    # a HepMC file is an input of ddsim, not a file an evaluator can read collections from: the
    # simulation another evaluator needs must not run first only for this one to fail
    spec = problem(
        hepmc_doc,
        "test.cell_count",
        "test.truth",
        objectives=(CELLS, TRUTH),
        generation_file="zh.hepmc",
    )
    inputs = {"geometry": geometry}
    if supplied:  # k4opt evaluate --input generation=zh.hepmc, with the gun in the problem
        spec = problem(hepmc_doc, "test.cell_count", "test.truth", objectives=(CELLS, TRUTH))
        inputs["gamma/generation"] = tmp_path / "zh.hepmc"
    with pytest.raises(
        EvaluationContractError,
        match=r"'test\.truth' reads collections from the 'generation' stage of gamma, but its file "
        r"zh\.hepmc is no EDM4hep file that podio reads \(\.root or \.sio\)",
    ):
        _evaluate_candidate(
            in_directory(spec, tmp_path),
            {CELL: 6},
            implementations(cell_count(), Truth()),
            inputs=inputs,
            workdir=tmp_path / "run",
        )
    assert log == [] and not (tmp_path / "run").exists()


class Unpackable(Event):
    """An event with a collection podio cannot unpack, which no evaluator reads."""

    def get(self, name: str) -> Any:
        if name == "Broken":
            raise RuntimeError("no dictionary for the collection type")
        return super().get(name)


def test_a_collection_no_evaluator_reads_cannot_fail_the_ones_that_read_the_file(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(
        monkeypatch, [Unpackable(Broken=Hits([1]), **{COLLECTION: Hits([1, 2])})] * 3
    )
    spec = problem(doc, "test.cell_count", objectives=(CELLS,))
    candidate = evaluate(spec, [cell_count()], tmp_path, geometry)
    assert candidate.status is Status.VALID and measured(candidate, CELLS).value == 2


# --- one implementation, bound several times ---


BUSY = ("simulation", "ecal_barrel.busy_events")


class Threshold(Evaluator):
    """The events with at least ``min_hits`` hits; one instance per binding and candidate."""

    name = "test.threshold"
    version = "1"
    stage = "simulation"
    collections = {"hits": Collection("SimCalorimeterHit")}
    settings = {"min_hits": 1}
    metrics = {BUSY: "1"}
    created: list[Threshold] = []

    def __init__(self) -> None:
        Threshold.created.append(self)
        self.asked: list[int] = []

    def evaluate(self, request: EvaluationRequest) -> Any:
        least = request.settings["min_hits"]
        self.asked.append(least)
        return {BUSY: sum(len(event.hits) >= least for event in self.events(request))}


def thresholds(doc: dict[str, Any], *settings: dict[str, Any], **extra: Any) -> ProblemSpec:
    """``test.threshold`` bound once with each of ``settings``: as itself, then as ``t1``, ``t2``,
    ..., with an objective on each binding's value."""
    doc = copy.deepcopy(doc)
    doc.pop("constraints", None)
    doc["samples"]["gamma"] = {
        "generation": {"executor": "gun", **GUN},
        "simulation": {"executor": "ddsim", "events": 3},
    }
    bindings = ["test.threshold", *(f"t{i}" for i in range(1, len(settings)))]
    doc["evaluators"] = [
        {"name": "test.threshold", "version": "1", "stage": "simulation"}
        | ({} if binding == "test.threshold" else {"as": binding})
        | {"collections": {"hits": COLLECTION}, "settings": given}
        for binding, given in zip(bindings, settings, strict=True)
    ]
    doc["objectives"] = [
        {"evaluator": binding, "stage": BUSY[0], "metric": BUSY[1], "direction": "minimize"}
        for binding in bindings
    ]
    return parse_problem(doc | extra)


def test_two_bindings_of_one_implementation_read_one_simulation_with_their_own_settings(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    Threshold.created.clear()
    spec = thresholds(doc, {"min_hits": 1}, {"min_hits": 3})
    assert spec.to_dict()["evaluators"][1] == {
        "name": "test.threshold",
        "as": "t1",
        "version": "1",
        "stage": "simulation",
        "samples": ["gamma"],
        "collections": {"hits": COLLECTION},
        "settings": {"min_hits": 3},
    }
    candidate = evaluate(spec, [Threshold], tmp_path, geometry)  # the class, given once
    assert log == ["ddsim"]  # one simulation, read by both bindings
    assert candidate.status is Status.VALID
    # events with 4, 2 and 4 hits: all have one, two have three
    assert candidate.value(MetricRef("test.threshold", *BUSY, "gamma")).value == 3
    assert candidate.value(MetricRef("t1", *BUSY, "gamma")).value == 2
    assert list(candidate.evaluations) == ["gamma/test.threshold", "gamma/t1"]
    assert {r.evaluator_name for r in candidate.records} == {"test.threshold"}
    assert reads(candidate) == {
        "test.threshold": ("geometry", "gamma/generation", "gamma/simulation"),
        "t1": ("geometry", "gamma/generation", "gamma/simulation"),
    }
    # the probe, then an instance of each binding of its own, each asked once
    probe, first, second = Threshold.created
    assert (probe.asked, first.asked, second.asked) == ([], [1], [3])
    assert sorted(
        p.name for p in (tmp_path / "run" / "samples" / "gamma" / "evaluators").iterdir()
    ) == [
        "t1",
        "test.threshold",
    ]


def test_a_binding_of_another_version_is_named_by_its_binding(
    doc: dict[str, Any], tmp_path: Path, log: list[str]
) -> None:
    spec = thresholds(doc, {"min_hits": 1}, {"min_hits": 3})
    entries = (spec.evaluators[0], dataclasses.replace(spec.evaluators[1], version="2"))
    spec = dataclasses.replace(spec, evaluators=entries)
    with pytest.raises(
        EvaluationContractError,
        match=r"problem declares evaluator 'test\.threshold' \(as 't1'\) version '2', got "
        r"version '1'",
    ):
        _evaluate_candidate(spec, {CELL: 6}, implementations(Threshold))
    assert log == []


@pytest.mark.parametrize(
    ("first", "second", "warned"),
    [
        ({"min_hits": 1}, {"min_hits": 1}, True),
        ({}, {"min_hits": 1}, True),  # the default filled in
        ({"min_hits": 1}, {"min_hits": 3}, False),
    ],
)
def test_two_bindings_that_measure_the_same_are_a_warning(
    doc: dict[str, Any],
    tmp_path: Path,
    geometry: Path,
    log: list[str],
    first: dict[str, Any],
    second: dict[str, Any],
    warned: bool,
) -> None:
    spec = thresholds(doc, first, second)
    candidate = evaluate(spec, [Threshold], tmp_path, geometry)
    assert candidate.status is Status.VALID  # both are evaluated all the same
    expected = (
        "evaluators 'test.threshold' and 't1' bind test.threshold with the same settings and "
        "collections on gamma; each is evaluated separately"
    )
    assert candidate.warnings == ((expected,) if warned else ())


class CutThreshold(Threshold):
    """:class:`Threshold` with a mapping setting too."""

    settings = {"min_hits": 1, "cuts": {}}


@pytest.mark.parametrize(
    ("second", "warned"),
    [({"high": 2, "low": 1}, True), ({"low": 1, "high": 3}, False)],
)
def test_bindings_with_a_mapping_setting_are_the_same_whatever_its_key_order(
    doc: dict[str, Any],
    tmp_path: Path,
    geometry: Path,
    log: list[str],
    second: dict[str, Any],
    warned: bool,
) -> None:
    spec = thresholds(doc, {"cuts": {"low": 1, "high": 2}}, {"cuts": second})
    candidate = evaluate(spec, [CutThreshold], tmp_path, geometry)
    assert candidate.status is Status.VALID
    assert len(candidate.warnings) == (1 if warned else 0)


def test_an_objective_on_the_metric_of_a_binding_needs_its_binding_to_measure_it(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = thresholds(doc, {"min_hits": 1}, {"min_hits": 3})
    objectives = (dataclasses.replace(spec.objectives[1], metric="nothing"),)
    spec = dataclasses.replace(spec, objectives=objectives)
    with pytest.raises(
        EvaluationContractError,
        match=r"evaluator 't1' does not measure simulation/nothing; supported: "
        r"simulation/ecal_barrel\.busy_events",
    ):
        evaluate(spec, [Threshold], tmp_path, geometry)
    assert log == []


# --- combined evaluators: each role is given what its implementation declares ---------------

SELECTED = ("analysis", "selected.sum_weights")  # test.yields
GENERATED = ("analysis", "generated.sum_weights")
SUM = ("analysis", "sum")  # test.combiner
SIGNAL = {"sample": "gamma", "evaluator": "test.yields"}
BOTH = {"samples": ["gamma", "pion"], "evaluator": "test.yields"}


def combiner(**declared: Role) -> type[Evaluator]:
    """``test.combiner``, a combined evaluator with the roles ``declared``: the sum of the
    selected weights of every evaluation its roles read."""

    class Combiner(Evaluator):
        name, version = "test.combiner", "1"
        roles = declared or None
        metrics = {SUM: "1"}

        def evaluate(self, request: EvaluationRequest) -> Any:
            items = [item for items in request.roles.values() for item in items]
            return {SUM: sum(i.measurement(*SELECTED).value for i in items if i.binding)}

    return Combiner


def combined_problem(doc: dict[str, Any], *entries: dict[str, Any]) -> ProblemSpec:
    """Two samples, ``gamma`` and ``pion``, each measured by ``test.yields``, and the combined
    ``entries`` (of ``test.combiner`` unless they say otherwise) with an objective on the sum
    of each."""
    doc = copy.deepcopy(doc)
    doc["samples"] = {
        name: {
            "generation": {"executor": "gun", **GUN, "particle": particle},
            "simulation": {"executor": "ddsim", "events": 3},
        }
        for name, particle in (("gamma", "gamma"), ("pion", "pi-"))
    }
    doc["evaluators"] = [
        {"name": "test.yields", "version": "1", "stage": "simulation"}
        | {"samples": ["gamma", "pion"], "collections": {"hits": COLLECTION}},
        *({"name": "test.combiner", "version": "1"} | entry for entry in entries),
    ]
    bindings = [entry.get("as", entry.get("name", "test.combiner")) for entry in entries]
    doc["objectives"] = [
        {"evaluator": binding, "stage": "analysis", "metric": "sum", "direction": "maximize"}
        for binding in bindings
    ]
    doc["constraints"] = []
    return parse_problem(doc)


@pytest.mark.parametrize(
    ("declared", "roles", "message"),
    [
        (
            {"signal": Role(metrics=[SELECTED])},
            {"signal": SIGNAL, "extra": SIGNAL},
            "evaluator 'test.combiner': its roles are signal; unknown extra",
        ),
        (
            {"signal": Role(), "backgrounds": Role(many=True)},
            {"signal": SIGNAL},
            "evaluator 'test.combiner': its roles are signal, backgrounds; missing backgrounds",
        ),
        (
            {"signal": Role()},
            {"signal": BOTH},
            "evaluator 'test.combiner': role 'signal' reads one item, but the problem gives 2: "
            "gamma/test.yields, pion/test.yields",
        ),
        (
            {"signal": Role(metrics=[SELECTED, ("analysis", "other")])},
            {"signal": SIGNAL},
            "role 'signal' reads analysis/other of gamma/test.yields, but test.yields does not "
            "declare it",
        ),
        (
            {"signal": Role(artifacts=["histogram", "covariance"])},
            {"signal": SIGNAL},
            "role 'signal' reads the artifact(s) covariance of gamma/test.yields, but "
            "test.yields does not declare it",
        ),
        (
            {"signal": Role(stage="simulation")},
            {"signal": SIGNAL},
            "role 'signal' reads the simulation file of a sample, as {sample: ..., stage: "
            "simulation}; got gamma/test.yields",
        ),
        (
            {"signal": Role()},
            {"signal": {"sample": "gamma", "stage": "simulation"}},
            "role 'signal' reads evaluations, as {sample: ..., evaluator: ...}; got the stage "
            "gamma/simulation",
        ),
    ],
)
def test_roles_that_do_not_fit_their_implementation_are_refused_before_anything_runs(
    doc: dict[str, Any], log: list[str], declared: dict[str, Role], roles: Any, message: str
) -> None:
    spec = combined_problem(doc, {"roles": roles})
    with pytest.raises(EvaluationContractError, match=re.escape(message)):
        _evaluate_candidate(spec, {CELL: 6}, implementations(yields(), combiner(**declared)))
    assert log == []


def test_a_combined_evaluator_and_its_entry_must_agree_that_it_is_one(
    doc: dict[str, Any], log: list[str]
) -> None:
    spec = combined_problem(doc, {"roles": {"signal": SIGNAL}})
    with pytest.raises(
        EvaluationContractError,
        match=re.escape(
            "evaluator 'test.combiner' declares no roles, so it is no combined evaluator; remove "
            "roles: from its entry"
        ),
    ):
        _evaluate_candidate(spec, {CELL: 6}, implementations(yields(), combiner()))
    spec = combined_problem(doc, {})
    with pytest.raises(
        EvaluationContractError,
        match=re.escape(
            "evaluator 'test.combiner' is a combined evaluator with the roles signal; say what "
            "each reads under roles: in its entry"
        ),
    ):
        _evaluate_candidate(
            spec, {CELL: 6}, implementations(yields(), combiner(signal=Role(metrics=[SELECTED])))
        )
    assert log == []


def test_malformed_settings_of_a_combined_evaluator_are_refused_before_anything_runs(
    doc: dict[str, Any], log: list[str]
) -> None:
    roles = {"signal": SIGNAL, "backgrounds": {"sample": "pion", "evaluator": "test.yields"}}
    entry = {"name": "test.toy_significance", "roles": roles}
    spec = combined_problem(doc, entry | {"settings": {"luminosity_invab": "much"}})
    with pytest.raises(
        EvaluationContractError, match=re.escape("settings.luminosity_invab: expected float")
    ):
        _evaluate_candidate(spec, {CELL: 6}, implementations(yields(), toy_significance()))
    assert log == []


def test_one_evaluation_may_fill_two_roles(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = combined_problem(doc, {"roles": {"signal": SIGNAL, "all": BOTH}})
    both = combiner(signal=Role(metrics=[SELECTED]), all=Role(many=True, metrics=[SELECTED]))
    candidate = evaluate(spec, [yields(), both], tmp_path, geometry)
    assert candidate.status is Status.VALID
    assert candidate.value(MetricRef("test.combiner", *SUM)).value == 3 + 3 + 3
    assert reads(candidate)["test.combiner"] == (
        "geometry",
        "gamma/test.yields",
        "pion/test.yields",
    )


def test_combined_evaluators_that_share_an_evaluation_read_it_made_once(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    made: list[str] = []

    class Counted(yields()):  # type: ignore[misc]
        def evaluate(self, request: EvaluationRequest) -> Any:
            made.append(str(request.workdir))
            return super().evaluate(request)

    spec = combined_problem(
        doc, {"as": "c1", "roles": {"signal": SIGNAL}}, {"as": "c2", "roles": {"signal": BOTH}}
    )
    many = combiner(signal=Role(many=True, metrics=[SELECTED]))
    candidate = evaluate(spec, [Counted, many], tmp_path, geometry)
    assert candidate.status is Status.VALID
    assert log.count("ddsim") == 2 and len(made) == 2  # each sample simulated and measured once
    # the same metric of two combined evaluators: two values, each its own evaluator's
    assert [ref.label for ref in spec.references] == ["c1/analysis/sum", "c2/analysis/sum"]
    assert candidate.value(MetricRef("c1", *SUM)).value == 3
    assert candidate.value(MetricRef("c2", *SUM)).value == 6


def test_a_valid_evaluation_a_role_reads_must_carry_the_metrics_the_role_reads(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    class Partial(yields()):  # type: ignore[misc]
        def evaluate(self, request: EvaluationRequest) -> Any:
            result = super().evaluate(request)
            kept = [m for m in result.measurements if (m.stage, m.metric) != GENERATED]
            return dataclasses.replace(result, measurements=tuple(kept))

    spec = combined_problem(doc, {"roles": {"signal": SIGNAL}})
    reading = combiner(signal=Role(metrics=[SELECTED, GENERATED]))
    with pytest.raises(
        EvaluationContractError,
        match=re.escape(
            "evaluator 'gamma/test.yields' returned a valid result without required metric(s): "
            "analysis/generated.sum_weights"
        ),
    ):
        evaluate(spec, [Partial, reading], tmp_path, geometry)
