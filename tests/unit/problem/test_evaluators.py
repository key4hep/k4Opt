"""Several evaluators of one candidate: each pipeline stage runs once, each evaluator measures on
its own, and the candidate's status and values follow from theirs (fake ddsim, no Key4hep)."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import run as ddsim_run
from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    Measurement,
    Status,
)
from k4opt.problem import ProblemSpec, _evaluate, _stages, parse_problem
from k4opt.problem._evaluate import CandidateRecord, _evaluate_candidate
from tests.support import simulation
from tests.support.evaluators import cell_count, hit_energy, parameter_echo
from tests.support.simulation import COLLECTION, events_with

CELL = "constants.ECal_cell_size"
GUN = {
    "particle": "gamma",
    "energy": {"value": 10, "unit": "GeV"},
    "direction": {"distribution": "uniform", "theta": [60, 120], "phi": [0, 360], "unit": "deg"},
}
CELLS = ("simulation", "ecal_barrel.mean_hit_cells")  # test.cell_count
ENERGY = ("simulation", "ecal_barrel.mean_hit_energy")  # test.hit_energy
ECHO = ("analysis", "candidate.first_value")  # test.parameter_echo
SETTINGS = {
    name: {"collection": COLLECTION}
    for name in ("test.cell_count", "test.hit_energy", "test.busiest")
}


def problem(
    doc: dict[str, Any],
    *names: str,
    objectives: tuple[tuple[str, str], ...] = (CELLS,),
    constraints: tuple[dict[str, Any], ...] = (),
    **evaluation: Any,
) -> ProblemSpec:
    """The shared problem with evaluators ``names`` (version 1) and these objectives; an
    ``evaluation`` key given as ``None`` is left out."""
    doc = copy.deepcopy(doc)
    doc["evaluation"].update({"events": 3, "generation": GUN} | evaluation)
    for key, value in evaluation.items():
        if value is None:
            del doc["evaluation"][key]
    doc["evaluation"]["evaluators"] = [
        {"name": name, "version": "1"} | ({"settings": SETTINGS[name]} if name in SETTINGS else {})
        for name in names
    ]
    doc["objectives"] = [{"stage": s, "metric": m, "direction": "minimize"} for s, m in objectives]
    doc["constraints"] = list(constraints)
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
    fake = ddsim_run.subprocess.run

    def logged(argv: list[str], **kwargs: Any) -> Any:
        if Path(argv[0]).name == "ddsim":
            events.append("ddsim")
        return fake(argv, **kwargs)

    monkeypatch.setattr(ddsim_run.subprocess, "run", logged)
    return events


class Spy:
    """Passes everything on to ``inner`` and logs each ``check`` and ``evaluate``."""

    def __init__(self, inner: Any, log: list[str]) -> None:
        self.inner, self.log = inner, log
        self.name, self.version = inner.name, inner.version
        self.requires, self.metrics = inner.requires, inner.metrics
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
        spec, {CELL: 6}, evaluators, inputs={"geometry": geometry}, workdir=tmp_path / "run"
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
    events_file = tmp_path / "run" / "simulation" / "events.edm4hep.root"
    assert cells.evaluated[0].inputs["simulation"] == events_file
    assert energy.evaluated[0].inputs["simulation"] == events_file
    assert candidate.status is Status.VALID
    first, second = candidate.records
    assert [(m.stage, m.metric) for m in first.result.measurements] == [CELLS]
    assert [(m.stage, m.metric) for m in second.result.measurements] == [ENERGY]
    assert candidate.value(*CELLS).value == 3.0  # distinct cells per event: 3, 2, 4
    assert candidate.value(*ENERGY).value == pytest.approx(10 / 3)  # 1 GeV per hit: 4, 2, 4
    assert candidate.owners == {CELLS: "test.cell_count", ENERGY: "test.hit_energy"}
    assert list(candidate.stages.provenance) == ["generation", "simulation"]
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["evaluators", "simulation"]


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


def test_an_evaluator_that_requires_no_stage_runs_no_simulation(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    # without a simulation, neither events nor a generation are needed
    spec = problem(doc, "test.parameter_echo", objectives=(ECHO,), events=None, generation=None)
    candidate = evaluate(spec, [parameter_echo()], tmp_path, geometry)
    assert log == [] and not (tmp_path / "run" / "simulation").exists()
    assert candidate.stages.provenance == {} and candidate.reads == {"test.parameter_echo": ()}
    assert candidate.status is Status.VALID and candidate.value(*ECHO).value == 6


class SampleSize(Evaluator):
    """Reads only the generation stage: the size of the sample file."""

    name = "test.sample_size"
    version = "1"
    requires = "generation"
    metrics = {("analysis", "sample.bytes"): "B"}

    def __init__(self) -> None:
        self.seen: list[EvaluationRequest] = []

    def evaluate(self, request: EvaluationRequest) -> Any:
        self.seen.append(request)
        return {("analysis", "sample.bytes"): request.inputs["generation"].stat().st_size}


@pytest.fixture
def sample_doc(doc: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "zh.hepmc").write_bytes(b"HepMC::Version 3\n")
    return doc


def with_sample(spec: ProblemSpec, tmp_path: Path) -> ProblemSpec:
    return dataclasses.replace(spec, _directory=tmp_path)


def test_each_evaluator_reads_the_files_of_the_stages_it_needs_and_no_others(
    sample_doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = problem(
        sample_doc,
        "test.cell_count",
        "test.sample_size",
        objectives=(CELLS, ("analysis", "sample.bytes")),
        generation={"sample": "zh.hepmc"},
    )
    cells, size = Spy(cell_count(), log), SampleSize()
    candidate = evaluate(with_sample(spec, tmp_path), [cells, size], tmp_path, geometry)
    assert log.count("ddsim") == 1
    # simulation needs generation, so its reader gets both files; the sample reader only its own
    assert set(cells.evaluated[0].inputs) == {"geometry", "generation", "simulation"}
    assert set(size.seen[0].inputs) == {"geometry", "generation"}
    assert candidate.reads == {
        "test.cell_count": ("generation", "simulation"),
        "test.sample_size": ("generation",),
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
    for field in ("candidate", "settings", "events", "seed", "backend_args", "generation"):
        assert getattr(first, field) == getattr(second, field)
    assert set(first.inputs) == set(second.inputs)


# --- a failure only concerns the evaluators that read the failed stage ---


def test_an_independent_evaluator_keeps_its_result_when_the_simulation_fails(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, returncode=2, stderr="G4Exception: bad volume")
    spec = problem(doc, "test.cell_count", "test.parameter_echo", objectives=(CELLS, ECHO))
    candidate = evaluate(spec, [cell_count(), parameter_echo()], tmp_path, geometry)
    assert len(fake.calls) == 1
    cells, echo = candidate.records
    assert cells.result.status is Status.FAILED and cells.elapsed_seconds is None
    assert "ddsim exited with status 2" in (cells.result.message or "")
    assert echo.result.status is Status.VALID and echo.elapsed_seconds is not None
    # the evaluation is incomplete, but the measurement that was made is kept
    assert candidate.status is Status.FAILED
    assert candidate.value(*CELLS) is None
    assert candidate.value(*ECHO).value == 6
    assert (candidate.message or "").startswith("test.cell_count failed: ddsim exited")
    assert not (tmp_path / "run" / "evaluators" / "test.cell_count").exists()


def test_a_generation_reader_keeps_its_result_when_the_simulation_fails(
    sample_doc: dict[str, Any], tmp_path: Path, geometry: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2)
    spec = problem(
        sample_doc,
        "test.cell_count",
        "test.sample_size",
        objectives=(CELLS, ("analysis", "sample.bytes")),
        generation={"sample": "zh.hepmc"},
    )
    size = SampleSize()
    candidate = evaluate(with_sample(spec, tmp_path), [cell_count(), size], tmp_path, geometry)
    cells, sized = candidate.records
    assert cells.result.status is Status.FAILED
    assert sized.result.status is Status.VALID and set(size.seen[0].inputs) == {
        "geometry",
        "generation",
    }


@pytest.mark.parametrize("readable", [True, False])
def test_a_supplied_simulation_does_not_depend_on_the_generation_another_evaluator_runs(
    sample_doc: dict[str, Any],
    tmp_path: Path,
    log: list[str],
    monkeypatch: pytest.MonkeyPatch,
    readable: bool,
) -> None:
    # the caller's simulation file bypasses generation, which runs only for the sample reader:
    # its file never reaches the simulation reader, and its failure does not fail that reader
    spec = problem(
        sample_doc,
        "test.cell_count",
        "test.sample_size",
        objectives=(CELLS, ("analysis", "sample.bytes")),
        generation={"sample": "zh.hepmc"},
    )
    sample, events_file = tmp_path / "zh.hepmc", tmp_path / "mine.edm4hep.root"
    events_file.write_bytes(b"fake")
    if not readable:
        real_open = Path.open

        def unreadable(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self == sample:
                raise PermissionError("denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", unreadable)
    cells = Spy(cell_count(), log)
    candidate = _evaluate_candidate(
        with_sample(spec, tmp_path),
        {CELL: 6},
        [cells, SampleSize()],
        inputs={"simulation": events_file},
    )
    assert "ddsim" not in log
    assert set(cells.evaluated[0].inputs) == {"simulation"}
    assert candidate.reads["test.cell_count"] == ("simulation",)
    counted, sized = candidate.records
    assert counted.result.status is Status.VALID
    assert candidate.value(*CELLS).value == 3.0
    if readable:
        assert sized.result.status is Status.VALID and candidate.status is Status.VALID
    else:
        assert sized.result.status is Status.FAILED
        assert "cannot read the sample" in (sized.result.message or "")


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
    candidate = _evaluate_candidate(spec, {CELL: 6}, [toy("a", first), toy("b", second)])
    assert candidate.status == status
    assert candidate.message == message
    for name, own in (("a", first), ("b", second)):
        record = candidate.record(name)
        assert record.result.status == own
        # a valid result gives the objective; an invalid one keeps its diagnostic in its record
        assert (candidate.value("analysis", f"{name}.value") is not None) == (own == "valid")
        assert len(record.result.measurements) == (0 if own == "failed" else 1)


# --- everything that can be checked is checked before anything runs ---


class Rival(Evaluator):
    """Declares a metric of ``test.cell_count``."""

    name = "test.rival"
    version = "1"
    requires = "simulation"
    metrics = {"ecal_barrel.mean_hit_cells": "1"}


class Silent(Evaluator):
    """Declares no metrics."""

    name = "test.silent"
    version = "1"


class Ahead(Evaluator):
    """Requires a stage that does not exist yet."""

    name = "test.ahead"
    version = "1"
    requires = "reconstruction"
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
            ("test.cell_count", "test.rival"),
            (cell_count, Rival),
            {},
            r"simulation/ecal_barrel\.mean_hit_cells is declared by evaluators "
            r"'test\.cell_count' and 'test\.rival'",
        ),
        (
            ("test.cell_count", "test.silent"),
            (cell_count, Silent),
            {},
            r"evaluator\(s\) 'test\.silent' declare no metrics",
        ),
        (
            ("test.cell_count", "test.hit_energy"),
            (cell_count, hit_energy),
            {"objectives": (CELLS, ("simulation", "nothing"))},
            r"no evaluator measures simulation/nothing; 'test\.cell_count' measures "
            r"simulation/ecal_barrel\.mean_hit_cells; 'test\.hit_energy' measures",
        ),
        (
            ("test.cell_count", "test.hit_energy"),
            (cell_count, hit_energy),
            {"constraints": ({"stage": ENERGY[0], "metric": ENERGY[1], "max": 1, "unit": "MeV"},)},
            r"declares unit 'MeV' but evaluator 'test\.hit_energy' measures it in 'GeV'",
        ),
        (
            ("test.cell_count", "test.ahead"),
            (cell_count, Ahead),
            {},
            r"unknown stage 'reconstruction'",
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
    entries = (
        spec.evaluation.evaluators[0],
        dataclasses.replace(spec.evaluation.evaluators[1], version="2"),
    )
    spec = dataclasses.replace(
        spec, evaluation=dataclasses.replace(spec.evaluation, evaluators=entries)
    )
    with pytest.raises(
        EvaluationContractError, match=r"'test\.hit_energy' version '2', got version '1'"
    ):
        _evaluate_candidate(spec, {CELL: 6}, [cell_count(), hit_energy()])
    assert log == []


def test_invalid_settings_of_the_second_evaluator_are_refused_before_anything_runs(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    spec = problem(doc, "test.cell_count", "test.hit_energy")
    entries = (
        spec.evaluation.evaluators[0],
        dataclasses.replace(spec.evaluation.evaluators[1], settings={"collection": 3}),
    )
    spec = dataclasses.replace(
        spec, evaluation=dataclasses.replace(spec.evaluation, evaluators=entries)
    )
    with pytest.raises(EvaluationContractError, match=r"settings\.collection: expected str"):
        evaluate(spec, [cell_count(), hit_energy()], tmp_path, geometry)
    assert log == [] and not (tmp_path / "run").exists()


def test_a_metric_of_another_evaluator_is_a_contract_error(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    class Greedy(Evaluator):
        name = "test.greedy"
        version = "1"
        metrics = {("analysis", "greedy.value"): "1"}

        def evaluate(self, request: EvaluationRequest) -> Any:
            return {("analysis", "greedy.value"): 1, CELLS: 99}  # not its metric

    spec = problem(doc, "test.cell_count", "test.greedy")
    with pytest.raises(
        EvaluationContractError,
        match=r"'test\.greedy' reported simulation/ecal_barrel\.mean_hit_cells, a metric of "
        r"evaluator 'test\.cell_count'",
    ):
        evaluate(spec, [cell_count(), Greedy()], tmp_path, geometry)


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
        spec, {CELL: 6}, [plotter("a"), plotter("b"), parameter_echo()], workdir=tmp_path
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
    monkeypatch.setattr(_stages, "time", clock)
    monkeypatch.setattr(_evaluate, "time", clock)
    simulation.install(monkeypatch)
    fake = ddsim_run.subprocess.run

    def slow_ddsim(argv: list[str], **kwargs: Any) -> Any:
        clock.now += 10.0 if Path(argv[0]).name == "ddsim" else 0.0
        return fake(argv, **kwargs)

    monkeypatch.setattr(ddsim_run.subprocess, "run", slow_ddsim)

    class Slow(Spy):
        def evaluate(self, request: EvaluationRequest) -> Any:
            clock.now += 1.0
            return super().evaluate(request)

    spec = problem(doc, "test.cell_count", "test.hit_energy", objectives=(CELLS, ENERGY))
    candidate: CandidateRecord = evaluate(
        spec, [Slow(cell_count(), []), Slow(hit_energy(), [])], tmp_path, geometry
    )
    assert candidate.stages.seconds == {"generation": 0.0, "simulation": 10.0}
    assert [record.elapsed_seconds for record in candidate.records] == [1.0, 1.0]
    assert candidate.elapsed_seconds == 12.0


# --- pipeline stages and metric labels are different things ---


def test_analysis_metrics_need_no_analysis_stage(
    doc: dict[str, Any], tmp_path: Path, geometry: Path, log: list[str]
) -> None:
    class Busiest(Evaluator):
        name = "test.busiest"
        version = "1"
        requires = "simulation"
        settings = {"collection": str}
        metrics = {("analysis", "ecal.max_hits"): "1"}

        def evaluate(self, request: EvaluationRequest) -> Any:
            hits = self.read(request, request.settings["collection"], "SimCalorimeterHit")
            return {("analysis", "ecal.max_hits"): max(len(event) for event in hits)}

    spec = problem(doc, "test.busiest", objectives=(("analysis", "ecal.max_hits"),))
    candidate = evaluate(spec, [Busiest()], tmp_path, geometry)
    assert list(candidate.stages.provenance) == ["generation", "simulation"]
    assert candidate.value("analysis", "ecal.max_hits").value == 4
