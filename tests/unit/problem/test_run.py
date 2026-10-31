"""``run_candidate`` and ``k4opt evaluate`` with a fake ddsim, on the mini geometry."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt import __version__
from k4opt.cli import main
from k4opt.ddsim import run as ddsim_run
from k4opt.errors import K4OptError
from k4opt.evaluation import (
    Artifact,
    Collection,
    EvaluationContractError,
    EvaluationRequest,
    Quantity,
    collection_names,
    metric_units,
    result_from,
    with_defaults,
)
from k4opt.geometry import GeometrySource, Override
from k4opt.problem import (
    SUMMARY_NAME,
    Assessment,
    CandidateError,
    EvaluationSession,
    ProblemConfigError,
    RecordError,
    RunError,
    RunOutcome,
    _stages,
    load_problem,
    parse_problem,
)
from k4opt.problem._run import _parse_candidate, _xml_value, assemble
from k4opt.problem._run import run_candidate as _run_candidate
from k4opt.problem._tasks import Kind, Plan, RunTask, TaskId, TaskResult, blocked
from k4opt.revision import code_fingerprint, code_identity
from tests.support import simulation
from tests.support.common import tree_digest
from tests.support.evaluators import (
    CELL_COUNT,
    HIT_ENERGY,
    PARAMETER_ECHO,
    calo_photon,
    cell_count,
    hit_energy,
    implementations,
    parameter_echo,
)
from tests.support.geometry import MINI_COMPACT, MINI_GEO
from tests.support.simulation import COLLECTION, events_with

RADIUS = "constants.InnerTracker_Barrel_radius_1"
# the mini geometry has no calorimeter, so ddsim is not expected to write the collection the test
# evaluator reads; the fake ddsim writes it anyway, and the output decides
UNEXPECTED = (
    f"evaluator 'test.cell_count': ddsim is not expected to write the collection {COLLECTION!r} "
    f"(hits) for {MINI_COMPACT}"
)
EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "problem"


def run_candidate(
    problem: Path, assignments: list[str], output: Path, **changes: Any
) -> RunOutcome:
    """``run_candidate`` with the test evaluator unless the test passes ``evaluators`` of its own:
    classes, or instances to watch."""
    changes["implementations"] = implementations(*changes.pop("evaluators", [cell_count()]))
    return _run_candidate(problem, assignments, output, **changes)


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def problem_file(tmp_path: Path, doc: dict[str, Any]) -> Path:
    doc["geometry"] = MINI_COMPACT
    doc["parameters"][0]["geometry"] = {"constant": "InnerTracker_Barrel_radius_1"}
    doc["parameters"][0]["bounds"] = [100.0, 600.0]
    doc["samples"]["gamma"]["simulation"].update(events=3, seed=5)
    doc["evaluators"] = [
        {
            "name": "test.cell_count",
            "version": "1",
            "stage": "simulation",
            "collections": {"hits": COLLECTION},
        }
    ]
    doc["objectives"] = [
        {"stage": "simulation", "metric": "ecal_barrel.mean_hit_cells", "direction": "minimize"}
    ]
    del doc["constraints"]
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> simulation.FakeDdsim:
    return simulation.install(monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9]))


@pytest.fixture
def guard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every discovery and materialization (and lets them run)."""
    calls: list[str] = []
    discover, materialize = GeometrySource.discover, GeometrySource.materialize

    def spy_discover(self: GeometrySource) -> Any:
        calls.append("discover")
        return discover(self)

    def spy_materialize(self: GeometrySource, *args: Any, **kwargs: Any) -> Any:
        calls.append("materialize")
        return materialize(self, *args, **kwargs)

    monkeypatch.setattr(GeometrySource, "discover", spy_discover)
    monkeypatch.setattr(GeometrySource, "materialize", spy_materialize)
    return calls


def test_candidate_is_materialized_with_the_cached_catalog_and_simulated(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[list[Override], Any]] = []
    materialize = GeometrySource.materialize

    def spy(self: GeometrySource, overrides: Any, output: Any, catalog: Any = None) -> Any:
        seen.append((list(overrides), catalog))
        return materialize(self, overrides, output, catalog)

    monkeypatch.setattr(GeometrySource, "materialize", spy)
    source_before = tree_digest(MINI_GEO)
    output = tmp_path / "runs" / "a"

    outcome = run_candidate(problem_file, [f"{RADIUS}=450.0"], output)

    ((overrides, catalog),) = seen
    assert overrides == [
        Override(target=RADIUS, value="450.0*mm", file="compact/det_v1/tracker.xml")
    ]
    assert catalog is not None  # the one discovered by load_problem, not rediscovered
    assert tree_digest(MINI_GEO) == source_before
    snapshot = output / "geometry"
    assert 'value="450.0*mm"' in (snapshot / "compact/det_v1/tracker.xml").read_text()
    ((argv, _),) = fake.calls
    assert f"--compactFile={snapshot / MINI_COMPACT}" in argv
    assert outcome.candidate.status == "valid"
    assert outcome.candidate.records[0].request.inputs["geometry"] == snapshot / MINI_COMPACT


def test_geometry_is_discovered_once_per_run(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert guard == ["discover", "materialize"]


def test_summary_is_written_and_truthful(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    on_disk = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    assert on_disk == json.loads(json.dumps(outcome.summary))

    assert on_disk["schema"] == "k4opt.run/1"
    assert (on_disk["status"], on_disk["message"]) == ("valid", None)
    # every stage was made from this candidate
    assert (on_disk["represents_candidate"], on_disk["supplied"]) == (True, [])
    assert on_disk["candidate"] == {RADIUS: 450.0}
    # the implementation once, and its binding's work
    (implementation,) = on_disk["implementations"].values()
    assert list(on_disk["implementations"]) == ["test.cell_count"]
    assert (implementation["version"], implementation["stage"]) == ("1", "simulation")
    # its sample, with the stages and the evaluations of the sample; none of the candidate
    assert list(on_disk["samples"]) == ["gamma"]
    assert list(on_disk["samples"]["gamma"]) == ["stages", "evaluations"]
    assert on_disk["evaluations"] == {}
    (evaluator,) = on_disk["samples"]["gamma"]["evaluations"].values()
    assert list(on_disk["samples"]["gamma"]["evaluations"]) == ["test.cell_count"]
    assert evaluator["implementation"] == "test.cell_count"
    assert (evaluator["status"], evaluator["message"]) == ("valid", None)
    (measurement,) = evaluator["measurements"]
    assert measurement["metric"] == "ecal_barrel.mean_hit_cells"
    assert measurement["value"] == 3.0  # distinct cells per event: 3, 2, 4
    # the objective, from the evaluator it names
    assert on_disk["metrics"] == [
        {
            "evaluator": "test.cell_count",
            "sample": "gamma",
            "stage": "simulation",
            "metric": "ecal_barrel.mean_hit_cells",
            "label": "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
        }
        | {k: measurement[k] for k in ("value", "unit", "uncertainty")}
    ]
    assert "events" not in on_disk  # each simulation says what it read
    assert outcome.events == {"gamma": (3, 3)}
    assert on_disk["geometry"]["fingerprint"] == outcome.fingerprint
    assert on_disk["geometry"]["status"] == "done"
    # each file in the run directory by its path relative to it, which it was made in
    assert on_disk["run_dir"] == str(tmp_path / "a")
    assert on_disk["geometry"]["manifest"] == "geometry/k4opt-manifest.json"
    assert (tmp_path / "a" / on_disk["geometry"]["compact"]).is_file()
    assert on_disk["geometry"]["changes"] == [
        {"target": RADIUS, "old_value": "400*mm", "new_value": "450.0*mm"}
    ]
    # the stages, recorded once, apart from the evaluators that read them
    assert list(on_disk["samples"]["gamma"]["stages"]) == ["generation", "simulation"]
    simulated = on_disk["samples"]["gamma"]["stages"]["simulation"]
    assert simulated["argv"][0] == "/stack/bin/ddsim"
    assert (simulated["returncode"], simulated["events_read"]) == (0, 3)
    # each a task: how it ended, and the tasks it read
    assert (simulated["status"], simulated["message"]) == ("done", None)
    assert (simulated["reads"], simulated["blocked_by"]) == (["geometry", "gamma/generation"], [])
    assert on_disk["samples"]["gamma"]["stages"]["generation"]["reads"] == []
    assert evaluator["settings"] == {"declared": {}, "used": {}}
    assert evaluator["collections"] == {"hits": COLLECTION}
    (warning,) = on_disk["warnings"]
    assert warning.startswith(UNEXPECTED) and warning.endswith("The simulation's output decides.")
    # the code of the evaluator, which its name and version only label
    source = Path(implementation["source"]["path"])
    assert source.name == "cell_count.py"
    assert implementation["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert implementation["source"]["executed"] is True  # loaded as FILE.py:Class
    assert evaluator["provenance"] == {} and evaluator["artifacts"] == {}  # none of its own
    assert evaluator["reads"] == ["geometry", "gamma/generation", "gamma/simulation"]
    assert evaluator["blocked_by"] == []
    assert evaluator["workdir"] == "samples/gamma/evaluators/test.cell_count"
    assert (tmp_path / "a" / evaluator["workdir"]).is_dir()
    # every second once: the geometry, each stage, the evaluator's own call, and the candidate
    # around them
    seconds = [stage["elapsed_seconds"] for stage in on_disk["samples"]["gamma"]["stages"].values()]
    seconds += [on_disk["geometry"]["elapsed_seconds"], evaluator["elapsed_seconds"]]
    assert min(seconds) >= 0 and sum(seconds) <= on_disk["elapsed_seconds"]
    # what the simulation made in its directory: its output, by reference, never hashed, and
    # its logs
    assert simulated["directory"] == "samples/gamma/simulation"
    events = tmp_path / "a" / "samples" / "gamma" / "simulation" / "events.edm4hep.root"
    assert simulated["outputs"] == {
        "events": {
            "ref": "gamma/simulation#events",
            "path": "samples/gamma/simulation/events.edm4hep.root",
            "bytes": events.stat().st_size,
            "sha256": None,
        }
    }
    assert list(simulated["logs"]) == ["stdout", "stderr", "command"]
    for path in simulated["logs"].values():
        assert (tmp_path / "a" / path).is_file()
        assert path.startswith("samples/gamma/simulation/")
    assert "paths" not in on_disk and "cwd" not in simulated
    assert (
        on_disk["samples"]["gamma"]["stages"]["generation"]["particle_gun"]["particle"] == "gamma"
    )
    assert on_disk["environment"]["declared"] == {"channel": "nightly", "version": "2026-10-20"}
    assert on_disk["environment"]["observed"]["ddsim"] == "/stack/bin/ddsim"
    assert on_disk["environment"]["observed"]["k4opt"] == __version__
    # what the version only labels: the content of the code that ran
    assert on_disk["environment"]["observed"]["k4opt_source"] == code_identity()
    assert on_disk["problem"]["path"] == str(problem_file.absolute())
    assert on_disk["candidate_key"] == [[RADIUS, "continuous", "num", "45e1"]]
    # the record's own identity, made for no study, and the candidate judged
    assert str(uuid.UUID(on_disk["id"])) == on_disk["id"]
    assert on_disk["run"] is None
    assert on_disk["assessment"] == {
        "feasible": True,  # no constraints
        "objectives": [
            {
                "evaluator": "test.cell_count",
                "sample": "gamma",
                "stage": "simulation",
                "metric": "ecal_barrel.mean_hit_cells",
                "direction": "minimize",
            }
            | {k: measurement[k] for k in ("value", "unit", "uncertainty")}
        ],
        "limits": [],
    }
    assert Assessment.from_summary(load_problem(problem_file), on_disk) == outcome.assessment
    assert "revision" in implementation["source"]  # of the checkout that tracks it, if one does
    # what it declares: every measurement of a metric it declares carries the declared unit
    assert implementation["metrics"] == [
        {"stage": "simulation", "metric": "ecal_barrel.mean_hit_cells", "unit": "1"}
    ]


def test_two_candidates_give_distinct_geometries(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    first = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    second = run_candidate(problem_file, [f"{RADIUS}=500"], tmp_path / "b")
    assert first.fingerprint != second.fingerprint
    tracker = "geometry/compact/det_v1/tracker.xml"
    assert (tmp_path / "a" / tracker).read_text() != (tmp_path / "b" / tracker).read_text()


@pytest.mark.parametrize(
    "assignments",
    [
        [],  # missing parameter
        [f"{RADIUS}=99"],  # below the bounds
        [f"{RADIUS}=601"],
        [f"{RADIUS}=nan"],
        [f"{RADIUS}=1_000"],
        [f"{RADIUS}=abc"],
        [f"{RADIUS}"],
        [f"{RADIUS}=450", f"{RADIUS}=460"],
        [f"{RADIUS}=450", "constants.nothing=1"],
        ["InnerTracker_Barrel_radius_1=450"],  # not the canonical reference
    ],
)
def test_invalid_candidates_are_rejected_before_any_side_effect(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
    assignments: list[str],
) -> None:
    output = tmp_path / "runs" / "a"
    with pytest.raises(CandidateError):
        run_candidate(problem_file, assignments, output)
    assert not output.exists()
    assert not output.parent.exists()
    assert "materialize" not in guard
    assert fake.calls == []


def test_a_conflicting_ddsim_option_is_rejected_before_any_side_effect(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["samples"]["gamma"]["simulation"]["numberOfEvents"] = 5
    problem_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(ProblemConfigError, match=r"numberOfEvents: .* set events instead"):
        run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert not (tmp_path / "a").exists()
    assert "discover" not in guard


def test_another_evaluator_is_rejected_before_any_side_effect(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluators"][0]["name"] = "calo.single_particle_shower"
    problem_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(EvaluationContractError, match="calo\\.single_particle_shower"):
        run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert not (tmp_path / "a").exists()
    assert "materialize" not in guard


def test_an_unmeasurable_metric_is_rejected_before_any_side_effect(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["objectives"][0]["metric"] = "nonexistent"
    problem_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(EvaluationContractError, match="simulation/nonexistent"):
        run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert not (tmp_path / "a").exists()
    assert "materialize" not in guard


def test_a_constraint_unit_mismatch_is_rejected_before_any_side_effect(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["constraints"] = [
        {"stage": "simulation", "metric": "ecal_barrel.mean_hit_cells", "max": 11, "unit": "mm"}
    ]
    problem_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(EvaluationContractError, match="declares unit 'mm'"):
        run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert not (tmp_path / "a").exists()
    assert "materialize" not in guard
    assert fake.calls == []


def test_an_existing_output_is_never_overwritten(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    output = tmp_path / "a"
    output.mkdir()
    (output / "keep.txt").write_text("data")
    with pytest.raises(RunError, match="already exists"):
        run_candidate(problem_file, [f"{RADIUS}=450"], output)
    assert [p.name for p in output.iterdir()] == ["keep.txt"]


def test_an_unusable_output_location_is_a_clean_error(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    blocker = tmp_path / "existing-file"
    blocker.write_text("data")
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(blocker / "run")
    )
    assert (code, out) == (1, "")
    assert errors(err).startswith("k4opt: error: ")
    assert "materialization failed" in err
    assert blocker.read_text() == "data"
    assert fake.calls == []


def test_numeric_categories_keep_exact_integers(doc: dict[str, Any]) -> None:
    big = 2**53
    doc["parameters"][0] = {
        "geometry": {"constant": "n"},
        "type": "categorical",
        "choices": [big, big + 1, 0.5],
    }
    spec = parse_problem(doc)
    assert _parse_candidate(spec, [f"constants.n={big + 1}"]) == {"constants.n": big + 1}
    assert _parse_candidate(spec, [f"constants.n={big}"]) == {"constants.n": big}
    assert _parse_candidate(spec, ["constants.n=0.5"]) == {"constants.n": 0.5}
    from k4opt.problem._evaluate import _validate_candidate

    _, key_a = _validate_candidate(spec, {"constants.n": big})
    _, key_b = _validate_candidate(spec, _parse_candidate(spec, [f"constants.n={big + 1}"]))
    assert key_a != key_b  # two different candidates stay different


def test_selector_references_with_equals_signs_can_be_passed(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    ref = "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"
    doc = yaml.safe_load(problem_file.read_text())
    doc["parameters"] = [{"geometry": {"target": ref}, "type": "integer", "bounds": [10, 100]}]
    problem_file.write_text(yaml.safe_dump(doc))
    assert _parse_candidate(load_problem(problem_file), [f"{ref}=90"]) == {ref: 90}
    outcome = run_candidate(problem_file, [f"{ref}=90"], tmp_path / "a")
    assert outcome.summary["candidate"] == {ref: 90}
    with pytest.raises(CandidateError, match="REFERENCE=VALUE"):
        _parse_candidate(load_problem(problem_file), [f"{ref}"])


@pytest.mark.parametrize(
    "text",
    ["9007199254740993.0", "9.007199254740993e15", "9007199254740993.00", "+9007199254740993"],
)
def test_every_spelling_of_a_large_integer_category_stays_exact(
    doc: dict[str, Any], text: str
) -> None:
    big = 2**53
    doc["parameters"][0] = {
        "geometry": {"constant": "n"},
        "type": "categorical",
        "choices": [big, big + 1, 1e23, 0.5],
    }
    spec = parse_problem(doc)
    assert _parse_candidate(spec, [f"constants.n={text}"]) == {"constants.n": big + 1}
    assert _parse_candidate(spec, ["constants.n=9007199254740992.0"]) == {"constants.n": big}
    assert _parse_candidate(spec, ["constants.n=1e23"]) == {"constants.n": 1e23}
    assert _parse_candidate(spec, ["constants.n=0.50"]) == {"constants.n": 0.5}


def test_an_undeclared_large_integer_is_not_rounded_into_a_choice(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    big = 2**53
    doc = yaml.safe_load(problem_file.read_text())
    doc["parameters"] = [
        {
            "geometry": {"constant": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": [big, big + 2],
        }
    ]
    problem_file.write_text(yaml.safe_dump(doc))
    for text in (str(big + 1), f"{big + 1}.0", "9.007199254740993e15"):
        with pytest.raises(CandidateError, match="not one of"):
            run_candidate(problem_file, [f"{RADIUS}={text}"], tmp_path / "a")
    assert not (tmp_path / "a").exists()
    assert fake.calls == []


def test_a_value_that_only_rounds_to_a_choice_is_rejected(doc: dict[str, Any]) -> None:
    doc["parameters"][0] = {
        "geometry": {"constant": "n"},
        "type": "categorical",
        "choices": [1, 2],
    }
    spec = parse_problem(doc)
    for text in ("1.0000000000000001", "2.0000000000000001", "1.5"):
        with pytest.raises(CandidateError, match="not one of"):
            _parse_candidate(spec, [f"constants.n={text}"])
    assert _parse_candidate(spec, ["constants.n=1.0"]) == {"constants.n": 1}


def test_a_decimal_that_rounds_to_a_float_choice_is_rejected(doc: dict[str, Any]) -> None:
    doc["parameters"][0] = {
        "geometry": {"constant": "n"},
        "type": "categorical",
        "choices": [2.0**53, 0.1],
    }
    spec = parse_problem(doc)
    with pytest.raises(CandidateError, match="not one of"):
        _parse_candidate(spec, [f"constants.n={2**53 + 1}"])
    assert _parse_candidate(spec, ["constants.n=0.1"]) == {"constants.n": 0.1}
    assert _parse_candidate(spec, [f"constants.n={2**53}"]) == {"constants.n": 2.0**53}


@pytest.mark.parametrize("text", [str(2**53 + 1), str(10**400), f"-{2**53 + 1}"])
def test_integer_literals_of_a_continuous_parameter_stay_exact(
    doc: dict[str, Any], text: str
) -> None:
    doc["parameters"][0]["bounds"] = [-(10**401), 10**401]
    spec = parse_problem(doc)
    (value,) = _parse_candidate(spec, [f"constants.ECal_cell_size={text}"]).values()
    assert value == int(text)
    assert isinstance(value, int)  # not rounded through float, no overflow to infinity
    assert _parse_candidate(spec, ["constants.ECal_cell_size=6"]) == {"constants.ECal_cell_size": 6}
    (small,) = _parse_candidate(spec, ["constants.ECal_cell_size=6"]).values()
    assert isinstance(small, float)  # ordinary values stay floats: "6.0*mm"


@pytest.mark.parametrize("kind", ["continuous", "integer"])
def test_an_integer_with_more_digits_than_python_converts_is_a_candidate_error(
    doc: dict[str, Any], kind: str
) -> None:
    doc["parameters"][0] = {"geometry": {"constant": "n"}, "type": kind, "bounds": [0, 10]}
    text = "1" * (sys.get_int_max_str_digits() + 1)
    with pytest.raises(CandidateError, match="too many digits"):
        _parse_candidate(parse_problem(doc), [f"constants.n={text}"])


def test_a_home_relative_problem_path_is_recorded_expanded(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(problem_file.parent))
    outcome = run_candidate(f"~/{problem_file.name}", [f"{RADIUS}=450"], tmp_path / "a")
    problem = outcome.summary["problem"]
    assert isinstance(problem, dict)
    assert problem["path"] == str(problem_file)
    assert Path(problem["path"]).is_file()


def test_an_output_inside_the_geometry_source_creates_nothing(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "k4geo"
    shutil.copytree(MINI_GEO, source)
    monkeypatch.setenv("K4GEO", str(source))
    before = tree_digest(source)
    with pytest.raises(K4OptError):
        run_candidate(problem_file, [f"{RADIUS}=450"], source / "new-parent" / "run")
    assert not (source / "new-parent").exists()
    assert tree_digest(source) == before
    assert fake.calls == []


def test_a_failed_simulation_is_summarized_with_its_logs(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2, stderr="G4Exception: bad volume")
    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    summary = json.loads(outcome.summary_path.read_text())
    assert summary["status"] == "failed"
    # the stage that failed, then what it kept from running
    assert summary["message"].startswith("gamma/simulation failed: ddsim exited with status 2")
    assert summary["message"].endswith("G4Exception: bad volume; blocked: gamma/test.cell_count")
    (evaluator,) = summary["samples"]["gamma"]["evaluations"].values()
    blocked = ("blocked", "not run: gamma/simulation failed")
    assert (evaluator["status"], evaluator["message"]) == blocked
    assert evaluator["blocked_by"] == ["gamma/simulation"]
    assert evaluator["measurements"] == [] and evaluator["elapsed_seconds"] is None
    assert evaluator["workdir"] is None  # not called, so no directory was made for it
    assert summary["metrics"][0]["value"] is None  # never a made-up number
    assert summary["samples"]["gamma"]["stages"]["simulation"]["returncode"] == 2
    assert summary["samples"]["gamma"]["stages"]["simulation"]["status"] == "failed"
    assert "bad volume" in summary["samples"]["gamma"]["stages"]["simulation"]["message"]
    failed = summary["samples"]["gamma"]["stages"]["simulation"]
    assert failed["outputs"] == {}  # what ddsim left behind is no output
    assert "bad volume" in (tmp_path / "a" / failed["logs"]["stderr"]).read_text()


@pytest.mark.parametrize(
    ("parameter", "value", "expected"),
    [
        ({"type": "continuous", "unit": "mm"}, 6.0, "6.0*mm"),
        ({"type": "continuous", "unit": "deg"}, 12, "12*deg"),
        ({"type": "continuous"}, 0.5, "0.5"),
        ({"type": "integer"}, 90, "90"),
        ({"type": "categorical", "choices": ["a", "b"]}, "b", "b"),
        ({"type": "categorical", "choices": [True, False]}, True, "true"),
    ],
)
def test_values_become_dd4hep_literals(
    doc: dict[str, Any], parameter: dict[str, Any], value: object, expected: str
) -> None:
    doc["parameters"][0] = {"geometry": {"constant": "x"}, **parameter}
    if "bounds" not in parameter and "choices" not in parameter:
        doc["parameters"][0]["bounds"] = [0, 100]
    spec = parse_problem(doc)
    assert _xml_value(spec.parameters[0], value) == expected  # type: ignore[arg-type]


def test_an_unsupported_unit_is_refused(doc: dict[str, Any]) -> None:
    doc["parameters"][0]["unit"] = "furlong"
    with pytest.raises(CandidateError, match="furlong"):
        _xml_value(parse_problem(doc).parameters[0], 6.0)


def test_text_values_are_read_by_the_parameter_kind(
    problem_file: Path, doc: dict[str, Any]
) -> None:
    spec = load_problem(problem_file)
    assert _parse_candidate(spec, [f"{RADIUS}=6"]) == {RADIUS: 6.0}
    assert _parse_candidate(spec, [f"{RADIUS}=1e2"]) == {RADIUS: 100.0}
    assert _parse_candidate(spec, [f"{RADIUS}=-.5"]) == {RADIUS: -0.5}
    doc["parameters"][0] = {"geometry": {"constant": "n"}, "type": "integer", "bounds": [1, 9]}
    integer = parse_problem(doc)
    assert _parse_candidate(integer, ["constants.n=4"]) == {"constants.n": 4}
    with pytest.raises(CandidateError, match="integer"):
        _parse_candidate(integer, ["constants.n=4.0"])


def test_the_example_problem_is_valid_and_matches_the_example_evaluator() -> None:
    problem = parse_problem(yaml.safe_load((EXAMPLES / "cld-ecal-granularity.yaml").read_text()))
    evaluator = calo_photon()()
    (entry,) = problem.evaluators
    assert (entry.name, entry.version, entry.stage) == (
        evaluator.name,
        evaluator.version,
        evaluator.stage,
    )
    # the objectives leave the one evaluator out, and name it once read
    assert [o.label for o in problem.objectives] == [
        "gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual",
        "gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells",
    ]
    assert metric_units(evaluator) == {
        ("simulation", "ecal_barrel.angular_residual"): "mrad",
        ("simulation", "ecal_barrel.fired_cells"): "1",
    }
    # the request the framework builds: the threshold in GeV, the hits and the default particles
    request = EvaluationRequest(
        settings=with_defaults(evaluator, entry.settings),
        collections=collection_names(evaluator, entry.collections),
    )
    evaluator.check(request)
    assert request.settings == {"threshold": 70e-6}
    assert request.collections == {"hits": "ECalBarrelCollection", "particles": "MCParticles"}


# --- the command ---------------------------------------------------------------------------


def errors(err: str) -> str:
    """``err`` without the warnings printed before the run."""
    return "".join(line for line in err.splitlines(True) if not line.startswith("k4opt: warning:"))


def evaluate(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(["evaluate", "--evaluator", CELL_COUNT, *argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_command_prints_the_result_and_exits_zero(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 0, err
    assert "status:      valid" in out
    assert "evaluator:   test.cell_count 1 on gamma: valid" in out
    assert "simulation/ecal_barrel.mean_hit_cells = 3 +- 0.57735 [1]" in out
    assert "feasible:    yes" in out and "limit:" not in out  # the problem has no constraints
    assert "events:      gamma: 3 of 3" in out
    assert "fingerprint: " in out
    assert f"summary:     {tmp_path / 'a' / SUMMARY_NAME}" in out
    assert "ddsim.stderr.log" in out
    assert err.startswith(f"k4opt: warning: {UNEXPECTED}")  # printed before anything ran


def test_command_exits_nonzero_for_a_failed_run(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=1, stderr="boom")
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 1
    assert "status:      failed" in out
    assert "feasible:    not assessed" in out
    assert "events:" not in out  # ddsim wrote no file, so no events were counted
    assert "k4opt: evaluation failed: gamma/simulation failed: ddsim exited with status 1" in err
    assert "evaluator:   test.cell_count 1 on gamma: blocked" in out


def test_command_exits_nonzero_for_an_invalid_run(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, events_with([], [], []))
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 1
    assert "status:      invalid" in out
    assert "metric:" not in out
    assert "no hits" in err


def test_command_reports_a_bad_candidate_without_creating_anything(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=9999", "--output", str(tmp_path / "a")
    )
    assert code == 1
    assert out == ""
    assert err.startswith("k4opt: error: ")
    assert "outside the bounds" in err
    assert not (tmp_path / "a").exists()


def test_command_requires_an_output_directory(problem_file: Path) -> None:
    with pytest.raises(SystemExit) as info:
        evaluate(str(problem_file), "--param", f"{RADIUS}=450")
    assert info.value.code == 2


def test_a_generation_file_is_simulated_and_its_hash_recorded(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    hepmc = tmp_path / "zh.hepmc"
    hepmc.write_bytes(b"HepMC::Version 3\n")
    doc = yaml.safe_load(problem_file.read_text(encoding="utf-8"))
    doc["samples"]["gamma"]["generation"] = {"file": "zh.hepmc"}  # next to the problem file
    problem_file.write_text(yaml.safe_dump(doc), encoding="utf-8")

    outcome = run_candidate(problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "s")

    ((argv, _),) = fake.calls
    assert f"--inputFiles={hepmc}" in argv and not any(a.startswith("--gun") for a in argv)
    generation = outcome.summary["samples"]["gamma"]["stages"]["generation"]  # type: ignore[index]
    assert generation["file"] == str(hepmc)
    assert generation["sha256"] == hashlib.sha256(hepmc.read_bytes()).hexdigest()
    assert outcome.candidate.status == "valid" and outcome.candidate.represents_candidate


def test_a_missing_generation_file_is_a_clean_error_before_any_side_effect(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text(encoding="utf-8"))
    doc["samples"]["gamma"]["generation"] = {"file": "gone.hepmc"}
    problem_file.write_text(yaml.safe_dump(doc), encoding="utf-8")
    output = tmp_path / "runs" / "s"
    with pytest.raises(EvaluationContractError, match=r"samples\.gamma\.generation\.file"):
        run_candidate(problem_file, [f"{RADIUS}=450.0"], output)
    assert not output.exists() and fake.calls == []


# --- stage files supplied on the command line ----------------------------------------------


def test_a_supplied_simulation_is_evaluated_but_never_counts_as_the_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    events_file = tmp_path / "earlier.edm4hep.root"
    events_file.write_bytes(b"fake")
    code, out, err = evaluate(
        str(problem_file),
        "--param",
        f"{RADIUS}=450",
        "--input",
        f"gamma/simulation={events_file}",
        "--output",
        str(tmp_path / "a"),
    )
    assert code == 0, err
    assert fake.calls == []  # nothing was simulated
    assert f"supplied:    gamma/simulation from {events_file}" in out
    assert "this is no evaluation of the candidate" in out
    assert "feasible:    not assessed" in out  # whatever it measured
    summary = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    supplied = (False, ["gamma/simulation"])
    assert (summary["represents_candidate"], summary["supplied"]) == supplied
    assert summary["samples"]["gamma"]["stages"] == {
        "simulation": {
            "status": "done",
            "message": None,
            "provided": str(events_file),
            "bytes": 4,
            "sha256": hashlib.sha256(b"fake").hexdigest(),
            "events_read": 3,
            # it ran nowhere and made nothing: the file is the caller's, by its path
            "directory": None,
            "outputs": {},
            "logs": {},
            "reads": [],  # it is given, so it reads nothing
            "blocked_by": [],
            "elapsed_seconds": summary["samples"]["gamma"]["stages"]["simulation"][
                "elapsed_seconds"
            ],
        }
    }
    assert summary["metrics"][0]["value"] == 3.0


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ("gamma/simulation", "expected SAMPLE/STAGE=FILE"),
        ("gamma/reconstruction={file}", "unknown stage; use generation, simulation"),
        ("gamma/simulation={missing}", "is not a file"),
        ("gamma/generation={file}", "no evaluator of the problem reads that stage of gamma"),
        # a stage of which sample: the problem says, even with only one
        (
            "simulation={file}",
            "input 'simulation': name the sample too, as SAMPLE/simulation; the problem's "
            "samples are gamma",
        ),
        ("pion/simulation={file}", "unknown sample 'pion'; the samples are gamma"),
    ],
)
def test_unusable_inputs_are_rejected_before_any_side_effect(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
    given: str,
    message: str,
) -> None:
    supplied = tmp_path / "mine.root"
    supplied.write_bytes(b"fake")
    argument = given.format(file=supplied, missing=tmp_path / "gone.root")
    if message.startswith("no evaluator"):
        # the evaluator reads the simulation, which the caller supplies too: generation is unused
        argument = [argument, f"gamma/simulation={supplied}"]
    output = tmp_path / "a"
    with pytest.raises(RunError, match=message):
        run_candidate(
            problem_file,
            [f"{RADIUS}=450"],
            output,
            inputs=argument if isinstance(argument, list) else [argument],
        )
    assert not output.exists() and fake.calls == [] and "materialize" not in guard


def test_an_input_given_twice_is_rejected(tmp_path: Path, problem_file: Path) -> None:
    supplied = tmp_path / "mine.root"
    supplied.write_bytes(b"fake")
    with pytest.raises(RunError, match="given more than once"):
        run_candidate(
            problem_file,
            [f"{RADIUS}=450"],
            tmp_path / "a",
            inputs=[f"gamma/simulation={supplied}", f"gamma/simulation={supplied}"],
        )


def test_the_command_needs_an_evaluator(problem_file: Path, tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(SystemExit):
        main(
            ["evaluate", str(problem_file), "--output", str(tmp_path / "o")], stdout=out, stderr=err
        )
    assert not (tmp_path / "o").exists()


def test_the_command_reports_an_unloadable_evaluator_without_creating_anything(
    problem_file: Path, tmp_path: Path
) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        [
            "evaluate",
            str(problem_file),
            "--evaluator",
            "gone.py:Nope",
            "--output",
            str(tmp_path / "o"),
        ],
        stdout=out,
        stderr=err,
    )
    assert code == 1 and "does not exist" in err.getvalue()
    assert not (tmp_path / "o").exists()


class Spy:
    """The test evaluator, recording the requests of check() and evaluate()."""

    def __init__(self) -> None:
        self.inner: Any = cell_count()()
        self.name, self.version = self.inner.name, self.inner.version
        self.stage, self.metrics = self.inner.stage, self.inner.metrics
        self.collections = self.inner.collections
        self.checked: list[Any] = []
        self.evaluated: list[Any] = []

    def check(self, request: Any) -> None:
        self.checked.append(request)
        self.inner.check(request)

    def evaluate(self, request: Any) -> Any:
        self.evaluated.append(request)
        return self.inner.evaluate(request)


def test_the_evaluator_provenance_is_saved_in_the_summary(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    class Annotating(Spy):
        def evaluate(self, request: Any) -> Any:
            result = result_from(self.inner, self.inner.evaluate(request))
            return dataclasses.replace(result, provenance={"selection": "barrel", "algo": [2, 1]})

    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "a", evaluators=[Annotating()]
    )
    on_disk = json.loads(outcome.summary_path.read_text())
    (evaluator,) = on_disk["samples"]["gamma"]["evaluations"].values()
    assert evaluator["provenance"] == {"selection": "barrel", "algo": [2, 1]}
    assert (
        on_disk["samples"]["gamma"]["stages"]["simulation"]["returncode"] == 0
    )  # the stage's record stays apart


def test_the_preflight_request_has_the_conditions_the_evaluation_gets(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    spy = Spy()
    run_candidate(problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "a", evaluators=[spy])
    (evaluated,) = spy.evaluated
    # The validated request is reused after materialization; check() runs exactly once.
    assert len(spy.checked) == 1
    for checked in spy.checked:
        assert (checked.events, checked.seed) == (3, 5)  # those of the simulation it reads
        for field in ("settings", "events", "seed", "candidate"):
            assert getattr(checked, field) == getattr(evaluated, field)


@pytest.mark.parametrize("invalid", ["settings", "collections"])
def test_a_plain_evaluator_is_validated_before_materialization(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, invalid: str
) -> None:
    class Plain:
        name = "test.cell_count"
        version = "1"
        stage = "simulation"
        collections = {"hits": cell_count().collections["hits"]}
        metrics = cell_count().metrics
        settings = {"threshold": Quantity("GeV", default=1)}

        def evaluate(self, request: Any) -> Any:
            pytest.fail("invalid requests must be rejected before execution")

    doc = yaml.safe_load(problem_file.read_text())
    if invalid == "settings":
        doc["evaluators"][0]["settings"] = {"threshold": {"value": 1, "unit": "bananas"}}
    else:
        doc["evaluators"][0]["collections"] = {}
    problem_file.write_text(yaml.safe_dump(doc))
    supplied = tmp_path / "events.root"
    supplied.write_bytes(b"fake")
    with pytest.raises(EvaluationContractError, match=invalid):
        run_candidate(
            problem_file,
            [f"{RADIUS}=450"],
            tmp_path / "a",
            evaluators=[Plain()],
            inputs=[f"gamma/simulation={supplied}"],
        )
    assert not (tmp_path / "a").exists() and fake.calls == []


def test_summary_preserves_the_definition_loaded_before_execution(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    definition = load_problem(problem_file).to_dict()

    class Editing(Spy):
        def evaluate(self, request: Any) -> Any:
            problem_file.write_text("replaced while evaluating")
            return super().evaluate(request)

    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[Editing()])
    assert outcome.summary["problem"]["definition"] == definition
    assert json.loads(outcome.summary_path.read_text())["problem"]["definition"] == definition


def test_summary_write_failure_is_a_controlled_error(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from k4opt.problem import _run

    def denied(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("summary is not writable")

    monkeypatch.setattr(_run, "write_json", denied)
    with pytest.raises(RunError, match="cannot write summary"):
        run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert len(fake.calls) == 1
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()


def test_the_candidate_time_counts_the_checks_and_every_task_the_geometry_too(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from k4opt.problem import _evaluate, _run

    class Clock:
        now = 0.0

        def perf_counter(self) -> float:
            return self.now

    clock = Clock()
    for module in (_evaluate, _run):
        monkeypatch.setattr(module, "time", clock)
    materialize = GeometrySource.materialize

    def slow_materialize(self: GeometrySource, *args: Any, **kwargs: Any) -> Any:
        clock.now += 100.0
        return materialize(self, *args, **kwargs)

    class Checking(Spy):
        def check(self, request: Any) -> None:
            clock.now += 0.5
            super().check(request)

    monkeypatch.setattr(GeometrySource, "materialize", slow_materialize)
    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[Checking()]
    )
    assert outcome.summary["elapsed_seconds"] == 100.5
    assert outcome.summary["geometry"]["elapsed_seconds"] == 100.0  # type: ignore[index]


def test_an_unwritable_evaluator_directory_is_a_recorded_failure(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mkdir = Path.mkdir

    def denied(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name == "test.cell_count":
            raise PermissionError("evaluator directory is not writable")
        mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert outcome.candidate.status == "failed"
    assert "cannot create evaluator directory" in outcome.candidate.message
    assert outcome.summary_path.is_file()


# --- several evaluators of one candidate ---------------------------------------------------

CELLS = ("simulation", "ecal_barrel.mean_hit_cells")
ENERGY = ("simulation", "ecal_barrel.mean_hit_energy")
ECHO = ("analysis", "candidate.first_value")
OWNERS = {CELLS: "test.cell_count", ENERGY: "test.hit_energy", ECHO: "test.parameter_echo"}


def declare(problem_file: Path, names: list[str], objectives: list[tuple[str, str]]) -> None:
    """Make the problem declare the test evaluators ``names`` and these objectives, each on the
    metric of the evaluator that measures it (:data:`OWNERS`)."""
    doc = yaml.safe_load(problem_file.read_text())
    reads = {"stage": "simulation", "collections": {"hits": COLLECTION}}
    doc["evaluators"] = [
        {"name": name, "version": "1"} | ({} if name == "test.parameter_echo" else reads)
        for name in names
    ]
    doc["objectives"] = [
        {"evaluator": OWNERS[(s, m)], "stage": s, "metric": m, "direction": "minimize"}
        for s, m in objectives
    ]
    problem_file.write_text(yaml.safe_dump(doc))


def evaluate_with(references: list[str], *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    flags = [arg for reference in references for arg in ("--evaluator", reference)]
    code = main(["evaluate", *flags, *argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_the_command_measures_with_several_evaluators_on_one_simulation(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    declare(problem_file, ["test.cell_count", "test.hit_energy"], [CELLS, ENERGY])
    output = tmp_path / "a"
    # in any order on the command line; reported in the order of the problem
    code, out, err = evaluate_with(
        [HIT_ENERGY, CELL_COUNT],
        str(problem_file),
        "--param",
        f"{RADIUS}=450",
        "--output",
        str(output),
    )
    assert code == 0, err
    assert len(fake.calls) == 1  # one simulation for both
    assert sorted(p.name for p in output.iterdir()) == ["geometry", "samples", SUMMARY_NAME]
    assert sorted(p.name for p in (output / "samples" / "gamma").iterdir()) == [
        "evaluators",
        "simulation",
    ]
    assert sorted(p.name for p in (output / "samples" / "gamma" / "evaluators").iterdir()) == [
        "test.cell_count",
        "test.hit_energy",
    ]
    assert out.index("evaluator:   test.cell_count 1 on gamma: valid") < out.index(
        "evaluator:   test.hit_energy 1 on gamma: valid"
    )
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["status"] == "valid"
    evaluations = summary["samples"]["gamma"]["evaluations"]
    assert list(evaluations) == ["test.cell_count", "test.hit_energy"]
    assert [m["label"] for m in summary["metrics"]] == [
        "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
        "gamma/test.hit_energy/simulation/ecal_barrel.mean_hit_energy",
    ]
    assert summary["metrics"][1]["value"] == pytest.approx(10 / 3)  # 4, 2 and 4 hits of 1 GeV
    assert list(summary["samples"]["gamma"]["stages"]) == ["generation", "simulation"]


def test_the_command_reports_each_binding_of_an_implementation_given_once(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluators"].append(doc["evaluators"][0] | {"as": "cells_again"})
    doc["objectives"][0]["evaluator"] = "test.cell_count"
    doc["objectives"].append(doc["objectives"][0] | {"evaluator": "cells_again"})
    problem_file.write_text(yaml.safe_dump(doc))
    output = tmp_path / "a"
    code, out, err = evaluate_with(
        [CELL_COUNT], str(problem_file), "--param", f"{RADIUS}=450", "--output", str(output)
    )
    assert code == 0, err
    assert len(fake.calls) == 1
    assert "evaluator:   test.cell_count 1 on gamma: valid" in out
    assert "evaluator:   cells_again (test.cell_count 1) on gamma: valid" in out
    # two bindings that measure the same twice are worth a word, not a refusal
    assert "evaluators 'test.cell_count' and 'cells_again' bind test.cell_count" in err
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert list(summary["samples"]["gamma"]["evaluations"]) == ["test.cell_count", "cells_again"]
    assert sorted(p.name for p in (output / "samples" / "gamma" / "evaluators").iterdir()) == [
        "cells_again",
        "test.cell_count",
    ]


def test_a_failed_simulation_keeps_the_result_of_an_evaluator_that_needs_none(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2, stderr="G4Exception: bad volume")
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])
    output = tmp_path / "a"
    code, out, err = evaluate_with(
        [CELL_COUNT, PARAMETER_ECHO],
        str(problem_file),
        "--param",
        f"{RADIUS}=450",
        "--output",
        str(output),
    )
    assert code == 1
    assert errors(err).startswith("k4opt: evaluation failed: gamma/simulation failed: ddsim exited")
    assert "evaluator:   test.parameter_echo 1: valid" in out
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["status"] == "failed"
    (cells,) = summary["samples"]["gamma"]["evaluations"].values()
    (echo,) = summary["evaluations"].values()
    assert (cells["status"], cells["elapsed_seconds"], cells["workdir"]) == ("blocked", None, None)
    assert echo["status"] == "valid" and echo["measurements"][0]["value"] == 450.0
    # the measurement that was made is kept; the missing one is missing, not made up
    assert [m["value"] for m in summary["metrics"]] == [None, 450.0]
    assert summary["samples"]["gamma"]["stages"]["simulation"]["message"].startswith(
        "ddsim exited with status 2"
    )


@pytest.mark.parametrize(
    ("declared", "given", "message"),
    [
        (
            ["test.cell_count", "test.hit_energy"],
            [CELL_COUNT],
            "no evaluator given for 'test.hit_energy' version '1'",
        ),
        (["test.cell_count"], [CELL_COUNT, HIT_ENERGY], "declares no evaluator 'test.hit_energy'"),
        (["test.cell_count"], [CELL_COUNT, CELL_COUNT], "'test.cell_count' is given twice"),
    ],
)
def test_evaluators_that_do_not_fit_the_problem_are_refused_before_any_side_effect(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
    declared: list[str],
    given: list[str],
    message: str,
) -> None:
    declare(problem_file, declared, [CELLS])
    output = tmp_path / "a"
    code, out, err = evaluate_with(
        given, str(problem_file), "--param", f"{RADIUS}=450", "--output", str(output)
    )
    assert (code, out) == (1, "")
    assert err.startswith("k4opt: error: ") and message in err
    assert not output.exists() and fake.calls == [] and "materialize" not in guard


def test_the_summary_is_the_same_for_the_same_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    declare(
        problem_file,
        ["test.cell_count", "test.hit_energy", "test.parameter_echo"],
        [CELLS, ENERGY, ECHO],
    )
    first = run_candidate(
        problem_file,
        [f"{RADIUS}=450"],
        tmp_path / "a",
        evaluators=[cell_count(), hit_energy(), parameter_echo()],
    )
    second = run_candidate(
        problem_file,
        [f"{RADIUS}=450"],
        tmp_path / "b",
        evaluators=[parameter_echo(), hit_energy(), cell_count()],
    )

    def normalized(outcome: RunOutcome) -> str:
        """The summary as written, without its directory, its times and its own id."""
        data = json.loads(outcome.summary_path.read_text().replace(str(outcome.run_dir), "<run>"))
        del data["id"]

        def untimed(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: 0 if k == "elapsed_seconds" else untimed(v) for k, v in node.items()}
            return [untimed(item) for item in node] if isinstance(node, list) else node

        return json.dumps(untimed(data))  # key order included

    assert normalized(first) == normalized(second)
    assert first.summary["id"] != second.summary["id"]  # two records, each its own identity


def test_a_misspelled_collection_warns_before_the_run_and_fails_after_it(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluators"][0]["collections"] = {"hits": "InnerTrackerBarelCollection"}
    problem_file.write_text(yaml.safe_dump(doc))
    code, _, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    # the geometry suggests the readout meant, but only the simulation's output decides
    assert err.startswith(
        "k4opt: warning: evaluator 'test.cell_count': ddsim is not expected to write the "
        "collection 'InnerTrackerBarelCollection' (hits) for "
        f"{MINI_COMPACT}; did you mean 'InnerTrackerBarrelCollection'?"
    )
    assert code == 1 and len(fake.calls) == 1
    summary = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    (evaluator,) = summary["samples"]["gamma"]["evaluations"].values()
    assert (evaluator["status"], evaluator["elapsed_seconds"]) == ("failed", None)
    assert evaluator["message"] == (
        "collection 'InnerTrackerBarelCollection' (hits) is not in events.edm4hep.root, which "
        f"holds: {COLLECTION}"
    )
    assert len(summary["warnings"]) == 1


def test_a_setting_with_a_unit_is_recorded_as_written_and_as_received(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluators"][0]["settings"] = {"threshold": {"value": 70, "unit": "keV"}}
    problem_file.write_text(yaml.safe_dump(doc))
    gated = type("Gated", (cell_count(),), {"settings": {"threshold": Quantity("GeV")}})
    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[gated])
    on_disk = json.loads(outcome.summary_path.read_text())
    (evaluator,) = on_disk["samples"]["gamma"]["evaluations"].values()
    assert evaluator["settings"] == {
        "declared": {"threshold": {"value": 70, "unit": "keV"}},
        "used": {"threshold": 70e-6},
    }


# --- one session, many candidates ----------------------------------------------------------


def candidates(requests: list[Any]) -> list[object]:
    """The radius each request was made for."""
    return [request.candidate[RADIUS] for request in requests]


def test_a_session_checks_and_measures_each_candidate_with_a_request_of_its_own(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    spy = Spy()
    session = EvaluationSession.open(problem_file, implementations(spy))
    session.check({RADIUS: 300.0})  # checks, and creates and runs nothing
    first = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    second = session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    assert candidates(spy.checked) == [300.0, 450.0, 500.0]  # never the request of another
    assert candidates(spy.evaluated) == [450.0, 500.0]
    assert guard == ["discover", "materialize", "materialize"]  # the geometry discovered once
    assert len(fake.calls) == 2 and sorted(p.name for p in tmp_path.iterdir()) == [
        "a",
        "b",
        "problem.yaml",
    ]
    assert first.summary["candidate"] == {RADIUS: 450.0}
    assert second.summary["candidate"] == {RADIUS: 500.0}
    assert first.fingerprint != second.fingerprint
    assert first.summary["id"] != second.summary["id"]


def test_a_check_that_refuses_one_candidate_refuses_only_that_one(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    class Picky(Spy):
        def check(self, request: Any) -> None:
            super().check(request)
            if request.candidate[RADIUS] == 500.0:
                raise EvaluationContractError("this evaluator cannot measure a radius of 500 mm")

    session = EvaluationSession.open(problem_file, [Picky])
    with pytest.raises(EvaluationContractError, match="500 mm"):
        session.evaluate({RADIUS: 500.0}, tmp_path / "a")
    assert not (tmp_path / "a").exists() and fake.calls == [] and "materialize" not in guard
    assert session.evaluate({RADIUS: 450.0}, tmp_path / "b").candidate.status == "valid"


# --- an evaluator class, and a new instance of it for every candidate ----------------------


def test_every_candidate_gets_new_instances_of_its_evaluators(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    created: list[Spy] = []

    class Counted(Spy):
        def __init__(self) -> None:
            super().__init__()
            created.append(self)

    session = EvaluationSession.open(problem_file, [Counted])
    session.check({RADIUS: 300.0})
    session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    # one instance binds the class when the session opens and is asked nothing; then each
    # candidate has an instance of its own, so nothing one keeps reaches another candidate
    probe, *instances = created
    assert (probe.checked, probe.evaluated) == ([], [])
    assert [(candidates(i.checked), candidates(i.evaluated)) for i in instances] == [
        ([300.0], []),
        ([450.0], [450.0]),
        ([500.0], [500.0]),
    ]


def test_an_evaluator_that_cannot_be_created_stops_before_anything_is_created(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    class Broken(Spy):
        def __init__(self) -> None:
            raise RuntimeError("no model file")

    with pytest.raises(EvaluationContractError, match=r"Broken\(\) failed: RuntimeError: no model"):
        EvaluationSession.open(problem_file, [Broken])
    created: list[Spy] = []

    class Once(Spy):  # a constructor that fails later, for a candidate
        def __init__(self) -> None:
            if created:
                raise RuntimeError("one instance per process")
            super().__init__()
            created.append(self)

    session = EvaluationSession.open(problem_file, [Once])
    with pytest.raises(EvaluationContractError, match="one instance per process"):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert not (tmp_path / "a").exists() and fake.calls == [] and "materialize" not in guard


@pytest.mark.parametrize(
    ("field", "later", "message"),
    [
        ("name", "test.other", "name 'test.other', not 'test.cell_count'"),
        ("version", "2", "version '2', not '1'"),
        ("metrics", {"other": "1"}, "metrics .*'other'"),
        # the defaults and collections the request is made with: another trial would be
        # measured otherwise under the same problem and study
        ("settings", {"threshold": Quantity("GeV", default=1.0)}, "settings .*'threshold'"),
        ("collections", {"hits": Collection("SimTrackerHit")}, "collections .*SimTrackerHit"),
        # what its results are checked against
        ("artifacts", {"other": "a file"}, "artifacts .*'other'"),
    ],
)
def test_a_new_instance_must_declare_what_the_first_did(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
    field: str,
    later: Any,
    message: str,
) -> None:
    created: list[Spy] = []

    class Drifting(Spy):
        def __init__(self) -> None:
            super().__init__()
            if created:  # the record names what the first instance declared
                setattr(self, field, later)
            created.append(self)

    session = EvaluationSession.open(problem_file, [Drifting])
    with pytest.raises(
        EvaluationContractError, match=rf"a new instance of .*Drifting has {message}"
    ):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert not (tmp_path / "a").exists() and fake.calls == [] and "materialize" not in guard


def test_only_an_evaluator_class_is_accepted(problem_file: Path) -> None:
    def build() -> Any:  # could set up the instance in ways the class's source does not say
        return cell_count()()

    for given, kind in ((cell_count()(), "CellCount"), (build, "function")):
        with pytest.raises(
            EvaluationContractError, match=f"got a {kind} object, not an evaluator class"
        ):
            EvaluationSession.open(problem_file, [given])


def test_a_nested_default_an_evaluator_changes_stops_the_next_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    class Mutating(Spy):
        settings = {"cuts": {"min": 1}}  # a default: the problem sets no settings

        def evaluate(self, request: Any) -> Any:
            assert request.settings == {"cuts": {"min": 1}}
            self.settings["cuts"]["min"] = 2  # the class's own dict, shared by every instance
            return super().evaluate(request)

    session = EvaluationSession.open(problem_file, [Mutating])
    assert session.evaluate({RADIUS: 450.0}, tmp_path / "a").candidate.status == "valid"
    with pytest.raises(EvaluationContractError, match=r"settings \{'cuts': \{'min': 2\}\}"):
        session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    assert not (tmp_path / "b").exists() and len(fake.calls) == 1


def test_a_session_keeps_the_problem_it_loaded(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    definition = session.spec.to_dict()
    doc = yaml.safe_load(problem_file.read_text())
    doc["parameters"][0]["bounds"] = [100.0, 200.0]  # an edit during a campaign
    problem_file.write_text(yaml.safe_dump(doc))
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.summary["problem"]["definition"] == definition  # type: ignore[index]
    assert session.problem_path == problem_file.absolute()


@pytest.fixture
def named_files(tmp_path: Path, problem_file: Path) -> tuple[Path, Path]:
    """The problem reads a generation file and a file a ddsim option names."""
    hepmc, table = tmp_path / "zh.hepmc", tmp_path / "particle.tbl"
    hepmc.write_bytes(b"HepMC::Version 3\n")
    table.write_bytes(b"22 gamma\n")
    doc = yaml.safe_load(problem_file.read_text())
    doc["samples"]["gamma"]["generation"] = {"file": hepmc.name}  # next to the problem file
    doc["samples"]["gamma"]["simulation"]["physics.pdgfile"] = table.name
    problem_file.write_text(yaml.safe_dump(doc))
    return hepmc, table


def test_the_identity_of_a_session_is_what_it_evaluates_with(
    problem_file: Path, named_files: tuple[Path, Path]
) -> None:
    hepmc, table = named_files
    session = EvaluationSession.open(problem_file, [cell_count()])
    identity = session.identity()
    assert identity["problem"] == {
        "path": str(problem_file.absolute()),
        "definition": session.spec.to_dict(),
    }
    assert identity["inputs"] == {
        "samples.gamma.generation.file": {
            "path": str(hepmc),
            "bytes": hepmc.stat().st_size,
            "sha256": hashlib.sha256(hepmc.read_bytes()).hexdigest(),
        },
        "samples.gamma.simulation.physics.pdgfile": {
            "path": str(table),
            "bytes": table.stat().st_size,
            "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        },
    }
    implementations = identity["implementations"]
    assert isinstance(implementations, dict) and list(implementations) == ["test.cell_count"]
    evaluator = implementations["test.cell_count"]
    assert (evaluator["version"], evaluator["stage"]) == ("1", "simulation")
    assert (
        evaluator["source"]["sha256"]
        == hashlib.sha256(Path(CELL_COUNT.split(":")[0]).read_bytes()).hexdigest()
    )
    baseline = GeometrySource(MINI_COMPACT).fingerprint([])
    assert identity["geometry"] == {
        "compact": MINI_COMPACT,
        "source_root": str(MINI_GEO.resolve()),
        "fingerprint": baseline,
        "externals": GeometrySource(MINI_COMPACT).external_files(),
    }
    assert identity["environment"]["K4GEO"] == str(MINI_GEO)  # type: ignore[index]
    assert identity["k4opt"]["sha256"] == code_fingerprint()  # type: ignore[index]
    assert identity["k4opt"]["version"] == __version__  # type: ignore[index]
    identity["inputs"].clear()  # type: ignore[attr-defined]
    assert session.identity()["inputs"]  # a fresh copy on every call
    assert json.loads(json.dumps(session.identity())) == session.identity()


@pytest.mark.parametrize("change", ["edit", "remove"])
def test_a_file_that_changes_stops_the_session_before_its_next_candidate(
    tmp_path: Path,
    problem_file: Path,
    named_files: tuple[Path, Path],
    fake: simulation.FakeDdsim,
    change: str,
) -> None:
    hepmc, _ = named_files
    session = EvaluationSession.open(problem_file, [cell_count()])
    session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    if change == "edit":
        hepmc.write_bytes(b"HepMC::Version 3\nE 0\n")  # the same path, other events
        now = "17 bytes"
    else:
        hepmc.unlink()
        now = "unreadable now"
    message = rf"samples\.gamma\.generation\.file: {hepmc} changed after the session opened .*{now}"
    with pytest.raises(RunError, match=message):
        session.verify_inputs()
    with pytest.raises(RunError, match=message):
        session.check({RADIUS: 500.0})
    with pytest.raises(RunError, match=message):
        session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    assert not (tmp_path / "b").exists() and len(fake.calls) == 1


def test_the_code_that_ran_is_identified_before_it_runs(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.problem import _run

    identities = iter(({"sha256": "opened", "git": None}, {"sha256": "edited", "git": None}))
    monkeypatch.setattr(_run, "code_identity", lambda: next(identities))
    session = EvaluationSession.open(problem_file, [cell_count()])
    next(identities)  # the checkout is edited while the session runs
    for name in ("a", "b"):
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / name)
        observed = outcome.summary["environment"]["observed"]  # type: ignore[index]
        assert observed["k4opt_source"] == {"sha256": "opened", "git": None}
    assert session.identity()["k4opt"]["sha256"] == "opened"  # type: ignore[index]
    monkeypatch.setenv("KEY4HEP_STACK", "/another/stack")  # the process changes, the session not
    assert session.identity()["environment"]["KEY4HEP_STACK"] != "/another/stack"  # type: ignore[index]


@pytest.mark.parametrize("name", ["KEY4HEP_STACK", "K4GEO"])
def test_a_session_runs_its_candidates_in_the_environment_it_opened_in(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv("KEY4HEP_STACK", "/stack/2026-10-20")
    session = EvaluationSession.open(problem_file, [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    observed = outcome.summary["environment"]["observed"]  # type: ignore[index]
    identity = session.identity()["environment"]
    assert {k: observed[k] for k in ("KEY4HEP_STACK", "K4GEO")} == identity
    monkeypatch.setenv(name, "/somewhere/else")
    with pytest.raises(RunError, match=rf"{name} changed after the session opened"):
        session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    assert not (tmp_path / "b").exists() and len(fake.calls) == 1


def test_a_session_does_not_open_once_k4opt_changed_on_disk(
    problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt import revision

    code_fingerprint()  # the process identified the code it runs
    monkeypatch.setattr(revision, "source_digest", lambda directory: "edited since")
    with pytest.raises(RunError, match="k4opt's code changed on disk after this process"):
        EvaluationSession.open(problem_file, [cell_count()])


def test_a_session_evaluates_no_candidate_once_k4opt_changed_on_disk(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt import revision

    session = EvaluationSession.open(problem_file, [cell_count()])
    session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    monkeypatch.setattr(revision, "source_digest", lambda directory: "edited since")
    message = "k4opt's code changed on disk after this process"
    with pytest.raises(RunError, match=message):
        session.check({RADIUS: 500.0})
    with pytest.raises(RunError, match=message):
        session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    assert not (tmp_path / "b").exists() and len(fake.calls) == 1


DETECTOR_TYPES = "${DD4hepINSTALL}/DDDetectors/compact/detector_types.xml"  # the mini geometry's


@pytest.fixture
def dd4hep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The file the mini geometry includes through ``${DD4hepINSTALL}``, in the test's own tree."""
    types = tmp_path / "dd4hep" / "DDDetectors" / "compact" / "detector_types.xml"
    types.parent.mkdir(parents=True)
    types.write_text('<define><constant name="DetType_TRACKER" value="0x1"/></define>')
    monkeypatch.setenv("DD4hepINSTALL", str(tmp_path / "dd4hep"))
    return types


def test_a_record_names_what_the_geometry_reads_outside_its_root(
    tmp_path: Path, problem_file: Path, dd4hep: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    read = {"path": str(dd4hep), "sha256": hashlib.sha256(dd4hep.read_bytes()).hexdigest()}
    expected = {DETECTOR_TYPES: read | {"files": {}}}
    assert outcome.summary["geometry"]["externals"] == expected  # type: ignore[index]
    assert session.identity()["geometry"]["externals"] == expected  # type: ignore[index]
    dd4hep.write_text("<define/>")  # between two candidates
    message = r"geometry: \$\{DD4hepINSTALL\}/.*detector_types\.xml does not read the files it read"
    with pytest.raises(RunError, match=message):
        session.check({RADIUS: 500.0})
    assert len(fake.calls) == 1


@pytest.mark.parametrize("change", ["generation file", "external file", "k4opt code"])
def test_what_changes_while_a_candidate_runs_leaves_it_without_a_record(
    tmp_path: Path,
    problem_file: Path,
    named_files: tuple[Path, Path],
    dd4hep: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from k4opt import revision

    hepmc, _ = named_files
    session = EvaluationSession.open(problem_file, [cell_count()])

    def changing(argv: list[str], **kwargs: Any) -> Any:  # while ddsim reads the files
        if change == "generation file":
            hepmc.write_bytes(b"HepMC::Version 3\nE 0\n")
        elif change == "external file":
            dd4hep.write_text("<define/>")
        else:
            monkeypatch.setattr(revision, "source_digest", lambda directory: "edited since")
        return fake(argv, **kwargs)

    monkeypatch.setattr(ddsim_run, "_execute", changing)
    message = {
        "generation file": rf"samples\.gamma\.generation\.file: {hepmc} changed after the session",
        "external file": r"geometry: \$\{DD4hepINSTALL\}/.*detector_types\.xml does not read",
        "k4opt code": "k4opt's code changed on disk",
    }[change]
    with pytest.raises(RunError, match=rf"{message}.*while the candidate in .*/a ran"):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    simulated = tmp_path / "a" / "samples" / "gamma" / "simulation"
    assert simulated.is_dir() and len(fake.calls) == 1
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()
    with pytest.raises(RunError, match=message):  # nor is the next candidate evaluated
        session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    assert not (tmp_path / "b").exists() and len(fake.calls) == 1


SUPPORT = "InnerTrackerSupportCAD.shape.ref"  # cad/support.stl in the mini geometry


@pytest.mark.parametrize("when", ["between candidates", "while it runs"])
def test_a_candidate_records_and_checks_the_external_files_its_values_name(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
    when: str,
) -> None:
    cad = tmp_path / "cad"
    cad.mkdir()
    model = cad / "b.stl"
    model.write_text("solid b")
    monkeypatch.setenv("CAD", str(cad))
    data = yaml.safe_load(problem_file.read_text())
    choices = ["cad/support.stl", "${CAD}/b.stl"]
    data["parameters"].append(
        {"geometry": {"target": SUPPORT}, "type": "categorical", "choices": choices}
    )
    problem_file.write_text(yaml.safe_dump(data))
    session = EvaluationSession.open(problem_file, [cell_count()])
    opened = session.identity()["geometry"]["externals"]  # type: ignore[index]
    assert "${CAD}/b.stl" not in opened

    outcome = session.evaluate({RADIUS: 450.0, SUPPORT: "${CAD}/b.stl"}, tmp_path / "a")
    geometry = outcome.summary["geometry"]
    read = {"path": str(model), "sha256": hashlib.sha256(b"solid b").hexdigest()}
    assert geometry["candidate_externals"] == {"${CAD}/b.stl": read}  # type: ignore[index]
    assert geometry["externals"] == opened  # type: ignore[index]
    unchanged = session.evaluate({RADIUS: 450.0, SUPPORT: "cad/support.stl"}, tmp_path / "b")
    assert unchanged.summary["geometry"]["candidate_externals"] == {}  # type: ignore[index]

    if when == "between candidates":
        model.write_text("solid other")
        message = r"\$\{CAD\}/b\.stl does not read the files it read for an earlier candidate"
    else:

        def changing(argv: list[str], **kwargs: Any) -> Any:  # while ddsim reads the file
            model.write_text("solid other")
            return fake(argv, **kwargs)

        monkeypatch.setattr(ddsim_run, "_execute", changing)
        message = r"\$\{CAD\}/b\.stl does not read the files it read when the candidate started"
    with pytest.raises(RunError, match=rf"geometry: {message}.*which has no {SUMMARY_NAME}"):
        session.evaluate({RADIUS: 500.0, SUPPORT: "${CAD}/b.stl"}, tmp_path / "c")
    assert not (tmp_path / "c" / SUMMARY_NAME).exists()
    assert len(fake.calls) == (2 if when == "between candidates" else 3)


def test_a_simulation_that_runs_out_of_time_fails_its_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stuck(argv: list[str], **kwargs: Any) -> None:
        fake.calls.append((argv, kwargs))
        return None  # killed when its time ran out

    monkeypatch.setattr(ddsim_run, "_execute", stuck)
    session = EvaluationSession.open(problem_file, [cell_count()], timeout=90)
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == "failed"
    assert "ddsim ran longer than 90 s and was killed" in (outcome.candidate.message or "")
    ((_, kwargs),) = fake.calls
    assert kwargs["timeout"] == 90.0
    simulated = outcome.summary["samples"]["gamma"]["stages"]["simulation"]  # type: ignore[index]
    assert (simulated["timeout"], simulated["returncode"]) == (90.0, None)
    assert "ran longer than 90 s" in simulated["message"]
    assert (tmp_path / "a" / "samples" / "gamma" / "simulation" / "ddsim.stderr.log").is_file()


def test_a_session_without_a_timeout_lets_ddsim_run(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    ((_, kwargs),) = fake.calls
    assert kwargs["timeout"] is None
    assert outcome.summary["samples"]["gamma"]["stages"]["simulation"]["timeout"] is None  # type: ignore[index]


@pytest.mark.parametrize("timeout", [0, -5.0, float("nan"), float("inf"), True, "60"])
def test_a_timeout_must_be_a_positive_number_of_seconds(
    problem_file: Path, timeout: object
) -> None:
    with pytest.raises(RunError, match="timeout must be a positive number of seconds"):
        EvaluationSession.open(problem_file, [cell_count()], timeout=timeout)  # type: ignore[arg-type]


def test_command_passes_its_timeout_to_ddsim(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    output = str(tmp_path / "a")
    code, _, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--timeout", "1.5", "--output", output
    )
    assert code == 0, err
    ((_, kwargs),) = fake.calls
    assert kwargs["timeout"] == 1.5
    code, _, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--timeout", "0", "--output", output + "b"
    )
    assert code == 1 and "timeout must be a positive number of seconds, got 0.0" in err


def test_a_run_reference_is_recorded_as_given(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    reference = {"study": "5b0d…", "trial": 7}
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a", run=reference)
    reference["trial"] = 8  # the record keeps its own copy
    assert json.loads(outcome.summary_path.read_text())["run"] == {"study": "5b0d…", "trial": 7}
    for bad, message in (({"when": float("nan")}, "non-finite"), ([1, 2], "must be a mapping")):
        with pytest.raises(RunError, match=message):
            session.evaluate({RADIUS: 450.0}, tmp_path / "b", run=bad)  # type: ignore[arg-type]
        assert not (tmp_path / "b").exists()


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"gamma/reconstruction": "{file}"}, "input 'gamma/reconstruction': unknown stage"),
        ({"gamma/simulation": "{missing}"}, "is not a file"),
        (
            {"gamma/generation": "{file}", "gamma/simulation": "{file}"},
            "input gamma/generation: no evaluator",
        ),
        ({"simulation": "{file}"}, "input 'simulation': name the sample too"),
    ],
)
def test_unusable_inputs_of_a_session_are_refused_when_it_opens(
    tmp_path: Path, problem_file: Path, inputs: dict[str, str], message: str
) -> None:
    supplied = tmp_path / "mine.root"
    supplied.write_bytes(b"fake")
    given = {
        stage: text.format(file=supplied, missing=tmp_path / "gone.root")
        for stage, text in inputs.items()
    }
    with pytest.raises(RunError, match=message):
        EvaluationSession.open(problem_file, [cell_count()], inputs=given)


def constrain(problem_file: Path, **limits: float) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["constraints"] = [{"stage": "simulation", "metric": "ecal_barrel.mean_hit_cells", **limits}]
    problem_file.write_text(yaml.safe_dump(doc))


def test_command_prints_each_limit_and_whether_the_candidate_is_feasible(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    constrain(problem_file, min=1, max=2)  # the candidate fires 3 cells per event
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 0, err  # a valid evaluation, whatever the limits say
    limit = "limit:       gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells"
    assert f"{limit} >= 1: satisfied, margin -2 [1]" in out
    assert f"{limit} <= 2: violated, margin 1 [1]" in out
    assert "feasible:    no" in out
    summary = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    assert [(limit["bound"], limit["satisfied"]) for limit in summary["assessment"]["limits"]] == [
        ("min", True),
        ("max", False),
    ]


def test_command_reports_limits_it_could_not_judge(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=1, stderr="boom")
    constrain(problem_file, max=2)
    code, out, _ = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 1
    limit = "limit:       gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells"
    assert f"{limit} <= 2: not assessed" in out
    assert "feasible:    not assessed" in out


def test_a_supplied_file_that_changes_stops_the_session_before_its_next_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    events_file = tmp_path / "earlier.edm4hep.root"
    events_file.write_bytes(b"fake")
    supplied = {"gamma/simulation": events_file}
    session = EvaluationSession.open(problem_file, [cell_count()], inputs=supplied)
    assert session.identity()["inputs"] == {
        "supplied.gamma.simulation": {
            "path": str(events_file),
            "bytes": 4,
            "sha256": hashlib.sha256(b"fake").hexdigest(),
        }
    }
    session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    events_file.write_bytes(b"other events")  # the same path, other events
    with pytest.raises(
        RunError, match=r"supplied\.gamma\.simulation: .* changed after the session"
    ):
        session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    assert not (tmp_path / "b").exists() and fake.calls == []


# --- a candidate's work is tasks: ordered by what they read, recorded by their results ---


def latest_ready(plan: Plan, run: RunTask, order: list[str]) -> dict[TaskId, TaskResult]:
    """An executor that runs, of the tasks whose needs all have results, always the last in plan
    order: another order in which every task comes after what it reads. ``order`` receives it."""
    results: dict[TaskId, TaskResult] = {}
    pending = list(plan.tasks)
    while pending:
        task = [t for t in pending if all(need in results for need in t.needs)][-1]
        pending.remove(task)
        order.append(task.id.label)
        reads = {need: results[need] for need in task.needs}
        results[task.id] = blocked(task, results) or run(task, reads)
    return results


def comparable(summary: Any, run_dir: Path) -> Any:
    """``summary`` without what differs from one evaluation of a candidate to the next: its id,
    the times and where it ran."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in ("id", "elapsed_seconds")}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(json.loads(json.dumps(summary).replace(str(run_dir), "RUN")))


@pytest.mark.parametrize("ddsim", [{}, {"returncode": 2, "stderr": "G4Exception: bad volume"}])
def test_the_record_does_not_depend_on_the_order_the_tasks_ran_in(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch, ddsim: dict[str, Any]
) -> None:
    from k4opt.problem import _evaluate

    simulation.install(monkeypatch, **ddsim)
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])
    seen: list[bool] = []

    class Echo(parameter_echo()):  # type: ignore[misc]
        def evaluate(self, request: EvaluationRequest) -> Any:
            # every evaluator reads the geometry task, so the geometry is there in any order
            seen.append(request.inputs["geometry"].is_file())
            return super().evaluate(request)

    session = EvaluationSession.open(problem_file, [cell_count(), Echo])
    first = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    order: list[str] = []
    monkeypatch.setattr(_evaluate, "run_tasks", lambda plan, run: latest_ready(plan, run, order))
    second = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    planned = [task.id.label for task in second.candidate.plan.tasks]
    assert order != planned and sorted(order) == sorted(planned)
    assert order[:2] == ["gamma/generation", "geometry"]  # the gun first: it reads no geometry
    assert seen == [True, True]
    assert comparable(second.summary, tmp_path / "b") == comparable(first.summary, tmp_path / "a")
    assert second.summary["status"] == ("valid" if not ddsim else "failed")


def test_the_record_is_assembled_from_the_task_results_alone(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a", run={"label": "golden"})
    written = json.loads(outcome.summary_path.read_text())
    shutil.rmtree(tmp_path / "a")  # nothing it makes the record from is read from disk
    again = assemble(
        written["id"],
        session.problem_path,
        session.spec,
        {RADIUS: 450.0},
        outcome.candidate,
        outcome.assessment,
        tmp_path / "a",
        session._taken,
        {"label": "golden"},
        session._process,
    )
    assert json.loads(json.dumps(again)) == written
    tasks = [
        written["geometry"],
        *written["samples"]["gamma"]["stages"].values(),
        *written["samples"]["gamma"]["evaluations"].values(),
    ]
    assert len(tasks) == 4 and written["evaluations"] == {}  # no evaluation of the candidate
    assert all(task["status"] in ("done", "valid") for task in tasks)


def change(data: dict[str, Any], path: str, value: Any) -> None:
    """Set the entry at ``path`` of ``data``, its keys separated by ``/`` (list items by index),
    to ``value``."""
    *parents, last = path.split("/")
    for key in parents:
        data = data[int(key)] if isinstance(data, list) else data[key]
    if isinstance(data, list):
        data[int(last)] = value
    else:
        data[last] = value


CELLED = "samples/gamma/evaluations/test.cell_count"
SIMULATED = "samples/gamma/stages/simulation"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        # what is blocked, by what, and why: re-derived from what each task reads
        (
            f"{CELLED}/status",
            "failed",
            "evaluator 'gamma/test.cell_count' is failed (blocked by ['gamma/simulation'], "
            "message 'not run: gamma/simulation failed'), but what it reads makes it blocked by "
            "['gamma/simulation']",
        ),
        (f"{CELLED}/blocked_by", [], "is blocked (blocked by [], message"),
        (
            f"{CELLED}/blocked_by",
            ["gamma/generation"],
            "is blocked (blocked by ['gamma/generation']",
        ),
        (f"{CELLED}/message", "not run: gamma/generation failed", "but what it reads makes it"),
        (
            f"{SIMULATED}/status",
            "done",
            "stage 'gamma/simulation': outputs [], but a stage that is done makes events in its "
            "directory",
        ),
        (
            f"{SIMULATED}/status",
            "blocked",
            "stage 'gamma/simulation' is blocked, but every task it reads succeeded",
        ),
        (f"{SIMULATED}/message", None, "stage 'gamma/simulation': a task that is failed needs a"),
        (f"{SIMULATED}/status", "crashed", "status 'crashed' is none of done, failed"),
        (
            "evaluations/test.parameter_echo/blocked_by",
            ["gamma/simulation"],
            "is blocked, but every task it reads",
        ),
        # what each task reads is the problem's, not the record's to say
        (
            f"{CELLED}/reads",
            ["geometry", "gamma/simulation"],
            "evaluator 'gamma/test.cell_count' reads ['geometry', 'gamma/simulation'], but the "
            "problem has it read ['geometry', 'gamma/generation', 'gamma/simulation']",
        ),
        (f"{SIMULATED}/reads", ["geometry"], "stage 'gamma/simulation' reads ['geometry'], but"),
        ("samples/gamma/stages/generation", None, "samples.gamma.stages.generation must be a map"),
        ("samples/gamma/stages", {}, "samples.gamma.stages are [], but the problem makes"),
        ("samples/gamma", None, "samples.gamma must be a mapping"),
        ("samples", {}, "its samples are [], but the problem's are ['gamma']"),
        ("geometry/status", "failed", "geometry: status 'failed', but it is always done"),
        ("supplied", ["gamma/simulation"], "but the problem makes ['simulation']"),
        ("supplied", ["simulation"], "supplied ['simulation'] must list stages of the problem's"),
        ("supplied", ["pion/simulation"], "must list stages of the problem's samples"),
        ("supplied", ["geometry"], "supplied ['geometry'] must list stages"),
        ("supplied", ["gamma/simulation"] * 2, "must list each stage once"),
        # where an evaluation is recorded says what it measured: a sample, or the candidate
        (
            "samples/gamma/evaluations",
            {},
            "samples.gamma.evaluations are [], but the problem's are ['test.cell_count']",
        ),
        (
            "evaluations/test.cell_count",
            {},
            "its evaluations are ['test.parameter_echo', 'test.cell_count'], but the problem's "
            "are ['test.parameter_echo']",
        ),
        ("metrics/0/sample", None, "its metrics are [['test.cell_count', None, "),
        # what a binding was configured with is the problem's, not the record's to say
        (
            f"{CELLED}/collections/hits",
            "OtherCollection",
            "evaluator 'gamma/test.cell_count': collections {'hits': 'OtherCollection'}, but the "
            "problem gives {'hits': ",
        ),
        (
            f"{CELLED}/settings/declared",
            {"threshold": 1},
            "evaluator 'gamma/test.cell_count': settings.declared {'threshold': 1}, but the "
            "problem",
        ),
        (
            "implementations/test.cell_count/stage",
            "generation",
            "implementation 'test.cell_count': stage 'generation', but the problem has it read "
            "'simulation'",
        ),
        ("message", "gamma/simulation failed: boom", "but its tasks make it"),
        ("status", "valid", "status is 'valid', but its tasks make it 'failed'"),
    ],
)
def test_a_record_whose_tasks_disagree_with_its_plan_is_refused(
    tmp_path: Path,
    problem_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: Any,
    message: str,
) -> None:
    simulation.install(monkeypatch, returncode=2, stderr="G4Exception: bad volume")
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])
    session = EvaluationSession.open(problem_file, [cell_count(), parameter_echo()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    data = json.loads(outcome.summary_path.read_text())
    Assessment.from_summary(session.spec, data)  # as written, it holds up
    # the sample's evaluation, blocked by its simulation, then the candidate's
    assert [e["status"] for e in data["samples"]["gamma"]["evaluations"].values()] == ["blocked"]
    assert [e["status"] for e in data["evaluations"].values()] == ["valid"]
    change(data, path, value)
    with pytest.raises(RecordError, match=re.escape(message)):
        Assessment.from_summary(session.spec, data)


def test_a_blocked_evaluator_that_measured_is_refused(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2)
    session = EvaluationSession.open(problem_file, [cell_count()])
    data = json.loads(session.evaluate({RADIUS: 450.0}, tmp_path / "a").summary_path.read_text())
    data["samples"]["gamma"]["evaluations"]["test.cell_count"]["measurements"] = [
        {"stage": "simulation", "metric": "ecal_barrel.mean_hit_cells", "value": 3.0}
        | {"unit": "1", "uncertainty": None}
    ]
    with pytest.raises(RecordError, match="it is blocked, so it was never called, yet it measured"):
        Assessment.from_summary(session.spec, data)


# --- several samples: each simulated once, on one geometry, measured on its own -------------


def two_samples(problem_file: Path, **pion: Any) -> dict[str, Any]:
    """Give the problem a second sample, ``pion`` (negative pions, unless ``pion`` says other
    stages), which the evaluator measures too, with an objective on each sample; the document."""
    doc = yaml.safe_load(problem_file.read_text())
    stages = copy.deepcopy(doc["samples"]["gamma"])
    stages["generation"]["particle"] = "pi-"
    stages["simulation"]["seed"] = 6
    doc["samples"]["pion"] = pion or stages
    doc["evaluators"][0]["samples"] = ["gamma", "pion"]
    objective = doc["objectives"][0]
    doc["objectives"] = [objective | {"sample": "gamma"}, objective | {"sample": "pion"}]
    problem_file.write_text(yaml.safe_dump(doc))
    return doc


def particle(argv: list[str]) -> str:
    """The particle a ddsim command line fires."""
    return next(a for a in argv if a.startswith("--gun.particle=")).partition("=")[2]


def test_each_sample_is_simulated_once_on_the_one_geometry_and_measured_on_its_own(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    two_samples(problem_file)
    session = EvaluationSession.open(problem_file, [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert guard == ["discover", "materialize"]  # one geometry for both
    assert [particle(argv) for argv, _ in fake.calls] == ["gamma", "pi-"]
    compact = str(tmp_path / "a" / "geometry" / MINI_COMPACT)
    assert all(f"--compactFile={compact}" in argv for argv, _ in fake.calls)
    # each sample's stages and evaluations in a directory of its own
    for sample in ("gamma", "pion"):
        made = tmp_path / "a" / "samples" / sample
        assert sorted(p.name for p in made.iterdir()) == ["evaluators", "simulation"]
        assert (made / "evaluators" / "test.cell_count").is_dir()
    record = outcome.candidate
    assert record.status == "valid"
    assert list(record.evaluations) == ["gamma/test.cell_count", "pion/test.cell_count"]
    gamma, pion = record.records
    assert gamma.request.seed == 5 and pion.request.seed == 6  # each its own sample's
    assert gamma.request.inputs["simulation"] != pion.request.inputs["simulation"]
    summary = outcome.summary
    assert list(summary["samples"]) == ["gamma", "pion"]  # type: ignore[call-overload]
    assert [m["label"] for m in summary["metrics"]] == [  # type: ignore[attr-defined]
        "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
        "pion/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
    ]
    assert [o.label for o in outcome.assessment.objectives or ()] == [
        m["label"]
        for m in summary["metrics"]  # type: ignore[attr-defined]
    ]
    assert outcome.events == {"gamma": (3, 3), "pion": (3, 3)}
    written = json.loads(outcome.summary_path.read_text())
    assert Assessment.from_summary(session.spec, written) == outcome.assessment
    # one warning for the binding, however many samples ddsim simulates for it
    assert len([w for w in session.warnings if w.startswith(UNEXPECTED)]) == 1


def test_the_command_reports_each_evaluation_with_its_sample(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    two_samples(problem_file)
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 0, err
    assert out.index("evaluator:   test.cell_count 1 on gamma: valid") < out.index(
        "evaluator:   test.cell_count 1 on pion: valid"
    )
    assert "events:      gamma: 3 of 3" in out and "events:      pion: 3 of 3" in out
    stdout = tmp_path / "a" / "samples" / "pion" / "simulation" / "ddsim.stdout.log"
    assert f"stdout:      pion: {stdout}" in out


def test_a_sample_whose_simulation_fails_blocks_only_its_own_evaluations(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9]))

    def pions_crash(argv: list[str], **kwargs: Any) -> Any:
        code = fake(argv, **kwargs)
        return 2 if particle(argv) == "pi-" else code

    monkeypatch.setattr(ddsim_run, "_execute", pions_crash)
    doc = two_samples(problem_file)
    doc["evaluators"].append({"name": "test.parameter_echo", "version": "1"})
    doc["objectives"].append(
        {"evaluator": "test.parameter_echo", "stage": ECHO[0], "metric": ECHO[1]}
        | {"direction": "minimize"}
    )
    problem_file.write_text(yaml.safe_dump(doc))
    session = EvaluationSession.open(problem_file, [cell_count(), parameter_echo()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    record = outcome.candidate
    statuses = {label: r.result.status for label, r in record.evaluations.items()}
    assert statuses == {
        "gamma/test.cell_count": "valid",
        "pion/test.cell_count": "blocked",
        "test.parameter_echo": "valid",
    }
    assert record.status == "failed"
    message = record.message or ""
    assert message.startswith("pion/simulation failed: ddsim exited with status 2")
    assert message.endswith("; blocked: pion/test.cell_count")
    # what was measured is kept, and nothing stands in for what was not
    assert [m["value"] for m in outcome.summary["metrics"]] == [3.0, None, 450.0]  # type: ignore[attr-defined]
    assert outcome.assessment.objectives is None
    written = json.loads(outcome.summary_path.read_text())
    assert written["samples"]["pion"]["evaluations"]["test.cell_count"]["blocked_by"] == [
        "pion/simulation"
    ]
    assert written["samples"]["gamma"]["stages"]["simulation"]["status"] == "done"
    Assessment.from_summary(session.spec, written)  # and the record holds up


def test_a_file_read_by_two_samples_is_hashed_once_and_named_for_each(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    from k4opt.problem._stages import _sha256

    hepmc = tmp_path / "zh.hepmc"  # next to the problem file
    hepmc.write_bytes(b"HepMC::Version 3\n")
    doc = yaml.safe_load(problem_file.read_text())
    doc["samples"]["gamma"]["generation"] = {"file": hepmc.name}
    pion = copy.deepcopy(doc["samples"]["gamma"]) | {"simulation": {"executor": "ddsim"}}
    pion["simulation"] |= {"events": 3, "seed": 6}
    problem_file.write_text(yaml.safe_dump(doc))
    two_samples(problem_file, **pion)
    _sha256.cache_clear()
    session = EvaluationSession.open(problem_file, [cell_count()])
    inputs = session.identity()["inputs"]
    assert list(inputs) == ["samples.gamma.generation.file", "samples.pion.generation.file"]
    assert inputs["samples.gamma.generation.file"] == inputs["samples.pion.generation.file"]
    assert _sha256.cache_info().misses == 1  # read once
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert [f"--inputFiles={hepmc}" in argv for argv, _ in fake.calls] == [True, True]
    assert outcome.candidate.status == "valid"


def test_a_supplied_stage_of_one_sample_leaves_the_other_simulated(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    two_samples(problem_file)
    events_file = tmp_path / "earlier.edm4hep.root"
    events_file.write_bytes(b"fake")
    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450"], tmp_path / "a", inputs=[f"pion/simulation={events_file}"]
    )
    assert [particle(argv) for argv, _ in fake.calls] == ["gamma"]
    summary = outcome.summary
    assert (summary["represents_candidate"], summary["supplied"]) == (False, ["pion/simulation"])
    assert list(summary["samples"]["pion"]["stages"]) == ["simulation"]  # type: ignore[index]
    gamma = summary["samples"]["gamma"]["stages"]  # type: ignore[index]
    assert list(gamma) == ["generation", "simulation"]
    assert outcome.assessment.feasible is None  # measured, but not judged


def test_a_file_that_changes_while_the_second_sample_runs_leaves_no_record(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, events_with([1], [2], [3]))
    hepmc = tmp_path / "zh.hepmc"
    hepmc.write_bytes(b"HepMC::Version 3\n")
    doc = two_samples(problem_file)
    doc["samples"]["gamma"]["generation"] = {"file": hepmc.name}
    problem_file.write_text(yaml.safe_dump(doc))

    def rewriting(argv: list[str], **kwargs: Any) -> Any:
        if len(fake.calls) == 1:  # the second simulation, of pion, after gamma read the file
            hepmc.write_bytes(b"HepMC::Version 3\nE 1\n")
        return fake(argv, **kwargs)

    monkeypatch.setattr(ddsim_run, "_execute", rewriting)
    session = EvaluationSession.open(problem_file, [cell_count()])
    with pytest.raises(RunError, match=r"samples\.gamma\.generation\.file: .* changed after"):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert len(fake.calls) == 2
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()
    for sample in ("gamma", "pion"):  # what ran is kept, to look at
        assert (tmp_path / "a" / "samples" / sample / "simulation").is_dir()


def test_samples_with_identical_stages_are_a_warning(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    two_samples(problem_file, **doc["samples"]["gamma"])
    session = EvaluationSession.open(problem_file, [cell_count()])
    assert (
        "samples gamma and pion have identical stages; each is simulated separately"
        in session.warnings
    )
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert len(fake.calls) == 2  # each all the same
    assert outcome.candidate.status == "valid"


# --- the files of a candidate: named by reference, relative to the run, hashed where claimed ---

HISTOGRAM = "hit cells per event, JSON"


def plotting(**declared: str) -> type[Any]:
    """``test.cell_count``, writing the cells per event as the artifact ``histogram`` in its own
    directory, and declaring the artifacts ``declared`` (``histogram``, unless given)."""

    class Plotting(cell_count()):  # type: ignore[misc]
        artifacts = declared or {"histogram": HISTOGRAM}

        def evaluate(self, request: EvaluationRequest) -> Any:
            measured = result_from(self, super().evaluate(request))
            assert request.workdir is not None
            histogram = request.workdir / "histogram.json"
            histogram.write_text(json.dumps([3, 2, 4]))
            return dataclasses.replace(measured, artifacts={"histogram": histogram})

    return Plotting


def paths_in(value: Any, at: str = "") -> list[tuple[str, str]]:
    """Every text of a summary, with where it is (``samples/gamma/stages/simulation/argv``)."""
    if isinstance(value, dict):
        return [found for k, v in value.items() for found in paths_in(v, f"{at}/{k}")]
    if isinstance(value, list):
        return [found for item in value for found in paths_in(item, at)]
    return [(at, value)] if isinstance(value, str) else []


def test_a_record_stays_true_wherever_its_run_directory_is_moved(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [plotting()])
    made = tmp_path / "a"
    session.evaluate({RADIUS: 450.0}, made)
    summary = json.loads((made / SUMMARY_NAME).read_text())
    # the run directory once, where it was made; the exact ddsim command as it ran
    named = {at for at, text in paths_in(summary) if str(made) in text}
    assert named == {"/run_dir", "/samples/gamma/stages/simulation/argv"}
    moved = tmp_path / "archive" / "a"
    moved.parent.mkdir()
    shutil.move(made, moved)
    Assessment.from_summary(session.spec, summary)  # holds up as it is
    stages = summary["samples"]["gamma"]["stages"]
    (evaluation,) = summary["samples"]["gamma"]["evaluations"].values()
    files = [
        summary["geometry"]["compact"],
        summary["geometry"]["manifest"],
        stages["simulation"]["outputs"]["events"]["path"],
        *stages["simulation"]["logs"].values(),
        evaluation["artifacts"]["histogram"]["path"],
    ]
    assert all((moved / path).is_file() for path in files)  # each read from where it is now
    histogram = evaluation["artifacts"]["histogram"]
    assert (
        hashlib.sha256((moved / histogram["path"]).read_bytes()).hexdigest() == histogram["sha256"]
    )


def test_an_artifact_is_recorded_by_reference_path_size_and_hash(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[plotting()]
    )
    summary = json.loads(outcome.summary_path.read_text())
    (evaluation,) = summary["samples"]["gamma"]["evaluations"].values()
    path = (
        tmp_path / "a" / "samples" / "gamma" / "evaluators" / "test.cell_count" / "histogram.json"
    )
    assert evaluation["artifacts"] == {
        "histogram": {
            "ref": "gamma/test.cell_count#histogram",
            "path": "samples/gamma/evaluators/test.cell_count/histogram.json",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    # what the implementation declares, as the record of every binding of it is checked against
    assert summary["implementations"]["test.cell_count"]["artifacts"] == ["histogram"]
    (made,) = outcome.candidate.results[
        TaskId(Kind.EVALUATOR, "test.cell_count", "gamma")
    ].made.values()
    assert made.artifact == Artifact(
        "gamma/test.cell_count#histogram",
        path,
        path.stat().st_size,
        evaluation["artifacts"]["histogram"]["sha256"],
    )
    without = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "b")
    assert without.summary["implementations"]["test.cell_count"]["artifacts"] is None


def test_an_artifact_is_hashed_once_and_an_edm4hep_output_never(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed: list[str] = []

    def sha256(path: str, *key: int) -> str:
        hashed.append(path)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    monkeypatch.setattr(_stages, "_sha256", sha256)  # uncached: every hash is counted
    run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[plotting()])
    artifact = tmp_path / "a" / "samples" / "gamma" / "evaluators" / "test.cell_count"
    assert hashed == [str(artifact / "histogram.json")]  # a gun reads no file


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        # a file the record names, changed after its task made it: the record would lie
        ("histogram", "gamma/test.cell_count#histogram changed after it was made"),
        ("events", "gamma/simulation#events changed after it was made"),
    ],
)
def test_a_file_that_changes_after_it_was_made_leaves_no_summary(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, changed: str, message: str
) -> None:
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])
    run_dir = tmp_path / "a"

    class Meddling(parameter_echo()):  # type: ignore[misc]
        """Measures the candidate after the sample was measured, and changes one of its files."""

        def evaluate(self, request: EvaluationRequest) -> Any:
            sample = run_dir / "samples" / "gamma"
            files = {
                "histogram": sample / "evaluators" / "test.cell_count" / "histogram.json",
                "events": sample / "simulation" / "events.edm4hep.root",
            }
            with files[changed].open("ab") as handle:
                handle.write(b"more")
            return super().evaluate(request)

    session = EvaluationSession.open(problem_file, [plotting(), Meddling])
    with pytest.raises(RunError, match=re.escape(message)):
        session.evaluate({RADIUS: 450.0}, run_dir)
    assert not (run_dir / SUMMARY_NAME).exists()


def test_a_hashed_file_whose_stat_changed_but_not_its_bytes_is_unchanged(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    run_dir = tmp_path / "a"
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])

    class Touching(parameter_echo()):  # type: ignore[misc]
        def evaluate(self, request: EvaluationRequest) -> Any:
            histogram = run_dir / "samples/gamma/evaluators/test.cell_count/histogram.json"
            histogram.write_bytes(histogram.read_bytes())  # rewritten as it was: hashed again
            return super().evaluate(request)

    outcome = EvaluationSession.open(problem_file, [plotting(), Touching]).evaluate(
        {RADIUS: 450.0}, run_dir
    )
    assert outcome.candidate.status == "valid" and outcome.summary_path.is_file()


VALID = "samples/gamma/evaluations/test.cell_count"
OUTPUT = "samples/gamma/stages/simulation/outputs/events"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        # a content identity only where one was computed
        (f"{VALID}/artifacts/histogram/sha256", None, "sha256 None, but such a file is hashed"),
        (f"{OUTPUT}/sha256", "a" * 64, "but such a file is never hashed"),
        (f"{VALID}/artifacts/histogram/sha256", "A" * 64, "sha256 must be 64 lowercase hex"),
        # each by its reference, its own and none other
        (f"{VALID}/artifacts/histogram/ref", "gamma/simulation#events", "ref 'gamma/simulation#"),
        (f"{OUTPUT}/ref", "gamma/simulation#output", "not 'gamma/simulation#events'"),
        (f"{OUTPUT}/bytes", -1, "bytes must be a size, got -1"),
        # relative to the run directory, in the task's own directory
        (f"{OUTPUT}/path", "/data/events.edm4hep.root", "is not a plain path relative to the run"),
        (f"{VALID}/artifacts/histogram/path", "samples/gamma/../x", "not a plain path relative"),
        (f"{VALID}/artifacts/histogram/path", "samples/gamma/x", "not in the task's own directory"),
        ("samples/gamma/stages/simulation/logs/stdout", "geometry/x.log", "not in the task's own"),
        ("geometry/compact", "/abs/compact.xml", "geometry: compact: '/abs/compact.xml' is not a"),
        ("samples/gamma/stages/simulation/directory", "elsewhere", "directory 'elsewhere', but"),
        (f"{VALID}/workdir", "evaluators/x", "workdir 'evaluators/x', but its directory is"),
        ("run_dir", None, "run_dir must be a nonblank string"),
        # what a stage that is done made, and what an evaluator declares, the record holds
        ("samples/gamma/stages/simulation/outputs", {}, "outputs [], but a stage that is done"),
        (f"{VALID}/artifacts", {}, "it is valid without its declared artifacts ['histogram']"),
        ("implementations/test.cell_count/artifacts", ["plot"], "artifacts ['histogram'], which"),
        ("implementations/test.cell_count/artifacts", ["plot", "plot"], "names one twice"),
        ("implementations/test.cell_count/artifacts", ["a#b"], "'a#b' is no artifact name"),
    ],
)
def test_a_record_whose_files_are_not_as_made_is_refused(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    path: str,
    value: Any,
    message: str,
) -> None:
    session = EvaluationSession.open(problem_file, [plotting()])
    data = json.loads(session.evaluate({RADIUS: 450.0}, tmp_path / "a").summary_path.read_text())
    Assessment.from_summary(session.spec, data)  # as written, it holds up
    change(data, path, value)
    with pytest.raises(RecordError, match=re.escape(message)):
        Assessment.from_summary(session.spec, data)


def test_a_blocked_evaluation_has_no_directory_and_no_artifacts(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2)
    session = EvaluationSession.open(problem_file, [plotting()])
    data = json.loads(session.evaluate({RADIUS: 450.0}, tmp_path / "a").summary_path.read_text())
    Assessment.from_summary(session.spec, data)  # blocked: valid without its artifact is fine
    change(data, f"{VALID}/workdir", "samples/gamma/evaluators/test.cell_count")
    with pytest.raises(RecordError, match="it is blocked, so it was never called, yet has a work"):
        Assessment.from_summary(session.spec, data)
