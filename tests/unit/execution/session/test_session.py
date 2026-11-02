"""``EvaluationSession`` and ``run_candidate`` with a fake ddsim, on the mini geometry: one
candidate, many candidates, and a new evaluator instance for each."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt import __version__
from k4opt.ddsim import run as ddsim_run
from k4opt.errors import K4OptError
from k4opt.evaluation import (
    CandidateRejected,
    Collection,
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRequest,
    Quantity,
    collection_names,
    metric_units,
    with_defaults,
)
from k4opt.execution import EvaluationSession, Rejected, RunError, RunOutcome
from k4opt.geometry import GeometrySource, Override
from k4opt.problem import CandidateError, ProblemConfigError, load_problem, parse_problem
from k4opt.problem.candidate import configure, parse_candidate
from k4opt.record import SUMMARY_NAME, Assessment, RecordError
from k4opt.revision import code_fingerprint, code_identity
from tests.support import simulation
from tests.support.common import tree_digest
from tests.support.evaluators import (
    CELL_COUNT,
    calo_photon,
    cell_count,
    implementations,
)
from tests.support.geometry import MINI_COMPACT, MINI_GEO
from tests.support.simulation import COLLECTION
from tests.unit.execution.session.conftest import (
    RADIUS,
    UNEXPECTED,
    Spy,
    errors,
    evaluate,
    run_candidate,
    spied,
)

EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "problem"
# an evaluator setting, which takes any number; the geometry takes those a double holds
SETTING = {"evaluators": {"calo.single_particle_shower": "n"}}


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
        Override(
            target="constants.InnerTracker_Barrel_radius_1",
            value="450.0*mm",
            file="compact/det_v1/tracker.xml",
        )
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
    assert on_disk["geometry"]["fingerprint"] == outcome.geometry_fingerprint
    assert on_disk["geometry"]["status"] == "done"
    # each file in the run directory by its path relative to it, which it was made in
    assert on_disk["run_dir"] == str(tmp_path / "a")
    assert on_disk["geometry"]["manifest"] == "geometry/k4opt-manifest.json"
    assert (tmp_path / "a" / on_disk["geometry"]["compact"]).is_file()
    resolved = {
        "parameter": RADIUS,
        "target": "constants.InnerTracker_Barrel_radius_1",
        "file": "compact/det_v1/tracker.xml",
        "locator": "/lccdd/define/constant[2]",
        "attribute": "value",
    }
    assert on_disk["geometry"]["changes"] == [
        resolved | {"old_value": "400*mm", "new_value": "450.0*mm"}
    ]
    assert on_disk["geometry"]["varies"] == [RADIUS]
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
    assert first.geometry_fingerprint != second.geometry_fingerprint
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
        [f"{RADIUS}=450", "nothing=1"],
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
    doc["parameters"] = {
        "n": {"targets": SETTING, "type": "categorical", "choices": [big, big + 1, 0.5]}
    }
    spec = parse_problem(doc)
    assert parse_candidate(spec, [f"n={big + 1}"]) == {"n": big + 1}
    assert parse_candidate(spec, [f"n={big}"]) == {"n": big}
    assert parse_candidate(spec, ["n=0.5"]) == {"n": 0.5}
    from k4opt.problem.candidate import validate_candidate

    _, key_a = validate_candidate(spec, {"n": big})
    _, key_b = validate_candidate(spec, parse_candidate(spec, [f"n={big + 1}"]))
    assert key_a != key_b  # two different candidates stay different


def test_a_structured_target_is_set_by_the_name_of_its_parameter(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    ref = "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"
    doc = yaml.safe_load(problem_file.read_text())
    doc["parameters"] = {
        "nphi": {"targets": {"geometry": ref}, "type": "integer", "bounds": [10, 100]}
    }
    problem_file.write_text(yaml.safe_dump(doc))
    assert parse_candidate(load_problem(problem_file), ["nphi=90"]) == {"nphi": 90}
    outcome = run_candidate(problem_file, ["nphi=90"], tmp_path / "a")
    assert outcome.summary["candidate"] == {"nphi": 90}
    with pytest.raises(CandidateError, match="NAME=VALUE"):
        parse_candidate(load_problem(problem_file), ["nphi"])


def test_a_value_may_hold_an_equals_sign(doc: dict[str, Any]) -> None:
    doc["parameters"] = {
        "mode": {
            "targets": {"geometry": "Mode"},
            "type": "categorical",
            "choices": ["a=b", "c"],
        }
    }
    assert parse_candidate(parse_problem(doc), ["mode=a=b"]) == {"mode": "a=b"}


@pytest.mark.parametrize(
    "text",
    ["9007199254740993.0", "9.007199254740993e15", "9007199254740993.00", "+9007199254740993"],
)
def test_every_spelling_of_a_large_integer_category_stays_exact(
    doc: dict[str, Any], text: str
) -> None:
    big = 2**53
    doc["parameters"] = {
        "n": {"targets": SETTING, "type": "categorical", "choices": [big, big + 1, 1e23, 0.5]}
    }
    spec = parse_problem(doc)
    assert parse_candidate(spec, [f"n={text}"]) == {"n": big + 1}
    assert parse_candidate(spec, ["n=9007199254740992.0"]) == {"n": big}
    assert parse_candidate(spec, ["n=1e23"]) == {"n": 1e23}
    assert parse_candidate(spec, ["n=0.50"]) == {"n": 0.5}


def test_an_undeclared_large_integer_is_not_rounded_into_a_choice(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    big = 2**53
    doc = yaml.safe_load(problem_file.read_text())
    doc["parameters"] = {
        RADIUS: {
            "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": [big, big + 2],
        }
    }
    problem_file.write_text(yaml.safe_dump(doc))
    for text in (str(big + 1), f"{big + 1}.0", "9.007199254740993e15"):
        with pytest.raises(CandidateError, match="not one of"):
            run_candidate(problem_file, [f"{RADIUS}={text}"], tmp_path / "a")
    assert not (tmp_path / "a").exists()
    assert fake.calls == []


def test_a_value_that_only_rounds_to_a_choice_is_rejected(doc: dict[str, Any]) -> None:
    doc["parameters"] = {
        "n": {
            "targets": {"geometry": "n"},
            "type": "categorical",
            "choices": [1, 2],
        }
    }
    spec = parse_problem(doc)
    for text in ("1.0000000000000001", "2.0000000000000001", "1.5"):
        with pytest.raises(CandidateError, match="not one of"):
            parse_candidate(spec, [f"n={text}"])
    assert parse_candidate(spec, ["n=1.0"]) == {"n": 1}


def test_a_decimal_that_rounds_to_a_float_choice_is_rejected(doc: dict[str, Any]) -> None:
    doc["parameters"] = {
        "n": {
            "targets": {"geometry": "n"},
            "type": "categorical",
            "choices": [2.0**53, 0.1],
        }
    }
    spec = parse_problem(doc)
    with pytest.raises(CandidateError, match="not one of"):
        parse_candidate(spec, [f"n={2**53 + 1}"])
    assert parse_candidate(spec, ["n=0.1"]) == {"n": 0.1}
    assert parse_candidate(spec, [f"n={2**53}"]) == {"n": 2.0**53}


@pytest.mark.parametrize("text", [str(2**53 + 1), str(10**400), f"-{2**53 + 1}"])
def test_integer_literals_of_a_continuous_parameter_stay_exact(
    doc: dict[str, Any], text: str
) -> None:
    doc["parameters"]["ecal_cell_size"]["bounds"] = [-(10**401), 10**401]
    doc["parameters"]["ecal_cell_size"]["targets"] = SETTING
    del doc["parameters"]["ecal_cell_size"]["unit"]
    spec = parse_problem(doc)
    (value,) = parse_candidate(spec, [f"ecal_cell_size={text}"]).values()
    assert value == int(text)
    assert isinstance(value, int)  # not rounded through float, no overflow to infinity
    assert parse_candidate(spec, ["ecal_cell_size=6"]) == {"ecal_cell_size": 6}
    (small,) = parse_candidate(spec, ["ecal_cell_size=6"]).values()
    assert isinstance(small, float)  # ordinary values stay floats: "6.0*mm"


@pytest.mark.parametrize("kind", ["continuous", "integer"])
def test_an_integer_with_more_digits_than_python_converts_is_a_candidate_error(
    doc: dict[str, Any], kind: str
) -> None:
    doc["parameters"] = {"n": {"targets": {"geometry": "n"}, "type": kind, "bounds": [0, 10]}}
    text = "1" * (sys.get_int_max_str_digits() + 1)
    with pytest.raises(CandidateError, match="too many digits"):
        parse_candidate(parse_problem(doc), [f"n={text}"])


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
    doc["parameters"] = {"x": {"targets": {"geometry": "x"}, **parameter}}
    if "bounds" not in parameter and "choices" not in parameter:
        doc["parameters"]["x"]["bounds"] = [0, 100]
    spec = parse_problem(doc)
    assert configure(spec, {"x": value}).geometry == (expected,)  # type: ignore[dict-item]


def test_an_unsupported_unit_is_refused_when_the_problem_is_read(doc: dict[str, Any]) -> None:
    doc["parameters"]["ecal_cell_size"]["unit"] = "furlong"
    with pytest.raises(ProblemConfigError, match="unit 'furlong' cannot be written"):
        parse_problem(doc)


def test_text_values_are_read_by_the_parameter_kind(
    problem_file: Path, doc: dict[str, Any]
) -> None:
    spec = load_problem(problem_file)
    assert parse_candidate(spec, [f"{RADIUS}=6"]) == {RADIUS: 6.0}
    assert parse_candidate(spec, [f"{RADIUS}=1e2"]) == {RADIUS: 100.0}
    assert parse_candidate(spec, [f"{RADIUS}=-.5"]) == {RADIUS: -0.5}
    doc["parameters"] = {"n": {"targets": {"geometry": "n"}, "type": "integer", "bounds": [1, 9]}}
    integer = parse_problem(doc)
    assert parse_candidate(integer, ["n=4"]) == {"n": 4}
    with pytest.raises(CandidateError, match="integer"):
        parse_candidate(integer, ["n=4.0"])


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


# --- one session, many candidates ----------------------------------------------------------


def candidates(requests: list[Any]) -> list[object]:
    """The radius each request was made for."""
    return [request.settings[RADIUS] for request in requests]


def test_a_session_checks_and_measures_each_candidate_with_a_request_of_its_own(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    spy = Spy()
    session = EvaluationSession.open(spied(problem_file), implementations(spy))
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
    assert first.geometry_fingerprint != second.geometry_fingerprint
    assert first.summary["id"] != second.summary["id"]


def test_a_check_that_rejects_one_candidate_rejects_only_that_one(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    class Picky(Spy):
        @classmethod
        def check(cls, request: Any) -> None:
            super().check(request)
            if request.settings[RADIUS] == 500.0:
                cls.reject("this evaluator cannot measure a radius of 500 mm")

    session = EvaluationSession.open(spied(problem_file), [Picky])
    rejected = session.evaluate({RADIUS: 500.0}, tmp_path / "a")
    assert rejected == Rejected(
        "gamma/test.cell_count", "this evaluator cannot measure a radius of 500 mm"
    )
    assert not (tmp_path / "a").exists() and fake.calls == [] and "materialize" not in guard
    with pytest.raises(CandidateRejected, match="500 mm"):
        session.check({RADIUS: 500.0})
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert isinstance(outcome, RunOutcome) and outcome.candidate.status == "valid"


@pytest.mark.parametrize("required", [False, True])
def test_only_a_required_evaluation_s_check_rejects_its_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, required: bool
) -> None:
    class Veto(Spy):
        name = "test.veto"

        @classmethod
        def check(cls, request: Any) -> None:
            cls.reject("the readout cannot be built")

    doc = yaml.safe_load(problem_file.read_text())
    doc["objectives"][0]["evaluator"] = "test.cell_count"
    veto = {"name": "test.veto", "version": "1", "stage": "simulation"}
    doc["evaluators"].append(veto | {"collections": {"hits": COLLECTION}, "required": required})
    if not required:
        del doc["evaluators"][-1]["required"]
    problem_file.write_text(yaml.safe_dump(doc))
    session = EvaluationSession.open(problem_file, [cell_count(), Veto])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    if required:  # the candidate's status depends on it: its veto rejects the candidate
        assert outcome == Rejected("gamma/test.veto", "the readout cannot be built")
        assert not (tmp_path / "a").exists() and fake.calls == []
        return
    # a diagnostic decides nothing of the candidate, so it cannot veto it either: it alone is
    # not run, recorded invalid, and the candidate is evaluated as without it
    assert isinstance(outcome, RunOutcome) and outcome.candidate.status == "valid"
    (diagnostic,) = outcome.candidate.diagnostics
    assert (diagnostic.id.label, diagnostic.status) == ("gamma/test.veto", "invalid")
    assert diagnostic.message == (
        "not run: its check() rejected the candidate (the readout cannot be built)"
    )
    summary = json.loads(outcome.summary_path.read_text())
    spec = load_problem(problem_file)
    assessment = Assessment.from_summary(spec, summary)
    assert assessment.feasible is True and summary["diagnostics"][0]["status"] == "invalid"
    # read back strictly: never called, so a record that says it measured, printed or was invalid
    # for another reason does not hold up
    evaluations = summary["samples"]["gamma"]["evaluations"]
    measured = evaluations["test.cell_count"]["measurements"]
    for key, value, refusal in (
        ("measurements", measured, "never called, yet it holds measurements"),
        ("log", "samples/gamma/evaluators/test.veto/evaluator.log", "yet it holds log"),
        ("message", "unusable", "only a diagnostic whose check\\(\\) rejected the candidate"),
    ):
        forged = copy.deepcopy(summary)
        forged["samples"]["gamma"]["evaluations"]["test.veto"][key] = value
        with pytest.raises(RecordError, match=refusal):
            Assessment.from_summary(spec, forged)


@pytest.mark.parametrize("ending", [EvaluationFailed, EvaluationInvalid, ValueError])
def test_a_check_that_ends_otherwise_than_rejecting_stops_the_session_s_evaluation(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, ending: type[Exception]
) -> None:
    class Broken(Spy):
        @classmethod
        def check(cls, request: Any) -> None:
            raise ending("cannot tell")

    session = EvaluationSession.open(problem_file, [Broken])
    expected = ValueError if ending is ValueError else EvaluationContractError
    with pytest.raises(expected, match="cannot tell"):  # a bug, not a candidate's verdict
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert not (tmp_path / "a").exists() and fake.calls == []


def test_a_check_must_be_a_classmethod(problem_file: Path) -> None:
    class Bound(Spy):
        def check(self, request: Any) -> None:  # type: ignore[override]
            pass

    with pytest.raises(EvaluationContractError, match=r"Bound\.check must be a classmethod"):
        EvaluationSession.open(problem_file, [Bound])


def test_a_declaration_computed_by_code_is_refused(problem_file: Path) -> None:
    class Computed(Spy):
        @property
        def settings(self) -> dict[str, Any]:  # type: ignore[override]
            return {}

    with pytest.raises(EvaluationContractError, match=r"Computed\.settings is a property"):
        EvaluationSession.open(problem_file, [Computed])


# --- an evaluator class, and one new instance of it for every evaluation --------------------


def test_every_evaluation_gets_one_new_instance_where_it_runs(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    created: list[Spy] = []

    class Counted(Spy):
        def __init__(self) -> None:
            super().__init__()
            created.append(self)

    session = EvaluationSession.open(spied(problem_file), [Counted])
    session.check({RADIUS: 300.0})
    assert created == []  # binding and checking create nothing
    session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    # each evaluation has an instance of its own, so nothing one keeps reaches another candidate
    assert [candidates(i.evaluated) for i in created] == [[450.0], [500.0]]


def test_an_evaluator_that_cannot_be_created_stops_its_candidate_without_a_record(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    class Broken(Spy):
        def __init__(self) -> None:
            raise RuntimeError("no model file")

    session = EvaluationSession.open(problem_file, [Broken])  # nothing is created to bind it
    with pytest.raises(EvaluationContractError, match=r"Broken\(\) failed: RuntimeError: no model"):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    # created where it runs, after what it reads: the simulation ran, and nothing is recorded
    assert len(fake.calls) == 1 and not (tmp_path / "a" / "summary.json").exists()


@pytest.mark.parametrize(
    ("field", "later"),
    [
        ("name", "test.other"),
        ("version", "2"),
        ("metrics", {"other": "1"}),
        ("settings", {"threshold": Quantity("GeV", default=1.0)}),
        ("collections", {"hits": Collection("SimTrackerHit")}),
        ("artifacts", {"other": "a file"}),
    ],
)
def test_an_instance_declares_nothing_of_its_own(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, field: str, later: Any
) -> None:
    class Drifting(Spy):
        def __init__(self) -> None:
            super().__init__()
            setattr(self, field, later)  # the record names what the class declared

    session = EvaluationSession.open(problem_file, [Drifting])
    with pytest.raises(EvaluationContractError, match=rf"an instance of .*Drifting sets {field}"):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert not (tmp_path / "a" / "summary.json").exists()


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
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert isinstance(outcome, RunOutcome) and outcome.candidate.status == "valid"
    # the next candidate's preflight finds the class changed, before anything of it is created
    with pytest.raises(EvaluationContractError, match=r"settings \{'cuts': \{'min': 2\}\}"):
        session.evaluate({RADIUS: 500.0}, tmp_path / "b")
    assert not (tmp_path / "b").exists() and len(fake.calls) == 1


def test_a_session_keeps_the_problem_it_loaded(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    definition = session.spec.to_dict()
    doc = yaml.safe_load(problem_file.read_text())
    doc["parameters"][RADIUS]["bounds"] = [100.0, 200.0]  # an edit during a campaign
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
        # the attribute each geometry destination sets, as the session resolved it
        "targets": [
            {
                "parameter": RADIUS,
                "target": "constants.InnerTracker_Barrel_radius_1",
                "file": "compact/det_v1/tracker.xml",
                "locator": "/lccdd/define/constant[2]",
                "attribute": "value",
                "value": "400*mm",
            }
        ],
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
    from k4opt.execution import _baseline

    identities = iter(({"sha256": "opened", "git": None}, {"sha256": "edited", "git": None}))
    monkeypatch.setattr(_baseline, "code_identity", lambda: next(identities))
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
    data["parameters"]["support"] = {
        "targets": {"geometry": SUPPORT},
        "type": "categorical",
        "choices": choices,
    }
    problem_file.write_text(yaml.safe_dump(data))
    session = EvaluationSession.open(problem_file, [cell_count()])
    opened = session.identity()["geometry"]["externals"]  # type: ignore[index]
    assert "${CAD}/b.stl" not in opened

    outcome = session.evaluate({RADIUS: 450.0, "support": "${CAD}/b.stl"}, tmp_path / "a")
    geometry = outcome.summary["geometry"]
    read = {"path": str(model), "sha256": hashlib.sha256(b"solid b").hexdigest()}
    assert geometry["candidate_externals"] == {"${CAD}/b.stl": read}  # type: ignore[index]
    assert geometry["externals"] == opened  # type: ignore[index]
    unchanged = session.evaluate({RADIUS: 450.0, "support": "cad/support.stl"}, tmp_path / "b")
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
        session.evaluate({RADIUS: 500.0, "support": "${CAD}/b.stl"}, tmp_path / "c")
    assert not (tmp_path / "c" / SUMMARY_NAME).exists()
    assert len(fake.calls) == (2 if when == "between candidates" else 3)


def limit_simulation(problem_file: Path, seconds: float) -> Path:
    """``problem_file`` whose simulation of ``gamma`` may take ``seconds``."""
    doc = yaml.safe_load(problem_file.read_text())
    doc["samples"]["gamma"]["simulation"]["timeout"] = seconds
    problem_file.write_text(yaml.safe_dump(doc))
    return problem_file


def test_a_simulation_that_runs_out_of_time_fails_its_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stuck(argv: list[str], **kwargs: Any) -> None:
        fake.calls.append((argv, kwargs))
        return None  # killed when its time ran out

    monkeypatch.setattr(ddsim_run, "_execute", stuck)
    # inline too: a stage's limit stops the program it runs
    session = EvaluationSession.open(limit_simulation(problem_file, 90), [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert isinstance(outcome, RunOutcome)
    assert outcome.candidate.status == "failed"
    assert "ddsim ran longer than 90 s and was killed" in (outcome.candidate.message or "")
    ((_, kwargs),) = fake.calls
    assert kwargs["timeout"] == 90.0
    simulated = outcome.summary["samples"]["gamma"]["stages"]["simulation"]  # type: ignore[index]
    assert (simulated["timeout"], simulated["returncode"]) == (90.0, None)
    assert simulated["definition"]["timeout"] == 90  # what the problem says, recorded with it
    assert "ran longer than 90 s" in simulated["message"]
    assert (tmp_path / "a" / "samples" / "gamma" / "simulation" / "ddsim.stderr.log").is_file()


def test_a_problem_without_a_timeout_lets_ddsim_run(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert isinstance(outcome, RunOutcome)
    ((_, kwargs),) = fake.calls
    assert kwargs["timeout"] is None
    assert outcome.summary["samples"]["gamma"]["stages"]["simulation"]["timeout"] is None  # type: ignore[index]


def test_the_command_passes_the_problem_s_timeout_to_ddsim(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    problem = limit_simulation(problem_file, 1.5)
    output = str(tmp_path / "a")
    code, _, err = evaluate(str(problem), "--param", f"{RADIUS}=450", "--output", output)
    assert code == 0, err
    ((_, kwargs),) = fake.calls
    assert kwargs["timeout"] == 1.5


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
