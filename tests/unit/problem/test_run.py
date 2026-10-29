"""``run_candidate`` and ``k4opt evaluate`` with a fake ddsim, on the mini geometry."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.ddsim import DdsimError
from k4opt.errors import K4OptError
from k4opt.evaluation import EvaluationContractError, metric_units
from k4opt.geometry import GeometrySource, Override
from k4opt.problem import CandidateError, load_problem, parse_problem
from k4opt.problem._run import SUMMARY_NAME, RunOutcome, _parse_candidate, _xml_value
from k4opt.problem._run import run_candidate as _run_candidate
from k4opt.problem.errors import RunError
from tests.support import simulation
from tests.support.common import tree_digest
from tests.support.evaluators import CALO_HIT_MULTIPLICITY, calo_hit_multiplicity
from tests.support.geometry import MINI_COMPACT, MINI_GEO
from tests.support.simulation import COLLECTION, events_with

RADIUS = "constants.InnerTracker_Barrel_radius_1"
EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "problem"


def run_candidate(
    problem: Path, assignments: list[str], output: Path, **changes: Any
) -> RunOutcome:
    """``run_candidate`` with the example evaluator unless the test passes another."""
    changes.setdefault("evaluator", calo_hit_multiplicity())
    return _run_candidate(problem, assignments, output, **changes)


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def problem_file(tmp_path: Path, doc: dict[str, Any]) -> Path:
    doc["geometry"] = {"entry": MINI_COMPACT}
    doc["parameters"][0]["geometry"] = {"constant": "InnerTracker_Barrel_radius_1"}
    doc["parameters"][0]["bounds"] = [100.0, 600.0]
    evaluation = doc["evaluation"]
    evaluation.update(evaluator="calo.hit_multiplicity", events=3, seed=5)
    evaluation["settings"] = {"collection": COLLECTION}
    evaluation["generation"] = {
        "particle": "gamma",
        "energy": {"value": 10, "unit": "GeV"},
        "direction": {
            "distribution": "uniform",
            "theta": [60, 120],
            "phi": [0, 360],
            "unit": "deg",
        },
    }
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
    assert outcome.record.result.status == "valid"
    assert outcome.record.request.inputs["geometry"] == snapshot / MINI_COMPACT


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

    assert on_disk["status"] == "valid"
    assert on_disk["candidate"] == {RADIUS: 450.0}
    assert on_disk["evaluator"] == {"name": "calo.hit_multiplicity", "version": "1"}
    (measurement,) = on_disk["measurements"]
    assert measurement["metric"] == "ecal_barrel.mean_hit_cells"
    assert measurement["value"] == 3.0  # distinct cells per event: 3, 2, 4
    assert on_disk["events"] == {"requested": 3, "observed": 3}
    assert on_disk["geometry"]["fingerprint"] == outcome.fingerprint
    assert Path(on_disk["geometry"]["manifest"]).is_file()
    assert on_disk["geometry"]["changes"] == [
        {"target": RADIUS, "old_value": "400*mm", "new_value": "450.0*mm"}
    ]
    assert on_disk["simulation"]["command"][0] == "/stack/bin/ddsim"
    assert on_disk["simulation"]["returncode"] == 0
    assert on_disk["settings"] == {"collection": COLLECTION}
    # the files of the stages, named <stage>.<role>: what the real-ddsim test reads back
    paths = on_disk["paths"]
    assert paths["run_dir"] == str(tmp_path / "a")
    for role in ("stdout", "stderr", "command", "edm4hep"):
        assert Path(paths[f"simulation.{role}"]).is_file()
    assert {"stdout", "stderr", "edm4hep"}.isdisjoint(paths)  # no unprefixed leftovers
    assert on_disk["generation"]["particle_gun"]["particle"] == "gamma"
    assert on_disk["environment"]["declared"] == {"channel": "nightly", "version": "2026-10-20"}
    assert on_disk["environment"]["observed"]["ddsim"] == "/stack/bin/ddsim"
    assert on_disk["problem"]["path"] == str(problem_file.absolute())
    assert on_disk["candidate_key"] == [[RADIUS, "continuous", "num", "45e1"]]


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


def test_a_conflicting_backend_argument_is_rejected_before_any_side_effect(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluation"]["backend"]["ddsim"]["args"] += ["--numberOfEvents", "5"]
    problem_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(DdsimError, match="manages"):
        run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert not (tmp_path / "a").exists()
    assert "materialize" not in guard


def test_another_evaluator_is_rejected_before_any_side_effect(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluation"]["evaluator"] = "calo.single_particle_shower"
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
    assert err.startswith("k4opt: error: ")
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
    assert "G4Exception: bad volume" in summary["message"]
    assert summary["measurements"] == []
    assert summary["simulation"]["returncode"] == 2
    assert "bad volume" in Path(summary["paths"]["simulation.stderr"]).read_text()


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
    problem = parse_problem(yaml.safe_load((EXAMPLES / "cld-ecal-hit-count.yaml").read_text()))
    evaluator = calo_hit_multiplicity()
    assert (problem.evaluation.evaluator, problem.evaluation.version) == (
        evaluator.name,
        evaluator.version,
    )
    assert [(o.stage, o.metric) for o in problem.objectives] == [
        ("simulation", "ecal_barrel.mean_hit_cells")
    ]
    assert metric_units(evaluator) == {("simulation", "ecal_barrel.mean_hit_cells"): "1"}


# --- the command ---------------------------------------------------------------------------


def evaluate(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(["evaluate", "--evaluator", CALO_HIT_MULTIPLICITY, *argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_command_prints_the_result_and_exits_zero(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 0, err
    assert "status:      valid" in out
    assert "simulation/ecal_barrel.mean_hit_cells = 3 +- 0.57735 [1]" in out
    assert "events:      3 of 3" in out
    assert "fingerprint: " in out
    assert f"summary:     {tmp_path / 'a' / SUMMARY_NAME}" in out
    assert "ddsim.stderr.log" in out
    assert err == ""


def test_command_exits_nonzero_for_a_failed_run(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=1, stderr="boom")
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 1
    assert "status:      failed" in out
    assert "k4opt: evaluation failed: ddsim exited with status 1" in err


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


def test_a_problem_with_a_sample_simulates_it_and_records_its_hash(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    sample = tmp_path / "zh.hepmc"
    sample.write_bytes(b"HepMC::Version 3\n")
    doc = yaml.safe_load(problem_file.read_text(encoding="utf-8"))
    doc["evaluation"]["generation"] = {"sample": "zh.hepmc"}  # next to the problem file
    problem_file.write_text(yaml.safe_dump(doc), encoding="utf-8")

    outcome = run_candidate(problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "s")

    ((argv, _),) = fake.calls
    assert f"--inputFiles={sample}" in argv and not any(a.startswith("--gun") for a in argv)
    generation = outcome.summary["generation"]
    assert isinstance(generation, dict)
    assert generation["sample"] == str(sample)
    assert generation["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert outcome.record.result.status == "valid"


def test_a_missing_sample_is_a_clean_error_before_any_side_effect(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text(encoding="utf-8"))
    doc["evaluation"]["generation"] = {"sample": "gone.hepmc"}
    problem_file.write_text(yaml.safe_dump(doc), encoding="utf-8")
    output = tmp_path / "runs" / "s"
    with pytest.raises(EvaluationContractError, match=r"generation\.sample"):
        run_candidate(problem_file, [f"{RADIUS}=450.0"], output)
    assert not output.exists() and fake.calls == []


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
    """The example evaluator, recording the requests of check() and evaluate()."""

    def __init__(self) -> None:
        self.inner: Any = calo_hit_multiplicity()
        self.name, self.version = self.inner.name, self.inner.version
        self.requires, self.metrics = self.inner.requires, self.inner.metrics
        self.checked: list[Any] = []
        self.evaluated: list[Any] = []

    def check(self, request: Any) -> None:
        self.checked.append(request)
        self.inner.check(request)

    def evaluate(self, request: Any) -> Any:
        self.evaluated.append(request)
        return self.inner.evaluate(request)


def test_the_preflight_request_has_the_generation_the_evaluation_gets(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    spy = Spy()
    run_candidate(problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "a", evaluator=spy)
    (evaluated,) = spy.evaluated
    # once before anything is created and once more when the candidate is evaluated
    assert len(spy.checked) == 2
    for checked in spy.checked:
        assert checked.generation["particle"] == "gamma"
        for field in ("generation", "settings", "events", "seed", "backend_args", "candidate"):
            assert getattr(checked, field) == getattr(evaluated, field)
