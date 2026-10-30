"""``run_candidate`` and ``k4opt evaluate`` with a fake ddsim, on the mini geometry."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt import __version__
from k4opt.cli import main
from k4opt.errors import K4OptError
from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationRequest,
    Quantity,
    collection_names,
    metric_units,
    result_from,
    with_defaults,
)
from k4opt.geometry import GeometrySource, Override
from k4opt.problem import CandidateError, ProblemConfigError, load_problem, parse_problem
from k4opt.problem._run import SUMMARY_NAME, RunOutcome, _parse_candidate, _xml_value
from k4opt.problem._run import run_candidate as _run_candidate
from k4opt.problem.errors import RunError
from tests.support import simulation
from tests.support.common import tree_digest
from tests.support.evaluators import (
    CELL_COUNT,
    HIT_ENERGY,
    PARAMETER_ECHO,
    calo_photon,
    cell_count,
    hit_energy,
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
    """``run_candidate`` with the test evaluator unless the test passes others."""
    changes.setdefault("evaluators", [cell_count()])
    return _run_candidate(problem, assignments, output, **changes)


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def problem_file(tmp_path: Path, doc: dict[str, Any]) -> Path:
    doc["geometry"] = MINI_COMPACT
    doc["parameters"][0]["geometry"] = {"constant": "InnerTracker_Barrel_radius_1"}
    doc["parameters"][0]["bounds"] = [100.0, 600.0]
    doc["stages"]["simulation"].update(events=3, seed=5)
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
    (evaluator,) = on_disk["evaluators"]
    assert (evaluator["name"], evaluator["version"]) == ("test.cell_count", "1")
    assert (evaluator["status"], evaluator["message"]) == ("valid", None)
    (measurement,) = evaluator["measurements"]
    assert measurement["metric"] == "ecal_barrel.mean_hit_cells"
    assert measurement["value"] == 3.0  # distinct cells per event: 3, 2, 4
    # the objective, from the evaluator that owns it
    assert on_disk["metrics"] == [
        {
            "stage": "simulation",
            "metric": "ecal_barrel.mean_hit_cells",
            "evaluator": "test.cell_count",
        }
        | {k: measurement[k] for k in ("value", "unit", "uncertainty")}
    ]
    assert on_disk["events"] == {"requested": 3, "observed": 3}
    assert on_disk["geometry"]["fingerprint"] == outcome.fingerprint
    assert Path(on_disk["geometry"]["manifest"]).is_file()
    assert on_disk["geometry"]["changes"] == [
        {"target": RADIUS, "old_value": "400*mm", "new_value": "450.0*mm"}
    ]
    # the stages, recorded once, apart from the evaluators that read them
    assert list(on_disk["stages"]) == ["generation", "simulation"]
    simulated = on_disk["stages"]["simulation"]
    assert simulated["argv"][0] == "/stack/bin/ddsim"
    assert (simulated["returncode"], simulated["events_read"]) == (0, 3)
    assert "error" not in simulated
    assert evaluator["settings"] == {"declared": {}, "used": {}}
    assert evaluator["collections"] == {"hits": COLLECTION}
    (warning,) = on_disk["warnings"]
    assert warning.startswith(UNEXPECTED) and warning.endswith("The simulation's output decides.")
    assert evaluator["stage"] == "simulation"
    # the code of the evaluator, which its name and version only label
    source = Path(evaluator["source"]["path"])
    assert source.name == "cell_count.py"
    assert evaluator["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert evaluator["provenance"] == {} and evaluator["artifacts"] == {}  # none of its own
    assert evaluator["stages"] == ["generation", "simulation"]
    assert evaluator["workdir"] == str(tmp_path / "a" / "evaluators" / "test.cell_count")
    # every second once: each stage, the evaluator's own call, and the candidate around them
    seconds = [stage["elapsed_seconds"] for stage in on_disk["stages"].values()]
    seconds.append(evaluator["elapsed_seconds"])
    assert min(seconds) >= 0 and sum(seconds) <= on_disk["elapsed_seconds"]
    # the files of the stages, named <stage>.<role>: what the real-ddsim test reads back
    paths = on_disk["paths"]
    assert paths["run_dir"] == str(tmp_path / "a")
    for role in ("stdout", "stderr", "command", "edm4hep"):
        assert Path(paths[f"simulation.{role}"]).is_file()
    assert {"stdout", "stderr", "edm4hep"}.isdisjoint(paths)  # no unprefixed leftovers
    assert on_disk["stages"]["generation"]["particle_gun"]["particle"] == "gamma"
    assert on_disk["environment"]["declared"] == {"channel": "nightly", "version": "2026-10-20"}
    assert on_disk["environment"]["observed"]["ddsim"] == "/stack/bin/ddsim"
    assert on_disk["environment"]["observed"]["k4opt"] == __version__
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


def test_a_conflicting_ddsim_option_is_rejected_before_any_side_effect(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["stages"]["simulation"]["numberOfEvents"] = 5
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
    assert summary["message"].startswith("test.cell_count failed: ddsim exited with status 2")
    assert "G4Exception: bad volume" in summary["message"]
    (evaluator,) = summary["evaluators"]
    assert evaluator["measurements"] == [] and evaluator["elapsed_seconds"] is None
    assert evaluator["workdir"] is None  # not called, so no directory was made for it
    assert summary["metrics"][0]["value"] is None  # never a made-up number
    assert summary["stages"]["simulation"]["returncode"] == 2
    assert "bad volume" in summary["stages"]["simulation"]["error"]
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
    problem = parse_problem(yaml.safe_load((EXAMPLES / "cld-ecal-granularity.yaml").read_text()))
    evaluator = calo_photon()
    (entry,) = problem.evaluators
    assert (entry.name, entry.version, entry.stage) == (
        evaluator.name,
        evaluator.version,
        evaluator.stage,
    )
    assert [(o.stage, o.metric) for o in problem.objectives] == [
        ("simulation", "ecal_barrel.angular_residual"),
        ("simulation", "ecal_barrel.fired_cells"),
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
    assert "evaluator:   test.cell_count 1: valid" in out
    assert "simulation/ecal_barrel.mean_hit_cells = 3 +- 0.57735 [1]" in out
    assert "events:      3 of 3" in out
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
    assert "events:" not in out  # ddsim wrote no file, so no events were counted
    assert "k4opt: evaluation failed: test.cell_count failed: ddsim exited with status 1" in err


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
    sample = tmp_path / "zh.hepmc"
    sample.write_bytes(b"HepMC::Version 3\n")
    doc = yaml.safe_load(problem_file.read_text(encoding="utf-8"))
    doc["stages"]["generation"] = {"file": "zh.hepmc"}  # next to the problem file
    problem_file.write_text(yaml.safe_dump(doc), encoding="utf-8")

    outcome = run_candidate(problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "s")

    ((argv, _),) = fake.calls
    assert f"--inputFiles={sample}" in argv and not any(a.startswith("--gun") for a in argv)
    generation = outcome.summary["stages"]["generation"]  # type: ignore[index]
    assert generation["file"] == str(sample)
    assert generation["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert outcome.candidate.status == "valid" and outcome.candidate.represents_candidate


def test_a_missing_generation_file_is_a_clean_error_before_any_side_effect(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text(encoding="utf-8"))
    doc["stages"]["generation"] = {"file": "gone.hepmc"}
    problem_file.write_text(yaml.safe_dump(doc), encoding="utf-8")
    output = tmp_path / "runs" / "s"
    with pytest.raises(EvaluationContractError, match=r"stages\.generation\.file"):
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
        f"simulation={events_file}",
        "--output",
        str(tmp_path / "a"),
    )
    assert code == 0, err
    assert fake.calls == []  # nothing was simulated
    assert f"supplied:    simulation from {events_file}" in out
    assert "this is no evaluation of the candidate" in out
    summary = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    assert (summary["represents_candidate"], summary["supplied"]) == (False, ["simulation"])
    assert summary["stages"] == {
        "simulation": {
            "provided": str(events_file),
            "bytes": 4,
            "sha256": hashlib.sha256(b"fake").hexdigest(),
            "events_read": 3,
            "elapsed_seconds": summary["stages"]["simulation"]["elapsed_seconds"],
        }
    }
    assert summary["metrics"][0]["value"] == 3.0


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ("simulation", "expected STAGE=FILE"),
        ("reconstruction={file}", "unknown stage; use generation, simulation"),
        ("simulation={missing}", "is not a file"),
        ("generation={file}", "no evaluator of the problem reads that stage"),
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
        argument = [argument, f"simulation={supplied}"]
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
            inputs=[f"simulation={supplied}", f"simulation={supplied}"],
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
        self.inner: Any = cell_count()
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
    (evaluator,) = on_disk["evaluators"]
    assert evaluator["provenance"] == {"selection": "barrel", "algo": [2, 1]}
    assert on_disk["stages"]["simulation"]["returncode"] == 0  # the stage's record stays apart


def test_the_preflight_request_has_the_conditions_the_evaluation_gets(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    spy = Spy()
    run_candidate(problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "a", evaluators=[spy])
    (evaluated,) = spy.evaluated
    # once before anything is created and once more when the candidate is evaluated
    assert len(spy.checked) == 2
    for checked in spy.checked:
        assert (checked.events, checked.seed) == (3, 5)  # those of the simulation it reads
        for field in ("settings", "events", "seed", "candidate"):
            assert getattr(checked, field) == getattr(evaluated, field)


# --- several evaluators of one candidate ---------------------------------------------------

CELLS = ("simulation", "ecal_barrel.mean_hit_cells")
ENERGY = ("simulation", "ecal_barrel.mean_hit_energy")
ECHO = ("analysis", "candidate.first_value")


def declare(problem_file: Path, names: list[str], objectives: list[tuple[str, str]]) -> None:
    """Make the problem declare the test evaluators ``names`` and these objectives."""
    doc = yaml.safe_load(problem_file.read_text())
    reads = {"stage": "simulation", "collections": {"hits": COLLECTION}}
    doc["evaluators"] = [
        {"name": name, "version": "1"} | ({} if name == "test.parameter_echo" else reads)
        for name in names
    ]
    doc["objectives"] = [{"stage": s, "metric": m, "direction": "minimize"} for s, m in objectives]
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
    assert sorted(p.name for p in output.iterdir()) == [
        "evaluators",
        "geometry",
        "simulation",
        SUMMARY_NAME,
    ]
    assert sorted(p.name for p in (output / "evaluators").iterdir()) == [
        "test.cell_count",
        "test.hit_energy",
    ]
    assert out.index("evaluator:   test.cell_count 1: valid") < out.index(
        "evaluator:   test.hit_energy 1: valid"
    )
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["status"] == "valid"
    assert [e["name"] for e in summary["evaluators"]] == ["test.cell_count", "test.hit_energy"]
    assert [(m["metric"], m["evaluator"]) for m in summary["metrics"]] == [
        ("ecal_barrel.mean_hit_cells", "test.cell_count"),
        ("ecal_barrel.mean_hit_energy", "test.hit_energy"),
    ]
    assert summary["metrics"][1]["value"] == pytest.approx(10 / 3)  # 4, 2 and 4 hits of 1 GeV
    assert list(summary["stages"]) == ["generation", "simulation"]


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
    assert errors(err).startswith("k4opt: evaluation failed: test.cell_count failed: ddsim exited")
    assert "evaluator:   test.parameter_echo 1: valid" in out
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["status"] == "failed"
    cells, echo = summary["evaluators"]
    assert (cells["status"], cells["elapsed_seconds"], cells["workdir"]) == ("failed", None, None)
    assert echo["status"] == "valid" and echo["measurements"][0]["value"] == 450.0
    # the measurement that was made is kept; the missing one is missing, not made up
    assert [m["value"] for m in summary["metrics"]] == [None, 450.0]
    assert summary["stages"]["simulation"]["error"].startswith("ddsim exited with status 2")


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
        """The summary as written, without its directory and its times."""
        data = json.loads(outcome.summary_path.read_text().replace(str(outcome.run_dir), "<run>"))

        def untimed(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: 0 if k == "elapsed_seconds" else untimed(v) for k, v in node.items()}
            return [untimed(item) for item in node] if isinstance(node, list) else node

        return json.dumps(untimed(data))  # key order included

    assert normalized(first) == normalized(second)


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
    (evaluator,) = summary["evaluators"]
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
    gated = type("Gated", (type(cell_count()),), {"settings": {"threshold": Quantity("GeV")}})()
    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[gated])
    on_disk = json.loads(outcome.summary_path.read_text())
    (evaluator,) = on_disk["evaluators"]
    assert evaluator["settings"] == {
        "declared": {"threshold": {"value": 70, "unit": "keV"}},
        "used": {"threshold": 70e-6},
    }
