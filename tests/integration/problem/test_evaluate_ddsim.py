"""``k4opt evaluate`` on the real CLD geometry with the real ddsim (needs the Key4hep stack).

Skipped without ``$K4GEO``, ``ddsim`` and ``podio``. Runs two candidates of a few events each,
one more with two evaluators that share its simulation, and one of two samples (photons of 10 and
50 GeV); about a minute per simulation, mostly ddsim start-up. Verify by hand with::

    source setup.sh && pytest tests/integration/problem -m integration -s
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.problem import Assessment, parse_problem
from tests.support.evaluators import CALO_PHOTON, CELL_COUNT
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "problem" / "cld-ecal-granularity.yaml"
CELL = "constants.ECal_cell_size"
COLLECTION = "ECalBarrelCollection"
EVENTS = 4
SAMPLE = "gamma_10gev"  # the example's one sample
RESIDUAL_LIMIT = {
    "stage": "simulation",
    "metric": "ecal_barrel.angular_residual",
    "max": 1.0,
    "unit": "mrad",
}


@pytest.fixture(scope="module")
def stack() -> None:
    k4geo = os.environ.get("K4GEO")
    if not k4geo or not (Path(k4geo) / CLD_COMPACT).is_file():
        pytest.skip("$K4GEO with the CLD geometry is required")
    if shutil.which("ddsim") is None or importlib.util.find_spec("podio") is None:
        pytest.skip("ddsim and podio from the Key4hep stack are required")


@pytest.fixture(scope="module")
def problem(tmp_path_factory: pytest.TempPathFactory, stack: None) -> Path:
    doc = yaml.safe_load(EXAMPLE.read_text())
    doc["samples"]["gamma_10gev"]["simulation"]["events"] = EVENTS
    doc["constraints"] = [RESIDUAL_LIMIT]  # the one of docs/problem.md
    path = tmp_path_factory.mktemp("problem") / "problem.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def evaluate(
    problem: Path, size: str, output: Path, evaluators: tuple[str, ...] = (CALO_PHOTON,)
) -> dict[str, Any]:
    out, err = io.StringIO(), io.StringIO()
    flags = [arg for evaluator in evaluators for arg in ("--evaluator", evaluator)]
    code = main(
        ["evaluate", str(problem), *flags, "--param", f"{CELL}={size}", "--output", str(output)],
        stdout=out,
        stderr=err,
    )
    assert code == 0, out.getvalue() + err.getvalue()
    assert "status:      valid" in out.getvalue()
    summary: dict[str, Any] = json.loads((output / "summary.json").read_text())
    return summary


def _geometry_files() -> dict[Path, str]:
    root = Path(os.environ["K4GEO"]) / Path(CLD_COMPACT).parent
    return {path: path.read_text() for path in sorted(root.glob("*.xml"))}


@pytest.fixture(scope="module")
def installed_before(stack: None) -> dict[Path, str]:
    return _geometry_files()


@pytest.fixture(scope="module")
def runs(
    problem: Path, installed_before: dict[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, dict[str, Any]]:
    base = tmp_path_factory.mktemp("runs")
    return {size: evaluate(problem, size, base / size) for size in ("4.0", "8.0")}


def stage_file(run: dict[str, Any], sample: str, role: str) -> Path:
    """The file of the simulation of ``sample`` a record names: its output (``events``) or a log,
    by its path relative to the run directory."""
    stage = run["samples"][sample]["stages"]["simulation"]
    path = stage["outputs"]["events"]["path"] if role == "events" else stage["logs"][role]
    return Path(run["run_dir"]) / path


def compact(run: dict[str, Any]) -> Path:
    return Path(run["run_dir"]) / run["geometry"]["compact"]


def test_target_is_present_in_the_installed_geometry(stack: None) -> None:
    root = Path(os.environ["K4GEO"]) / Path(CLD_COMPACT).parent
    assert any('name="ECal_cell_size"' in f.read_text() for f in root.glob("ECalBarrel*.xml"))


def test_two_candidates_give_distinct_modified_geometries(
    runs: dict[str, dict[str, Any]],
) -> None:
    assert runs["4.0"]["geometry"]["fingerprint"] != runs["8.0"]["geometry"]["fingerprint"]
    for size, run in runs.items():
        (change,) = run["geometry"]["changes"]
        assert change["new_value"] == f"{size}*mm"
        snapshots = compact(run).parent.glob("ECalBarrel*.xml")
        assert any(f'name="ECal_cell_size" value="{size}*mm"' in f.read_text() for f in snapshots)


def test_the_installed_geometry_is_untouched(
    runs: dict[str, dict[str, Any]], installed_before: dict[Path, str]
) -> None:
    assert _geometry_files() == installed_before


def test_ddsim_simulated_the_materialized_geometry(runs: dict[str, dict[str, Any]]) -> None:
    for run in runs.values():
        log = stage_file(run, SAMPLE, "stdout").read_text()
        assert str(compact(run)) in log
        assert run["geometry"]["source_root"] not in log  # nothing loaded from the original


def test_metrics_come_from_the_observed_events(runs: dict[str, dict[str, Any]]) -> None:
    from podio.reading import get_reader

    for run in runs.values():
        assert run["status"] == "valid"
        assert run["samples"][SAMPLE]["stages"]["simulation"]["events_read"] == EVENTS
        (evaluator,) = run["samples"][SAMPLE]["evaluations"].values()
        collection = evaluator["collections"]["hits"]
        assert evaluator["settings"]["used"] == {"threshold": 70e-6}  # 70 keV, in GeV
        assert run["warnings"] == []  # CLD has the ECAL barrel readout the evaluator reads
        counts = []
        for event in get_reader(str(stage_file(run, SAMPLE, "events"))).get("events"):
            (photon,) = [p for p in event.get("MCParticles") if p.getGeneratorStatus() == 1]
            if not photon.isDecayedInTracker():
                counts.append(sum(hit.getEnergy() >= 70e-6 for hit in event.get(collection)))
        measured = {m["metric"]: m for m in evaluator["measurements"]}
        cells, residual = (
            measured["ecal_barrel.fired_cells"],
            measured["ecal_barrel.angular_residual"],
        )
        assert cells["value"] == pytest.approx(statistics.fmean(counts))
        assert cells["unit"] == "1"
        assert residual["unit"] == "mrad"
        assert 0 < residual["value"] < 20  # a 10 GeV photon: about a mrad
        assert stage_file(run, SAMPLE, "stderr").is_file()
        assert run["samples"][SAMPLE]["stages"]["simulation"]["outputs"]["events"]["sha256"] is None
        assert run["environment"]["declared"]["version"] == "2026-10-20"
        assert run["samples"]["gamma_10gev"]["stages"]["simulation"]["returncode"] == 0


@pytest.fixture(scope="module")
def shared(problem: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One candidate measured by the example evaluator and the test evaluator together."""
    doc = yaml.safe_load(problem.read_text())
    for item in (*doc["objectives"], *doc.get("constraints", [])):  # of one of two evaluators now
        item["evaluator"] = "calo.photon"
    doc["evaluators"].append(
        {
            "name": "test.cell_count",
            "version": "1",
            "stage": "simulation",
            "collections": {"hits": COLLECTION},
        }
    )
    both = problem.with_name("both.yaml")
    both.write_text(yaml.safe_dump(doc))
    output = tmp_path_factory.mktemp("shared") / "run"
    return evaluate(both, "6.0", output, (CALO_PHOTON, CELL_COUNT))


def test_two_evaluators_read_one_simulation(shared: dict[str, Any]) -> None:
    from podio.reading import get_reader

    run_dir = Path(shared["run_dir"])
    assert len(list(run_dir.glob("**/ddsim.json"))) == 1  # ddsim ran once
    photon, cells = shared["samples"][SAMPLE]["evaluations"].values()
    assert (photon["implementation"], photon["status"]) == ("calo.photon", "valid")
    assert (cells["implementation"], cells["status"]) == ("test.cell_count", "valid")
    chain = ["geometry", f"{SAMPLE}/generation", f"{SAMPLE}/simulation"]
    assert photon["reads"] == cells["reads"] == chain
    assert shared["samples"]["gamma_10gev"]["stages"]["simulation"]["status"] == "done"
    # the test evaluator's value, recomputed from the one file both read
    per_event = [
        len({hit.getCellID() for hit in event.get(COLLECTION)})
        for event in get_reader(str(stage_file(shared, SAMPLE, "events"))).get("events")
    ]
    (measurement,) = cells["measurements"]
    assert measurement["value"] == pytest.approx(statistics.fmean(per_event))
    # the objectives are calo.photon's, each read from it
    assert {m["evaluator"] for m in shared["metrics"]} == {"calo.photon"}


def test_the_records_hold_the_assessment_their_measurements_give(
    runs: dict[str, dict[str, Any]], shared: dict[str, Any]
) -> None:
    for run in (*runs.values(), shared):
        assessment = Assessment.from_summary(parse_problem(run["problem"]["definition"]), run)
        assert assessment.objectives is not None
        assert [m.metric for m in assessment.objectives] == [
            "ecal_barrel.angular_residual",
            "ecal_barrel.fired_cells",
        ]
        (limit,) = assessment.limits
        residual = assessment.objectives[0].value
        assert (limit.name, limit.value, limit.unit) == (
            f"{SAMPLE}/calo.photon/simulation/ecal_barrel.angular_residual:max",
            residual,
            "mrad",
        )
        assert assessment.feasible is (residual <= 1.0)  # four photons: either can happen


@pytest.fixture(scope="module")
def two_samples(problem: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One candidate measured by the example evaluator on photons of 10 and of 50 GeV."""
    doc = yaml.safe_load(problem.read_text())
    harder = copy.deepcopy(doc["samples"][SAMPLE])
    harder["generation"]["energy"]["value"] = 50
    harder["simulation"]["seed"] = 43
    doc["samples"]["gamma_50gev"] = harder
    doc["evaluators"][0]["samples"] = [SAMPLE, "gamma_50gev"]
    objectives = doc["objectives"]
    doc["objectives"] = [o | {"sample": s} for s in (SAMPLE, "gamma_50gev") for o in objectives]
    doc["constraints"][0]["sample"] = SAMPLE
    path = problem.with_name("two-samples.yaml")
    path.write_text(yaml.safe_dump(doc))
    return evaluate(path, "6.0", tmp_path_factory.mktemp("samples") / "run")


def test_two_samples_are_simulated_on_one_geometry_and_measured_each_on_its_own(
    two_samples: dict[str, Any],
) -> None:
    from podio.reading import get_reader

    run_dir = Path(two_samples["run_dir"])
    assert sorted(
        p.parent.parent.name for p in run_dir.glob("samples/*/simulation/ddsim.json")
    ) == [
        "gamma_10gev",
        "gamma_50gev",
    ]
    assert len(list(run_dir.glob("**/k4opt-manifest.json"))) == 1  # one geometry for both
    for sample, energy in ((SAMPLE, 10.0), ("gamma_50gev", 50.0)):
        part = two_samples["samples"][sample]
        assert part["stages"]["simulation"]["events_read"] == EVENTS
        (evaluator,) = part["evaluations"].values()
        assert evaluator["status"] == "valid"
        assert evaluator["reads"] == ["geometry", f"{sample}/generation", f"{sample}/simulation"]
        # each sample's file holds its own photons
        for event in get_reader(str(stage_file(two_samples, sample, "events"))).get("events"):
            (photon,) = [p for p in event.get("MCParticles") if p.getGeneratorStatus() == 1]
            assert photon.getEnergy() == pytest.approx(energy, rel=1e-4)  # stored as floats
    assert [m["label"] for m in two_samples["metrics"]][:4] == [
        f"{sample}/calo.photon/simulation/ecal_barrel.{metric}"
        for sample in (SAMPLE, "gamma_50gev")
        for metric in ("angular_residual", "fired_cells")
    ]
    assessment = Assessment.from_summary(
        parse_problem(two_samples["problem"]["definition"]), two_samples
    )
    assert assessment.objectives is not None and len(assessment.objectives) == 4


def test_pi0_example_combines_hashed_histograms_from_two_real_samples(
    tmp_path: Path, stack: None
) -> None:
    from podio.reading import get_reader

    example = EXAMPLE.with_name("cld-ecal-pi0-rejection.yaml")
    doc = yaml.safe_load(example.read_text())
    for sample in doc["samples"].values():
        sample["simulation"]["events"] = EVENTS
    path = tmp_path / "pi0.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    sources = example.parent.parent / "evaluators"
    run = evaluate(
        path,
        "8.0",
        tmp_path / "run",
        tuple(
            f"{sources / file}:{name}"
            for file, name in (
                ("shower_shape.py", "ShowerShape"),
                ("separation.py", "Separation"),
                ("ecal_channels.py", "EcalChannels"),
            )
        ),
    )
    root = Path(run["run_dir"])
    histograms = {}
    for sample, pdg in (("gamma_20gev", 22), ("pi0_20gev", 111)):
        assert run["samples"][sample]["stages"]["simulation"]["events_read"] == EVENTS
        for event in get_reader(str(stage_file(run, sample, "events"))).get("events"):
            (primary,) = [p for p in event.get("MCParticles") if p.getGeneratorStatus() == 1]
            assert primary.getPDG() == pdg
        artifact = run["samples"][sample]["evaluations"]["calo.shower_shape"]["artifacts"][
            "histogram"
        ]
        assert artifact["sha256"] is not None
        histograms[sample] = json.loads((root / artifact["path"]).read_text())
        histogram = histograms[sample]
        assert sum(histogram["counts"]) + histogram["underflow"] + histogram["overflow"] == EVENTS
    combined = run["evaluations"]["pi0_rejection"]
    assert combined["roles"] == {
        "signal": ["gamma_20gev/calo.shower_shape"],
        "background": ["pi0_20gev/calo.shower_shape"],
    }
    metrics = {m["metric"]: m["value"] for m in combined["measurements"]}
    background = histograms["pi0_20gev"]
    cut_index = background["edges"].index(metrics["width_cut"])
    assert metrics["background_rejection"] == pytest.approx(
        1 - sum(background["counts"][:cut_index]) / EVENTS
    )
    assert metrics["signal_efficiency"] >= 0.9
    assert len(list(root.glob("**/ddsim.json"))) == 2
    assert len(list(root.glob("**/k4opt-manifest.json"))) == 1
    assert run["evaluations"]["readout"]["status"] == "valid"
    assert Assessment.from_summary(parse_problem(run["problem"]["definition"]), run).feasible
