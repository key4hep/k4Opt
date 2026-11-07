"""``k4opt evaluate`` on the real CLD geometry with the real ddsim (needs the Key4hep stack).

Skipped without ``$K4GEO``, ``ddsim`` and ``podio``. Runs two candidates of a few events each,
one more with two evaluators that share its simulation, and one of two samples (photons of 10 and
50 GeV); about a minute per simulation, mostly ddsim start-up. Verify by hand with::

    source setup.sh && pytest tests/integration/problem -m integration -s
"""

from __future__ import annotations

import copy
import hashlib
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
from k4opt.evaluation import load_evaluator
from k4opt.execution import EvaluationSession, Rejected
from k4opt.problem import parse_problem
from k4opt.record import Assessment
from k4opt.software import Software
from tests.support.common import scientific
from tests.support.evaluators import CALO_PHOTON, CELL_COUNT
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "problem" / "cld-ecal-granularity.yaml"
CELL = "ecal_cell_size"
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
    problem: Path,
    size: str,
    output: Path,
    evaluators: tuple[str, ...] = (CALO_PHOTON,),
    workers: int | str = "inline",
) -> dict[str, Any]:
    out, err = io.StringIO(), io.StringIO()
    flags = [arg for evaluator in evaluators for arg in ("--evaluator", evaluator)]
    flags += ["--param", f"{CELL}={size}", "--workers", str(workers), "--output", str(output)]
    code = main(["evaluate", str(problem), *flags], stdout=out, stderr=err)
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
        # hashed when the stage ended, so that a stage reusing it can check it is still that file
        events = stage_file(run, SAMPLE, "events")
        output = run["samples"][SAMPLE]["stages"]["simulation"]["outputs"]["events"]
        assert output["sha256"] == hashlib.sha256(events.read_bytes()).hexdigest()
        software = run["environment"]["observed"]["software"]
        assert software["id"] == Software.now().id
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


@pytest.fixture(scope="module")
def pi0_runs(
    tmp_path_factory: pytest.TempPathFactory, stack: None
) -> dict[int | str, dict[str, Any]]:
    """The pi0-rejection example at a few events, evaluated inline and with four workers."""
    example = EXAMPLE.with_name("cld-ecal-pi0-rejection.yaml")
    doc = yaml.safe_load(example.read_text())
    doc["stages"]["simulation"]["events"] = EVENTS  # shared by both samples
    base = tmp_path_factory.mktemp("pi0")
    path = base / "pi0.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    sources = example.parent.parent / "evaluators"
    evaluators = tuple(
        f"{sources / file}:{name}"
        for file, name in (
            ("shower_shape.py", "ShowerShape"),
            ("separation.py", "Separation"),
            ("ecal_channels.py", "EcalChannels"),
        )
    )
    return {
        workers: evaluate(path, "8.0", base / f"workers{workers}", evaluators, workers)
        for workers in ("inline", 4)
    }


def test_pi0_example_combines_hashed_histograms_from_two_real_samples(
    pi0_runs: dict[int | str, dict[str, Any]],
) -> None:
    from podio.reading import get_reader

    run = pi0_runs["inline"]
    root = Path(run["run_dir"])
    histograms = {}
    for sample, pdg in (("gamma_50gev", 22), ("pi0_50gev", 111)):
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
        "signal": ["gamma_50gev/calo.shower_shape"],
        "background": ["pi0_50gev/calo.shower_shape"],
    }
    metrics = {m["metric"]: m["value"] for m in combined["measurements"]}
    background = histograms["pi0_50gev"]
    cut_index = background["edges"].index(metrics["cut"])
    assert metrics["background_rejection"] == pytest.approx(
        1 - sum(background["counts"][:cut_index]) / EVENTS
    )
    assert metrics["signal_efficiency"] >= 0.9
    assert len(list(root.glob("**/ddsim.json"))) == 2
    assert len(list(root.glob("**/k4opt-manifest.json"))) == 1
    assert run["evaluations"]["readout"]["status"] == "valid"
    assert Assessment.from_summary(parse_problem(run["problem"]["definition"]), run).feasible


def _unhashed(run: dict[str, Any]) -> dict[str, Any]:
    """``run`` without the SHA-256 of its stages' outputs."""
    run = copy.deepcopy(run)
    for sample in run["samples"].values():
        for stage in sample["stages"].values():
            for output in (stage.get("outputs") or {}).values():
                output.pop("sha256", None)
    return run


def _signatures(run: dict[str, Any]) -> dict[str, Any]:
    """The SHA-256 of each stage signature of ``run``, by sample and stage."""
    return {
        f"{name}/{stage_name}": (stage.get("signature") or {}).get("sha256")
        for name, sample in run["samples"].items()
        for stage_name, stage in sample["stages"].items()
    }


def test_pi0_example_gives_the_same_record_with_four_workers(
    pi0_runs: dict[int | str, dict[str, Any]],
) -> None:
    one, four = pi0_runs["inline"], pi0_runs[4]
    print(f"pi0 example: {one['elapsed_seconds']:.1f} s inline, ", end="")
    print(f"{four['elapsed_seconds']:.1f} s with four")
    assert four["environment"]["observed"]["execution"] == {"backend": "process", "workers": 4}
    # ROOT writes the same events a few bytes apart (its files carry when and where they were
    # written), so the outputs' sizes and hashes differ; everything else, the stages' signatures
    # included, is the same
    assert scientific(_unhashed(one), frozenset({"bytes"})) == scientific(
        _unhashed(four), frozenset({"bytes"})
    )
    assert _signatures(one) == _signatures(four)
    for sample in ("gamma_50gev", "pi0_50gev"):
        evaluation = four["samples"][sample]["evaluations"]["calo.shower_shape"]
        assert (Path(four["run_dir"]) / evaluation["log"]).is_file()
        histogram = evaluation["artifacts"]["histogram"]
        assert histogram == one["samples"][sample]["evaluations"]["calo.shower_shape"]["artifacts"][
            "histogram"
        ] | {"path": histogram["path"]}


def test_a_worker_that_read_events_with_podio_runs_the_next_simulation(
    problem: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """ROOT may leave threads behind in a worker that read a file; ddsim still starts there, by
    a launcher of its own rather than code run between fork and exec."""
    base = tmp_path_factory.mktemp("reused")
    with EvaluationSession.open(problem, [load_evaluator(CALO_PHOTON)], workers=2) as session:
        # the worker of a simulation counts its events with podio, and runs the next one too
        runs = [session.evaluate({CELL: size}, base / str(size)) for size in (6.0, 7.0)]
    for run in runs:
        assert run.candidate.status == "valid", run.candidate.message
        assert run.events == {SAMPLE: (EVENTS, EVENTS)}


def test_parameters_set_a_ddsim_option_and_the_gun_energy_that_ddsim_accepts(
    problem: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A range cut and the photon energy, each a parameter: the installed ddsim takes
    ``--physics.rangecut=0.5*mm`` and ``--gun.energy=12.0*GeV``, and the file it writes holds
    photons of that energy, which the evaluator measures."""
    from podio.reading import get_reader

    doc = yaml.safe_load(problem.read_text())
    del doc["samples"][SAMPLE]["generation"]["energy"]
    doc["parameters"] |= {
        "range_cut": {
            "type": "continuous",
            "bounds": [0.1, 1.0],
            "unit": "mm",
            "targets": {"simulation": "physics.rangecut"},
        },
        "photon_energy": {
            "type": "continuous",
            "bounds": [5.0, 50.0],
            "unit": "GeV",
            "targets": {"generation": "energy"},
        },
    }
    base = tmp_path_factory.mktemp("bound")
    path = base / "problem.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))  # the parameters in this order
    candidate = {CELL: 6.0, "range_cut": 0.5, "photon_energy": 12.0}
    with EvaluationSession.open(path, [load_evaluator(CALO_PHOTON)]) as session:
        outcome = session.evaluate(candidate, base / "run")
        spec = session.spec
    assert outcome.candidate.status == "valid", outcome.candidate.message
    stages = outcome.summary["samples"][SAMPLE]["stages"]  # type: ignore[index]
    argv = stages["simulation"]["argv"]
    assert "--physics.rangecut=0.5*mm" in argv and "--gun.energy=12.0*GeV" in argv
    assert stages["simulation"]["definition"]["physics.rangecut"] == "0.5*mm"
    assert stages["generation"]["definition"]["energy"] == {"value": 12.0, "unit": "GeV"}
    assert stages["simulation"]["varies"] == [CELL, "range_cut", "photon_energy"]
    events = Path(outcome.run_dir) / stages["simulation"]["outputs"]["events"]["path"]
    for event in get_reader(str(events)).get("events"):
        (primary,) = [p for p in event.get("MCParticles") if p.getGeneratorStatus() == 1]
        assert primary.getEnergy() == pytest.approx(12.0, rel=1e-6)
    assert Assessment.from_summary(spec, outcome.summary) == outcome.assessment


def test_the_example_of_every_destination_previews_on_the_installed_geometry(
    stack: None,
) -> None:
    """``--dry-run`` on the real CLD geometry: what each parameter sets, and nothing simulated."""
    example = EXAMPLE.with_name("cld-ecal-bindings.yaml")
    out, err = io.StringIO(), io.StringIO()
    params = ["ecal_cell_size=8.0", "range_cut=0.5", "cell_threshold=70"]
    argv = ["evaluate", str(example), "--evaluator", CALO_PHOTON, "--dry-run"]
    code = main([*argv, *(arg for p in params for arg in ("--param", p))], stdout=out, stderr=err)
    assert code == 0, out.getvalue() + err.getvalue()
    lines = out.getvalue().splitlines()
    at = lines.index("ecal_cell_size = 8.0 mm")
    assert lines[at + 1].startswith("  geometry: constants.ECal_cell_size (")
    assert lines[at + 2] == "    5.1*mm -> 8.0*mm"
    at = lines.index("range_cut = 0.5 mm")
    assert lines[at + 1 : at + 4] == [
        "  simulation: physics.rangecut",
        "    gamma_10gev: 0.5*mm",
        "    gamma_50gev: 0.5*mm",
    ]
    at = lines.index("cell_threshold = 70.0 keV")
    assert lines[at + 1 : at + 3] == [
        "  evaluator calo.photon: threshold",
        "    gamma_10gev, gamma_50gev: 70.0 keV",
    ]
    assert lines[-1] == "No simulation executed; no evaluation output created."


def test_a_simulation_killed_at_its_time_limit_fails_its_sample_and_the_session_goes_on(
    problem: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The real ddsim, killed at the time limit the problem gives one sample's simulation (it
    takes longer than 2 s only to start): that sample fails with its logs, the other sample of
    the candidate is measured, and the next candidate runs in the same session's worker."""
    doc = yaml.safe_load(problem.read_text())
    unlimited = copy.deepcopy(doc["samples"][SAMPLE])
    doc["samples"]["unlimited"] = unlimited
    doc["samples"][SAMPLE]["simulation"]["timeout"] = 2
    doc["evaluators"][0]["samples"] = [SAMPLE, "unlimited"]
    objectives = doc["objectives"]
    doc["objectives"] = [o | {"sample": s} for s in (SAMPLE, "unlimited") for o in objectives]
    doc["constraints"][0]["sample"] = SAMPLE
    path = problem.with_name("limited.yaml")
    path.write_text(yaml.safe_dump(doc))
    base = tmp_path_factory.mktemp("limited")
    with EvaluationSession.open(path, [load_evaluator(CALO_PHOTON)], workers=1) as session:
        runs = [session.evaluate({CELL: size}, base / str(size)) for size in (6.0, 7.0)]
    for run in runs:
        assert not isinstance(run, Rejected)
        summary = json.loads(run.summary_path.read_text())
        killed = summary["samples"][SAMPLE]["stages"]["simulation"]
        assert killed["status"] == "failed" and killed["timeout"] == 2
        assert "ddsim ran longer than 2 s and was killed" in killed["message"]
        assert (run.run_dir / killed["logs"]["stderr"]).is_file()
        measured = summary["samples"]["unlimited"]["evaluations"]["calo.photon"]
        assert measured["status"] == "valid"
        assert summary["status"] == "failed"  # an execution failure, recorded: no error
        assessment = Assessment.from_summary(
            parse_problem(summary["problem"]["definition"]), summary
        )
        assert assessment.feasible is None and assessment.objectives is None
