"""``k4opt evaluate`` on the real CLD geometry with the real ddsim (needs the Key4hep stack).

Skipped without ``$K4GEO``, ``ddsim`` and ``podio``. Runs two candidates of a few events each;
about a minute each, mostly ddsim start-up. Verify by hand with::

    source setup.sh && pytest tests/integration/problem -m integration -s
"""

from __future__ import annotations

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
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "problem" / "cld-ecal-hit-count.yaml"
CELL = "constants.ECal_cell_size"
EVENTS = 4


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
    doc["evaluation"]["events"] = EVENTS
    path = tmp_path_factory.mktemp("problem") / "problem.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def evaluate(problem: Path, size: str, output: Path) -> dict[str, Any]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["evaluate", str(problem), "--param", f"{CELL}={size}", "--output", str(output)],
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
        snapshots = Path(run["geometry"]["compact"]).parent.glob("ECalBarrel*.xml")
        assert any(f'name="ECal_cell_size" value="{size}*mm"' in f.read_text() for f in snapshots)


def test_the_installed_geometry_is_untouched(
    runs: dict[str, dict[str, Any]], installed_before: dict[Path, str]
) -> None:
    assert _geometry_files() == installed_before


def test_ddsim_simulated_the_materialized_geometry(runs: dict[str, dict[str, Any]]) -> None:
    for run in runs.values():
        log = Path(run["paths"]["stdout"]).read_text()
        assert run["geometry"]["compact"] in log
        assert run["geometry"]["source_root"] not in log  # nothing loaded from the original


def test_metric_comes_from_the_observed_events(runs: dict[str, dict[str, Any]]) -> None:
    from podio.reading import get_reader

    for run in runs.values():
        assert run["status"] == "valid"
        assert run["events"] == {"requested": EVENTS, "observed": EVENTS}
        collection = run["simulation"]["collection"]
        events = get_reader(run["paths"]["edm4hep"]).get("events")
        counts = [len({hit.getCellID() for hit in event.get(collection)}) for event in events]
        (measurement,) = run["measurements"]
        assert len(counts) == EVENTS
        assert measurement["value"] == pytest.approx(statistics.fmean(counts))
        assert measurement["uncertainty"] == pytest.approx(statistics.stdev(counts) / EVENTS**0.5)
        assert measurement["unit"] == "1"
        assert Path(run["paths"]["stderr"]).is_file()
        assert run["environment"]["declared"]["version"] == "2026-10-20"
        assert run["simulation"]["returncode"] == 0
