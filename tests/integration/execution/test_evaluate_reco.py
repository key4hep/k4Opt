"""``k4opt evaluate`` of the reconstruction example: the real ddsim, then CLD's reconstruction
with k4run, on a modified CLD geometry (needs the Key4hep stack).

Skipped without ``$K4GEO``, ``$CLDCONFIG``, ``ddsim``, ``k4run`` and ``podio``. One candidate of a
few 50 GeV neutral pions in worker processes, then the same simulation supplied to reconstruct
again; a few minutes, mostly ddsim.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.k4run import properties
from tests.support.evaluators import EXAMPLES
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

EXAMPLE = EXAMPLES / "problem" / "cld-ecal-pi0-reco.yaml"
EVALUATORS = (
    f"{EXAMPLES / 'evaluators' / 'pfo_pi0.py'}:PfoPi0",
    f"{EXAMPLES / 'evaluators' / 'ecal_channels.py'}:EcalChannels",
)
EVENTS = 5


@pytest.fixture(scope="module")
def problem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    k4geo = os.environ.get("K4GEO")
    if not k4geo or not (Path(k4geo) / CLD_COMPACT).is_file() or not os.environ.get("CLDCONFIG"):
        pytest.skip("$K4GEO with the CLD geometry and $CLDCONFIG are required")
    if not all(shutil.which(p) for p in ("ddsim", "k4run")) or not importlib.util.find_spec(
        "podio"
    ):
        pytest.skip("ddsim, k4run and podio from the Key4hep stack are required")
    doc = yaml.safe_load(EXAMPLE.read_text())
    doc["samples"]["pi0_50gev"]["simulation"]["events"] = EVENTS
    path = tmp_path_factory.mktemp("problem") / "problem.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True))
    return path


def evaluate(problem: Path, output: Path, *extra: str) -> tuple[str, dict[str, Any]]:
    out, err = io.StringIO(), io.StringIO()
    flags = [arg for evaluator in EVALUATORS for arg in ("--evaluator", evaluator)]
    argv = [*flags, "--param", "ecal_cell_size=8.0", "--output", str(output), *extra]
    code = main(["evaluate", str(problem), *argv], stdout=out, stderr=err)
    assert code == 0, out.getvalue() + err.getvalue()
    return out.getvalue(), json.loads((output / "summary.json").read_text())


@pytest.fixture(scope="module")
def run(problem: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    _, summary = evaluate(problem, tmp_path_factory.mktemp("runs") / "run", "--workers", "2")
    return summary


def test_the_candidate_is_reconstructed_on_its_own_geometry(run: dict[str, Any]) -> None:
    assert run["status"] == "valid"
    stage = run["samples"]["pi0_50gev"]["stages"]["reconstruction"]
    assert stage["status"] == "done" and stage["events_read"] == EVENTS
    root = Path(run["run_dir"])
    resolved = properties((root / stage["logs"]["joboptions"]).read_text())
    assert resolved["GeoSvc.detectors"] == [str(root / run["geometry"]["compact"])]
    assert "directory/CLDReconstruction.py" in stage["files"]
    k4run = run["environment"]["observed"]["programs"]["k4run"]
    assert stage["argv"][0] == k4run["path"] and k4run["path"].endswith("/k4run")
    resolved_fraction = next(m for m in run["metrics"] if m["metric"] == "pi0.resolved")
    assert 0 <= resolved_fraction["value"] <= 1


def test_a_supplied_simulation_is_reconstructed_again(
    problem: Path, run: dict[str, Any], tmp_path: Path
) -> None:
    root = Path(run["run_dir"])
    simulated = (
        root / run["samples"]["pi0_50gev"]["stages"]["simulation"]["outputs"]["events"]["path"]
    )
    out, summary = evaluate(
        problem, tmp_path / "again", "--input", f"pi0_50gev/simulation={simulated}"
    )
    assert "supplied:    pi0_50gev/simulation" in out
    stage = summary["samples"]["pi0_50gev"]["stages"]["reconstruction"]
    assert stage["status"] == "done" and stage["events_read"] == EVENTS
    # the same events, reconstructed again: the same measurement
    assert [m["value"] for m in summary["metrics"]] == [m["value"] for m in run["metrics"]]
