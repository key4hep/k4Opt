"""``k4opt optimize`` of the geometry and the reconstruction at once: the ECAL cell size and an
argument of CLD's reconstruction (``--cms``), with the real ddsim and k4run (needs the Key4hep
stack and Optuna).

Skipped without ``$K4GEO``, ``$CLDCONFIG``, ``ddsim``, ``k4run``, ``podio`` or Optuna. Two
candidates of a few 50 GeV neutral pions, two at a time, then a resume for one more; a few
minutes, mostly ddsim. Each trial must have simulated and reconstructed on its own geometry, with
its own value of the argument, every simulated event.
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
EVENTS = 4


@pytest.fixture(scope="module")
def problem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    k4geo = os.environ.get("K4GEO")
    if not k4geo or not (Path(k4geo) / CLD_COMPACT).is_file() or not os.environ.get("CLDCONFIG"):
        pytest.skip("$K4GEO with the CLD geometry and $CLDCONFIG are required")
    if not all(shutil.which(p) for p in ("ddsim", "k4run")) or not importlib.util.find_spec(
        "podio"
    ):
        pytest.skip("ddsim, k4run and podio from the Key4hep stack are required")
    pytest.importorskip("optuna")
    doc = yaml.safe_load(EXAMPLE.read_text())
    sample = doc["samples"]["pi0_50gev"]
    sample["simulation"]["events"] = EVENTS
    del sample["reconstruction"]["cms"]  # a parameter sets it
    doc["parameters"]["reco_cms"] = {
        "title": "Centre-of-mass energy the reconstruction is configured for",
        "type": "categorical",
        "choices": [240, 365],
        "targets": {"reconstruction": "cms"},
    }
    path = tmp_path_factory.mktemp("problem") / "joint.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True))
    return path


def optimize(problem: Path, output: Path, *argv: str) -> str:
    out, err = io.StringIO(), io.StringIO()
    flags = [arg for evaluator in EVALUATORS for arg in ("--evaluator", evaluator)]
    command = ["optimize", str(problem), *flags, "--output", str(output), *argv]
    code = main(command, stdout=out, stderr=err)
    assert code == 0, out.getvalue() + err.getvalue()
    return out.getvalue()


def record(study: Path, number: int) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((study / f"trial-{number:06d}" / "summary.json").read_text())
    return data


def test_a_joint_study_reconstructs_each_candidate_on_its_own_geometry_and_settings(
    problem: Path, tmp_path: Path
) -> None:
    output = tmp_path / "joint"
    first = optimize(problem, output, "--trials", "2", "--seed", "3", "--workers", "2")
    assert "trials:   2: 2 valid" in first, first
    resumed = optimize(problem, output, "--trials", "1", "--resume", "--workers", "1")
    assert "trials:   3: 3 valid" in resumed, resumed
    study = json.loads((output / "study.json").read_text())
    for number in range(3):
        summary = record(output, number)
        run = output / f"trial-{number:06d}"
        candidate = summary["candidate"]
        stages = summary["samples"]["pi0_50gev"]["stages"]
        simulated, reconstructed = stages["simulation"], stages["reconstruction"]
        assert simulated["events_read"] == reconstructed["events_read"] == EVENTS
        # the candidate's own geometry: its cell size, read by the simulation and the job
        (change,) = summary["geometry"]["changes"]
        assert change["new_value"] == f"{candidate['ecal_cell_size']!r}*mm"
        compact = str(run / summary["geometry"]["compact"])
        assert f"--compactFile={compact}" in simulated["argv"]
        resolved = properties((run / reconstructed["logs"]["joboptions"]).read_text())
        assert resolved["GeoSvc.detectors"] == [compact]
        # the reconstruction's own argument, as the candidate sets it
        assert reconstructed["definition"]["cms"] == candidate["reco_cms"]
        assert f"--cms={candidate['reco_cms']}" in reconstructed["argv"]
        assert reconstructed["varies"] == ["ecal_cell_size", "reco_cms"]
        assert simulated["varies"] == ["ecal_cell_size"]
        # every trial ran the programs the study identified
        assert summary["environment"]["observed"]["programs"] == study["programs"]
