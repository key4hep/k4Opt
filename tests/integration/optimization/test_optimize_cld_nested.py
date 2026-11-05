"""The nested example with the real ddsim and k4run (needs the Key4hep stack and Optuna): for each
ECAL cell size, a study of Pandora's ECAL hit threshold, whose best trial is reconstructed again on
new events.

Skipped without ``$K4GEO``, ``$CLDCONFIG``, ``ddsim``, ``k4run``, ``podio`` or Optuna. Two cell
sizes of three tuning trials each on a few 50 GeV neutral pions, the second after a resume; about
five minutes, mostly ddsim and k4run.
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
from k4opt.optimization import load_study
from k4opt.problem.candidate import replicate_seed
from tests.support.evaluators import EXAMPLES
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

EXAMPLE = EXAMPLES / "nested" / "cld-pi0-nested.yaml"
EVALUATORS = (
    f"{EXAMPLES / 'evaluators' / 'pfo_pi0.py'}:PfoPi0",
    f"{EXAMPLES / 'evaluators' / 'ecal_channels.py'}:EcalChannels",
)
EVENTS = 6
TUNING = 3
SEED = 42  # the simulation's
PANDORA = "MyDDMarlinPandora_10ns.Parameters"
THRESHOLD = f"{PANDORA}[ECalMipThreshold]"


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
    doc["samples"]["pi0_50gev"]["simulation"]["events"] = EVENTS
    doc["optimization"]["inner"]["trials"] = TUNING
    path = tmp_path_factory.mktemp("problem") / "nested.yaml"
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


def test_each_cell_size_is_judged_at_its_best_threshold_on_new_events(
    problem: Path, tmp_path: Path
) -> None:
    output = tmp_path / "nested"
    first = optimize(problem, output, "--trials", "1", "--seed", "3", "--workers", "4")
    assert first.startswith("For each ECAL cell size (1 trial):"), first
    assert "simulation 2 · reconstruction 4 · pfo.pi0 4 · readout 1" in first, first
    resumed = optimize(problem, output, "--trials", "1", "--resume", "--workers", "4")
    assert "For each" not in resumed, resumed
    study = load_study(output)
    assert len(study.trials) == 2
    for trial in study.trials:
        inner, selected = trial.inner, trial.selected
        assert inner is not None and selected is not None, trial.message
        assert len(inner.trials) == TUNING
        assert inner.scope.fixed == {"ecal_cell_size": trial.candidate["ecal_cell_size"]}
        assert selected["values"] == {"ecal_mip_threshold": trial.candidate["ecal_mip_threshold"]}
        # the trial's record: the best threshold, reconstructed on new events
        summary = record(output, trial.number)
        stages = summary["samples"]["pi0_50gev"]["stages"]
        simulated, reconstructed = stages["simulation"], stages["reconstruction"]
        assert (summary["level"], summary["replicate"]) == (0, 1)
        assert simulated["definition"]["seed"] == replicate_seed(SEED, 1)
        assert simulated["events_read"] == reconstructed["events_read"] == EVENTS
        assert reconstructed["definition"][THRESHOLD] == trial.candidate["ecal_mip_threshold"]
        run = output / f"trial-{trial.number:06d}"
        resolved = properties((run / reconstructed["logs"]["joboptions"]).read_text())
        held = resolved[PANDORA]
        assert isinstance(held, dict)
        assert held["ECalMipThreshold"] == [repr(trial.candidate["ecal_mip_threshold"])]
        # the tuning: one simulation, on the problem's seed, a reconstruction for each threshold
        made = []
        for tuned in inner.trials:
            assert inner.directory is not None
            stages = record(inner.directory, tuned.number)["samples"]["pi0_50gev"]["stages"]
            assert stages["simulation"]["definition"]["seed"] == SEED
            made.append("reused" not in stages["simulation"])
            assert "reused" not in stages["reconstruction"]
        assert sum(made) == 1, made
