"""End-to-end check with DD4hep itself: a snapshot must load exactly like its source.

A byte-identical copy of the reachable files proves little if a resource was missed or an
include broke; loading both with DD4hep does. Needs the DD4hep Python bindings (the Key4hep
stack) and a k4geo tree; skipped otherwise. DD4hep keeps global state, so each load runs in
a fresh process.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from k4opt.geometry import GeometrySource, Override
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

LOADER = """
import json, sys
import dd4hep
description = dd4hep.Detector.getInstance()
description.fromXML(sys.argv[1])
summary = {
    "detectors": sorted(str(item.first) for item in description.detectors()),
    "world_daughters": int(description.world().volume().GetNdaughters()),
    "probe": float(description.constantAsDouble(sys.argv[2])),
}
sys.stdout.write("\\n@@SUMMARY@@" + json.dumps(summary) + "\\n")
"""
CONSTANT = "CentralBeamPipe_rmax"


def load_with_dd4hep(compact: Path) -> dict[str, object]:
    run = subprocess.run(
        [sys.executable, "-c", LOADER, str(compact), CONSTANT],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert "@@SUMMARY@@" in run.stdout, run.stderr[-500:]
    summary: dict[str, object] = json.loads(run.stdout.split("@@SUMMARY@@")[1].splitlines()[0])
    return summary


@pytest.fixture(scope="module")
def dd4hep_available() -> None:
    if importlib.util.find_spec("dd4hep") is None:
        pytest.skip("the DD4hep Python bindings are not importable (source the Key4hep stack)")


@pytest.mark.usefixtures("dd4hep_available")
def test_dd4hep_loads_snapshots_like_the_source(k4geo_root: Path, tmp_path: Path) -> None:
    source = GeometrySource(CLD_COMPACT, root=k4geo_root)
    catalog = source.discover()
    original = load_with_dd4hep(source.compact_path)

    untouched = source.materialize([], tmp_path / "untouched", catalog)
    assert load_with_dd4hep(untouched.compact_path) == original

    target = f"constants.{CONSTANT}"
    old = catalog.lookup(target).value
    edited = source.materialize([Override(target, f"{old}+0.1*mm")], tmp_path / "edited", catalog)
    changed = load_with_dd4hep(edited.compact_path)
    assert changed["detectors"] == original["detectors"]
    assert changed["world_daughters"] == original["world_daughters"]
    assert isinstance(changed["probe"], float) and isinstance(original["probe"], float)
    assert changed["probe"] == pytest.approx(original["probe"] + 0.01)  # 0.1 mm in cm
