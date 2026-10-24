"""Smoke test against a real k4geo tree (the Key4hep stack's ``$K4GEO`` or ``K4GEO_DIR``).

Run inside a Key4hep environment (``source setup.sh``) or with
``K4GEO_DIR=/path/to/k4geo python -m pytest tests/integration``. Skipped when
neither is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from k4opt.geometry import GeometrySource, Override
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

INNER_TRACKER = "FCCee/CLD/compact/CLD_o2_v09/InnerTracker_o2_v08.xml"


def test_cld_discovery_and_materialization(
    k4geo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = GeometrySource(CLD_COMPACT, root=k4geo_root)
    monkeypatch.setenv("K4GEO", str(k4geo_root))
    assert GeometrySource(CLD_COMPACT) == source, "relative geometry must resolve against $K4GEO"

    catalog = source.discover()
    assert catalog.files[0] == CLD_COMPACT
    assert INNER_TRACKER in catalog.files
    assert any(e.ref.startswith("${DD4hepINSTALL}") for e in catalog.externals)

    radius = catalog.lookup("constants.InnerTracker_Barrel_radius_1")
    nphi = catalog.lookup("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi")
    assert radius.file == INNER_TRACKER and radius.value == "400*mm"
    assert nphi.file == INNER_TRACKER and nphi.value == "43*2"
    assert catalog.lookup("constants.SolenoidField").value == "2*tesla"

    result = source.materialize(
        [
            Override("constants.InnerTracker_Barrel_radius_1", "405*mm"),
            Override("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi", "90"),
        ],
        tmp_path / "cld-test",
        catalog,
    )
    assert result.compact_path == tmp_path / "cld-test" / CLD_COMPACT
    assert result.compact_path.is_file()
    assert [f.path for f in result.files] == [INNER_TRACKER]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["source"]["compact"] == CLD_COMPACT
    assert manifest["reachable_files"] == list(catalog.files)

    snapshot = GeometrySource(CLD_COMPACT, root=result.output_root).discover()
    assert snapshot.lookup("constants.InnerTracker_Barrel_radius_1").value == "405*mm"
    assert snapshot.lookup("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi").value == "90"
    assert len(snapshot) == len(catalog)
