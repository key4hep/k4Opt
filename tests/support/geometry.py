"""Constants for the geometry tests."""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "geometry"
MINI_GEO = FIXTURES / "mini_geo"
MINI_COMPACT = "compact/det_v1/det_v1.xml"
CLD_COMPACT = "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml"
