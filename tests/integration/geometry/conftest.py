"""Fixtures for the geometry integration tests (need a real k4geo tree)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.support.geometry import CLD_COMPACT


@pytest.fixture
def k4geo_root() -> Path:
    """A real k4geo tree: ``$K4GEO_DIR`` or the stack's ``$K4GEO``; skips when unavailable."""
    root = os.environ.get("K4GEO_DIR") or os.environ.get("K4GEO")
    if not root:
        pytest.skip("neither K4GEO_DIR nor K4GEO is set; skipping k4geo integration test")
    if not (Path(root) / CLD_COMPACT).is_file():
        pytest.skip(f"{CLD_COMPACT} not found below {root}")
    return Path(root)
