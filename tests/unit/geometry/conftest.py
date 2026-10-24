"""Fixtures for the geometry unit tests."""

from __future__ import annotations

import pytest

from k4opt.geometry import Catalog, GeometrySource
from tests.support.geometry import MINI_COMPACT, MINI_GEO


@pytest.fixture
def mini_source() -> GeometrySource:
    return GeometrySource(MINI_COMPACT, root=MINI_GEO)


@pytest.fixture
def mini_catalog(mini_source: GeometrySource) -> Catalog:
    return mini_source.discover()
