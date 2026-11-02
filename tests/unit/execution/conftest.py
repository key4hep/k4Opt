"""Shared fixtures of the execution unit tests."""

from __future__ import annotations

import pytest

from tests.support.problems import document


@pytest.fixture
def doc() -> dict[str, object]:
    """A fresh, valid problem document (plain dict) that a test may mutate."""
    return document()
