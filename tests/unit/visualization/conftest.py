"""Shared fixtures of the visualization tests."""

from __future__ import annotations

import pytest

from tests.support.studies import mini_environment


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    mini_environment(monkeypatch)
