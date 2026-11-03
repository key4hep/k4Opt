"""Shared fixtures of the dashboard tests: the app as Streamlit runs it, on real publications."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.support.studies import mini_environment

DASHBOARD = Path(__file__).resolve().parents[3] / "dashboard"


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    mini_environment(monkeypatch)
    monkeypatch.delenv("K4OPT_DASHBOARD_DATA", raising=False)
    monkeypatch.syspath_prepend(str(DASHBOARD))  # its modules, as Streamlit imports them


@pytest.fixture(autouse=True)
def fresh_caches() -> None:
    """Nothing a test read is kept for the next: Streamlit's caches are the process's."""
    sys.modules.pop("data", None)
    try:
        import streamlit as st
    except ModuleNotFoundError:  # the tables need no Streamlit
        return
    st.cache_resource.clear()
