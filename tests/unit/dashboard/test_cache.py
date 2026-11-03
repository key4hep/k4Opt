"""What the dashboard keeps of the studies it read: by snapshot, while their snapshots add up to
its budget, the least recently used dropped first."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("streamlit")


def test_studies_are_kept_within_a_budget_of_their_snapshots_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data

    monkeypatch.setattr(data, "STUDY_BYTES", 100)
    studies: dict[str, Any] = {name: object() for name in "abcd"}

    def key(name: str) -> tuple[str, str, str]:
        return ("pub", f"{name}.json", name)

    data._keep(key("a"), studies["a"], 40)
    data._keep(key("b"), studies["b"], 40)
    assert data._held(key("a")) is studies["a"]  # used: b is now the oldest
    data._keep(key("c"), studies["c"], 40)  # 120 > 100: b goes
    assert (data._held(key("b")), data._held(key("a"))) == (None, studies["a"])
    data._keep(key("d"), studies["d"], 500)  # larger than the budget alone: kept alone
    assert [data._held(key(name)) for name in "acd"] == [None, None, studies["d"]]
