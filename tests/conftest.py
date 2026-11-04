"""Root conftest; helpers live in tests/support.

A session warns when its workers may not fit in the machine's memory
(:func:`k4opt.execution._session._memory_warning`). The tests do not depend on the machine they
run on: they see as much memory as any number of workers needs, unless one says otherwise.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def ample_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    from k4opt.execution import _session

    monkeypatch.setattr(_session, "usable_memory", lambda: None)
