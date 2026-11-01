"""A fake Key4hep stack for tests whose tasks run in worker processes: a ``ddsim`` script and a
``podio`` package (``fake_stack/``), which the workers find as the real ones, on ``PATH`` and
the coordinator's ``sys.path``. The coordinator itself reads the same files through a patched
:func:`k4opt.evaluation.edm4hep._frames`, so the real ``podio`` it may import stays untouched."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import edm4hep
from tests.support.simulation import COLLECTION, Event, Hits

FAKE_STACK = Path(__file__).with_name("fake_stack")


class Reader:
    """A podio reader of a file the fake ``ddsim`` wrote."""

    categories = ("events",)

    def __init__(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.events = [Event(**{COLLECTION: Hits(cells)}) for cells in data["events"]]

    def get(self, category: str) -> list[Event]:
        return self.events


def install(monkeypatch: pytest.MonkeyPatch, marks: Path | None = None, **told: Any) -> None:
    """Run the fake ``ddsim`` and read its files with the fake ``podio``, here and in every
    worker started from now on. ``marks`` is the directory the fake ``ddsim`` marks its start and
    end in; ``told`` says what it does for each sample (see ``fake_stack/bin/ddsim``)."""
    monkeypatch.setenv("PATH", f"{FAKE_STACK / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.syspath_prepend(str(FAKE_STACK / "python"))
    monkeypatch.setattr(edm4hep, "_frames", lambda path: Reader(str(path)).events)
    control = dict(told) | ({} if marks is None else {"dir": str(marks)})
    monkeypatch.setenv("FAKE_DDSIM", json.dumps(control))


def eventually(condition: Callable[[], bool], timeout: float = 30.0) -> bool:
    """Whether ``condition`` holds within ``timeout`` seconds, asked every few milliseconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def alive(pid: int) -> bool:
    """Whether the process ``pid`` exists and is no zombie."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


def workers(parent: int | None = None) -> list[int]:
    """The worker processes (:mod:`k4opt.problem._worker`) of the process ``parent`` (this one by
    default) that are alive."""
    parent = os.getpid() if parent is None else parent
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        ppid = int(stat.rpartition(")")[2].split()[1])
        if ppid == parent and b"k4opt.problem._worker" in command and alive(int(entry.name)):
            found.append(int(entry.name))
    return found


def started(marks: Path) -> dict[str, int]:
    """The pid of each fake ddsim that started, by ``<run>.<sample>``."""
    return {
        path.name.removesuffix(".started"): int(path.read_text())
        for path in marks.glob("*.started")
    }
