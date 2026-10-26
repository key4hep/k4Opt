"""A fake ``ddsim`` run and fake podio events, for tests that need no Key4hep stack."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import run as ddsim_run
from k4opt.evaluators import calo_hits

COLLECTION = "ECalBarrelCollection"
DDSIM = "/stack/bin/ddsim"
_run = subprocess.run


@dataclass
class Hit:
    cell: int

    def getCellID(self) -> int:  # noqa: N802 - podio's spelling
        return self.cell


class Hits(list[Any]):
    """A podio-like collection of hits."""

    def __init__(self, cells: Sequence[int], type_name: str = calo_hits.HIT_TYPE) -> None:
        super().__init__(Hit(c) for c in cells)
        self.type_name = type_name

    def getValueTypeName(self) -> str:  # noqa: N802
        return self.type_name


class Event:
    def __init__(self, **collections: Any) -> None:
        self.collections = collections

    def getAvailableCollections(self) -> tuple[str, ...]:  # noqa: N802
        return tuple(self.collections)

    def get(self, name: str) -> Any:
        return self.collections[name]


def events_with(*cells_per_event: Sequence[int]) -> list[Event]:
    return [Event(**{COLLECTION: Hits(cells)}) for cells in cells_per_event]


@dataclass
class FakeDdsim:
    """Replaces ``subprocess.run``; records the call and writes the files ddsim would."""

    returncode: int = 0
    write_output: bool = True
    stderr: str = ""
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if Path(argv[0]).name != "ddsim":  # subprocess is shared: the geometry component runs git
            return _run(argv, **kwargs)
        self.calls.append((argv, kwargs))
        kwargs["stderr"].write(self.stderr.encode())
        if self.write_output:
            output = next(a for a in argv if a.startswith("--outputFile="))
            Path(output.partition("=")[2]).write_bytes(b"fake")
        return subprocess.CompletedProcess(argv, self.returncode)


def install(
    monkeypatch: pytest.MonkeyPatch, events: Sequence[Any] | None = None, **ddsim: Any
) -> FakeDdsim:
    """Make the evaluator run ``FakeDdsim`` and read ``events`` instead of the real thing."""
    fake = FakeDdsim(**ddsim)
    monkeypatch.setattr(ddsim_run.subprocess, "run", fake)
    monkeypatch.setattr(ddsim_run.shutil, "which", lambda name: DDSIM)
    monkeypatch.setattr(calo_hits, "_open_events", lambda path: iter(events or []))
    return fake
