"""A fake ``ddsim`` run and fake podio events, for tests that need no Key4hep stack."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import run as ddsim_run
from k4opt.evaluation import edm4hep

COLLECTION = "ECalBarrelCollection"
HIT_TYPE = "edm4hep::SimCalorimeterHit"
PARTICLE_TYPE = "edm4hep::MCParticle"
DDSIM = "/stack/bin/ddsim"

type Triple = tuple[float, float, float]


@dataclass
class Vector:
    x: float
    y: float
    z: float


@dataclass
class Hit:
    cell: int
    energy: float = 1.0  # GeV
    position: Triple = (0.0, 0.0, 0.0)  # mm

    def getCellID(self) -> int:  # noqa: N802 - podio's spelling
        return self.cell

    def getEnergy(self) -> float:  # noqa: N802
        return self.energy

    def getPosition(self) -> Vector:  # noqa: N802
        return Vector(*self.position)


@dataclass
class Particle:
    momentum: Triple  # GeV
    vertex: Triple = (0.0, 0.0, 0.0)  # mm
    status: int = 1  # generator status: 1 for the particle of the gun
    decayed_in_tracker: bool = False
    pdg: int = 22  # a photon

    def getGeneratorStatus(self) -> int:  # noqa: N802
        return self.status

    def getPDG(self) -> int:  # noqa: N802
        return self.pdg

    def getMomentum(self) -> Vector:  # noqa: N802
        return Vector(*self.momentum)

    def getVertex(self) -> Vector:  # noqa: N802
        return Vector(*self.vertex)

    def isDecayedInTracker(self) -> bool:  # noqa: N802
        return self.decayed_in_tracker


class Collection(list[Any]):
    """A podio-like collection: a list that knows the type of its elements."""

    def __init__(self, elements: Sequence[Any], type_name: str) -> None:
        super().__init__(elements)
        self.type_name = type_name

    def getValueTypeName(self) -> str:  # noqa: N802
        return self.type_name


class Hits(Collection):
    """Calorimeter hits, given as cell IDs or as ``Hit``."""

    def __init__(self, cells: Sequence[int | Hit], type_name: str = HIT_TYPE) -> None:
        super().__init__([c if isinstance(c, Hit) else Hit(c) for c in cells], type_name)


class Particles(Collection):
    def __init__(self, particles: Sequence[Particle], type_name: str = PARTICLE_TYPE) -> None:
        super().__init__(particles, type_name)


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
    """Replaces the ddsim process (``k4opt.ddsim.run._execute``); records the call and writes
    the files ddsim would."""

    returncode: int = 0
    write_output: bool = True
    stderr: str = ""
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: Any) -> int | None:
        self.calls.append((argv, kwargs))
        kwargs["stderr"].write(self.stderr.encode())
        if self.write_output:
            output = next(a for a in argv if a.startswith("--outputFile="))
            Path(output.partition("=")[2]).write_bytes(b"fake")
        return self.returncode


def install_events(monkeypatch: pytest.MonkeyPatch, events: Sequence[Any] | None = None) -> None:
    """Make k4opt read ``events`` instead of an EDM4hep file; no ddsim is involved."""
    monkeypatch.setattr(edm4hep, "_frames", lambda path: list(events or []))


def install(
    monkeypatch: pytest.MonkeyPatch, events: Sequence[Any] | None = None, **ddsim: Any
) -> FakeDdsim:
    """Make k4opt run ``FakeDdsim`` and read ``events`` instead of the real thing.

    Without ``events`` the file holds as many one-hit events as the last run requested.
    """
    fake = FakeDdsim(**ddsim)
    monkeypatch.setattr(ddsim_run, "_execute", fake)
    monkeypatch.setattr(ddsim_run.shutil, "which", lambda name: DDSIM)

    def read(path: Path) -> list[Any]:
        if events is not None:
            return list(events)
        argv = fake.calls[-1][0]
        requested = int(
            next(a for a in argv if a.startswith("--numberOfEvents=")).partition("=")[2]
        )
        return events_with(*[[1]] * requested)

    monkeypatch.setattr(edm4hep, "_frames", read)
    return fake
