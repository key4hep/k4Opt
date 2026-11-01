"""``get_reader`` of the fake ``podio``: the events of a file the fake ``ddsim`` wrote.

Self-contained: a worker process imports it as ``podio.reading``, where ``tests`` need not be
importable (the repository's ``tests`` is no package, and another one may be on the path).
"""

from __future__ import annotations

import json
from pathlib import Path

COLLECTION = "ECalBarrelCollection"
HIT_TYPE = "edm4hep::SimCalorimeterHit"


class Hit:
    def __init__(self, cell: int) -> None:
        self.cell = cell

    def getCellID(self) -> int:  # noqa: N802 - podio's spelling
        return self.cell

    def getEnergy(self) -> float:  # noqa: N802
        return 1.0  # GeV


class Hits(list[Hit]):
    def getValueTypeName(self) -> str:  # noqa: N802
        return HIT_TYPE


class Event:
    def __init__(self, cells: list[int]) -> None:
        self.hits = Hits(Hit(cell) for cell in cells)

    def getAvailableCollections(self) -> tuple[str, ...]:  # noqa: N802
        return (COLLECTION,)

    def get(self, name: str) -> Hits:
        return self.hits


class Reader:
    """A podio reader of a file the fake ``ddsim`` wrote: one hit per cell of each event."""

    categories = ("events",)

    def __init__(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.events = [Event(cells) for cells in data["events"]]

    def get(self, category: str) -> list[Event]:
        return self.events


def get_reader(path: str) -> Reader:
    return Reader(path)
