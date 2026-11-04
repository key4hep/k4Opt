"""ddsim makes the same events from the same seed (needs the Key4hep stack): what makes a stage
reused by its signature the stage that would have run, not only one like it.

Skipped without ``$K4GEO`` with the CLD geometry, ``ddsim`` or ``podio``. Simulates a few photons
on CLD four times at once, twice on one thread and twice on two, and compares the events by their
content (the particles and the ECAL hits), not by bytes: a ROOT file holds the time it was
written. With several threads Geant4 seeds each event from the run's seed, so the events are
the same, but they may be written in another order.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import simulate
from k4opt.frozen import freeze
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

EVENTS = 3
GUN = {
    "particle": "gamma",
    "energy": {"value": 10, "unit": "GeV"},
    "direction": {"distribution": "uniform", "theta": [80, 100], "phi": [0, 360], "unit": "deg"},
}


def _events(path: Path) -> list[tuple[Any, ...]]:
    """Each event of the EDM4hep file ``path`` as what it holds: its generated particles and its
    ECAL barrel hits, sorted."""
    from podio.reading import get_reader

    events = []
    for frame in get_reader(str(path)).get("events"):
        particles = sorted(
            (p.getPDG(), round(p.getMomentum().x, 6), round(p.getMomentum().y, 6))
            for p in frame.get("MCParticles")
            if p.getGeneratorStatus() == 1
        )
        hits = sorted(
            (h.getCellID(), round(h.getEnergy(), 9)) for h in frame.get("ECalBarrelCollection")
        )
        events.append((tuple(particles), tuple(hits)))
    return events


def test_the_same_seed_makes_the_same_events(tmp_path: Path) -> None:
    k4geo = os.environ.get("K4GEO")
    if not k4geo or not (Path(k4geo) / CLD_COMPACT).is_file():
        pytest.skip("$K4GEO with the CLD geometry is required")
    if shutil.which("ddsim") is None or importlib.util.find_spec("podio") is None:
        pytest.skip("ddsim and podio from the Key4hep stack are required")
    gun = freeze(GUN, "gun", ValueError)

    def run(name: str, threads: int) -> list[tuple[Any, ...]]:
        made = simulate(
            compact=Path(k4geo) / CLD_COMPACT,
            workdir=tmp_path / name,
            events=EVENTS,
            seed=42,
            particle_gun=gun,
            options={"numberOfThreads": threads},
        )
        assert made.error is None, made.error
        return _events(made.output)

    with ThreadPoolExecutor(4) as pool:
        runs = dict(
            zip(
                ("one", "one again", "two", "two again"),
                pool.map(run, ("one", "one again", "two", "two again"), (1, 1, 2, 2)),
                strict=True,
            )
        )
    assert len(runs["one"]) == EVENTS and any(hits for _, hits in runs["one"])
    assert runs["one again"] == runs["one"]  # one thread: the same events, in the same order
    assert sorted(runs["two"]) == sorted(runs["two again"])  # two: the same, in any order
