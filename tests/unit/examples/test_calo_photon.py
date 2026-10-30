"""The example evaluator calo.photon reading fake podio events; no ddsim is involved.

The photons fly along x from the origin and the cells sit 2000 mm away, so a centroid displaced
by ``d`` mm sideways is ``atan(d / 2000)`` off the photon's direction.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRequest,
    Status,
    collection_names,
    result_from,
    with_defaults,
)
from tests.support import simulation
from tests.support.evaluators import calo_photon
from tests.support.simulation import COLLECTION, Event, Hit, Hits, Particle, Particles

# what the evaluator receives for the problem's threshold of 70 keV, and the collections
SETTINGS: dict[str, Any] = {"threshold": 7e-05}
COLLECTIONS = {"hits": COLLECTION, "particles": "MCParticles"}
PHOTON = Particle(momentum=(10.0, 0.0, 0.0))
FIRES = 1e-3  # GeV, well above the threshold of 70 keV
RESIDUAL = "ecal_barrel.angular_residual"
CELLS = "ecal_barrel.fired_cells"


def mrad(sideways: float) -> float:
    return 1e3 * math.atan(sideways / 2000)


def cell(y: float, energy: float = FIRES, z: float = 0.0) -> Hit:
    return Hit(cell=0, energy=energy, position=(2000.0, y, z))


def event(hits: Sequence[Hit], *particles: Particle) -> Event:
    return Event(**{COLLECTION: Hits(hits), "MCParticles": Particles(list(particles or [PHOTON]))})


def request(tmp_path: Path, **changes: Any) -> EvaluationRequest:
    events_file = tmp_path / "events.edm4hep.root"
    events_file.write_bytes(b"fake")
    fields: dict[str, Any] = {
        "settings": SETTINGS,
        "collections": COLLECTIONS,
        "inputs": {"simulation": events_file},
    }
    return EvaluationRequest(**(fields | changes))


def evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *events: Event) -> dict[str, Any]:
    simulation.install_events(monkeypatch, events)
    evaluator = calo_photon()
    result = result_from(evaluator, evaluator.evaluate(request(tmp_path)))
    assert result.status is Status.VALID
    return {m.metric: m for m in result.measurements}


def test_the_evaluator_asks_for_the_simulation_stage() -> None:
    assert calo_photon().stage == "simulation"


def test_valid_run_reports_means_and_standard_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # centroids 1 mm and 0 mm off the photon's line; 2 and 1 fired cells
    measured = evaluate(tmp_path, monkeypatch, event([cell(0), cell(2)]), event([cell(0)]))
    residual, cells = measured[RESIDUAL], measured[CELLS]
    assert (residual.stage, residual.unit, cells.unit) == ("simulation", "mrad", "1")
    assert residual.value == pytest.approx(mrad(1) / 2)
    assert residual.uncertainty == pytest.approx(mrad(1) / 2)  # s / sqrt(N) with ddof=1
    assert (cells.value, cells.uncertainty) == (pytest.approx(1.5), pytest.approx(0.5))


def test_the_centroid_is_energy_weighted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    measured = evaluate(tmp_path, monkeypatch, event([cell(0, 3 * FIRES), cell(4, FIRES)]))
    assert measured[RESIDUAL].value == pytest.approx(mrad(1))


def test_cells_below_the_threshold_neither_fire_nor_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = [cell(0), cell(500, 69e-6), cell(0, 70e-6)]  # the last one is exactly at threshold
    measured = evaluate(tmp_path, monkeypatch, event(hits))
    assert measured[CELLS].value == 2
    assert measured[RESIDUAL].value == pytest.approx(0, abs=1e-9)


def test_the_direction_starts_at_the_photon_vertex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    displaced = Particle(momentum=(10.0, 0.0, 0.0), vertex=(0.0, 0.0, 10.0))
    measured = evaluate(tmp_path, monkeypatch, event([cell(0, z=10.0)], displaced))
    assert measured[RESIDUAL].value == pytest.approx(0, abs=1e-9)


def test_photons_converted_in_the_tracker_are_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted = Particle(momentum=(10.0, 0.0, 0.0), decayed_in_tracker=True)
    secondary = Particle(momentum=(1.0, 1.0, 0.0), status=0)  # ignored: not from the gun
    measured = evaluate(
        tmp_path,
        monkeypatch,
        event([cell(0), cell(2)], PHOTON, secondary),
        event([cell(300), cell(0), cell(1)], converted),
    )
    assert (measured[CELLS].value, measured[CELLS].uncertainty) == (2, None)
    assert measured[RESIDUAL].value == pytest.approx(mrad(1))


def test_a_photon_without_fired_cells_counts_zero_cells_and_no_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measured = evaluate(tmp_path, monkeypatch, event([cell(0), cell(2)]), event([cell(9, 1e-6)]))
    assert measured[CELLS].value == 1
    assert (measured[RESIDUAL].value, measured[RESIDUAL].uncertainty) == (
        pytest.approx(mrad(1)),
        None,
    )


@pytest.mark.parametrize(
    "events",
    [
        [event([]), event([cell(0, 1e-6)])],
        [event([cell(0)], Particle(momentum=(10.0, 0.0, 0.0), decayed_in_tracker=True))],
    ],
)
def test_nothing_to_measure_is_invalid_not_a_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[Event]
) -> None:
    simulation.install_events(monkeypatch, events)
    with pytest.raises(EvaluationInvalid, match="no photon reached the calorimeter"):
        calo_photon().evaluate(request(tmp_path))


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([Event(), Event()], "'ECalBarrelCollection' is missing in event 0"),
        ([Event(**{COLLECTION: Hits([cell(0)])})], "'MCParticles' is missing in event 0"),
        (
            [Event(**{COLLECTION: Hits([1], "edm4hep::MCParticle")})],
            "expected 'edm4hep::SimCalorimeterHit'",
        ),
        ([event([cell(0)], PHOTON, PHOTON)], "one generated particle per event, found 2"),
        ([event([cell(0)], Particle((1.0, 0.0, 0.0), status=0))], "found 0"),
        ([event([cell(0)], Particle((10.0, 0.0, 0.0), pdg=11))], "expected a photon .* PDG 11"),
    ],
)
def test_an_unusable_simulation_output_fails_never_gives_a_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[Event], message: str
) -> None:
    simulation.install_events(monkeypatch, events)
    with pytest.raises(EvaluationFailed, match=message):
        calo_photon().evaluate(request(tmp_path))


@pytest.mark.parametrize(
    "settings",
    [
        {},  # the threshold is part of the metric definition: no default
        {"threshold": "70 keV"},
        {**SETTINGS, "collection": COLLECTION},  # a collection is no setting
        {**SETTINGS, "generator": {"particle": "gamma"}},  # configures the simulation stage
    ],
)
def test_unsupported_settings_raise(tmp_path: Path, settings: dict[str, Any]) -> None:
    with pytest.raises(EvaluationContractError):
        calo_photon().check(request(tmp_path, settings=settings))


@pytest.mark.parametrize(
    ("threshold", "received"),
    [
        ({"value": 70, "unit": "keV"}, 70e-6),  # exactly the float an author would write
        ({"value": 0.07, "unit": "MeV"}, 70e-6),
        ({"value": 1e-4, "unit": "GeV"}, 1e-4),
    ],
)
def test_the_threshold_is_received_in_gev(threshold: dict[str, Any], received: float) -> None:
    assert with_defaults(calo_photon(), {"threshold": threshold}) == {"threshold": received}


@pytest.mark.parametrize(
    ("threshold", "message"),
    [
        (7e-05, "needs value and unit"),
        ({"value": 70}, "needs value and unit"),
        ({"value": 70, "unit": "mm"}, r"unit: must be a unit of energy \(eV, keV, MeV, GeV, TeV\)"),
        ({"value": "70", "unit": "keV"}, "value: must be a finite number"),
    ],
)
def test_a_threshold_without_an_energy_unit_is_refused(threshold: Any, message: str) -> None:
    with pytest.raises(EvaluationContractError, match=rf"settings\.threshold.*{message}"):
        with_defaults(calo_photon(), {"threshold": threshold})


def test_the_problem_names_the_hits_and_may_rename_the_particles() -> None:
    assert collection_names(calo_photon(), {"hits": COLLECTION}) == COLLECTIONS
    renamed = {"hits": COLLECTION, "particles": "Truth"}
    assert collection_names(calo_photon(), renamed) == renamed
    with pytest.raises(EvaluationContractError, match=r"missing \['hits'\]"):
        collection_names(calo_photon(), {})
    with pytest.raises(EvaluationContractError, match=r"unknown \['hit'\]"):
        collection_names(calo_photon(), {"hit": COLLECTION})


def test_importing_the_evaluator_does_not_import_podio() -> None:
    code = "import sys, k4opt.evaluation, k4opt.cli; assert 'podio' not in sys.modules"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
