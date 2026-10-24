"""The calo.hit_multiplicity evaluator with a fake ddsim and fake podio events."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import DdsimError
from k4opt.ddsim import run as ddsim_run
from k4opt.evaluation import EvaluationContractError, EvaluationRequest
from k4opt.evaluators import CaloHitMultiplicity, calo_hits
from k4opt.evaluators.calo_hits import mean_and_error, read_cell_counts
from tests.support import simulation
from tests.support.simulation import COLLECTION, Event, Hits, events_with

SETTINGS: dict[str, Any] = {
    "readouts": {"ecal_barrel": COLLECTION},
    "generator": {
        "particle": "gamma",
        "energy": {"value": 10, "unit": "GeV"},
        "direction": {
            "distribution": "uniform_solid_angle",
            "theta": [60, 120],
            "phi": [0, 360],
            "unit": "deg",
        },
    },
}


def request(tmp_path: Path, events: int = 3, **changes: Any) -> EvaluationRequest:
    geometry = tmp_path / "compact.xml"
    geometry.write_text("<lccdd/>", encoding="utf-8")
    fields: dict[str, Any] = {
        "settings": SETTINGS,
        "inputs": {"geometry": geometry},
        "workdir": tmp_path / "run",
        "events": events,
        "seed": 7,
        "backend_args": {"ddsim": ("--physics.list", "FTFP_BERT")},
    }
    return EvaluationRequest(**(fields | changes))


def test_mean_and_standard_error_of_the_mean() -> None:
    mean, error = mean_and_error([2, 4, 4, 4, 5, 5, 7, 9])
    assert mean == 5
    assert error == pytest.approx(math.sqrt(32 / 7) / math.sqrt(8))  # s with ddof=1


def test_a_single_event_has_no_uncertainty_not_a_zero_one() -> None:
    assert mean_and_error([12]) == (12, None)


def test_identical_events_have_a_zero_standard_error() -> None:
    assert mean_and_error([5, 5, 5]) == (5, 0)


def test_counts_distinct_cell_ids_including_duplicates_and_empty_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation.install(monkeypatch, events_with([1, 1, 2], [], [5, 6, 7, 5, 6]))
    assert read_cell_counts(Path("x.root"), COLLECTION) == [2, 0, 3]


def test_valid_run_reports_the_mean_of_distinct_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9, 9, 9]))
    result = CaloHitMultiplicity().evaluate(request(tmp_path))

    assert result.status == "valid"
    (m,) = result.measurements
    assert (m.stage, m.metric, m.unit) == ("simulation", "ecal_barrel.mean_hit_cells", "1")
    assert m.value == pytest.approx(3.0)  # counts 3, 2, 4
    assert m.uncertainty == pytest.approx(1 / math.sqrt(3))
    assert result.provenance["events_read"] == 3
    assert result.provenance["returncode"] == 0
    assert set(result.artifacts) == {"stdout", "stderr", "command", "edm4hep"}
    assert all(path.is_file() for path in result.artifacts.values())

    ((argv, kwargs),) = fake.calls
    assert isinstance(argv, list)
    assert argv[0] == "/stack/bin/ddsim"
    assert argv[-2:] == ["--physics.list", "FTFP_BERT"]
    assert f"--compactFile={tmp_path / 'compact.xml'}" in argv
    assert "--numberOfEvents=3" in argv
    assert "--random.seed=7" in argv
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == tmp_path / "run" / "simulation"
    assert result.provenance["argv"] == tuple(argv)


def test_a_single_event_run_is_valid_without_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, events_with([1, 2, 3]))
    result = CaloHitMultiplicity().evaluate(request(tmp_path, events=1))
    assert result.status == "valid"
    assert result.measurements[0].value == 3
    assert result.measurements[0].uncertainty is None


@pytest.mark.parametrize(
    ("ddsim", "events", "message"),
    [
        ({"returncode": 3, "stderr": "boom: bad geometry"}, events_with([1], [2], [3]), "status 3"),
        ({"returncode": 3, "stderr": "boom: bad geometry"}, events_with([1], [2], [3]), "boom"),
        ({"write_output": False}, events_with([1], [2], [3]), "wrote no"),
        ({}, events_with([1], [2]), "2 of 3 events"),
        ({}, events_with([1], [2], [3], [4]), "4 of 3 events"),
        ({}, [Event(), Event(), Event()], "missing in event 0"),
        (
            {},
            [Event(**{COLLECTION: Hits([1], "edm4hep::MCParticle")})] * 3,
            "expected 'edm4hep::SimCalorimeterHit'",
        ),
    ],
)
def test_execution_problems_are_failed_never_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ddsim: dict[str, Any],
    events: list[Event],
    message: str,
) -> None:
    simulation.install(monkeypatch, events, **ddsim)
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "failed"
    assert message in (result.message or "")
    assert result.measurements == ()


def test_a_nonzero_exit_is_failed_even_when_the_output_looks_fine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, events_with([1], [2], [3]), returncode=1)
    assert CaloHitMultiplicity().evaluate(request(tmp_path)).status == "failed"


def test_a_collection_missing_in_a_later_event_is_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [*events_with([1], [2]), Event()]
    simulation.install(monkeypatch, events)
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "failed"
    assert "missing in event 2" in (result.message or "")


def test_hits_without_cell_id_are_failed_not_counted_as_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Bare:  # a hit type that has no cellID
        pass

    class NoCells(Hits):
        def __init__(self) -> None:
            super().__init__([])
            self.append(Bare())

    simulation.install(monkeypatch, [Event(**{COLLECTION: NoCells()})] * 3)
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "failed"
    assert "cellID" in (result.message or "")


class CppException(Exception):  # noqa: N818 - stands for cppyy's std::runtime_error
    """Like a C++ exception seen from Python: an Exception, but not a RuntimeError."""


def test_c_plus_plus_exceptions_of_the_reader_are_failed_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(Event):
        def getAvailableCollections(self) -> tuple[str, ...]:  # noqa: N802
            return (COLLECTION,)

        def get(self, name: str) -> Any:
            raise CppException("unsupported schema version")

    simulation.install(monkeypatch, [*events_with([1]), Broken()])
    result = CaloHitMultiplicity().evaluate(request(tmp_path, events=2))
    assert result.status == "failed"
    assert "CppException: unsupported schema version" in (result.message or "")
    assert result.measurements == ()
    assert "edm4hep" in result.artifacts  # the files stay inspectable


def test_c_plus_plus_exceptions_when_opening_the_file_are_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)

    def opening(path: Path) -> Any:
        raise CppException("incompatible file")
        yield  # pragma: no cover

    monkeypatch.setattr(calo_hits, "_open_events", opening)
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "failed"
    assert "incompatible file" in (result.message or "")


def test_unreadable_output_is_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    simulation.install(monkeypatch)

    def broken(path: Path) -> Any:
        raise calo_hits._ReadError("cannot read events.edm4hep.root: truncated")
        yield  # pragma: no cover

    monkeypatch.setattr(calo_hits, "_open_events", broken)
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "failed"
    assert "truncated" in (result.message or "")


def test_a_file_without_events_is_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    simulation.install(monkeypatch, [])
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "invalid"
    assert result.measurements == ()
    assert "no events" in (result.message or "")


def test_no_hits_in_any_event_is_invalid_not_a_zero_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, events_with([], [], []))
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "invalid"
    assert result.measurements == ()
    assert "no hits" in (result.message or "")


def test_ddsim_missing_from_the_environment_is_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, events_with([1], [2], [3]))
    monkeypatch.setattr(ddsim_run.shutil, "which", lambda name: None)
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "failed"
    assert "ddsim not found" in (result.message or "")


def test_ddsim_that_cannot_start_is_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    simulation.install(monkeypatch)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(ddsim_run.subprocess, "run", refuse)
    result = CaloHitMultiplicity().evaluate(request(tmp_path))
    assert result.status == "failed"
    assert "denied" in (result.message or "")


@pytest.mark.parametrize(
    "changes",
    [
        {"events": None},
        {"settings": {"generator": SETTINGS["generator"]}},
        {"settings": SETTINGS | {"readouts": {"ecal_barrel": ""}}},
        {"settings": SETTINGS | {"readouts": {"ecal_barrel": COLLECTION, "hcal": "H"}}},
        {"settings": SETTINGS | {"extra": 1}},
        {"settings": {**SETTINGS, "generator": {**SETTINGS["generator"], "particle": ""}}},
        {"backend_args": {"ddsim": ("--numberOfEvents", "5")}},
        {"backend_args": {"k4run": ("--x",)}},
    ],
)
def test_misuse_raises_before_anything_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    fake = simulation.install(monkeypatch)
    with pytest.raises((EvaluationContractError, DdsimError)):
        CaloHitMultiplicity().evaluate(request(tmp_path, **changes))
    assert not (tmp_path / "run").exists()
    assert fake.calls == []


@pytest.mark.parametrize("missing", ["geometry", "workdir"])
def test_a_missing_geometry_or_workdir_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    simulation.install(monkeypatch)
    changes: dict[str, Any] = {"inputs": {}} if missing == "geometry" else {"workdir": None}
    with pytest.raises(EvaluationContractError):
        CaloHitMultiplicity().evaluate(request(tmp_path, **changes))


def test_an_existing_simulation_directory_is_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, events_with([1], [2], [3]))
    (tmp_path / "run" / "simulation").mkdir(parents=True)
    with pytest.raises(DdsimError, match="already exists"):
        CaloHitMultiplicity().evaluate(request(tmp_path))
    assert fake.calls == []


def test_importing_the_evaluator_does_not_import_podio() -> None:
    code = "import sys, k4opt.evaluation, k4opt.cli; assert 'podio' not in sys.modules"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
