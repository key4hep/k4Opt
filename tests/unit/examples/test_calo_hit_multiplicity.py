"""The example evaluator calo.hit_multiplicity reading fake podio events; no ddsim is involved."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import EvaluationContractError, EvaluationRequest, Status
from tests.support import simulation
from tests.support.evaluators import calo_hit_multiplicity, calo_hits
from tests.support.simulation import COLLECTION, Event, Hits, events_with

SETTINGS: dict[str, Any] = {"readouts": {"ecal_barrel": COLLECTION}}


def request(tmp_path: Path, events: int = 3, **changes: Any) -> EvaluationRequest:
    events_file = tmp_path / "events.edm4hep.root"
    events_file.write_bytes(b"fake")
    fields: dict[str, Any] = {
        "settings": SETTINGS,
        "inputs": {"simulation": events_file},
        "events": events,
    }
    return EvaluationRequest(**(fields | changes))


def evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_events: list[Event], **changes: Any
) -> Any:
    simulation.install_events(monkeypatch, fake_events)
    return calo_hit_multiplicity().evaluate(request(tmp_path, **changes))


def test_the_evaluator_asks_for_the_simulation_stage() -> None:
    assert calo_hit_multiplicity().requires == "simulation"


def test_mean_and_standard_error_of_the_mean() -> None:
    mean, error = calo_hits.mean_and_error([2, 4, 4, 4, 5, 5, 7, 9])
    assert mean == 5
    assert error == pytest.approx(math.sqrt(32 / 7) / math.sqrt(8))  # s with ddof=1


def test_a_single_event_has_no_uncertainty_not_a_zero_one() -> None:
    assert calo_hits.mean_and_error([12]) == (12, None)


def test_identical_events_have_a_zero_standard_error() -> None:
    assert calo_hits.mean_and_error([5, 5, 5]) == (5, 0)


def test_counts_distinct_cell_ids_including_duplicates_and_empty_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation.install_events(monkeypatch, events_with([1, 1, 2], [], [5, 6, 7, 5, 6]))
    assert calo_hits.read_cell_counts(Path("x.root"), COLLECTION) == [2, 0, 3]


def test_valid_run_reports_the_mean_of_distinct_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evaluate(tmp_path, monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9, 9, 9]))
    assert result.status is Status.VALID
    (m,) = result.measurements
    assert (m.stage, m.metric, m.unit) == ("simulation", "ecal_barrel.mean_hit_cells", "1")
    assert m.value == pytest.approx(3.0)  # counts 3, 2, 4
    assert m.uncertainty == pytest.approx(1 / math.sqrt(3))
    assert result.provenance["events_read"] == 3
    assert result.provenance["collection"] == COLLECTION


def test_a_single_event_run_is_valid_without_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evaluate(tmp_path, monkeypatch, events_with([1, 2, 3]), events=1)
    assert result.status is Status.VALID
    assert result.measurements[0].value == 3
    assert result.measurements[0].uncertainty is None


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (events_with([1], [2]), "2 of 3 events"),
        (events_with([1], [2], [3], [4]), "4 of 3 events"),
        ([Event(), Event(), Event()], "missing in event 0"),
        ([*events_with([1], [2]), Event()], "missing in event 2"),
        (
            [Event(**{COLLECTION: Hits([1], "edm4hep::MCParticle")})] * 3,
            "expected 'edm4hep::SimCalorimeterHit'",
        ),
    ],
)
def test_an_unusable_simulation_output_is_failed_never_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[Event], message: str
) -> None:
    result = evaluate(tmp_path, monkeypatch, events)
    assert result.status is Status.FAILED
    assert message in (result.message or "")
    assert result.measurements == ()


def test_hits_without_cell_id_are_failed_not_counted_as_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Bare:  # a hit type that has no cellID
        pass

    class NoCells(Hits):
        def __init__(self) -> None:
            super().__init__([])
            self.append(Bare())

    result = evaluate(tmp_path, monkeypatch, [Event(**{COLLECTION: NoCells()})] * 3)
    assert result.status is Status.FAILED
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

    result = evaluate(tmp_path, monkeypatch, [*events_with([1]), Broken()], events=2)
    assert result.status is Status.FAILED
    assert "CppException: unsupported schema version" in (result.message or "")
    assert result.measurements == ()


def test_c_plus_plus_exceptions_when_opening_the_file_are_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def opening(path: Path) -> Any:
        raise CppException("incompatible file")
        yield  # pragma: no cover

    monkeypatch.setattr(calo_hits, "_open_events", opening)
    result = calo_hit_multiplicity().evaluate(request(tmp_path))
    assert result.status is Status.FAILED
    assert "incompatible file" in (result.message or "")


def test_unreadable_output_is_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(path: Path) -> Any:
        raise calo_hits._ReadError("cannot read events.edm4hep.root: truncated")
        yield  # pragma: no cover

    monkeypatch.setattr(calo_hits, "_open_events", broken)
    result = calo_hit_multiplicity().evaluate(request(tmp_path))
    assert result.status is Status.FAILED
    assert "truncated" in (result.message or "")


def test_a_file_without_events_is_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = evaluate(tmp_path, monkeypatch, [])
    assert result.status is Status.INVALID
    assert result.measurements == ()
    assert "no events" in (result.message or "")


def test_no_hits_in_any_event_is_invalid_not_a_zero_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evaluate(tmp_path, monkeypatch, events_with([], [], []))
    assert result.status is Status.INVALID
    assert result.measurements == ()
    assert "no hits" in (result.message or "")


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"readouts": {"ecal_barrel": ""}},
        {"readouts": {"ecal_barrel": COLLECTION, "hcal": "H"}},
        {**SETTINGS, "extra": 1},
        {**SETTINGS, "generator": {"particle": "gamma"}},  # configures the simulation stage
    ],
)
def test_unsupported_settings_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: dict[str, Any]
) -> None:
    simulation.install_events(monkeypatch, events_with([1], [2], [3]))
    with pytest.raises(EvaluationContractError):
        calo_hit_multiplicity().check(request(tmp_path, settings=settings))
    with pytest.raises(EvaluationContractError):
        calo_hit_multiplicity().evaluate(request(tmp_path, settings=settings))


@pytest.mark.parametrize("inputs", [{}, "missing"])
def test_a_missing_simulation_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inputs: Any
) -> None:
    simulation.install_events(monkeypatch, events_with([1]))
    if inputs == "missing":
        inputs = {"simulation": tmp_path / "gone.root"}
    with pytest.raises(EvaluationContractError, match="inputs"):
        calo_hit_multiplicity().evaluate(request(tmp_path, inputs=inputs))


def test_importing_the_evaluator_does_not_import_podio() -> None:
    code = "import sys, k4opt.evaluation, k4opt.cli; assert 'podio' not in sys.modules"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
