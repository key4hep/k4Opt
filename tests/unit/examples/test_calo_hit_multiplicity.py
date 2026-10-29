"""The example evaluator calo.hit_multiplicity reading fake podio events; no ddsim is involved."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRequest,
    Status,
    result_from,
)
from tests.support import simulation
from tests.support.evaluators import calo_hit_multiplicity
from tests.support.simulation import COLLECTION, Event, Hits, events_with

SETTINGS: dict[str, Any] = {"collection": COLLECTION}


def request(tmp_path: Path, **changes: Any) -> EvaluationRequest:
    events_file = tmp_path / "events.edm4hep.root"
    events_file.write_bytes(b"fake")
    fields: dict[str, Any] = {"settings": SETTINGS, "inputs": {"simulation": events_file}}
    return EvaluationRequest(**(fields | changes))


def evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_events: list[Event], **changes: Any
) -> Any:
    simulation.install_events(monkeypatch, fake_events)
    evaluator = calo_hit_multiplicity()
    return result_from(evaluator, evaluator.evaluate(request(tmp_path, **changes)))


def test_the_evaluator_asks_for_the_simulation_stage() -> None:
    assert calo_hit_multiplicity().requires == "simulation"


def test_valid_run_reports_the_mean_of_distinct_cells_and_its_standard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # duplicate cells count once: counts 3, 2, 4
    result = evaluate(tmp_path, monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9, 9, 9]))
    assert result.status is Status.VALID
    (m,) = result.measurements
    assert (m.stage, m.metric, m.unit) == ("simulation", "ecal_barrel.mean_hit_cells", "1")
    assert m.value == pytest.approx(3.0)
    assert m.uncertainty == pytest.approx(1 / math.sqrt(3))  # s / sqrt(N) with ddof=1


def test_an_empty_event_counts_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = evaluate(tmp_path, monkeypatch, events_with([1, 1, 2], [], [5, 6, 7, 5, 6]))
    assert result.measurements[0].value == pytest.approx(5 / 3)  # counts 2, 0, 3


def test_a_single_event_run_is_valid_without_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evaluate(tmp_path, monkeypatch, events_with([1, 2, 3]))
    assert result.status is Status.VALID
    assert result.measurements[0].value == 3
    assert result.measurements[0].uncertainty is None


def test_identical_events_have_a_zero_standard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evaluate(tmp_path, monkeypatch, events_with([1, 2], [3, 4], [5, 6]))
    assert result.measurements[0].uncertainty == 0


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([Event(), Event()], "missing in event 0"),
        ([*events_with([1]), Event()], "missing in event 1"),
        (
            [Event(**{COLLECTION: Hits([1], "edm4hep::MCParticle")})] * 2,
            "expected 'edm4hep::SimCalorimeterHit'",
        ),
    ],
)
def test_an_unusable_simulation_output_fails_never_gives_a_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[Event], message: str
) -> None:
    with pytest.raises(EvaluationFailed, match=message):
        evaluate(tmp_path, monkeypatch, events)


def test_no_hits_in_any_event_is_invalid_not_a_zero_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(EvaluationInvalid, match="no hits"):
        evaluate(tmp_path, monkeypatch, events_with([], [], []))


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"collection": 3},
        {"collection": ""},
        {"collection": "   "},
        {"readouts": {"ecal_barrel": COLLECTION}},
        {**SETTINGS, "extra": 1},
        {**SETTINGS, "generator": {"particle": "gamma"}},  # configures the simulation stage
    ],
)
def test_unsupported_settings_raise(tmp_path: Path, settings: dict[str, Any]) -> None:
    with pytest.raises(EvaluationContractError):
        calo_hit_multiplicity().check(request(tmp_path, settings=settings))


def test_importing_the_evaluator_does_not_import_podio() -> None:
    code = "import sys, k4opt.evaluation, k4opt.cli; assert 'podio' not in sys.modules"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
