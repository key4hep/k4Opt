"""Known distributions check the example physics arithmetic and its histogram boundaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import (
    Artifact,
    EvaluationInvalid,
    EvaluationRequest,
    RoleInput,
    load_evaluator,
)
from tests.support import simulation

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "evaluators"


def example(file: str, name: str) -> Any:
    return load_evaluator(f"{EXAMPLES / file}:{name}")()


def histogram(tmp_path: Path, name: str, counts: list[int], overflow: int = 0) -> RoleInput:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps({"edges": [0, 1, 2, 3], "counts": counts, "underflow": 0, "overflow": overflow})
    )
    return RoleInput(
        name,
        binding="shape",
        artifacts={
            "histogram": Artifact(f"{name}/shape#histogram", path, path.stat().st_size, None)
        },
    )


def test_separation_uses_whole_bins_and_includes_background_overflow(tmp_path: Path) -> None:
    evaluator = example("separation.py", "Separation")
    request = EvaluationRequest(
        settings={"signal_efficiency": 0.9},
        roles={
            "signal": (histogram(tmp_path, "gamma", [8, 2, 0]),),
            "background": (histogram(tmp_path, "pi0", [1, 2, 2], overflow=5),),
        },
    )
    evaluator.check(request)
    assert evaluator.evaluate(request) == {
        ("analysis", "background_rejection"): pytest.approx(0.7),
        ("analysis", "signal_efficiency"): 1.0,
        ("analysis", "width_cut"): 2,
    }


@pytest.mark.parametrize(
    ("counts", "overflow", "message"),
    [
        ([0, 0, 0], 0, "empty width histogram"),
        ([1, 0, 0], 9, "needs overflow events"),
    ],
)
def test_separation_refuses_empty_or_unresolved_signal(
    tmp_path: Path, counts: list[int], overflow: int, message: str
) -> None:
    request = EvaluationRequest(
        settings={"signal_efficiency": 0.9},
        roles={
            "signal": (histogram(tmp_path, "gamma", counts, overflow),),
            "background": (histogram(tmp_path, "pi0", [1, 2, 3]),),
        },
    )
    with pytest.raises(EvaluationInvalid, match=message):
        example("separation.py", "Separation").evaluate(request)


def test_shower_width_is_transverse_and_energy_weighted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits = [simulation.Hit(1, 1, (100, -3, 0)), simulation.Hit(2, 1, (100, 3, 0))]
    simulation.install_events(monkeypatch, [simulation.Event(hits=simulation.Hits(hits))])
    request = EvaluationRequest(
        settings={"threshold": 0.0, "bins": {"low": 0, "high": 2, "count": 2}},
        inputs={"simulation": tmp_path / "events.root"},
        collections={"hits": "hits"},
        workdir=tmp_path,
    )
    evaluator = example("shower_shape.py", "ShowerShape")
    evaluator.check(request)
    result = evaluator.evaluate(request)
    assert result.measurements[0].value == pytest.approx(3.0)
    assert json.loads(result.artifacts["histogram"].read_text()) == {
        "edges": [0, 1, 2],
        "counts": [0, 0],
        "underflow": 0,
        "overflow": 1,
    }


def test_channel_cost_scales_with_inverse_cell_area() -> None:
    evaluator = example("ecal_channels.py", "EcalChannels")
    values = []
    for cell in (5.0, 10.0):
        request = EvaluationRequest(
            candidate={"constants.ECal_cell_size": cell},
            settings={"layers": 40, "radius": 2150, "length": 4420},
        )
        evaluator.check(request)
        values.append(evaluator.evaluate(request)[("geometry", "ecal_barrel.channels")])
    assert values[0] == pytest.approx(2 * math.pi * 2150 * 4420 * 40 / 25)
    assert values[0] == pytest.approx(4 * values[1])
