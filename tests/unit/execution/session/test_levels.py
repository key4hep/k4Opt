"""A candidate evaluated for a level of a nested optimization is judged by that level's criteria:
the tasks they read are required, the record names their metrics and its assessment judges them,
and a reader judges it by the level it says it was made for."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.execution import EvaluationSession, RunOutcome, RunRequest
from k4opt.problem import CandidateError
from k4opt.record import SUMMARY_NAME, Assessment, RecordError
from tests.support import simulation
from tests.support.evaluators import cell_count, trade_off
from tests.unit.execution.session.conftest import RADIUS

CELLS = "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells"
A = "test.trade_off/analysis/a"


@pytest.fixture
def nested_file(problem_file: Path) -> Path:
    """The problem with a second parameter, the setting ``x`` of ``test.trade_off``, tuned for
    each radius by ``a``, a metric no objective of the problem names; ``test.trade_off`` is
    invalid below 0.5."""
    doc = yaml.safe_load(problem_file.read_text())
    doc["parameters"]["x"] = {
        "type": "continuous",
        "bounds": [0.1, 1.0],
        "targets": {"evaluators": {"test.trade_off": "x"}},
    }
    doc["evaluators"].append(
        {"name": "test.trade_off", "version": "1", "settings": {"invalid_below": 0.5}}
    )
    doc["objectives"][0] |= {"evaluator": "test.cell_count", "sample": "gamma"}
    doc["optimization"] = {
        "parameters": [RADIUS],
        "inner": {
            "parameters": ["x"],
            "trials": 3,
            "objective": {
                "evaluator": "test.trade_off",
                "stage": "analysis",
                "metric": "a",
                "direction": "minimize",
            },
        },
    }
    problem_file.write_text(yaml.safe_dump(doc))
    return problem_file


def evaluate(session: EvaluationSession, x: float, output: Path, level: int) -> dict[str, Any]:
    outcome = session.evaluate({RADIUS: 450.0, "x": x}, output, level=level)
    data: dict[str, Any] = json.loads((outcome.run_dir / SUMMARY_NAME).read_text())
    return data


def test_each_level_requires_names_and_judges_its_own_criteria(
    tmp_path: Path, nested_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(nested_file, [cell_count(), trade_off()]) as session:
        top = evaluate(session, 0.8, tmp_path / "top", level=0)
        inner = evaluate(session, 0.8, tmp_path / "inner", level=1)
        for record in (top, inner):
            Assessment.from_summary(session.spec, record)  # each holds up as what it says
    assert (top["level"], inner["level"]) == (0, 1)
    assert [m["label"] for m in top["metrics"]] == [CELLS]
    assert [m["label"] for m in inner["metrics"]] == [A]
    assert [o["metric"] for o in top["assessment"]["objectives"]] == ["ecal_barrel.mean_hit_cells"]
    assert [o["metric"] for o in inner["assessment"]["objectives"]] == ["a"]
    assert "test.trade_off" not in top["required"] and "test.trade_off" in inner["required"]
    assert "gamma/test.cell_count" in top["required"]
    assert "gamma/test.cell_count" not in inner["required"]
    # what a level does not judge by is measured all the same
    assert inner["samples"]["gamma"]["evaluations"]["test.cell_count"]["status"] == "valid"
    assert top["evaluations"]["test.trade_off"]["status"] == "valid"


def test_a_candidate_only_its_inner_criteria_reject_is_valid_at_the_top(
    tmp_path: Path, nested_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(nested_file, [cell_count(), trade_off()]) as session:
        top = evaluate(session, 0.2, tmp_path / "top", level=0)
        inner = evaluate(session, 0.2, tmp_path / "inner", level=1)
    assert top["status"] == "valid"  # test.trade_off is a diagnostic there
    assert [d["task"] for d in top["diagnostics"]] == ["test.trade_off"]
    assert inner["status"] == "invalid" and inner["assessment"]["objectives"] is None


@pytest.mark.parametrize("level", [1, 2, -1, None, True])
def test_a_record_judged_by_another_level_is_refused(
    tmp_path: Path, nested_file: Path, fake: simulation.FakeDdsim, level: object
) -> None:
    with EvaluationSession.open(nested_file, [cell_count(), trade_off()]) as session:
        record = evaluate(session, 0.8, tmp_path / "top", level=0)
        record["level"] = level
        with pytest.raises(RecordError, match=r"level is|required is|its metrics are"):
            Assessment.from_summary(session.spec, record)


@pytest.mark.parametrize("level", [2, -1, True])
def test_a_level_the_problem_does_not_have_is_refused_before_anything_is_created(
    tmp_path: Path, nested_file: Path, fake: simulation.FakeDdsim, level: object
) -> None:
    with (
        EvaluationSession.open(nested_file, [cell_count(), trade_off()]) as session,
        pytest.raises(CandidateError, match="the problem's levels are 0 to 1"),
    ):
        session.evaluate({RADIUS: 450.0, "x": 0.8}, tmp_path / "a", level=level)  # type: ignore[arg-type]
    assert not (tmp_path / "a").exists() and fake.calls == []


def test_the_diagnostics_of_a_session_are_those_of_a_level(
    nested_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(nested_file, [cell_count(), trade_off()]) as session:
        assert session.diagnostics() == session.diagnostics(0) == ("test.trade_off",)
        assert session.diagnostics(1) == ("gamma/test.cell_count",)
        notice = session.notice(1)
        assert notice is not None and "gamma/test.cell_count" in notice


def test_candidates_of_two_levels_in_one_evaluation_share_their_stages(
    tmp_path: Path, nested_file: Path, fake: simulation.FakeDdsim
) -> None:
    requests = [
        RunRequest({RADIUS: 450.0, "x": 0.8}, tmp_path / name, {"name": name}, level=level)
        for name, level in (("top", 0), ("inner", 1))
    ]
    with (
        EvaluationSession.open(nested_file, [cell_count(), trade_off()]) as session,
        session.evaluate_each(requests) as evaluations,
    ):
        ended = {request.run["name"]: outcome for request, outcome in evaluations}  # type: ignore[index]
        for outcome in ended.values():
            assert isinstance(outcome, RunOutcome), outcome
            Assessment.from_summary(session.spec, dict(outcome.summary))
    top, inner = (ended[name].summary for name in ("top", "inner"))
    assert len(fake.calls) == 1  # one simulation: the stages do not depend on the level
    assert inner["samples"]["gamma"]["stages"]["simulation"]["reused"]["record"] == top["id"]
    assert [m["label"] for m in top["metrics"]] == [CELLS]
    assert [m["label"] for m in inner["metrics"]] == [A]
