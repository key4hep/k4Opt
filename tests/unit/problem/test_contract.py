"""What a problem asks of its implementations, refused with the same words live and from a record.

Each case is a declaration or a result a live session refuses (:mod:`k4opt.problem.contract`),
and the record of an accepted candidate edited to say the same: reading it back refuses it as
the session refused the evaluator, so a record holds up only where a live evaluation would have.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
    Quantity,
    load_evaluator,
)
from k4opt.execution import EvaluationSession
from k4opt.record import Assessment, RecordError
from tests.support.geometry import MINI_COMPACT, MINI_GEO

RADIUS = "radius"
ECHO = ("analysis", "candidate.first_value")
ECHOED = "evaluations/test.echo"  # the work of the test.echo binding in a record
DECLARED = "implementations/test.echo"  # what its implementation declares, in a record


class Echo:
    """The candidate's radius as its metric, with a note as its artifact; reads no stage."""

    name = "test.echo"
    version = "1"
    settings = {"x": Quantity("mm")}  # the candidate's radius, which it sets
    metrics: dict[tuple[str, str], str] | None = {ECHO: "1"}
    artifacts: dict[str, str] | None = {"note": "what it measured"}
    unit = "1"  # what it measures in
    returned = ("note",)  # the artifacts it returns

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        value = request.settings["x"]
        assert request.workdir is not None
        files = {}
        for name in self.returned:
            files[name] = request.workdir / f"{name}.txt"
            files[name].write_text(str(value))
        return EvaluationResult.valid([Measurement(*ECHO, value, self.unit)], artifacts=files)


class Silent(Echo):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return EvaluationResult.valid([])


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def problem_file(tmp_path: Path) -> Path:
    """The radius as the parameter, ``test.echo`` its only evaluator, its metric the objective
    and constrained in ``1``."""
    metric = {"evaluator": "test.echo", "stage": ECHO[0], "metric": ECHO[1]}
    data = {
        "schema_version": "k4opt.problem/v1alpha1",
        "name": "contract",
        "environment": {"key4hep": {"channel": "nightly", "version": "2026-10-20"}},
        "geometry": MINI_COMPACT,
        "parameters": {
            RADIUS: {
                "targets": {
                    "geometry": "InnerTracker_Barrel_radius_1",
                    "evaluators": {"test.echo": "x"},
                },
                "type": "continuous",
                "bounds": [100.0, 600.0],
                "unit": "mm",
            }
        },
        "evaluators": [{"name": "test.echo", "version": "1"}],
        "objectives": [metric | {"direction": "minimize"}],
        "constraints": [metric | {"max": 500, "unit": "1"}],
    }
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(data: dict[str, Any]) -> None:
        *parents, last = path.split("/")
        node: Any = data
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        node[last] = value

    return change


def _measured_in(unit: str) -> Callable[[dict[str, Any]], None]:
    return _set(f"{ECHOED}/measurements/0/unit", unit)


def _undeclared_metrics_measured_in(unit: str) -> Callable[[dict[str, Any]], None]:
    def change(data: dict[str, Any]) -> None:
        _set(f"{DECLARED}/metrics", None)(data)
        _measured_in(unit)(data)

    return change


# (what the live implementation declares or does, how the record is edited to say the same)
CASES: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], None]]] = {
    "another version": ({"version": "2"}, _set(f"{DECLARED}/version", "2")),
    "another stage": ({"stage": "simulation"}, _set(f"{DECLARED}/stage", "simulation")),
    "an objective metric it does not declare": (
        {"metrics": {("analysis", "other"): "1"}},
        _set(f"{DECLARED}/metrics", [{"stage": "analysis", "metric": "other", "unit": "1"}]),
    ),
    "a constraint unit it does not declare": (
        {"metrics": {ECHO: "mm"}},
        _set(f"{DECLARED}/metrics/0/unit", "mm"),
    ),
    "a measurement in another unit than declared": ({"unit": "mm"}, _measured_in("mm")),
    "a measurement in another unit than a constraint's": (
        {"metrics": None, "unit": "mm"},
        _undeclared_metrics_measured_in("mm"),
    ),
    "a valid result without the metric the problem names": (
        {"evaluate": Silent.evaluate},
        _set(f"{ECHOED}/measurements", []),
    ),
    "an artifact it does not declare": (
        {"artifacts": {"plot": "a plot"}, "returned": ("note",)},
        _set(f"{DECLARED}/artifacts", ["plot"]),
    ),
    "a valid result without an artifact it declares": (
        {"artifacts": {"note": "what it measured", "plot": "a plot"}},
        _set(f"{DECLARED}/artifacts", ["note", "plot"]),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_record_is_refused_with_the_words_a_live_session_refuses_it_with(
    tmp_path: Path, problem_file: Path, case: str
) -> None:
    declared, edit = CASES[case]
    variant = type("Echo", (Echo,), declared)
    with (
        pytest.raises(EvaluationContractError) as live,
        EvaluationSession.open(problem_file, [variant]) as session,
    ):
        session.evaluate({RADIUS: 450.0}, tmp_path / "live")
    with EvaluationSession.open(problem_file, [Echo]) as session:
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "recorded")
        spec = session.spec
    summary = json.loads(outcome.summary_path.read_text())
    assert Assessment.from_summary(spec, summary).feasible is True  # as it was made
    edit(summary)
    with pytest.raises(RecordError) as recorded:
        Assessment.from_summary(spec, summary)
    assert str(recorded.value) == str(live.value)


def test_a_record_is_read_back_without_its_implementation(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the record holds what the implementation declared: reading it needs neither its file nor
    # its module, which are gone here
    monkeypatch.setattr(sys, "path", list(sys.path))  # load_evaluator puts tmp_path first
    source = tmp_path / "echo_gone.py"
    source.write_text(
        "from k4opt.evaluation import EvaluationResult, Measurement, Quantity\n"
        "class Echo:\n"
        "    name = 'test.echo'\n"
        "    version = '1'\n"
        "    settings = {'x': Quantity('mm')}\n"
        f"    metrics = {{{ECHO!r}: '1'}}\n"
        "    def evaluate(self, request):\n"
        "        value = request.settings['x']\n"
        f"        return EvaluationResult.valid([Measurement(*{ECHO!r}, value, '1')])\n"
    )
    loaded = load_evaluator(f"{source}:Echo")
    with EvaluationSession.open(problem_file, [loaded]) as session:
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        spec = session.spec
    for name in [name for name, m in sys.modules.items() if m is sys.modules[loaded.__module__]]:
        del sys.modules[name]
    del loaded, session
    source.unlink()
    summary = json.loads(outcome.summary_path.read_text())
    assert summary["implementations"]["test.echo"]["source"]["path"] == str(source)
    assessment = Assessment.from_summary(spec, summary)
    assert assessment.to_dict() == summary["assessment"] and assessment.feasible is True
