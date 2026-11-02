"""The lifecycle of an evaluator, alike in the session's own process and in worker processes: its
class's ``check`` once per evaluation, where the session runs, before anything of the candidate is
created; then one new instance per evaluation, where the evaluation runs, created right before its
``evaluate()``. Logged by ``test.lifecycle`` from whatever process did it."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.execution import EvaluationSession, RunOutcome, RunRequest
from tests.support import stack
from tests.support.evaluators import cell_count, lifecycle
from tests.support.geometry import MINI_COMPACT
from tests.support.problems import add_targets
from tests.unit.execution.test_parallel import mini_k4geo  # noqa: F401 - a fixture

RADIUS = "radius"


def problem(path: Path, doc: dict[str, Any]) -> Path:
    """A problem whose radius sets the geometry and the setting ``x`` of ``test.lifecycle``,
    which measures the candidate and reads no stage."""
    doc = copy.deepcopy(doc)
    doc["geometry"] = MINI_COMPACT
    parameter = {
        "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
        "type": "continuous",
        "bounds": [100.0, 600.0],
        "unit": "mm",
    }
    doc["parameters"] = {RADIUS: add_targets(parameter, evaluators={"test.lifecycle": "x"})}
    doc["evaluators"] = [
        {"name": "test.lifecycle", "version": "1"},
        # what simulates the sample, which a problem needs
        {
            "name": "test.cell_count",
            "version": "1",
            "stage": "simulation",
            "collections": {"hits": "ECalBarrelCollection"},
        },
    ]
    doc["objectives"] = [
        {"evaluator": "test.lifecycle", "stage": "analysis", "metric": "x", "direction": "minimize"}
    ]
    del doc["constraints"]
    path.write_text(yaml.safe_dump(doc))
    return path


@pytest.mark.parametrize("workers", ["inline", 2])
def test_each_evaluation_checks_its_request_and_creates_one_instance_where_it_runs(
    tmp_path: Path, doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch, workers: int | str
) -> None:
    log = tmp_path / "lifecycle.log"
    monkeypatch.setenv("LIFECYCLE_LOG", str(log))
    stack.install(monkeypatch)
    values = [150.0, 300.0, 450.0]
    requests = [RunRequest({RADIUS: x}, tmp_path / f"c{i}") for i, x in enumerate(values)]
    path = problem(tmp_path / "p.yaml", doc)
    evaluators = [lifecycle(), cell_count()]
    with EvaluationSession.open(path, evaluators, workers=workers) as session:  # type: ignore[arg-type]
        assert not log.exists()  # binding the class created nothing and ran none of its code
        with session.evaluate_each(requests) as evaluations:
            outcomes = [outcome for _, outcome in evaluations]
    assert all(isinstance(o, RunOutcome) and o.candidate.status == "valid" for o in outcomes)
    lines = [line.split() for line in log.read_text().splitlines()]
    here = str(os.getpid())
    checks = [rest for kind, *rest in lines if kind == "check"]
    inits = [rest for kind, *rest in lines if kind == "init"]
    evaluated = [rest for kind, *rest in lines if kind == "evaluate"]
    # the check of each candidate's request, once, in the session's own process
    assert checks == [[here, str(x)] for x in values]
    # one instance per evaluation, no other, created where it evaluates: never in the session's
    # process with workers, which would only discard it
    assert len(inits) == len(evaluated) == len(values)
    assert [pid for pid, _ in inits] == [pid for pid, *_ in evaluated]
    assert {pid == here for pid, _ in inits} == {workers == "inline"}
    # each evaluates its own candidate, from a fresh state: nothing an instance kept reaches
    # another candidate
    assert sorted(float(x) for _, _, x, _ in evaluated) == values
    assert {calls for *_, calls in evaluated} == {"0"}
    assert [instance for _, instance in inits] == [instance for _, instance, *_ in evaluated]
