"""Shared fixtures for the problem unit tests."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping

import pytest

from k4opt.problem._schedule import Scheduler
from k4opt.problem._tasks import Plan, Task, TaskId, TaskResult

_BASE: dict[str, object] = {
    "schema_version": "k4opt.problem/v1alpha1",
    "name": "demo",
    "environment": {"key4hep": {"channel": "nightly", "version": "2026-10-20"}},
    "geometry": "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml",
    "parameters": [
        {
            "geometry": {"constant": "ECal_cell_size"},
            "type": "continuous",
            "bounds": [4.0, 8.0],
            "unit": "mm",
        }
    ],
    "samples": {
        "gamma": {
            "generation": {
                "executor": "gun",
                "particle": "gamma",
                "energy": {"value": 10, "unit": "GeV"},
                "direction": {
                    "distribution": "uniform",
                    "theta": [80, 100],
                    "phi": [0, 360],
                    "unit": "deg",
                },
            },
            "simulation": {
                "executor": "ddsim",
                "events": 1000,
                "seed": 42,
                "physics.list": "FTFP_BERT",
            },
        },
    },
    "evaluators": [
        {
            "name": "calo.single_particle_shower",
            "version": "1",
            "stage": "simulation",
            "settings": {"collection": "ECalBarrelCollection"},
        }
    ],
    "objectives": [{"stage": "simulation", "metric": "a.rms", "direction": "minimize"}],
    "constraints": [{"stage": "simulation", "metric": "a.cells", "max": 100}],
}


@pytest.fixture
def doc() -> dict[str, object]:
    """A fresh, valid problem document (plain dict) that a test may mutate."""
    return copy.deepcopy(_BASE)


def run_in_order(
    plan: Plan, run: Callable[[Task, Mapping[TaskId, TaskResult]], TaskResult]
) -> dict[TaskId, TaskResult]:
    """The result of each task of ``plan``, in plan order, from ``run`` called by the scheduler
    with one slot, as a session with one job runs them: each task once, after what it reads,
    given the results of those tasks only; a task whose needs did not all succeed is blocked and
    not run."""
    scheduler = Scheduler(1)
    scheduler.add("candidate", plan)
    while (item := scheduler.next()) is not None:
        _, task = item
        done = scheduler.results("candidate")
        scheduler.finished("candidate", run(task, {need: done[need] for need in task.needs}))
    results = scheduler.remove("candidate")
    return {task.id: results[task.id] for task in plan.tasks}
