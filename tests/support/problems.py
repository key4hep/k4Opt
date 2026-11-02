"""Problems the unit tests start from, and the order a session with one job runs a plan in."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from k4opt.execution._schedule import Scheduler
from k4opt.problem.plan import Plan, Task, TaskId
from k4opt.record.results import TaskResult

_BASE: dict[str, object] = {
    "schema_version": "k4opt.problem/v1alpha1",
    "name": "demo",
    "environment": {"key4hep": {"channel": "nightly", "version": "2026-10-20"}},
    "geometry": "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml",
    "parameters": {
        "ecal_cell_size": {
            "type": "continuous",
            "bounds": [4.0, 8.0],
            "unit": "mm",
            "sets": {"geometry": {"constant": "ECal_cell_size"}},
        }
    },
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


BASE = _BASE
"""A valid problem document; :func:`document` gives a copy to change."""


def document() -> dict[str, object]:
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


def two_samples(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` with a second sample, ``pion``, which the evaluator measures too."""
    pion = copy.deepcopy(doc["samples"]["gamma"])
    pion["generation"]["particle"] = "pi-"
    pion["simulation"]["seed"] = 43
    doc["samples"]["pion"] = pion
    doc["evaluators"][0]["samples"] = ["gamma", "pion"]
    for item in (*doc["objectives"], *doc["constraints"]):
        item["sample"] = "gamma"
    return doc
