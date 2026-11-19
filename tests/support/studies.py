"""Real studies for the tests of what reads studies back (publication, visualization, dashboard).

Each runs ``k4opt optimize`` with the test evaluator ``test.trade_off`` (``a = x`` in mm,
``b = 1/x`` in 1/mm) on the mini geometry, in a fraction of a second, and is clearly test data:
its problem is called ``trade``. Each is a grid study, so that nothing here needs Optuna; it needs
the environment :func:`mini_environment` sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.geometry import MINI_GEO
from tests.support.software import VIEW

# what each radius gives with invalid_below 200, failed_below 150 and the limit a <= 400
MIXED = {
    300: "feasible",
    500: "infeasible",
    170: "invalid",
    120: "failed",
    350: "feasible",
    450: "infeasible",
    250: "feasible",
    400: "feasible",
}
TRADE = {"name": "test.trade_off", "version": "1"}


def mini_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The k4geo tree and Key4hep stack a study of the mini geometry is made in."""
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    monkeypatch.setenv("KEY4HEP_STACK", str(VIEW))


def grid_of(document: dict[str, Any], trials: int) -> dict[str, list[float]]:
    """Grid values of each continuous parameter of the problem ``document``: ``trials`` points
    spread evenly over its bounds, both included (one point: the middle)."""
    grid = {}
    for name, parameter in document["parameters"].items():
        if parameter["type"] == "continuous":
            low, high = parameter["bounds"]
            step = (high - low) / max(trials - 1, 1)
            points = [low + step * n for n in range(trials)] if trials > 1 else [(low + high) / 2]
            grid[name] = points
    return grid


def study(
    directory: Path, trials: int, *, implementations: list[Any] | None = None, **changes: Any
) -> Path:
    """A grid study of ``trials`` trials in ``directory`` (fewer if the grid has fewer points),
    its problem written beside it, measured by ``implementations`` (``test.trade_off`` by
    default); ``changes`` are those of :func:`tests.unit.optimization.conftest.problem_document`."""
    from k4opt.optimization import optimize
    from tests.support.evaluators import trade_off
    from tests.unit.optimization.conftest import problem_document

    document = problem_document(**changes)
    problem = directory.with_name(f"{directory.name}.yaml")
    problem.write_text(yaml.safe_dump(document), encoding="utf-8")
    grid = grid_of(document, trials)
    optimize(
        problem,
        implementations or [trade_off()],
        directory,
        trials=trials,
        strategy="grid",
        grid=grid,
    )
    return directory


def mixed_study(directory: Path, trials: int = 8) -> Path:
    """Two objectives (``a`` and ``b``, both minimized) and the limit ``a <= 400 mm``, whose
    trials end every way a record can, the first four in this order: feasible, infeasible,
    invalid and failed (a grid of the eight choices of :data:`MIXED`, of which ``trials`` are
    run)."""
    return study(
        directory,
        trials,
        parameter={
            "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": list(MIXED),
        },
        evaluators=[TRADE | {"settings": {"invalid_below": 200, "failed_below": 150}}],
        constraints=[{"stage": "analysis", "metric": "a", "max": 400, "unit": "mm"}],
    )


def nested_study(directory: Path, monkeypatch: pytest.MonkeyPatch, trials: int = 2) -> Path:
    """A nested grid study of ``trials`` trials in ``directory``: the radius of the geometry at
    the top, compared by the cells of a simulation (fake ddsim, in this process); for each radius,
    a study of 3 trials of the setting ``x`` of ``test.candidate_check`` (1.5, 1.0 and 2.0), tuned
    by its metric ``analysis/seen`` (minimized), which no objective of the problem names."""
    from k4opt.optimization import optimize
    from tests.support import simulation
    from tests.support.evaluators import candidate_check, cell_count
    from tests.unit.optimization.conftest import GUN, RADIUS, problem_document

    cells = {"stage": "simulation", "metric": "ecal_barrel.mean_hit_cells", "direction": "minimize"}
    document = problem_document(
        evaluators=[
            {
                "name": "test.cell_count",
                "version": "1",
                "stage": "simulation",
                "collections": {"hits": "ECalBarrelCollection"},
            },
            {"name": "test.candidate_check", "version": "1"},
        ],
        objectives=[("test.cell_count", *cells.values())],
    )
    document["parameters"][RADIUS]["targets"] = {"geometry": "InnerTracker_Barrel_radius_1"}
    document["parameters"]["x"] = {
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "targets": {"evaluators": {"test.candidate_check": "x"}},
    }
    document["samples"] = {
        "gamma": {
            "generation": GUN | {"particle": "gamma"},
            "simulation": {"executor": "ddsim", "events": 2, "seed": 1},
        }
    }
    seen = {"evaluator": "test.candidate_check", "stage": "analysis", "metric": "seen"}
    document["optimization"] = {
        "parameters": [RADIUS],
        "inner": {"parameters": ["x"], "trials": 3, "objective": seen | {"direction": "minimize"}},
    }
    problem = directory.with_name(f"{directory.name}.yaml")
    problem.write_text(yaml.safe_dump(document), encoding="utf-8")
    simulation.install(monkeypatch)
    grid = grid_of(document, trials) | {"x": [1.5, 1.0, 2.0]}
    implementations = [cell_count(), candidate_check()]
    optimize(problem, implementations, directory, trials=trials, strategy="grid", grid=grid)
    return directory
