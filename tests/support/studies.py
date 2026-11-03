"""Real studies for the tests of what reads studies back (publication, visualization, dashboard).

Each runs ``k4opt optimize`` with the test evaluator ``test.trade_off`` (``a = x`` in mm,
``b = 1/x`` in 1/mm) on the mini geometry, in a fraction of a second, and is clearly test data:
its problem is called ``trade``. Needs Optuna, and the environment :func:`mini_environment`
sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.geometry import MINI_GEO

# what each radius gives with invalid_below 200, failed_below 150 and the limit a <= 400
MIXED = {300: "feasible", 500: "infeasible", 170: "invalid", 120: "failed"}
TRADE = {"name": "test.trade_off", "version": "1"}


def mini_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The k4geo tree and Key4hep stack a study of the mini geometry is made in."""
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    monkeypatch.setenv("KEY4HEP_STACK", "/stack/2026-10-20")


def study(
    directory: Path, trials: int, *, implementations: list[Any] | None = None, **changes: Any
) -> Path:
    """A study of ``trials`` trials in ``directory``, its problem written beside it, measured by
    ``implementations`` (``test.trade_off`` by default); ``changes`` are those of
    :func:`tests.unit.optimization.conftest.problem_document`."""
    from k4opt.optimization import optimize
    from tests.support.evaluators import trade_off
    from tests.unit.optimization.conftest import problem_document

    problem = directory.with_name(f"{directory.name}.yaml")
    problem.write_text(yaml.safe_dump(problem_document(**changes)), encoding="utf-8")
    optimize(problem, implementations or [trade_off()], directory, trials=trials, seed=1)
    return directory


def mixed_study(directory: Path, trials: int = 8) -> Path:
    """Two objectives (``a`` and ``b``, both minimized) and the limit ``a <= 400 mm``, whose
    trials end every way a record can: feasible, infeasible, invalid and failed."""
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
