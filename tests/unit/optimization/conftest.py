"""Shared fixtures of the optimization tests: problems on the mini geometry whose evaluators
read no stage, so a study of a few trials runs in a fraction of a second."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.geometry import MINI_COMPACT, MINI_GEO

RADIUS = "constants.InnerTracker_Barrel_radius_1"
A = ("analysis", "a")
B = ("analysis", "b")


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    monkeypatch.setenv("KEY4HEP_STACK", "/stack/2026-10-20")


def problem_document(
    *,
    evaluators: list[dict[str, Any]] | None = None,
    objectives: list[tuple[str, ...]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    parameter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A problem measured by ``test.trade_off`` unless it says otherwise. An objective is
    ``(stage, metric, direction)``, or ``(evaluator, stage, metric, direction)`` for a problem
    of several evaluators."""
    document: dict[str, Any] = {
        "schema_version": "k4opt.problem/v1alpha1",
        "name": "trade",
        "environment": {"key4hep": {"channel": "nightly", "version": "2026-10-20"}},
        "geometry": MINI_COMPACT,
        "parameters": [
            parameter
            or {
                "geometry": {"constant": "InnerTracker_Barrel_radius_1"},
                "type": "continuous",
                "bounds": [100.0, 600.0],
                "unit": "mm",
            }
        ],
        "evaluators": evaluators or [{"name": "test.trade_off", "version": "1"}],
        "objectives": [
            dict(zip(("evaluator", "stage", "metric", "direction")[-len(o) :], o, strict=True))
            for o in objectives or [(*A, "minimize"), (*B, "minimize")]
        ],
    }
    if constraints:
        document["constraints"] = constraints
    return document


@pytest.fixture
def write_problem(tmp_path: Path) -> Callable[..., Path]:
    """Write a :func:`problem_document` to ``tmp_path/problem.yaml``; its path."""

    def write(**changes: Any) -> Path:
        path = tmp_path / "problem.yaml"
        path.write_text(yaml.safe_dump(problem_document(**changes)), encoding="utf-8")
        return path

    return write


GUN = {
    "executor": "gun",
    "energy": {"value": 10, "unit": "GeV"},
    "direction": {"distribution": "uniform", "theta": [80, 100], "phi": [0, 360], "unit": "deg"},
}
CELLS = ("simulation", "ecal_barrel.mean_hit_cells")


@pytest.fixture
def sampled_problem(write_problem: Callable[..., Path]) -> Path:
    """``test.cell_count`` on two samples, ``gamma`` and ``pion``, of 2 events each, with an
    objective on each; a test installs the fake ddsim (:mod:`tests.support.simulation`)."""
    path = write_problem(
        evaluators=[
            {
                "name": "test.cell_count",
                "version": "1",
                "stage": "simulation",
                "samples": ["gamma", "pion"],
                "collections": {"hits": "ECalBarrelCollection"},
            }
        ],
        objectives=[("test.cell_count", *CELLS, "minimize")],
    )
    document = yaml.safe_load(path.read_text())
    document["samples"] = {
        name: {
            "generation": GUN | {"particle": particle},
            "simulation": {"executor": "ddsim", "events": 2, "seed": seed},
        }
        for name, particle, seed in (("gamma", "gamma", 1), ("pion", "pi-", 2))
    }
    objective = document["objectives"][0]
    document["objectives"] = [objective | {"sample": "gamma"}, objective | {"sample": "pion"}]
    document["objectives"][1]["direction"] = "maximize"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture
def combined_problem(write_problem: Callable[..., Path]) -> Path:
    """``test.yields`` on a signal sample, ``sig`` (photons), and a background, ``bkg`` (pions),
    of 2 events each, combined by ``test.toy_significance`` as ``significance``, whose expected
    significance is the objective; a test installs the fake ddsim."""
    yields = {"name": "test.yields", "version": "1", "stage": "simulation"}
    path = write_problem(
        evaluators=[
            yields | {"samples": ["sig", "bkg"], "collections": {"hits": "ECalBarrelCollection"}},
            {
                "name": "test.toy_significance",
                "as": "significance",
                "version": "1",
                "roles": {
                    "signal": {"sample": "sig", "evaluator": "test.yields"},
                    "backgrounds": {"sample": "bkg", "evaluator": "test.yields"},
                },
                "settings": {"luminosity_invab": 1.0},
            },
        ],
        objectives=[("significance", "analysis", "expected_significance", "maximize")],
    )
    document = yaml.safe_load(path.read_text())
    document["samples"] = {
        name: {
            "properties": {"cross_section": {"value": cross_section, "unit": "fb"}},
            "generation": GUN | {"particle": particle},
            "simulation": {"executor": "ddsim", "events": 2, "seed": seed},
        }
        for name, particle, seed, cross_section in (("sig", "gamma", 1, 2), ("bkg", "pi-", 2, 30))
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


TRADE = {"name": "test.trade_off", "version": "1"}
# what each radius gives, with invalid_below 200, failed_below 150 and the limit a <= 400
OUTCOME = {300: "feasible", 500: "infeasible", 170: "invalid", 120: "failed"}


@pytest.fixture
def mixed_problem(write_problem: Callable[..., Path]) -> Path:
    """A problem whose trials end every way a record can, some of them with equal values."""
    return write_problem(
        parameter={
            "geometry": {"constant": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": list(OUTCOME),
        },
        evaluators=[TRADE | {"settings": {"invalid_below": 200, "failed_below": 150}}],
        constraints=[{"stage": "analysis", "metric": "a", "max": 400, "unit": "mm"}],
    )


@pytest.fixture
def diagnosed_problem(write_problem: Callable[..., Path]) -> Path:
    """``test.trade_off``, whose metrics are the objectives, and a diagnostic: the same
    implementation bound ``as: check``, which no objective or constraint reads and which fails
    below 350 mm."""
    return write_problem(
        evaluators=[TRADE, TRADE | {"as": "check", "settings": {"failed_below": 350}}],
        objectives=[("test.trade_off", *A, "minimize"), ("test.trade_off", *B, "minimize")],
    )


@dataclass
class Log:
    """What a study reported while it ran."""

    warnings: list[str] = field(default_factory=list)
    trials: list[Any] = field(default_factory=list)

    def kwargs(self) -> dict[str, Any]:
        return {"warn": self.warnings.append, "progress": self.trials.append}


def summary(study: Path, number: int) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (study / f"trial-{number:06d}" / "summary.json").read_text(encoding="utf-8")
    )
    return data


def optuna_study(study: Path) -> Any:
    """The Optuna study of a study directory, read from its journal."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    document = json.loads((study / "study.json").read_text(encoding="utf-8"))
    path = str(study / "study.journal")
    storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    return optuna.load_study(study_name=document["optimizer"]["study"], storage=storage)


def left_running(output: Path, problem: Path, directory: bool = False) -> int:
    """A trial asked for and left RUNNING, as a session killed just after asking leaves it."""
    from k4opt.optimization._optuna.space import search_space
    from k4opt.problem import load_problem

    study = optuna_study(output)
    trial = study.ask(fixed_distributions=search_space(load_problem(problem)))
    if directory:
        name = f"trial-{trial.number:06d}"
        trial.set_user_attr("k4opt.dir", name)
        (output / name / "geometry").mkdir(parents=True)
        (output / name / "geometry" / "partial.xml").write_text("<lccdd>")
    return int(trial.number)


def evaluator_from(directory: Path, name: str, source: str) -> Any:
    """The evaluator class ``Probe`` of ``source``, written to ``directory/name.py`` and loaded
    as ``k4opt optimize --evaluator`` loads one, so that its source is identified."""
    from k4opt.evaluation import load_evaluator

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return load_evaluator(f"{path}:Probe")
