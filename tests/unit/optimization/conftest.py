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
    objectives: list[tuple[str, str, str]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    parameter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A problem measured by ``test.trade_off`` unless it says otherwise."""
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
            {"stage": s, "metric": m, "direction": d}
            for s, m, d in objectives or [(*A, "minimize"), (*B, "minimize")]
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


def evaluator_from(directory: Path, name: str, source: str) -> Any:
    """The evaluator class ``Probe`` of ``source``, written to ``directory/name.py`` and loaded
    as ``k4opt optimize --evaluator`` loads one, so that its source is identified."""
    from k4opt.evaluation import load_evaluator

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return load_evaluator(f"{path}:Probe")
