"""Shared fixtures of the optimization tests: problems on the mini geometry whose evaluators
read no stage, so a study of a few trials runs in a fraction of a second."""

from __future__ import annotations

import copy
import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.geometry import MINI_COMPACT, MINI_GEO
from tests.support.problems import add_targets
from tests.support.software import VIEW

RADIUS = "radius"
# each reads the candidate from its setting x
READERS = ("test.trade_off", "test.candidate_check", "test.file_size", "test.slashed")
A = ("analysis", "a")
B = ("analysis", "b")


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    monkeypatch.setenv("KEY4HEP_STACK", str(VIEW))


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
    evaluators = evaluators or [{"name": "test.trade_off", "version": "1"}]
    parameter = copy.deepcopy(parameter) or {
        "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
        "type": "continuous",
        "bounds": [100.0, 600.0],
    }
    # the evaluators that read the candidate read it from their setting x, which it sets
    readers = dict.fromkeys(e.get("as", e["name"]) for e in evaluators if e["name"] in READERS)
    if readers:
        add_targets(parameter, evaluators=dict.fromkeys(readers, "x"))
    document: dict[str, Any] = {
        "schema_version": "k4opt.problem/v1alpha1",
        "name": "trade",
        "geometry": MINI_COMPACT,
        "parameters": {RADIUS: parameter},
        "evaluators": evaluators,
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
            "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
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


def document(study: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((study / "study.json").read_text(encoding="utf-8"))
    return data


def ledger(study: Path) -> Any:
    """The ledger of a study directory, read as ``load_study`` reads it."""
    from k4opt.optimization._ledger import read_ledger

    return read_ledger(study, document(study)["id"])


def endings(study: Path) -> list[str | None]:
    """How each trial of a study ended, by number, as its ledger holds it (``None``: running)."""
    return [None if e.ending is None else e.ending.kind for e in ledger(study).entries]


def proposal(problem: Path) -> dict[str, Any]:
    """Values of every parameter of ``problem``, each in its domain: a midpoint, the lower
    integer bound, the first choice."""
    from k4opt.problem import load_problem

    values: dict[str, Any] = {}
    for parameter in load_problem(problem).parameters:
        if parameter.choices is not None:
            values[parameter.name] = parameter.choices[0]
        else:
            assert parameter.bounds is not None
            low, high = parameter.bounds
            values[parameter.name] = low if parameter.kind == "integer" else (low + high) / 2
    return values


def left_running(
    output: Path,
    problem: Path,
    directory: bool = False,
    values: dict[str, Any] | None = None,
    note: dict[str, Any] | None = None,
) -> int:
    """A trial proposed and left running, as a session killed just after its proposal leaves it:
    with ``values`` (those of :func:`proposal` by default) and what the strategy noted of them,
    its directory partly made if ``directory``."""
    from k4opt.optimization._ledger import LedgerWriter

    writer, _ = LedgerWriter.open(output, document(output)["id"])
    try:
        number = writer.next_number
        searched = document(output)["scope"]["searched"]
        if values is None:
            values = {k: v for k, v in proposal(problem).items() if k in searched}
        writer.session("2026-11-19T00:00:00+00:00")
        writer.proposed(number, values, note or {})
    finally:
        writer.close()
    if directory:
        name = f"trial-{number:06d}"
        (output / name / "geometry").mkdir(parents=True)
        (output / name / "geometry" / "partial.xml").write_text("<lccdd>")
    return number


GRID = {RADIUS: [150.0, 250.0, 350.0, 450.0, 550.0, 600.0]}
"""Grid values of the radius of :func:`problem_document`'s problem."""

STRATEGY_KWARGS: dict[str, dict[str, Any]] = {
    "optuna": {"seed": 1},
    "grid": {"strategy": "grid", "grid": GRID},
}
"""How ``optimize`` runs a new study of :func:`problem_document`'s problem with each strategy."""


@pytest.fixture(params=list(STRATEGY_KWARGS))
def strategy(request: pytest.FixtureRequest) -> str:
    """Each strategy in turn; Optuna's only where Optuna is installed."""
    if request.param == "optuna":
        pytest.importorskip("optuna")
    name: str = request.param
    return name


@dataclass
class Told:
    """What every strategy opened while it is installed proposed and was told: each proposal and
    each feedback as ``(level, study, number, ...)``, where ``study`` is ``top``, or for a study
    below it ``<parent study id>:<parent trial>``; ``events`` holds both in the order they came,
    as ``(proposed|told, level, study, number)``."""

    events: list[tuple[str, int, str, int]] = field(default_factory=list)
    proposed: list[tuple[int, str, int, Any]] = field(default_factory=list)
    told: list[tuple[int, str, int, Any]] = field(default_factory=list)
    # each strategy opened, with the trials it was made from
    opened: list[tuple[int, str, list[Any]]] = field(default_factory=list)

    def order(self, study: str) -> list[tuple[int, int]]:
        """When each trial of ``study`` was proposed and told, as positions in :attr:`events`,
        in number order."""
        at: dict[tuple[str, int], int] = {}
        for index, (kind, _, name, number) in enumerate(self.events):
            if name == study:
                at.setdefault((kind, number), index)
        numbers = sorted({number for _, number in at})
        return [(at[("proposed", n)], at.get(("told", n), len(self.events))) for n in numbers]

    def feedback(self, level: int = 0) -> dict[int, Any]:
        """The feedback of each trial of the studies of ``level``, by number (of the last)."""
        return {number: told for at, _, number, told in self.told if at == level}


@pytest.fixture
def told(monkeypatch: pytest.MonkeyPatch) -> Told:
    """Record what every strategy a study opens from now on proposes and is told."""
    from k4opt.optimization import _tree

    log, real = Told(), _tree.open_strategy

    class Recording:
        def __init__(self, inner: Any, level: int, study: str) -> None:
            self.inner, self.level, self.study = inner, level, study

        def propose(self, number: int) -> Any:
            answer = self.inner.propose(number)
            log.proposed.append((self.level, self.study, number, answer))
            log.events.append(("proposed", self.level, self.study, number))
            return answer

        def tell(self, number: int, feedback: Any) -> None:
            log.told.append((self.level, self.study, number, feedback))
            log.events.append(("told", self.level, self.study, number))
            self.inner.tell(number, feedback)

    def opening(recorded: Any, scope: Any, history: Any) -> Any:
        parent = scope.parent
        study = "top" if parent is None else f"{parent['study']}:{parent['trial']}"
        log.opened.append((scope.level, study, list(history)))
        return Recording(real(recorded, scope, history), scope.level, study)

    monkeypatch.setattr(_tree, "open_strategy", opening)
    return log


def evaluator_from(directory: Path, name: str, source: str) -> Any:
    """The evaluator class ``Probe`` of ``source``, written to ``directory/name.py`` and loaded
    as ``k4opt optimize --evaluator`` loads one, so that its source is identified."""
    from k4opt.evaluation import load_evaluator

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return load_evaluator(f"{path}:Probe")
