"""Fixtures and helpers of the session tests: ``run_candidate``, ``k4opt evaluate``
and ``EvaluationSession`` with a fake ddsim, on the mini geometry."""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.geometry import GeometrySource
from k4opt.problem import (
    RunOutcome,
)
from k4opt.problem._run import run_candidate as _run_candidate
from tests.support import simulation
from tests.support.evaluators import (
    CELL_COUNT,
    cell_count,
    implementations,
)
from tests.support.geometry import MINI_COMPACT, MINI_GEO
from tests.support.simulation import COLLECTION, events_with

RADIUS = "constants.InnerTracker_Barrel_radius_1"


# the mini geometry has no calorimeter, so ddsim is not expected to write the collection the test
# evaluator reads; the fake ddsim writes it anyway, and the output decides
UNEXPECTED = (
    f"evaluator 'test.cell_count': ddsim is not expected to write the collection {COLLECTION!r} "
    f"(hits) for {MINI_COMPACT}"
)


def run_candidate(
    problem: Path, assignments: list[str], output: Path, **changes: Any
) -> RunOutcome:
    """``run_candidate`` with the test evaluator unless the test passes ``evaluators`` of its own:
    classes, or instances to watch."""
    changes["implementations"] = implementations(*changes.pop("evaluators", [cell_count()]))
    return _run_candidate(problem, assignments, output, **changes)


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def problem_file(tmp_path: Path, doc: dict[str, Any]) -> Path:
    doc["geometry"] = MINI_COMPACT
    doc["parameters"][0]["geometry"] = {"constant": "InnerTracker_Barrel_radius_1"}
    doc["parameters"][0]["bounds"] = [100.0, 600.0]
    doc["samples"]["gamma"]["simulation"].update(events=3, seed=5)
    doc["evaluators"] = [
        {
            "name": "test.cell_count",
            "version": "1",
            "stage": "simulation",
            "collections": {"hits": COLLECTION},
        }
    ]
    doc["objectives"] = [
        {"stage": "simulation", "metric": "ecal_barrel.mean_hit_cells", "direction": "minimize"}
    ]
    del doc["constraints"]
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> simulation.FakeDdsim:
    return simulation.install(monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9]))


@pytest.fixture
def guard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every discovery and materialization (and lets them run)."""
    calls: list[str] = []
    discover, materialize = GeometrySource.discover, GeometrySource.materialize

    def spy_discover(self: GeometrySource) -> Any:
        calls.append("discover")
        return discover(self)

    def spy_materialize(self: GeometrySource, *args: Any, **kwargs: Any) -> Any:
        calls.append("materialize")
        return materialize(self, *args, **kwargs)

    monkeypatch.setattr(GeometrySource, "discover", spy_discover)
    monkeypatch.setattr(GeometrySource, "materialize", spy_materialize)
    return calls


def errors(err: str) -> str:
    """``err`` without the warnings printed before the run."""
    return "".join(line for line in err.splitlines(True) if not line.startswith("k4opt: warning:"))


def evaluate(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(["evaluate", "--evaluator", CELL_COUNT, *argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


class Spy:
    """The test evaluator, recording the requests of check() and evaluate()."""

    def __init__(self) -> None:
        self.inner: Any = cell_count()()
        self.name, self.version = self.inner.name, self.inner.version
        self.stage, self.metrics = self.inner.stage, self.inner.metrics
        self.collections = self.inner.collections
        self.checked: list[Any] = []
        self.evaluated: list[Any] = []

    def check(self, request: Any) -> None:
        self.checked.append(request)
        self.inner.check(request)

    def evaluate(self, request: Any) -> Any:
        self.evaluated.append(request)
        return self.inner.evaluate(request)


CELLS = ("simulation", "ecal_barrel.mean_hit_cells")


ENERGY = ("simulation", "ecal_barrel.mean_hit_energy")


ECHO = ("analysis", "candidate.first_value")


OWNERS = {CELLS: "test.cell_count", ENERGY: "test.hit_energy", ECHO: "test.parameter_echo"}


def declare(problem_file: Path, names: list[str], objectives: list[tuple[str, str]]) -> None:
    """Make the problem declare the test evaluators ``names`` and these objectives, each on the
    metric of the evaluator that measures it (:data:`OWNERS`)."""
    doc = yaml.safe_load(problem_file.read_text())
    reads = {"stage": "simulation", "collections": {"hits": COLLECTION}}
    doc["evaluators"] = [
        {"name": name, "version": "1"} | ({} if name == "test.parameter_echo" else reads)
        for name in names
    ]
    doc["objectives"] = [
        {"evaluator": OWNERS[(s, m)], "stage": s, "metric": m, "direction": "minimize"}
        for s, m in objectives
    ]
    problem_file.write_text(yaml.safe_dump(doc))


def change(data: dict[str, Any], path: str, value: Any) -> None:
    """Set the entry at ``path`` of ``data``, its keys separated by ``/`` (list items by index),
    to ``value``."""
    *parents, last = path.split("/")
    for key in parents:
        data = data[int(key)] if isinstance(data, list) else data[key]
    if isinstance(data, list):
        data[int(last)] = value
    else:
        data[last] = value


def two_samples(problem_file: Path, **pion: Any) -> dict[str, Any]:
    """Give the problem a second sample, ``pion`` (negative pions, unless ``pion`` says other
    stages), which the evaluator measures too, with an objective on each sample; the document."""
    doc = yaml.safe_load(problem_file.read_text())
    stages = copy.deepcopy(doc["samples"]["gamma"])
    stages["generation"]["particle"] = "pi-"
    stages["simulation"]["seed"] = 6
    doc["samples"]["pion"] = pion or stages
    doc["evaluators"][0]["samples"] = ["gamma", "pion"]
    objective = doc["objectives"][0]
    doc["objectives"] = [objective | {"sample": "gamma"}, objective | {"sample": "pion"}]
    problem_file.write_text(yaml.safe_dump(doc))
    return doc


def particle(argv: list[str]) -> str:
    """The particle a ddsim command line fires."""
    return next(a for a in argv if a.startswith("--gun.particle=")).partition("=")[2]


SIGNIFICANCE = ("analysis", "expected_significance")


def combined(problem_file: Path, **changes: Any) -> dict[str, Any]:
    """Make the problem a toy signal over two backgrounds: the samples ``sig`` (photons), ``zz``
    and ``ww`` (pions), each with a cross section, measured by ``test.yields`` and combined by
    ``test.toy_significance`` as ``significance``, whose significance is the objective;
    ``changes`` replace entries of the document. The document."""
    doc = yaml.safe_load(problem_file.read_text())
    gamma = doc["samples"].pop("gamma")
    for seed, (name, particle, cross_section) in enumerate(
        [("sig", "gamma", 200), ("zz", "pi-", 1360), ("ww", "pi+", 16400)], start=1
    ):
        stages = copy.deepcopy(gamma)
        stages["generation"]["particle"] = particle
        stages["simulation"]["seed"] = seed
        properties = {"cross_section": {"value": cross_section, "unit": "fb"}}
        doc["samples"][name] = {"properties": properties, **stages}
    doc["evaluators"] = [
        {
            "name": "test.yields",
            "version": "1",
            "stage": "simulation",
            "samples": ["sig", "zz", "ww"],
            "collections": {"hits": COLLECTION},
        },
        {
            "name": "test.toy_significance",
            "as": "significance",
            "version": "1",
            # in another order than the evaluator declares them, which does not matter
            "roles": {
                "backgrounds": {"samples": ["zz", "ww"], "evaluator": "test.yields"},
                "signal": {"sample": "sig", "evaluator": "test.yields"},
            },
            "settings": {"luminosity_invab": 10.8},
        },
    ]
    doc["objectives"] = [
        {"evaluator": "significance", "stage": "analysis", "metric": SIGNIFICANCE[1]}
        | {"direction": "maximize"}
    ]
    doc |= changes
    problem_file.write_text(yaml.safe_dump(doc, sort_keys=False))
    return doc
