"""Loading a problem resolves its geometry once; parse_problem stays pure."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.geometry import Catalog, GeometryError, GeometrySource, IncludeError, SourceError
from k4opt.problem import ProblemConfigError, load_problem, parse_problem
from k4opt.problem.config import resolve_problem
from tests.support.geometry import MINI_COMPACT, MINI_GEO

pytestmark = pytest.mark.usefixtures("mini_k4geo")


@pytest.fixture
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def problem_doc(doc: dict[str, Any]) -> dict[str, Any]:
    doc["geometry"] = MINI_COMPACT
    cell = doc["parameters"].pop("ecal_cell_size")
    doc["parameters"]["radius"] = cell | {"targets": {"geometry": "InnerTracker_Barrel_radius_1"}}
    return doc


def write_problem(tmp_path: Path, doc: dict[str, Any]) -> Path:
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_load_resolves_a_constant(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    problem = resolve_problem(write_problem(tmp_path, problem_doc))
    (target,) = problem.targets
    assert target.target == "constants.InnerTracker_Barrel_radius_1"
    assert target.value == "400*mm"  # raw text, neither evaluated nor converted
    assert target.file == "compact/det_v1/tracker.xml"
    assert target.line is not None
    assert problem.source == GeometrySource(MINI_COMPACT)
    assert problem.path == tmp_path / "problem.yaml"
    assert problem.directory == tmp_path


def test_load_resolves_a_structured_target(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    ref = "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"
    problem_doc["parameters"]["radius"] = {
        "targets": {"geometry": ref},
        "type": "integer",
        "bounds": [10, 100],
    }
    (target,) = resolve_problem(write_problem(tmp_path, problem_doc)).targets
    assert (target.target, target.attribute, target.value) == (ref, "nphi", "43*2")


def test_parameters_resolve_in_order_with_one_discovery(
    tmp_path: Path, problem_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem_doc["parameters"]["world_side"] = {
        "targets": {"geometry": "world_side"},
        "type": "continuous",
        "bounds": [1.0, 2.0],
    }
    calls: list[GeometrySource] = []
    discover = GeometrySource.discover

    def counting(self: GeometrySource) -> Any:
        calls.append(self)
        return discover(self)

    monkeypatch.setattr(GeometrySource, "discover", counting)
    problem = resolve_problem(write_problem(tmp_path, problem_doc))
    assert len(calls) == 1
    assert [t.target for t in problem.targets] == [
        "constants.InnerTracker_Barrel_radius_1",
        "constants.world_side",
    ]
    assert problem.catalog.root == problem.source.root


def test_one_parameter_sets_several_attributes(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    problem_doc["parameters"]["radius"]["targets"] = {
        "geometry": ["InnerTracker_Barrel_radius_1", "world_side"]
    }
    problem = resolve_problem(write_problem(tmp_path, problem_doc))
    assert [t.target for t in problem.targets] == [
        "constants.InnerTracker_Barrel_radius_1",
        "constants.world_side",
    ]
    assert [p.name for p, _ in problem.spec.geometry_destinations] == ["radius", "radius"]


def test_two_destinations_of_one_attribute_are_refused(
    tmp_path: Path, problem_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # two references that a geometry resolves to one attribute: refused when it is loaded,
    # not when the first candidate's geometry is made
    problem_doc["parameters"]["side"] = {
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "targets": {"geometry": "world_side"},
    }
    lookup = Catalog.lookup

    def same(self: Catalog, target: str, file: str | None = None) -> Any:
        return lookup(self, "constants.InnerTracker_Barrel_radius_1", file)

    monkeypatch.setattr(Catalog, "lookup", same)
    with pytest.raises(ProblemConfigError, match=r"which parameters\.radius sets already"):
        load_problem(write_problem(tmp_path, problem_doc))


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("InnerTracker_Barrel_radius_9", "unknown target"),
        ("constants.InnerTracker_Barrel_radius_9", "unknown target"),
        ("not a [reference", "not a [reference"),
    ],
)
def test_missing_or_malformed_target(
    tmp_path: Path, problem_doc: dict[str, Any], selector: str, message: str
) -> None:
    problem_doc["parameters"]["other"] = problem_doc["parameters"]["radius"] | {
        "targets": {"geometry": selector}
    }
    with pytest.raises(ProblemConfigError, match=r"parameters\.other\.targets") as info:
        load_problem(write_problem(tmp_path, problem_doc))
    assert message in str(info.value)
    causes = []
    cause: BaseException | None = info.value
    while cause is not None:
        causes.append(cause)
        cause = cause.__cause__
    assert any(isinstance(cause, GeometryError) for cause in causes)


def test_a_malformed_target_needs_no_geometry(problem_doc: dict[str, Any]) -> None:
    problem_doc["parameters"]["radius"]["targets"] = {"geometry": "not a [reference"}
    with pytest.raises(ProblemConfigError, match=r"parameters\.radius\.targets\.geometry: invalid"):
        parse_problem(problem_doc)


def test_ambiguous_target_is_not_guessed(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    problem_doc["parameters"]["radius"]["targets"] = {"geometry": "Tracker_cooling_gap"}
    with pytest.raises(ProblemConfigError) as info:
        load_problem(write_problem(tmp_path, problem_doc))
    message = str(info.value)
    assert "parameters.radius.targets.geometry" in message
    assert "'constants.Tracker_cooling_gap'" in message
    assert "cannot currently be selected unambiguously" in message
    assert "compact/det_v1/tracker.xml" in message
    assert "compact/common/shared_constants.xml" in message


def test_missing_k4geo(
    tmp_path: Path, problem_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("K4GEO")
    with pytest.raises(SourceError, match=r"\$K4GEO is not set"):
        load_problem(write_problem(tmp_path, problem_doc))


def test_missing_geometry_file(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    problem_doc["geometry"] = "compact/det_v1/nope.xml"
    with pytest.raises(SourceError, match="not found"):
        load_problem(write_problem(tmp_path, problem_doc))


def test_broken_include(
    tmp_path: Path, problem_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "k4geo"
    shutil.copytree(MINI_GEO, tree)
    (tree / "compact/det_v1/tracker.xml").unlink()
    monkeypatch.setenv("K4GEO", str(tree))
    with pytest.raises(IncludeError, match=r"tracker\.xml"):
        load_problem(write_problem(tmp_path, problem_doc))


def test_parse_problem_needs_no_geometry(
    problem_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("K4GEO")
    problem_doc["geometry"] = "does/not/exist.xml"
    assert parse_problem(problem_doc).geometry == "does/not/exist.xml"


def test_a_loaded_problem_is_the_parsed_one(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    loaded = load_problem(write_problem(tmp_path, problem_doc))
    parsed = parse_problem(problem_doc, source=tmp_path / "problem.yaml")
    assert loaded == parsed
    assert hash(loaded) == hash(parsed)
    assert loaded.to_dict() == parsed.to_dict()
    assert parse_problem(loaded.to_dict()) == loaded


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_loading_is_read_only_and_runs_no_tools(
    tmp_path: Path, problem_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected process: {args}")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    before = _tree_hash(MINI_GEO)
    path = write_problem(tmp_path, problem_doc)
    load_problem(path)
    assert _tree_hash(MINI_GEO) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["problem.yaml"]  # no snapshot
