"""Loading a problem resolves its geometry once; parse_problem stays pure."""

from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.geometry import GeometryError, GeometrySource, IncludeError, SourceError
from k4opt.problem import ProblemConfigError, ProblemSpec, load_problem, parse_problem
from tests.support.geometry import MINI_COMPACT, MINI_GEO

pytestmark = pytest.mark.usefixtures("mini_k4geo")


@pytest.fixture
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def problem_doc(doc: dict[str, Any]) -> dict[str, Any]:
    doc["geometry"] = MINI_COMPACT
    doc["parameters"][0]["geometry"] = {"constant": "InnerTracker_Barrel_radius_1"}
    return doc


def write_problem(tmp_path: Path, doc: dict[str, Any]) -> Path:
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def targets(problem: ProblemSpec) -> tuple[Any, ...]:
    return problem._targets


def test_load_resolves_a_constant(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    problem = load_problem(write_problem(tmp_path, problem_doc))
    (target,) = targets(problem)
    assert target.target == "constants.InnerTracker_Barrel_radius_1"
    assert target.value == "400*mm"  # raw text, neither evaluated nor converted
    assert target.file == "compact/det_v1/tracker.xml"
    assert target.line is not None
    assert problem._source == GeometrySource(MINI_COMPACT)


def test_load_resolves_a_structured_target(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    ref = "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"
    problem_doc["parameters"][0] = {
        "geometry": {"target": ref},
        "type": "integer",
        "bounds": [10, 100],
    }
    (target,) = targets(load_problem(write_problem(tmp_path, problem_doc)))
    assert (target.target, target.attribute, target.value) == (ref, "nphi", "43*2")


def test_parameters_resolve_in_order_with_one_discovery(
    tmp_path: Path, problem_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem_doc["parameters"].append(
        {"geometry": {"constant": "world_side"}, "type": "continuous", "bounds": [1.0, 2.0]}
    )
    calls: list[GeometrySource] = []
    discover = GeometrySource.discover

    def counting(self: GeometrySource) -> Any:
        calls.append(self)
        return discover(self)

    monkeypatch.setattr(GeometrySource, "discover", counting)
    problem = load_problem(write_problem(tmp_path, problem_doc))
    assert len(calls) == 1
    assert [t.target for t in targets(problem)] == [
        "constants.InnerTracker_Barrel_radius_1",
        "constants.world_side",
    ]
    assert problem._catalog is not None
    assert problem._catalog.root == problem._source.root  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ({"constant": "InnerTracker_Barrel_radius_9"}, "unknown target"),
        ({"target": "constants.InnerTracker_Barrel_radius_9"}, "unknown target"),
        ({"target": "not a [reference"}, "not a [reference"),
    ],
)
def test_missing_or_malformed_target(
    tmp_path: Path, problem_doc: dict[str, Any], selector: dict[str, str], message: str
) -> None:
    problem_doc["parameters"].insert(0, copy.deepcopy(problem_doc["parameters"][0]))
    problem_doc["parameters"][1]["geometry"] = selector
    with pytest.raises(ProblemConfigError, match=r"parameters\[1\]\.geometry") as info:
        load_problem(write_problem(tmp_path, problem_doc))
    assert message in str(info.value)
    assert isinstance(info.value.__cause__, GeometryError)


def test_ambiguous_target_is_not_guessed(tmp_path: Path, problem_doc: dict[str, Any]) -> None:
    problem_doc["parameters"][0]["geometry"] = {"constant": "Tracker_cooling_gap"}
    with pytest.raises(ProblemConfigError) as info:
        load_problem(write_problem(tmp_path, problem_doc))
    message = str(info.value)
    assert "parameters[0].geometry" in message
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
    problem = parse_problem(problem_doc)
    assert (problem._source, problem._catalog, problem._targets) == (None, None, ())


def test_runtime_state_is_not_part_of_the_problem(
    tmp_path: Path, problem_doc: dict[str, Any]
) -> None:
    loaded = load_problem(write_problem(tmp_path, problem_doc))
    parsed = parse_problem(problem_doc, source=tmp_path / "problem.yaml")
    assert loaded == parsed
    assert hash(loaded) == hash(parsed)
    assert loaded.to_dict() == parsed.to_dict()
    assert parse_problem(loaded.to_dict()) == loaded
    assert "_catalog" not in repr(loaded)


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
