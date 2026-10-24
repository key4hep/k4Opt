"""Tests for geometry path resolution (``$K4GEO``, absolute paths, explicit roots)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from k4opt.geometry import GeometrySource, IncludeError, SourceError
from k4opt.geometry.source import git_toplevel
from tests.support.geometry import MINI_COMPACT, MINI_GEO

REAL_MINI_GEO = Path(os.path.realpath(MINI_GEO))


def test_relative_geometry_resolves_against_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    source = GeometrySource(MINI_COMPACT)
    assert source.root == REAL_MINI_GEO
    assert source.compact == MINI_COMPACT
    assert source.compact_path.is_file()


def test_relative_geometry_without_k4geo_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K4GEO", raising=False)
    with pytest.raises(SourceError, match=r"\$K4GEO is not set"):
        GeometrySource(MINI_COMPACT)


def test_k4geo_must_be_a_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("K4GEO", str(tmp_path / "missing"))
    with pytest.raises(SourceError, match="not an existing directory"):
        GeometrySource(MINI_COMPACT)


def test_environment_variables_are_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K4GEO", raising=False)
    monkeypatch.setenv("MINIGEO", str(MINI_GEO))
    source = GeometrySource("$MINIGEO/compact/det_v1/det_v1.xml")
    # the fixtures live inside this repository, so the git checkout is the detected root
    expected_root = git_toplevel(REAL_MINI_GEO) or REAL_MINI_GEO / "compact" / "det_v1"
    assert source.root == expected_root
    assert source.compact == (REAL_MINI_GEO / MINI_COMPACT).relative_to(source.root).as_posix()
    assert source.compact_path == REAL_MINI_GEO / MINI_COMPACT


def test_unset_environment_variable_is_an_error() -> None:
    with pytest.raises(SourceError, match="not set"):
        GeometrySource("$K4OPT_SURELY_UNSET/x.xml")


def test_absolute_path_inside_k4geo_uses_k4geo_as_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    source = GeometrySource(str(MINI_GEO / MINI_COMPACT))
    assert source.root == REAL_MINI_GEO
    assert source.compact == MINI_COMPACT


def test_absolute_path_outside_git_and_k4geo_falls_back_to_its_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("K4GEO", raising=False)
    geo = tmp_path / "geo"
    shutil.copytree(MINI_GEO, geo)
    if git_toplevel(geo) is not None:
        pytest.skip("temporary directory unexpectedly lies inside a git checkout")
    source = GeometrySource(str(geo / MINI_COMPACT))
    assert source.root == Path(os.path.realpath(geo / "compact" / "det_v1"))
    assert source.compact == "det_v1.xml"
    with pytest.raises(IncludeError, match="set 'root'"):
        source.discover()
    explicit = GeometrySource(str(geo / MINI_COMPACT), root=geo)
    assert explicit.compact == MINI_COMPACT
    assert len(explicit.discover()) > 0


def test_explicit_root_validation(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not an existing directory"):
        GeometrySource("x.xml", root=tmp_path / "missing")
    with pytest.raises(SourceError, match="not found"):
        GeometrySource("compact/det_v1/missing.xml", root=MINI_GEO)
    with pytest.raises(SourceError, match="escapes"):
        GeometrySource("../compact/det_v1/det_v1.xml", root=MINI_GEO / "compact")
    with pytest.raises(SourceError, match="does not lie inside"):
        GeometrySource(str(MINI_GEO / MINI_COMPACT), root=MINI_GEO / "compact" / "det_v2")
    with pytest.raises(SourceError, match="not found"):
        GeometrySource(str(tmp_path / "nope.xml"))
    with pytest.raises(SourceError, match="empty"):
        GeometrySource("  ", root=MINI_GEO)


def test_paths_are_normalised_and_sources_compare_equal() -> None:
    a = GeometrySource("compact/det_v1/../det_v1/./det_v1.xml", root=MINI_GEO)
    b = GeometrySource(str(MINI_GEO / MINI_COMPACT), root=MINI_GEO)
    assert a.compact == MINI_COMPACT
    assert a == b and hash(a) == hash(b)
    assert "det_v1.xml" in repr(a)
    assert a != GeometrySource("compact/det_v2/det_v2.xml", root=MINI_GEO)
