"""Atomic result publication: a partial or competing write never replaces a result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from k4opt import jsonio


def test_success_is_complete_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    jsonio.write_json(path, {"result": [1, 2]})
    assert json.loads(path.read_text()) == {"result": [1, 2]}
    assert list(tmp_path.iterdir()) == [path]


def test_the_file_is_staged_beside_its_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AFS and EOS refuse a hard link from another directory
    link = jsonio.os.link
    staged: list[Path] = []

    def recording(source: Path, destination: Path) -> None:
        staged.append(Path(source))
        link(source, destination)

    monkeypatch.setattr(jsonio.os, "link", recording)
    jsonio.write_json(tmp_path / "result.json", {})
    assert [path.parent for path in staged] == [tmp_path]


def test_the_file_has_the_permissions_of_an_ordinary_file(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.write_text("")
    jsonio.write_json(tmp_path / "result.json", {})
    assert (tmp_path / "result.json").stat().st_mode == ordinary.stat().st_mode


def test_failed_serialization_leaves_no_partial_file(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        jsonio.write_json(tmp_path / "result.json", {"partial": 1, "bad": object()})
    assert list(tmp_path.iterdir()) == []


def test_failed_publication_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(jsonio.os, "link", denied)
    with pytest.raises(PermissionError):
        jsonio.write_json(tmp_path / "result.json", {"complete": True})
    assert list(tmp_path.iterdir()) == []


def test_a_concurrently_created_destination_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    link = jsonio.os.link

    def competing(staged: Path, destination: Path) -> None:
        path.write_text("someone else's result")
        link(staged, destination)

    monkeypatch.setattr(jsonio.os, "link", competing)
    with pytest.raises(FileExistsError):
        jsonio.write_json(path, {"ours": True})
    assert path.read_text() == "someone else's result"
    assert list(tmp_path.iterdir()) == [path]
