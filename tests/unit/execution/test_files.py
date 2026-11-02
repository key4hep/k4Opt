"""The files a task made, watched for changes until its candidate ends (no Key4hep)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from k4opt.execution._files import Made, Observed, Tree, changed_files
from k4opt.problem.plan import GEOMETRY, evaluator_task, stage_task

SIMULATION = stage_task("simulation", "gamma")


# --- the files a task made, and whether they are still what its record says ---


def test_a_made_file_is_named_by_its_task_and_hashed_only_if_asked(tmp_path: Path) -> None:
    path = tmp_path / "events.root"
    path.write_bytes(b"events")
    unhashed = Made.file(SIMULATION, "events", path, hashed=False)
    assert unhashed.artifact.ref == "gamma/simulation#events"
    assert (unhashed.artifact.path, unhashed.artifact.bytes) == (path, 6)
    assert unhashed.artifact.sha256 is None
    hashed = Made.file(evaluator_task("plot", "gamma"), "histogram", path, hashed=True)
    assert hashed.artifact.ref == "gamma/plot#histogram"
    assert hashed.artifact.sha256 is not None and len(hashed.artifact.sha256) == 64
    with pytest.raises(OSError, match="is no file"):
        Made.file(SIMULATION, "events", tmp_path, hashed=False)


def test_a_file_replaced_while_it_is_hashed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.execution import _files

    path = tmp_path / "histogram.json"
    path.write_bytes(b"old")
    identity = _files.file_identity

    def replaced(file: Path) -> dict[str, object]:
        file.write_bytes(b"new, and longer")  # between the stat and the hash
        return identity(file)

    monkeypatch.setattr(_files, "file_identity", replaced)
    # the size of one version with the hash of another, which changed() would then trust
    with pytest.raises(OSError, match="changed while it was hashed"):
        Made.file(evaluator_task("plot", "gamma"), "histogram", path, hashed=True)


@pytest.mark.parametrize("hashed", [False, True])
def test_a_made_file_is_changed_once_its_stat_or_its_bytes_are(
    tmp_path: Path, hashed: bool
) -> None:
    path = tmp_path / "events.root"
    path.write_bytes(b"events")
    made = Made.file(SIMULATION, "events", path, hashed=hashed)
    assert not made.changed()
    os.utime(path, ns=(1, 1))  # another stat, the same bytes: only a hash tells them apart
    assert made.changed() is not hashed
    # other bytes, of another size: a rewrite of the same size within one tick of the kernel's
    # clock keeps every stat (the limit _sha256 documents)
    path.write_bytes(b"other events")
    assert made.changed()
    path.unlink()
    assert made.changed()


def test_the_changed_files_are_those_of_the_tasks_observed(tmp_path: Path) -> None:
    kept, rewritten = tmp_path / "kept.root", tmp_path / "rewritten.root"
    for path in (kept, rewritten):
        path.write_bytes(b"events")
    pion = stage_task("simulation", "pion")
    observed = {
        SIMULATION: Observed({"events": Made.file(SIMULATION, "events", kept, hashed=False)}),
        GEOMETRY: Observed(),
        pion: Observed({"events": Made.file(pion, "events", rewritten, hashed=False)}),
    }
    rewritten.write_bytes(b"more events")
    assert changed_files(observed) == ["pion/simulation#events"]


def test_a_tree_is_changed_by_any_file_in_it_that_changes_goes_or_is_added(
    tmp_path: Path,
) -> None:
    (tmp_path / "compact").mkdir()
    (tmp_path / "compact" / "a.xml").write_text("<a/>")
    (tmp_path / "b.xml").write_text("<b/>")
    tree = Tree.of(tmp_path)
    assert list(tree.stats) == ["b.xml", "compact/a.xml"] and tree.changed() == []
    (tmp_path / "compact" / "a.xml").write_text("<a>edited</a>")
    (tmp_path / "b.xml").unlink()
    (tmp_path / "c.xml").write_text("<c/>")
    assert tree.changed() == ["b.xml", "compact/a.xml", "c.xml"]
    observed = {GEOMETRY: Observed(tree=tree)}
    assert changed_files(observed) == ["geometry/b.xml", "geometry/compact/a.xml", "geometry/c.xml"]
