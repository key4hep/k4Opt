"""``k4opt.revision``: git revisions of files and the identity of k4opt's own code."""

from __future__ import annotations

import shutil
from pathlib import Path

import k4opt
from k4opt import geometry
from k4opt.revision import (
    GitInfo,
    code_fingerprint,
    code_identity,
    code_unchanged,
    source_digest,
    tracked_revision,
)
from tests.support.common import git, git_checkout, requires_git, write_tree

PACKAGE = {"pkg/__init__.py": "x = 1\n", "pkg/sub/mod.py": "y = 2\n", "pkg/data.txt": "not code\n"}


def test_the_digest_follows_the_code_not_its_location(tmp_path: Path) -> None:
    first = write_tree(tmp_path / "a", PACKAGE) / "pkg"
    second = write_tree(tmp_path / "elsewhere" / "b", PACKAGE) / "pkg"
    assert source_digest(first) == source_digest(second)
    (second / "data.txt").write_text("other data\n")  # only .py files are code
    (second / "__pycache__").mkdir()
    (second / "__pycache__" / "mod.cpython-313.pyc").write_bytes(b"\0")
    assert source_digest(first) == source_digest(second)


def test_an_edit_a_new_file_or_a_rename_changes_the_digest(tmp_path: Path) -> None:
    package = write_tree(tmp_path, PACKAGE) / "pkg"
    before = source_digest(package)
    (package / "sub" / "mod.py").write_text("y = 3\n")
    edited = source_digest(package)
    (package / "new.py").write_text("")
    added = source_digest(package)
    (package / "sub" / "mod.py").rename(package / "sub" / "other.py")
    renamed = source_digest(package)
    assert len({before, edited, added, renamed}) == 4


def test_the_running_code_is_identified_by_its_files() -> None:
    package = Path(k4opt.__file__).parent
    assert code_fingerprint() == source_digest(package)
    assert code_unchanged()
    identity = code_identity()
    assert identity["sha256"] == code_fingerprint()
    checkout = identity["git"]  # None when k4opt is installed rather than checked out
    assert checkout is None or set(checkout) == {"commit", "dirty"}


def test_the_geometry_component_still_exports_git_info() -> None:
    assert geometry.GitInfo is GitInfo


@requires_git
def test_a_file_tracked_by_a_checkout_has_its_revision(tmp_path: Path) -> None:
    commit = git_checkout(tmp_path, {"study/evaluator.py": "", "study/helper.py": ""})
    revision = tracked_revision(tmp_path / "study" / "evaluator.py")
    assert revision is not None
    assert (revision.commit, revision.dirty) == (commit, False)
    (tmp_path / "study" / "helper.py").write_text("changed = True\n")  # another tracked file
    revision = tracked_revision(tmp_path / "study" / "evaluator.py")
    assert revision is not None and revision.dirty


@requires_git
def test_a_file_no_checkout_tracks_has_no_revision(tmp_path: Path) -> None:
    git_checkout(tmp_path / "repo", {"tracked.py": ""})
    (tmp_path / "repo" / "new.py").write_text("")  # not in any commit of that checkout
    assert tracked_revision(tmp_path / "repo" / "new.py") is None
    git(tmp_path / "repo", "add", "new.py")  # staged: the checkout tracks it, but has changes
    staged = tracked_revision(tmp_path / "repo" / "new.py")
    assert staged is not None and staged.dirty
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copy(tmp_path / "repo" / "tracked.py", outside)
    assert tracked_revision(outside / "tracked.py") is None
