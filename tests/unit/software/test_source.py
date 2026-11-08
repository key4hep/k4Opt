"""Where local changes come from: each local build linked to its checkout, the checkout's complete
state against the revision the nightly built, and nothing written into the checkout."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from k4opt.software import Software, source
from k4opt.software.source import build_id, checkout, origins
from k4opt.software.stack import Release
from tests.support.checkouts import build, elf, git, variables
from tests.support.checkouts import checkout as make_checkout
from tests.support.software import lcg_view

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def software(**values: str) -> Software:
    return Software.now(values, site=[])


def release(tmp_path: Path) -> Release:
    found = Release.of(str(lcg_view(tmp_path / "stack")))
    assert found is not None
    return found


def untouched(repo: Path) -> tuple[bytes, int]:
    """What a look must not change in ``repo``: its index and how many objects it holds."""
    index = (repo / ".git" / "index").read_bytes()
    objects = sum(len(files) for _, _, files in os.walk(repo / ".git" / "objects"))
    return hashlib.sha256(index).digest(), objects


def test_the_build_id_is_the_one_the_linker_wrote(tmp_path: Path) -> None:
    library = tmp_path / "libX.so"
    library.write_bytes(elf("0123456789abcdef", b"payload"))
    assert build_id(library) == "0123456789abcdef"
    library.write_bytes(b"#!/bin/sh\n")
    assert build_id(library) is None
    assert build_id(tmp_path / "none") is None


@pytest.mark.skipif(shutil.which("readelf") is None, reason="needs readelf")
def test_the_build_id_of_a_real_library_is_what_readelf_says() -> None:
    target = os.path.realpath(shutil.which("true") or "/bin/true")
    notes = subprocess.run(["readelf", "-n", target], capture_output=True, text=True).stdout
    said = [line.split(":")[1].strip() for line in notes.splitlines() if "Build ID" in line]
    assert build_id(target) == (said[0] if said else None)


def test_a_local_build_is_linked_to_its_checkout_and_diffed_against_the_nightly(
    tmp_path: Path,
) -> None:
    repo = make_checkout(tmp_path)
    nightly = git(repo, "rev-parse", "--short=7", "HEAD")
    # the nightly built this commit; the user committed once since, then edited and added a file
    (repo / "README.md").write_text("DDMarlinPandora, patched\n")
    git(repo, "commit", "-qam", "local")
    source_file = repo / "src" / "DDCaloHitCreator.cc"
    source_file.write_text("energy = 1.05 * eCalToEMGeV * hit;\n")
    (repo / "NOTES.txt").write_text("x1.05\n")
    (repo / "debug.log").write_text("ignored\n")
    prefix = build(repo)
    (prefix / "lib" / "libDDMarlinPandora.so.0.14.0").touch()  # built after the edit
    stack = release(tmp_path)
    built = replace(stack.packages["ddmarlinpandora"], revision=nightly)
    stack = replace(stack, packages={**stack.packages, "ddmarlinpandora": built})
    before = untouched(repo)
    (found,) = origins(software(**variables(prefix)).local, stack)
    assert untouched(repo) == before  # nothing written into the checkout
    assert (found.kind, found.prefix, found.package) == ("build", str(prefix), "ddmarlinpandora")
    assert found.link == "verified" and found.build == str(repo / "build")
    assert [b.file for b in found.binaries] == ["lib/libDDMarlinPandora.so.0.14.0"]
    assert found.binaries[0].build_id == "9854a49d"
    assert found.stale is False and found.rebuildable
    state = found.git
    assert state is not None
    assert (state.base_is, state.base[:7] if state.base else None) == ("nightly", nightly)
    assert state.ahead == 1 and state.untracked == 1 and not state.fetch
    assert state.uncommitted == {"files": 2, "insertions": 2, "deletions": 1}
    # the installation, the build tree and what git ignores are not the source
    assert sorted(state.changed) == ["NOTES.txt", "README.md", "src/DDCaloHitCreator.cc"]
    assert state.remote is not None and state.remote.endswith("DDMarlinPandora.git")
    # the patch rebuilds the state from the nightly's revision, in a fresh clone
    fresh = tmp_path / "fresh"
    subprocess.run(["git", "clone", "-q", state.remote, str(fresh)], check=True)
    git(fresh, "checkout", "-q", nightly)
    assert state.patch is not None
    subprocess.run(["git", "-C", str(fresh), "apply", "-"], input=state.patch, check=True)
    assert (fresh / "src" / "DDCaloHitCreator.cc").read_text() == source_file.read_text()
    assert (fresh / "NOTES.txt").read_text() == "x1.05\n"
    assert not (fresh / "install").exists() and not (fresh / "debug.log").exists()


def test_a_source_edited_after_the_build_is_stale(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    library = prefix / "lib" / "libDDMarlinPandora.so.0.14.0"
    os.utime(library, ns=(1_000_000_000, 1_000_000_000))
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 2 * hit;\n")
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.stale is True
    # what is kept is not what the binary was built from: never rebuildable
    assert found.link == "verified" and not found.rebuildable


def test_a_source_file_removed_after_the_build_is_stale(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    os.utime(prefix / "lib" / "libDDMarlinPandora.so.0.14.0", ns=(2_000_000_000, 2_000_000_000))
    shutil.rmtree(repo / "src")  # its directory went too: the checkout's own time says when
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.git is not None and "src/DDCaloHitCreator.cc" in found.git.changed
    assert found.stale is True and not found.rebuildable


def test_a_source_file_removed_before_the_build_is_not_stale(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    os.utime(prefix / "lib" / "libDDMarlinPandora.so.0.14.0", ns=(2_000_000_000, 2_000_000_000))
    (repo / "README.md").unlink()
    os.utime(repo, ns=(1_000_000_000, 1_000_000_000))
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.git is not None and found.git.changed == ("README.md",)
    assert found.stale is False and found.rebuildable


@pytest.mark.parametrize("failing", ["--name-only", "--binary"])
def test_a_diff_that_fails_is_no_clean_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 2 * hit;\n")
    real = source._git

    def git_failing(cwd: str, *args: str, **options: Any) -> bytes | None:
        return None if "diff" in args and failing in args else real(cwd, *args, **options)

    monkeypatch.setattr(source, "_git", git_failing)
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.git is None and not found.rebuildable
    assert found.problem is not None and "cannot be diffed" in found.problem


def test_the_remote_recorded_is_one_that_holds_the_base(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    url = git(repo, "remote", "get-url", "origin")
    git(repo, "remote", "rename", "origin", "upstream")
    git(repo, "remote", "add", "origin", "/nowhere/else.git")  # has no ref of it
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.git is not None and found.git.base_is == "public"
    assert found.git.remote == url and found.rebuildable


def test_a_base_no_known_remote_can_give_is_not_rebuildable(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    git(repo, "remote", "remove", "origin")
    git(repo, "update-ref", "refs/remotes/gone/master", "HEAD")  # a ref left of a remote now gone
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.git is not None and found.git.base_is == "public" and found.git.remote is None
    assert not found.rebuildable


def test_a_build_keeps_what_it_installed_beside_its_binaries(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    (prefix / "python").mkdir()
    (prefix / "python" / "steer.py").write_text("edited = True\n")
    values = {**variables(prefix), "PYTHONPATH": str(prefix / "python")}
    (found,) = origins(software(**values).local, None)
    path, sha256 = found.files["python/steer.py"]
    assert path == str(prefix / "python" / "steer.py")
    assert sha256 == hashlib.sha256(b"edited = True\n").hexdigest()
    assert not any(name.startswith("lib/libDDMarlinPandora") for name in found.files)


def test_binaries_the_build_tree_no_longer_has_make_the_link_likely(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo, build_id="aaaa", installed="bbbb")  # rebuilt since it was installed
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.link == "likely"
    assert "differ from those installed" in found.evidence[-1]
    assert found.package is None and found.git is not None and found.git.base_is == "public"


def test_an_installation_in_a_checkout_without_a_build_tree_is_adjacent(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    shutil.rmtree(repo / "build")
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.link == "adjacent" and found.build is None and not found.rebuildable


def test_the_nightly_revision_missing_from_the_checkout_needs_a_fetch(tmp_path: Path) -> None:
    repo = make_checkout(tmp_path)
    prefix = build(repo)
    (found,) = origins(software(**variables(prefix)).local, release(tmp_path))  # 821384f
    assert found.git is not None and found.git.fetch and found.git.base_is == "public"


def test_local_code_in_no_checkout_is_its_files(tmp_path: Path) -> None:
    helpers = tmp_path / "myhelpers"
    helpers.mkdir()
    (helpers / "ecal_tweak.py").write_text("def scale(): return 1.05\n")
    (helpers / "notes.txt").write_text("not importable\n")
    (found,) = origins(software(PYTHONPATH=str(helpers)).local, None)
    assert found.kind == "directory" and found.git is None
    assert list(found.files) == ["ecal_tweak.py"]


def test_a_virtual_environment_is_its_distributions(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    packages = venv / "lib" / "python3.14" / "site-packages"
    (packages / "numpy-2.5.4.dist-info").mkdir(parents=True)
    (packages / "numpy").mkdir()
    (packages / "numpy" / "__init__.py").write_text("")
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (found,) = origins(software(PYTHONPATH=str(packages)).local, None)
    assert found.kind == "environment" and found.distributions == ("numpy==2.5.4",)


def test_a_checkout_with_no_remote_is_kept_whole(tmp_path: Path) -> None:
    repo = tmp_path / "mine"
    repo.mkdir()
    (repo / "helper.py").write_text("X = 1\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "only")
    found = checkout(str(repo), None)
    assert found is not None and found.git is not None
    assert found.git.base is None and found.git.base_is == "none"
    assert found.git.changed == ("helper.py",)


def test_a_large_untracked_file_is_listed_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source, "LARGE", 10)
    repo = make_checkout(tmp_path)
    (repo / "big.root").write_bytes(b"x" * 100)
    found = checkout(str(repo), None)
    assert found is not None and found.git is not None
    assert found.git.large == ("big.root",) and found.git.changed == ()


def test_a_checkout_no_remote_has_is_rebuildable_from_its_whole_tree(tmp_path: Path) -> None:
    repo = tmp_path / "Mine"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.cc").write_text("int a = 1;\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "local only")
    prefix = build(repo)
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.git is not None and found.git.base_is == "none" and found.git.base is None
    assert "src/a.cc" in found.git.changed and found.rebuildable


def test_a_large_untracked_file_makes_a_build_not_rebuildable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source, "LARGE", 10)
    repo = make_checkout(tmp_path)
    (repo / "table.bin").write_bytes(b"x" * 100)  # the only change, not kept
    prefix = build(repo)
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.git is not None and found.git.clean and found.git.large == ("table.bin",)
    assert not found.rebuildable


def test_a_local_binary_of_no_build_is_identified_never_listed_to_keep(tmp_path: Path) -> None:
    helpers = tmp_path / "plugins"
    helpers.mkdir()
    (helpers / "libPatch.so").write_bytes(elf("abcd", b"code"))
    (helpers / "patch.py").write_text("X = 1\n")
    (found,) = origins(software(PYTHONPATH=str(helpers), LD_LIBRARY_PATH=str(helpers)).local, None)
    assert found.kind == "directory" and list(found.files) == ["patch.py"]
    assert [(b.file, b.build_id) for b in found.binaries] == [("libPatch.so", "abcd")]


def test_a_build_keeps_the_cmake_cache_it_was_configured_with(tmp_path: Path) -> None:
    prefix = build(make_checkout(tmp_path))
    (found,) = origins(software(**variables(prefix)).local, None)
    assert found.configuration == str(tmp_path / "DDMarlinPandora" / "build" / "CMakeCache.txt")


@pytest.mark.parametrize(
    ("url", "public"),
    [
        ("https://user:token@gitlab.cern.ch/a/b.git", "https://gitlab.cern.ch/a/b.git"),
        ("https://token@github.com/a/b.git", "https://github.com/a/b.git"),
        ("ssh://git@github.com/a/b.git", "ssh://git@github.com/a/b.git"),
        ("ssh://git:secret@host:7999/a/b.git", "ssh://git@host:7999/a/b.git"),
        ("git@github.com:a/b.git", "git@github.com:a/b.git"),
        ("/home/me/repo", "/home/me/repo"),
    ],
)
def test_a_remote_is_recorded_without_secrets_but_with_its_ssh_user(url: str, public: str) -> None:
    assert source.public_url(url) == public
