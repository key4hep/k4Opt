"""A study as a physicist's local work leaves it, for the tests of what is published of it: a
real study, run with a local build and a helper directory in front of the stack, whose stages are
then said to have read them (test studies run no program that could be traced)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from k4opt.software import keep, source
from k4opt.software.keep import DIRECTORY, USED, keep_inputs
from tests.support.checkouts import build, checkout, git, variables
from tests.support.studies import mixed_study

STAGE = "pi0/reconstruction"
SECRET = "ghp_" + "a1B2" * 10


def identity(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def read_by_stage(study: Path, *paths: Path) -> None:
    """Note in the study that a stage read ``paths``, as its footprint would."""
    used = study / DIRECTORY / USED
    document = json.loads(used.read_text())
    files = {str(path): identity(path) for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    document["footprints"][digest] = {"stage": STAGE, "files": files}
    used.write_text(json.dumps(document))


def add_input(study: Path, path: Path) -> None:
    """Note in the study that a stage read the input ``path``, and keep it."""
    keep_inputs(study / DIRECTORY, [(str(path), identity(path), [STAGE], "observed")])


def local_study(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    trials: int = 2,
    *,
    private: bool = False,
    untracked: tuple[str, ...] = (),
    links: dict[str, str] | None = None,
) -> Path:
    """A study in ``root/study`` run on a build of DDMarlinPandora with one edited file, one
    untracked header it installed, one untracked note and one untracked binary, a directory of
    helpers of which a stage read one, and a calibration file that stage read. ``private``: no
    remote has its checkout, so that its patch holds every file; ``untracked``: more files of it;
    ``links``: untracked links in it, by name, to what they lead to."""
    repo = checkout(root / "src")
    clean = checkout(root / "k4opt", "k4Opt")  # k4opt's own, whatever the tests run from
    monkeypatch.setattr(
        keep, "checkout", lambda path, release: source.checkout(str(clean), release)
    )
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 1.05 * eCalToEMGeV * hit;\n")
    (repo / "New.h").write_text("class New;\n")
    (repo / "notes.txt").write_text("not for anyone\n")
    (repo / "blob.dat").write_bytes(b"\0\1\2binary")
    if private:
        git(repo, "remote", "remove", "origin")
    for name in untracked:
        (repo / name).write_text("// compiled in\n")
    for name, target in (links or {}).items():
        (repo / name).symlink_to(target)
    past = time.time() - 60  # all of it before the build: not stale
    extra = [repo / name for name in untracked]
    for path in (
        *repo.rglob("*.cc"),
        repo / "New.h",
        repo / "notes.txt",
        repo / "blob.dat",
        *extra,
    ):
        os.utime(path, (past, past))
    prefix = build(repo)
    (prefix / "include" / "New.h").write_text("class New;\n")
    helpers = root / "helpers"
    helpers.mkdir()
    (helpers / "ecal_tweak.py").write_text("SCALE = 1.05\n")
    (helpers / "unrelated.py").write_text("IDEA = 'unpublished'\n")
    values = variables(prefix)
    values["LD_LIBRARY_PATH"] = f"{prefix / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    values["PYTHONPATH"] = f"{helpers}:{os.environ.get('PYTHONPATH', '')}"
    for name, value in values.items():
        monkeypatch.setenv(name, value.rstrip(":"))
    study = mixed_study(root / "study", trials)
    library = prefix / "lib" / "libDDMarlinPandora.so.0.14.0"
    read_by_stage(study, library, helpers / "ecal_tweak.py")
    calibration = root / "data" / "ecal_scale.json"
    calibration.parent.mkdir()
    calibration.write_text('{"scale": 1.05}\n')
    add_input(study, calibration)
    return study
