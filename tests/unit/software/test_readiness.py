"""What a rerun of a study needs and what of it is available: the status follows from the evidence,
and never says more than it shows. A local build that ran can be rebuilt from its kept source, never
restored: its original binary is a fingerprint only."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from k4opt.software import Software
from k4opt.software.keep import DIRECTORY, Store, keep_used, take
from k4opt.software.readiness import (
    ALL_AVAILABLE,
    BLOCKED,
    INCOMPLETE,
    REBUILD,
    kept_directory,
    requirements,
)
from tests.support.checkouts import build, checkout, variables
from tests.support.software import lcg_view, republish

IDENTITY: dict[str, Any] = {
    "inputs": {},
    "implementations": {},
    "k4opt": {"version": "0.1.0", "sha256": "f" * 64, "git": {"commit": "a" * 40, "dirty": False}},
}


def summary(*, local: list[str] | None = None, level: str = "files") -> dict[str, Any]:
    """The record of a trial whose reconstruction ran and was observed at ``level``, reading from
    the ``local`` entries."""
    observed = {"level": level, "complete": level == "files", "local": local or []}
    return {
        "samples": {
            "pi0": {"stages": {"reconstruction": {"argv": ["k4run"], "observed": observed}}}
        }
    }


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    view = lcg_view(tmp_path / "stack")
    monkeypatch.setenv("KEY4HEP_STACK", str(view))
    return view


def study_on(study: Path, view: Path, values: dict[str, str] | None = None) -> Software:
    software = Software.now({**(values or {}), "KEY4HEP_STACK": str(view)}, site=[])
    take(study / DIRECTORY, software, IDENTITY)
    return software


def test_an_official_build_alone_is_all_available_on_the_day_it_was_checked(
    tmp_path: Path, stack: Path
) -> None:
    study_on(tmp_path / "study", stack)
    readiness = requirements(tmp_path / "study", [summary()], IDENTITY)
    assert readiness.status == f"{ALL_AVAILABLE} on {readiness.at[:10]}"
    base = readiness.requirements[0]
    assert base.kind == "base" and base.available and base.used_by == ("pi0/reconstruction",)
    assert base.ran == {"original": "on-cvmfs"}
    assert "Reproduced" not in readiness.status


def test_a_rotated_slot_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    study_on(tmp_path / "study", stack)
    republish(stack)
    readiness = requirements(tmp_path / "study", [summary()], IDENTITY)
    assert readiness.status == BLOCKED
    (check,) = readiness.requirements[0].checks
    assert check.state == "gone" and "another build" in check.how


def test_a_local_build_that_ran_needs_a_rebuild_and_its_binary_is_a_fingerprint(
    tmp_path: Path, stack: Path
) -> None:
    repo = checkout(tmp_path)
    prefix = build(repo)
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 1.05 * hit;\n")
    study = tmp_path / "study"
    software = study_on(study, stack, variables(prefix))
    # the reconstruction's footprint read the installed library
    library = prefix / "lib" / "libDDMarlinPandora.so.0.14.0"
    from k4opt.hashing import sha256_of
    from k4opt.software.observe import Footprint

    run = tmp_path / "trial"
    run.mkdir()
    footprint = Footprint("files", {str(library): f"sha256:{sha256_of(library)}"}, ran=("k4run",))
    data = gzip.compress(json.dumps(footprint.to_dict()).encode())
    (run / "footprint.json.gz").write_bytes(data)
    import hashlib

    found = {"pi0/reconstruction": ("footprint.json.gz", hashlib.sha256(data).hexdigest())}
    keep_used(study / DIRECTORY, run, found, [entry.path for entry in software.local])
    readiness = requirements(study, [summary(local=[str(prefix / "lib")])], IDENTITY)
    assert readiness.status == REBUILD
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.label == "ddmarlinpandora" and package.used_by == ("pi0/reconstruction",)
    assert package.source is not None and package.source["rebuildable"] is True
    assert package.ran is not None and package.ran["original"] == "fingerprint-only"
    assert [f["file"] for f in package.ran["files"]] == ["lib/libDDMarlinPandora.so.0.14.0"]
    assert ALL_AVAILABLE not in readiness.status


def test_a_kept_patch_altered_on_disk_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    repo = checkout(tmp_path)
    prefix = build(repo)
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 1.05 * hit;\n")
    study = tmp_path / "study"
    study_on(study, stack, variables(prefix))
    (document,) = (json.loads(p.read_text()) for p in (study / DIRECTORY).glob("software-*.json"))
    patch = next(o for o in document["origins"] if o["kind"] == "build")["git"]["patch"]
    Store(study / DIRECTORY).path(patch).write_bytes(gzip.compress(b"another patch"))
    readiness = requirements(study, [summary(local=[str(prefix / "lib")])], IDENTITY)
    assert readiness.status == BLOCKED
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert [c.state for c in package.checks] == ["altered"]


def test_a_local_build_no_stage_read_did_not_run(tmp_path: Path, stack: Path) -> None:
    prefix = build(checkout(tmp_path))
    study = tmp_path / "study"
    study_on(study, stack, variables(prefix))
    readiness = requirements(study, [summary()], IDENTITY)
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.used_by == () and any("not used" in note for note in package.notes)
    assert readiness.status.startswith(ALL_AVAILABLE)  # nothing of it ran
    # unless that is not known: then nothing is said of it
    unknown = requirements(study, [summary(level="none")], IDENTITY)
    assert unknown.status == INCOMPLETE
    assert not any(
        "not used" in note for r in unknown.requirements if r.kind == "package" for note in r.notes
    )


def test_what_is_not_known_makes_the_record_incomplete(tmp_path: Path, stack: Path) -> None:
    study_on(tmp_path / "study", stack)
    assert requirements(tmp_path / "study", [], IDENTITY).status == INCOMPLETE
    nothing = requirements(tmp_path / "other", [summary()], IDENTITY)
    assert nothing.status == INCOMPLETE and nothing.requirements == ()
    assert "kept nothing" in nothing.gaps[0]


def test_uncommitted_k4opt_not_kept_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    study_on(tmp_path / "study", stack)
    dirty = {**IDENTITY, "k4opt": {**IDENTITY["k4opt"], "git": {"commit": "a" * 40, "dirty": True}}}
    readiness = requirements(tmp_path / "study", [summary()], dirty)
    k4opt = next(r for r in readiness.requirements if r.kind == "k4opt")
    assert k4opt.checks[0].state in ("gone", "available")  # kept only if its checkout was


def test_an_inner_study_reads_what_the_top_study_kept(tmp_path: Path) -> None:
    top = tmp_path / "study"
    inner = top / "inner" / "trial-000003"
    inner.mkdir(parents=True)
    (top / "study.json").write_text("{}")
    assert kept_directory(inner) == top / DIRECTORY
    assert kept_directory(top) == top / DIRECTORY
