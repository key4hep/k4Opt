"""What a rerun of a study needs and what of it is available: the status follows from the evidence,
and never says more than it shows. A local build that ran can be rebuilt from its kept source, never
restored: its original binary is a fingerprint only."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from k4opt.software import Software, keep, source
from k4opt.software.keep import DIRECTORY, INPUTS, USED, Store, keep_inputs, keep_used, take
from k4opt.software.observe import identity
from k4opt.software.readiness import (
    ALL_AVAILABLE,
    BLOCKED,
    INCOMPLETE,
    REBUILD,
    UNVERIFIED,
    Requirement,
    kept_directory,
    records,
    requirements,
)
from tests.support.checkouts import build, checkout, elf, git, variables
from tests.support.software import lcg_view, republish

IDENTITY: dict[str, Any] = {
    "inputs": {},
    "implementations": {},
    "k4opt": {"version": "0.1.0", "sha256": "f" * 64, "git": {"commit": "a" * 40, "dirty": False}},
}


@pytest.fixture(autouse=True)
def public_k4opt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """k4opt's own checkout, as the studies of these tests keep it: clean, at a commit a remote
    has, whatever the checkout the tests run from is."""
    repo = checkout(tmp_path / "k4opt", "k4Opt")
    IDENTITY["k4opt"]["git"] = {"commit": git(repo, "rev-parse", "HEAD"), "dirty": False}
    monkeypatch.setattr(keep, "checkout", lambda path, release: source.checkout(str(repo), release))


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
    readiness = requirements(tmp_path / "study", IDENTITY, [summary()])
    assert readiness.status == f"{ALL_AVAILABLE} on {readiness.at[:10]}"
    base = readiness.requirements[0]
    assert base.kind == "base" and base.available and base.used_by == ("pi0/reconstruction",)
    assert base.ran == {"original": "on-cvmfs"}
    assert "Reproduced" not in readiness.status


def test_a_rotated_slot_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    study_on(tmp_path / "study", stack)
    republish(stack)
    readiness = requirements(tmp_path / "study", IDENTITY, [summary()])
    assert readiness.status == BLOCKED
    (check,) = readiness.requirements[0].checks
    assert check.state == "gone" and "another build" in check.how


def test_a_local_build_that_ran_needs_a_rebuild_and_its_binary_is_a_fingerprint(
    tmp_path: Path, stack: Path
) -> None:
    repo = checkout(tmp_path)
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 1.05 * hit;\n")
    prefix = build(repo)  # built after the edit
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
    found = {"pi0/reconstruction": ("footprint.json.gz", hashlib.sha256(data).hexdigest())}
    keep_used(study / DIRECTORY, run, found, [entry.path for entry in software.local])
    readiness = requirements(study, IDENTITY, [summary(local=[str(prefix / "lib")])])
    assert readiness.status == REBUILD
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.label == "ddmarlinpandora" and package.used_by == ("pi0/reconstruction",)
    assert package.source is not None and package.source["rebuildable"] is True
    assert package.ran is not None and package.ran["original"] == "fingerprint-only"
    assert [f["file"] for f in package.ran["files"]] == ["lib/libDDMarlinPandora.so.0.14.0"]
    assert ALL_AVAILABLE not in readiness.status


def test_a_kept_patch_altered_on_disk_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    repo = checkout(tmp_path)
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 1.05 * hit;\n")
    prefix = build(repo)  # built after the edit
    study = tmp_path / "study"
    study_on(study, stack, variables(prefix))
    (document,) = (json.loads(p.read_text()) for p in (study / DIRECTORY).glob("software-*.json"))
    patch = next(o for o in document["origins"] if o["kind"] == "build")["git"]["patch"]
    Store(study / DIRECTORY).path(patch).write_bytes(gzip.compress(b"another patch"))
    readiness = requirements(study, IDENTITY, [summary(local=[str(prefix / "lib")])])
    assert readiness.status == BLOCKED
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert "altered" in [c.state for c in package.checks]


def test_a_local_build_no_stage_read_did_not_run(tmp_path: Path, stack: Path) -> None:
    prefix = build(checkout(tmp_path))
    study = tmp_path / "study"
    study_on(study, stack, variables(prefix))
    readiness = requirements(study, IDENTITY, [summary()])
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.used_by == () and any("not used" in note for note in package.notes)
    assert readiness.status.startswith(ALL_AVAILABLE)  # nothing of it ran
    # unless that is not known: then nothing is said of it
    unknown = requirements(study, IDENTITY, [summary(level="none")])
    assert unknown.status == INCOMPLETE
    assert not any(
        "not used" in note for r in unknown.requirements if r.kind == "package" for note in r.notes
    )


def test_what_is_not_known_makes_the_record_incomplete(tmp_path: Path, stack: Path) -> None:
    study_on(tmp_path / "study", stack)
    assert requirements(tmp_path / "study", IDENTITY, []).status == INCOMPLETE
    nothing = requirements(tmp_path / "other", IDENTITY, [summary()])
    assert nothing.status == INCOMPLETE and nothing.requirements == ()
    assert "kept nothing" in nothing.gaps[0]


def test_uncommitted_k4opt_not_kept_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    study_on(tmp_path / "study", stack)
    dirty = {**IDENTITY, "k4opt": {**IDENTITY["k4opt"], "git": {"commit": "a" * 40, "dirty": True}}}
    readiness = requirements(tmp_path / "study", dirty, [summary()])
    k4opt = next(r for r in readiness.requirements if r.kind == "k4opt")
    assert k4opt.checks[0].state in ("gone", "available")  # kept only if its checkout was


def test_an_inner_study_reads_what_the_top_study_kept(tmp_path: Path) -> None:
    top = tmp_path / "study"
    inner = top / "inner" / "trial-000003"
    inner.mkdir(parents=True)
    (top / "study.json").write_text("{}")
    assert kept_directory(inner) == top / DIRECTORY
    assert kept_directory(top) == top / DIRECTORY


def test_a_source_changed_after_its_build_is_not_known_to_be_what_ran(
    tmp_path: Path, stack: Path
) -> None:
    repo = checkout(tmp_path)
    prefix = build(repo)
    library = prefix / "lib" / "libDDMarlinPandora.so.0.14.0"
    os.utime(library, ns=(1_000_000_000, 1_000_000_000))  # built long ago
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 2 * hit;\n")  # edited since
    study = tmp_path / "study"
    study_on(study, stack, variables(prefix))
    readiness = requirements(study, IDENTITY, [summary(local=[str(prefix / "lib")])])
    assert readiness.status == INCOMPLETE  # neither "rebuild needed" nor "blocked"
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.source is not None
    assert package.source["stale"] is True and package.source["rebuildable"] is False
    assert any("not known to be what they were built from" in gap for gap in readiness.gaps)


def test_a_source_that_changed_between_sessions_is_not_known_to_be_what_ran(
    tmp_path: Path, stack: Path
) -> None:
    repo = checkout(tmp_path)
    prefix = build(repo)
    study = tmp_path / "study"
    study_on(study, stack, variables(prefix))
    notes = repo / "NOTES.txt"
    notes.write_text("edited before the resume\n")
    os.utime(notes, ns=(1_000_000_000, 1_000_000_000))  # older than the build: not stale by time
    study_on(study, stack, variables(prefix))  # a second session: another state
    assert len(list((study / DIRECTORY).glob("software-*.json"))) == 2
    readiness = requirements(study, IDENTITY, [summary(local=[str(prefix / "lib")])])
    assert readiness.status == INCOMPLETE
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.source is not None and package.source["stale"] is True
    assert any("between sessions" in note for note in package.notes)


def test_what_could_not_be_checked_is_not_said_to_be_available(tmp_path: Path, stack: Path) -> None:
    venv = tmp_path / "venv"
    packages = venv / "lib" / "python3.14" / "site-packages"
    (packages / "numpy-2.5.4.dist-info").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    study_on(tmp_path / "study", stack, {"PYTHONPATH": str(packages)})
    readiness = requirements(tmp_path / "study", IDENTITY, [summary()])
    assert readiness.status == UNVERIFIED
    environment = next(r for r in readiness.requirements if r.kind == "environment")
    assert not environment.available
    assert not Requirement("x", "code", "x", "x").available  # no check: not known


def test_a_rebuild_and_what_could_not_be_checked_are_both_said(tmp_path: Path, stack: Path) -> None:
    repo = checkout(tmp_path)
    prefix = build(repo)
    venv = tmp_path / "venv" / "lib" / "python3.14" / "site-packages"
    venv.mkdir(parents=True)
    (tmp_path / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    study = tmp_path / "study"
    study_on(study, stack, {**variables(prefix), "PYTHONPATH": str(venv)})
    readiness = requirements(study, IDENTITY, [summary(local=[str(prefix / "lib")])])
    assert readiness.status == f"{REBUILD} · availability not established"


def test_the_records_of_every_level_of_a_nested_study_count(tmp_path: Path, stack: Path) -> None:
    prefix = build(checkout(tmp_path))
    top = tmp_path / "study"
    study_on(top, stack, variables(prefix))
    (top / "study.json").write_text("{}")
    inner = top / "inner" / "trial-000000"
    (inner / "trial-000001").mkdir(parents=True)
    (inner / "study.json").write_text("{}")
    used = summary(local=[str(prefix / "lib")])
    (inner / "trial-000001" / "summary.json").write_text(json.dumps(used))
    (top / "trial-000000").mkdir()
    (top / "trial-000000" / "summary.json").write_text(json.dumps(summary()))
    (snapshot,) = (top / DIRECTORY).glob("software-*.json")
    for trial in ("trial-000000", "inner/trial-000000/trial-000001"):
        keep_used(top / DIRECTORY, top, {}, [], trial, snapshot.name)
    assert [trial for trial, _ in records(top)] == [
        "trial-000000",
        "inner/trial-000000/trial-000001",
    ]
    readiness = requirements(inner, IDENTITY)  # an inner study's view is the whole tree's
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.trials == ("inner/trial-000000/trial-000001",)
    assert readiness.status == REBUILD


def test_a_stage_that_ran_without_saying_what_it_used_leaves_a_gap(
    tmp_path: Path, stack: Path
) -> None:
    study_on(tmp_path / "study", stack)
    record = summary()
    del record["samples"]["pi0"]["stages"]["reconstruction"]["observed"]
    readiness = requirements(tmp_path / "study", IDENTITY, [record])
    assert readiness.status == INCOMPLETE
    assert "does not say what it used" in readiness.gaps[0]


def test_a_reused_stage_is_what_the_stage_that_made_it_used(tmp_path: Path, stack: Path) -> None:
    prefix = build(checkout(tmp_path))
    study_on(tmp_path / "study", stack, variables(prefix))
    other = tmp_path / "other-study" / "trial-000004"
    other.mkdir(parents=True)
    (other / "summary.json").write_text(json.dumps(summary(local=[str(prefix / "lib")])))
    reused = {"reused": {"run_dir": str(other), "task": "pi0/reconstruction"}, "status": "done"}
    record = {"samples": {"pi0": {"stages": {"reconstruction": reused}}}}
    readiness = requirements(tmp_path / "study", IDENTITY, [record])
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.used_by == ("pi0/reconstruction",) and readiness.status == REBUILD
    (other / "summary.json").unlink()  # its record gone: what it used is not known
    gone = requirements(tmp_path / "study", IDENTITY, [record])
    assert gone.status == INCOMPLETE and "whose record does not say" in gone.gaps[0]


def test_a_used_build_with_untracked_files_not_kept_is_incomplete(
    tmp_path: Path, stack: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source, "LARGE", 10)
    repo = checkout(tmp_path)
    (repo / "table.bin").write_bytes(b"x" * 100)
    prefix = build(repo)
    study_on(tmp_path / "study", stack, variables(prefix))
    readiness = requirements(tmp_path / "study", IDENTITY, [summary(local=[str(prefix / "lib")])])
    assert readiness.status == INCOMPLETE
    assert any("too large to keep" in gap for gap in readiness.gaps)


def test_an_evaluator_module_not_kept_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    evaluator = tmp_path / "ev.py"
    evaluator.write_text("from helper import f\n")
    sha256 = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    implementation = {"version": "1", "source": {"path": str(evaluator), "sha256": sha256}}
    implementation["source"]["helpers"] = {"helper.py": "0" * 64}  # type: ignore[index]
    identity = {**IDENTITY, "implementations": {"e": implementation}}
    software = Software.now({"KEY4HEP_STACK": str(stack)}, site=[])
    take(tmp_path / "study" / DIRECTORY, software, identity)
    readiness = requirements(tmp_path / "study", identity, [summary()])
    assert readiness.status == BLOCKED
    evaluator_requirement = next(r for r in readiness.requirements if r.kind == "evaluator")
    assert [c.state for c in evaluator_requirement.checks] == ["available", "gone"]


def test_a_kept_record_altered_on_disk_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    study_on(tmp_path / "study", stack)
    (kept,) = (tmp_path / "study" / DIRECTORY).glob("software-*.json")
    kept.write_text(kept.read_text().replace('"lcg"', '"lcg" '))
    readiness = requirements(tmp_path / "study", IDENTITY, [summary()])
    assert readiness.status == BLOCKED and readiness.altered == (kept.name,)


def test_an_input_whose_record_names_other_bytes_is_altered(tmp_path: Path, stack: Path) -> None:
    study_on(tmp_path / "study", stack)
    file = tmp_path / "scale.json"
    file.write_text("1.05")
    digest = hashlib.sha256(b"1.05").hexdigest()
    directory = tmp_path / "study" / DIRECTORY
    keep_inputs(directory, [(str(file), f"sha256:{digest}", ["pi0/reconstruction"], "observed")])
    other = Store(directory).put(b"2.0")
    document = json.loads((directory / INPUTS).read_text())
    document["inputs"][0]["kept"] = other  # a valid blob, of other bytes
    (directory / INPUTS).write_text(json.dumps(document))
    readiness = requirements(tmp_path / "study", IDENTITY, [summary()])
    assert readiness.status == BLOCKED


def test_only_the_states_the_trials_ran_with_count(tmp_path: Path, stack: Path) -> None:
    repo = checkout(tmp_path)
    prefix = build(repo)
    study = tmp_path / "study"
    software = Software.now({**variables(prefix), "KEY4HEP_STACK": str(stack)}, site=[])
    first = take(study / DIRECTORY, software, IDENTITY)
    notes = repo / "NOTES.txt"
    notes.write_text("edited, but no trial ran after\n")
    os.utime(notes, ns=(1_000_000_000, 1_000_000_000))
    take(study / DIRECTORY, software, IDENTITY)
    keep_used(study / DIRECTORY, study, {}, [], "record 0", first.path.name)
    readiness = requirements(study, IDENTITY, [summary(local=[str(prefix / "lib")])])
    assert readiness.status == REBUILD  # the second state ran nothing: not "changed between"


def test_a_library_counts_as_run_where_it_was_read_not_by_its_bytes(
    tmp_path: Path, stack: Path
) -> None:
    from k4opt.hashing import sha256_of
    from k4opt.software.observe import Footprint

    first = build(checkout(tmp_path / "a"))
    second = build(checkout(tmp_path / "b"))  # the same bytes, another installation
    study = tmp_path / "study"
    values = variables(first)
    values["LD_LIBRARY_PATH"] = f"{first / 'lib'}:{second / 'lib'}"
    software = study_on(study, stack, values)
    library = first / "lib" / "libDDMarlinPandora.so.0.14.0"
    run = tmp_path / "run"
    run.mkdir()
    footprint = Footprint("files", {str(library): f"sha256:{sha256_of(library)}"}, ran=("k4run",))
    data = gzip.compress(json.dumps(footprint.to_dict()).encode())
    (run / "footprint.json.gz").write_bytes(data)
    import hashlib

    found = {"pi0/reconstruction": ("footprint.json.gz", hashlib.sha256(data).hexdigest())}
    keep_used(study / DIRECTORY, run, found, [e.path for e in software.local])
    readiness = requirements(study, IDENTITY, [summary(local=[str(first / "lib")])])
    used = {r.label: r.used_by for r in readiness.requirements if r.kind == "package"}
    assert list(used.values()).count(()) == 1  # the other installation did not run


def in_trial(study: Path, name: str, record: dict[str, Any]) -> None:
    (study / name).mkdir(parents=True)
    (study / name / "summary.json").write_text(json.dumps(record))


def test_every_installation_trials_ran_from_is_required(tmp_path: Path, stack: Path) -> None:
    first, second = (build(checkout(tmp_path / name)) for name in "ab")
    study = tmp_path / "study"
    records = []
    for number, prefix in enumerate((first, second)):
        software = Software.now({**variables(prefix), "KEY4HEP_STACK": str(stack)}, site=[])
        taken = take(study / DIRECTORY, software, IDENTITY)
        keep_used(study / DIRECTORY, study, {}, [], f"record {number}", taken.path.name)
        records.append(summary(local=[str(prefix / "lib")]))
    readiness = requirements(study, IDENTITY, records)
    packages = [r for r in readiness.requirements if r.kind == "package"]
    assert len({r.id for r in packages}) == 2
    assert sorted(r.trials for r in packages) == [("record 0",), ("record 1",)]
    assert {str(first), str(second)} == {r.label.partition(" at ")[2] for r in packages}


def test_a_state_a_trial_ran_with_gone_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    repo = checkout(tmp_path)
    prefix = build(repo)
    study = tmp_path / "study"
    software = Software.now({**variables(prefix), "KEY4HEP_STACK": str(stack)}, site=[])
    first = take(study / DIRECTORY, software, IDENTITY)
    keep_used(study / DIRECTORY, study, {}, [], "record 0", first.path.name)
    (repo / "NOTES.txt").write_text("another state\n")
    second = take(study / DIRECTORY, software, IDENTITY)
    keep_used(study / DIRECTORY, study, {}, [], "record 1", second.path.name)
    first.path.unlink()
    record = summary(local=[str(prefix / "lib")])
    readiness = requirements(study, IDENTITY, [record, record])
    assert readiness.status == BLOCKED and readiness.altered == (first.path.name,)


def test_evidence_gone_is_not_evidence_of_nothing(tmp_path: Path, stack: Path) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    in_trial(study, "trial-000000", summary())
    (snapshot,) = (study / DIRECTORY).glob("software-*.json")
    readiness = requirements(study, IDENTITY)  # no used.json at all
    assert readiness.status == INCOMPLETE and any(USED in gap for gap in readiness.gaps)
    keep_used(study / DIRECTORY, study, {}, [], "trial-000000", snapshot.name)
    assert requirements(study, IDENTITY).gaps == ()  # noted, and empty: nothing is missing
    in_trial(study, "trial-000001", summary())  # its record was never noted
    readiness = requirements(study, IDENTITY)
    assert readiness.status == INCOMPLETE and any("trial-000001" in g for g in readiness.gaps)


def test_a_footprint_the_study_did_not_note_leaves_a_gap(tmp_path: Path, stack: Path) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    (snapshot,) = (study / DIRECTORY).glob("software-*.json")
    keep_used(study / DIRECTORY, study, {}, [], "trial-000000", snapshot.name)
    record = summary()
    stage = record["samples"]["pi0"]["stages"]["reconstruction"]
    stage["logs"] = {"footprint": "footprint.json.gz"}
    stage["observed"]["footprint"] = {"sha256": "a" * 64}
    in_trial(study, "trial-000000", record)
    readiness = requirements(study, IDENTITY)
    assert readiness.status == INCOMPLETE
    assert readiness.gaps == ("what pi0/reconstruction read of local builds was not noted",)


@pytest.mark.parametrize("name", [USED, INPUTS])
@pytest.mark.parametrize("text", ["{", '{"schema": "other"}'])
def test_a_damaged_index_is_said_and_the_rest_is_shown(
    tmp_path: Path, stack: Path, name: str, text: str
) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    (study / DIRECTORY / name).write_text(text)
    readiness = requirements(study, IDENTITY, [summary()])
    assert readiness.status == BLOCKED and readiness.altered == (name,)
    assert readiness.requirements[0].kind == "base"


@pytest.mark.parametrize("field", ["salt", "identity"])
def test_a_commitment_that_is_no_hex_is_an_altered_input(
    tmp_path: Path, stack: Path, field: str
) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    file = tmp_path / "scale.json"
    file.write_text("1.05")
    digest = hashlib.sha256(b"1.05").hexdigest()
    directory = study / DIRECTORY
    keep_inputs(directory, [(str(file), f"sha256:{digest}", ["pi0/reconstruction"], "observed")])
    document = json.loads((directory / INPUTS).read_text())
    document["inputs"][0][field] = "sha256:zz" if field == "identity" else "zz"
    (directory / INPUTS).write_text(json.dumps(document))
    readiness = requirements(study, IDENTITY, [summary()])
    assert readiness.status == BLOCKED
    (kept,) = (r for r in readiness.requirements if r.kind == "input")
    assert kept.checks[0].state == "altered"


def test_a_pipe_still_there_vouches_for_nothing(tmp_path: Path, stack: Path) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    pipe = tmp_path / "calibration"
    os.mkfifo(pipe)
    keep_inputs(study / DIRECTORY, [(str(pipe), "special", ["pi0/reconstruction"], "observed")])
    readiness = requirements(study, IDENTITY, [summary()])
    assert readiness.status == UNVERIFIED
    (kept,) = (r for r in readiness.requirements if r.kind == "input")
    assert kept.checks[0].state == "not-checked"


def test_a_stage_that_used_the_network_leaves_a_gap(tmp_path: Path, stack: Path) -> None:
    prefix = build(checkout(tmp_path))
    study = tmp_path / "study"
    study_on(study, stack, variables(prefix))
    record = summary()
    record["samples"]["pi0"]["stages"]["reconstruction"]["observed"]["network"] = ["1.2.3.4:443"]
    readiness = requirements(study, IDENTITY, [record])
    assert readiness.status == INCOMPLETE
    assert "pi0/reconstruction reached the network (1.2.3.4:443)" in readiness.gaps[0]
    # what it read from files is still known: a local build it did not read did not run
    assert next(r for r in readiness.requirements if r.kind == "package").unused


def test_a_file_read_that_could_not_be_identified_leaves_a_gap(tmp_path: Path, stack: Path) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    record = summary()
    record["samples"]["pi0"]["stages"]["reconstruction"]["observed"]["undeclared"] = {"/x/y": None}
    readiness = requirements(study, IDENTITY, [record])
    assert readiness.status == INCOMPLETE and "/x/y" in readiness.gaps[0]


def lost_patch(tmp_path: Path, stack: Path) -> tuple[Path, Path]:
    """A study on a local build whose kept patch is gone; its prefix."""
    repo = checkout(tmp_path)
    prefix = build(repo)
    notes = repo / "NOTES.txt"
    notes.write_text("a local change\n")
    os.utime(notes, ns=(1_000_000_000, 1_000_000_000))
    study = tmp_path / "study"
    study_on(study, stack, variables(prefix))
    (snapshot,) = (study / DIRECTORY).glob("software-*.json")
    document = json.loads(snapshot.read_text())
    (origin,) = (o for o in document["origins"] if o["kind"] == "build")
    Store(study / DIRECTORY).path(origin["git"]["patch"]).unlink()
    return study, prefix


def test_a_build_that_did_not_run_blocks_nothing(tmp_path: Path, stack: Path) -> None:
    study, _ = lost_patch(tmp_path, stack)
    readiness = requirements(study, IDENTITY, [summary()])
    assert readiness.status == f"{ALL_AVAILABLE} on {readiness.at[:10]}"
    package = next(r for r in readiness.requirements if r.kind == "package")
    assert package.unused and package.missing


def test_a_build_that_ran_is_blocked_by_its_lost_patch(tmp_path: Path, stack: Path) -> None:
    study, prefix = lost_patch(tmp_path, stack)
    readiness = requirements(study, IDENTITY, [summary(local=[str(prefix / "lib")])])
    assert readiness.status == BLOCKED


def test_a_build_not_known_to_be_unused_still_blocks(tmp_path: Path, stack: Path) -> None:
    study, _ = lost_patch(tmp_path, stack)
    readiness = requirements(study, IDENTITY, [summary(level="none")])
    assert readiness.status == BLOCKED


def test_an_installed_module_edited_by_hand_is_kept_as_it_ran(tmp_path: Path, stack: Path) -> None:
    prefix = build(checkout(tmp_path))
    (prefix / "python").mkdir()
    (prefix / "python" / "steer.py").write_text("edited = True\n")
    study = tmp_path / "study"
    values = {**variables(prefix), "PYTHONPATH": str(prefix / "python")}
    study_on(study, stack, values)
    (snapshot,) = (study / DIRECTORY).glob("software-*.json")
    (origin,) = (o for o in json.loads(snapshot.read_text())["origins"] if o["kind"] == "build")
    kept = origin["files"]["python/steer.py"]["kept"]
    assert Store(study / DIRECTORY).get(kept) == b"edited = True\n"
    Store(study / DIRECTORY).path(kept).unlink()
    readiness = requirements(study, IDENTITY, [summary(local=[str(prefix / "python")])])
    assert readiness.status == BLOCKED


def test_local_code_whose_base_no_known_remote_gives_is_not_established(
    tmp_path: Path, stack: Path
) -> None:
    repo = checkout(tmp_path)
    git(repo, "remote", "remove", "origin")
    git(repo, "update-ref", "refs/remotes/gone/master", "HEAD")
    code = repo / "src"
    study = tmp_path / "study"
    study_on(study, stack, {"PYTHONPATH": str(code)})
    readiness = requirements(study, IDENTITY, [summary(local=[str(code)])])
    assert readiness.status == UNVERIFIED


def test_an_input_the_records_name_and_the_index_lacks_leaves_a_gap(
    tmp_path: Path, stack: Path
) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    file = tmp_path / "scale.json"
    file.write_text("1")
    then = f"sha256:{hashlib.sha256(b'1').hexdigest()}"
    keep_inputs(study / DIRECTORY, [(str(file), then, ["pi0/reconstruction"], "observed")])
    (study / DIRECTORY / INPUTS).unlink()
    readiness = requirements(study, IDENTITY, [summary_with(undeclared={str(file): then})])
    assert readiness.status == INCOMPLETE and str(file) in readiness.gaps[0]


def test_a_declared_input_the_index_lacks_leaves_a_gap(tmp_path: Path, stack: Path) -> None:
    file = tmp_path / "scale.json"
    file.write_text("1")
    entry = {"path": str(file), "sha256": hashlib.sha256(b"1").hexdigest()}
    identity = {**IDENTITY, "inputs": {"samples.pi0.simulation.steering": entry}}
    study = tmp_path / "study"
    software = Software.now({"KEY4HEP_STACK": str(stack)}, site=[])
    take(study / DIRECTORY, software, identity)
    assert not requirements(study, identity, [summary()]).gaps
    (study / DIRECTORY / INPUTS).unlink()
    assert str(file) in requirements(study, identity, [summary()]).gaps[0]


def summary_with(**observed: object) -> dict[str, Any]:
    record = summary()
    record["samples"]["pi0"]["stages"]["reconstruction"]["observed"].update(observed)
    return record


def test_an_output_reused_from_another_study_says_whose_build_made_it(
    tmp_path: Path, stack: Path
) -> None:
    theirs, ours = (build(checkout(tmp_path / name)) for name in "ab")
    other = tmp_path / "other"
    study_on(other, stack, variables(theirs))
    in_trial(other, "trial-000000", summary(local=[str(theirs / "lib")]))
    study = tmp_path / "study"
    study_on(study, stack, variables(ours))
    run = str(other / "trial-000000")
    stage = {"reused": {"run_dir": run, "task": "pi0/reconstruction"}}
    record = {"samples": {"pi0": {"stages": {"reconstruction": stage}}}}
    readiness = requirements(study, IDENTITY, [record])
    assert readiness.status == INCOMPLETE
    assert str(theirs / "lib") in readiness.gaps[0] and run in readiness.gaps[0]
    assert not next(r for r in readiness.requirements if r.kind == "package").unused


def test_a_listed_directory_is_said_only_once_it_changed(tmp_path: Path, stack: Path) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    listed = tmp_path / "calibrations"
    listed.mkdir()
    record = summary_with(undeclared={str(listed): identity(str(listed))})
    assert not any(
        r.id.startswith("listing:") for r in requirements(study, IDENTITY, [record]).requirements
    )
    (listed / "new.json").write_text("{}")
    readiness = requirements(study, IDENTITY, [record])
    assert readiness.status == UNVERIFIED  # not gone: any Python program lists its own path
    assert readiness.requirements[-2].id == f"listing:{listed}"


def test_a_footprint_record_of_another_shape_is_said(tmp_path: Path, stack: Path) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    document = {
        "schema": "k4opt.used/1",
        "trials": {},
        "footprints": {"abc": {"stage": "pi0/reconstruction", "files": ["/x"]}},
    }
    (study / DIRECTORY / USED).write_text(json.dumps(document))
    readiness = requirements(study, IDENTITY, [summary()])
    assert readiness.status == BLOCKED and readiness.altered == (USED,)


def test_a_local_geometry_is_required_as_it_was_kept(tmp_path: Path, stack: Path) -> None:
    root = tmp_path / "mine"
    shutil.copytree(Path(__file__).parents[2] / "fixtures" / "geometry" / "mini_geo", root)
    compact = "compact/det_v1/det_v1.xml"
    geometry = {"compact": compact, "source_root": str(root), "externals": {}}
    identity_ = {**IDENTITY, "geometry": geometry}
    study = tmp_path / "study"
    software = Software.now({"KEY4HEP_STACK": str(stack)}, site=[])
    taken = take(study / DIRECTORY, software, identity_)
    shutil.rmtree(root)  # what is kept is all a rerun needs of it
    readiness = requirements(study, identity_, [summary()])
    assert readiness.status == f"{ALL_AVAILABLE} on {readiness.at[:10]}"
    assert next(r for r in readiness.requirements if r.kind == "geometry").label == compact
    kept = json.loads(taken.path.read_text())["geometry"]["files"][compact]["kept"]
    Store(study / DIRECTORY).path(kept).unlink()
    assert requirements(study, identity_, [summary()]).status == BLOCKED


def test_an_ignored_module_that_ran_is_required(tmp_path: Path, stack: Path) -> None:
    repo = checkout(tmp_path)
    (repo / ".gitignore").write_text("generated/\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "ignore")
    git(repo, "push", "-q", "origin", "HEAD")
    (repo / "generated").mkdir()
    (repo / "generated" / "__init__.py").write_text("")
    (repo / "generated" / "calibration.py").write_text("constant = 1\n")
    study = tmp_path / "study"
    taken = take(
        study / DIRECTORY,
        Software.now({"PYTHONPATH": str(repo), "KEY4HEP_STACK": str(stack)}, site=[]),
        IDENTITY,
    )
    (origin,) = (
        o for o in json.loads(taken.path.read_text())["origins"] if o["prefix"] == str(repo)
    )
    kept = origin["files"]["generated/calibration.py"]["kept"]
    assert Store(study / DIRECTORY).get(kept) == b"constant = 1\n"
    Store(study / DIRECTORY).path(kept).unlink()
    readiness = requirements(study, IDENTITY, [summary(local=[str(repo)])])
    assert readiness.status == BLOCKED


def test_a_compiled_input_gone_blocks_a_rerun(tmp_path: Path, stack: Path) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    helper = tmp_path / "helper"
    helper.write_bytes(elf("abcd"))
    then = f"sha256:{hashlib.sha256(helper.read_bytes()).hexdigest()}"
    keep_inputs(study / DIRECTORY, [(str(helper), then, ["pi0/reconstruction"], "observed")])
    assert requirements(study, IDENTITY, [summary()]).status.startswith(ALL_AVAILABLE)
    helper.unlink()
    assert requirements(study, IDENTITY, [summary()]).status == BLOCKED


def test_a_candidates_geometry_file_the_index_lacks_leaves_a_gap(
    tmp_path: Path, stack: Path
) -> None:
    study = tmp_path / "study"
    study_on(study, stack)
    root = tmp_path / "mine"
    record = summary()
    record["geometry"] = {
        "status": "done",
        "source_root": str(root),
        "candidate_files": {"alternative.xml": "a" * 64},
    }
    readiness = requirements(study, IDENTITY, [record])
    assert readiness.status == INCOMPLETE and str(root / "alternative.xml") in readiness.gaps[0]
