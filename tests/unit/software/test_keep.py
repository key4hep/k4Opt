"""What a study keeps privately for a rerun: bytes by their digest, checked on every read, within
its caps; the state of local builds, the steering a job directory changes, the inputs no one else
identifies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from k4opt.software import Software, SoftwareError, keep
from k4opt.software.keep import Store, commitment, keep_inputs, keep_used, take
from k4opt.software.stack import Release
from tests.support.checkouts import build, checkout, variables
from tests.support.software import lcg_view


def one(inputs: list[dict[str, Any]], path: str) -> dict[str, Any]:
    """The one kept input read at ``path``."""
    (entry,) = (entry for entry in inputs if entry["path"] == path)
    return entry


def test_bytes_kept_come_back_checked(tmp_path: Path) -> None:
    store = Store(tmp_path)
    sha256 = store.put(b"a patch")
    assert sha256 == hashlib.sha256(b"a patch").hexdigest()
    assert store.get(sha256) == b"a patch" and store.state(sha256) == "available"
    assert store.put(b"a patch") == sha256  # once
    assert store.state("0" * 64) == "gone"


def test_bytes_altered_on_disk_are_never_given_back(tmp_path: Path) -> None:
    store = Store(tmp_path)
    sha256 = store.put(b"original")
    import gzip

    store.path(sha256).write_bytes(gzip.compress(b"tampered"))
    assert store.state(sha256) == "altered"
    with pytest.raises(SoftwareError, match="altered"):
        store.get(sha256)
    store.path(sha256).write_bytes(b"not gzip")
    assert store.state(sha256) == "altered"


def test_a_study_keeps_no_more_than_its_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(keep, "MAX_KEPT", 64)
    store = Store(tmp_path)
    store.put(b"small")
    with pytest.raises(SoftwareError, match="at most"):
        store.put(bytes(range(256)) * 4)


def test_a_file_is_kept_only_as_the_stage_read_it_and_not_too_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "scale.json"
    file.write_text('{"scale": 1.05}')
    store = Store(tmp_path / "kept")
    with pytest.raises(SoftwareError, match="no longer the file"):
        store.put_file(file, "0" * 64)
    monkeypatch.setattr(keep, "MAX_INPUT", 4)
    with pytest.raises(SoftwareError, match="larger than"):
        store.put_file(file)


def identity_of(job: Path, label: str = "samples.pi0.reconstruction") -> dict[str, Any]:
    """A session identity whose reconstruction runs from the job directory ``job``."""
    inputs = {}
    for path in sorted(p for p in job.rglob("*") if p.is_file()):
        relative = path.relative_to(job).as_posix()
        inputs[f"{label}.directory/{relative}"] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"inputs": inputs, "implementations": {}}


def test_a_session_keeps_the_release_its_local_builds_and_its_changed_steering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = lcg_view(tmp_path / "stack")
    monkeypatch.setenv("KEY4HEP_STACK", str(view))
    release = Release.of(str(view))
    assert release is not None
    repo = checkout(tmp_path)
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 1.05 * eCalToEMGeV * hit;\n")
    prefix = build(repo)  # built after the edit
    # a copy of the nightly's CLDConfig: one file changed, one added, one removed
    original = Path(release.packages["cldconfig"].prefix) / "share" / "CLDConfig"
    job = shutil.copytree(original, tmp_path / "mine" / "CLDConfig")
    (job / "ParticleFlow" / "Pandora.py").write_text("calibration = 1.1\n")
    (job / "mine.py").write_text("extra = True\n")
    (job / "event_display.py").unlink()
    software = Software.now({**variables(prefix), "KEY4HEP_STACK": str(view)}, site=[])
    directory = tmp_path / "study" / "software"
    taken = take(directory, software, identity_of(job))
    document = json.loads(taken.path.read_text())
    store = Store(directory)
    manifests = document["release"]["manifests"]
    assert store.get(manifests["LCG_externals"]) == release.manifests["LCG_externals"]
    built = next(o for o in document["origins"] if o["kind"] == "build")
    assert (built["package"], built["link"], built["rebuildable"]) == (
        "ddmarlinpandora",
        "verified",
        True,
    )
    assert b"1.05 * eCalToEMGeV" in store.get(built["git"]["patch"])
    assert built["binaries"][0]["build_id"] == "9854a49d"
    (steering,) = document["steering"]
    assert (steering["stage"], steering["package"], steering["revision"]) == (
        "pi0/reconstruction",
        "cldconfig",
        "8879e4e",
    )
    assert steering["unchanged"] == 1 and steering["removed"] == ["event_display.py"]
    assert (
        store.get(steering["changed"]["ParticleFlow/Pandora.py"]["kept"]) == b"calibration = 1.1\n"
    )
    assert store.get(steering["added"]["mine.py"]["kept"]) == b"extra = True\n"
    # no binary is kept: only bytes the store wrote are blobs, and none is the library
    library = (prefix / "lib" / "libDDMarlinPandora.so.0.14.0").read_bytes()
    assert not store.path(hashlib.sha256(library).hexdigest()).exists()
    # the same state again adds nothing
    again = take(directory, software, identity_of(job))
    assert again.path == taken.path
    assert len(list(directory.glob("software-*.json"))) == 1


def test_the_stacks_own_job_directory_keeps_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = lcg_view(tmp_path / "stack")
    monkeypatch.setenv("KEY4HEP_STACK", str(view))
    release = Release.of(str(view))
    assert release is not None
    job = Path(release.packages["cldconfig"].prefix) / "share" / "CLDConfig"
    taken = take(tmp_path / "kept", Software.now({}, site=[]), identity_of(job))
    (steering,) = json.loads(taken.path.read_text())["steering"]
    assert steering["original"] == steering["directory"]
    assert steering["changed"] == {} and steering["added"] == {} and steering["unchanged"] == 3


def test_an_input_is_kept_with_a_salted_commitment_and_the_variable_that_names_it(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    scale = private / "ecal_scale.json"
    scale.write_text('{"scale": 1.05}')
    sha256 = hashlib.sha256(scale.read_bytes()).hexdigest()
    directory = tmp_path / "kept"
    problems = keep_inputs(
        directory,
        [(str(scale), f"sha256:{sha256}", ["pi0/reconstruction"], "observed")],
        {"MY_ECAL_SCALE_FILE": str(scale), "HOME": str(tmp_path / "home")},
    )
    assert problems == []
    entry = one(keep.kept_inputs(directory), str(scale))
    assert entry["variable"] == "MY_ECAL_SCALE_FILE" and entry["bytes"] == 15
    assert entry["commitment"] == commitment(entry["salt"], sha256) != sha256
    assert Store(directory).get(entry["kept"]) == scale.read_bytes()
    # read again by another stage: one entry, both stages, the same salt
    keep_inputs(directory, [(str(scale), f"sha256:{sha256}", ["pi0/analysis"], "observed")])
    again = one(keep.kept_inputs(directory), str(scale))
    assert again["stages"] == ["pi0/analysis", "pi0/reconstruction"]
    assert again["salt"] == entry["salt"]


def test_an_input_changed_since_its_stage_read_it_is_not_kept(tmp_path: Path) -> None:
    file = tmp_path / "calibration.json"
    file.write_text("v2")
    then = hashlib.sha256(b"v1").hexdigest()
    (problem,) = keep_inputs(
        tmp_path / "kept", [(str(file), f"sha256:{then}", ["s/r"], "observed")]
    )
    assert "no longer the file" in problem
    entry = one(keep.kept_inputs(tmp_path / "kept"), str(file))
    assert entry["kept"] is None and "no longer" in entry["why"]


def test_an_input_on_cvmfs_is_identified_not_copied(tmp_path: Path) -> None:
    path = "/cvmfs/other.cern.ch/data/map.root"
    assert keep_inputs(tmp_path, [(path, "cvmfs:abcd", ["s/r"], "observed")]) == []
    entry = one(keep.kept_inputs(tmp_path), path)
    assert entry["kept"] is None and entry["commitment"] is None


def test_what_local_builds_a_stage_read_is_noted_once_per_footprint(tmp_path: Path) -> None:
    import gzip

    from k4opt.software.observe import Footprint

    run = tmp_path / "trial-000000"
    stage = run / "pi0" / "reconstruction"
    stage.mkdir(parents=True)
    footprint = Footprint(
        "files",
        {
            "/home/me/install/lib/libX.so.1": "sha256:aa",
            "/cvmfs/sft.cern.ch/lib/libY.so": "cvmfs:b",
        },
        ran=("/usr/bin/python3",),
    )
    data = gzip.compress(json.dumps(footprint.to_dict()).encode())
    (stage / "footprint.json.gz").write_bytes(data)
    sha256 = hashlib.sha256(data).hexdigest()
    found = {"pi0/reconstruction": ("pi0/reconstruction/footprint.json.gz", sha256)}
    keep_used(tmp_path / "kept", run, found, ["/home/me/install/lib"])
    used = json.loads((tmp_path / "kept" / keep.USED).read_text())["footprints"]
    assert used == {
        sha256: {
            "stage": "pi0/reconstruction",
            "files": {"/home/me/install/lib/libX.so.1": "sha256:aa"},
        }
    }


def test_a_listed_directory_is_no_input_and_home_names_no_file(tmp_path: Path) -> None:
    rootrc = tmp_path / ".rootrc"
    rootrc.write_text("Unix.*.Root.MacroPath: .\n")
    sha256 = hashlib.sha256(rootrc.read_bytes()).hexdigest()
    found = [
        (str(tmp_path), "names:" + "0" * 64, ["s/r"], "observed"),
        (str(rootrc), f"sha256:{sha256}", ["s/r"], "observed"),
    ]
    keep_inputs(tmp_path / "kept", found, {"HOME": str(tmp_path), "PYTHONPATH": "/x"})
    (entry,) = keep.kept_inputs(tmp_path / "kept")
    assert entry["path"] == str(rootrc) and entry["variable"] is None


def test_what_a_study_keeps_only_its_owner_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import stat as stats

    monkeypatch.setenv("KEY4HEP_STACK", str(lcg_view(tmp_path / "stack")))
    old = os.umask(0o022)
    try:
        directory = tmp_path / "study" / "software"
        directory.mkdir(parents=True, mode=0o755)  # as an older k4opt, or a person, made it
        scale = tmp_path / "scale.json"
        scale.write_text("1.05")
        digest = hashlib.sha256(b"1.05").hexdigest()
        take(directory, Software.now({}, site=[]), {"inputs": {}, "implementations": {}})
        keep_inputs(directory, [(str(scale), f"sha256:{digest}", ["s/r"], "observed")])
    finally:
        os.umask(old)
    for path in [directory, *directory.rglob("*")]:
        mode = stats.S_IMODE(path.stat().st_mode)
        assert mode == (0o700 if path.is_dir() else 0o600), (path, oct(mode))


def test_a_path_read_as_another_file_is_another_input(tmp_path: Path) -> None:
    file = tmp_path / "calibration.json"
    kept = tmp_path / "kept"
    for version, trial in ((b"A", "trial-000000"), (b"B", "trial-000001")):
        file.write_bytes(version)
        digest = hashlib.sha256(version).hexdigest()
        found = [(str(file), f"sha256:{digest}", ["pi0/reconstruction"], "observed")]
        keep_inputs(kept, found, trial=trial)
    entries = keep.kept_inputs(kept)
    assert [(e["trials"], Store(kept).get(e["kept"])) for e in entries] == [
        (["trial-000000"], b"A"),
        (["trial-000001"], b"B"),
    ]


def test_an_evaluator_module_that_cannot_be_kept_is_listed_as_not_kept(tmp_path: Path) -> None:
    evaluator = tmp_path / "ev.py"
    evaluator.write_text("from helper import f\n")
    sha256 = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    identity = {
        "inputs": {},
        "implementations": {
            "e": {
                "source": {
                    "path": str(evaluator),
                    "sha256": sha256,
                    "helpers": {"helper.py": "0" * 64},
                }
            }
        },
    }
    taken = take(tmp_path / "kept", Software.now({}, site=[]), identity)
    files = json.loads(taken.path.read_text())["evaluators"]["e"]["files"]
    assert files["ev.py"]["kept"] == sha256 and files["helper.py"]["kept"] is None
    assert any("helper.py not found" in problem for problem in taken.problems)


def test_a_file_two_stages_declare_is_kept_for_both(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text("1.0")
    entry = {"path": str(calibration), "sha256": hashlib.sha256(b"1.0").hexdigest()}
    identity = {
        "inputs": {
            "samples.pi0.simulation.steering": entry,
            "samples.pi0.reconstruction.calibration": entry,
        },
        "implementations": {},
    }
    take(tmp_path / "kept", Software.now({}, site=[]), identity, {})
    (kept,) = keep.kept_inputs(tmp_path / "kept")
    assert kept["stages"] == ["pi0/reconstruction", "pi0/simulation"]


def test_steering_keeps_what_its_bytes_do_not_say(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    script = job / "run.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    shared = job / "shared.py"
    shared.write_text("x = 1\n")
    (job / "linked.py").symlink_to("shared.py")
    inputs = {
        f"samples.pi0.reconstruction.directory/{path.name}": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in job.iterdir()
    }
    taken = take(tmp_path / "kept", Software.now({}, site=[]), {"inputs": inputs}, {})
    (steering,) = json.loads(taken.path.read_text())["steering"]
    added = steering["added"]
    assert added["run.sh"]["executable"] is True and added["run.sh"]["link"] is None
    assert added["shared.py"]["executable"] is False
    assert added["linked.py"]["link"] == "shared.py"


def test_listing_steering_ends_on_links_back_up_the_tree(tmp_path: Path) -> None:
    root = tmp_path / "steering"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a" / "one.py").write_text("1")
    (root / "b" / "two.py").write_text("2")
    os.symlink("..", root / "a" / "up")
    os.symlink(".", root / "a" / "self")
    os.symlink("../b", root / "a" / "to_b")  # and b back to a: a cycle no prefix test sees
    os.symlink("../a", root / "b" / "to_a")
    found = keep._listing(str(root))
    assert {"a/one.py", "b/two.py", "a/to_b/two.py", "b/to_a/one.py"} <= found.keys()
    assert len(found) < 12


@pytest.mark.parametrize(
    "text",
    ["{", '{"schema": "other", "inputs": []}', '{"schema": "k4opt.inputs/1", "inputs": [1]}'],
)
def test_a_damaged_index_of_inputs_is_refused(tmp_path: Path, text: str) -> None:
    (tmp_path / keep.INPUTS).write_text(text)
    with pytest.raises(SoftwareError):
        keep.kept_inputs(tmp_path)
