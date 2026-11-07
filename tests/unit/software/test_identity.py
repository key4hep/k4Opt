"""What candidates run on: one build whatever place holds it, and every variable compared unless
it is known to be irrelevant."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import pytest

from k4opt.software import IGNORED, Build, Software, SoftwareError, System
from tests.support.software import PLATFORM, lcg_view, manifest, republish


def software(stack: Path | str | None = None, **variables: str) -> Software:
    """What a shell with only ``variables`` and ``KEY4HEP_STACK`` runs on."""
    return Software.now({**variables, **({} if stack is None else {"KEY4HEP_STACK": str(stack)})})


def test_an_lcg_view_is_identified_by_its_time_and_its_manifests(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    build = Build.of(str(view))
    assert (build.kind, build.release, build.platform) == ("lcg", "devkey-head", PLATFORM)
    assert build.built == "Sat Nov  7 04:17:32 2026"
    assert build.label == "LCG devkey-head, built Sat Nov 7 04:17:32 2026"
    assert sorted(build.digests) == ["LCG_contrib", "LCG_externals", "setup"]
    assert all(build.digests.values())


def test_the_same_path_with_another_build_is_another_build(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    before = software(view)
    republish(view)
    after = software(view)
    assert before.id != after.id
    assert before.differences(after) == [
        "the Key4hep build is LCG devkey-head, built Sat Nov 14 04:17:32 2026, not LCG "
        "devkey-head, built Sat Nov 7 04:17:32 2026"
    ]


@pytest.mark.parametrize("name", ["externals", "contrib"])
def test_a_manifest_alone_tells_two_builds_apart(tmp_path: Path, name: str) -> None:
    view = lcg_view(tmp_path)
    before = software(view)
    manifest(view, name).write_text("another build\n")
    (said,) = before.differences(software(view))
    assert said.endswith(f"is another one (other LCG_{name})")


def test_a_manifest_that_went_is_no_longer_the_same_build(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    before = software(view)
    manifest(view).unlink()
    assert Build.of(str(view)).digests["LCG_externals"] is None
    assert before.id != software(view).id


def test_latest_is_the_weekday_it_points_to(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    latest = view.parents[2] / "latest" / PLATFORM / "setup.sh"
    assert latest.is_file()
    assert Build.of(str(latest)) == Build.of(str(view))
    assert software(latest).id == software(view).id


def test_a_build_is_the_same_wherever_it_is(tmp_path: Path) -> None:
    here, there = lcg_view(tmp_path / "a"), lcg_view(tmp_path / "b")
    assert software(here).id == software(there).id


def test_any_other_stack_is_identified_by_the_content_of_its_setup(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    here, there = tmp_path / "a" / "setup.sh", tmp_path / "b" / "setup.sh"
    here.write_text("export A=1\n")
    there.write_text("export A=1\n")
    build = Build.of(str(here))
    assert (build.kind, build.label) == ("unrecognized", "unrecognized stack")
    assert build == Build.of(str(there))
    there.write_text("export A=2\n")
    assert build != Build.of(str(there))


def test_a_view_without_its_time_is_not_taken_for_one(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    view.write_text("export A=1\n")
    assert Build.of(str(view)).kind == "unrecognized"


def test_a_stack_that_is_no_file_is_identified_by_its_text() -> None:
    first, second = Build.of("/stack/2026-10-20"), Build.of("/stack/2026-11-01")
    assert first.kind == "unrecognized" and list(first.digests) == ["text"]
    assert first != second


def test_no_stack_is_an_unrecognized_build_with_nothing_to_identify_it() -> None:
    assert Build.of(None) == Build.of("") == Build("unrecognized", None, None, None, {})


@pytest.mark.parametrize(
    "name",
    [
        "MARLIN_DLL",
        "GAUDI_PLUGIN_PATH",
        "LD_PRELOAD",
        "G4LEDATA",
        "MY_SCALE",
        "HOME",
        "TZ",
        "PATH",  # ROOT runs g++ through it, an evaluator any program
        "XDG_CONFIG_HOME",  # where matplotlib and others read their settings
        "GIT_DIR",
    ],
)
def test_every_variable_is_compared_by_its_digest_known_or_not(name: str) -> None:
    first, second = software(**{name: "one"}), software(**{name: "two"})
    assert first.id != second.id != software().id
    assert first.differences(second) == [f"the environment differs in {name}"]
    assert first.differences(software()) == [f"the environment differs in {name}"]
    assert "one" not in json.dumps(first.to_dict())  # kept as a digest: a value can be private


@pytest.mark.parametrize(
    "name",
    [
        "_",
        "OLDPWD",
        "COLUMNS",
        "SSH_CONNECTION",
        "XDG_RUNTIME_DIR",
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "PYTEST_CURRENT_TEST",
        "CPPYY_API_PATH",
        "CPPYY_NO_ROOT_FILTER",
        "XDG_SESSION_ID",
        "GIT_PAGER",
        "K4GEO",
    ],
)
def test_a_variable_known_to_be_irrelevant_is_not_compared(name: str) -> None:
    assert software(**{name: "one"}).id == software(**{name: "two"}).id == software().id


def test_an_exported_shell_function_is_code_and_compared() -> None:
    # bash -c myhelper prints first in one, second in the other
    first = software(**{"BASH_FUNC_myhelper%%": "() {  printf first\n}"})
    second = software(**{"BASH_FUNC_myhelper%%": "() {  printf second\n}"})
    assert first.differences(second) == ["the environment differs in BASH_FUNC_myhelper%%"]


def test_what_a_listed_prefix_leaves_out_is_only_what_it_starts() -> None:
    assert "SSH_*" in IGNORED
    assert software(SSHFS_OPTIONS="one").id != software(SSHFS_OPTIONS="two").id


def test_a_value_that_is_no_text_is_still_a_value() -> None:
    undecodable = os.fsdecode(b"\xff")
    assert software(MY_SCALE=undecodable).id != software(MY_SCALE="?").id


def test_another_system_is_another_software(monkeypatch: pytest.MonkeyPatch) -> None:
    before = software()
    monkeypatch.setattr(
        "k4opt.software.identity.platform.libc_ver", lambda: ("glibc", "2.99"), raising=True
    )
    (said,) = before.differences(software())
    assert said.startswith("the system is ") and "glibc 2.99" in said


def test_the_system_says_only_what_it_knows(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> dict[str, str]:
        raise OSError("no os-release")

    monkeypatch.setattr("k4opt.software.identity.platform.freedesktop_os_release", missing)
    monkeypatch.setattr("k4opt.software.identity.platform.libc_ver", lambda: ("", ""))
    monkeypatch.setattr("k4opt.software.identity.platform.machine", lambda: "")
    assert System.now() == System(None, None, None)
    assert System.now().label == "unknown"


def test_the_major_version_of_the_system_is_what_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    release = {"ID": "almalinux", "VERSION_ID": "9.8"}
    monkeypatch.setattr("k4opt.software.identity.platform.freedesktop_os_release", lambda: release)
    assert System.now().os == "almalinux 9"


def test_the_same_shell_has_no_differences_and_one_id(tmp_path: Path) -> None:
    view = lcg_view(tmp_path)
    first, second = software(view, A="1"), software(view, A="1")
    assert first.id == second.id and first.differences(second) == []


def test_it_reads_back_from_what_it_writes_and_crosses_to_a_worker(tmp_path: Path) -> None:
    written = software(lcg_view(tmp_path), MY_SCALE="1")
    data = json.loads(json.dumps(written.to_dict()))
    assert Software.from_dict(data) == written
    assert pickle.loads(pickle.dumps(written)) == written


def _tampered(key: str, value: Any, tmp_path: Path) -> dict[str, Any]:
    data = software(lcg_view(tmp_path), MY_SCALE="1").to_dict()
    return {**data, key: value}


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("id", "0" * 64, "is not that of its parts"),
        ("variables", {}, "is not that of its parts"),
        ("variables", {"MY_SCALE": "1"}, "neither digests by name nor one"),
        ("variables", [], "neither digests by name nor one"),
        ("build", None, "the build is not an object"),
        ("build", {"kind": "lcg"}, "the build is not an object"),
        ("system", {}, "the system is not an object"),
    ],
)
def test_what_is_not_software_is_refused(
    tmp_path: Path, key: str, value: Any, message: str
) -> None:
    with pytest.raises(SoftwareError, match=message):
        Software.from_dict(_tampered(key, value, tmp_path))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("kind", "conda", "no kind of build"),
        ("release", 1, "is 1, not text"),
        ("digests", [], "not an object"),
        ("digests", {"setup": 1}, "is 1, not text"),
    ],
)
def test_what_is_not_a_build_is_refused(key: str, value: Any, message: str) -> None:
    data = {**Build.of(None).to_dict(), key: value}
    with pytest.raises(SoftwareError, match=message):
        Build.from_dict(data)
