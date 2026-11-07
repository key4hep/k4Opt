"""Local builds, by content: a variable that leads programs to a local build is identified by what
can be loaded through it, never by where the build is."""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from k4opt.software import Software, overlays
from k4opt.software.overlays import SITE, identified, sites


def software(**variables: str) -> Software:
    """What a shell with only ``variables`` runs on, with no site directories."""
    return Software.now(variables, site=[])


def install(prefix: Path, library: bytes = b"ELF v1") -> Path:
    """A local build installed at ``prefix``, as ``k4_local_repo`` lays one out."""
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib" / "libDDMarlinPandora.so").write_bytes(library)
    (prefix / "lib" / "DDMarlinPandora.components").write_text("v1 libDDMarlinPandora.so:Algo\n")
    (prefix / "python" / "ddmarlin").mkdir(parents=True)
    (prefix / "python" / "ddmarlin" / "__init__.py").write_text("SCALE = 1.0\n")
    return prefix


def at(prefix: Path) -> dict[str, str]:
    """The variables that put the build at ``prefix`` in front of the stack."""
    return {
        "LD_LIBRARY_PATH": f"{prefix / 'lib'}:/cvmfs/stack/lib",
        "PYTHONPATH": f"{prefix / 'python'}:/cvmfs/stack/python",
        "MARLIN_DLL": f"{prefix / 'lib' / 'libDDMarlinPandora.so'}",
    }


def test_a_library_rebuilt_in_place_is_other_software(tmp_path: Path) -> None:
    prefix = install(tmp_path / "install")
    before = software(**at(prefix))
    (prefix / "lib" / "libDDMarlinPandora.so").write_bytes(b"ELF v2")  # the same path
    after = software(**at(prefix))
    assert before.id != after.id
    assert before.differences(after) == [
        f"the files LD_LIBRARY_PATH leads to in {prefix / 'lib'} changed",
        f"the files MARLIN_DLL leads to in {prefix / 'lib' / 'libDDMarlinPandora.so'} changed",
    ]


def test_the_same_build_at_another_prefix_is_the_same_software(tmp_path: Path) -> None:
    here = install(tmp_path / "here")
    there = shutil.copytree(here, tmp_path / "elsewhere" / "install")
    first, second = software(**at(here)), software(**at(there))
    assert first.id == second.id and first.differences(second) == []
    assert {entry.path for entry in first.local} != {entry.path for entry in second.local}


def test_what_the_stack_and_the_system_hold_stays_as_text(tmp_path: Path) -> None:
    value = "/cvmfs/stack/lib:/usr/lib64:relative/lib"
    assert identified("LD_LIBRARY_PATH", value) == (value, ())


def test_a_variable_of_no_role_stays_as_text(tmp_path: Path) -> None:
    install(tmp_path / "install")
    assert identified("MY_INSTALL", str(tmp_path / "install")) == (str(tmp_path / "install"), ())


def test_a_library_directory_is_what_it_holds_by_name_not_below(tmp_path: Path) -> None:
    prefix = install(tmp_path / "install")
    before = software(**at(prefix))
    (prefix / "lib" / "cmake").mkdir()
    (prefix / "lib" / "cmake" / "Config.cmake").write_text("# build files, never loaded\n")
    assert software(**at(prefix)).id == before.id
    (prefix / "lib" / "libNew.so").write_bytes(b"a library a plugin scan may find")
    assert software(**at(prefix)).id != before.id


def test_python_is_identified_by_what_it_can_import(tmp_path: Path) -> None:
    root = tmp_path / "python"
    (root / "helper").mkdir(parents=True)
    (root / "helper" / "__init__.py").write_text("")
    (root / "helper" / "scale.json").write_text("1.0")  # data of a regular package
    (root / "tool.py").write_text("x = 1\n")
    (root / "runs" / "study").mkdir(parents=True)  # no package: outputs live here
    (root / "runs" / "study" / "summary.json").write_text("{}")
    (root / "py-venv").mkdir()  # not importable: no identifier
    (root / "py-venv" / "big.py").write_text("")
    (root / "notes.txt").write_text("")
    _, (entry,) = identified("PYTHONPATH", str(root))
    before = software(PYTHONPATH=str(root))
    assert entry.files == 3  # helper/__init__.py, helper/scale.json, tool.py
    for unimported in ("runs/study/summary.json", "py-venv/big.py", "notes.txt"):
        (root / unimported).write_text("changed")
    (root / "helper" / "__pycache__").mkdir()  # what Python writes as it imports
    (root / "helper" / "__pycache__" / "x.pyc").write_bytes(b"cached")
    assert software(PYTHONPATH=str(root)).id == before.id
    (root / "helper" / "scale.json").write_text("1.05")
    assert software(PYTHONPATH=str(root)).id != before.id


def test_a_namespace_package_is_identified_by_its_modules(tmp_path: Path) -> None:
    root = tmp_path / "python"
    (root / "space" / "inner").mkdir(parents=True)
    (root / "space" / "inner" / "mod.py").write_text("x = 1\n")
    before = software(PYTHONPATH=str(root))
    (root / "space" / "inner" / "mod.py").write_text("x = 22\n")  # another size: see hashing
    assert software(PYTHONPATH=str(root)).id != before.id


def test_package_metadata_is_identified(tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    (root / "ddmarlin-1.0.dist-info").mkdir(parents=True)
    (root / "ddmarlin-1.0.dist-info" / "entry_points.txt").write_text("[k4opt]\n")
    before = software(PYTHONPATH=str(root))
    (root / "ddmarlin-1.0.dist-info" / "entry_points.txt").write_text("[k4opt]\nplugin = x\n")
    assert software(PYTHONPATH=str(root)).id != before.id


def test_headers_are_identified_below_their_directory(tmp_path: Path) -> None:
    include = tmp_path / "include"
    (include / "edm4hep").mkdir(parents=True)
    (include / "edm4hep" / "Hit.h").write_text("struct Hit {};\n")
    before = software(ROOT_INCLUDE_PATH=str(include))
    (include / "edm4hep" / "Hit.h").write_text("struct Hit { int x; };\n")
    assert software(ROOT_INCLUDE_PATH=str(include)).id != before.id


def test_a_preloaded_library_is_its_content(tmp_path: Path) -> None:
    library = tmp_path / "libpatch.so"
    library.write_bytes(b"v1")
    before = software(LD_PRELOAD=f"{library} /usr/lib64/libm.so.6")
    library.write_bytes(b"v2, rebuilt")
    after = software(LD_PRELOAD=f"{library} /usr/lib64/libm.so.6")
    assert before.differences(after) == [f"the files LD_PRELOAD leads to in {library} changed"]


def test_an_entry_too_large_to_read_is_identified_by_its_very_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlays, "MAX_FILES", 2)
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in ("a.so", "b.so", "c.so"):
        (lib / name).write_bytes(name.encode())
    old = 1_600_000_000_000_000_000
    for path in lib.iterdir():
        os.utime(path, ns=(old, old))  # an install of long ago
    first = software(LD_LIBRARY_PATH=str(lib))
    (entry,) = first.by_stat
    assert (entry.variable, entry.files, entry.content) == ("LD_LIBRARY_PATH", 3, False)
    assert software(LD_LIBRARY_PATH=str(lib)).id == first.id
    time.sleep(0.01)
    (lib / "b.so").write_bytes(b"B.SO")  # another build of the same size ...
    os.utime(lib / "b.so", ns=(old, old))  # ... its mtime kept, as an extraction may
    assert software(LD_LIBRARY_PATH=str(lib)).id != first.id
    moved = shutil.copytree(lib, tmp_path / "elsewhere")
    assert software(LD_LIBRARY_PATH=str(moved)).id != first.id  # not known to be the same


def test_a_link_that_leads_nowhere_is_recorded_and_a_new_target_is_other_software(
    tmp_path: Path,
) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libX.so").symlink_to(tmp_path / "missing.so")
    before = software(LD_LIBRARY_PATH=str(lib))
    (tmp_path / "missing.so").write_bytes(b"now there")
    assert software(LD_LIBRARY_PATH=str(lib)).id != before.id


def test_the_site_directories_and_what_their_pth_files_add_are_identified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_dir, added = tmp_path / "site-packages", tmp_path / "src"
    site_dir.mkdir()
    added.mkdir()
    (added / "editable.py").write_text("x = 1\n")
    (site_dir / "a.pth").write_text(f"# a comment\nimport os\n{added}\n../missing\n")
    monkeypatch.setattr(overlays.site, "getsitepackages", lambda: [str(site_dir)])
    monkeypatch.setattr(overlays.site, "ENABLE_USER_SITE", False)
    assert sites() == [str(site_dir), str(added)]
    before = Software.now({})
    assert {(e.variable, e.path) for e in before.local} == {
        (SITE, str(site_dir)),
        (SITE, str(added)),
    }
    (added / "editable.py").write_text("x = 2\n")  # an editable install, edited
    (said,) = before.differences(Software.now({}))
    assert said == f"what Python imports from {added} changed"


def test_local_entries_are_recorded_where_they_are_and_read_back(tmp_path: Path) -> None:
    written = software(**at(install(tmp_path / "install")))
    data = written.to_dict()
    assert {entry["variable"] for entry in data["local"]} == {  # type: ignore[union-attr]
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "MARLIN_DLL",
    }
    assert Software.from_dict(data) == written


def test_a_directory_on_two_variables_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "python"
    root.mkdir()
    (root / "tool.py").write_text("")
    listed: list[str] = []
    python = overlays._LISTED["python"]

    def counting(
        directory: str, prefix: str, ancestors: frozenset[str], read: overlays._Read
    ) -> overlays.Listed:
        listed.append(directory)
        return python(directory, prefix, ancestors, read)

    monkeypatch.setitem(overlays._LISTED, "python", counting)
    monkeypatch.setattr(overlays.site, "getsitepackages", lambda: [str(root)])
    monkeypatch.setattr(overlays.site, "ENABLE_USER_SITE", False)
    Software.now({"PYTHONPATH": str(root)})
    assert listed == [str(root)]


def test_an_unreadable_file_is_recorded_as_such(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads every file")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libX.so").write_bytes(b"x")
    (lib / "libX.so").chmod(0)
    (entry,) = software(LD_LIBRARY_PATH=str(lib)).local
    assert entry.sha256 is not None and entry.files == 1


def test_the_libraries_a_wheel_bundles_are_identified(tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    (root / "numpy" / "_core").mkdir(parents=True)
    (root / "numpy" / "__init__.py").write_text("")
    (root / "numpy.libs").mkdir()
    (root / "numpy.libs" / "libscipy_openblas64_.so").write_bytes(b"v1")
    before = software(PYTHONPATH=str(root))
    (root / "numpy.libs" / "libscipy_openblas64_.so").write_bytes(b"v2, rebuilt")
    assert software(PYTHONPATH=str(root)).id != before.id


def test_a_header_directory_reached_through_a_link_is_identified(tmp_path: Path) -> None:
    include, build = tmp_path / "include", tmp_path / "build" / "include" / "Detector"
    include.mkdir()
    build.mkdir(parents=True)
    (build / "Hit.h").write_text("struct Hit {};\n")
    (include / "Detector").symlink_to(build)
    before = software(ROOT_INCLUDE_PATH=str(include))
    (build / "Hit.h").write_text("struct Hit { int x; };\n")
    assert software(ROOT_INCLUDE_PATH=str(include)).id != before.id


@pytest.mark.parametrize("variable", ["PYTHONPATH", "ROOT_INCLUDE_PATH"])
def test_a_link_back_up_the_tree_ends(tmp_path: Path, variable: str) -> None:
    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "helper.py").write_text("")
    (root / "pkg" / "Hit.h").write_text("")
    (root / "Track.h").write_text("")
    (root / "loop").symlink_to(root)  # an identifier: Python could import through it
    (root / "pkg" / "up").symlink_to(root)
    (entry,) = software(**{variable: str(root)}).local
    assert entry.files >= 2


@pytest.mark.parametrize("spelling", [".", "", "./python/.."])
def test_a_relative_entry_is_what_it_leads_to_from_where_k4opt_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    here, there = tmp_path / "studyA", tmp_path / "studyB"
    for directory, scale in ((here, "1.0"), (there, "2.0")):
        (directory / "python").mkdir(parents=True)
        (directory / "helper.py").write_text(f"SCALE = {scale}\n")
    monkeypatch.chdir(here)
    first = software(PYTHONPATH=spelling)
    (entry,) = first.local
    assert entry.path == str(here)
    monkeypatch.chdir(there)
    assert software(PYTHONPATH=spelling).id != first.id  # the same text, other code


def test_a_relative_entry_that_leads_nowhere_stays_as_it_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert identified("PYTHONPATH", "missing") == ("missing", ())


def test_only_what_is_loaded_from_a_library_directory_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an empty entry is the current directory, where a job's own log may be growing
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nohup.out").write_text("candidate 1\n")
    (tmp_path / "libLocal.so.1.2").write_bytes(b"v1")
    before = software(LD_LIBRARY_PATH=":/cvmfs/stack/lib")
    (tmp_path / "nohup.out").write_text("candidate 1\ncandidate 2\n")
    assert software(LD_LIBRARY_PATH=":/cvmfs/stack/lib").id == before.id
    for name, content in (
        ("Local.rootmap", "x"),
        ("Local.components", "x"),
        ("Local_rdict.pcm", "x"),
    ):
        (tmp_path / name).write_text(content)
        after = software(LD_LIBRARY_PATH=":/cvmfs/stack/lib")
        assert after.id != before.id, name
        before = after


@pytest.mark.parametrize("name", ["TrackerConstants", "Hit.C", "Hit.hpp"])
def test_any_file_an_include_may_name_is_identified(tmp_path: Path, name: str) -> None:
    include = tmp_path / "include" / "Detector"
    include.mkdir(parents=True)
    (include / name).write_text("int x = 1;\n")
    before = software(ROOT_INCLUDE_PATH=str(include.parent))
    (include / name).write_text("int x = 22;\n")
    assert software(ROOT_INCLUDE_PATH=str(include.parent)).id != before.id


@pytest.mark.parametrize("variable", ["LD_PRELOAD", "MARLIN_DLL"])
def test_a_relative_file_is_the_file_it_leads_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "libpatch.so").write_bytes(b"v1")
    before = software(**{variable: "./libpatch.so"})
    (entry,) = before.local
    assert entry.path == str(tmp_path / "libpatch.so")
    (tmp_path / "libpatch.so").write_bytes(b"v2, rebuilt")
    assert software(**{variable: "./libpatch.so"}).id != before.id


def test_a_bare_library_name_is_looked_up_on_the_library_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "libpatch.so").write_bytes(b"v1")
    assert identified("LD_PRELOAD", "libpatch.so") == ("libpatch.so", ())


def test_what_is_installed_under_usr_local_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(overlays.os.path, "exists", lambda path: True)
    assert overlays._is_local("/usr/local/lib")
    assert overlays._is_local("/usr/local/lib/python3.14/site-packages")
    assert not overlays._is_local("/usr/lib64")
    assert not overlays._is_local("/usr/localized")


def test_a_variable_named_like_the_site_directories_is_compared(tmp_path: Path) -> None:
    first, second = software(site="A"), software(site="B")
    assert first.id != second.id
    assert first.differences(second) == ["the environment differs in site"]


def test_entries_identified_at_once_in_threads_agree(tmp_path: Path) -> None:
    roots = []
    for index in range(8):
        root = tmp_path / f"python{index}" / "pkg"
        root.mkdir(parents=True)
        (root / "__init__.py").write_text(str(index))
        roots.append(str(root.parent))
    found: list[str] = []
    errors: list[BaseException] = []

    def look() -> None:
        try:
            found.append(software(PYTHONPATH=os.pathsep.join(roots)).id)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=look) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(set(found)) == 1 and len(found) == 8


def test_an_empty_include_entry_stays_as_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # k4_local_repo leaves an empty entry, which ROOT skips: a study written below where k4opt
    # runs must not change the software
    monkeypatch.chdir(tmp_path)
    value = ":/cvmfs/stack/include"
    before = software(ROOT_INCLUDE_PATH=value)
    assert before.local == ()
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "study.json").write_text("{}")
    assert software(ROOT_INCLUDE_PATH=value).id == before.id
    # "." adds the working directory for an #include <...>: what it holds counts
    (entry,) = software(ROOT_INCLUDE_PATH=".").local
    assert entry.path == str(tmp_path)
