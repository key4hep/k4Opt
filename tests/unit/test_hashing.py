"""Files by content (``k4opt.hashing``): each version of a file is read once per process, and a
worker can take what its coordinator read."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest

from k4opt import hashing
from k4opt.hashing import file_identity, known, learn, stat_of


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every file read from now on, with an empty cache."""
    read, seen = hashing._read, []

    def counting(name: str) -> str:
        seen.append(name)
        return read(name)

    monkeypatch.setattr(hashing, "_read", counting)
    monkeypatch.setattr(hashing, "_HASHES", {})
    return seen


def test_many_files_checked_again_and_again_are_each_read_once(
    tmp_path: Path, reads: list[str]
) -> None:
    # as a session checks a directory of data before each candidate, in the same order
    files = [tmp_path / f"{i}.dat" for i in range(100)]
    for file in files:
        file.write_bytes(file.name.encode())
    for _ in range(3):
        for file in files:
            file_identity(file)
    assert sorted(reads) == sorted(map(str, files))


def test_a_file_is_read_once_until_it_changes(tmp_path: Path, reads: list[str]) -> None:
    hepmc = tmp_path / "zh.hepmc"
    hepmc.write_bytes(b"one")
    first = file_identity(hepmc)
    for _ in range(3):
        assert file_identity(hepmc) == first
    assert reads == [str(hepmc)]
    hepmc.write_bytes(b"three")  # another size: a different file version
    changed = file_identity(hepmc)
    assert changed["sha256"] == hashlib.sha256(b"three").hexdigest() != first["sha256"]


def test_a_file_replaced_with_the_same_size_and_mtime_is_read_again(
    tmp_path: Path, reads: list[str]
) -> None:
    hepmc = tmp_path / "zh.hepmc"
    hepmc.write_bytes(b"AAAA")
    before = hepmc.stat()
    first = file_identity(hepmc)
    time.sleep(0.05)  # ctime comes from a coarse kernel clock
    hepmc.write_bytes(b"BBBB")  # same size ...
    os.utime(hepmc, ns=(before.st_atime_ns, before.st_mtime_ns))  # ... and the old mtime back
    second = file_identity(hepmc)
    assert first["sha256"] == hashlib.sha256(b"AAAA").hexdigest()
    assert second["sha256"] == hashlib.sha256(b"BBBB").hexdigest()
    assert len(reads) == 2


def test_what_one_process_read_another_takes_without_reading(
    tmp_path: Path, reads: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "libDDCore.so"
    library.write_bytes(b"ELF")
    taken = file_identity(library)
    handed = known()
    monkeypatch.setattr(hashing, "_HASHES", {})  # another process
    learn(handed)
    assert file_identity(library) == taken and reads == [str(library)]
    library.write_bytes(b"ELF, rebuilt")  # what it learned is of another version
    assert file_identity(library) != taken and len(reads) == 2


def test_a_file_that_cannot_be_read_has_no_stat_and_no_identity(tmp_path: Path) -> None:
    assert stat_of(tmp_path / "gone") is None
    assert hashing.readable_identity(tmp_path / "gone") is None
    with pytest.raises(OSError, match="cannot be read"):
        file_identity(tmp_path / "gone")
