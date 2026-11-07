"""Files by content: the SHA-256 of a file, hashed once per version of it.

A version is told apart from a rewritten one, on the filesystem that took it, by its ``stat``: its
size, times and inode (:data:`Stat`). The stats are machine-specific and never recorded. The hashes
this process took (:func:`known`) can be handed to another process on the same machine
(:func:`learn`), so that a worker does not read again what its coordinator already read.

Shared by the components that identify files: the execution (what a candidate reads and makes)
and the software (what a local build installed).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path

from k4opt.frozen import Json

type Stat = tuple[int, int, int, int]
"""What tells a file apart from a rewritten one: its size, mtime, ctime and inode."""

type Known = tuple[tuple[str, Stat, str], ...]
"""Hashes taken: each file's path, its stat when it was hashed and its SHA-256."""

_HASHES: dict[tuple[str, Stat], str] = {}
"""The SHA-256 of each version of a file this process read, by its path and stat. Not bounded: a
session checks every file it identified before each candidate, so a bound smaller than their number
would hash each of them every time; an entry is a few hundred bytes per version of a file."""


def stat_of(path: str | os.PathLike[str]) -> Stat | None:
    """The :data:`Stat` of the file at ``path`` (links followed); ``None`` if there is none."""
    try:
        status = os.stat(path)
    except OSError:
        return None
    return (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)


def sha256_of(path: str | os.PathLike[str], stat: Stat | None = None) -> str:
    """SHA-256 of the file at ``path``, whose stat is ``stat`` (taken now if ``None``): read once
    per version of the file in this process. A changed file is hashed again: writing to a file or
    replacing it changes its ctime or inode even if the content keeps its size and its mtime is
    restored. The kernel stamps ctime with a coarse clock (milliseconds), so a file rewritten within
    the same tick as it was hashed is not told apart. Raises :class:`OSError`."""
    name = os.fspath(path)
    if stat is None:
        status = os.stat(name)
        stat = (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)
    found = _HASHES.get((name, stat))
    if found is None:
        found = _HASHES[(name, stat)] = _read(name)
    return found


def read_sha256(path: str | os.PathLike[str]) -> str:
    """SHA-256 of the file at ``path``, read now and kept nowhere: for a file whose stat cannot
    vouch for its content (one changed within the kernel clock's last tick). Raises
    :class:`OSError`."""
    return _read(os.fspath(path))


def _read(name: str) -> str:
    """SHA-256 of the file ``name``, read now."""
    digest = hashlib.sha256()
    with Path(name).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Json]:
    """``bytes`` and ``sha256`` of the file at ``path``; raises :class:`OSError`."""
    stat = stat_of(path)
    if stat is None:
        raise OSError(f"{path} cannot be read")
    return {"bytes": stat[0], "sha256": sha256_of(path, stat)}


def readable_identity(path: Path) -> dict[str, Json] | None:
    """:func:`file_identity`, or ``None`` for a file that cannot be read: whatever reads it
    reports that."""
    try:
        return file_identity(path)
    except OSError:
        return None


def known() -> Known:
    """Every hash this process took, for :func:`learn` in another process on this machine."""
    return tuple((name, stat, sha256) for (name, stat), sha256 in _HASHES.items())


def learn(hashes: Iterable[tuple[str, Stat, str]]) -> None:
    """Take ``hashes`` (:func:`known` of another process on this machine) as this process's own:
    a file with the same path and stat is the same version of it."""
    for name, (size, mtime_ns, ctime_ns, inode), sha256 in hashes:
        _HASHES[(name, (size, mtime_ns, ctime_ns, inode))] = sha256
