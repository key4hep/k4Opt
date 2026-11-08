"""Publish complete files (JSON records, exports) atomically: without replacing an existing
destination (:func:`publish`), or, for the one file that is rewritten (a publication's catalog),
replacing it whole (:func:`replace`)."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    """Serialize ``value`` as indented JSON and :func:`publish` it at ``path``.

    Serialization finishes before anything is written, so a value that cannot be serialized
    leaves no file. Filesystem and serialization errors propagate to the component that owns the
    file.
    """
    publish(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


PRIVATE = 0o600
"""The permissions of a file only its owner may read (``private``)."""


def _staged(path: Path, text: str | bytes, private: bool = False) -> Path:
    """``text`` (UTF-8, or the bytes as they are) in a new hidden sibling of ``path``, flushed to
    disk and closed; readable by its owner alone if ``private``, from the moment it exists."""
    staged = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    # exclusive, so never another writer's file
    descriptor = os.open(
        staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE if private else 0o666
    )
    handle = os.fdopen(descriptor, "wb")
    try:
        with handle:
            handle.write(text.encode() if isinstance(text, str) else text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def publish(path: Path, text: str | bytes, private: bool = False) -> None:
    """Write ``text`` (UTF-8, or the bytes as they are) into a hidden temporary sibling, then
    publish it with an exclusive hard link.

    Writing, flushing to disk (fsync) and closing finish before the destination becomes visible.
    Without the fsync, EOS shows the destination of a hard link to unsynced bytes as empty for a
    moment, so a reader that reads it at once gets nothing (measured: 5 of 1500). Both a failed
    write and a destination created concurrently leave that destination intact, and an existing
    destination is never replaced (:class:`FileExistsError`). The sibling shares the
    destination's directory, since AFS and EOS refuse hard links between directories, and gets
    the permissions of an ordinary new file, or only its owner's if ``private``. Filesystem errors
    propagate to the caller.
    """
    staged = _staged(path, text, private)
    try:
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def replace(path: Path, text: str, private: bool = False) -> None:
    """Write ``text`` (UTF-8) as :func:`publish` does, then move it over ``path``, which may exist.

    A reader of ``path`` sees the file before or after, whole, never a part of either: the rename
    is atomic within the directory (also on EOS and AFS), and the bytes are on disk before it. A
    failed write leaves ``path`` as it was. Concurrent writers are the caller's to keep apart: the
    last rename wins. ``private`` as for :func:`publish`. Filesystem errors propagate to the
    caller.
    """
    staged = _staged(path, text, private)
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
