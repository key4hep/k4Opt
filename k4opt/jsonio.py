"""Publish complete JSON files atomically without replacing an existing destination."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    """Serialize into a hidden temporary sibling, then publish with an exclusive hard link.

    Serialization, flushing to disk (fsync) and closing finish before the destination becomes
    visible. Without the fsync, EOS shows the destination of a hard link to unsynced bytes as
    empty for a moment, so a reader that reads it at once gets nothing (measured: 5 of 1500).
    Both a failed write and a destination created concurrently leave that destination intact.
    The sibling shares the destination's directory, since AFS and EOS refuse hard links
    between directories, and gets the permissions of an ordinary new file.
    Filesystem and serialization errors propagate to the component that owns the file.
    """
    staged = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    handle = staged.open("x", encoding="utf-8")  # exclusive, so never another writer's file
    try:
        with handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)
