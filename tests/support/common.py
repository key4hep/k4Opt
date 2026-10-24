"""Helpers shared by all test modules."""

from __future__ import annotations

import hashlib
import os
import textwrap
from pathlib import Path


def write_tree(base: Path, files: dict[str, str]) -> Path:
    """Write ``{relative_path: content}`` below ``base`` (dedenting content) and return ``base``."""
    for rel, content in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return base


def tree_digest(root: Path) -> dict[str, str]:
    """Return ``{relative_path: sha256}`` for every file below ``root`` (symlinks by target)."""
    digest: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                digest[rel] = "symlink:" + os.readlink(path)
            else:
                digest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
