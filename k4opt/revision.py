"""Revisions of code and data, shared by the components that record provenance.

The geometry records the git checkout its files come from, an evaluator's source the checkout
that tracks its file, and a run the k4opt code that made it. Git only labels a revision:
uncommitted edits show as ``dirty``, and a file that no checkout tracks has no revision at all.
k4opt's own code is identified by its content, :func:`code_fingerprint`, which does not depend on
git or on where the package is installed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from k4opt.frozen import Json


class GitInfo:
    """Git revision of a checkout."""

    __slots__ = ("commit", "dirty")

    def __init__(self, commit: str, dirty: bool) -> None:
        self.commit = commit
        self.dirty = dirty
        """True when tracked files have uncommitted modifications (untracked files are ignored)."""

    def to_dict(self) -> dict[str, Any]:
        return {"commit": self.commit, "dirty": self.dirty}

    def __repr__(self) -> str:
        return f"GitInfo(commit={self.commit!r}, dirty={self.dirty!r})"


def _git(args: list[str], cwd: Path, timeout: int = 30) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def detect_git(root: Path) -> GitInfo | None:
    """Return the git revision of ``root`` or ``None`` when unavailable."""
    head = _git(["rev-parse", "HEAD"], root)
    if head is None:
        return None
    status = _git(["status", "--porcelain", "--untracked-files=no"], root, timeout=60)
    if status is None:
        return None
    return GitInfo(commit=head.strip(), dirty=bool(status.strip()))


def git_toplevel(path: Path) -> Path | None:
    """Return the real path of the git work tree containing ``path``, if any."""
    top = _git(["rev-parse", "--show-toplevel"], path)
    if not top or not top.strip():
        return None
    return Path(os.path.realpath(top.strip()))


def tracked_revision(path: Path) -> GitInfo | None:
    """The revision of the git checkout that tracks the file ``path``.

    ``None`` outside a checkout, and for a file the checkout around it does not track (an ignored
    or new file): that checkout's commit would not contain it.
    """
    if _git(["ls-files", "--error-unmatch", "--", path.name], path.parent) is None:
        return None
    return detect_git(path.parent)


def source_digest(directory: Path) -> str:
    """SHA-256 over the relative path and the content of every ``.py`` file below ``directory``.

    Equal for two copies of the same files wherever they are; any edit, added or renamed file
    changes it.
    """
    files = {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*.py")
        if path.is_file()
    }
    canonical = json.dumps(dict(sorted(files.items())), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@functools.cache
def code_fingerprint() -> str:
    """The :func:`source_digest` of the k4opt package, taken once per process.

    It identifies the files as they are when first asked for, which a run does before it
    evaluates anything.
    """
    return source_digest(Path(__file__).parent)


@functools.cache
def _code_revision() -> GitInfo | None:
    return tracked_revision(Path(__file__))


def code_identity() -> dict[str, Json]:
    """``sha256`` (:func:`code_fingerprint`) and ``git`` (the commit and dirty flag of the
    checkout that tracks k4opt, ``None`` if none does) of the running k4opt."""
    git = _code_revision()
    return {"sha256": code_fingerprint(), "git": None if git is None else git.to_dict()}
