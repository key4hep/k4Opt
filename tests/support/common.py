"""Helpers shared by all test modules."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def git(cwd: Path, *args: str) -> str:
    """Run ``git`` in ``cwd`` without the user's or the system's configuration; its output."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.org",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.org",
    }
    run = subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)
    return run.stdout.decode()


def git_checkout(path: Path, files: dict[str, str]) -> str:
    """A new git checkout at ``path`` with ``files`` committed; the commit."""
    write_tree(path, files)
    git(path, "init", "--quiet")
    git(path, "add", *files)
    git(path, "commit", "--quiet", "--message", "files")
    return git(path, "rev-parse", "HEAD").strip()


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
