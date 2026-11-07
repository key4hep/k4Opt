"""Helpers shared by all test modules."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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


def scientific(summary: dict[str, Any], dropped: frozenset[str] = frozenset()) -> dict[str, Any]:
    """What a summary says of its candidate, without where and how fast it ran, nor how its
    tasks were executed (its ``id``, ``run_dir``, times, peak memory, ``execution`` and logs), nor
    the keys ``dropped``; the run directory in a path reads ``<run>``."""
    gone = {"id", "run_dir", "elapsed_seconds", "peak_memory_bytes", "log", "execution", *dropped}

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key not in gone}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    stripped: dict[str, Any] = strip(
        json.loads(json.dumps(summary).replace(summary["run_dir"], "<run>"))
    )
    return stripped


@contextlib.contextmanager
def set_beneath_python(name: str, value: str) -> Iterator[None]:
    """``name`` set as a library does with C's ``setenv`` (ROOT's interpreter does it for
    ``CLING_PREBUILT_MODULE_PATH``): in the process's environment, not in ``os.environ``."""
    libc = ctypes.CDLL(None)
    assert libc.setenv(name.encode(), value.encode(), 1) == 0 and name not in os.environ
    try:
        yield
    finally:
        libc.unsetenv(name.encode())


class _Uncached(dict[tuple[str, Any], str]):
    """The hash cache of :mod:`k4opt.hashing`, which keeps nothing below ``root``."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = f"{root}/"

    def __setitem__(self, key: tuple[str, Any], value: str) -> None:
        if not key[0].startswith(self.root):
            super().__setitem__(key, value)


def counted_hashes(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[str]:
    """Every file below ``root`` that :mod:`k4opt.hashing` reads from now on, in order, each read
    counted (none cached); the files of the software stay cached as before."""
    from k4opt import hashing

    read, hashed = hashing._read, []

    def counting(name: str) -> str:
        if name.startswith(f"{root}/"):
            hashed.append(name)
        return read(name)

    monkeypatch.setattr(hashing, "_read", counting)
    monkeypatch.setattr(hashing, "_HASHES", _Uncached(root) | hashing._HASHES)
    return hashed
