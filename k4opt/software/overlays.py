"""Local builds, by content: what a variable that decides what programs load leads to outside the
stack and the system.

A local build (``k4_local_repo``, a helper directory, a virtual environment, the user's
site-packages) is put in front of the stack through variables such as ``LD_LIBRARY_PATH`` and
``PYTHONPATH``. Their text says where it is, not what it is: a library rebuilt in place keeps the
text, and the same build at another prefix changes it. So each entry of such a variable that is a
local file or directory is replaced by the SHA-256 of what can be loaded through it, as its
variable uses it (its *role*, :data:`ROLES`); the rest of the value stays as it is. Entries under
``/cvmfs`` and the system's directories stay as text: the build and the system identify them.

What can be loaded through an entry, by role:

- ``libraries``: the files directly in the directory, which a library or a plugin is looked up
  among by name (the loader, ROOT's and DD4hep's ``.rootmap`` and ``.components`` scans);
- ``headers``: every file below the directory, which an ``#include`` names relative to it;
- ``python``: what Python can import from the directory: its modules (``.py``, ``.pyc``, ``.so``)
  and ``.pth`` files, every file of a regular package (one with an ``__init__``), with its data,
  the modules of a namespace package (a directory whose name is an identifier), and the package
  metadata (``*.dist-info``, ``*.egg-info``) that ``importlib.metadata`` reads; never
  ``__pycache__``, which Python writes as it imports;
- ``files``: each entry is a file (``MARLIN_DLL``, ``LD_PRELOAD``).

An entry that is a file is its content, whatever its role (a ``.zip`` on ``PYTHONPATH``). An entry
with more than :data:`MAX_FILES` files or :data:`MAX_BYTES` bytes is not identified by content:
it stays as text, and :class:`Local` says so (``sha256`` is ``None``).

Python also imports from the interpreter's site directories, which no variable names: those of a
virtual environment and the user's (``pip install --user``), and the directories their ``.pth``
files add. They are identified as one more variable of role ``python``, :data:`SITE`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import site
import stat
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from k4opt.hashing import Stat, sha256_of

type Role = Literal["libraries", "headers", "python", "files"]

ROLES: dict[str, Role] = {
    "LD_LIBRARY_PATH": "libraries",
    "ROOT_LIBRARY_PATH": "libraries",
    "GAUDI_PLUGIN_PATH": "libraries",
    "ROOT_INCLUDE_PATH": "headers",
    "PYTHONPATH": "python",
    "MARLIN_DLL": "files",
    "LD_PRELOAD": "files",
}
"""The variables whose local entries are identified by content, with what is loaded through them."""

SITE = "site"
"""The name under which the site directories of k4opt's interpreter are identified (:func:`sites`),
as a variable of role ``python``: what a Python program of the same interpreter imports besides
``PYTHONPATH``. ddsim and k4run run in the first ``python3`` on ``PATH``, k4opt's own in an
environment set up as documented."""

SHARED = ("/cvmfs", "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc")
"""Where entries are identified by their text: the stack (by its build) and the system."""

MAX_FILES = 50_000
MAX_BYTES = 4 << 30

_MODULES = (".py", ".pyc", ".so", ".pth")
_METADATA = (".dist-info", ".egg-info")
_TOKEN = "<{}>"

type Listed = Iterator[tuple[str, str, Stat | None]]
"""Files that can be loaded through an entry: each one's name relative to it, its path and its
stat (``None`` for one that cannot be read)."""


@dataclass(frozen=True, slots=True)
class Local:
    """A local entry of ``variable``: where it is (``path``), the SHA-256 of what can be loaded
    through it (``None`` if it is too large to identify by content), and how many ``files`` and
    ``bytes`` that is."""

    variable: str
    path: str
    sha256: str | None
    files: int
    bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "variable": self.variable,
            "path": self.path,
            "sha256": self.sha256,
            "files": self.files,
            "bytes": self.bytes,
        }


type Seen = dict[tuple[Role, str], Local]
"""What one look at the environment already identified, by role and entry: a directory on both
``PYTHONPATH`` and among the site directories is read once."""


def identified(
    variable: str, value: str, seen: Seen | None = None
) -> tuple[str, tuple[Local, ...]]:
    """``value`` of ``variable`` with each local entry replaced by its content, and those entries;
    ``value`` as it is for a variable of no role."""
    role = ROLES.get(variable)
    if role is None:
        return value, ()
    split = re.split(r"[:\s]+", value) if variable == "LD_PRELOAD" else value.split(os.pathsep)
    return _replaced(variable, role, split, {} if seen is None else seen)


def sites() -> list[str]:
    """The site directories of this interpreter, the user's included where Python reads it, each
    followed by the directories its ``.pth`` files add, in the order Python reads them."""
    found: list[str] = []
    user = [site.getusersitepackages()] if site.ENABLE_USER_SITE else []
    for directory in [*site.getsitepackages(), *user]:
        found.append(directory)
        for name in sorted(_names(directory)):
            if name.endswith(".pth"):
                found += _added(directory, name)
    return list(dict.fromkeys(found))


def _added(directory: str, name: str) -> list[str]:
    """The directories the ``.pth`` file ``name`` in ``directory`` adds, as :mod:`site` reads it:
    one per line but comments, ``import`` lines and directories that do not exist."""
    try:
        with open(os.path.join(directory, name), encoding="utf-8", errors="replace") as pth:
            lines = [line.rstrip() for line in pth]
    except OSError:
        return []
    paths = (
        os.path.normpath(os.path.join(directory, line))
        for line in lines
        if line and not line.startswith(("#", "import ", "import\t"))
    )
    return [path for path in paths if os.path.isdir(path)]


def site_identified(
    directories: Sequence[str], seen: Seen | None = None
) -> tuple[str, tuple[Local, ...]]:
    """The site ``directories`` (:func:`sites`) as :data:`SITE` is identified, each local one by
    what Python can import from it."""
    return _replaced(SITE, "python", list(directories), {} if seen is None else seen)


def _replaced(
    variable: str, role: Role, entries: list[str], seen: Seen
) -> tuple[str, tuple[Local, ...]]:
    kept, local = [], []
    for entry in entries:
        if not _is_local(entry):
            kept.append(entry)
            continue
        if (role, entry) not in seen:
            seen[(role, entry)] = _identify(variable, role, entry)
        found = replace(seen[(role, entry)], variable=variable)
        local.append(found)
        kept.append(entry if found.sha256 is None else _TOKEN.format(found.sha256))
    return os.pathsep.join(kept), tuple(local)


def _is_local(entry: str) -> bool:
    if not os.path.isabs(entry) or not os.path.exists(entry):
        return False
    return not any(entry == root or entry.startswith(f"{root}/") for root in SHARED)


class _TooLargeError(Exception):
    pass


def _identify(variable: str, role: Role, entry: str) -> Local:
    files: list[tuple[str, str]] = []
    size = 0
    try:
        listed = _file(entry) if os.path.isfile(entry) else _LISTED[role](entry, "")
        for relative, path, status in listed:
            if status is None:  # a link that leads nowhere, or a file gone meanwhile
                files.append((relative, "unreadable"))
                continue
            size += status[0]
            if len(files) >= MAX_FILES or size > MAX_BYTES:
                raise _TooLargeError
            try:
                files.append((relative, sha256_of(path, status)))
            except OSError:
                files.append((relative, "unreadable"))
    except _TooLargeError:
        return Local(variable, entry, None, len(files), size)
    text = json.dumps(sorted(files), separators=(",", ":"))
    return Local(variable, entry, hashlib.sha256(text.encode()).hexdigest(), len(files), size)


def _stat(entry: os.DirEntry[str]) -> Stat | None:
    try:
        status = entry.stat()  # links followed, as a program that opens it does
    except OSError:
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    return (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)


def _scan(directory: str) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(directory) as found:
            return list(found)
    except OSError:
        return []


def _names(directory: str) -> list[str]:
    return [entry.name for entry in _scan(directory)]


def _is_dir(entry: os.DirEntry[str], follow: bool = True) -> bool:
    try:
        return entry.is_dir(follow_symlinks=follow)
    except OSError:
        return False


def _is_file(entry: os.DirEntry[str]) -> bool:
    """A file to load: a regular one, or a link that leads nowhere (recorded as unreadable)."""
    try:
        return entry.is_file() or (entry.is_symlink() and not os.path.exists(entry.path))
    except OSError:
        return False


def _file(path: str) -> Listed:
    try:
        status = os.stat(path)
    except OSError:
        yield "", path, None
        return
    yield "", path, (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)


def _libraries(directory: str, prefix: str) -> Listed:
    for entry in _scan(directory):
        if _is_file(entry):
            yield f"{prefix}{entry.name}", entry.path, _stat(entry)


def _headers(directory: str, prefix: str) -> Listed:
    for entry in _scan(directory):
        if _is_dir(entry, follow=False):
            yield from _headers(entry.path, f"{prefix}{entry.name}/")
        elif _is_file(entry):
            yield f"{prefix}{entry.name}", entry.path, _stat(entry)


def _python(directory: str, prefix: str, top: bool = True) -> Listed:
    for entry in _scan(directory):
        if entry.name == "__pycache__":
            continue
        if _is_dir(entry):
            if top and entry.name.endswith(_METADATA):
                yield from _package(entry.path, f"{entry.name}/")
            elif entry.name.isidentifier():
                inside = _scan(entry.path)
                if any(item.name.startswith("__init__.") for item in inside):
                    yield from _package(entry.path, f"{prefix}{entry.name}/", inside)
                else:
                    yield from _python(entry.path, f"{prefix}{entry.name}/", top=False)
        elif _is_file(entry) and entry.name.endswith(_MODULES):
            yield f"{prefix}{entry.name}", entry.path, _stat(entry)


def _package(directory: str, prefix: str, scanned: list[os.DirEntry[str]] | None = None) -> Listed:
    for entry in _scan(directory) if scanned is None else scanned:
        if entry.name == "__pycache__":
            continue
        if _is_dir(entry):
            yield from _package(entry.path, f"{prefix}{entry.name}/")
        elif _is_file(entry):
            yield f"{prefix}{entry.name}", entry.path, _stat(entry)


def _nothing(directory: str, prefix: str) -> Listed:
    """What a variable of files loads through a directory on it: nothing."""
    return iter(())


_LISTED = {"libraries": _libraries, "headers": _headers, "python": _python, "files": _nothing}
