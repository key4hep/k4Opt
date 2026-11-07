"""Local builds, by content: what a variable that decides what programs load leads to outside the
stack and the system.

A local build (``k4_local_repo``, a helper directory, a virtual environment, the user's
site-packages) is put in front of the stack through variables such as ``LD_LIBRARY_PATH`` and
``PYTHONPATH``. Their text says where it is, not what it is: a library rebuilt in place keeps the
text, and the same build at another prefix changes it. So each entry of such a variable that is a
local file or directory is replaced by the SHA-256 of what can be loaded through it, as its
variable uses it (its *role*, :data:`ROLES`); the rest of the value stays as it is. Entries under
``/cvmfs`` and the system's directories stay as text: the build and the system identify them, but
``/usr/local``, where software is installed by hand. A relative entry is what it leads to from the
directory k4opt runs in, where k4opt's process and its workers resolve it.

What can be loaded through an entry, by role:

- ``libraries``: the libraries directly in the directory, which the loader looks up by name, and
  the registries and dictionaries ROOT, DD4hep and Gaudi read there (``.rootmap``, ``.components``,
  ``.pcm``);
- ``headers``: every file below the directory, which an ``#include`` may name relative to it,
  with any extension or none; but for an empty entry (which ``k4_local_repo`` leaves), which
  ROOT skips;
- ``python``: what Python can import from the directory: its modules (``.py``, ``.pyc``, ``.so``)
  and ``.pth`` files, every file of a regular package (one with an ``__init__``), with its data,
  the modules of a namespace package (a directory whose name is an identifier), and the package
  metadata (``*.dist-info``, ``*.egg-info``) that ``importlib.metadata`` reads, and the native
  libraries a wheel bundles (``*.libs``); never ``__pycache__``, which Python writes as it imports;
- ``files``: each entry is a file (``MARLIN_DLL``, ``LD_PRELOAD``).

An entry that is a file is its content, whatever its role (a ``.zip`` on ``PYTHONPATH``). An entry
with more than :data:`MAX_FILES` files or :data:`MAX_BYTES` bytes is identified by each file's
size and mtime rather than by its content, which would be read on every session; :class:`Local`
says so. Links are followed, as a lookup through them is; one back up the tree is not.

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
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from k4opt.hashing import Stat, read_sha256, sha256_of, stat_of

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
"""What the local entries of the site directories of k4opt's interpreter (:func:`sites`) name as
their variable, of role ``python``; the software holds their digest apart from the variables, so
that no variable can stand for it: what a Python program of the same interpreter imports besides
``PYTHONPATH``. ddsim and k4run run in the first ``python3`` on ``PATH``, k4opt's own in an
environment set up as documented."""

SHARED = ("/cvmfs", "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc")
"""Where entries are identified by their text: the stack (by its build) and the system."""
INSTALLED = ("/usr/local",)
"""Where in :data:`SHARED` software is installed by hand or by ``pip``, not by the system: local."""

MAX_FILES = 50_000
MAX_BYTES = 4 << 30

_MODULES = (".py", ".pyc", ".so", ".pth")
_LIBRARY = re.compile(r"\.so(\.[\w.]*)?$|\.(rootmap|components|pcm)$")
"""What is loaded from a library directory: libraries (``libX.so``, ``libX.so.1.2``), and the
registries a plugin is looked up in and the dictionaries ROOT reads (``.rootmap``, ``.components``,
``.pcm``); never what else lies there (build files, a log being written)."""
_BESIDE = (".dist-info", ".egg-info", ".libs")
"""Directories beside the packages of a site directory that no import names but Python reads:
package metadata, and the native libraries a wheel bundles (``numpy.libs``), which its extension
modules load through their run path."""
_TOKEN = "<{}>"

type _Read = list[tuple[str, Stat | None]]
"""The directories a walk scanned, with their stat when it scanned them."""

type Listed = Iterator[tuple[str, str, Stat | None]]
"""Files that can be loaded through an entry: each one's name relative to it, its path and its
stat (``None`` for one that cannot be read)."""


@dataclass(frozen=True, slots=True)
class Local:
    """A local entry of ``variable``: where it is (``path``), the SHA-256 of what can be loaded
    through it, and how many ``files`` and ``bytes`` that is; by their ``content``, or, for an entry
    too large to read on every session, by each file's size and mtime (``content`` false): a
    rebuild still changes it, a copy that keeps mtimes does not."""

    variable: str
    path: str
    sha256: str
    files: int
    bytes: int
    content: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "variable": self.variable,
            "path": self.path,
            "sha256": self.sha256,
            "files": self.files,
            "bytes": self.bytes,
            "content": self.content,
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
    kept: list[str] = []
    local: list[Local] = []
    for given in entries:
        entry = given
        if not os.path.isabs(given) and (role != "files" or "/" in given):
            # "" and "." are the current directory, as "./libpatch.so" is a file in it; a bare
            # library name (LD_PRELOAD=libpatch.so) is looked up on the library path instead
            entry = os.path.normpath(os.path.join(os.getcwd(), given))
        if role == "headers" and not given:
            # ROOT skips an empty entry, and looks in the working directory for an #include "..."
            # anyway; "." does add it, for an #include <...> too: identified as any other
            kept.append(given)
            continue
        if not _is_local(entry):  # a relative one that leads nowhere: as written, wherever it is
            kept.append(entry if os.path.exists(entry) else given)
            continue
        if (role, entry) not in seen:
            seen[(role, entry)] = _identify(variable, role, entry)
        found = replace(seen[(role, entry)], variable=variable)
        local.append(found)
        kept.append(_TOKEN.format(found.sha256))
    return os.pathsep.join(kept), tuple(local)


def _is_local(entry: str) -> bool:
    if not os.path.isabs(entry) or not os.path.exists(entry):
        return False
    if _under(entry, INSTALLED):
        return True
    return not _under(entry, SHARED)


def _under(entry: str, roots: tuple[str, ...]) -> bool:
    return any(entry == root or entry.startswith(f"{root}/") for root in roots)


@dataclass(frozen=True, slots=True)
class _Taken:
    """What a look at an entry found, and the stat of every file and directory it read: while
    none of them changed, neither did what can be loaded through the entry."""

    local: Local
    stats: tuple[tuple[str, Stat | None], ...]

    def unchanged(self) -> bool:
        return all(stat_of(path) == status for path, status in self.stats)


_TAKEN: dict[tuple[Role, str], _Taken] = {}
"""The last look at each entry in this process: a later look stats what it read (tens of
milliseconds for a virtual environment) and walks and hashes again only if something changed."""

RECENT_NS = 2_000_000_000
"""How recent a change may be for a stat not to vouch for what it is: a file or directory changed
twice within one tick of the kernel's coarse clock (milliseconds, a second on some network
filesystems) keeps its stat. What changed more recently is read again on every look, as git
treats a "racily clean" file."""


def _recent(status: Stat | None, now: int) -> bool:
    return status is not None and max(status[1], status[2]) > now - RECENT_NS


def _identify(variable: str, role: Role, entry: str) -> Local:
    taken = _TAKEN.get((role, entry))
    if taken is not None and taken.unchanged():
        return taken.local
    now = time.time_ns()
    local, stats = _walk(variable, role, entry, now)
    if not any(_recent(status, now) for _, status in stats):
        _TAKEN[(role, entry)] = _Taken(local, tuple(stats))
    else:
        _TAKEN.pop((role, entry), None)
    return local


def _walk(
    variable: str, role: Role, entry: str, now: int
) -> tuple[Local, list[tuple[str, Stat | None]]]:
    """What can be loaded through ``entry``, and the stat of each file of it: by content up to
    :data:`MAX_FILES` files and :data:`MAX_BYTES` bytes, by each file's size and mtime beyond."""
    read: list[tuple[str, Stat | None]] = []  # the directories scanned, whose names it lists
    if os.path.isfile(entry):
        listed = list(_file(entry))
    else:
        listed = list(_LISTED[role](entry, "", frozenset(), read))
    size = sum(status[0] for _, _, status in listed if status is not None)
    content = len(listed) <= MAX_FILES and size <= MAX_BYTES
    files: list[tuple[str, object]] = []
    for relative, path, status in listed:
        if status is None:  # a link that leads nowhere, or a file gone meanwhile
            files.append((relative, "unreadable"))
        elif not content:  # these very files: any write changes its ctime, which nothing sets back
            files.append((relative, list(status)))
        else:
            try:
                recent = _recent(status, now)
                files.append((relative, read_sha256(path) if recent else sha256_of(path, status)))
            except OSError:
                files.append((relative, "unreadable"))
    text = json.dumps(sorted(files), separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    stats = [*read, *((path, status) for _, path, status in listed)]
    return Local(variable, entry, digest, len(listed), size, content), stats


def _stat(entry: os.DirEntry[str]) -> Stat | None:
    try:
        status = entry.stat()  # links followed, as a program that opens it does
    except OSError:
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    return (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)


def _scan(directory: str, read: _Read | None = None) -> list[os.DirEntry[str]]:
    if read is not None:  # what it holds by name changes only with its own stat
        read.append((directory, stat_of(directory)))
    try:
        with os.scandir(directory) as found:
            return list(found)
    except OSError:
        return []


def _names(directory: str) -> list[str]:
    return [entry.name for entry in _scan(directory)]


def _within(directory: str, ancestors: frozenset[str]) -> frozenset[str] | None:
    """``ancestors`` with ``directory`` where its links lead; ``None`` if it is one of them (a
    link back up the tree), which would never end."""
    real = os.path.realpath(directory)
    return None if real in ancestors else ancestors | {real}


def _is_dir(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_dir()  # links followed, as a lookup through it does
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


def _libraries(directory: str, prefix: str, ancestors: frozenset[str], read: _Read) -> Listed:
    for entry in _scan(directory, read):
        if _is_file(entry) and _LIBRARY.search(entry.name):
            yield f"{prefix}{entry.name}", entry.path, _stat(entry)


def _headers(directory: str, prefix: str, ancestors: frozenset[str], read: _Read) -> Listed:
    inside = _within(directory, ancestors)
    if inside is None:
        return
    for entry in _scan(directory, read):
        if _is_dir(entry):
            yield from _headers(entry.path, f"{prefix}{entry.name}/", inside, read)
        elif _is_file(entry):  # an #include may name any file: "TrackerConstants", "Hit.C"
            yield f"{prefix}{entry.name}", entry.path, _stat(entry)


def _python(
    directory: str, prefix: str, ancestors: frozenset[str], read: _Read, top: bool = True
) -> Listed:
    inside = _within(directory, ancestors)
    if inside is None:
        return
    for entry in _scan(directory, read):
        if entry.name == "__pycache__":
            continue
        if _is_dir(entry):
            if top and entry.name.endswith(_BESIDE):
                yield from _package(entry.path, f"{entry.name}/", inside, read)
            elif entry.name.isidentifier():
                scanned = _scan(entry.path, read)
                if any(item.name.startswith("__init__.") for item in scanned):
                    yield from _package(entry.path, f"{prefix}{entry.name}/", inside, read, scanned)
                else:
                    yield from _python(
                        entry.path, f"{prefix}{entry.name}/", inside, read, top=False
                    )
        elif _is_file(entry) and entry.name.endswith(_MODULES):
            yield f"{prefix}{entry.name}", entry.path, _stat(entry)


def _package(
    directory: str,
    prefix: str,
    ancestors: frozenset[str],
    read: _Read,
    scanned: list[os.DirEntry[str]] | None = None,
) -> Listed:
    inside = _within(directory, ancestors)
    if inside is None:
        return
    for entry in _scan(directory, read) if scanned is None else scanned:
        if entry.name == "__pycache__":
            continue
        if _is_dir(entry):
            yield from _package(entry.path, f"{prefix}{entry.name}/", inside, read)
        elif _is_file(entry):
            yield f"{prefix}{entry.name}", entry.path, _stat(entry)


def _nothing(directory: str, prefix: str, ancestors: frozenset[str], read: _Read) -> Listed:
    """What a variable of files loads through a directory on it: nothing."""
    return iter(())


_LISTED = {"libraries": _libraries, "headers": _headers, "python": _python, "files": _nothing}
