"""What a stage used: the *footprint* of one observed run of a program, read from its raw traces.

A program run observed at the ``files`` level (:data:`k4opt.program.Observation`) leaves one trace
by process, which says each file it opened, each it failed to open, each program it started and
each connection it made. :func:`observe` reads them into a :class:`Footprint`, writes it beside the
stage's logs (:data:`FOOTPRINT`, gzipped JSON, kept private) and removes the traces; what the
record keeps of it is short (:meth:`Footprint.summary`): the packages of the stack the stage used,
the local entries it read from, the files it read that nothing else identifies, and whether it
used the network.

Each file the stage read is identified as it is when the stage ended: a file on CVMFS by the
content key CVMFS keeps of it, without reading it, any other by its SHA-256 (:mod:`k4opt.hashing`),
and a directory by the names in it (what a scan of it found). A file reached through a link is kept
under both names, so that a link retargeted to another file is seen though the old file stays. What
the stage made, what it was given and what it wrote, and the kernel's own files (:data:`IGNORED`),
are not part of it. A footprint *holds* (:meth:`Footprint.changes`) while each file it read is the
same, each directory holds the same names and no path it failed to find exists: so far as opening
files goes, the stage would find what it found. A file that could not be identified, or that
changed while the stage ran, never holds: what is not known is taken to differ.

A footprint is *complete* when its traces could be read to the end: every file the run opened is
in it. That is not every dependency of the run:

what was decided from a file's size or time, or from whether it exists, without opening it
(``stat``), an import that failed in Python, which answers from a listing it keeps, and a file the
stage read and also wrote, which counts as its own, are not in it. With the same software, the
first two can only concern files outside the stack and the local builds.

At the ``libraries`` level only the libraries the loader loaded are known (``LD_DEBUG``): enough to
say which packages ran, not what else was read, so such a footprint never holds for another study.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import time
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from k4opt.frozen import Json
from k4opt.hashing import read_sha256, sha256_of
from k4opt.program import Observation
from k4opt.software.errors import SoftwareError
from k4opt.software.overlays import _BESIDE, _LIBRARY, _MODULES, ROLES, SITE, Local, _recent

FOOTPRINT = "footprint.json.gz"
"""The file in a stage's directory that holds its footprint."""

SCHEMA = "k4opt.footprint/1"

IGNORED = ("/proc", "/sys", "/dev", "/run", "/var/run", "/var/lib/sss")
"""Where what a stage reads is the kernel's or the session's, never an input: the processes, the
devices, and the caches of the user database, which change as anyone logs in."""

SYSTEM = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc")
"""Where the system's package manager installs: identified, roughly, by the system
(:class:`~k4opt.software.System`), never listed among a stage's undeclared inputs; ``/usr/local``,
where software is installed by hand, is not the system's, nor are ``/opt`` and ``/var``."""

SERVICES = (
    "/var/run/nscd/socket",
    "/run/nscd/socket",
    "/var/lib/sss/pipes",
    "/run/systemd/journal",
    "/dev/log",
    "/run/dbus/system_bus_socket",
    "/var/run/dbus/system_bus_socket",
)
"""The local sockets of the system's own services (the user database, the journal, the system
bus), a connection to which is no input; one to any other may answer with anything."""

_LOCAL_SYSTEM = ("/usr/local",)

STACK = (
    "/cvmfs/sft.cern.ch",
    "/cvmfs/sft-nightlies.cern.ch",
    "/cvmfs/sw.hsf.org",
    "/cvmfs/sw-nightlies.hsf.org",
    "/cvmfs/geant4.cern.ch",
)
"""The CVMFS repositories the stack and its data come from: the LCG releases and nightlies, the
Key4hep ones and Geant4's data, all in directories of one version each, which the build and the
variables that name them (``G4LEDATA``) identify. Any other repository holds data like any other
filesystem: what a stage reads there is an undeclared input."""

_PACKAGE = re.compile(
    r"^/cvmfs/[^/]+/lcg/(?:nightlies/[^/]+/[^/]+|releases)/(?:[^/]+/)*?"
    r"(?P<name>[^/]+)/(?P<version>[^/]+)/(?:x86_64|aarch64)-[^/]+/"
)
"""A file of an LCG package, in a nightly or a release: the package's name and version."""

_REGISTRY = (".rootmap", ".components")
"""Files every package on the library path has read, whether its code runs or not: the registries
ROOT and Gaudi look a plugin up in."""

_MAKES = frozenset(
    (b"rename", b"renameat", b"renameat2", b"link", b"linkat", b"symlink", b"symlinkat", b"mkdir")
) | {b"mkdirat"}
"""The calls by which a program makes a path other than by opening it: the path is the last one
they name (``rename(from, to)``, ``symlinkat(target, dirfd, path)``)."""
_SENDS = frozenset((b"connect", b"sendto", b"sendmsg"))
"""The calls that name where a socket reaches: a connection, or a datagram sent without one."""
_CALLS = _MAKES | _SENDS | {b"open", b"openat", b"openat2", b"execve", b"execveat", b"socket"}
_RESULT = re.compile(
    rb"^(?P<ret>-?\d+|\?)(?:<(?P<fd>[^>\\]*(?:\\.[^>\\]*)*)>)?(?: (?P<errno>E[A-Z0-9]+)\b.*)?$"
)
"""What a call returned, as strace writes it after ``) = ``: a number, with the path of the
descriptor (``3</usr/lib64/libc.so.6>``), or ``-1`` and the error (``-1 ENOENT (No such ...)``), or
``?`` for a process killed in the call."""
_STRING = rb'"([^"\\]*(?:\\.[^"\\]*)*)"'
_DIRFD = re.compile(rb"^(?:AT_FDCWD|-?\d+)(?:<([^>\\]*(?:\\.[^>\\]*)*)>)?, " + _STRING)
_NAMED = re.compile(rb"(?:(?:AT_FDCWD|-?\d+)(?:<([^>\\]*(?:\\.[^>\\]*)*)>)?, )?" + _STRING)
_PATH = re.compile(rb"^" + _STRING)
_ESCAPE = re.compile(rb"\\(x[0-9a-fA-F]{2}|[0-7]{1,3}|.)")
_SIMPLE = {b"n": b"\n", b"t": b"\t", b"r": b"\r", b"v": b"\v", b"f": b"\f"}
_WRITES = re.compile(rb"\b(O_WRONLY|O_TRUNC|O_TMPFILE)\b")
"""An open whose file the program wrote, never read: write-only, emptied, or a new nameless file.
Read-write is not one (a database opened to be queried), nor is create without exclusive (an
existing file opened as it is): such a file is read, and one the run then changed has no
identity."""
_CREATES = (re.compile(rb"\bO_CREAT\b"), re.compile(rb"\bO_EXCL\b"))
"""An open that makes the file anew, or fails: a file of the run's own."""
_UNREADABLE = 5
"""How many lines a footprint keeps of those its traces held that could not be read."""
_INET = re.compile(rb"sa_family=(AF_INET6?)\b(?P<rest>.*)")
_UNIX = re.compile(
    rb'sa_family=AF_UNIX, sun_path=(?P<abstract>@?)"(?P<path>[^"\\]*(?:\\.[^"\\]*)*)"'
)
_SOCKET = re.compile(rb"^(?P<family>AF_INET6?)\b")
_LOADED = re.compile(rb"^\s*\d+:\s+calling init: (?P<path>.+?)\s*$")
MISSING = "missing"
"""How a stage failed to open a path that was not there."""
DENIED = "denied"
"""How a stage failed to open a path it was refused."""
_MISSING = (b"ENOENT", b"ENOTDIR")
_DENIED = (b"EACCES", b"EPERM")
_SPECIAL = "special"
"""The identity of a device, a pipe or a socket: none that says what it gives."""
_LISTED = "names:"
"""What the identity of a directory starts with."""
_DELETED = " (deleted)"
"""What the kernel adds to the path of a descriptor whose file was removed."""


def _unescape(text: bytes) -> str:
    """A string as strace writes it (C escapes), as a path."""
    if b"\\" not in text:
        return os.fsdecode(text)

    def one(match: re.Match[bytes]) -> bytes:
        code = match.group(1)
        if code[:1] == b"x":
            return bytes([int(code[1:], 16)])
        if code[:1].isdigit():
            return bytes([int(code, 8) & 0xFF])
        return _SIMPLE.get(code, code)

    return os.fsdecode(_ESCAPE.sub(one, text))


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def identity(path: str, since: int | None = None) -> str | None:
    """What identifies ``path`` now, ``None`` if it cannot be read or changed at or after the time
    ``since`` (in nanoseconds): then what it is now need not be what a run that began at that time
    read. ``names:`` and the SHA-256 of the names in a directory (but ``__pycache__``, which Python
    makes as it imports, whoever runs it), ``cvmfs:`` and the content key
    CVMFS keeps of a file on it (read without the file), ``sha256:`` and the SHA-256 of any other
    file, read once per version of it (:func:`k4opt.hashing.sha256_of`), or each time while it
    changed just now, and ``special`` for a device, a pipe or a socket, which may give other data
    every time it is read and so never vouches for what was read (:meth:`Footprint.changes`)."""
    try:
        status = os.stat(path)
        if since is not None and max(status.st_mtime_ns, status.st_ctime_ns) >= since:
            return None
        if stat.S_ISDIR(status.st_mode):  # what Python makes of its own accord is no name
            names = "\0".join(sorted(set(os.listdir(path)) - {"__pycache__"}))
            return _LISTED + hashlib.sha256(os.fsencode(names)).hexdigest()
        if not stat.S_ISREG(status.st_mode):
            return _SPECIAL
        if path.startswith("/cvmfs/"):
            for key in ("user.hash", "user.chunk_list"):
                try:
                    value = os.getxattr(path, key)
                except OSError:
                    continue
                return "cvmfs:" + hashlib.sha256(value).hexdigest()
        stamp = (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)
        if _recent(stamp, time.time_ns()):  # its stat cannot tell it from a rewrite in that tick
            return "sha256:" + read_sha256(path)
        return "sha256:" + sha256_of(path, stamp)
    except OSError:
        return None


@dataclass(frozen=True, slots=True)
class Footprint:
    """What one observed run of a program read, found missing and started.

    ``level`` is how it was observed. ``read`` holds each file and directory it opened, by its
    path, with its :func:`identity`; ``missing`` each path it failed to find, ``denied`` each it was
    refused, ``ran`` each program it started and ``connected`` what it reached outside this
    machine's own services: each network address it connected to, each network socket it made
    (``AF_INET socket``: a datagram needs no connection) and each other local socket it connected
    to (``unix:<path>``), whose answers may come from anywhere.
    ``unread`` counts the lines of its traces that could not be read, of which ``unreadable``
    holds the first few, or why it is incomplete: a footprint with any is incomplete, and so is
    one at another level than ``files``. ``made`` holds the paths it wrote or made outside what
    the stage was given or made in its own directory: what a path another stage looked for in
    vain may have become since (a cache), by the study's own doing."""

    level: Observation
    read: Mapping[str, str | None] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    ran: tuple[str, ...] = ()
    connected: tuple[str, ...] = ()
    unread: int = 0
    unreadable: tuple[str, ...] = ()
    made: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether it says every file the run opened."""
        return self.level == "files" and self.unread == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "level": self.level,
            "read": dict(sorted(self.read.items())),
            "missing": list(self.missing),
            "denied": list(self.denied),
            "ran": list(self.ran),
            "connected": list(self.connected),
            "unread": self.unread,
            "unreadable": list(self.unreadable),
            "made": list(self.made),
        }

    @classmethod
    def from_dict(cls, data: object) -> Footprint:
        """The footprint :meth:`to_dict` wrote; raises :class:`SoftwareError`."""
        keys = {"schema", "level", "read", "missing", "denied", "ran", "connected", "unread"}
        keys |= {"unreadable", "made"}
        if not isinstance(data, Mapping) or set(data) != keys or data["schema"] != SCHEMA:
            raise SoftwareError(f"no footprint of {SCHEMA}")
        read, unread = data["read"], data["unread"]
        if data["level"] not in ("files", "libraries") or not isinstance(read, Mapping):
            raise SoftwareError("a footprint's level or files are not what a footprint holds")
        if not all(isinstance(v, str) or v is None for v in read.values()):
            raise SoftwareError("a footprint's files are not identities by path")
        names = ("missing", "denied", "ran", "connected", "unreadable", "made")
        lists = [data[key] for key in names]
        if not all(isinstance(x, list) and all(isinstance(i, str) for i in x) for x in lists):
            raise SoftwareError("a footprint's paths are not lists of paths")
        if isinstance(unread, bool) or not isinstance(unread, int) or unread < 0:
            raise SoftwareError(f"a footprint's unread lines are {unread!r}, no count")
        missing, denied, ran, connected, unreadable, made = (
            tuple(cast(list[str], x)) for x in lists
        )
        return cls(
            data["level"], dict(read), missing, denied, ran, connected, unread, unreadable, made
        )

    def changes(self, at_most: int = 3) -> list[str]:
        """What no longer holds of it, each said for a person (the first ``at_most`` found);
        empty if every file it read is the same, every directory holds the same names, nothing
        it failed to find exists and nothing it was refused can be read now. A file that could
        not be identified when the stage ended, or changed while it ran, does not hold: what it
        read is not known."""
        found: list[str] = []
        for path, then in self.read.items():
            if then is None or then == _SPECIAL:  # a pipe gives what it is sent, each time
                found.append(f"{path} could not be identified")
            elif identity(path) != then:
                found.append(f"{path} changed" if os.path.lexists(path) else f"{path} went")
            if len(found) >= at_most:
                return found
        for path in self.missing:
            if os.path.lexists(path):
                found.append(f"{path} appeared")
                if len(found) >= at_most:
                    return found
        for path in self.denied:
            if os.access(path, os.R_OK):
                found.append(f"{path} can be read now")
                if len(found) >= at_most:
                    return found
        return found

    def looked(self, local: Sequence[Local]) -> dict[str, str]:
        """The paths it looked for in vain, of those outside the stack, the system and what the
        software identifies (:func:`covered`), each with how: :data:`MISSING` or :data:`DENIED`
        (refused). What a session and a resume watch, as inputs (:func:`findable`)."""
        failed = dict.fromkeys(self.denied, DENIED) | dict.fromkeys(self.missing, MISSING)
        return {
            path: how
            for path, how in failed.items()
            if _undeclared(path) and not covered(path, local)
        }

    def appeared(self, local: Sequence[Local], made: Collection[str] = ()) -> list[str]:
        """The paths it looked for in vain that a stage would find now (:meth:`looked`,
        :func:`findable`), but those in ``made``: what the stages of a study made themselves. What a
        resume refuses, as an input that changed."""
        return [path for path in findable(self.looked(local)) if path not in made]

    def summary(self, local: Sequence[Local]) -> dict[str, Json]:
        """What a stage's record keeps of it: how it was observed (``level``), whether that is
        complete, the ``packages`` of the stack whose files it used (an LCG package's name and
        version; never one it only listed, or whose registries alone it read), the ``local``
        entries it read a file from (their paths, of those in ``local``), the ``undeclared`` files
        and directories it read that neither the stack, the system, the software (:func:`covered`)
        nor the problem identify (with their identity), and what it reached of the ``network``
        (:attr:`connected`)."""
        # what it read, not what it listed: Python lists every directory of its path, used or not
        files = [p for p, then in self.read.items() if not (then or "").startswith(_LISTED)]
        packages: dict[str, str] = {}
        for path in files:
            found = _PACKAGE.match(path)
            if found is not None and not path.endswith(_REGISTRY):
                packages[found["name"]] = found["version"]
        roots = _roots(local)
        used = sorted({roots[root] for path in files for root in roots if _under(path, [root])})
        undeclared: dict[str, Json] = {}
        if self.level == "files":
            undeclared = {
                path: then
                for path, then in sorted(self.read.items())
                if _undeclared(path) and not covered(path, local)
            }
        return {
            "level": self.level,
            "complete": self.complete,
            "packages": cast(Json, dict(sorted(packages.items()))),
            "local": cast(Json, used),
            "undeclared": cast(Json, undeclared),
            "network": cast(Json, list(self.connected)),
        }


def _undeclared(path: str) -> bool:
    """Whether ``path`` is outside what the stack and the system identify."""
    if _under(path, STACK):
        return False
    return not _under(path, SYSTEM) or _under(path, _LOCAL_SYSTEM)


def covered(path: str, local: Sequence[Local]) -> bool:
    """Whether the software identifies the file ``path`` by its content: as one of what can be
    loaded through one of the ``local`` entries, by the entry's role
    (:mod:`k4opt.software.overlays`). A data file beside the libraries of a local build is not, nor
    is a directory: what they hold is watched as any undeclared input. A Python package's data is
    taken as not identified either, which is stricter than the software."""
    for entry in local:
        role = "python" if entry.variable == SITE else ROLES.get(entry.variable)
        for root in {entry.path.rstrip("/"), os.path.realpath(entry.path)}:
            if path == root:
                return os.path.isfile(root)
            if not path.startswith(f"{root}/") or os.path.isdir(path):
                continue
            parts = path[len(root) + 1 :].split("/")
            if role == "headers":
                return True
            if role == "libraries" and len(parts) == 1 and _LIBRARY.search(parts[0]):
                return True
            if role == "python" and (
                "__pycache__" in parts or parts[-1].endswith(_MODULES) or parts[0].endswith(_BESIDE)
            ):
                return True
    return False


def _roots(local: Sequence[Local]) -> dict[str, str]:
    """Where each local entry is, its links followed too, by that place: the entry's path."""
    roots: dict[str, str] = {}
    for entry in local:
        roots[entry.path] = entry.path
        roots[os.path.realpath(entry.path)] = entry.path
    return roots


@dataclass(slots=True)
class _Seen:
    read: dict[str, None] = field(default_factory=dict)
    missing: dict[str, None] = field(default_factory=dict)
    denied: dict[str, None] = field(default_factory=dict)
    ran: dict[str, None] = field(default_factory=dict)
    connected: dict[str, None] = field(default_factory=dict)
    written: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)  # a path opened: the file it reached
    sockets: dict[str, None] = field(default_factory=dict)  # local sockets no one was at
    # files reached by a relative path from a directory the trace does not say
    unresolved: list[str] = field(default_factory=list)
    unread: int = 0
    unreadable: list[str] = field(default_factory=list)


def _named(given: str, base: str | None) -> str | None:
    """The path ``given`` to a call from the directory ``base``, as the system walks it: a
    ``..`` after a link is not the directory before it, so a path with one is kept as it is;
    ``None`` for a relative path from a directory the trace does not say."""
    if not os.path.isabs(given):
        if base is None or not os.path.isabs(base):
            return None
        given = os.path.join(base, given)
    return given if ".." in given.split("/") else os.path.normpath(given)


def _opened(seen: _Seen, call: bytes, args: bytes, match: re.Match[bytes]) -> None:
    found = _DIRFD.match(args) if call != b"open" else _PATH.match(args)
    if found is None:
        seen.unread += 1
        return
    if call == b"open":
        base, given = None, _unescape(found.group(1))
    else:
        base = None if found.group(1) is None else _unescape(found.group(1))
        given = _unescape(found.group(2))
    flags = args[found.end() :]
    if match["ret"] != b"?" and int(match["ret"]) >= 0:
        fd = match["fd"]
        if fd is None:
            seen.unread += 1
            return
        path = _unescape(fd).removesuffix(_DELETED)
        named = _named(given, base)
        if named is None:  # what it was opened by, and whether by a link: unknown, if it matters
            seen.unresolved.append(path)
        if _WRITES.search(flags) or all(flag.search(flags) for flag in _CREATES):
            seen.written.add(path)
            seen.written.add(path if named is None else named)
        else:
            seen.read[path] = None
            if named is not None and named != path:
                seen.aliases[named] = path
        return
    found_at = _named(given, base)  # where the system looked: a .. after a link is not undone
    if found_at is None:  # relative to a directory no one knows
        seen.unread += 1
        return
    path = found_at
    if match["errno"] in _MISSING:
        seen.missing[path] = None
    elif match["errno"] in _DENIED:
        seen.denied[path] = None
    elif match["errno"] == b"EEXIST":  # a probe that found it there: what is there decides
        seen.read[path] = None


def _started(seen: _Seen, args: bytes, match: re.Match[bytes]) -> None:
    named = _PATH.match(args) if not args.startswith(b"AT_FDCWD") else None
    if named is None:
        named = _DIRFD.match(args)
        given = None if named is None else _unescape(named.group(2))
    else:
        given = _unescape(named.group(1))
    if given is None or not os.path.isabs(given):
        seen.unread += 1  # started from a relative path: its directory is not said
        return
    path = _named(given, None) or given
    if match["ret"] == b"0" or match["ret"] == b"?":
        seen.ran[path] = None
        seen.read[path] = None
    elif match["errno"] in _MISSING:
        seen.missing[path] = None
    elif match["errno"] in _DENIED:
        seen.denied[path] = None


def _connected(seen: _Seen, args: bytes, match: re.Match[bytes]) -> None:
    local = _UNIX.search(args)
    if local is not None:
        path = _unescape(local["path"])
        if local["abstract"]:  # a name no file has: whoever takes it later answers
            seen.connected[f"unix:@{path}"] = None
            return
        if _under(path, SERVICES):
            return
        if match["errno"] not in _MISSING:  # connected, connecting or refused: it decides
            seen.connected[f"unix:{path}"] = None
        elif os.path.isabs(path):  # no one there: one that comes would be another input
            seen.sockets[path] = None  # missing wherever it is, under /run as anywhere
        else:  # relative to a directory the trace does not say
            seen.unread += 1
        return
    found = _INET.search(args)
    if found is None:
        return
    address = re.findall(rb'"([^"]*)"|htons\((\d+)\)', found["rest"])
    host = next((a.decode(errors="replace") for a, _ in address if a), "?")
    port = next((p.decode() for _, p in address if p), "?")
    seen.connected[f"{host}:{port}"] = None


def _made(seen: _Seen, args: bytes, match: re.Match[bytes]) -> None:
    """A path the program made without opening it is its own, as one it wrote. One named relative
    to a directory the trace does not say stays unknown, and so stays missing if it was looked
    for: a footprint that then never holds, which costs a run and never a wrong reuse."""
    if match["ret"] != b"0":
        return
    names = list(_NAMED.finditer(args))
    if not names:
        return
    named = names[-1]
    base = None if named.group(1) is None else _unescape(named.group(1))
    path = _named(_unescape(named.group(2)), base)
    if path is not None:
        seen.written.add(path)
        seen.written.add(os.path.normpath(path))


def _socket(seen: _Seen, args: bytes, match: re.Match[bytes]) -> None:
    """A network socket is network use, connected or not: a datagram needs no ``connect``. The
    programs of the stack make none; those they do make are local."""
    found = _SOCKET.match(args)
    if found is not None and match["ret"] not in (b"-1", b"?"):
        seen.connected[f"{found['family'].decode()} socket"] = None


def _lines(trace: Path, prefix: str) -> Iterator[bytes]:
    for path in sorted(trace.glob(f"{prefix}.*")):
        with path.open("rb") as handle:
            yield from handle


def _traced(trace: Path) -> _Seen:
    """What the strace traces in ``trace`` say."""
    seen = _Seen()
    for line in _lines(trace, "t"):
        line = line.rstrip(b"\n")
        if not line or line.startswith((b"+++", b"---")):
            continue
        before = seen.unread
        _line(seen, line)
        if seen.unread > before and len(seen.unreadable) < _UNREADABLE:
            seen.unreadable.append(line[:500].decode(errors="replace"))
    return seen


def _line(seen: _Seen, line: bytes) -> None:
    """What one line of a trace says, into ``seen``."""
    # split at the last " = ", after which strace writes what the call returned, at column 40 at
    # least, with spaces before it for a short call; a path with that in it fails the result,
    # never misleads
    head, said, result = line.rpartition(b" = ")
    head = head.rstrip(b" ")
    call, _, args = head[:-1].partition(b"(")
    match = _RESULT.match(result) if said and head.endswith(b")") and call in _CALLS else None
    if match is None:
        seen.unread += 1
        return
    if call.startswith(b"open"):
        _opened(seen, call, args, match)
    elif call.startswith(b"execve"):
        _started(seen, args, match)
    elif call == b"socket":
        _socket(seen, args, match)
    elif call in _MAKES:
        _made(seen, args, match)
    else:  # a connection, or a datagram that says where it goes
        _connected(seen, args, match)


def _loaded(trace: Path) -> _Seen:
    """What the loader's traces in ``trace`` say: the libraries it loaded."""
    seen = _Seen()
    for line in _lines(trace, "ld"):
        found = _LOADED.match(line)
        if found is not None:
            seen.read[os.path.realpath(os.fsdecode(found["path"]))] = None
    return seen


def footprint(
    trace: Path, level: Observation, own: Sequence[Path], since: int | None = None
) -> Footprint:
    """The footprint of the run whose raw traces are in ``trace``, observed at ``level``
    (``files`` or ``libraries``): what it read, found missing and started outside ``own`` (the
    stage's directory, the candidate's, and the files it was given or that the problem names) and
    :data:`IGNORED`, each file identified now, unless it changed at or after ``since``, when the
    run began (:func:`identity`). A file reached through a link is kept under the path it was
    opened by too, with no identity if that path leads elsewhere now or a link on its way changed
    since the run began. A path it wrote or made (by a rename, a link, a directory) is none of
    these, wherever it is. A path it failed to find stays missing though it exists now, unless it
    made it: another process may have, while it ran. Only ``__pycache__`` is never missing: Python
    checks what it finds there against the source, whoever wrote it.
    Traces in which no program started are incomplete, whatever they say: every run starts one,
    so they are what a tracer that died or a full disk left."""
    seen = _traced(trace) if level == "files" else _loaded(trace)
    if level == "files" and not seen.ran:
        seen.unread += 1
        seen.unreadable.append("no program started")
    skipped = [*IGNORED, *(os.fspath(path) for path in own)]
    skipped += [os.path.realpath(path) for path in own]

    def kept(paths: Iterable[str]) -> list[str]:
        return [p for p in paths if p not in seen.written and not _under(p, skipped)]

    # a relative path that led into what the stage made or was given is none of its inputs
    for path in kept(seen.unresolved):
        seen.unread += 1
        if len(seen.unreadable) < _UNREADABLE:
            seen.unreadable.append(f"opened by a path relative to an unknown directory: {path}")

    read = {path: identity(path, since) for path in kept(seen.read)}
    for path in kept(seen.ran):  # a program started through a link retargeted since: not known
        if path in read and not _steady(path, None, since):
            read[path] = None
    for name in kept(seen.aliases):  # /etc/mtab leads into /proc: not an input either
        reached = seen.aliases[name]
        if reached in seen.written or _under(reached, IGNORED) or name in read:
            continue
        read[name] = identity(name, since) if _steady(name, reached, since) else None
    absent = [path for path in kept(seen.missing) if "__pycache__" not in path.split("/")]
    absent += [path for path in seen.sockets if path not in seen.written and path not in absent]
    made = [path for path in sorted(seen.written) if not _under(path, skipped)]
    return Footprint(
        level,
        read,
        tuple(sorted(absent)),
        tuple(sorted(kept(seen.denied))),
        tuple(sorted(kept(seen.ran))),
        tuple(sorted(seen.connected)),
        seen.unread,
        tuple(seen.unreadable),
        tuple(made),
    )


def _steady(name: str, reached: str | None, since: int | None) -> bool:
    """Whether the path ``name`` still leads to the file ``reached`` (if it is known), and no link
    on its way changed at or after ``since``: one retargeted while the run ran, even if back
    since."""
    if reached is not None and os.path.realpath(name) != reached:
        return False
    if since is None:
        return True
    prefix = ""
    for part in name.split("/")[1:]:
        prefix += f"/{part}"
        try:
            status = os.lstat(prefix)
        except OSError:
            return False
        if stat.S_ISLNK(status.st_mode) and max(status.st_mtime_ns, status.st_ctime_ns) >= since:
            return False
    return True


def observe(
    run_trace: Path | None,
    level: Observation,
    *,
    directory: Path,
    own: Sequence[Path],
    local: Sequence[Local] = (),
    since: int | None = None,
) -> tuple[dict[str, Json], Path | None]:
    """What a stage's record says of how the run whose raw traces are in ``run_trace`` was
    observed (:meth:`Footprint.summary`, of the ``local`` entries, with the size and SHA-256 of
    its ``footprint``), and its footprint file: written into ``directory`` as :data:`FOOTPRINT`,
    the traces removed; ``since`` is when the run began (:func:`footprint`). Unobserved
    (``run_trace`` ``None``), the record says only its ``level``,
    ``none``; traces that cannot be read make no footprint, and the record says why (``problem``).
    Never raises for the traces: a stage runs whether or not it can be observed."""
    if run_trace is None or level == "none":
        return {"level": "none", "complete": False}, None
    try:
        found = footprint(run_trace, level, own, since)
    except OSError as exc:
        return {"level": level, "complete": False, "problem": f"traces unreadable: {exc}"}, None
    finally:
        shutil.rmtree(run_trace, ignore_errors=True)
    # ASCII: a path need not be text, and comes back as it was
    data = json.dumps(found.to_dict(), separators=(",", ":")).encode()
    compressed = gzip.compress(data, mtime=0)
    path = directory / FOOTPRINT
    summary = found.summary(local)
    try:
        path.write_bytes(compressed)
    except OSError as exc:
        return {**summary, "complete": False, "problem": f"footprint not kept: {exc}"}, None
    sha256 = hashlib.sha256(compressed).hexdigest()
    return {**summary, "footprint": {"bytes": len(compressed), "sha256": sha256}}, path


def load(path: Path) -> Footprint:
    """The footprint in the file ``path`` (:data:`FOOTPRINT`); raises :class:`SoftwareError` if it
    cannot be read or is none."""
    try:
        data = json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, ValueError, EOFError) as exc:
        raise SoftwareError(f"{path}: no footprint ({exc})") from exc
    return Footprint.from_dict(data)


def findable(looked: Mapping[str, str]) -> list[str]:
    """Each of the paths ``looked`` for in vain (path: how, :meth:`Footprint.looked`) that a stage
    would find now: one that was missing exists, one that was refused can be read."""
    return [
        path
        for path, how in looked.items()
        if (os.access(path, os.R_OK) if how == DENIED else os.path.lexists(path))
    ]


def changed_inputs(undeclared: Mapping[str, Json]) -> list[str]:
    """Each of ``undeclared`` (path: identity, as :meth:`Footprint.summary` keeps them) that is
    not what it was, by its path."""
    return [path for path, then in undeclared.items() if identity(path) != then]


__all__ = [
    "DENIED",
    "FOOTPRINT",
    "IGNORED",
    "MISSING",
    "SCHEMA",
    "SERVICES",
    "STACK",
    "SYSTEM",
    "Footprint",
    "changed_inputs",
    "covered",
    "findable",
    "footprint",
    "identity",
    "load",
    "observe",
]
