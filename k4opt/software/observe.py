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
and a directory by the names in it (what a scan of it found). What the stage made, what it was
given and what it wrote, and the kernel's own files (:data:`IGNORED`), are not part of it. A
footprint *holds* (:meth:`Footprint.changes`) while each file it read is the same, each directory
holds the same names and no path it failed to find exists: so far as opening files goes, the stage
would find what it found. A file that could not be identified never holds: what is not known is
taken to differ.

What a footprint cannot show: what was decided from a file's size or time, or from whether it
exists, without opening it (``stat``), an import that failed in Python, which answers from a
listing it keeps, and a file the stage read and also wrote, which counts as its own. With the same
software, the first two can only concern files outside the stack and the local builds.

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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from k4opt.frozen import Json
from k4opt.hashing import read_sha256, sha256_of
from k4opt.program import Observation
from k4opt.software.errors import SoftwareError
from k4opt.software.overlays import Local, _recent

FOOTPRINT = "footprint.json.gz"
"""The file in a stage's directory that holds its footprint."""

SCHEMA = "k4opt.footprint/1"

IGNORED = ("/proc", "/sys", "/dev", "/run", "/var/run", "/var/lib/sss")
"""Where what a stage reads is the kernel's or the session's, never an input: the processes, the
devices, and the caches of the user database, which change as anyone logs in."""

SYSTEM = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/var")
"""Where the system keeps what it installed: identified, roughly, by the system
(:class:`~k4opt.software.System`), never listed among a stage's undeclared inputs; ``/usr/local``,
where software is installed by hand, is not the system's."""

_LOCAL_SYSTEM = ("/usr/local",)

_PACKAGE = re.compile(
    r"^/cvmfs/[^/]+/lcg/(?:nightlies/[^/]+/[^/]+|releases)/(?:[^/]+/)*?"
    r"(?P<name>[^/]+)/(?P<version>[^/]+)/(?:x86_64|aarch64)-[^/]+/"
)
"""A file of an LCG package, in a nightly or a release: the package's name and version."""

_REGISTRY = (".rootmap", ".components")
"""Files every package on the library path has read, whether its code runs or not: the registries
ROOT and Gaudi look a plugin up in."""

_CALLS = frozenset((b"open", b"openat", b"openat2", b"execve", b"execveat", b"connect"))
_RESULT = re.compile(
    rb"^(?P<ret>-?\d+|\?)(?:<(?P<fd>[^>\\]*(?:\\.[^>\\]*)*)>)?(?: (?P<errno>E[A-Z0-9]+)\b.*)?$"
)
"""What a call returned, as strace writes it after ``) = ``: a number, with the path of the
descriptor (``3</usr/lib64/libc.so.6>``), or ``-1`` and the error (``-1 ENOENT (No such ...)``), or
``?`` for a process killed in the call."""
_STRING = rb'"([^"\\]*(?:\\.[^"\\]*)*)"'
_DIRFD = re.compile(rb"^(?:AT_FDCWD|-?\d+)(?:<([^>\\]*(?:\\.[^>\\]*)*)>)?, " + _STRING)
_PATH = re.compile(rb"^" + _STRING)
_ESCAPE = re.compile(rb"\\(x[0-9a-fA-F]{2}|[0-7]{1,3}|.)")
_SIMPLE = {b"n": b"\n", b"t": b"\t", b"r": b"\r", b"v": b"\v", b"f": b"\f"}
_WRITES = re.compile(rb"\b(O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND|O_TMPFILE)\b")
_INET = re.compile(rb"sa_family=(AF_INET6?)\b(?P<rest>.*)")
_LOADED = re.compile(rb"^\s*\d+:\s+calling init: (?P<path>.+?)\s*$")
_MISSING = (b"ENOENT", b"ENOTDIR")
_DENIED = (b"EACCES", b"EPERM")
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


def identity(path: str) -> str | None:
    """What identifies ``path`` now, ``None`` if it cannot be read: ``names:`` and the SHA-256 of
    the names in a directory, ``cvmfs:`` and the content key CVMFS keeps of a file on it (read
    without the file), ``sha256:`` and the SHA-256 of any other file, read once per version of it
    (:func:`k4opt.hashing.sha256_of`), or each time while it changed just now, and ``special`` for
    a device, a pipe or a socket."""
    try:
        status = os.stat(path)
        if stat.S_ISDIR(status.st_mode):
            names = "\0".join(sorted(os.listdir(path)))
            return _LISTED + hashlib.sha256(os.fsencode(names)).hexdigest()
        if not stat.S_ISREG(status.st_mode):
            return "special"
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
    refused, ``ran`` each program it started and ``connected`` each network address it connected to.
    ``unread`` counts the lines of its traces that could not be read: a footprint with any is
    incomplete, and so is one at another level than ``files``."""

    level: Observation
    read: Mapping[str, str | None] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    ran: tuple[str, ...] = ()
    connected: tuple[str, ...] = ()
    unread: int = 0

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
        }

    @classmethod
    def from_dict(cls, data: object) -> Footprint:
        """The footprint :meth:`to_dict` wrote; raises :class:`SoftwareError`."""
        keys = {"schema", "level", "read", "missing", "denied", "ran", "connected", "unread"}
        if not isinstance(data, Mapping) or set(data) != keys or data["schema"] != SCHEMA:
            raise SoftwareError(f"no footprint of {SCHEMA}")
        read, unread = data["read"], data["unread"]
        if data["level"] not in ("files", "libraries") or not isinstance(read, Mapping):
            raise SoftwareError("a footprint's level or files are not what a footprint holds")
        if not all(isinstance(v, str) or v is None for v in read.values()):
            raise SoftwareError("a footprint's files are not identities by path")
        lists = [data[key] for key in ("missing", "denied", "ran", "connected")]
        if not all(isinstance(x, list) and all(isinstance(i, str) for i in x) for x in lists):
            raise SoftwareError("a footprint's paths are not lists of paths")
        if isinstance(unread, bool) or not isinstance(unread, int) or unread < 0:
            raise SoftwareError(f"a footprint's unread lines are {unread!r}, no count")
        missing, denied, ran, connected = (tuple(cast(list[str], x)) for x in lists)
        return cls(data["level"], dict(read), missing, denied, ran, connected, unread)

    def changes(self, at_most: int = 3) -> list[str]:
        """What no longer holds of it, each said for a person (the first ``at_most`` found);
        empty if every file it read is the same, every directory holds the same names, nothing
        it failed to find exists and nothing it was refused can be read now. A file that could
        not be identified when the stage ended does not hold: what it was is not known."""
        found: list[str] = []
        for path, then in self.read.items():
            if then is None:
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

    def summary(self, local: Sequence[Local]) -> dict[str, Json]:
        """What a stage's record keeps of it: how it was observed (``level``), whether that is
        complete, the ``packages`` of the stack whose files it used (an LCG package's name and
        version; never one it only listed, or whose registries alone it read), the ``local``
        entries it read a file from (their paths, of those in ``local``), the ``undeclared`` files
        and directories it read that neither the stack, the system, a local build nor the problem
        identify (with their identity), and the ``network`` addresses it connected to."""
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
                if _undeclared(path) and not _under(path, roots)
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
    if path.startswith("/cvmfs/"):
        return False
    return not _under(path, SYSTEM) or _under(path, _LOCAL_SYSTEM)


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
    unread: int = 0


def _opened(seen: _Seen, call: bytes, args: bytes, match: re.Match[bytes]) -> None:
    named = _DIRFD.match(args) if call != b"open" else _PATH.match(args)
    if named is None:
        seen.unread += 1
        return
    if call == b"open":
        base, given = None, _unescape(named.group(1))
    else:
        base = None if named.group(1) is None else _unescape(named.group(1))
        given = _unescape(named.group(2))
    flags = args[named.end() :]
    if match["ret"] != b"?" and int(match["ret"]) >= 0:
        fd = match["fd"]
        if fd is None:
            seen.unread += 1
            return
        path = _unescape(fd).removesuffix(_DELETED)
        if _WRITES.search(flags):
            seen.written.add(path)
        else:
            seen.read[path] = None
        return
    if os.path.isabs(given):
        path = os.path.normpath(given)
    elif base is not None and os.path.isabs(base):
        path = os.path.normpath(os.path.join(base, given))
    else:  # relative to a directory no one knows
        seen.unread += 1
        return
    if match["errno"] in _MISSING:
        seen.missing[path] = None
    elif match["errno"] in _DENIED:
        seen.denied[path] = None


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
    path = os.path.normpath(given)
    if match["ret"] == b"0" or match["ret"] == b"?":
        seen.ran[path] = None
        seen.read[path] = None
    elif match["errno"] in _MISSING:
        seen.missing[path] = None
    elif match["errno"] in _DENIED:
        seen.denied[path] = None


def _connected(seen: _Seen, args: bytes) -> None:
    found = _INET.search(args)
    if found is None:  # a local socket (the user database, a display): no network
        return
    address = re.findall(rb'"([^"]*)"|htons\((\d+)\)', found["rest"])
    host = next((a.decode(errors="replace") for a, _ in address if a), "?")
    port = next((p.decode() for _, p in address if p), "?")
    seen.connected[f"{host}:{port}"] = None


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
        # split at the last ") = ": a path with that in it fails the result, never misleads
        head, said, result = line.rpartition(b") = ")
        call, _, args = head.partition(b"(")
        match = _RESULT.match(result) if said and call in _CALLS else None
        if match is None:
            seen.unread += 1
            continue
        if call.startswith(b"open"):
            _opened(seen, call, args, match)
        elif call.startswith(b"execve"):
            _started(seen, args, match)
        else:
            _connected(seen, args)
    return seen


def _loaded(trace: Path) -> _Seen:
    """What the loader's traces in ``trace`` say: the libraries it loaded."""
    seen = _Seen()
    for line in _lines(trace, "ld"):
        found = _LOADED.match(line)
        if found is not None:
            seen.read[os.path.realpath(os.fsdecode(found["path"]))] = None
    return seen


def footprint(trace: Path, level: Observation, own: Sequence[Path]) -> Footprint:
    """The footprint of the run whose raw traces are in ``trace``, observed at ``level``
    (``files`` or ``libraries``): what it read, found missing and started outside ``own`` (the
    stage's directory, the candidate's, and the files it was given or that the problem names) and
    :data:`IGNORED`, each file identified now. A file it wrote is none of these, wherever it is.
    Traces in which no program started are incomplete, whatever they say: every run starts one,
    so they are what a tracer that died or a full disk left."""
    seen = _traced(trace) if level == "files" else _loaded(trace)
    if level == "files" and not seen.ran:
        seen.unread += 1
    skipped = [*IGNORED, *(os.fspath(path) for path in own)]
    skipped += [os.path.realpath(path) for path in own]

    def kept(paths: Iterable[str]) -> list[str]:
        return [p for p in paths if p not in seen.written and not _under(p, skipped)]

    return Footprint(
        level,
        {path: identity(path) for path in kept(seen.read)},
        tuple(sorted(kept(seen.missing))),
        tuple(sorted(kept(seen.denied))),
        tuple(sorted(kept(seen.ran))),
        tuple(sorted(seen.connected)),
        seen.unread,
    )


def observe(
    run_trace: Path | None,
    level: Observation,
    *,
    directory: Path,
    own: Sequence[Path],
    local: Sequence[Local] = (),
) -> tuple[dict[str, Json], Path | None]:
    """What a stage's record says of how the run whose raw traces are in ``run_trace`` was
    observed (:meth:`Footprint.summary`, of the ``local`` entries, with the size and SHA-256 of
    its ``footprint``), and its footprint file: written into ``directory`` as :data:`FOOTPRINT`,
    the traces removed. Unobserved (``run_trace`` ``None``), the record says only its ``level``,
    ``none``; traces that cannot be read make no footprint, and the record says why (``problem``).
    Never raises for the traces: a stage runs whether or not it can be observed."""
    if run_trace is None or level == "none":
        return {"level": "none", "complete": False}, None
    try:
        found = footprint(run_trace, level, own)
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


def changed_inputs(undeclared: Mapping[str, Json]) -> list[str]:
    """Each of ``undeclared`` (path: identity, as :meth:`Footprint.summary` keeps them) that is
    not what it was, by its path."""
    return [path for path, then in undeclared.items() if identity(path) != then]


__all__ = [
    "FOOTPRINT",
    "IGNORED",
    "SCHEMA",
    "SYSTEM",
    "Footprint",
    "changed_inputs",
    "footprint",
    "identity",
    "load",
    "observe",
]
