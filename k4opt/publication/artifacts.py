"""What of a study may be published beside its snapshot, and its public form.

The study keeps privately what a rerun needs (:mod:`k4opt.software.keep`). :func:`offer` says what
of it is offered for publication: only what is known to have been built or used, never what merely
lies beside it, in two groups the author answers for separately:

- **code** that ran: the edits of a build's checkout against the revision the nightly built, a local
  directory's files a stage read, changed job files, the geometry's and the evaluators' own files;
- **inputs** the stages read, up to :data:`BUDGET` in all.

A checkout's patch is offered file by file (:func:`sections`). An untracked file is offered if a
stage read it or the build installed it; a directory's edits if a stage read them. A binary
(``GIT binary patch``) is held back unless named. Anything over the budget, or that a person left
out by name (``exclude``), is not published, and said; :func:`scan` looks through what would be
published for secrets, whose findings block publishing (:meth:`Offer.release`).

An :class:`Offer` is what the author reviews; a :class:`Release`, made from it, is what was
approved: :func:`~k4opt.publication.publish` publishes exactly that list.
:func:`public_form` makes the study's :class:`~k4opt.publication.SoftwareSheet`, always published.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import k4opt as k4opt_package
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.optimization import StudyResults
from k4opt.publication.errors import PublicationError
from k4opt.publication.sheet import (
    Access,
    Group,
    PublishedArtifact,
    PublishedFile,
    PublishedRequirement,
    SoftwareSheet,
    State,
    artifact_path,
)
from k4opt.software import SoftwareError
from k4opt.software.keep import Store
from k4opt.software.readiness import (
    ALL_AVAILABLE,
    BLOCKED,
    INCOMPLETE,
    Readiness,
    Requirement,
    requirements,
)

_K4OPT = os.path.dirname(os.path.abspath(k4opt_package.__file__))
BUDGET = 10_000_000
"""How many bytes of code, and of inputs, are published unless ``--include-large`` says more."""
NEEDS_ACCESS = "Not rerun · needs access"


@dataclass(frozen=True, slots=True)
class Section:
    """One file's part of a git patch: its ``name``, its text, and the lines it adds and removes
    (``binary``: a ``GIT binary patch``, of no lines)."""

    name: str
    data: bytes
    binary: bool
    added: int
    removed: int


_HEADER = re.compile(rb"^diff --git ", re.MULTILINE)
_ESCAPES = {b"a": 7, b"b": 8, b"t": 9, b"n": 10, b"v": 11, b"f": 12, b"r": 13}


def sections(patch: bytes) -> list[Section]:
    """The files of ``patch`` (a git diff of this form: no renames, so each header names one file
    twice; a line of a file's text begins with ``+``, ``-`` or a space, never a header)."""
    starts = [match.start() for match in _HEADER.finditer(patch)]
    found = []
    for begin, end in zip(starts, [*starts[1:], len(patch)], strict=False):
        text = patch[begin:end]
        name = _name(text[len(b"diff --git ") : text.index(b"\n")])
        binary = b"\nGIT binary patch\n" in text or b"\nBinary files " in text
        added = removed = 0
        if not binary:
            body = text[text.index(b"\n@@") :] if b"\n@@" in text else b""
            for line in body.split(b"\n"):
                added += line[:1] == b"+"
                removed += line[:1] == b"-"
        found.append(Section(name, text, binary, added, removed))
    return found


def _name(header: bytes) -> str:
    """The file a patch's header (``a/NAME b/NAME``, either quoted as git quotes it) names."""
    if header.startswith(b'"'):
        return _unquote(header)[2:].decode(errors="surrogateescape")
    return header[2 : (len(header) - 5) // 2 + 2].decode(errors="surrogateescape")


def _unquote(text: bytes) -> bytes:
    """The first C-style quoted string of ``text``, as git writes an unusual file name."""
    found = bytearray()
    index = 1
    while text[index : index + 1] != b'"':
        char = text[index : index + 1]
        if char != b"\\":
            found += char
            index += 1
        elif text[index + 1 : index + 2].isdigit():
            found.append(int(text[index + 1 : index + 4], 8))
            index += 4
        else:
            escaped = text[index + 1 : index + 2]
            found.append(_ESCAPES.get(escaped, escaped[0]))
            index += 2
    return bytes(found)


@dataclass(frozen=True, slots=True)
class Piece:
    """What could be published: a file, or (``base`` not ``None``) one file's part of a patch, of
    ``owner``, in a ``group``; which ``requirements`` it answers, the ``stages`` that used it, and
    the lines a change adds and removes, if counted."""

    group: Group
    owner: str
    name: str
    data: bytes
    requirements: tuple[str, ...]
    stages: tuple[str, ...]
    base: str | None = None
    added: int | None = None
    removed: int | None = None

    @property
    def label(self) -> str:
        return f"{self.owner}/{self.name}" if self.owner else self.name

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def is_patch(self) -> bool:
        return self.base is not None

    def matches(self, name: str) -> bool:
        """Whether ``name`` names it: its owner, its path, its label or its own file name."""
        return name in (self.owner, self.name, self.label, os.path.basename(self.name))


@dataclass(frozen=True, slots=True)
class Held:
    """A part of what the study kept that is not offered unless named, and why (``why``);
    ``needed`` if a rerun may need it. Naming it (``include``) offers it as ``piece``."""

    group: Group
    owner: str
    name: str
    why: str
    requirement: str
    needed: bool
    piece: Piece

    @property
    def label(self) -> str:
        return self.piece.label


@dataclass(frozen=True, slots=True)
class Finding:
    """A secret :func:`scan` saw: ``what`` it looks like, in which file (``label``), at ``line``."""

    label: str
    line: int
    what: str


# what :func:`scan` looks for, best effort: enough to stop an accident, no substitute for looking
_SECRETS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("a private key", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")),
    ("an AWS access key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{22,})\b")),
    ("a GitLab token", re.compile(rb"\bglpat-[\w-]{20,}\b")),
    ("a Slack token", re.compile(rb"\bxox[abprs]-[\w-]{10,}\b")),
    ("a Google API key", re.compile(rb"\bAIza[\w-]{35}\b")),
    (
        "a URL with a password",
        re.compile(rb"\b[a-z][a-z0-9+.-]{0,31}://[^/\s:@'\"${}]{1,256}:[^/\s@'\"${}]{1,256}@"),
    ),
    (
        "a secret assigned in the open",
        re.compile(
            rb"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
            rb"\b\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']"
        ),
    ),
)
_SECRET_NAMES = re.compile(
    r"(?:^|/)(?:\.netrc|\.pgpass|\.htpasswd|\.env|id_(?:rsa|dsa|ecdsa|ed25519)|credentials)$"
    r"|\.(?:pem|p12|pfx|key)$"
)


def scan(pieces: Iterable[Piece]) -> list[Finding]:
    """What of ``pieces`` looks like a secret, the first of each kind per file."""
    findings = []
    for piece in pieces:
        if _SECRET_NAMES.search(piece.name):
            findings.append(Finding(piece.label, 1, "a file of credentials, by its name"))
        for what, pattern in _SECRETS:
            match = pattern.search(piece.data)
            if match is not None:
                findings.append(Finding(piece.label, _line(piece, match.start()), what))
    return findings


_HUNK = re.compile(rb"^@@ -\d+(?:,\d+)? \+(\d+)")


def _line(piece: Piece, offset: int) -> int:
    """The line of the file ``offset`` of ``piece`` falls in; of the changed file, for a patch."""
    if not piece.is_patch:
        return piece.data.count(b"\n", 0, offset) + 1
    line = 1
    for raw in piece.data[: offset + 1].split(b"\n"):
        hunk = _HUNK.match(raw)
        if hunk is not None:
            line = int(hunk[1]) - 1
        elif raw[:1] in (b"+", b" "):
            line += 1
    return max(line, 1)


@dataclass(frozen=True, slots=True)
class Offer:
    """What may be published of a study, for its author to review.

    ``code`` and ``inputs`` are offered; ``large`` what passes the :data:`BUDGET` of its group
    (published only ``include_large``); ``held`` what is listed and not offered; ``excluded`` what
    the author left out by name; ``findings`` what :func:`scan` saw in what would be published. A
    study read from its JSON form offers nothing (``readiness`` is ``None``)."""

    readiness: Readiness | None
    code: tuple[Piece, ...] = ()
    inputs: tuple[Piece, ...] = ()
    large: tuple[Piece, ...] = ()
    held: tuple[Held, ...] = ()
    excluded: tuple[Piece, ...] = ()
    findings: tuple[Finding, ...] = ()
    include_large: bool = False

    def release(self, *, code: bool = False, inputs: bool | Collection[str] = False) -> Release:
        """What the author approved: all of ``code`` or none, and of ``inputs`` all (``True``), none
        or the ones named. Raises :class:`PublicationError` if a scan found a secret, or a name
        names no input offered."""
        if self.findings:
            raise PublicationError(
                f"{len(self.findings)} possible secrets in what would be published: "
                "remove or exclude those files first"
            )
        large = [p for p in self.large if self.include_large]
        all_inputs = [*self.inputs, *(p for p in large if p.group == "inputs")]
        chosen: list[Piece]
        if isinstance(inputs, bool):
            chosen = all_inputs if inputs else []
        else:
            chosen = [p for p in all_inputs if any(p.matches(name) for name in inputs)]
            unknown = [n for n in inputs if not any(p.matches(n) for p in all_inputs)]
            if unknown:
                raise PublicationError(
                    f"no input offered is named {', '.join(unknown)}; a large one takes "
                    "--include-large as well"
                )
        pieces = [*chosen]
        if code:
            pieces = [*self.code, *(p for p in large if p.group == "code"), *pieces]
        return Release(self, tuple(pieces))


@dataclass(frozen=True, slots=True)
class Release:
    """The ``pieces`` of an ``offer`` the author approved, and no other."""

    offer: Offer
    pieces: tuple[Piece, ...] = ()


def offer(
    results: StudyResults,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    include_large: bool = False,
) -> Offer:
    """What of ``results`` may be published (see the module): ``include`` names what is otherwise
    held back, ``exclude`` what to leave out, ``include_large`` publishes past the budget. Raises
    :class:`PublicationError` for a name that names nothing: a slip must not publish a file its
    author meant to leave out."""
    if results.directory is None:
        return Offer(None)
    readiness = requirements(results.directory, results.document)
    pieces, held = _gathered(readiness)
    named = [h for h in held if any(h.piece.matches(name) for name in include)]
    pieces = _merged([*pieces, *(h.piece for h in named)])
    held = [h for h in held if h not in named]
    excluded = [p for p in pieces if any(p.matches(name) for name in exclude)]
    pieces = [p for p in pieces if p not in excluded]
    known = [*pieces, *excluded, *(h.piece for h in held), *(h.piece for h in named)]
    unknown = [n for n in (*exclude, *include) if not any(p.matches(n) for p in known)]
    if unknown:
        raise PublicationError(f"nothing of this study is named {', '.join(unknown)}")
    code, large_code = _budget([p for p in pieces if p.group == "code"])
    inputs, large_inputs = _budget([p for p in pieces if p.group == "inputs"])
    large = (*large_code, *large_inputs)
    return Offer(
        readiness,
        code,
        inputs,
        large,
        tuple(held),
        tuple(excluded),
        tuple(scan([*code, *inputs, *(large if include_large else ())])),
        include_large,
    )


def _budget(pieces: Sequence[Piece]) -> tuple[tuple[Piece, ...], tuple[Piece, ...]]:
    """``pieces`` split by :data:`BUDGET`, the smallest first: as many fit as possible."""
    fits: list[Piece] = []
    over: list[Piece] = []
    used = 0
    for piece in sorted(pieces, key=lambda p: (p.size, p.label)):
        if used + piece.size <= BUDGET:
            fits.append(piece)
            used += piece.size
        else:
            over.append(piece)
    order = {piece.label: index for index, piece in enumerate(pieces)}
    return (
        tuple(sorted(fits, key=lambda p: order[p.label])),
        tuple(sorted(over, key=lambda p: order[p.label])),
    )


def _merged(pieces: Iterable[Piece]) -> list[Piece]:
    """``pieces``, each file once: two stages can run the same job file."""
    found: dict[tuple[str, str, bytes], Piece] = {}
    for piece in pieces:
        key = (piece.owner, piece.name, piece.data)
        held = found.get(key)
        found[key] = (
            piece
            if held is None
            else Piece(
                piece.group,
                piece.owner,
                piece.name,
                piece.data,
                tuple(dict.fromkeys((*held.requirements, *piece.requirements))),
                tuple(sorted({*held.stages, *piece.stages})),
                piece.base,
                piece.added,
                piece.removed,
            )
        )
    return list(found.values())


def _gathered(readiness: Readiness) -> tuple[list[Piece], list[Held]]:
    pieces: list[Piece] = []
    held: list[Held] = []
    assert readiness.store is not None
    store = readiness.store
    reads = _Reads(readiness.ran)
    k4opt = next(
        (str((r.entry or {}).get("checkout")) for r in readiness.requirements if r.kind == "k4opt"),
        None,
    )
    for requirement in readiness.requirements:
        if requirement.unused or requirement.entry is None:  # needed by no rerun of them
            continue
        if requirement.kind in ("package", "code"):
            _from_origin(requirement, store, reads, k4opt, pieces, held)
        elif requirement.kind == "steering":
            _from_steering(requirement, store, pieces, held)
        elif requirement.kind == "geometry":
            _from_files(requirement, store, "geometry", requirement.entry.get("files"), pieces)
        elif requirement.kind == "evaluator":
            path = str(requirement.entry.get("path"))
            if not path.startswith((f"{k4opt}/", f"{_K4OPT}/")):  # k4opt's own code otherwise
                owner = requirement.label.partition(" ")[0]
                _from_files(requirement, store, owner, requirement.entry.get("files"), pieces)
        elif requirement.kind == "input":
            _from_input(requirement, store, pieces, held)
    return pieces, held


class _Reads:
    """Which stages read a file, by where they read it or what it held."""

    def __init__(self, ran: Mapping[tuple[str, str], set[str]]) -> None:
        self.paths: dict[str, set[str]] = {}
        self.identities: dict[str, set[str]] = {}
        for (path, identity), stages in ran.items():
            self.paths.setdefault(path, set()).update(stages)
            self.identities.setdefault(identity, set()).update(stages)

    def stages(self, paths: Iterable[str] = (), sha256: object = None) -> tuple[str, ...]:
        found: set[str] = set()
        for path in paths:
            found |= self.paths.get(path, set())
        if isinstance(sha256, str) and sha256:
            found |= self.identities.get(f"sha256:{sha256}", set())
        return tuple(sorted(found))


def _kept(store: Store, kept: object) -> bytes | None:
    """The bytes kept as ``kept``; ``None`` if they are not there as they were, which the
    study's readiness says."""
    if not isinstance(kept, str):
        return None
    try:
        return store.get(kept)
    except SoftwareError:
        return None


def _places(checkout: str, name: str, entries: Iterable[str]) -> list[str]:
    """Where a stage could have read the file ``name`` of ``checkout``: there, or through a local
    entry, which may name the directory by another path."""
    path = os.path.join(checkout, name)
    found = [path]
    for entry in entries:
        real = os.path.realpath(entry)
        if path.startswith(f"{real}/"):
            found.append(os.path.join(entry, os.path.relpath(path, real)))
    return found


def _public_name(entry: Mapping[str, Any]) -> str:
    """What a local build or directory is called in public: never where it is."""
    return str(entry.get("package") or os.path.basename(str(entry.get("prefix")).rstrip("/")))


def _binary(data: bytes) -> bool:
    return b"\0" in data[:8000]


def _from_origin(
    requirement: Requirement,
    store: Store,
    reads: _Reads,
    k4opt: str | None,
    pieces: list[Piece],
    held: list[Held],
) -> None:
    origin = requirement.entry
    assert origin is not None
    owner = _public_name(origin)
    git = origin.get("git")
    is_build = origin.get("kind") == "build"
    entries = [str(e) for e in origin.get("entries") or ()]
    if isinstance(git, Mapping):
        patch = _kept(store, git.get("patch"))
        ours = git.get("checkout") == k4opt  # k4opt ran as k4opt, not through a program's reads
        for section in sections(patch or b""):
            _from_section(
                requirement, git, owner, section, is_build or ours, reads, entries, pieces, held
            )
    if is_build and isinstance(git, Mapping):
        return  # what it installed is what its source installs
    for name, file in (origin.get("files") or {}).items():
        if not isinstance(file, Mapping):
            continue
        data = _kept(store, file.get("kept"))
        if data is None:
            continue
        stages = reads.stages(sha256=file.get("sha256"))
        piece = Piece("code", owner, str(name), data, (requirement.id,), stages)
        if stages:
            pieces.append(piece)
        else:
            held.append(
                Held(
                    "code", owner, str(name), "not read by any stage", requirement.id, False, piece
                )
            )


def _from_section(
    requirement: Requirement,
    git: Mapping[str, Any],
    owner: str,
    section: Section,
    compiled: bool,
    reads: _Reads,
    entries: Iterable[str],
    pieces: list[Piece],
    held: list[Held],
) -> None:
    untracked = git.get("untracked_files") or {}
    sha256 = untracked.get(section.name)
    is_untracked = section.name in untracked
    whole = git.get("base_is") == "none"  # no public commit: its every file is in the patch
    base = str(git.get("base") or "")[:7] or "no public commit"
    stages = reads.stages(_places(str(git.get("checkout")), section.name, entries), sha256)
    installed = sha256 is not None and any(
        isinstance(f, Mapping) and f.get("sha256") == sha256
        for f in (requirement.entry or {}).get("files", {}).values()
    )
    piece = Piece(
        "code",
        owner,
        section.name,
        section.data,
        (requirement.id,),
        stages or requirement.used_by,
        base,
        None if section.binary else section.added,
        None if section.binary else section.removed,
    )
    why: str | None = None
    needed = False
    if is_untracked and not stages and not installed:
        why = "untracked, not used"
    elif section.binary:
        why, needed = "binary", True
    elif whole and not stages and not installed:
        why = "no public commit, not used"
    elif not compiled and not stages:
        why = "not read by any stage"
    if why is None:
        pieces.append(piece)
    else:
        held.append(Held("code", owner, section.name, why, requirement.id, needed, piece))


def _from_steering(
    requirement: Requirement, store: Store, pieces: list[Piece], held: list[Held]
) -> None:
    entry = requirement.entry
    assert entry is not None
    owner = str(entry.get("package") or requirement.label)
    original = entry.get("original")
    for name, file in {**(entry.get("changed") or {}), **(entry.get("added") or {})}.items():
        if not isinstance(file, Mapping):
            continue
        data = _kept(store, file.get("kept"))
        if data is None or str(name).endswith(".pyc") or "__pycache__/" in str(name):
            continue  # made again by whoever runs it
        if _binary(data):
            piece = Piece("code", owner, str(name), data, (requirement.id,), requirement.used_by)
            held.append(Held("code", owner, str(name), "binary", requirement.id, True, piece))
            continue
        added, removed = _counted(original, str(name), data)
        pieces.append(
            Piece(
                "code",
                owner,
                str(name),
                data,
                (requirement.id,),
                requirement.used_by,
                added=added,
                removed=removed,
            )
        )


def _counted(original: object, name: str, data: bytes) -> tuple[int | None, int | None]:
    """The lines a changed job file adds and removes against the stack's, where that can be read."""
    try:
        with open(os.path.join(str(original), name), "rb") as handle:
            theirs = handle.read().decode().splitlines()
        lines = list(difflib.ndiff(theirs, data.decode().splitlines()))
    except (OSError, ValueError):
        return None, None
    return sum(line.startswith("+ ") for line in lines), sum(
        line.startswith("- ") for line in lines
    )


def _from_files(
    requirement: Requirement, store: Store, owner: str, files: object, pieces: list[Piece]
) -> None:
    for name, file in (files if isinstance(files, Mapping) else {}).items():
        if not isinstance(file, Mapping):
            continue
        data = _kept(store, file.get("kept"))
        if data is not None:
            shown = os.path.basename(str(name)) if os.path.isabs(str(name)) else str(name)
            pieces.append(Piece("code", owner, shown, data, (requirement.id,), requirement.used_by))


def _from_input(
    requirement: Requirement, store: Store, pieces: list[Piece], held: list[Held]
) -> None:
    entry = requirement.entry
    assert entry is not None
    data = _kept(store, entry.get("kept"))
    if data is None:
        return
    path = str(entry.get("path"))
    name = os.path.basename(path)
    if entry.get("origin") == "geometry":  # a file a candidate's reference reached: code that ran
        pieces.append(Piece("code", "geometry", name, data, (requirement.id,), requirement.used_by))
        return
    piece = Piece("inputs", "", name, data, (requirement.id,), requirement.used_by)
    home = os.path.expanduser("~")
    if path.startswith(f"{home}/.") or name == "pyvenv.cfg":
        held.append(
            Held(
                "inputs",
                "",
                name,
                "home or environment configuration",
                requirement.id,
                False,
                piece,
            )
        )
    else:
        pieces.append(piece)


# --- the public form ---------------------------------------------------------------------------


_NOTES = (
    "a nightly:",
    "changed locally",
    "its source changed",
    "its CMake configuration",
    "binary only",
    "no remote has its history",
    "the nightly's revision",
    "k4opt's own checkout",
)
"""The notes of a requirement that hold no local path; any other is not published."""


def public_form(
    results: StudyResults,
    release: Release,
    snapshot: str,
    first: tuple[str, str] | None,
    artifacts: Sequence[PublishedArtifact],
) -> SoftwareSheet:
    """The sheet of ``results`` in the state of ``snapshot`` (a SHA-256): the public form of its
    requirements as checked just now, with what ``release`` approved. ``first`` is the time and
    status of the first publication of this snapshot, if it was published before."""
    readiness = release.offer.readiness
    assert readiness is not None
    access = _access(readiness, release)
    public = tuple(_requirement(r, access[r.id]) for r in readiness.requirements)
    status = _status(readiness, access)
    return SoftwareSheet(
        study=results.id,
        snapshot=snapshot,
        status=status,
        checked=readiness.at,
        first_checked=first[0] if first else readiness.at,
        first_status=first[1] if first else status,
        gaps=readiness.public_gaps,
        altered=readiness.altered,
        not_kept=len(readiness.problems),
        on_request=tuple(sorted({piece.label for piece in release.offer.excluded})),
        requirements=public,
        artifacts=tuple(artifacts),
    )


def _status(readiness: Readiness, access: Mapping[str, Access]) -> str:
    """The study's status (:attr:`Readiness.status`), and whether it needs what only its author can
    give: what a reader may do is what is published, not what the study kept."""
    status = readiness.status
    private = any(access[r.id] == "on request" for r in readiness.requirements if not r.unused)
    if not private or status.startswith((BLOCKED, INCOMPLETE)):
        return status
    return NEEDS_ACCESS if status.startswith(ALL_AVAILABLE) else f"{status} · needs access"


def _access(readiness: Readiness, release: Release) -> dict[str, Access]:
    offer_ = release.offer
    published = {r for piece in release.pieces for r in piece.requirements}
    withheld = {
        r
        for piece in [*offer_.code, *offer_.inputs, *offer_.large, *offer_.excluded]
        if piece not in release.pieces
        for r in piece.requirements
    }
    withheld |= {h.requirement for h in offer_.held if h.needed}
    found: dict[str, Access] = {}
    for r in readiness.requirements:
        if r.id in withheld:
            found[r.id] = "on request"
        elif r.id in published:
            found[r.id] = "public"
        else:
            found[r.id] = _unoffered(r)
    return found


def _unoffered(requirement: Requirement) -> Access:
    """Who can get a requirement of which nothing is published: the stack's own is public; what
    was not kept, or is no file to give, no one."""
    entry = requirement.entry or {}
    if requirement.kind in ("environment", "listing") or requirement.unused:
        return "none"
    if requirement.kind == "input":
        return "public" if str(entry.get("identity", "")).startswith("cvmfs:") else "none"
    if requirement.kind in ("package", "code"):
        rebuildable = (requirement.source or {}).get("rebuildable")
        return "none" if requirement.kind == "package" and not rebuildable else "public"
    return "public"


def _worst(requirement: Requirement) -> State:
    states = [check.state for check in requirement.checks]
    for state in ("altered", "gone", "not-checked"):
        if state in states:
            return state
    return "available" if states else "not-checked"


def _label(requirement: Requirement) -> str:
    entry = requirement.entry or {}
    if requirement.kind in ("package", "code"):
        return _public_name(entry)
    if requirement.kind == "environment":
        return "a Python environment"
    if requirement.id.startswith("listing:"):
        return "a directory a stage listed"
    return requirement.label


def _public_notes(requirement: Requirement) -> tuple[str, ...]:
    notes = [note for note in requirement.notes if note.startswith(_NOTES)]
    large = (requirement.entry or {}).get("git", {}) if requirement.entry else {}
    if isinstance(large, Mapping) and large.get("large"):
        notes.append(f"{len(large['large'])} untracked files too large to keep")
    return tuple(notes)


def _identified(requirement: Requirement) -> str:
    if requirement.kind == "input":  # says what it is: its size or variable is the author's
        return f"{(requirement.entry or {}).get('origin')} input"
    return requirement.identified


_FACTS = ("base_is", "changed", "link", "rebuildable", "stale", "incomplete")


def _facts(requirement: Requirement) -> dict[str, Json]:
    source: Mapping[str, Any] = requirement.source or {}
    entry = requirement.entry or {}
    facts: dict[str, Json]
    if requirement.kind in ("package", "code"):
        facts = {key: source[key] for key in _FACTS if key in source}
        if source.get("base"):
            facts["base"] = str(source["base"])[:12]
        facts["package"] = entry.get("package")
        return facts
    if requirement.kind == "base":
        return {"release": source.get("release"), "head_packages": source.get("head_packages")}
    if requirement.kind == "steering":
        return {
            "package": source.get("package"),
            "revision": source.get("revision"),
            "removed": len(source.get("removed") or ()),
        }
    if requirement.kind == "k4opt":
        return {"commit": source.get("commit")}
    if requirement.kind == "input":
        return {"commitment": source.get("commitment")}
    return {}


def _ran(requirement: Requirement) -> Mapping[str, Json] | None:
    ran = requirement.ran
    if ran is None:
        return None
    files = [
        {
            "file": os.path.basename(str(file["file"]))
            if os.path.isabs(str(file["file"]))
            else str(file["file"]),
            "sha256": file.get("sha256"),
            "build_id": file.get("build_id"),
        }
        for file in cast("Iterable[dict[str, Any]]", thaw(ran.get("files") or ()))
    ]
    return cast(
        "Mapping[str, Json]",
        freeze(
            {"original": ran.get("original"), "files": files},
            "the files that ran",
            PublicationError,
        ),
    )


def _requirement(requirement: Requirement, access: Access) -> PublishedRequirement:
    return PublishedRequirement(
        id=requirement.id if not requirement.id.startswith("listing:") else "listing",
        kind=requirement.kind,
        label=_label(requirement),
        identified=_identified(requirement),
        used_by=requirement.used_by,
        state=_worst(requirement),
        access=access,
        notes=_public_notes(requirement),
        facts=FrozenMap(_facts(requirement)),
        ran=_ran(requirement),
        unused=requirement.unused,
    )


def bundled(study: str, pieces: Iterable[Piece]) -> list[tuple[PublishedArtifact, bytes]]:
    """``pieces`` as the files published: a patch per owner and base, of the files approved, and
    each other file on its own."""
    patches: dict[tuple[str, str], list[Piece]] = {}
    found: list[tuple[PublishedArtifact, bytes]] = []
    for piece in pieces:
        if piece.is_patch:
            assert piece.base is not None
            patches.setdefault((piece.owner, piece.base), []).append(piece)
        else:
            one: tuple[PublishedFile, ...] = (
                PublishedFile(piece.name, piece.added, piece.removed),
            )
            found.append(
                _artifact(
                    study,
                    piece.group,
                    piece.owner,
                    piece.name,
                    "file",
                    piece.data,
                    piece.stages,
                    piece.requirements,
                    None,
                    one,
                )
            )
    for (owner, base), part in patches.items():
        data = b"".join(p.data for p in part)
        files = tuple(PublishedFile(p.name, p.added, p.removed) for p in part)
        stages = tuple(sorted({s for p in part for s in p.stages}))
        needs = tuple(dict.fromkeys(r for p in part for r in p.requirements))
        found.append(
            _artifact(
                study, "code", owner, f"{owner}.patch", "patch", data, stages, needs, base, files
            )
        )
    return found


def _artifact(
    study: str,
    group: Group,
    owner: str,
    name: str,
    kind: Literal["patch", "file"],
    data: bytes,
    stages: tuple[str, ...],
    needs: tuple[str, ...],
    base: str | None,
    files: tuple[PublishedFile, ...],
) -> tuple[PublishedArtifact, bytes]:
    sha256 = hashlib.sha256(data).hexdigest()
    return (
        PublishedArtifact(
            group,
            owner,
            name,
            kind,
            sha256,
            len(data),
            artifact_path(study, sha256),
            stages,
            needs,
            base,
            files,
        ),
        data,
    )
