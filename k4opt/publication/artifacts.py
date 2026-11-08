"""What of a study may be published beside its snapshot, and its public form.

The study keeps privately what a rerun needs (:mod:`k4opt.software.keep`). :func:`offer` says what
of it is offered for publication, in two groups the author answers for separately:

- **code** that ran, each a *unit* approved whole: a build (its checkout's whole kept state against
  the revision the nightly built, untracked files included, its CMake options, what it installed
  that a stage read), a directory of local code a stage read (its every kept file, or its
  checkout's state), k4opt's own checkout, a changed job directory, the geometry, an evaluator;
- **inputs** the stages read, each a unit of its own, up to :data:`BUDGET` in all.

Which files of a unit a rerun needs is not guessed: what is compiled is never read at run time, so a
build's whole state is offered, and a unit is published whole or not at all. Leaving out one of its
files (``exclude``) leaves out the unit, which is then on request. :func:`scan` looks through what
would be published for secrets, whose findings block publishing (:meth:`Offer.release`).

Every unit has an id that holds between publications (:class:`~k4opt.software.readiness.Requirement`
ids; ``link:<stage>/<file>`` for a job file that leads out of its directory), so a later
publication knows what an earlier one published by that id and the file's name, never by its bytes.

An :class:`Offer` is what the author reviews; a :class:`Release`, made from it, is what was
approved: :func:`~k4opt.publication.publish` publishes exactly that list.
:func:`public_form` makes the study's :class:`~k4opt.publication.SoftwareSheet`, always published.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

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
    Readiness,
    Requirement,
    requirements,
)

BUDGET = 10_000_000
"""How many bytes of code, and of inputs, are published unless ``--include-large`` says more."""
NEEDS_ACCESS = "Not rerun · needs access"
type Role = Literal["file", "generated", "external"]
type Kind = Literal["patch", "file", "generated", "external"]
_TOOLS = ("pyvenv.cfg",)
"""What says only where an environment's interpreter is: listed, and no rerun needs it from the
author; the environment is a requirement of its own."""


@dataclass(frozen=True, slots=True)
class Section:
    """One file's part of a git patch: its ``name``, its text, and the lines it adds and removes
    (``binary``: a ``GIT binary patch``, of no lines)."""

    name: str
    data: bytes
    binary: bool
    added: int
    removed: int
    deleted: bool = False
    link: bool = False


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
        head = text[: text.index(b"\n@@")] if b"\n@@" in text else text
        deleted, link = b"\ndeleted file mode " in head, b" mode 120000" in head
        found.append(Section(name, text, binary, added, removed, deleted, link))
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
    """A file that could be published, of the unit its first requirement names (``unit``), in a
    ``group``: ``owner`` is what the unit is called in public, ``requirements`` what it answers,
    ``stages`` the stages that used it. A patch (``base`` not ``None``) is a checkout's whole kept
    state against ``base``, its ``files`` each with the lines it adds and removes (``None`` for a
    binary one); ``added`` and ``removed`` count the lines of a change, if counted. ``role`` says
    what any other piece is: a file of its unit as it is (``file``), one k4opt makes for it
    (``generated``: options, a manifest, aliases, a list of distributions) or one the geometry
    reached out of its tree, named by its content (``external``). ``paths`` are where its files are
    on the author's machine, never published: by them, two units that hold one file are known."""

    group: Group
    owner: str
    name: str
    data: bytes
    requirements: tuple[str, ...]
    stages: tuple[str, ...]
    base: str | None = None
    added: int | None = None
    removed: int | None = None
    place: str | None = None
    """Where an input was read: how its author can name it. Never published."""
    files: tuple[tuple[str, int | None, int | None], ...] = ()
    role: Role = "file"
    paths: tuple[str, ...] = ()

    @property
    def unit(self) -> str:
        return self.requirements[0]

    @property
    def kind(self) -> Kind:
        return "patch" if self.base is not None else self.role

    @property
    def key(self) -> tuple[str, str, str]:
        """What it is, between publications: its unit, its kind and its name. No two pieces of a
        study have one (:func:`_unique`), so a generated file never stands for a real one."""
        return self.unit, self.kind, self.name

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
        """Whether ``name`` names it: its owner, its name or label, its own file name, a file of
        its patch, where it was read, or the id of what it answers (``input:7c1e09a2b4f3``)."""
        if not name.strip():  # an input has no owner: no name is no selection of every input
            return False
        inside = [file for file, _, _ in self.files]
        return name in (
            self.owner,
            self.name,
            self.label,
            os.path.basename(self.name),
            self.place,
            *self.requirements,
            *inside,
            *(f"{self.owner}/{file}" for file in inside),
            *(os.path.basename(file) for file in inside),
        )


@dataclass(frozen=True, slots=True)
class Lost:
    """What a unit should hold and whose kept copy is gone: its identity (:attr:`Piece.key`), what
    it answers, and for a patch the commit it is against. An earlier publication's copy of it can
    stand in for it (:func:`still_public`)."""

    key: tuple[str, str, str]
    requirements: tuple[str, ...]
    base: str | None = None


def _unique(pieces: Iterable[Piece]) -> list[Piece]:
    """``pieces``, each identity once: equal ones merged (what they answer, who used them, where
    they are), and two of one identity and other bytes refused, since a reader could not tell
    them apart. A file two references reach is published once."""
    found: dict[tuple[str, str, str], Piece] = {}
    for piece in pieces:
        held = found.get(piece.key)
        if held is None:
            found[piece.key] = piece
        elif (held.data, held.base) != (piece.data, piece.base):
            raise PublicationError(
                f"{piece.unit} holds two different files called {piece.name}: nothing is "
                "published, since a reader could not tell them apart"
            )
        else:
            found[piece.key] = replace(
                held,
                requirements=tuple(dict.fromkeys((*held.requirements, *piece.requirements))),
                stages=tuple(sorted({*held.stages, *piece.stages})),
                paths=tuple(dict.fromkeys((*held.paths, *piece.paths))),
            )
    return list(found.values())


def _left_out(pieces: Sequence[Piece], names: Sequence[str]) -> set[str]:
    """The units ``names`` leave out: those holding a file a name names, then every unit holding
    one of the files of those (where they are on the author's machine) or answering for one of
    them, until none is added: a file left out is left out wherever it is."""
    units = {p.unit for p in pieces if any(p.matches(name) for name in names)}
    while True:
        paths = {path for p in pieces if p.unit in units for path in p.paths}
        more = {
            p.unit
            for p in pieces
            if p.unit not in units
            and (paths.intersection(p.paths) or units.intersection(p.requirements[1:]))
        }
        if not more:
            return units
        units |= more


def _refuse_empty(names: Iterable[str]) -> None:
    """A script whose variable is unset passes an empty name: that must select nothing."""
    if any(not name.strip() for name in names):
        raise PublicationError("an empty name names no file: check what was passed")


def _units(pieces: Iterable[Piece]) -> dict[str, list[Piece]]:
    """``pieces`` by their unit, in their order."""
    found: dict[str, list[Piece]] = {}
    for piece in pieces:
        found.setdefault(piece.unit, []).append(piece)
    return found


def _named(pieces: Iterable[Piece], name: str) -> list[Piece]:
    """The pieces of the unit ``name`` names. A name that fits several units is refused, with how
    to name each: approving one must never approve two."""
    units = {u: part for u, part in _units(pieces).items() if any(p.matches(name) for p in part)}
    if len(units) > 1:
        each = "; ".join(
            f"{u}: {part[0].place or part[0].owner or part[0].name}" for u, part in units.items()
        )
        raise PublicationError(f"{name} names {len(units)} files; name one of them: {each}")
    return [piece for part in units.values() for piece in part]


@dataclass(frozen=True, slots=True)
class Held:
    """A unit of what the study kept that is not offered unless named, and why (``why``); ``needed``
    if a rerun may need it. Naming it (``include``) offers its ``pieces``."""

    group: Group
    owner: str
    name: str
    why: str
    requirement: str
    needed: bool
    pieces: tuple[Piece, ...]

    @property
    def label(self) -> str:
        return self.name


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
            rb"(?i)(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
            rb"\b\s*[:=]\s*(?:[\"'](?!/)[^\"'\s]{8,256}[\"']"
            # unquoted: what a token looks like, not a name, a call or a path
            rb"|(?=[\w+=-]{0,255}\d)[\w+=-]{8,256}(?![\w+=(.\[/-]))"
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
        # a patch file by file, so that a finding names the file it is in
        parts = (
            [
                replace(piece, name=f"{piece.owner}/{s.name}", owner="", data=s.data)
                for s in sections(piece.data)
            ]
            if piece.is_patch
            else [piece]
        )
        for part in parts:
            if _SECRET_NAMES.search(part.name):
                findings.append(Finding(part.label, 1, "a file of credentials, by its name"))
            for what, pattern in _SECRETS:
                match = pattern.search(part.data)
                if match is not None:
                    findings.append(Finding(part.label, _line(part, match.start()), what))
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

    ``code`` and ``inputs`` are offered, unit by unit; ``large`` the units past the :data:`BUDGET`
    of their group (published only ``include_large``); ``held`` the units listed and not offered;
    ``excluded`` the units the author left out by name; ``findings`` what :func:`scan` saw in what
    would be published; ``lost`` what should be kept and is not (the unit and the file's name). A
    study read from its JSON form offers nothing (``readiness`` is ``None``)."""

    readiness: Readiness | None
    code: tuple[Piece, ...] = ()
    inputs: tuple[Piece, ...] = ()
    large: tuple[Piece, ...] = ()
    held: tuple[Held, ...] = ()
    excluded: tuple[Piece, ...] = ()
    findings: tuple[Finding, ...] = ()
    include_large: bool = False
    study: str = ""
    """The id of the study it was made from."""
    snapshot: str = ""
    """The SHA-256 of that study's snapshot when it was reviewed."""
    lost: tuple[Lost, ...] = ()

    @property
    def everything(self) -> tuple[Piece, ...]:
        """All it knows of what could be published, whatever became of each."""
        held = (p for h in self.held for p in h.pieces)
        return (*self.code, *self.inputs, *self.large, *self.excluded, *held)

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
            _refuse_empty(inputs)
            named = {name: _named(all_inputs, name) for name in inputs}
            unknown = [name for name, found in named.items() if not found]
            if unknown:
                raise PublicationError(
                    f"no input offered is named {', '.join(unknown)}; a large one takes "
                    "--include-large as well"
                )
            units = {p.unit for found in named.values() for p in found}
            chosen = [p for p in all_inputs if p.unit in units]
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
    include_inputs: Sequence[str] = (),
    exclude: Sequence[str] = (),
    include_large: bool = False,
) -> Offer:
    """What of ``results`` may be published (see the module): ``include`` names the code units
    otherwise held back and ``include_inputs`` the inputs, each only of its own kind, so that
    naming an input never opts in code; ``exclude`` leaves out the units it names a file of,
    ``include_large`` publishes past the budget. Raises :class:`PublicationError` for a name that
    names nothing: a slip must not publish a file its author meant to leave out."""
    _refuse_empty((*include, *include_inputs, *exclude))
    snapshot = hashlib.sha256(results.to_json().encode()).hexdigest()
    if results.directory is None:
        return Offer(None, study=results.id, snapshot=snapshot)
    readiness = requirements(results.directory, results.document)
    pieces, held, lost = _gathered(readiness)
    of = {
        group: [p for h in held if h.group == group for p in h.pieces]
        for group in ("code", "inputs")
    }
    wanted = {p.unit for name in include for p in _named(of["code"], name)}
    wanted |= {p.unit for name in include_inputs for p in _named(of["inputs"], name)}
    named = [h for h in held if h.requirement in wanted]
    pieces = [*pieces, *(p for h in named for p in h.pieces)]
    held = [h for h in held if h not in named]
    left_out = _left_out(pieces, exclude)
    excluded = [p for p in pieces if p.unit in left_out]
    pieces = [p for p in pieces if p.unit not in left_out]
    known = [*pieces, *excluded, *(p for h in [*held, *named] for p in h.pieces)]
    unknown = [n for n in exclude if not any(p.matches(n) for p in known)]
    unknown += [
        name
        for names, group in ((include, "code"), (include_inputs, "inputs"))
        for name in names
        if not any(p.group == group and p.matches(name) for p in known)
    ]
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
        results.id,
        snapshot,
        tuple(lost),
    )


def _budget(pieces: Sequence[Piece]) -> tuple[tuple[Piece, ...], tuple[Piece, ...]]:
    """``pieces`` split by :data:`BUDGET`, unit by unit, the smallest first: as many fit as
    possible, and a unit is never split."""
    fits: set[str] = set()
    used = 0
    units = _units(pieces)
    for unit, part in sorted(units.items(), key=lambda kv: (sum(p.size for p in kv[1]), kv[0])):
        size = sum(p.size for p in part)
        if used + size <= BUDGET:
            fits.add(unit)
            used += size
    return (
        tuple(p for p in pieces if p.unit in fits),
        tuple(p for p in pieces if p.unit not in fits),
    )


def _gathered(readiness: Readiness) -> tuple[list[Piece], list[Held], list[Lost]]:
    """What the study kept that could be published, by unit: offered, held back, and what should
    be kept and is not. A file that is an input is published as that input alone, with its own
    consent, never through a unit of code that holds it too, and the input then answers for that
    unit as well."""
    pieces: list[Piece] = []
    held: list[Held] = []
    lost: list[Lost] = []
    if readiness.store is None:  # nothing kept is left to offer; its status says so
        return pieces, held, lost
    store = readiness.store
    reads = _Reads(readiness.ran, not readiness.gaps)
    k4opt = next(
        (str((r.entry or {}).get("checkout")) for r in readiness.requirements if r.kind == "k4opt"),
        None,
    )
    geometry_root = next(
        (str((r.entry or {}).get("root")) for r in readiness.requirements if r.kind == "geometry"),
        None,
    )
    inputs = _inputs_by_place(readiness)
    attached: dict[str, set[str]] = {}  # input: the units of code that hold it too
    for requirement in readiness.requirements:
        if requirement.kind == "environment":
            _from_environment(requirement, reads, pieces, held)
            continue
        if requirement.unused or requirement.entry is None:  # needed by no rerun of them
            continue
        if requirement.kind in ("package", "code"):
            _from_origin(requirement, store, reads, k4opt, inputs, attached, pieces, held, lost)
        elif requirement.kind == "steering":
            _from_steering(requirement, store, pieces, lost)
        elif requirement.kind == "geometry":
            files = requirement.entry.get("files")
            root = str(requirement.entry.get("root") or "")
            _from_files(requirement, store, "geometry", files, root, pieces, lost)
            aliases = _aliases(requirement)
            if aliases is not None:
                pieces.append(aliases)
        elif requirement.kind == "evaluator":  # its own unit, wherever its file is
            owner = requirement.label.partition(" ")[0]
            files = requirement.entry.get("files")
            beside = os.path.dirname(str(requirement.entry.get("path") or ""))
            _from_files(requirement, store, owner, files, beside, pieces, lost)
        elif requirement.kind == "input":
            _from_input(requirement, store, geometry_root, pieces, held, lost)

    def answering(piece: Piece) -> Piece:
        more = sorted(attached.get(piece.unit, ()))
        if piece.group != "inputs" or not more:
            return piece
        return replace(piece, requirements=tuple(dict.fromkeys((*piece.requirements, *more))))

    pieces = _unique(answering(piece) for piece in pieces)
    held = [replace(h, pieces=tuple(_unique(answering(p) for p in h.pieces))) for h in held]
    _unique([*pieces, *(p for h in held for p in h.pieces)])  # one identity, offered or held
    return pieces, held, lost


def _inputs_by_place(readiness: Readiness) -> dict[str, str]:
    """Each place an input that takes its own consent is at, its unit: the files the stages read
    as data (not those only a candidate's geometry reached, which are code), and the files a job
    directory links to out of it."""
    found: dict[str, str] = {}
    for r in readiness.requirements:
        entry = r.entry or {}
        if r.kind == "input" and r.entry is not None and entry.get("origin") != "geometry":
            path = str(entry.get("path"))
            found |= {path: r.id, os.path.realpath(path): r.id}
        elif r.kind == "steering" and r.entry is not None:
            for name, file in {
                **(entry.get("changed") or {}),
                **(entry.get("added") or {}),
            }.items():
                if isinstance(file, Mapping) and _leaves(str(name), file):
                    place = str(file.get("outside") or file.get("link"))
                    found |= {place: f"link:{r.label}/{name}"}
    return found


_CACHED = re.compile(r"^(.*)/__pycache__/([^/.]+)\.[^/.]+(?:\.opt-\d+)?\.pyc$")


class _Reads:
    """Which stages read a file, by where they read it: equal bytes elsewhere say nothing of this
    copy, its digest only that what was read there is this file. A module Python read from its
    cache was used as much as if it had read its source, which it then never opens. ``complete``
    says that every stage was observed: only then is a file no stage read known not to be used."""

    def __init__(self, ran: Mapping[tuple[str, str], set[str]], complete: bool) -> None:
        self.complete = complete
        self.paths: dict[str, list[tuple[str | None, set[str]]]] = {}
        for (path, identity), stages in ran.items():
            self.paths.setdefault(path, []).append((identity, stages))
            cached = _CACHED.match(path)
            if cached is not None:  # a valid cache vouches for its source, by its own digest
                self.paths.setdefault(f"{cached[1]}/{cached[2]}.py", []).append((None, stages))

    def stages(self, paths: Iterable[str], sha256: object = None) -> tuple[str, ...]:
        """The stages that read one of ``paths``, as the file of this ``sha256`` if it is known."""
        wanted = f"sha256:{sha256}" if isinstance(sha256, str) and sha256 else None
        found: set[str] = set()
        for path in paths:
            for identity, stages in self.paths.get(path, ()):
                if wanted is None or identity is None or identity == wanted:
                    found |= stages
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
    found = [path, os.path.realpath(path)]
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
    inputs: Mapping[str, str],
    attached: dict[str, set[str]],
    pieces: list[Piece],
    held: list[Held],
    lost: list[Lost],
) -> None:
    """A build or a directory of local code, as one unit: its checkout's whole kept state, a
    build's options and what it installed that a stage read, a directory's kept files. A build
    is offered, since what is compiled is never read at run time; k4opt's own checkout, since it
    ran; a directory of code if a stage read from it, or if that is not known. A file of it that
    is an input (``inputs``, by where it is) is left to that input, which answers for this unit
    too (``attached``)."""
    origin = requirement.entry
    assert origin is not None
    owner = _public_name(origin)
    git = origin.get("git")
    is_build = origin.get("kind") == "build"
    entries = [str(e) for e in origin.get("entries") or ()]
    rid = requirement.id
    unit: list[Piece] = []
    used = bool(requirement.used_by)

    def an_input(places: Iterable[str]) -> bool:
        holder = next((inputs[place] for place in places if place in inputs), None)
        if holder is not None:
            attached.setdefault(holder, set()).add(rid)
        return holder is not None

    if isinstance(git, Mapping) and git.get("changed"):
        base = str(git.get("base") or "")[:7] or "no public commit"
        name = f"{owner}.patch"
        patch = _kept(store, git.get("patch"))
        if patch is None:
            lost.append(Lost((rid, "patch", name), (rid,), base))
        else:
            checkout = str(git.get("checkout"))
            parts, held_at = [], []
            for part in sections(patch):
                places = _places(checkout, part.name, entries)
                used = used or bool(reads.stages(places))
                if not an_input(places):
                    parts.append(part)
                    held_at += places
            if parts:
                counted = tuple(
                    (p.name, None if p.binary else p.added, None if p.binary else p.removed)
                    for p in parts
                )
                unit.append(
                    Piece(
                        "code",
                        owner,
                        name,
                        b"".join(p.data for p in parts),
                        (rid,),
                        requirement.used_by,
                        base,
                        sum(p.added for p in parts if not p.binary),
                        sum(p.removed for p in parts if not p.binary),
                        files=counted,
                        paths=tuple(dict.fromkeys(held_at)),
                    )
                )
    if is_build:
        options = _kept(store, origin.get("configuration"))
        if options is not None:
            unit.append(
                Piece(
                    "code",
                    owner,
                    OPTIONS,
                    _options(options),
                    (rid,),
                    requirement.used_by,
                    role="generated",
                )
            )
        elif origin.get("configuration"):
            lost.append(Lost((rid, "generated", OPTIONS), (rid,)))
    sourced = is_build and isinstance(git, Mapping)
    roots = [str(origin.get("prefix")), *entries]
    roots += [os.path.realpath(root) for root in roots]
    for name, file in (origin.get("files") or {}).items():
        if not isinstance(file, Mapping):
            continue
        places = [os.path.join(root, str(name)) for root in roots]
        stages = reads.stages(places, file.get("sha256"))
        used = used or bool(stages)
        if sourced and not stages:
            continue  # what a build installed and nothing read: what its source installs
        # what a build installed may differ from what its source installs: what ran is offered
        shown = f"installed/{name}" if sourced else str(name)
        if an_input(places):
            continue
        data = _kept(store, file.get("kept"))
        if data is None:
            lost.append(Lost((rid, "file", shown), (rid,)))
            continue
        unit.append(
            Piece(
                "code",
                owner,
                shown,
                data,
                (rid,),
                stages or requirement.used_by,
                paths=tuple(dict.fromkeys(places)),
            )
        )
    if not unit:
        return
    ours = isinstance(git, Mapping) and git.get("checkout") == k4opt
    if is_build or ours or used or not reads.complete:
        pieces.extend(unit)
    else:
        held.append(Held("code", owner, owner, "not read by any stage", rid, False, tuple(unit)))


OPTIONS = "cmake-options.txt"
_OPTION = re.compile(r"^([A-Za-z_][\w.+-]*):(BOOL|STRING|PATH|FILEPATH|UNINITIALIZED)=(.*)$")
_PUBLIC_PATHS = ("/cvmfs/", "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/etc/")
"""Where a path names the stack or the system, which are public: such a setting is published."""
_SENSITIVE = re.compile(
    r"(?i)pass(?:word|wd|phrase)?|pwd|secret|token|credential|api[_-]?key|private[_-]?key|auth"
)


def _options(cache: bytes) -> bytes:
    """The options of a CMake cache that hold no path and whose name says no credential, an empty
    one too (it differs from none): what a rebuild needs of it, without where the build was made.
    Those left out are counted, and are the author's to give; the paths CMake found by itself are
    not settings."""
    kept, left = [], 0
    for line in cache.decode(errors="replace").splitlines():
        option = _OPTION.match(line)
        if option is None:
            continue
        name, value = option[1], option[3]
        if name == "CMAKE_INSTALL_PREFIX":  # where to install: a rerun chooses its own
            continue
        public = all(part.startswith(_PUBLIC_PATHS) for part in value.split(";") if part)
        if _SENSITIVE.search(name) or ("/" in value and not public):
            left += 1
        else:
            kept.append(f"{name}={value}")
    note = (
        [f"# {left} more options hold local paths or credentials: ask the author"] if left else []
    )
    return "\n".join([*note, *sorted(kept)]).encode() + b"\n"


def _from_steering(
    requirement: Requirement, store: Store, pieces: list[Piece], lost: list[Lost]
) -> None:
    """A changed job directory, as one unit: its changed and added files and what they do not say
    (``job-files.json``). A file that leads out of the directory is another file's bytes, data: an
    input of its own (``link:<stage>/<file>``), where it lies in the job, whether its kept copy is
    there or gone."""
    entry = requirement.entry
    assert entry is not None
    owner = str(entry.get("package") or requirement.label)
    original = entry.get("original")
    directory = str(entry.get("directory") or "")
    files = {**(entry.get("changed") or {}), **(entry.get("added") or {})}
    manifest = _manifest(entry, files)
    if manifest is not None:
        pieces.append(
            Piece(
                "code",
                owner,
                "job-files.json",
                manifest,
                (requirement.id,),
                requirement.used_by,
                role="generated",
            )
        )
    for name, file in files.items():
        if not isinstance(file, Mapping) or _made_again(str(name)):
            continue  # made again by whoever runs it
        leaves = _leaves(str(name), file)
        link = f"link:{requirement.label}/{name}"
        answers = (link, requirement.id) if leaves else (requirement.id,)
        data = _kept(store, file.get("kept"))
        if data is None:
            lost.append(Lost((answers[0], "file", str(name)), answers))
            continue
        if leaves:  # another file's bytes, not this directory's
            place = str(file.get("outside") or file.get("link"))
            pieces.append(
                Piece(
                    "inputs",
                    "",
                    str(name),
                    data,
                    answers,
                    requirement.used_by,
                    place=place,
                    paths=(place, os.path.realpath(place)),
                )
            )
            continue
        added, removed = _counted(original, str(name), data)
        pieces.append(
            Piece(
                "code",
                owner,
                str(name),
                data,
                answers,
                requirement.used_by,
                added=added,
                removed=removed,
                paths=(os.path.join(directory, str(name)),) if directory else (),
            )
        )


def _made_again(name: str) -> bool:
    return name.endswith(".pyc") or "__pycache__/" in name


def _leaves(name: str, file: Mapping[str, Any]) -> bool:
    """Whether the file ``name`` of a job directory is really out of it, other than on the stack,
    which is public: by what the study saw every link lead to, and in any case by its own link."""
    if file.get("outside"):
        return True
    link = file.get("link")
    if not isinstance(link, str) or link.startswith("/cvmfs/"):
        return False
    return os.path.isabs(link) or os.path.normpath(
        os.path.join(os.path.dirname(name), link)
    ).startswith("..")


def _manifest(entry: Mapping[str, Any], files: Mapping[str, Any]) -> bytes | None:
    """What the files of a job directory do not say of it: which files of the stack's package it
    removed, whether each of its files is executable (``false`` too: taking the bit away is a change
    of its own) and which are links within it; ``None`` if it changed nothing. A link that leads
    out of it names a place of the author's, and is left out with the file it is."""
    described = {
        str(name): file
        for name, file in files.items()
        if isinstance(file, Mapping) and not _made_again(str(name)) and not _leaves(str(name), file)
    }
    links = {
        name: file["link"]
        for name, file in described.items()
        if isinstance(file.get("link"), str) and not os.path.isabs(file["link"])
    }
    removed = sorted(str(name) for name in entry.get("removed") or ())
    if not (described or removed):
        return None
    document = {
        "package": entry.get("package"),
        "revision": entry.get("revision"),
        "removed": removed,
        "executable": {
            name: file["executable"]
            for name, file in sorted(described.items())
            if isinstance(file.get("executable"), bool)
        },
        "links": dict(sorted(links.items())),
    }
    return (json.dumps(document, indent=2) + "\n").encode()


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
    requirement: Requirement,
    store: Store,
    owner: str,
    files: object,
    root: str,
    pieces: list[Piece],
    lost: list[Lost],
) -> None:
    """The kept files of a geometry or an evaluator (``root``: the directory their names are
    relative to), as one unit; one found outside the geometry's tree is named by its content, as
    the snapshot's geometry names it by its SHA-256, so equal ones are one file."""
    for name, file in (files if isinstance(files, Mapping) else {}).items():
        if not isinstance(file, Mapping):
            continue
        shown, role, path = str(name), "file", os.path.join(root, str(name))
        if os.path.isabs(shown):
            shown = f"externals/{str(file.get('sha256'))[:16]}/{os.path.basename(shown)}"
            role, path = "external", str(name)
        data = _kept(store, file.get("kept"))
        if data is None:
            lost.append(Lost((requirement.id, role, shown), (requirement.id,)))
            continue
        pieces.append(
            Piece(
                "code",
                owner,
                shown,
                data,
                (requirement.id,),
                requirement.used_by,
                role=cast(Role, role),
                paths=(path, os.path.realpath(path)),
            )
        )


def _aliases(requirement: Requirement) -> Piece | None:
    """The other names of a geometry's files, which its references can use, as a file; those that
    name a place of the author's are left out."""
    aliases = {
        str(name): target
        for name, target in ((requirement.entry or {}).get("aliases") or {}).items()
        if isinstance(target, str) and not os.path.isabs(str(name)) and not os.path.isabs(target)
    }
    if not aliases:
        return None
    data = (json.dumps(aliases, indent=2, sort_keys=True) + "\n").encode()
    return Piece(
        "code",
        "geometry",
        "aliases.json",
        data,
        (requirement.id,),
        requirement.used_by,
        role="generated",
    )


def _from_environment(
    requirement: Requirement, reads: _Reads, pieces: list[Piece], held: list[Held]
) -> None:
    """A Python environment, as the list of its distributions from an index (``requirements.txt``,
    for ``pip install -r``): what a rerun recreates it from. Offered if a stage used it, or if that
    is not known; its files are not kept, and neither is where it is. One installed from a local
    path or a repository is not on the list, and the environment is then not all of it."""
    listed = (requirement.source or {}).get("distributions")
    if not isinstance(listed, (list, tuple)) or not listed:
        return
    data = "".join(f"{name}\n" for name in sorted(map(str, listed), key=str.lower)).encode()
    piece = Piece(
        "code",
        "python-environment",
        "requirements.txt",
        data,
        (requirement.id,),
        requirement.used_by,
        role="generated",
    )
    if requirement.used_by or not reads.complete:
        pieces.append(piece)
    else:
        why = "not used by any stage"
        held.append(Held("code", piece.owner, piece.owner, why, requirement.id, False, (piece,)))


_CREDENTIALS = (
    ".ssh/",
    ".gnupg/",
    ".aws/",
    ".kube/",
    ".docker/",
    ".netrc",
    ".pgpass",
    ".git-credentials",
)
"""What of a home directory holds the account's credentials: never offered, whatever read it."""


def _from_input(
    requirement: Requirement,
    store: Store,
    geometry_root: str | None,
    pieces: list[Piece],
    held: list[Held],
    lost: list[Lost],
) -> None:
    entry = requirement.entry
    assert entry is not None
    path = str(entry.get("path"))
    name = os.path.basename(path)
    data = _kept(store, entry.get("kept"))
    kept = isinstance(entry.get("kept"), str)
    paths = (path, os.path.realpath(path))
    if entry.get("origin") == "geometry":  # a file a candidate's reference reached: code that ran
        if geometry_root is not None and path.startswith(f"{geometry_root}/"):
            shown, role = os.path.relpath(path, geometry_root), "file"
        else:
            digest = str(entry.get("identity", "")).removeprefix("sha256:")
            shown, role = f"externals/{digest[:16]}/{name}", "external"
        if data is None:
            if kept:
                lost.append(Lost((requirement.id, role, shown), (requirement.id,)))
            return
        pieces.append(
            Piece(
                "code",
                "geometry",
                shown,
                data,
                (requirement.id,),
                requirement.used_by,
                role=cast(Role, role),
                paths=paths,
            )
        )
        return
    if data is None:
        if kept:
            lost.append(Lost((requirement.id, "file", name), (requirement.id,)))
        return
    piece = Piece(
        "inputs", "", name, data, (requirement.id,), requirement.used_by, place=path, paths=paths
    )
    if name in _TOOLS:
        why = "an environment's marker"
        held.append(Held("inputs", "", name, why, requirement.id, False, (piece,)))
    elif os.path.relpath(path, os.path.expanduser("~")).startswith(_CREDENTIALS):
        why = "credentials of the account"
        held.append(Held("inputs", "", name, why, requirement.id, True, (piece,)))
    else:  # other home configuration (.rootrc) is an input like any, asked for with the inputs
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
    "distributions not installed from an index",
)
"""The notes of a requirement that hold no local path; any other is not published."""


def still_public(
    release: Release, published: Iterable[tuple[PublishedArtifact, bytes]]
) -> tuple[list[PublishedArtifact], dict[tuple[str, str, str], bytes]]:
    """What of ``published`` (files an earlier publication holds, with their bytes) is still of this
    study though ``release`` did not approve it: those files, and their bytes by identity
    (:attr:`Piece.key`). A publication is never taken back, so what it holds stays readable and is
    not said to be private, however the author answers this time; but only what the study still
    has: a file its unit holds now with the same bytes (a patch against the same commit), or one the
    study still needs whose kept copy is gone (:class:`Lost`), which the checked published copy
    stands in for. A file its unit no longer holds belongs to another state of it, and is dropped;
    so is one of another unit, even of equal bytes."""
    offer_ = release.offer
    now = {piece.key: piece for piece in offer_.everything}
    gone = {item.key: item for item in offer_.lost}
    approved = {piece.key for piece in release.pieces}
    kept: dict[tuple[str, str, str], PublishedArtifact] = {}
    earlier: dict[tuple[str, str, str], bytes] = {}
    for artifact, data in published:
        if not artifact.requirements:
            continue
        key = (artifact.requirements[0], artifact.kind, artifact.name)
        if key in approved or key in kept:
            continue
        piece = now.get(key)
        if piece is not None:
            if (piece.data, piece.base) != (data, artifact.base):
                continue
            artifact = replace(artifact, requirements=piece.requirements, stages=piece.stages)
        elif key in gone:
            if gone[key].base is not None and gone[key].base != artifact.base:
                continue
            artifact = replace(artifact, requirements=gone[key].requirements)
        else:
            continue
        kept[key], earlier[key] = artifact, data
    return list(kept.values()), earlier


def public_form(
    results: StudyResults,
    release: Release,
    snapshot: str,
    first: tuple[str, str] | None,
    artifacts: Sequence[PublishedArtifact],
    earlier: Mapping[tuple[str, str, str], bytes] | None = None,
) -> SoftwareSheet:
    """The sheet of ``results`` in the state of ``snapshot`` (a SHA-256): the public form of its
    requirements as checked just now, with what ``release`` approved. ``first`` is the time and
    status of the first publication of this snapshot, if it was published before; ``earlier`` the
    files an earlier publication already made public, by unit and name (:func:`still_public`)."""
    readiness = release.offer.readiness
    assert readiness is not None
    published = {piece.key: piece.data for piece in release.pieces} | dict(earlier or {})
    access = _access(readiness, release, published)
    public = tuple(_requirement(r, access[r.id], published) for r in readiness.requirements)
    status = _status(readiness, access)
    return SoftwareSheet(
        study=results.id,
        snapshot=snapshot,
        status=status,
        checked=readiness.at,
        first_checked=first[0] if first else readiness.at,
        first_status=first[1] if first else status,
        gaps=readiness.public_gaps,
        altered=_altered(readiness.altered),
        not_kept=len(readiness.problems),
        on_request=tuple(
            sorted(
                {
                    piece.owner or piece.name
                    for piece in release.offer.excluded
                    if piece.key not in published
                }
            )
        ),
        requirements=public,
        artifacts=tuple(artifacts),
    )


def _altered(names: Iterable[str]) -> tuple[str, ...]:
    """What was altered, in public: a kept state is named by the digest of its private content."""
    said = (
        "a kept record of what the study ran on" if n.startswith("software-") else n for n in names
    )
    return tuple(dict.fromkeys(said))


def _status(readiness: Readiness, access: Mapping[str, Access]) -> str:
    """The study's status (:attr:`Readiness.status`), and whether it needs what only its author can
    give: what a reader may do is what is published, not what the study kept."""
    status = readiness.status
    private = any(access[r.id] == "on request" for r in readiness.requirements if not r.unused)
    if not private:
        return status  # what blocks or is unknown is said first; what is private, beside it
    return NEEDS_ACCESS if status.startswith(ALL_AVAILABLE) else f"{status} · needs access"


def _access(
    readiness: Readiness, release: Release, published: Mapping[tuple[str, str, str], bytes]
) -> dict[str, Access]:
    """Who can get each requirement: ``public`` if everything of it a rerun needs is published
    (now or before) or public by nature, ``on request`` if some of it is the author's alone,
    ``none`` if something of it is gone and no published copy stands in for it. A build needs its
    source and its options as well: options published in part leave the rest to ask for; an
    environment with distributions not from an index needs those from the author."""
    offer_ = release.offer
    needed = [
        *offer_.code,
        *offer_.inputs,
        *offer_.large,
        *offer_.excluded,
        *(p for h in offer_.held if h.needed for p in h.pieces),
    ]
    found: dict[str, Access] = {}
    for r in readiness.requirements:
        mine = [p for p in needed if r.id in p.requirements]
        gone = [item.key for item in offer_.lost if r.id in item.requirements]
        if any(key not in published for key in gone) or (r.missing and not gone):
            found[r.id] = "none"  # whatever else is published of it, a rerun cannot have all of it
            continue
        withheld = any(p.key not in published for p in mine)
        if r.kind == "package":
            if not (r.source or {}).get("rebuildable"):
                found[r.id] = "none"  # no source a rebuild could start from
            elif withheld or _configuration(r, published) in ("partial", "on request"):
                found[r.id] = "on request"
            else:
                found[r.id] = "public"
        elif r.kind == "environment":
            if not mine:
                found[r.id] = _unoffered(r)
            elif withheld or (r.source or {}).get("unlisted"):
                found[r.id] = "on request"
            else:
                found[r.id] = "public"
        elif withheld:
            found[r.id] = "on request"
        elif mine or gone or any(key[0] == r.id for key in published):
            found[r.id] = "public"
        else:
            found[r.id] = _unoffered(r)
    return found


def _unoffered(requirement: Requirement) -> Access:
    """Who can get a requirement of which nothing is published: the stack's own is public; what
    was not kept, or is no file to give, no one."""
    entry = requirement.entry or {}
    if requirement.id.startswith("listing:") or requirement.unused:
        return "none"
    if requirement.kind == "environment":  # no list of its distributions to give
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


def _configuration(
    requirement: Requirement, published: Mapping[tuple[str, str, str], bytes]
) -> str:
    """Whether a reader can have what a build was configured with, all of it or a part (its options
    that hold a path or a credential are left out, and counted in its first line): its source alone
    rebuilds it with other options."""
    entry = requirement.entry or {}
    options = published.get((requirement.id, "generated", OPTIONS))
    if options is not None:
        return "partial" if options.startswith(b"#") else "published"
    return "on request" if entry.get("configuration") else "none"


def _requirement(
    requirement: Requirement, access: Access, published: Mapping[tuple[str, str, str], bytes]
) -> PublishedRequirement:
    facts = _facts(requirement)
    if requirement.kind == "package":
        facts["configuration"] = _configuration(requirement, published)
    tool = requirement.kind == "input" and requirement.label in _TOOLS
    return PublishedRequirement(
        id=requirement.id if not requirement.id.startswith("listing:") else "listing",
        kind=requirement.kind,
        label=_label(requirement),
        identified=_identified(requirement),
        used_by=requirement.used_by,
        state=_worst(requirement),
        access=access,
        notes=_public_notes(requirement),
        facts=FrozenMap(facts),
        ran=_ran(requirement),
        unused=requirement.unused or tool,
    )


def bundled(study: str, pieces: Iterable[Piece]) -> list[tuple[PublishedArtifact, bytes]]:
    """``pieces`` as the files published: each on its own, a patch with the files it changes."""
    found: list[tuple[PublishedArtifact, bytes]] = []
    for piece in pieces:
        if piece.is_patch:
            files = tuple(
                PublishedFile(name, added, removed) for name, added, removed in piece.files
            )
        else:
            files = (PublishedFile(piece.name, piece.added, piece.removed),)
        kind: Kind = piece.kind
        found.append(
            _artifact(
                study,
                piece.group,
                piece.owner,
                piece.name,
                kind,
                piece.data,
                piece.stages,
                piece.requirements,
                piece.base,
                files,
            )
        )
    return found


def _artifact(
    study: str,
    group: Group,
    owner: str,
    name: str,
    kind: Kind,
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
