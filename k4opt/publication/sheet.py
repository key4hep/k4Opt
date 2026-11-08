"""A study's *software sheet*: the public form of what a rerun of it needs.

A snapshot says what a study found; its sheet says what it ran on and what a rerun would need, as
much as may be read by anyone: the status, each requirement with its state and who can get it, and
the files published beside it. It is a file of its own, named by its content
(``studies/<id>/software/<sha256[:16]>.json``, :data:`SHEET_FORMAT`), since it changes with the date
of its check while the snapshot does not. Reading one is strict: it must be well formed, of the
study and the snapshot it is named for, and every artifact's path must be the one its study and
SHA-256 give, so that it names no other file and no URL.

What it holds is made field by field from the study's private records
(:func:`~k4opt.publication.artifacts.public_form`), never copied whole: no path, size, digest of an
unpublished file, variable name, host or user name is in it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, NoReturn, cast

from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.publication.catalog import loads
from k4opt.publication.errors import PublicationError

SHEET_FORMAT = "k4opt.software-sheet/1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

type Group = Literal["code", "inputs"]
type Access = Literal["public", "on request", "none"]
type State = Literal["available", "gone", "altered", "not-checked"]

_GROUPS: tuple[Group, ...] = ("code", "inputs")
_ACCESS: tuple[Access, ...] = ("public", "on request", "none")
_STATES: tuple[State, ...] = ("available", "gone", "altered", "not-checked")
_KINDS = ("patch", "file", "generated", "external")


def artifact_path(study: str, sha256: str) -> str:
    """Where the file of ``study`` with this SHA-256 is, relative to the publication."""
    if not _ID.fullmatch(study):
        raise PublicationError(f"{study!r} is not the id of a study (a UUID)")
    if not _SHA256.fullmatch(sha256):
        raise PublicationError(f"{sha256!r} is not a SHA-256")
    return f"studies/{study}/artifacts/{sha256[:16]}"


@dataclass(frozen=True, slots=True)
class PublishedFile:
    """One file of a published patch: its ``name`` and the lines it adds and removes (``None`` for
    a binary one)."""

    name: str
    added: int | None
    removed: int | None


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """A file published beside the snapshot: which ``group`` it was approved in, whose it is
    (``owner``: a package, a stage's job directory, an input's own name), its ``name``, whether it
    is a ``patch`` (against the commit ``base``, of the ``files``) or a whole ``file``, its
    ``sha256``, its size in ``bytes``, its ``path`` in the publication, the ``stages`` that used it
    and the ``requirements`` it answers."""

    group: Group
    owner: str
    name: str
    kind: Literal["patch", "file", "generated", "external"]
    sha256: str
    bytes: int
    path: str
    stages: tuple[str, ...]
    requirements: tuple[str, ...]
    base: str | None
    files: tuple[PublishedFile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "owner": self.owner,
            "name": self.name,
            "kind": self.kind,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "path": self.path,
            "stages": list(self.stages),
            "requirements": list(self.requirements),
            "base": self.base,
            "files": [
                {"name": file.name, "added": file.added, "removed": file.removed}
                for file in self.files
            ],
        }


@dataclass(frozen=True, slots=True)
class PublishedRequirement:
    """One thing a rerun needs, in public (:class:`~k4opt.software.readiness.Requirement`).

    ``state`` is what its check found (``available``, ``gone``, ``altered``, ``not-checked``),
    ``access`` who can get it: ``public`` from where it is or this publication, ``on request`` from
    the author, ``none`` if nothing was kept that anyone could be given. ``facts`` and ``ran`` say
    of a local build what is kept apart in the study: its source and what of it ran."""

    id: str
    kind: str
    label: str
    identified: str
    used_by: tuple[str, ...]
    state: State
    access: Access
    notes: tuple[str, ...]
    facts: Mapping[str, Json]
    ran: Mapping[str, Json] | None
    unused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "identified": self.identified,
            "used_by": list(self.used_by),
            "state": self.state,
            "access": self.access,
            "notes": list(self.notes),
            "facts": thaw(self.facts),
            "ran": None if self.ran is None else thaw(self.ran),
            "unused": self.unused,
        }


@dataclass(frozen=True, slots=True)
class SoftwareSheet:
    """What a study ran on and what a rerun needs, as published.

    ``study`` and ``snapshot`` (a SHA-256) say which study, in which state; ``status`` is the
    headline when it was ``checked``. ``first_checked`` and ``first_status`` are those of the first
    publication of this snapshot, which a later check does not change. ``gaps`` is what its records
    leave out, ``altered`` what it kept and is not what was written, ``not_kept`` how many things
    could not be kept, ``on_request`` the names of files the author left out."""

    study: str
    snapshot: str
    status: str
    checked: str
    first_checked: str
    first_status: str
    gaps: tuple[str, ...]
    altered: tuple[str, ...]
    not_kept: int
    on_request: tuple[str, ...]
    requirements: tuple[PublishedRequirement, ...]
    artifacts: tuple[PublishedArtifact, ...]

    def to_json(self) -> str:
        data = {
            "format": SHEET_FORMAT,
            "study": self.study,
            "snapshot": self.snapshot,
            "status": self.status,
            "checked": self.checked,
            "first_checked": self.first_checked,
            "first_status": self.first_status,
            "gaps": list(self.gaps),
            "altered": list(self.altered),
            "not_kept": self.not_kept,
            "on_request": list(self.on_request),
            "requirements": [requirement.to_dict() for requirement in self.requirements],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }
        return json.dumps(data, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: bytes | str, study: str, snapshot: str, where: str) -> SoftwareSheet:
        """The sheet ``text`` holds, checked, of ``study`` in the state of ``snapshot`` (a
        SHA-256): :class:`PublicationError` naming ``where`` if it is malformed in any part."""
        data = _object(loads(text, where), where, _SHEET)
        if data["format"] != SHEET_FORMAT:
            _refuse(f"{where}: format", f"is {data['format']!r}, not {SHEET_FORMAT!r}")
        if data["study"] != study or data["snapshot"] != snapshot:
            _refuse(where, "is not the sheet of the study and snapshot its catalog entry names")
        not_kept = data["not_kept"]
        if isinstance(not_kept, bool) or not isinstance(not_kept, int) or not_kept < 0:
            _refuse(f"{where}: not_kept", f"is {not_kept!r}, not a count")
        return cls(
            study=study,
            snapshot=snapshot,
            status=_text(data["status"], f"{where}: status"),
            checked=_time(data["checked"], f"{where}: checked"),
            first_checked=_time(data["first_checked"], f"{where}: first_checked"),
            first_status=_text(data["first_status"], f"{where}: first_status"),
            gaps=_texts(data["gaps"], f"{where}: gaps"),
            altered=_texts(data["altered"], f"{where}: altered"),
            not_kept=not_kept,
            on_request=_texts(data["on_request"], f"{where}: on_request"),
            requirements=tuple(
                _requirement(entry, f"{where}: requirements[{index}]")
                for index, entry in enumerate(_list(data["requirements"], f"{where}: requirements"))
            ),
            artifacts=_distinct(
                tuple(
                    _artifact(entry, study, f"{where}: artifacts[{index}]")
                    for index, entry in enumerate(_list(data["artifacts"], f"{where}: artifacts"))
                ),
                where,
            ),
        )


_SHEET = (
    "format",
    "study",
    "snapshot",
    "status",
    "checked",
    "first_checked",
    "first_status",
    "gaps",
    "altered",
    "not_kept",
    "on_request",
    "requirements",
    "artifacts",
)
_REQUIREMENT = (
    "id",
    "kind",
    "label",
    "identified",
    "used_by",
    "state",
    "access",
    "notes",
    "facts",
    "ran",
    "unused",
)
_ARTIFACT = (
    "group",
    "owner",
    "name",
    "kind",
    "sha256",
    "bytes",
    "path",
    "stages",
    "requirements",
    "base",
    "files",
)


def _distinct(
    artifacts: tuple[PublishedArtifact, ...], where: str
) -> tuple[PublishedArtifact, ...]:
    """``artifacts``, refused if two are one file of one unit: what each is must be clear."""
    seen = [(a.requirements[:1], a.kind, a.name) for a in artifacts]
    if len(set(seen)) != len(seen):
        _refuse(f"{where}: artifacts", "name one file of one unit twice")
    return artifacts


def _requirement(value: object, where: str) -> PublishedRequirement:
    data = _object(value, where, _REQUIREMENT)
    if not isinstance(data["unused"], bool):
        _refuse(f"{where}.unused", f"is {data['unused']!r}, not true or false")
    facts = freeze(_object(data["facts"], f"{where}.facts"), where, PublicationError)
    ran = data["ran"]
    if ran is not None:
        ran = freeze(_object(ran, f"{where}.ran"), where, PublicationError)
    return PublishedRequirement(
        id=_text(data["id"], f"{where}.id"),
        kind=_text(data["kind"], f"{where}.kind"),
        label=_text(data["label"], f"{where}.label"),
        identified=_text(data["identified"], f"{where}.identified"),
        used_by=_texts(data["used_by"], f"{where}.used_by"),
        state=cast(State, _one_of(data["state"], _STATES, f"{where}.state")),
        access=cast(Access, _one_of(data["access"], _ACCESS, f"{where}.access")),
        notes=_texts(data["notes"], f"{where}.notes"),
        facts=cast(Mapping[str, Json], facts),
        ran=cast("Mapping[str, Json] | None", ran),
        unused=data["unused"],
    )


def _artifact(value: object, study: str, where: str) -> PublishedArtifact:
    data = _object(value, where, _ARTIFACT)
    sha256, size = data["sha256"], data["bytes"]
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        _refuse(f"{where}.sha256", f"is {sha256!r}, not a SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        _refuse(f"{where}.bytes", f"is {size!r}, not a size")
    if data["path"] != artifact_path(study, sha256):
        _refuse(f"{where}.path", f"is {data['path']!r}, not the path its study and SHA-256 give")
    base = data["base"]
    return PublishedArtifact(
        group=cast(Group, _one_of(data["group"], _GROUPS, f"{where}.group")),
        owner=_text(data["owner"], f"{where}.owner"),
        name=_text(data["name"], f"{where}.name"),
        kind=cast(
            Literal["patch", "file", "generated", "external"],
            _one_of(data["kind"], _KINDS, f"{where}.kind"),
        ),
        sha256=sha256,
        bytes=size,
        path=data["path"],
        stages=_texts(data["stages"], f"{where}.stages"),
        requirements=_texts(data["requirements"], f"{where}.requirements"),
        base=None if base is None else _text(base, f"{where}.base"),
        files=tuple(
            _file(entry, f"{where}.files[{index}]")
            for index, entry in enumerate(_list(data["files"], f"{where}.files"))
        ),
    )


def _file(value: object, where: str) -> PublishedFile:
    data = _object(value, where, ("name", "added", "removed"))
    counts = [data["added"], data["removed"]]
    if not all(
        count is None or (isinstance(count, int) and not isinstance(count, bool) and count >= 0)
        for count in counts
    ):
        _refuse(where, f"has the line counts {counts!r}")
    return PublishedFile(_text(data["name"], f"{where}.name"), counts[0], counts[1])


def _refuse(where: str, problem: str) -> NoReturn:
    raise PublicationError(f"{where} {problem}")


def _object(value: object, where: str, keys: tuple[str, ...] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, (dict, FrozenMap)):
        _refuse(where, "is not an object")
    if keys is not None and set(value) != set(keys):
        _refuse(where, f"has the keys {sorted(value)}, not {sorted(keys)}")
    return value


def _list(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        _refuse(where, "is not a list")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str):
        _refuse(where, f"is {value!r}, not text")
    return value


def _time(value: object, where: str) -> str:
    text = _text(value, where)
    try:
        datetime.fromisoformat(text)
    except ValueError:
        _refuse(where, f"is {text!r}, no time")
    return text


def _texts(value: object, where: str) -> tuple[str, ...]:
    return tuple(_text(item, f"{where}[{index}]") for index, item in enumerate(_list(value, where)))


def _one_of(value: object, allowed: tuple[str, ...], where: str) -> str:
    if value not in allowed:
        _refuse(where, f"is {value!r}, not one of {list(allowed)}")
    return str(value)
