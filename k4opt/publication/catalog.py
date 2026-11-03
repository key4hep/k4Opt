"""The catalog of a publication: which studies it holds, each by its latest snapshot.

A publication is a directory, served as static files or copied anywhere::

    catalog.json                                       k4opt.catalog/1, the one file rewritten
    studies/<study id>/snapshots/<sha256[:16]>.json    each the JSON form of a study, never changed

A snapshot is the JSON form of a study (:meth:`~k4opt.optimization.StudyResults.to_json`), named
by its content, so a name stands for one content forever. The catalog has an entry for each study,
by the study's own id, with what a list of studies shows (its name, when it was created and
published, how many trials ended how, how many parameters, objectives and constraints) and the
snapshot it is published at: its path, SHA-256 and size. Reading one is strict: every entry must
be well formed, its path must be the one its id and SHA-256 give (so it names no other file and
no URL), and a malformed entry refuses the whole catalog, so that no study goes missing
unnoticed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from k4opt.frozen import FrozenMap
from k4opt.optimization import OUTCOMES, StudyResults
from k4opt.publication.errors import PublicationError

CATALOG_NAME = "catalog.json"
CATALOG_FORMAT = "k4opt.catalog/1"
MAX_CATALOG_BYTES = 16 * 2**20
"""The largest catalog read: some ten thousand studies."""
MAX_SNAPSHOT_BYTES = 512 * 2**20
"""The largest snapshot a catalog may name: a study of a few hundred thousand trials."""

_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENTRY = (
    "id",
    "name",
    "problem",
    "created",
    "published",
    "trials",
    "outcomes",
    "parameters",
    "objectives",
    "constraints",
    "snapshot",
)


def snapshot_path(study: str, sha256: str) -> str:
    """Where the snapshot of ``study`` with this SHA-256 is, relative to the publication; only a
    study's own id, a UUID, names a directory, so that no id leads out of the publication."""
    if not _ID.fullmatch(study):
        raise PublicationError(f"{study!r} is not the id of a study (a UUID)")
    if not _SHA256.fullmatch(sha256):
        raise PublicationError(f"{sha256!r} is not a SHA-256")
    return f"studies/{study}/snapshots/{sha256[:16]}.json"


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A published snapshot: its ``path`` in the publication, its ``sha256`` and its size in
    ``bytes``, which a reader checks it by."""

    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """A study of the catalog, as its latest published snapshot holds it.

    ``id``, ``name`` and ``created`` are the study's; ``problem`` the name of its problem (``None``
    if it has none); ``published`` when this snapshot was published (UTC, ISO 8601). ``trials``
    is how many trials the study had, ``outcomes`` how many ended how, every outcome a key: a
    ``running`` trial was running when it was published. ``parameters``, ``objectives`` and
    ``constraints`` count those of its problem.
    """

    id: str
    name: str
    problem: str | None
    created: str
    published: str
    trials: int
    outcomes: Mapping[str, int]
    parameters: int
    objectives: int
    constraints: int
    snapshot: Snapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", FrozenMap(self.outcomes))

    @property
    def finished(self) -> int:
        """How many of its trials had ended when it was published: all but the running ones."""
        return self.trials - self.outcomes["running"]

    @classmethod
    def of(cls, results: StudyResults, snapshot: Snapshot, published: str) -> CatalogEntry:
        """The entry of ``results``, published at ``snapshot`` at the time ``published``."""
        spec = results.problem
        return cls(
            id=results.id,
            name=results.name,
            problem=spec.name,
            created=results.created,
            published=published,
            trials=len(results.trials),
            outcomes=results.outcomes,
            parameters=len(spec.parameters),
            objectives=len(spec.objectives),
            constraints=len(spec.constraints),
            snapshot=snapshot,
        )

    def check(self, results: StudyResults) -> None:
        """Refuse ``results`` read from this entry's snapshot if it is not the study the entry
        describes, as the entry describes it."""
        found = CatalogEntry.of(results, self.snapshot, self.published)
        if found != self:
            differ = [key for key in _ENTRY if getattr(found, key) != getattr(self, key)]
            raise PublicationError(
                f"{self.snapshot.path} is not the study its catalog entry describes: its "
                f"{', '.join(differ)} differ"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "problem": self.problem,
            "created": self.created,
            "published": self.published,
            "trials": self.trials,
            "outcomes": dict(self.outcomes),
            "parameters": self.parameters,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "snapshot": self.snapshot.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object, where: str) -> CatalogEntry:
        """The entry ``data`` holds, checked; :class:`PublicationError` naming ``where``."""
        entry = _object(data, where, _ENTRY)
        study = entry["id"]
        if not isinstance(study, str) or not _ID.fullmatch(study):
            _refuse(f"{where}.id", f"is {study!r}, not the id of a study")
        where = f"the entry of study {study}"
        counts = {
            key: _count(entry[key], f"{where}: {key}")
            for key in ("trials", "parameters", "objectives", "constraints")
        }
        outcomes = _object(entry["outcomes"], f"{where}: outcomes", OUTCOMES)
        tally = {key: _count(value, f"{where}: outcomes.{key}") for key, value in outcomes.items()}
        if sum(tally.values()) != counts["trials"]:
            _refuse(
                f"{where}: outcomes", f"count {sum(tally.values())} trials, not {counts['trials']}"
            )
        snapshot = _object(entry["snapshot"], f"{where}: snapshot", ("path", "sha256", "bytes"))
        sha256, size = snapshot["sha256"], snapshot["bytes"]
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            _refuse(f"{where}: snapshot.sha256", f"is {sha256!r}, not a SHA-256")
        if _count(size, f"{where}: snapshot.bytes") > MAX_SNAPSHOT_BYTES:
            _refuse(f"{where}: snapshot.bytes", f"is {size}, more than {MAX_SNAPSHOT_BYTES}")
        if snapshot["path"] != snapshot_path(study, sha256):
            _refuse(
                f"{where}: snapshot.path",
                f"is {snapshot['path']!r}, not {snapshot_path(study, sha256)!r}, the path its id "
                "and SHA-256 give",
            )
        for key in ("name", "created", "published"):
            _text(entry[key], f"{where}: {key}")
        if entry["problem"] is not None:
            _text(entry["problem"], f"{where}: problem")
        return cls(
            id=study,
            name=entry["name"],
            problem=entry["problem"],
            created=entry["created"],
            published=entry["published"],
            outcomes=tally,
            snapshot=Snapshot(snapshot["path"], sha256, size),
            **counts,
        )


@dataclass(frozen=True, slots=True)
class Catalog:
    """The studies a publication holds, an entry each, by id."""

    studies: tuple[CatalogEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "studies", tuple(sorted(self.studies, key=lambda e: e.id)))

    def study(self, study: str) -> CatalogEntry | None:
        """The entry of the study with the id ``study``, ``None`` if it has none."""
        return next((entry for entry in self.studies if entry.id == study), None)

    def with_entry(self, entry: CatalogEntry) -> Catalog:
        """This catalog with ``entry`` in place of the entry of its study, or added."""
        return Catalog((*(e for e in self.studies if e.id != entry.id), entry))

    def to_json(self) -> str:
        data = {"format": CATALOG_FORMAT, "studies": [e.to_dict() for e in self.studies]}
        return json.dumps(data, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: bytes | str) -> Catalog:
        """The catalog ``text`` holds, checked: :class:`PublicationError` if it is malformed in
        any part."""
        data = loads(text, CATALOG_NAME)
        catalog = _object(data, CATALOG_NAME, ("format", "studies"))
        if catalog["format"] != CATALOG_FORMAT:
            _refuse(f"{CATALOG_NAME}: format", f"is {catalog['format']!r}, not {CATALOG_FORMAT!r}")
        studies = catalog["studies"]
        if not isinstance(studies, list):
            _refuse(f"{CATALOG_NAME}: studies", "is not a list")
        entries = [
            CatalogEntry.from_dict(entry, f"{CATALOG_NAME}: studies[{index}]")
            for index, entry in enumerate(studies)
        ]
        ids = [entry.id for entry in entries]
        if len(set(ids)) != len(ids):
            _refuse(f"{CATALOG_NAME}: studies", "name a study more than once")
        return cls(tuple(entries))


def loads(text: bytes | str, where: str) -> Any:
    """JSON as a publication holds it: no ``NaN`` or ``Infinity``, which JSON does not have."""

    def refused(constant: str) -> NoReturn:
        _refuse(where, f"holds {constant}, which is no JSON number")

    try:
        return json.loads(text, parse_constant=refused)
    except ValueError as exc:  # UnicodeDecodeError included
        raise PublicationError(f"{where} is not JSON: {exc}") from None


def _refuse(where: str, problem: str) -> NoReturn:
    raise PublicationError(f"{where} {problem}")


def _object(value: object, where: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _refuse(where, "is not an object")
    if set(value) != set(keys):
        _refuse(where, f"has the keys {sorted(value)}, not {sorted(keys)}")
    return value


def _count(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(where, f"is {value!r}, not a count")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str):
        _refuse(where, f"is {value!r}, not text")
    return value
