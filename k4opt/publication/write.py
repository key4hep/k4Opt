"""Publishing a study: its snapshot and what its author approved first, then its catalog entry.

:func:`publish` writes the JSON form of a study that :func:`~k4opt.optimization.load_study` has
read and checked into a publication directory: the snapshot, under the name its content gives,
written whole and never over another file, the files its author approved
(:class:`~k4opt.publication.Release`), its software sheet, then the study's entry in
``catalog.json``, replaced whole under ``.catalog.lock``. A reader of the directory sees the catalog
before or after, and every file a catalog names is complete before it is named. The study directory
is only read.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from k4opt import jsonio
from k4opt.optimization import StudyResults
from k4opt.publication.artifacts import Release, bundled, offer, public_form
from k4opt.publication.catalog import (
    CATALOG_NAME,
    MAX_SHEET_BYTES,
    MAX_SNAPSHOT_BYTES,
    Catalog,
    CatalogEntry,
    Snapshot,
    sheet_path,
    snapshot_path,
)
from k4opt.publication.errors import PublicationError
from k4opt.publication.sheet import PublishedArtifact, SoftwareSheet

LOCK_NAME = ".catalog.lock"
PUBLIC_PREFIXES = ("/cvmfs/",)
"""Absolute paths that are public by nature, which :func:`local_paths` does not report."""

# an absolute path in text: a / that starts the text or follows a space, a quote, = or (, then
# what a path holds; "1/mm", "GeV/c" and "${K4GEO}/x" are no absolute paths
_ABSOLUTE = re.compile(r"(?:^|(?<=[\s\"'=(]))/[\w.+-][^\s\"',;)]*")


def publish(
    results: StudyResults,
    root: str | os.PathLike[str],
    *,
    warn: Callable[[str], None] | None = None,
    release: Release | None = None,
) -> CatalogEntry:
    """Publish ``results`` in the publication directory ``root``; its catalog entry.

    ``root`` is made if it does not exist, and must not be inside the study directory. ``warn``
    is told, before anything is written, of each absolute path (outside ``/cvmfs``) that the
    snapshot carries in the problem's definition or a trial's message (:func:`local_paths`):
    what of the machine it was made on leaves with it. ``release`` is what its author approved of
    an :func:`~k4opt.publication.offer` of this study; without one, nothing but the snapshot and the
    software sheet is published. Publishing the same state of a study again checks what a rerun
    needs again and adds no snapshot: its software sheet says when it was last checked. Raises
    :class:`PublicationError` if the catalog is malformed, another publication holds its lock, or
    the catalog's entry of this study is further along (more
    trials, or more of them ended): a publisher that read the study earlier never takes it back
    to that state.
    """
    directory = Path(os.path.abspath(Path(root).expanduser()))
    study = results.directory
    if study is not None and Path(os.path.realpath(directory)).is_relative_to(
        os.path.realpath(study)
    ):
        raise PublicationError(
            f"{directory} is in the study directory {study}; publishing adds nothing to a study, "
            "so publish somewhere else"
        )
    for where, path in local_paths(results):
        (warn or (lambda message: None))(f"{where} holds the local path {path}; it is published")
    text = results.to_json()
    data = text.encode()
    if len(data) > MAX_SNAPSHOT_BYTES:  # a reader refuses it: none is written
        raise PublicationError(
            f"the snapshot of study {results.name} would be {len(data)} bytes, more than the "
            f"{MAX_SNAPSHOT_BYTES} a publication holds ({len(results.trials)} trials)"
        )
    sha256 = hashlib.sha256(data).hexdigest()
    snapshot = Snapshot(snapshot_path(results.id, sha256), sha256, len(data))
    approved = release if release is not None else Release(offer(results))
    _write_once(directory / snapshot.path, data)
    artifacts = bundled(results.id, approved.pieces)
    for artifact, content in artifacts:
        _write_once(directory / artifact.path, content)
    with _locked(directory):
        catalog = _read_catalog(directory)
        current = catalog.study(results.id)
        software = _sheet(directory, results, approved, sha256, current, [a for a, _ in artifacts])
        if current is not None and current.snapshot == snapshot:
            if current.software == software:
                return current
            entry = replace(current, software=software)
            _replace_catalog(directory, catalog.with_entry(entry))
            return entry
        entry = CatalogEntry.of(
            results, snapshot, datetime.now(UTC).isoformat(timespec="seconds"), software
        )
        if current is not None and (
            entry.trials < current.trials or entry.finished < current.finished
        ):
            raise PublicationError(
                f"{directory / CATALOG_NAME} holds a later state of study {results.name}: "
                f"{current.trials} trials, {current.finished} ended (published "
                f"{current.published}), not {entry.trials} and {entry.finished}; read the study "
                "again and publish what it holds now"
            )
        _replace_catalog(directory, catalog.with_entry(entry))
    return entry


def _replace_catalog(directory: Path, catalog: Catalog) -> None:
    try:
        jsonio.replace(directory / CATALOG_NAME, catalog.to_json())
    except OSError as exc:
        raise PublicationError(f"cannot write {directory / CATALOG_NAME}: {exc}") from exc


def _sheet(
    directory: Path,
    results: StudyResults,
    release: Release,
    snapshot: str,
    current: CatalogEntry | None,
    artifacts: list[PublishedArtifact],
) -> Snapshot | None:
    """Write the software sheet of ``results``, of the snapshot ``snapshot`` and what ``release``
    approved, keeping from the sheet ``current`` names of the same snapshot when it was first
    published and what it was found then; where it is. ``None`` for a study with no directory."""
    if release.offer.readiness is None:
        return None
    first = None
    if current is not None and current.software and current.snapshot.sha256 == snapshot:
        first = _first(directory, results.id, snapshot, current)
    text = public_form(results, release, snapshot, first, artifacts).to_json()
    data = text.encode()
    if len(data) > MAX_SHEET_BYTES:  # a reader refuses it: none is written
        raise PublicationError(
            f"the software sheet of study {results.name} would be {len(data)} bytes, more than "
            f"the {MAX_SHEET_BYTES} a publication holds"
        )
    digest = hashlib.sha256(data).hexdigest()
    sheet = Snapshot(sheet_path(results.id, digest), digest, len(data))
    _write_once(directory / sheet.path, data)
    return sheet


def _first(
    directory: Path, study: str, snapshot: str, current: CatalogEntry
) -> tuple[str, str] | None:
    """When the sheet ``current`` names says this snapshot was first published, and its status."""
    assert current.software is not None
    try:
        previous = SoftwareSheet.from_json(
            (directory / current.software.path).read_bytes(),
            study,
            snapshot,
            str(directory / current.software.path),
        )
    except (OSError, PublicationError):
        return None  # a missing or broken earlier sheet: the check made now is the first known
    return previous.first_checked, previous.first_status


def local_paths(results: StudyResults) -> list[tuple[str, str]]:
    """Each absolute path, outside :data:`PUBLIC_PREFIXES`, that the JSON form of ``results``
    carries where a person writes text: in the problem's definition (a stage's ``file``, an
    evaluator setting) and in the trials' messages and diagnostics, with where it is."""
    found: list[tuple[str, str]] = []

    def scan(value: object, where: str) -> None:
        if isinstance(value, str):
            found.extend(
                (where, path)
                for path in _ABSOLUTE.findall(value)
                if not path.startswith(PUBLIC_PREFIXES)
            )
        elif isinstance(value, Mapping):
            for key, item in value.items():
                scan(key, where)
                scan(item, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                scan(item, f"{where}[{index}]")

    scan(results.problem.to_dict(), "the problem's definition")
    for trial in results.trials:
        scan(trial.message, f"trial {trial.number}'s message")
        for diagnostic in trial.diagnostics:
            scan(diagnostic["message"], f"trial {trial.number}'s diagnostic {diagnostic['task']}")
    return found


def _write_once(path: Path, data: bytes) -> None:
    """``data`` at ``path``, written whole; a file there already must hold exactly ``data``."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        jsonio.publish(path, data)
    except FileExistsError:
        pass
    except OSError as exc:
        raise PublicationError(f"cannot write {path}: {exc}") from exc
    else:
        return
    try:
        held = path.read_bytes()
    except OSError as exc:
        raise PublicationError(f"cannot read {path}: {exc}") from exc
    if held != data:
        raise PublicationError(
            f"{path} exists with other bytes than the content it is named after; such a file "
            "is never replaced, so move it away once nothing reads it"
        )


def _read_catalog(directory: Path) -> Catalog:
    """The catalog of the publication; an empty one where there is none yet."""
    path = directory / CATALOG_NAME
    try:
        text = path.read_bytes()
    except FileNotFoundError:
        return Catalog()
    except OSError as exc:
        raise PublicationError(f"cannot read {path}: {exc}") from exc
    try:
        return Catalog.from_json(text)
    except PublicationError as exc:
        raise PublicationError(f"{path} is left as it is: {exc}") from None


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """``.catalog.lock`` while the catalog is read and rewritten, so that two publications never
    lose each other's entry; one left by a publication that was killed is never taken over."""
    path = directory / LOCK_NAME
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
    except FileExistsError:
        raise PublicationError(
            f"{path} exists: another publication is updating the catalog, or one was killed "
            f"while it did; once none runs, remove {path} and publish again"
        ) from None
    except OSError as exc:
        raise PublicationError(f"cannot create {path}: {exc}") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
