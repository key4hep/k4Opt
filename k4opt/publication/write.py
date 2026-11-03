"""Publishing a study: its snapshot first, then its entry in the catalog.

:func:`publish` writes the JSON form of a study that :func:`~k4opt.optimization.load_study` has
read and checked into a publication directory: the snapshot, under the name its content gives,
written whole and never over another file, then the study's entry in ``catalog.json``, replaced
whole under ``.catalog.lock``. A reader of the directory sees the catalog before or after, and
every snapshot a catalog names is complete before it is named. The study directory is only read.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

from k4opt import jsonio
from k4opt.optimization import StudyResults
from k4opt.publication.catalog import (
    CATALOG_NAME,
    Catalog,
    CatalogEntry,
    Snapshot,
    snapshot_path,
)
from k4opt.publication.errors import PublicationError

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
) -> CatalogEntry:
    """Publish ``results`` in the publication directory ``root``; its catalog entry.

    ``root`` is made if it does not exist, and must not be inside the study directory. ``warn``
    is told, before anything is written, of each absolute path (outside ``/cvmfs``) that the
    snapshot carries in the problem's definition or a trial's message (:func:`local_paths`):
    what of the machine it was made on leaves with it. Publishing the same state of a study again
    changes nothing. Raises :class:`PublicationError` if the catalog is malformed, another
    publication holds its lock, or the catalog's entry of this study is further along (more
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
    sha256 = hashlib.sha256(data).hexdigest()
    snapshot = Snapshot(snapshot_path(results.id, sha256), sha256, len(data))
    _write_snapshot(directory / snapshot.path, text, data)
    with _locked(directory):
        catalog = _read_catalog(directory)
        current = catalog.study(results.id)
        if current is not None and current.snapshot == snapshot:
            return current
        entry = CatalogEntry.of(results, snapshot, datetime.now(UTC).isoformat(timespec="seconds"))
        if current is not None and (
            entry.trials < current.trials or entry.finished < current.finished
        ):
            raise PublicationError(
                f"{directory / CATALOG_NAME} holds a later state of study {results.name}: "
                f"{current.trials} trials, {current.finished} ended (published "
                f"{current.published}), not {entry.trials} and {entry.finished}; read the study "
                "again and publish what it holds now"
            )
        try:
            jsonio.replace(directory / CATALOG_NAME, catalog.with_entry(entry).to_json())
        except OSError as exc:
            raise PublicationError(f"cannot write {directory / CATALOG_NAME}: {exc}") from exc
    return entry


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


def _write_snapshot(path: Path, text: str, data: bytes) -> None:
    """The snapshot at ``path``, written whole; one there already must hold exactly ``data``."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        jsonio.publish(path, text)
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
            f"{path} exists with other bytes than the snapshot it is named after; a snapshot is "
            "never replaced, so move it away once nothing reads it"
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
