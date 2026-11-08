"""Reading a publication: from a directory or over HTTP(S), the same files and the same checks.

:func:`open_publication` takes where a publication is: a directory, or the ``http://`` or
``https://`` URL it is served at (by WebEOS, any static web server, an object store's web
endpoint). Either is read file by file, by the paths its catalog gives, never by listing a
directory, and no catalog entry can name a file outside the publication or another URL: its paths
are checked to be the ones the entry's own id and SHA-256 give
(:class:`~k4opt.publication.CatalogEntry`). Over HTTP, TLS certificates are verified, a
redirect is refused (a login page is no catalog) and every read is bounded in time and size. A
snapshot must be exactly the bytes its entry names, so a partial copy is never read as a study.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

from k4opt import __version__
from k4opt.optimization import OptimizationError, StudyResults
from k4opt.publication.catalog import (
    CATALOG_NAME,
    MAX_CATALOG_BYTES,
    Catalog,
    CatalogEntry,
    Snapshot,
    loads,
)
from k4opt.publication.errors import (
    NothingPublishedError,
    PublicationError,
    PublicationUnavailableError,
)
from k4opt.publication.sheet import PublishedArtifact, SoftwareSheet

TIMEOUT = 30.0
"""Seconds a server may take to connect, and between two parts of an answer."""
DEADLINE = 120.0
"""Seconds a whole answer may take to arrive, however steadily it trickles in (at most one
:data:`TIMEOUT` more, while the last part is awaited)."""
_CHUNK = 2**16


class _MissingFileError(Exception):
    """The file is not there: an answer, unlike a server that does not answer."""


class _Source(Protocol):
    location: str

    def read(self, relative: str, limit: int) -> bytes:
        """The file at ``relative`` (a path its catalog gives, or ``catalog.json``), at most
        ``limit`` bytes; :class:`_MissingFileError` if it is not there."""
        ...


class Publication:
    """A publication, read where it is: :meth:`catalog` says which studies it holds,
    :meth:`study` reads one of them. Nothing is cached here; every call reads again."""

    def __init__(self, source: _Source) -> None:
        self._source = source

    @property
    def location(self) -> str:
        """The directory or URL it is read from."""
        return self._source.location

    def catalog(self) -> Catalog:
        """The catalog, checked. Raises :class:`NothingPublishedError` if there is none,
        :class:`PublicationUnavailableError` if it cannot be read just now and
        :class:`PublicationError` if it is malformed."""
        try:
            data = self._source.read(CATALOG_NAME, MAX_CATALOG_BYTES)
        except _MissingFileError:
            raise NothingPublishedError(
                f"nothing is published at {self.location}: it holds no {CATALOG_NAME}"
            ) from None
        return Catalog.from_json(data)

    def study(self, entry: CatalogEntry) -> StudyResults:
        """The study ``entry`` names, read from its snapshot and checked: its bytes are exactly
        those the entry names, it reads back as the JSON form of a study
        (:meth:`~k4opt.optimization.StudyResults.from_dict`), and it is the study the entry
        describes. Raises :class:`PublicationUnavailableError` if it cannot be read just now or
        arrives short, and :class:`PublicationError` if it is missing or does not hold up."""
        where = f"{self.location}: {entry.snapshot.path}"
        data = self._checked(entry.snapshot, entry.snapshot.path)
        try:
            results = StudyResults.from_dict(loads(data, where))
        except OptimizationError as exc:
            raise PublicationError(f"{where}: {exc}") from None
        entry.check(results)
        return results

    def software(self, entry: CatalogEntry) -> SoftwareSheet | None:
        """The software sheet ``entry`` names, read and checked; ``None`` if it has none (a study
        published from its JSON form). Raises what :meth:`study` raises for a file."""
        if entry.software is None:
            return None
        data = self._checked(entry.software, entry.software.path)
        where = f"{self.location}: {entry.software.path}"
        return SoftwareSheet.from_json(data, entry.id, entry.snapshot.sha256, where)

    def artifact(self, artifact: PublishedArtifact) -> bytes:
        """The bytes of the file ``artifact`` names, exactly as its sheet says. Raises what
        :meth:`study` raises for a file."""
        where = Snapshot(artifact.path, artifact.sha256, artifact.bytes)
        return self._checked(where, artifact.path)

    def _checked(self, held: Snapshot, path: str) -> bytes:
        """The bytes of the file ``held`` names, exactly its size and SHA-256."""
        where = f"{self.location}: {path}"
        try:
            data = self._source.read(path, held.bytes)
        except _MissingFileError:
            raise PublicationError(
                f"{where} is not there, though the catalog names it: it was moved away, or the "
                "publication was copied in part"
            ) from None
        if len(data) < held.bytes:
            raise PublicationUnavailableError(
                f"{where} arrived with {len(data)} of its {held.bytes} bytes"
            )
        if hashlib.sha256(data).hexdigest() != held.sha256:
            raise PublicationError(f"{where} is not the file its catalog names (other bytes)")
        return data


def open_publication(location: str | os.PathLike[str]) -> Publication:
    """The publication at ``location``: an ``http://`` or ``https://`` URL, or a directory.
    Nothing is read yet. Raises :class:`PublicationError` for a URL of another kind."""
    text = os.fspath(location)
    scheme = urllib.parse.urlsplit(text).scheme if "://" in text else ""
    if scheme in ("http", "https"):
        return Publication(_Http(text))
    if scheme:
        raise PublicationError(
            f"{text}: a publication is read from a directory or an http(s) URL, not {scheme}://"
        )
    return Publication(_Local(Path(text).expanduser()))


class _Local:
    """A publication in a directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(root))
        self.location = str(self.root)

    def read(self, relative: str, limit: int) -> bytes:
        path = self.root / relative
        if not self.root.is_dir():
            raise NothingPublishedError(f"nothing is published at {self.root}: it is no directory")
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise PublicationError(f"{path} leads out of the publication {self.root}")
        try:
            with path.open("rb") as handle:
                data = handle.read(limit + 1)
        except FileNotFoundError:
            raise _MissingFileError from None
        except OSError as exc:
            raise PublicationUnavailableError(f"cannot read {path}: {exc.strerror or exc}") from exc
        if len(data) > limit:
            raise PublicationError(f"{path} is larger than the {limit} bytes it may be")
        return data


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None  # the 3xx answer is raised as it is


_OPENER = urllib.request.build_opener(_NoRedirect)


class _Http:
    """A publication served at an HTTP(S) URL."""

    def __init__(self, url: str) -> None:
        parts = urllib.parse.urlsplit(url)
        if not parts.netloc or parts.query or parts.fragment:
            raise PublicationError(f"{url}: a publication's URL names a host and no query")
        self.location = url.rstrip("/")

    def read(self, relative: str, limit: int) -> bytes:
        url = f"{self.location}/{relative}"  # a checked path: letters, digits, - . /
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",  # the catalog changes; a snapshot never does
                "User-Agent": f"k4opt/{__version__}",
            },
        )
        try:
            deadline = time.monotonic() + DEADLINE
            with _OPENER.open(request, timeout=TIMEOUT) as response:
                parts: list[bytes] = []
                size = 0
                while size <= limit:
                    if time.monotonic() > deadline:
                        raise PublicationUnavailableError(
                            f"{url} did not arrive within {DEADLINE:g} s ({size} bytes so far)"
                        )
                    part: bytes = response.read(min(_CHUNK, limit + 1 - size))
                    if not part:
                        break
                    parts.append(part)
                    size += len(part)
                data = b"".join(parts)
        except urllib.error.HTTPError as exc:
            exc.close()  # an error answer holds its connection too
            if exc.code in (404, 410):
                raise _MissingFileError from None
            if 300 <= exc.code < 400:
                raise PublicationError(
                    f"{url} redirects to {exc.headers.get('Location')}: a publication is read "
                    "where it is served, and a login page is no catalog; serve it without one"
                ) from None
            if exc.code >= 500 or exc.code in (408, 429):
                raise PublicationUnavailableError(f"{url}: the server answers {exc.code}") from None
            raise PublicationError(f"{url}: the server answers {exc.code} {exc.reason}") from None
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            raise PublicationUnavailableError(f"{url} cannot be read just now: {reason}") from None
        if len(data) > limit:
            raise PublicationError(f"{url} is larger than the {limit} bytes it may be")
        return data
