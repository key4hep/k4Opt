"""Where the dashboard reads its studies, and how what it read is kept.

``K4OPT_DASHBOARD_DATA`` names the publication (``docs/publication.md``): a directory or the
``https://`` URL it is served at. Unset or empty, the dashboard starts with nothing to show and
says how to set it.

Two kinds of data, cached two ways, for every viewer alike:

* the **catalog** changes whenever a study is published, so it is kept for :data:`CATALOG_TTL`
  seconds, and the sidebar's *Refresh* reads it again at once;
* a **snapshot** never changes, since its name is its content, so a study is kept by the path and
  SHA-256 of its snapshot, never stale; the latest read are kept while their snapshots add up to
  :data:`STUDY_BYTES` at most, so that memory stays bounded however large each study is.

What is kept is immutable (:class:`~k4opt.publication.Catalog`,
:class:`~k4opt.optimization.StudyResults`), so sharing it between viewers is safe. A read that
fails raises, and Streamlit keeps no exception: nothing of a failed or incomplete read is kept, and
the next interaction reads again.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import streamlit as st

from k4opt.errors import K4OptError
from k4opt.optimization import StudyResults
from k4opt.publication import (
    Catalog,
    CatalogEntry,
    NothingPublishedError,
    PublicationUnavailableError,
    PublishedArtifact,
    SoftwareSheet,
    open_publication,
)

from labels import plain

ENVIRONMENT = "K4OPT_DASHBOARD_DATA"
CATALOG_TTL = 60
STUDY_BYTES = 96 * 2**20
"""The studies kept, by the size of their snapshots: a study read back holds about 1.5 times its
snapshot's size (measured), so some 150 MB, within the 1 GiB the deployment allows."""


def location() -> str | None:
    """The publication the dashboard reads, ``None`` if none is configured."""
    return os.environ.get(ENVIRONMENT, "").strip() or None


@dataclass(frozen=True, slots=True)
class Read:
    """A catalog and when it was read."""

    catalog: Catalog
    at: datetime


@st.cache_resource(ttl=CATALOG_TTL, max_entries=4, show_spinner="Reading the catalog...")
def _catalog(where: str) -> Read:
    return Read(open_publication(where).catalog(), datetime.now(UTC))


_kept: OrderedDict[tuple[str, str, str], tuple[StudyResults, int]] = OrderedDict()
"""The studies read, by publication, snapshot path and SHA-256, the last used last, each with
its snapshot's size; for every viewer of this process."""
_keeping = threading.Lock()


def _held(key: tuple[str, str, str]) -> StudyResults | None:
    with _keeping:
        held = _kept.get(key)
        if held is None:
            return None
        _kept.move_to_end(key)
        return held[0]


def _keep(key: tuple[str, str, str], results: StudyResults, size: int) -> None:
    """Keep ``results``, then drop the least recently used until :data:`STUDY_BYTES` holds the
    rest; the one just read always stays, however large."""
    with _keeping:
        _kept[key] = (results, size)
        _kept.move_to_end(key)
        while len(_kept) > 1 and sum(size for _, size in _kept.values()) > STUDY_BYTES:
            _kept.popitem(last=False)


def refresh() -> None:
    """Read the catalog again on the next run."""
    _catalog.clear()


@dataclass(frozen=True, slots=True)
class Source:
    """What the dashboard reads this run: its ``location``, the catalog read from it, or why there
    is none (``problem``, shown at ``level``)."""

    location: str | None
    read: Read | None = None
    problem: str | None = None
    level: Literal["info", "warning", "error"] = "info"

    @property
    def catalog(self) -> Catalog | None:
        return None if self.read is None else self.read.catalog


def source() -> Source:
    """The catalog of the configured publication, or why there is none: nothing configured or
    published (info), a publication that cannot be read just now (warning), one that does not
    hold up (error)."""
    where = location()
    if where is None:
        return Source(
            None,
            problem=(
                f"No publication is configured. Set `{ENVIRONMENT}` to a publication directory or "
                "its URL, made with `k4opt publish STUDY --output DIR`, and start the dashboard "
                f"again: `{ENVIRONMENT}=published/ streamlit run dashboard/app.py`."
            ),
        )
    try:
        return Source(where, _catalog(where))
    except NothingPublishedError as exc:
        return Source(where, problem=f"Nothing is published here yet: {plain(exc)}.")
    except PublicationUnavailableError as exc:
        return Source(
            where,
            problem=f"The publication cannot be read just now ({plain(exc)}); it is read again on "
            "your "
            "next interaction, or with Refresh.",
            level="warning",
        )
    except K4OptError as exc:
        return Source(
            where, problem=f"The publication does not hold up: {plain(exc)}", level="error"
        )


_SOURCE = "k4opt.source"
KEPT = "k4opt.kept"
"""The ids of the studies the sidebar keeps this run, for the Studies page."""


def resolve() -> Source:
    """:func:`source`, read once per run by the app's entry point, before its page runs."""
    found = source()
    st.session_state[_SOURCE] = found
    return found


def current() -> Source:
    """What :func:`resolve` read in this run, for the page it runs."""
    found = st.session_state.get(_SOURCE)
    return found if isinstance(found, Source) else resolve()


def study(where: str, entry: CatalogEntry) -> StudyResults:
    """The study ``entry`` names, as its snapshot holds it (:meth:`Publication.study`), checked
    against ``entry`` on every run, also when it was read before for another entry naming the
    same snapshot; raises what that raises, and nothing of a failed read is kept."""
    key = (where, entry.snapshot.path, entry.snapshot.sha256)
    results = _held(key)
    if results is None:
        with st.spinner("Reading the study..."):
            results = open_publication(where).study(entry)
        _keep(key, results, entry.snapshot.bytes)
    entry.check(results)
    return results


@st.cache_resource(max_entries=64, show_spinner=False)
def _software(where: str, path: str, sha256: str, _entry: CatalogEntry) -> SoftwareSheet | None:
    return open_publication(where).software(_entry)


def software(where: str, entry: CatalogEntry) -> SoftwareSheet | None:
    """The software sheet of ``entry``, kept by its path and SHA-256 (its name is its content), or
    ``None`` if it has none; raises what reading it raises, and nothing of a failed read is kept."""
    if entry.software is None:
        return None
    return _software(where, entry.software.path, entry.software.sha256, entry)


@st.cache_resource(max_entries=32, show_spinner=False)
def _artifact(where: str, path: str, sha256: str, _artifact: PublishedArtifact) -> bytes:
    return open_publication(where).artifact(_artifact)


def artifact(where: str, published: PublishedArtifact) -> bytes:
    """The bytes of a published file, checked, kept by its path and SHA-256."""
    return _artifact(where, published.path, published.sha256, published)


def newest_first(catalog: Catalog) -> list[CatalogEntry]:
    """The studies of ``catalog``, the latest published first."""
    return sorted(catalog.studies, key=lambda entry: entry.published, reverse=True)


def show(found: Source) -> None:
    """Say why ``found`` has no catalog, or that its catalog names no study."""
    if found.problem is not None:
        {"info": st.info, "warning": st.warning, "error": st.error}[found.level](found.problem)
    elif found.catalog is not None and not found.catalog.studies:
        st.info(f"The catalog of {plain(found.location)} names no study yet.")
