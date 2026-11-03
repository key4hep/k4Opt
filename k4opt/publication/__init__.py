"""Publication component: studies as portable, read-only snapshots, and where to find them.

:func:`publish` writes a study that :func:`~k4opt.optimization.load_study` has read and checked
into a publication directory: a snapshot, the study's JSON form without any location
(:meth:`~k4opt.optimization.StudyResults.to_json`), named by its content and never changed, and
the study's entry in the directory's ``catalog.json``. The directory can be served as static
files (WebEOS, any web server) or copied anywhere; :func:`open_publication` reads it from a
directory or a URL alike, without Optuna, the study directory or the machine the study ran on.
See ``docs/publication.md``.

Typical use::

    from k4opt.optimization import load_study
    from k4opt.publication import open_publication, publish

    publish(load_study("runs/cld-ecal-study"), "published/")

    publication = open_publication("https://k4opt-data.web.cern.ch")  # or "published/"
    for entry in publication.catalog().studies:
        study = publication.study(entry)  # a StudyResults, checked
"""

from k4opt.publication.catalog import CATALOG_NAME, Catalog, CatalogEntry, Snapshot
from k4opt.publication.errors import (
    NothingPublishedError,
    PublicationError,
    PublicationUnavailableError,
)
from k4opt.publication.read import Publication, open_publication
from k4opt.publication.write import local_paths, publish

__all__ = [
    "CATALOG_NAME",
    "Catalog",
    "CatalogEntry",
    "NothingPublishedError",
    "Publication",
    "PublicationError",
    "PublicationUnavailableError",
    "Snapshot",
    "local_paths",
    "open_publication",
    "publish",
]
