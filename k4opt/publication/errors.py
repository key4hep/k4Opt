"""Exceptions raised by the publication component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class PublicationError(K4OptError):
    """A study cannot be published, or what a publication holds does not hold up.

    The publication directory is inside the study, its catalog is malformed or names a later state
    of the study, another publication holds its lock, or a catalog entry or a snapshot read back is
    not what it says. The message says which file and what to do; nothing is changed.
    """


class NothingPublishedError(PublicationError):
    """Nothing is published at the location: it holds no catalog."""


class PublicationUnavailableError(PublicationError):
    """The location did not answer as it should just now (a timeout, a refused connection, a
    server error, an incomplete transfer, an unreadable file); reading again may succeed."""
