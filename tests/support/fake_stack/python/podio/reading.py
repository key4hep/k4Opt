"""``get_reader`` of the fake ``podio``: the events of a file the fake ``ddsim`` wrote."""

from __future__ import annotations

from tests.support.stack import Reader


def get_reader(path: str) -> Reader:
    return Reader(path)
