"""Reading EDM4hep files for evaluators.

``podio`` comes from the Key4hep stack and is imported only when a file is opened, so this
module imports without it. Every problem with a file (``podio`` missing, a file that cannot be
opened, a missing category or collection, an unexpected element type, an exception of the C++
reader) is an :class:`EvaluationFailed`, which ``k4opt`` reports as a ``failed`` result.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from k4opt.evaluation.errors import EvaluationFailed


def _frames(path: Path) -> Any:
    """The events (podio frames) of an EDM4hep file, iterable and with a ``len``.

    Imports the Key4hep ``podio``. Counting asks the reader for the number of entries; only
    iterating reads the frames.
    """
    try:
        from podio.reading import get_reader
    except ImportError as exc:
        raise EvaluationFailed(
            f"podio is not available (is the Key4hep environment set up?): {exc}"
        ) from exc
    reader = get_reader(str(path))
    if "events" not in reader.categories:
        raise EvaluationFailed(f"{path.name} has no 'events' category")
    return reader.get("events")


@contextlib.contextmanager
def _unreadable(path: Path) -> Iterator[None]:
    try:
        yield
    except EvaluationFailed:
        raise
    except Exception as exc:
        # The Key4hep reader raises C++ exceptions (std::runtime_error, ROOT's RException)
        # that are not subclasses of Python's RuntimeError, so no narrower type is enough. They
        # mean an unusable file, not a bug in the caller.
        raise EvaluationFailed(f"cannot read {path.name}: {type(exc).__name__}: {exc}") from exc


def _type_name(name: Any) -> str:
    """The text of a podio type name, which is a C++ ``string_view`` (not a ``str``) in PyROOT."""
    return name if isinstance(name, str) else str(name.data())[: int(name.size())]


class _Elements(list[Any]):
    """The elements of one collection. podio elements do not own their data, the event (podio
    frame) does, so the list keeps its event alive: lists may be kept after reading on."""

    def __init__(self, elements: Any, event: Any) -> None:
        super().__init__(elements)
        self.event = event


def count_events(path: Path) -> int:
    """The number of events in the EDM4hep file ``path``."""
    with _unreadable(path):
        return len(_frames(path))


def read_collection(path: Path, collection: str, value_type: str) -> Iterator[Any]:
    """The collection ``collection`` of each event of ``path`` as a list, in file order.

    ``value_type`` is the podio type of its elements, for example
    ``"edm4hep::SimCalorimeterHit"``. An existing but empty collection is yielded like any
    other. A collection that is missing in an event, or holds another type, raises
    :class:`EvaluationFailed` when that event is reached, never an empty result. Each list keeps
    its event in memory, so lists can be kept and used after the next event is read. A file that
    ends before the number of events it records raises once the last event was read.
    """
    with _unreadable(path):
        frames = _frames(path)
        entries, read = len(frames), 0
        for index, event in enumerate(frames):
            read = index + 1
            if collection not in event.getAvailableCollections():
                raise EvaluationFailed(
                    f"collection {collection!r} is missing in event {index} of {path.name}"
                )
            elements = event.get(collection)
            found = _type_name(elements.getValueTypeName())
            if found != value_type:
                raise EvaluationFailed(
                    f"collection {collection!r} holds {found!r}, expected {value_type!r}"
                )
            # read here, where a failing reader is reported, not where the caller uses them
            yield _Elements(elements, event)
        # the event count checked by k4opt is the number of entries the file records; a damaged
        # file can stop earlier without raising
        if read != entries:
            raise EvaluationFailed(f"{path.name} records {entries} events, but {read} were read")
