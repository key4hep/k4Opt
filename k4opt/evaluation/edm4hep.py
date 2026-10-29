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


def _events(path: Path) -> Iterator[Any]:
    """The events (podio frames) of an EDM4hep file; imports the Key4hep ``podio``."""
    try:
        from podio.reading import get_reader
    except ImportError as exc:
        raise EvaluationFailed(
            f"podio is not available (is the Key4hep environment set up?): {exc}"
        ) from exc
    reader = get_reader(str(path))
    if "events" not in reader.categories:
        raise EvaluationFailed(f"{path.name} has no 'events' category")
    yield from reader.get("events")


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


def count_events(path: Path) -> int:
    """The number of events in the EDM4hep file ``path``."""
    with _unreadable(path):
        return sum(1 for _ in _events(path))


def read_collection(path: Path, collection: str, value_type: str) -> Iterator[Any]:
    """The collection ``collection`` of each event of ``path``, in file order.

    ``value_type`` is the podio type of its elements, for example
    ``"edm4hep::SimCalorimeterHit"``. An existing but empty collection is yielded like any
    other. A collection that is missing in an event, or holds another type, raises
    :class:`EvaluationFailed` when that event is reached, never an empty result.
    """
    with _unreadable(path):
        for index, event in enumerate(_events(path)):
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
            yield elements
