"""Reading EDM4hep files for evaluators, with fake podio events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import EvaluationFailed, count_events, edm4hep, read_collection
from tests.support import simulation
from tests.support.simulation import COLLECTION, HIT_TYPE, Event, Hits, events_with

PATH = Path("events.edm4hep.root")


class CppException(Exception):  # noqa: N818 - stands for cppyy's std::runtime_error
    """Like a C++ exception seen from Python: an Exception, but not a RuntimeError."""


def test_count_events(monkeypatch: pytest.MonkeyPatch) -> None:
    simulation.install_events(monkeypatch, events_with([1], [], [2, 3]))
    assert count_events(PATH) == 3


def test_the_collection_of_each_event_is_yielded_including_empty_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation.install_events(monkeypatch, events_with([1, 1, 2], [], [5]))
    sizes = [len(hits) for hits in read_collection(PATH, COLLECTION, HIT_TYPE)]
    assert sizes == [3, 0, 1]


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([Event()], "collection 'ECalBarrelCollection' is missing in event 0 of events.edm4hep"),
        ([*events_with([1]), Event()], "missing in event 1"),
        ([Event(**{COLLECTION: Hits([1], "edm4hep::MCParticle")})], "expected 'edm4hep::Sim"),
    ],
)
def test_a_missing_or_mistyped_collection_fails_never_reads_as_empty(
    monkeypatch: pytest.MonkeyPatch, events: list[Event], message: str
) -> None:
    simulation.install_events(monkeypatch, events)
    with pytest.raises(EvaluationFailed, match=message):
        list(read_collection(PATH, COLLECTION, HIT_TYPE))


def test_c_plus_plus_exceptions_of_the_reader_are_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broken(Event):
        def getAvailableCollections(self) -> tuple[str, ...]:  # noqa: N802
            return (COLLECTION,)

        def get(self, name: str) -> Any:
            raise CppException("unsupported schema version")

    simulation.install_events(monkeypatch, [*events_with([1]), Broken()])
    with pytest.raises(EvaluationFailed, match="CppException: unsupported schema version"):
        list(read_collection(PATH, COLLECTION, HIT_TYPE))


def test_c_plus_plus_exceptions_when_opening_are_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def opening(path: Path) -> Any:
        raise CppException("incompatible file")
        yield  # pragma: no cover

    monkeypatch.setattr(edm4hep, "_events", opening)
    with pytest.raises(EvaluationFailed, match=r"cannot read events\.edm4hep\.root.*incompatible"):
        count_events(PATH)


def test_a_podio_type_name_may_be_a_string_view() -> None:
    class StringView:
        def data(self) -> str:
            return "edm4hep::SimCalorimeterHit and more"

        def size(self) -> int:
            return len("edm4hep::SimCalorimeterHit")

    assert edm4hep._type_name(StringView()) == HIT_TYPE
    assert edm4hep._type_name(HIT_TYPE) == HIT_TYPE


def test_without_podio_reading_fails_with_a_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "podio", None)
    with pytest.raises(EvaluationFailed, match="podio is not available"):
        count_events(PATH)
