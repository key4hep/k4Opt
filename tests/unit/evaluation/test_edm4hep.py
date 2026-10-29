"""Reading EDM4hep files for evaluators, with fake podio events."""

from __future__ import annotations

import gc
import weakref
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


@pytest.mark.parametrize("name", ["SimCalorimeterHit", HIT_TYPE])
def test_the_edm4hep_prefix_of_the_type_is_optional(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    simulation.install_events(monkeypatch, events_with([1], [2]))
    assert len(list(read_collection(PATH, COLLECTION, name))) == 2


def test_a_short_name_does_not_match_a_type_of_another_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation.install_events(
        monkeypatch, [Event(**{COLLECTION: Hits([1], "other::SimCalorimeterHit")})]
    )
    with pytest.raises(EvaluationFailed, match="expected 'edm4hep::SimCalorimeterHit'"):
        list(read_collection(PATH, COLLECTION, "SimCalorimeterHit"))


def test_a_kept_collection_keeps_the_event_that_owns_its_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # podio elements point into the event's memory: freeing the event would leave them dangling
    events = events_with([1], [2])
    owners = [weakref.ref(event) for event in events]
    simulation.install_events(monkeypatch, events)
    kept = list(read_collection(PATH, COLLECTION, HIT_TYPE))
    events.clear()  # the fake reader no longer holds the events
    gc.collect()
    assert all(owner() is not None for owner in owners)
    assert [hit.getCellID() for hits in kept for hit in hits] == [1, 2]
    del kept
    gc.collect()
    assert all(owner() is None for owner in owners)


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


def test_counting_asks_the_reader_for_its_entries_and_reads_no_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Frames:
        def __len__(self) -> int:
            return 7

        def __iter__(self) -> Any:
            raise AssertionError("counting must not read the frames")

    monkeypatch.setattr(edm4hep, "_frames", lambda path: Frames())
    assert count_events(PATH) == 7


def test_c_plus_plus_exceptions_while_iterating_a_collection_are_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Elements:
        def getValueTypeName(self) -> str:  # noqa: N802
            return HIT_TYPE

        def __iter__(self) -> Any:
            raise CppException("corrupt buffer")

    simulation.install_events(monkeypatch, [Event(**{COLLECTION: Elements()})])
    with pytest.raises(EvaluationFailed, match="CppException: corrupt buffer"):
        list(read_collection(PATH, COLLECTION, HIT_TYPE))


def test_c_plus_plus_exceptions_when_opening_are_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def opening(path: Path) -> Any:
        raise CppException("incompatible file")

    monkeypatch.setattr(edm4hep, "_frames", opening)
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
