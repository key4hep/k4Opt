"""The ledger of a study, ``trials.jsonl``: one event per line, appended and synced by the one
process that holds the study's lock, read back strictly and without writing; a line a killed
process cut short is never read, and the next writer removes it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from k4opt.optimization import OptimizationError
from k4opt.optimization._ledger import (
    LEDGER_NAME,
    LedgerWriter,
    NotSetUpError,
    read_ledger,
)

STUDY = "2b9a6f0e-7c55-4d3e-9a0f-0d6b1f1e2c3a"
STRATEGY = "5" * 64  # the digest of the strategy the study was made with
RECORD = {"id": "r0", "sha256": "a" * 64}


def written(directory: Path) -> LedgerWriter:
    """A ledger with a session, two trials (one recorded, one running) and a third rejected."""
    writer = LedgerWriter.create(directory, STUDY, STRATEGY)
    writer.session("2026-11-19T10:00:00+00:00")
    writer.proposed(0, {"radius": 300.0}, {"point": 0})
    writer.recorded(0, "valid", RECORD)
    writer.proposed(1, {"radius": 400.0}, {"point": 1})
    writer.proposed(2, {"radius": 500.0}, {"point": 2})
    writer.rejected(2, "test.candidate_check", "cannot measure 500")
    return writer


def events(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (directory / LEDGER_NAME).read_text().splitlines()]


def rewrite(directory: Path, lines: list[dict[str, Any]]) -> None:
    (directory / LEDGER_NAME).write_text("".join(json.dumps(e) + "\n" for e in lines))


def test_a_ledger_reads_back_as_it_was_written(tmp_path: Path) -> None:
    writer = written(tmp_path)
    writer.close()
    ledger = read_ledger(tmp_path, STUDY, STRATEGY)
    assert ledger == writer.ledger
    assert ledger.study == STUDY and ledger.sessions == ("2026-11-19T10:00:00+00:00",)
    first, second, third = ledger.entries
    assert (first.ending.kind, first.ending.status, dict(first.ending.record)) == (
        "recorded",
        "valid",
        RECORD,
    )
    assert second.running and dict(second.values) == {"radius": 400.0}
    assert (third.ending.kind, third.ending.by) == ("rejected", "test.candidate_check")
    assert not ledger.exhausted and not any(e.recovered for e in ledger.entries)
    # one line per event, numbered, compact: what a person reads with any tool
    assert [e["seq"] for e in events(tmp_path)] == list(range(7))
    head = {"seq": 0, "event": "study", "study": STUDY, "strategy": STRATEGY}
    assert events(tmp_path)[0] == head  # the study, and the strategy it was made with


def test_a_trial_a_later_session_ends_is_recovered(tmp_path: Path) -> None:
    written(tmp_path).close()
    writer, cut = LedgerWriter.open(tmp_path, STUDY, STRATEGY)
    assert not cut
    writer.session("2026-11-19T11:00:00+00:00")
    writer.interrupted(1)
    writer.close()
    entries = read_ledger(tmp_path, STUDY, STRATEGY).entries
    assert [e.recovered for e in entries] == [False, True, False]


def test_a_line_cut_short_is_not_read_and_the_next_writer_removes_it(tmp_path: Path) -> None:
    written(tmp_path).close()
    path = tmp_path / LEDGER_NAME
    whole = path.read_bytes()
    path.write_bytes(whole + b'{"seq":7,"event":"recor')  # a process killed as it wrote it
    assert read_ledger(tmp_path, STUDY, STRATEGY).entries[1].running  # as if never written
    assert path.read_bytes().endswith(b'"event":"recor')  # reading changes nothing
    writer, cut = LedgerWriter.open(tmp_path, STUDY, STRATEGY)
    assert cut and path.read_bytes() == whole
    writer.session("2026-11-19T11:00:00+00:00")
    writer.close()
    assert len(read_ledger(tmp_path, STUDY, STRATEGY).sessions) == 2


def test_a_ledger_without_its_first_line_is_a_study_never_set_up(tmp_path: Path) -> None:
    with pytest.raises(NotSetUpError, match="does not exist"):
        read_ledger(tmp_path, STUDY, STRATEGY)
    (tmp_path / LEDGER_NAME).write_bytes(b'{"seq":0,"ev')
    with pytest.raises(NotSetUpError, match="holds no event"):
        read_ledger(tmp_path, STUDY, STRATEGY)


def _edit(index: int, **changes: Any) -> Any:
    def edit(lines: list[dict[str, Any]]) -> None:
        lines[index].update(changes)

    return edit


BROKEN = {
    "another study": (_edit(0, study="other"), "is the ledger of the study 'other'"),
    "another strategy": (
        _edit(0, strategy="6" * 64),
        "study.json records another strategy than the one the study was made with",
    ),
    "no study first": (lambda lines: lines.pop(0), "line 1 is not event 0 of the ledger"),
    "a gap": (lambda lines: lines.pop(3), "line 4 is not event 3"),
    "an unknown event": (_edit(4, event="told"), "is no event of a study's ledger"),
    "a key more": (_edit(3, values={"radius": 1.0}), "is no event of a study's ledger"),
    "out of order": (_edit(4, trial=5), "proposes trial 5, but the next trial is 1"),
    "ended twice": (
        lambda lines: lines.append({"seq": 7, "event": "error", "trial": 0, "message": "x"}),
        "trial 0 has ended already",
    ),
    "no such trial": (
        lambda lines: lines.append({"seq": 7, "event": "interrupted", "trial": 7}),
        "there is no trial 7",
    ),
    "a status of no record": (_edit(3, status="feasible"), "is not the status of a record"),
    "no record": (_edit(3, record={"id": "r0", "sha256": "short"}), "names no record"),
    "a message no text": (_edit(6, message=3), "3 is not text"),
    "values no scalars": (_edit(2, values={"radius": [1.0]}), "have no scalar values"),
    "before a session": (lambda lines: lines.pop(1), "before any session opened the study"),
    "a trial number no number": (_edit(4, trial=True), "True is no trial number"),
    "not JSON": (None, "line 3 is not JSON"),
}


@pytest.mark.parametrize("case", sorted(BROKEN))
def test_a_ledger_that_does_not_hold_up_is_refused_and_left_as_it_is(
    tmp_path: Path, case: str
) -> None:
    written(tmp_path).close()
    change, message = BROKEN[case]
    if change is None:
        path = tmp_path / LEDGER_NAME
        lines = path.read_bytes().split(b"\n")
        lines[2] = lines[2][:-1]  # a complete line, but not JSON
        path.write_bytes(b"\n".join(lines))
    else:
        lines = events(tmp_path)
        change(lines)
        for index, line in enumerate(lines):  # numbered again, where a case does not mean that
            if case not in ("a gap", "no study first"):
                line["seq"] = index
        rewrite(tmp_path, lines)
    before = (tmp_path / LEDGER_NAME).read_bytes()
    with pytest.raises(OptimizationError, match=message):
        read_ledger(tmp_path, STUDY, STRATEGY)
    with pytest.raises(OptimizationError, match=message):  # a writer opens none such either
        LedgerWriter.open(tmp_path, STUDY, STRATEGY)
    assert (tmp_path / LEDGER_NAME).read_bytes() == before


def test_a_writer_never_writes_what_a_reader_would_refuse(tmp_path: Path) -> None:
    writer = written(tmp_path)
    before = (tmp_path / LEDGER_NAME).read_bytes()
    for write in (
        lambda: writer.recorded(0, "valid", RECORD),  # ended already
        lambda: writer.proposed(7, {"radius": 1.0}, {}),  # not the next trial
        lambda: writer.selected(9, {"study": "s", "trial": 0, "values": {}}),
        lambda: writer.recorded(1, "feasible", RECORD),
        lambda: writer.proposed(3, {"radius": float("nan")}, {}),  # no JSON
    ):
        with pytest.raises((OptimizationError, ValueError)):
            write()
        assert writer.writable  # refused before anything was written
    assert (tmp_path / LEDGER_NAME).read_bytes() == before
    writer.exhaust()
    with pytest.raises(OptimizationError, match="after the strategy was exhausted"):
        writer.proposed(3, {"radius": 1.0}, {})
    writer.close()
    assert read_ledger(tmp_path, STUDY, STRATEGY).exhausted


def test_after_a_write_fails_nothing_more_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from k4opt.optimization import _ledger

    writer = written(tmp_path)
    write = os.write

    def full(descriptor: int, data: Any) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_ledger.os, "write", full)
    with pytest.raises(OptimizationError, match="No space left on device"):
        writer.interrupted(1)
    monkeypatch.setattr(_ledger.os, "write", write)
    assert not writer.writable
    with pytest.raises(OptimizationError, match="is not written any more after a write failed"):
        writer.interrupted(1)
    writer.close()
    assert read_ledger(tmp_path, STUDY, STRATEGY).entries[1].running  # settled by the next session


def test_a_new_ledger_is_never_made_over_another(tmp_path: Path) -> None:
    written(tmp_path).close()
    before = (tmp_path / LEDGER_NAME).read_bytes()
    with pytest.raises(OptimizationError, match="cannot create"):
        LedgerWriter.create(tmp_path, STUDY, STRATEGY)
    assert (tmp_path / LEDGER_NAME).read_bytes() == before
