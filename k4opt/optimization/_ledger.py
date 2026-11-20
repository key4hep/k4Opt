"""Internal: the ledger of a study, the one durable record of its trials; nothing of any strategy.

Every study directory holds ``trials.jsonl`` (:data:`LEDGER_NAME`): one JSON object per line, each
an *event*, numbered by ``seq`` from 0. Only the process that holds the study's lock writes it,
and only by appending a complete line, synced to disk before anything else happens
(:class:`LedgerWriter`): a line is written once it ends in a newline. A line a killed process
cut short is the last one and was never written: a reader leaves it out, and the next writer
removes it before it appends. :func:`read_ledger` reads it, strictly and without writing.

The events, each with exactly these keys besides ``seq`` and ``event``:

- ``study`` (``study``, ``strategy``): the first line, naming the study the ledger belongs to
  and committing to the strategy it was made with (the SHA-256 of what ``study.json`` records
  of it), so that a strategy changed afterwards is refused, not run or read as the study's;
- ``session`` (``started``): a session opened the study to run trials; the trials a later
  session ends are ``recovered``;
- ``proposed`` (``trial``, ``values``, ``proposal``): trial ``trial`` (numbered from 0, in
  order) with the values the strategy proposed for the searched parameters, checked, and what the
  strategy noted of it; the trial runs from here on;
- ``malformed`` (``trial``, ``proposal``, ``message``): a proposal that is no candidate of the
  study, ended as it is made;
- ``selected`` (``trial``, ``selected``): the best trial of the study of the level below that a
  running trial stands for (``study``, ``trial``, the ``values`` it adds), once;
- one ending of a running trial: ``recorded`` (``trial``, ``status`` of its record, ``record``:
  its ``id`` and ``sha256``), ``rejected`` (``trial``, ``by``, ``message``), ``unscored``
  (``trial``, ``message``), ``interrupted`` (``trial``) or ``error`` (``trial``, ``message``);
- ``exhausted``: the strategy has nothing more to propose; no trial is proposed after it.

What the trial's values rest on is its record, ``trial-NNNNNN/summary.json``, which the ledger
names by id and SHA-256: the ledger says which trials there are and how each ended, the records
what they measured.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from k4opt.evaluation import Scalar
from k4opt.frozen import FrozenMap, Json, freeze
from k4opt.optimization.errors import OptimizationError

LEDGER_NAME = "trials.jsonl"

RECORDED = ("valid", "invalid", "failed")
"""The statuses of a record."""

ENDINGS = ("recorded", "rejected", "unscored", "interrupted", "error")
"""How a running trial ends, as an event."""

_KEYS: Mapping[str, tuple[str, ...]] = {
    "study": ("study", "strategy"),
    "session": ("started",),
    "proposed": ("trial", "values", "proposal"),
    "malformed": ("trial", "proposal", "message"),
    "selected": ("trial", "selected"),
    "recorded": ("trial", "status", "record"),
    "rejected": ("trial", "by", "message"),
    "unscored": ("trial", "message"),
    "interrupted": ("trial",),
    "error": ("trial", "message"),
    "exhausted": (),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class NotSetUpError(OptimizationError):
    """The study has no ledger: the process that made its ``study.json`` stopped before it made
    one, or the file was moved away; only a study directory in which nothing ran tells the two
    apart."""


@dataclass(frozen=True, slots=True)
class Ending:
    """How a trial ended: ``kind`` is an event of :data:`ENDINGS`, or ``malformed``, in the
    session numbered ``session``; ``status`` and ``record`` (``id``, ``sha256``) are those of a
    ``recorded`` one, ``by`` names the task that rejected a ``rejected`` one."""

    kind: str
    session: int
    status: str | None = None
    record: Mapping[str, str] | None = None
    by: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class Entry:
    """One trial as the ledger holds it: proposed in the session numbered ``session`` with the
    ``values`` of the searched parameters (none for a malformed proposal) and what the strategy
    noted of it (``proposal``); the trial of the study below it ``selected``, and its
    ``ending``, ``None`` while it runs."""

    number: int
    session: int
    values: Mapping[str, Scalar]
    proposal: Mapping[str, Json]
    selected: Mapping[str, Json] | None = None
    ending: Ending | None = None

    @property
    def running(self) -> bool:
        return self.ending is None

    @property
    def recovered(self) -> bool:
        """Whether a later session than the one that proposed it ended it (a resume)."""
        return self.ending is not None and self.ending.session != self.session


@dataclass(frozen=True, slots=True)
class Ledger:
    """What a ledger holds: the study it belongs to, the sessions that opened it (their start
    times), every trial by number, and whether the strategy was exhausted."""

    study: str
    sessions: tuple[str, ...]
    entries: tuple[Entry, ...]
    exhausted: bool


@dataclass(slots=True)
class _State:
    """The ledger as its events so far make it, checked event by event: what a reader replays
    and a writer keeps, so that a writer never appends what a reader would refuse. ``seq`` is
    the number of the next event: the lines so far, the study's own first one among them."""

    study: str
    sessions: list[str] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    exhausted: bool = False
    seq: int = 1

    def apply(self, event: Mapping[str, Any], where: str) -> None:
        """Add ``event`` as the next; raise :class:`OptimizationError` (``where`` names the
        line), changing nothing, for one that is not the next a ledger can hold."""
        self._add(event, where)
        self.seq += 1

    def _add(self, event: Mapping[str, Any], where: str) -> None:
        kind = event.get("event")
        keys = _KEYS.get(kind) if isinstance(kind, str) else None
        if keys is None or set(event) != {"seq", "event", *keys}:
            raise OptimizationError(f"{where} is no event of a study's ledger: {event!r}")
        if kind == "study":
            raise OptimizationError(f"{where} names its study again")
        if kind == "session":
            _text(event["started"], where)
            self.sessions.append(event["started"])
            return
        if not self.sessions:
            raise OptimizationError(f"{where}: {kind} before any session opened the study")
        session = len(self.sessions) - 1
        if kind == "exhausted":
            if self.exhausted:
                raise OptimizationError(f"{where}: the strategy was exhausted already")
            self.exhausted = True
            return
        number = event["trial"]
        if isinstance(number, bool) or not isinstance(number, int):
            raise OptimizationError(f"{where}: trial {number!r} is no trial number")
        if kind in ("proposed", "malformed"):
            if number != len(self.entries):
                raise OptimizationError(
                    f"{where} proposes trial {number}, but the next trial is {len(self.entries)}"
                )
            if self.exhausted:
                raise OptimizationError(
                    f"{where} proposes a trial after the strategy was exhausted"
                )
            proposal = _object(event["proposal"], f"{where}'s proposal")
            if kind == "malformed":
                ending = Ending("malformed", session, message=_text(event["message"], where))
                self.entries.append(Entry(number, session, FrozenMap({}), proposal, None, ending))
            else:
                values = _object(event["values"], f"{where}'s values")
                bad = [k for k, v in values.items() if isinstance(v, (Mapping, tuple)) or v is None]
                if bad:
                    raise OptimizationError(f"{where}: {', '.join(bad)} have no scalar values")
                self.entries.append(Entry(number, session, values, proposal))  # type: ignore[arg-type]
            return
        if not 0 <= number < len(self.entries):
            raise OptimizationError(f"{where}: there is no trial {number}")
        entry = self.entries[number]
        if not entry.running:
            raise OptimizationError(f"{where}: trial {number} has ended already")
        if kind == "selected":
            if entry.selected is not None:
                raise OptimizationError(f"{where}: trial {number} selected already")
            selected = _object(event["selected"], f"{where}'s selection")
            if set(selected) != {"study", "trial", "values"}:
                raise OptimizationError(f"{where}: {selected!r} is no selection")
            self.entries[number] = replace(entry, selected=selected)
            return
        ending = Ending(str(kind), session)
        if kind == "recorded":
            status, record = event["status"], _object(event["record"], f"{where}'s record")
            if status not in RECORDED:
                raise OptimizationError(f"{where}: {status!r} is not the status of a record")
            if (
                set(record) != {"id", "sha256"}
                or not isinstance(record["id"], str)
                or not isinstance(record["sha256"], str)
                or not _SHA256.fullmatch(record["sha256"])
            ):
                raise OptimizationError(f"{where}: {record!r} names no record")
            ending = Ending(kind, session, status=status, record=record)  # type: ignore[arg-type]
        elif kind == "rejected":
            by, message = _text(event["by"], where), _text(event["message"], where)
            ending = Ending(kind, session, by=by, message=message)
        elif kind in ("unscored", "error"):
            ending = Ending(kind, session, message=_text(event["message"], where))
        self.entries[number] = replace(entry, ending=ending)

    def snapshot(self) -> Ledger:
        return Ledger(self.study, tuple(self.sessions), tuple(self.entries), self.exhausted)


def _text(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise OptimizationError(f"{where}: {value!r} is not text")
    return value


def _object(value: object, where: str) -> Mapping[str, Json]:
    frozen = freeze(value, where, OptimizationError)
    if not isinstance(frozen, Mapping):
        raise OptimizationError(f"{where} is not an object")
    return frozen


def _parse(line: bytes, index: int, path: Path) -> dict[str, Any]:
    where = f"{path.name} line {index + 1}"
    try:
        event = json.loads(line)
    except ValueError as exc:
        raise OptimizationError(f"{where} is not JSON: {exc}") from None
    if not isinstance(event, dict) or event.get("seq") != index or isinstance(event["seq"], bool):
        raise OptimizationError(f"{where} is not event {index} of the ledger: {line[:200]!r}")
    return event


def _replay(data: bytes, path: Path, study: str, strategy: str) -> tuple[_State, int]:
    """The state the complete lines of ``data`` make, and how many bytes they take: a last line
    without its newline was cut short as it was written, so it was never written. The first
    line must name the study ``study`` and commit to the strategy whose digest is
    ``strategy``."""
    end = data.rfind(b"\n") + 1
    lines = data[:end].split(b"\n")[:-1]
    if not lines:
        raise NotSetUpError(f"{path} holds no event: the study was never set up completely")
    head = _parse(lines[0], 0, path)
    if set(head) != {"seq", "event", "study", "strategy"} or head["event"] != "study":
        raise OptimizationError(f"{path.name} line 1 does not name the ledger's study")
    if head["study"] != study:
        raise OptimizationError(
            f"{path} is the ledger of the study {head['study']!r}, not of {study!r}"
        )
    if head["strategy"] != strategy:
        raise OptimizationError(
            f"study.json records another strategy than the one the study was made with: "
            f"{path.name} commits to the strategy {str(head['strategy'])[:12]}..., study.json "
            f"records {strategy[:12]}...; a study runs and reads with the strategy it began with"
        )
    state = _State(study)
    for index, line in enumerate(lines[1:], start=1):
        state.apply(_parse(line, index, path), f"{path.name} line {index + 1}")
    return state, end


def read_ledger(directory: Path, study: str, strategy: str) -> Ledger:
    """The ledger of the study ``study`` in ``directory``, made with the strategy whose digest is
    ``strategy`` (:func:`~k4opt.optimization._study.strategy_digest`), read without writing
    anything.

    Raises :class:`NotSetUpError` if there is none (or it names no study yet) and
    :class:`OptimizationError` for one that does not hold up: a line that is not the next event,
    an event a ledger cannot hold where it is (a trial ended twice, proposed out of order), or
    the ledger of another study, or of a study made with another strategy."""
    path = directory / LEDGER_NAME
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise NotSetUpError(
            f"{path} does not exist: the study's trials were never kept, or the file was moved away"
        ) from None
    except OSError as exc:
        raise OptimizationError(f"cannot read {path}: {exc.strerror or exc}") from None
    state, _ = _replay(data, path, study, strategy)
    return state.snapshot()


class LedgerWriter:
    """Appends the events of one study to its ledger, as the one process that holds the study's
    lock, each a complete line synced to disk; keeps the state they make
    (:attr:`ledger`).

    Every event is checked as a reader checks it before it is written. A write that does not
    complete, because it fails (:class:`OptimizationError`) or Ctrl-C or SIGTERM stops it
    (:class:`KeyboardInterrupt`), may or may not have reached the disk, whole or in part, so the
    writer writes nothing more (:attr:`writable`), and the state it keeps is no longer what the
    disk holds: what it was writing is settled by the next session, from what is on disk.
    """

    def __init__(self, path: Path, descriptor: int, state: _State) -> None:
        self.path = path
        self._descriptor: int | None = descriptor
        self._state = state
        self._broken: str | None = None

    @classmethod
    def create(cls, directory: Path, study: str, strategy: str) -> LedgerWriter:
        """The ledger of a new study, made with its first line, which commits to the strategy
        whose digest is ``strategy``."""
        path = directory / LEDGER_NAME
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o644)
        except OSError as exc:
            raise OptimizationError(f"cannot create {path}: {exc.strerror or exc}") from None
        writer = cls(path, descriptor, _State(study))
        try:
            writer._write(_line({"seq": 0, "event": "study", "study": study, "strategy": strategy}))
        except BaseException:  # a resume goes on from what reached the disk, if anything did
            writer.close()
            raise
        _sync_directory(directory)
        return writer

    @classmethod
    def open(cls, directory: Path, study: str, strategy: str) -> tuple[LedgerWriter, bool]:
        """The ledger of a study that has one, made with the strategy whose digest is
        ``strategy``, to append to, and whether a line cut short was removed from its end first
        (what a process killed as it wrote it left)."""
        path = directory / LEDGER_NAME
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_APPEND)
        except FileNotFoundError:
            raise NotSetUpError(f"{path} does not exist") from None
        except OSError as exc:
            raise OptimizationError(f"cannot open {path}: {exc.strerror or exc}") from None
        try:
            with open(descriptor, "rb", closefd=False) as handle:
                data = handle.read()
            state, end = _replay(data, path, study, strategy)
            cut = end < len(data)
            if cut:
                os.ftruncate(descriptor, end)
                os.fsync(descriptor)
        except BaseException as exc:
            os.close(descriptor)
            if isinstance(exc, OSError):
                raise OptimizationError(f"cannot read {path}: {exc.strerror or exc}") from None
            raise
        return cls(path, descriptor, state), cut

    @property
    def ledger(self) -> Ledger:
        return self._state.snapshot()

    @property
    def next_number(self) -> int:
        """The number the next trial proposed gets."""
        return len(self._state.entries)

    @property
    def exhausted(self) -> bool:
        return self._state.exhausted

    @property
    def session_number(self) -> int:
        """The number of the session writing (from 0)."""
        return len(self._state.sessions) - 1

    @property
    def writable(self) -> bool:
        """Whether the writer still writes: it is open, and every write it began completed."""
        return self._descriptor is not None and self._broken is None

    def entry(self, number: int) -> Entry:
        return self._state.entries[number]

    def session(self, started: str) -> None:
        self._append({"event": "session", "started": started})

    def proposed(
        self, number: int, values: Mapping[str, Scalar], proposal: Mapping[str, Json]
    ) -> None:
        event = {"event": "proposed", "trial": number, "values": dict(values), "proposal": proposal}
        self._append(event)

    def malformed(self, number: int, proposal: Mapping[str, Json], message: str) -> None:
        event = {"event": "malformed", "trial": number, "proposal": proposal, "message": message}
        self._append(event)

    def selected(self, number: int, selected: Mapping[str, Json]) -> None:
        self._append({"event": "selected", "trial": number, "selected": selected})

    def recorded(self, number: int, status: str, record: Mapping[str, str]) -> None:
        self._append({"event": "recorded", "trial": number, "status": status, "record": record})

    def rejected(self, number: int, by: str, message: str) -> None:
        self._append({"event": "rejected", "trial": number, "by": by, "message": message})

    def unscored(self, number: int, message: str) -> None:
        self._append({"event": "unscored", "trial": number, "message": message})

    def interrupted(self, number: int) -> None:
        self._append({"event": "interrupted", "trial": number})

    def error(self, number: int, message: str) -> None:
        self._append({"event": "error", "trial": number, "message": message})

    def exhaust(self) -> None:
        self._append({"event": "exhausted"})

    def _append(self, event: dict[str, Any]) -> None:
        if self._broken is not None:
            raise OptimizationError(
                f"{self.path} is not written any more after a write that did not complete "
                f"({self._broken})"
            )
        seq = self._state.seq
        line = _line({"seq": seq, **event})
        # checked as read back; from here until the line is on disk, whatever stops this leaves
        # the state ahead of the disk, or the line half written: nothing may be written after it
        self._broken = _UNSETTLED
        try:
            self._state.apply(json.loads(line), f"{self.path.name} line {seq + 1}")
        except OptimizationError:  # refused, before anything changed
            self._broken = None
            raise
        self._write(line)

    def _write(self, line: str) -> None:
        """Append ``line`` and sync it; the writer writes again only once it is on disk."""
        self._broken = _UNSETTLED
        data = memoryview(line.encode())
        try:
            if self._descriptor is None:
                raise OSError("the ledger is closed")
            while data:
                data = data[os.write(self._descriptor, data) :]
            os.fsync(self._descriptor)
        except OSError as exc:
            self._broken = str(exc.strerror or exc)
            raise OptimizationError(f"cannot write {self.path}: {self._broken}") from exc
        self._broken = None

    def close(self) -> None:
        if self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            os.close(descriptor)


_UNSETTLED = "interrupted"
"""Why a writer writes nothing more after a write that stopped without an error of its own
(Ctrl-C, SIGTERM)."""


def _line(event: Mapping[str, Any]) -> str:
    """The line of the ledger that holds ``event``: compact JSON, without NaN or infinity."""
    return json.dumps(event, default=_plain, separators=(",", ":"), allow_nan=False) + "\n"


def _plain(value: object) -> object:
    """A read-only mapping (:class:`~k4opt.frozen.FrozenMap`) as the JSON object it is."""
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{type(value).__name__} is not JSON")


def _sync_directory(directory: Path) -> None:
    """Make a new file's entry in ``directory`` durable, where the system allows it."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
