"""A study interrupted at any write to its ledger stays readable and resumable, and loses or
repeats no trial.

The ledger is written one line per event: the study, each session, each proposal, each ending.
The tests run a study once to count the lines, then once for each line, interrupted there: by
Ctrl-C or SIGTERM (a ``KeyboardInterrupt``, after which k4opt ends the trials still running
unless that write began; ``synced``: it came as the line was synced, so the line is on disk), by
a SIGKILL (nothing more is written, so the trial is left as far as it got; ``torn``: with half of
the line on disk), or by an ordinary error of that one write (a disk that fills up, after which
nothing more is written and k4opt stops the study). Every such study must then be read by
``load_study`` and resumed, each trial reading as its own kind or as ``interrupted``, every
trial proposed once and ended once, and a grid's points proposed in order, none twice.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from k4opt.optimization import OptimizationError, load_study, optimize
from k4opt.optimization._ledger import LedgerWriter
from tests.unit.optimization.conftest import RADIUS, ledger
from tests.unit.optimization.test_resume import rejecting

TRIALS = 4


class Killed(BaseException):
    """What a SIGKILL is to the test: nothing of the process runs after it, so neither does any
    write to the ledger."""


def _graded(write_problem: Callable[..., Path]) -> Path:
    """Trials of every kind but rejected: valid feasible and infeasible ones, invalid and failed."""
    return write_problem(
        evaluators=[
            {
                "name": "test.trade_off",
                "version": "1",
                "settings": {"invalid_below": 250.0, "failed_below": 180.0},
            }
        ],
        constraints=[{"stage": "analysis", "metric": "a", "max": 400, "unit": "mm"}],
    )


PROBLEMS: dict[str, tuple[Callable[[Callable[..., Path]], Path], str]] = {
    "graded": (_graded, "trade_off"),
    "rejecting": (rejecting, "candidate_check"),
}
GRIDS = {
    "graded": {RADIUS: [500.0, 300.0, 200.0, 150.0, 450.0, 350.0, 100.0, 600.0]},
    "rejecting": {RADIUS: [150, 250]},  # categorical: 250 is rejected
}


def _evaluators(kind: str) -> list[Any]:
    from tests.support import evaluators

    return [getattr(evaluators, kind)()]


@contextmanager
def _writes(at: int | None, how: str) -> Iterator[list[str]]:
    """Count the lines written to ledgers, by event; interrupt the line numbered ``at`` (from 0)
    as ``how`` says, and for a kill every line after it too."""
    seen: list[str] = []
    real = LedgerWriter._write
    dead = False

    def write(self: LedgerWriter, line: str) -> None:
        nonlocal dead
        if dead:
            raise Killed
        what = str(json.loads(line)["event"])
        if at is None or len(seen) != at:
            seen.append(what)
            real(self, line)
            return
        seen.append(what)
        if how == "sigterm":
            raise KeyboardInterrupt(f"interrupted at write {at}: {what}")
        if how == "synced":  # it came as the line was synced: the line is on disk
            sync = os.fsync

            def synced(descriptor: int) -> None:
                sync(descriptor)
                raise KeyboardInterrupt(f"interrupted at write {at}: {what}")

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(os, "fsync", synced)
                real(self, line)
            raise AssertionError("the sync raises")
        if how == "error":  # the write fails as a disk that fills up fails it
            self._broken = "No space left on device"
            raise OptimizationError(f"cannot write {self.path}: No space left on device")
        dead = True
        if how == "torn":  # half of the line reached the disk
            assert self._descriptor is not None
            os.write(self._descriptor, line.encode()[: len(line) // 2])
        raise Killed

    patch = pytest.MonkeyPatch()
    patch.setattr(LedgerWriter, "_write", write)
    try:
        yield seen
    finally:
        patch.undo()


def _study(problem: Path, output: Path, kind: str, name: str, strategy: str, **kwargs: Any) -> Any:
    if not kwargs.get("resume"):
        if strategy == "grid":
            kwargs |= {"strategy": "grid", "grid": GRIDS[name]}
        else:
            pytest.importorskip("optuna")
            kwargs["seed"] = 2
    return optimize(problem, _evaluators(kind), output, trials=TRIALS, **kwargs)


def _accounted(output: Path, strategy: str, name: str, before: int, where: str) -> None:
    """Every trial was proposed once, in number order, and ended once (the ledger refuses
    anything else); the resume added as many as it was asked for, or, for a grid, as many as it
    had points left; a grid's points were proposed in its order, none twice."""
    entries = ledger(output).entries
    assert [e.number for e in entries] == list(range(len(entries))), where
    assert all(e.ending is not None for e in entries), where
    if strategy == "grid":
        points = GRIDS[name][RADIUS]
        assert len(entries) == min(before + TRIALS, len(points)), where
        values = [e.values[RADIUS] for e in entries]
        assert values == points[: len(values)], where
        assert [e.proposal for e in entries] == [{"point": n} for n in range(len(entries))], where
    else:
        assert len(entries) == before + TRIALS, where


@pytest.mark.parametrize("how", ["sigterm", "synced", "sigkill", "torn", "error"])
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_an_interrupt_at_any_write_leaves_a_readable_and_resumable_study(
    tmp_path: Path, write_problem: Callable[..., Path], name: str, how: str, strategy: str
) -> None:
    # an ordinary error stops the study with an error of its own (OptimizationError)
    interrupt = {"sigterm": KeyboardInterrupt, "synced": KeyboardInterrupt, "error": Exception}
    stops = interrupt.get(how, Killed)
    make, kind = PROBLEMS[name]
    problem = make(write_problem)
    with _writes(None, how) as writes:
        _study(problem, tmp_path / "clean", kind, name, strategy)
    statuses = {t.status for t in load_study(tmp_path / "clean").trials}
    # the problem gives the trials the test means to interrupt
    assert statuses >= ({"rejected", "valid"} if name == "rejecting" else {"valid", "invalid"})
    for at, what in enumerate(writes):
        output = tmp_path / f"at-{at}"
        with _writes(at, how), pytest.raises(stops):
            _study(problem, output, kind, name, strategy)
        where = f"interrupted at write {at} ({what})"
        (output / ".lock").unlink(missing_ok=True)  # a kill leaves it
        if at == 0 and how != "synced":  # its ledger names no study: a resume sets it up
            with pytest.raises(OptimizationError, match="never set up"):
                load_study(output)
            before = 0
        else:
            try:
                read = load_study(output)
            except Exception as exc:  # pragma: no cover - the failure is the message
                pytest.fail(f"{where}: load_study refuses the study: {exc}")
            assert {t.status for t in read.trials} <= {*statuses, "interrupted", "running"}, where
            before = len(read.trials)
        left = [] if at == 0 else [e.number for e in ledger(output).entries if e.running]
        try:
            resumed = _study(problem, output, kind, name, strategy, resume=True)
        except Exception as exc:  # pragma: no cover - the failure is the message
            pytest.fail(f"{where}: the resume refuses the study: {exc}")
        assert "running" not in {t.status for t in resumed.trials}, where
        final = load_study(output)
        assert final.to_dict() == resumed.to_dict(), where
        assert {t.status for t in final.trials} <= {*statuses, "interrupted"}, where
        # what the interrupted session left running, the resume ended: those, and only those
        assert [t.number for t in final.trials if t.recovered] == left, where
        _accounted(output, strategy, name, before, where)
