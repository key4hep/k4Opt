"""A study interrupted at any write to the optimizer's storage stays readable and resumable.

Each trial is told in several writes (its marks, its constraint values, then ``tell``). The tests
run a study once to count them, then once for each write, interrupted there: by Ctrl-C or
SIGTERM (a ``KeyboardInterrupt``, after which k4opt closes the trials still running), or by a
SIGKILL (nothing more is written, so the trial is left as far as it got). Every such study must
then be read by ``load_study`` and resumed, each trial reading as its own kind or as
``interrupted``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import optuna
import pytest
from optuna.trial import Trial

from k4opt.optimization import load_study, optimize
from tests.unit.optimization.test_resume import rejecting

TRIALS = 4


class Killed(BaseException):
    """What a SIGKILL is to the test: nothing of the process runs after it, so neither does any
    write to the storage."""


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


def _evaluators(kind: str) -> list[Any]:
    from tests.support import evaluators

    return [getattr(evaluators, kind)()]


@contextmanager
def _writes(at: int | None, how: type[BaseException]) -> Iterator[list[str]]:
    """Count the writes of trials to the storage, by what they write; raise ``how`` at write
    number ``at`` (from 0), and for :class:`Killed` at every write after it too."""
    seen: list[str] = []
    real_attr, real_constraint = Trial.set_user_attr, Trial.set_constraint
    real_tell = optuna.Study.tell
    dead = False

    def count(what: str) -> None:
        nonlocal dead
        if dead:
            raise Killed
        if at is not None and len(seen) == at:
            dead = how is Killed
            seen.append(what)
            raise how(f"interrupted at write {at}: {what}")
        seen.append(what)

    def set_user_attr(self: Trial, key: str, value: Any) -> None:
        count(key)
        real_attr(self, key, value)

    def set_constraint(self: Trial, key: str, value: float) -> None:
        count(f"constraint {key}")
        real_constraint(self, key, value)

    def tell(self: optuna.Study, *args: Any, **kwargs: Any) -> Any:
        count("tell")
        return real_tell(self, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(Trial, "set_user_attr", set_user_attr)
    patch.setattr(Trial, "set_constraint", set_constraint)
    patch.setattr(optuna.Study, "tell", tell)
    try:
        yield seen
    finally:
        patch.undo()


def _study(problem: Path, output: Path, kind: str, **kwargs: Any) -> Any:
    return optimize(problem, _evaluators(kind), output, trials=TRIALS, **kwargs)


def _count(problem: Path, output: Path, kind: str) -> list[str]:
    with _writes(None, KeyboardInterrupt) as seen:
        _study(problem, output, kind, seed=2)
    return seen


@pytest.mark.parametrize("how", [KeyboardInterrupt, Killed], ids=["sigterm", "sigkill"])
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_an_interrupt_at_any_write_leaves_a_readable_and_resumable_study(
    tmp_path: Path, write_problem: Callable[..., Path], name: str, how: type[BaseException]
) -> None:
    make, kind = PROBLEMS[name]
    problem = make(write_problem)
    writes = _count(problem, tmp_path / "clean", kind)
    statuses = {t.status for t in load_study(tmp_path / "clean").trials}
    # the problem gives the trials the test means to interrupt
    assert statuses >= ({"rejected", "valid"} if name == "rejecting" else {"valid", "invalid"})
    for at, what in enumerate(writes):
        output = tmp_path / f"at-{at}"
        with _writes(at, how), pytest.raises(how):
            _study(problem, output, kind, seed=2)
        where = f"interrupted at write {at} ({what})"
        try:
            read = load_study(output)
        except Exception as exc:  # pragma: no cover - the failure is the message
            pytest.fail(f"{where}: load_study refuses the study: {exc}")
        assert {t.status for t in read.trials} <= {
            *statuses,
            "interrupted",
            "running",
        }, where
        try:
            resumed = _study(problem, output, kind, resume=True)
        except Exception as exc:  # pragma: no cover - the failure is the message
            pytest.fail(f"{where}: the resume refuses the study: {exc}")
        assert "running" not in {t.status for t in resumed.trials}, where
        assert {t.status for t in load_study(output).trials} <= {*statuses, "interrupted"}, where
