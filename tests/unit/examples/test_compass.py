"""The example strategy of one's own, compass search, driven with a study's context and the calls a
study makes, without evaluating anything: it finds what it is told is best, never compares a trial
without a value, does not care in which order a round's trials end, and is the same strategy when
it is told its trials again, as a study made again from its ledger tells it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from k4opt.optimization import (
    EXHAUSTED,
    Feedback,
    Proposal,
    StudyContext,
    TrialResult,
    load_strategy,
)
from k4opt.optimization._scope import StudyScope
from k4opt.problem import parse_problem
from tests.unit.optimization.conftest import RADIUS, problem_document

COMPASS = Path(__file__).resolve().parents[3] / "examples" / "strategies" / "compass.py"

type Measure = Callable[[float], float | None]


@pytest.fixture(scope="module")
def compass() -> Any:
    return load_strategy(f"{COMPASS}:CompassSearch")


def context(**parameter: Any) -> StudyContext:
    """The context of a study of the trade-off problem, its radius declared as ``parameter``
    says (a continuous one in [100, 600] by default)."""
    document = problem_document(parameter=parameter or None)
    return StudyContext.of(StudyScope.of(parse_problem(document)), "a-study")


def ended(number: int, radius: float, measure: Measure) -> Feedback:
    """How trial ``number`` of ``radius`` ended: valid with what ``measure`` gives as the first
    objective, invalid where it gives nothing."""
    value = measure(radius)
    if value is None:
        return Feedback("invalid", TrialResult(number, "invalid", {RADIUS: radius}))
    result = TrialResult(number, "valid", {RADIUS: radius})
    return Feedback("valid", result, (value, -radius), feasible=True)


def drive(
    strategy: Any, measure: Measure, *, slots: int = 2, reverse: bool = False, rounds: int = 1000
) -> list[tuple[str, int, Any]]:
    """Run ``strategy`` as a study with ``slots`` does, asking while a slot is free and telling
    what ran once nothing more is proposed, in order or ``reverse``d; every call it was told
    (``proposed``, ``told``), at most ``rounds`` batches of them."""
    calls: list[tuple[str, int, Any]] = []
    number = 0
    for _ in range(rounds):
        batch = []
        while len(batch) < slots:
            answer = strategy.propose(number)
            if not isinstance(answer, Proposal):
                break
            strategy.proposed(number, answer)
            calls.append(("proposed", number, answer))
            batch.append((number, float(answer.values[RADIUS])))
            number += 1
        if not batch:
            assert answer is EXHAUSTED  # nothing under way, nothing to propose: it is done
            return calls
        for told, radius in reversed(batch) if reverse else batch:
            feedback = ended(told, radius, measure)
            strategy.tell(told, feedback)
            calls.append(("told", told, feedback))
    return calls


def radii(calls: list[tuple[str, int, Any]]) -> list[float]:
    return [float(item.values[RADIUS]) for kind, _, item in calls if kind == "proposed"]


def test_it_finds_the_least_of_what_it_is_told_and_ends_by_itself(compass: Any) -> None:
    strategy = compass(context())
    proposed = radii(drive(strategy, lambda radius: (radius - 420.0) ** 2))
    assert proposed[0] == 350.0  # the middle of [100, 600] first
    assert len(set(proposed)) == len(proposed)  # a point measured once is not proposed again
    assert min(abs(radius - 420.0) for radius in proposed) < 10.0  # the last step: 2% of 500
    assert strategy.exhausted and len(proposed) < 40


def test_it_never_compares_a_trial_without_a_value(compass: Any) -> None:
    proposed = radii(drive(compass(context()), lambda radius: radius if radius >= 250 else None))
    best = min(radius for radius in proposed if radius >= 250)
    assert 250.0 <= best < 260.0  # as close to the edge of the valid ones as its last step


def test_the_order_a_rounds_trials_end_in_changes_nothing(compass: Any) -> None:
    def measure(radius: float) -> float:
        return abs(radius - 180.0)

    assert radii(drive(compass(context()), measure)) == radii(
        drive(compass(context()), measure, reverse=True)
    )


def test_told_its_trials_again_it_goes_on_where_it_was(compass: Any) -> None:
    def measure(radius: float) -> float:
        return (radius - 500.0) ** 2

    live = compass(context())
    calls = drive(live, measure, rounds=3)
    again = compass(context())  # as a study made again from its ledger makes it
    for kind, number, item in calls:
        (again.proposed if kind == "proposed" else again.tell)(number, item)
    assert again.propose(len(radii(calls))) == live.propose(len(radii(calls)))
    assert radii(drive(again, measure)) == radii(drive(live, measure))


def test_an_integer_parameter_takes_integers_and_a_categorical_one_is_refused(
    compass: Any,
) -> None:
    proposed = radii(drive(compass(context(type="integer", bounds=[1, 9])), lambda r: -r))
    assert all(float(radius).is_integer() for radius in proposed) and max(proposed) == 9
    with pytest.raises(ValueError, match="numeric parameters only"):
        compass(context(type="categorical", choices=[100.0, 200.0]))


@pytest.mark.parametrize("bounds", [[0, 1], [1, 2], [0, 2], [5, 6], [-1, 1]])
def test_a_narrow_integer_range_is_searched_whole(compass: Any, bounds: list[int]) -> None:
    # a step of a quarter of the range rounds to no move at all: it moves by one integer at least
    proposed = radii(drive(compass(context(type="integer", bounds=bounds)), lambda r: r))
    assert sorted(set(proposed)) == list(range(bounds[0], bounds[1] + 1))
