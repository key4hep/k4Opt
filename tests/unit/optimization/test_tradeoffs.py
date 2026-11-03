"""Trade-offs against a reference trial: which feasible trials improve, keep or give up how much
of each objective, in its direction, relative or in its unit, within the uncertainties or not."""

from __future__ import annotations

import pytest

from k4opt.frozen import FrozenMap
from k4opt.optimization import OptimizationError, Rule, StudyResults, TrialResult, trade_offs
from k4opt.optimization._results import pareto_front
from k4opt.problem import parse_problem
from k4opt.record import Assessment, ObjectiveValue
from tests.unit.optimization.conftest import RADIUS, problem_document

EVALUATOR = "test.trade_off"
RES, COST = f"{EVALUATOR}/analysis/a", f"{EVALUATOR}/analysis/b"


def study(
    trials: list[tuple[str, float, float]],
    *,
    directions: tuple[str, str] = ("minimize", "minimize"),
    uncertainty: float | None = None,
) -> StudyResults:
    """A study of the objectives ``a`` (a resolution) and ``b`` (a cost), each trial
    ``(outcome, a, b)``, each value with ``uncertainty``."""
    spec = parse_problem(
        problem_document(
            objectives=[("analysis", "a", directions[0]), ("analysis", "b", directions[1])]
        )
    )
    results = []
    for number, (outcome, a, b) in enumerate(trials):
        if outcome == "failed":
            results.append(TrialResult(number, "failed", {RADIUS: 100.0 + number}))
            continue
        measured = tuple(
            ObjectiveValue(EVALUATOR, "analysis", metric, direction, value, "mm", uncertainty)
            for metric, direction, value in zip(("a", "b"), directions, (a, b), strict=True)
        )
        feasible = outcome == "feasible"
        results.append(
            TrialResult(
                number, "valid", {RADIUS: 100.0 + number}, Assessment(measured, (), feasible)
            )
        )
    return StudyResults(
        "trade-0", "id", "now", spec, FrozenMap({}), tuple(results), pareto_front(results, spec)
    )


TRIALS = [
    ("feasible", 1.00, 100.0),  # 0: the reference
    ("feasible", 1.04, 85.0),  # 1: 4 % worse, 15 % cheaper
    ("feasible", 1.08, 70.0),  # 2: 8 % worse, 30 % cheaper
    ("feasible", 0.99, 95.0),  # 3: better and cheaper
    ("feasible", 1.02, 98.0),  # 4: 2 % worse, 2 % cheaper
    ("infeasible", 1.01, 50.0),  # 5: never a match
    ("failed", 0.0, 0.0),  # 6: no values
]


def test_give_up_some_resolution_for_a_cheaper_design() -> None:
    found = trade_offs(study(TRIALS), 0, [Rule(RES, "worsen", 0.05), Rule(COST, "improve", 0.10)])
    assert found.matches == (1,) and found.reference.number == 0
    assert found.compared == 4  # the feasible trials but the reference
    assert found.excludes == {RES: 1, COST: 2}  # trial 2 too blurry; trials 3 and 4 too dear
    assert found.change[1][RES] == pytest.approx(0.04) and found.change[1][COST] == -15.0


def test_an_amount_in_the_objectives_unit_and_keeping_a_value() -> None:
    results = study(TRIALS)
    absolute = trade_offs(results, 0, [Rule(COST, "improve", 25.0, relative=False)])
    assert absolute.matches == (2,)
    kept = trade_offs(results, 0, [Rule(RES, "keep")])
    assert kept.matches == (3,)
    assert trade_offs(results, 0, []).matches == (1, 2, 3, 4)  # no rule: every feasible one


def test_better_and_worse_follow_each_objectives_direction() -> None:
    maximized = [("feasible", 1.0, 100.0), ("feasible", 1.1, 120.0), ("feasible", 0.9, 130.0)]
    results = study(maximized, directions=("maximize", "maximize"))
    assert trade_offs(results, 0, [Rule(COST, "improve", 0.1), Rule(RES, "keep")]).matches == (1,)
    assert trade_offs(results, 0, [Rule(RES, "worsen", 0.15)]).matches == (1, 2)


def test_differences_within_the_uncertainties_count_as_none() -> None:
    results = study(TRIALS, uncertainty=0.03)  # sqrt(2) * 0.03 = 0.042 on each difference
    noisy = trade_offs(results, 0, [Rule(RES, "keep")])
    assert noisy.matches == (1, 3, 4)  # 4 % and 2 % worse are within the uncertainty
    assert trade_offs(results, 0, [Rule(RES, "keep")], within_uncertainty=False).matches == (3,)
    # an improvement must exceed the uncertainty: 1 % better is no improvement at all
    assert trade_offs(results, 0, [Rule(RES, "improve", 0.0)]).matches == (1, 3, 4)
    assert trade_offs(results, 0, [Rule(RES, "improve", 0.001)]).matches == ()


def test_what_cannot_be_asked_is_refused() -> None:
    results = study([*TRIALS, ("feasible", 0.0, 0.0)])
    with pytest.raises(OptimizationError, match="trial 6 is failed and has no objective values"):
        trade_offs(results, 6, [])
    with pytest.raises(OptimizationError, match="the study has no trial 99"):
        trade_offs(results, 99, [])
    with pytest.raises(OptimizationError, match="the study has no objective 'nothing'"):
        trade_offs(results, 0, [Rule("nothing", "keep")])
    with pytest.raises(OptimizationError, match="two rules on"):
        trade_offs(results, 0, [Rule(RES, "keep"), Rule(RES, "worsen", 0.1)])
    with pytest.raises(OptimizationError, match=r"has .*= 0, of which no share can be asked"):
        trade_offs(results, 7, [Rule(RES, "worsen", 0.1)])
    assert trade_offs(results, 7, [Rule(RES, "worsen", 0.1, relative=False)]).compared == 5
    with pytest.raises(OptimizationError, match="it must be zero or more"):
        Rule(RES, "worsen", -0.1)
    with pytest.raises(OptimizationError, match="improves, worsens or keeps"):
        Rule(RES, "maybe")  # type: ignore[arg-type]
