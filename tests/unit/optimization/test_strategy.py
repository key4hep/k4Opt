"""What a study and its strategy say to each other, whatever the strategy: scripted strategies
that propose what a study must not trust (malformed proposals), wait for trials, or are
exhausted; and Optuna's TPE, which proposes what a sampler holding the study all along would."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from k4opt.optimization import OptimizationError, StudyResults, _tree, load_study, optimize
from k4opt.optimization._proposal import EXHAUSTED, WAITING, Answer, Feedback, Past, Proposal
from tests.support.evaluators import candidate_check, trade_off
from tests.unit.optimization.conftest import RADIUS, endings, ledger

GRID = {"strategy": "grid", "grid": {RADIUS: [300.0]}}
"""What ``study.json`` records; the scripted strategies below take its place when it runs."""


class Scripted:
    """A strategy that answers ``script`` in turn, each an answer, a proposal, or a callable of
    what it was told so far; it keeps what it was made with and what it is told."""

    def __init__(self, script: list[Any], history: list[Past]) -> None:
        self.script, self.history = iter(script), history
        self.told: list[tuple[int, Feedback]] = []
        self.proposed: list[tuple[int, int]] = []  # each number, with how many were told then
        self.exhausted = False  # it says so by answering EXHAUSTED

    def propose(self, number: int) -> Proposal | Answer:
        answer = next(self.script, EXHAUSTED)
        answer = answer(self.told) if callable(answer) else answer
        if isinstance(answer, Proposal):
            self.proposed.append((number, len(self.told)))
        return answer

    def tell(self, number: int, feedback: Feedback) -> None:
        self.told.append((number, feedback))


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[Any]], list[Scripted]]:
    """Run the studies made from now on with :class:`Scripted` strategies of a script; the
    strategies made, in turn."""
    made: list[Scripted] = []

    def install(script: list[Any]) -> list[Scripted]:
        def opening(recorded: Any, scope: Any, history: Any) -> Scripted:
            strategy = Scripted(list(script), list(history))
            made.append(strategy)
            return strategy

        monkeypatch.setattr(_tree, "open_strategy", opening)
        return made

    return install


def run(problem: Path, output: Path, trials: int, **kwargs: Any) -> StudyResults:
    kwargs = kwargs if kwargs.get("resume") else GRID | kwargs
    evaluators = kwargs.pop("evaluators", [trade_off()])
    return optimize(problem, evaluators, output, trials=trials, **kwargs)


MALFORMED = [
    (
        Proposal({RADIUS: 700.0}),
        r"outside the study's domain: radius: 700\.0 is outside the bounds",
    ),
    (Proposal({RADIUS: "far"}), "radius: expected a number, got 'far'"),
    (Proposal({RADIUS: True}), "expected a number, got True"),
    (Proposal({}), "no value of radius"),
    (Proposal({RADIUS: 300.0, "depth": 1}), "'depth', which the study does not search"),
    (Proposal({RADIUS: 300.0}, {"model": object()}), "note of its proposal is not plain JSON"),
    ({RADIUS: 300.0}, r"proposed \{'radius': 300\.0\}, not a proposal"),
    (None, "the strategy proposed None, not a proposal"),
]


def test_a_malformed_proposal_ends_its_trial_unevaluated_and_the_study_goes_on(
    tmp_path: Path, write_problem: Callable[..., Path], scripted: Any
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    made = scripted([proposal for proposal, _ in MALFORMED] + [Proposal({RADIUS: 300.0})])
    report = run(problem, output, len(MALFORMED) + 1)
    *wrong, right = report.trials
    assert [t.status for t in wrong] == ["malformed"] * len(MALFORMED)
    assert right.status == "valid"
    for trial, (_, message) in zip(wrong, MALFORMED, strict=True):
        assert trial.message is not None
        assert __import__("re").search(message, trial.message), trial.message
        # nothing of it was made: no directory, no record, no value of what it proposed
        assert not (output / f"trial-{trial.number:06d}").exists()
        assert (trial.record, trial.assessment, dict(trial.candidate)) == (None, None, {})
    (strategy,) = made
    statuses = [(n, f.status, f.values, f.record) for n, f in strategy.told]
    assert statuses[: len(MALFORMED)] == [(n, "malformed", None, None) for n in range(8)]
    assert report.outcomes["malformed"] == len(MALFORMED) and report.outcomes["rejected"] == 0
    assert load_study(output).to_dict() == report.to_dict()
    assert StudyResults.from_dict(json.loads(report.to_json())).to_dict() == report.to_dict()
    assert endings(output) == ["malformed"] * len(MALFORMED) + ["recorded"]


def test_a_malformed_proposal_is_not_a_rejected_candidate(
    tmp_path: Path, write_problem: Callable[..., Path], scripted: Any
) -> None:
    problem = write_problem(
        evaluators=[{"name": "test.candidate_check", "version": "1", "settings": {"reject": 250}}],
        objectives=[("analysis", "seen", "minimize")],
    )
    scripted([Proposal({RADIUS: 250.0}), Proposal({RADIUS: 50.0}), Proposal({RADIUS: 300.0})])
    report = run(problem, tmp_path / "s", 3, evaluators=[candidate_check()])
    assert [t.status for t in report.trials] == ["rejected", "malformed", "valid"]
    rejected, malformed, _ = report.trials
    assert rejected.message == "test.candidate_check: this evaluator cannot measure 250.0"
    assert rejected.candidate == {RADIUS: 250.0}  # a candidate of the study, which it refused
    assert malformed.candidate == {} and "outside the bounds" in (malformed.message or "")


def test_a_strategy_that_waits_with_nothing_under_way_stops_the_study(
    tmp_path: Path, write_problem: Callable[..., Path], scripted: Any
) -> None:
    scripted([Proposal({RADIUS: 300.0}), WAITING])
    with pytest.raises(OptimizationError, match="waits for a trial to end, but none of the"):
        run(write_problem(), tmp_path / "s", 3)
    assert endings(tmp_path / "s") == ["recorded"]  # what ended before stays
    assert not (tmp_path / "s" / ".lock").exists()


def test_a_strategy_that_waits_is_asked_again_once_a_trial_ended(
    tmp_path: Path, write_problem: Callable[..., Path], scripted: Any
) -> None:
    # two at once, then waiting until both were told, then two more: a batch strategy
    batch = iter([Proposal({RADIUS: 150.0}), Proposal({RADIUS: 175.0})])
    later = iter([Proposal({RADIUS: 200.0}), Proposal({RADIUS: 225.0})])

    def step(told: list[Any]) -> Any:
        if len(told) < 2:
            return next(batch, WAITING)
        return next(later, EXHAUSTED)

    made = scripted([step] * 8)
    report = run(write_problem(), tmp_path / "s", 4, workers=2)
    assert [t.status for t in report.trials] == ["valid"] * 4
    (strategy,) = made
    # the first two proposed before anything was told, the others once both had been
    assert strategy.proposed == [(0, 0), (1, 0), (2, 2), (3, 2)]
    assert sorted(n for n, _ in strategy.told) == [0, 1, 2, 3]
    assert not report.exhausted  # its budget ended first


def test_a_strategy_exhausted_at_once_runs_no_trial(
    tmp_path: Path, write_problem: Callable[..., Path], scripted: Any
) -> None:
    scripted([EXHAUSTED])
    report = run(write_problem(), tmp_path / "s", 3)
    assert report.trials == () and report.exhausted
    assert ledger(tmp_path / "s").exhausted


def test_a_resumed_strategy_is_made_from_every_trial_of_the_study(
    tmp_path: Path, write_problem: Callable[..., Path], scripted: Any
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    made = scripted([Proposal({RADIUS: 150.0}, {"why": "low"}), Proposal({RADIUS: 50.0})])
    run(problem, output, 2)
    run(problem, output, 0, resume=True)
    first, resumed = made
    assert first.history == []
    assert [
        (p.number, dict(p.values), dict(p.note), p.feedback.status) for p in resumed.history
    ] == [
        (0, {RADIUS: 150.0}, {"why": "low"}, "valid"),
        (1, {}, {}, "malformed"),
    ]
    assert resumed.history[0].feedback == first.told[0][1]  # as it was told live


# --- Optuna's TPE ----------------------------------------------------------------------------


def test_optuna_proposes_what_a_sampler_that_held_the_study_all_along_would(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    optuna = pytest.importorskip("optuna")
    from optuna.trial import TrialState, create_trial

    from k4opt.optimization._optuna import CONSTANT_LIAR, STARTUP_TRIALS, search_space, session_seed
    from k4opt.problem import load_problem

    settings = {"invalid_below": 250.0}  # some trials without values: FAIL for TPE
    problem = write_problem(
        evaluators=[{"name": "test.trade_off", "version": "1", "settings": settings}]
    )
    output, seed = tmp_path / "s", 17
    first = optimize(problem, [trade_off()], output, trials=12, seed=seed)
    resumed = optimize(problem, [trade_off()], output, trials=4, resume=True)
    space = search_space(load_problem(problem))

    def tpe(session: int) -> Any:
        sampler = optuna.samplers.TPESampler(
            seed=session_seed(seed, session),
            n_startup_trials=STARTUP_TRIALS,
            constant_liar=CONSTANT_LIAR,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    def tell(study: Any, trial: Any, result: Any) -> None:
        if result.status == "valid":
            study.tell(trial, [float(v) for v in result.objectives.values()])
        else:
            study.tell(trial, state=TrialState.FAIL)

    # one session after another, each told every trial as it ended: the proposals of the study
    alone = tpe(0)
    for result in first.trials:
        trial = alone.ask(fixed_distributions=space)
        assert trial.params == dict(result.candidate)
        tell(alone, trial, result)
    # the resumed session: a sampler seeded anew, holding every trial of the first session
    again = tpe(12)
    for result in first.trials:
        values = (
            [float(v) for v in result.objectives.values()] if result.status == "valid" else None
        )
        state = TrialState.COMPLETE if values else TrialState.FAIL
        again.add_trial(
            create_trial(
                state=state, values=values, params=dict(result.candidate), distributions=space
            )
        )
    for result in resumed.trials[12:]:
        trial = again.ask(fixed_distributions=space)
        assert trial.params == dict(result.candidate)
        tell(again, trial, result)
    assert {t.status for t in first.trials} == {"valid", "invalid"}  # both kinds, to mean much
