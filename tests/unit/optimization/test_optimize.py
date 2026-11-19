"""``optimize``: studies of the trade-off problem on the mini geometry, with evaluators that read
no stage (and, once, two that share a fake ddsim)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.evaluation import EvaluationContractError
from k4opt.execution import Evaluations, RunError
from k4opt.optimization import STUDY_NAME, OptimizationError, load_study, optimize
from k4opt.optimization._render import trial_line
from k4opt.optimization._scope import limit_names
from k4opt.problem import load_problem
from k4opt.record import Assessment
from tests.support import simulation
from tests.support.evaluators import (
    candidate_check,
    cell_count,
    hit_energy,
    toy_significance,
    trade_off,
    yields,
)
from tests.support.simulation import COLLECTION
from tests.unit.optimization.conftest import (
    GUN,
    RADIUS,
    STRATEGY_KWARGS,
    Log,
    Told,
    endings,
    evaluator_from,
    ledger,
    summary,
)

TRADE = {"name": "test.trade_off", "version": "1"}


def study(
    problem: Path,
    output: Path,
    trials: int = 3,
    evaluators: list[Any] | None = None,
    strategy: str = "optuna",
    **kwargs: Any,
) -> Any:
    """A study with ``strategy``, as :data:`STRATEGY_KWARGS` runs it, unless ``kwargs`` say how."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
        kwargs.setdefault("seed", 1)
    elif "strategy" not in kwargs:
        kwargs = STRATEGY_KWARGS[strategy] | kwargs
    return optimize(problem, evaluators or [trade_off()], output, trials=trials, **kwargs)


def test_two_objectives_are_told_in_the_problems_order_and_directions(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str, told: Told
) -> None:
    problem = write_problem(
        objectives=[("analysis", "b", "maximize"), ("analysis", "a", "minimize")]
    )
    report = study(problem, tmp_path / "s", strategy=strategy)
    for number, feedback in told.feedback().items():
        metrics = {m["metric"]: m["value"] for m in summary(tmp_path / "s", number)["metrics"]}
        assert feedback.values == (metrics["b"], metrics["a"])
        assert feedback.record == tmp_path / "s" / f"trial-{number:06d}" / "summary.json"
    assert tuple(o.label for o in report.problem.objectives) == (
        "test.trade_off/analysis/b",
        "test.trade_off/analysis/a",
    )
    (first, *_) = report.trials
    assert list(first.objectives) == ["test.trade_off/analysis/b", "test.trade_off/analysis/a"]


def test_one_objective_takes_the_same_path(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str, told: Told
) -> None:
    problem = write_problem(objectives=[("analysis", "a", "minimize")])
    report = study(problem, tmp_path / "s", strategy=strategy)
    assert [f.values for f in told.feedback().values()] == [
        (t.objectives["test.trade_off/analysis/a"],) for t in report.trials
    ]
    # every candidate is feasible without constraints, and the best one is the front
    assert report.pareto == (
        min(report.trials, key=lambda t: t.objectives["test.trade_off/analysis/a"]).number,
    )


def test_every_trial_has_a_directory_with_its_complete_record(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str
) -> None:
    output = tmp_path / "s"
    report = study(write_problem(), output, trials=4, strategy=strategy)
    assert sorted(p.name for p in output.iterdir()) == sorted(
        [STUDY_NAME, "software", "trials.jsonl", *[f"trial-{n:06d}" for n in range(4)]]
    )
    document = json.loads((output / STUDY_NAME).read_text())
    assert document["id"] == report.id and document["strategy"]["name"] == strategy
    held = ledger(output)
    assert held.study == document["id"] and len(held.sessions) == 1
    for entry, reported in zip(held.entries, report.trials, strict=True):
        name = f"trial-{entry.number:06d}"
        data = (output / name / "summary.json").read_bytes()
        record = json.loads(data)
        assert record["run"] == {"study": document["id"], "trial": entry.number}
        assert record["assessment"]["feasible"] is True
        assert entry.ending is not None and entry.ending.kind == "recorded"
        assert entry.ending.status == "valid"
        assert entry.ending.record == {
            "id": record["id"],
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        # what the strategy noted of its proposal: Optuna its session's seed, the grid the point
        assert set(entry.proposal) == ({"seed"} if strategy == "optuna" else {"point"})
        assert (reported.record, report.record_path(reported)) == (
            record["id"],
            output / name / "summary.json",
        )
        assert dict(entry.values) == record["candidate"]


def test_invalid_and_failed_candidates_are_told_without_values_and_the_study_goes_on(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str, told: Told
) -> None:
    settings = {"invalid_below": 350, "failed_below": 200}
    problem = write_problem(
        parameter={
            "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": [450, 300, 150],  # valid, invalid, failed; the probe takes 450
        },
        evaluators=[TRADE | {"settings": settings}],
    )
    trials = 12 if strategy == "optuna" else 3  # the grid has three points
    report = study(problem, tmp_path / "s", trials=trials, strategy=strategy, grid=None)
    assert len(report.trials) == trials  # nothing stopped it
    feedback = told.feedback()
    for reported in report.trials:
        x = reported.candidate[RADIUS]
        expected = "failed" if x < 200 else "invalid" if x < 350 else "valid"
        told_now = feedback[reported.number]
        assert reported.status == told_now.status == expected
        if expected == "valid":
            assert told_now.values is not None and reported.feasible is True
        else:
            assert told_now.values is None and told_now.feasible is None
            assert reported.objectives is None and reported.feasible is None
            assert f"x = {x} is below" in (told_now.message or "")
            assert told_now.constraints == {}  # never a made-up 0 or inf
    assert {t.status for t in report.trials} == {"valid", "invalid", "failed"}


def test_a_study_of_nothing_but_failures_runs_to_its_end_with_an_empty_front(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem(evaluators=[TRADE | {"settings": {"failed_below": 700}}])
    report = study(problem, tmp_path / "s", trials=5)
    assert [t.status for t in report.trials] == ["failed"] * 5
    assert report.pareto == ()


def test_without_a_feasible_candidate_the_front_is_empty(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str, told: Told
) -> None:
    constraints = [{"stage": "analysis", "metric": "a", "max": 50, "unit": "mm"}]
    report = study(write_problem(constraints=constraints), tmp_path / "s", 4, strategy=strategy)
    assert [(t.status, t.feasible) for t in report.trials] == [("valid", False)] * 4
    assert report.pareto == ()
    for trial in report.trials:  # told with their real objectives, infeasible
        feedback = told.feedback()[trial.number]
        assert feedback.status == "valid" and feedback.feasible is False
        assert feedback.values == tuple(trial.objectives.values())  # type: ignore[union-attr]
        x = trial.candidate[RADIUS]
        assert feedback.constraints == {"test.trade_off/analysis/a:max": x - 50}


def test_limits_on_several_metrics_are_told_in_units_of_their_scales(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str, told: Told
) -> None:
    # a in mm and b in 1/mm: the optimizer adds dimensionless violations, never mm and 1/mm
    constraints = [
        {"stage": "analysis", "metric": "a", "max": 300, "unit": "mm", "scale": 100},
        {"stage": "analysis", "metric": "b", "min": 0.0025, "scale": 0.001},
    ]
    report = study(write_problem(constraints=constraints), tmp_path / "s", 4, strategy=strategy)
    for trial in report.trials:
        assert trial.assessment is not None
        x = trial.candidate[RADIUS]
        margins = {limit.name: limit.margin for limit in trial.assessment.limits}
        assert margins == {  # the physical margins stay in the record
            "test.trade_off/analysis/a:max": x - 300,
            "test.trade_off/analysis/b:min": pytest.approx(0.0025 - 1 / x),
        }
        assert told.feedback()[trial.number].constraints == {
            "test.trade_off/analysis/a:max": pytest.approx((x - 300) / 100),
            "test.trade_off/analysis/b:min": pytest.approx((0.0025 - 1 / x) / 0.001),
        }
        # the verdict is the physical one, whatever the scale
        assert trial.feasible is (x <= 300 and 1 / x >= 0.0025)
    study_json = json.loads((tmp_path / "s" / STUDY_NAME).read_text())
    scales = [c.get("scale") for c in study_json["problem"]["definition"]["constraints"]]
    assert scales == [100, 0.001]  # part of the study's identity


def test_the_same_seed_proposes_the_same_candidates(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    problem = write_problem()
    first = study(problem, tmp_path / "a", trials=4, seed=5)
    second = study(problem, tmp_path / "b", trials=4, seed=5)
    assert [t.candidate for t in first.trials] == [t.candidate for t in second.trials]
    other = study(problem, tmp_path / "c", trials=4, seed=6)
    assert [t.candidate for t in other.trials] != [t.candidate for t in first.trials]


def test_check_and_evaluate_see_each_trials_own_candidate(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str
) -> None:
    problem = write_problem(
        evaluators=[{"name": "test.candidate_check", "version": "1"}],
        objectives=[("analysis", "seen", "minimize")],
    )
    checking = candidate_check()
    checking.instances.clear()  # type: ignore[attr-defined]  # those of earlier tests
    checking.checked.clear()  # type: ignore[attr-defined]
    report = study(problem, tmp_path / "s", trials=4, evaluators=[checking], strategy=strategy)
    proposed = [dict(t.candidate) for t in report.trials]
    # no instance binds the class: its check() is asked once per trial, about that trial alone,
    # and each trial creates one new instance, which evaluates that trial's values only
    values = [c[RADIUS] for c in proposed]
    assert checking.checked == values  # type: ignore[attr-defined]
    assert [i.evaluated for i in checking.instances] == [[v] for v in values]  # type: ignore[attr-defined]
    assert len(set(map(str, proposed))) == 4


def checked_problem(write_problem: Callable[..., Path], **settings: float) -> Path:
    """A problem whose radius is 150 or 250, measured by ``test.candidate_check``."""
    return write_problem(
        parameter={
            "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": [150, 250],
        },
        evaluators=[{"name": "test.candidate_check", "version": "1", "settings": settings}],
        objectives=[("analysis", "seen", "minimize")],
    )


def test_a_candidate_an_evaluator_rejects_is_rejected_and_the_study_goes_on(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str, told: Told
) -> None:
    problem = checked_problem(write_problem, reject=250)
    trials = 8 if strategy == "optuna" else 2
    evaluators = [candidate_check()]
    report = study(problem, tmp_path / "s", trials, evaluators, strategy=strategy, grid=None)
    statuses = {t.candidate[RADIUS]: t.status for t in report.trials}
    assert statuses == {150: "valid", 250: "rejected"}  # both proposed, and the study ran on
    for trial in report.trials:
        if trial.status != "rejected":
            continue
        # no values, no constraint values, no record and nothing created: never an evaluation
        feedback = told.feedback()[trial.number]
        assert (feedback.status, feedback.values, feedback.record) == ("rejected", None, None)
        assert not feedback.constraints
        assert trial.assessment is None and trial.record is None and trial.objectives is None
        assert trial.message == "test.candidate_check: this evaluator cannot measure 250"
        assert not (tmp_path / "s" / f"trial-{trial.number:06d}").exists()
        assert report.record_path(trial) is None
        assert "rejected" in trial_line(trial)
    assert set(report.pareto) <= {t.number for t in report.trials if t.status == "valid"}
    assert report.outcomes["rejected"] == sum(t.status == "rejected" for t in report.trials)
    assert load_study(tmp_path / "s") == report  # read back alike, rejections included


def test_a_check_that_ends_otherwise_than_rejecting_stops_the_study_at_it(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = checked_problem(write_problem, refuse=250)
    with pytest.raises(OptimizationError, match=r"trial (\d+) stops the study .*work with 250"):
        study(problem, tmp_path / "s", trials=20, evaluators=[candidate_check()])
    *before, stopped = ledger(tmp_path / "s").entries
    assert all(t.values[RADIUS] == 150 and t.ending.kind == "recorded" for t in before)
    assert stopped.ending.kind == "error" and stopped.values[RADIUS] == 250
    assert "EvaluationContractError" in stopped.ending.message
    assert load_study(tmp_path / "s").trial(stopped.number).status == "error"
    assert not (tmp_path / "s" / f"trial-{stopped.number:06d}").exists()  # nothing was created
    assert not (tmp_path / "s" / ".lock").exists()


def test_a_check_is_only_ever_asked_about_proposed_candidates(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    # it refuses the lower bound, which a continuous sampler does not propose: the study runs
    problem = write_problem(
        evaluators=[{"name": "test.candidate_check", "version": "1", "settings": {"reject": 100}}],
        objectives=[("analysis", "seen", "minimize")],
    )
    report = study(problem, tmp_path / "s", trials=4, evaluators=[candidate_check()])
    assert [t.status for t in report.trials] == ["valid"] * 4


BROKEN = {
    "missing": """
        from k4opt.evaluation import Evaluator

        class Probe(Evaluator):
            name, version = "test.trade_off", "1"
            settings = {"x": float}  # the candidate's value, which it sets
            metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

            def evaluate(self, request):
                return {("analysis", "a"): 1.0}
    """,
    "unit": """
        from k4opt.evaluation import EvaluationResult, Evaluator, Measurement

        class Probe(Evaluator):
            name, version = "test.trade_off", "1"
            settings = {"x": float}  # the candidate's value, which it sets
            metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

            def evaluate(self, request):
                a = Measurement("analysis", "a", 1.0, "cm")
                return EvaluationResult.valid([a, Measurement("analysis", "b", 1.0, "1/mm")])
    """,
    "bug": """
        from k4opt.evaluation import Evaluator

        class Probe(Evaluator):
            name, version = "test.trade_off", "1"
            settings = {"x": float}  # the candidate's value, which it sets
            metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

            def evaluate(self, request):
                return 1 / 0
    """,
}


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("missing", "EvaluationContractError: evaluator 'test.trade_off' returned a valid result"),
        ("unit", "EvaluationContractError: measurement analysis/a has unit 'cm'"),
        ("bug", "ZeroDivisionError: division by zero"),
    ],
)
def test_an_evaluator_that_breaks_the_contract_stops_the_study(
    tmp_path: Path, write_problem: Callable[..., Path], kind: str, error: str
) -> None:
    broken = evaluator_from(tmp_path / "evaluators", f"broken_{kind}", BROKEN[kind])
    with pytest.raises(OptimizationError, match="trial 0 stops the study") as info:
        study(write_problem(), tmp_path / "s", evaluators=[broken])
    assert error in str(info.value)
    ((trial,),) = [ledger(tmp_path / "s").entries]
    assert trial.ending.kind == "error" and trial.ending.message.startswith(error)
    assert not (tmp_path / "s" / "trial-000000" / "summary.json").exists()


def _tamper(monkeypatch: pytest.MonkeyPatch, change: Callable[[dict[str, Any]], None]) -> None:
    """Change each published record with ``change`` before the study reads it back."""
    following = Evaluations.__next__

    def tampering(self: Evaluations) -> Any:
        request, outcome = following(self)
        if not isinstance(outcome, Exception):
            data = json.loads(outcome.summary_path.read_text())
            change(data)
            outcome.summary_path.write_text(json.dumps(data))
        return request, outcome

    monkeypatch.setattr(Evaluations, "__next__", tampering)


def test_a_record_that_does_not_hold_up_stops_the_study(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def tampering(data: dict[str, Any]) -> None:
        data["metrics"][0]["value"] = 1.0  # not what its evaluator measured

    _tamper(monkeypatch, tampering)
    with pytest.raises(
        OptimizationError, match=r"trial 0 stops the study.*metric test\.trade_off/analysis/a"
    ):
        study(write_problem(), tmp_path / "s")
    assert endings(tmp_path / "s") == ["error"]  # its record is on disk, but not kept
    assert load_study(tmp_path / "s").trials[0].record is None


def test_a_record_of_other_external_geometry_files_stops_the_study(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def tampering(data: dict[str, Any]) -> None:
        for entry in data["geometry"]["externals"].values():
            entry["sha256"] = "0" * 64  # not what the study read

    _tamper(monkeypatch, tampering)
    message = r"trial 0 stops the study.*through the geometry's external references"
    with pytest.raises(OptimizationError, match=message):
        study(write_problem(), tmp_path / "s")


def test_the_strategy_is_told_what_the_record_on_disk_holds(
    tmp_path: Path, write_problem: Callable[..., Path], strategy: str, told: Told
) -> None:
    constraints = [{"stage": "analysis", "metric": "a", "min": 150, "max": 450, "unit": "mm"}]
    problem = write_problem(constraints=constraints)
    study(problem, tmp_path / "s", trials=6, strategy=strategy)
    spec = load_problem(problem)
    assert sorted(told.feedback()) == list(range(6))
    for number, feedback in told.feedback().items():
        assessment = Assessment.from_summary(spec, summary(tmp_path / "s", number))
        assert feedback.values == tuple(float(m.value) for m in assessment.objectives or ())
        assert feedback.constraints == {limit.name: limit.margin for limit in assessment.limits}
        assert feedback.feasible is assessment.feasible


def test_a_strategy_that_fails_to_propose_stops_the_study_as_an_error(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.optimization import _grid

    def failing(self: Any, number: int) -> Any:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(_grid.Grid, "propose", failing)  # outside any trial
    with pytest.raises(OptimizationError, match="the strategy could not propose trial 0: ") as info:
        study(write_problem(), tmp_path / "s", strategy="grid")
    assert "OSError: [Errno 5] Input/output error" in str(info.value)
    assert not (tmp_path / "s" / ".lock").exists() and endings(tmp_path / "s") == []


def test_a_bug_of_k4opt_is_not_dressed_up_as_a_study_error(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.optimization import _run

    def buggy(*args: Any, **kwargs: Any) -> Any:
        return 1 / 0

    monkeypatch.setattr(_run, "results_of", buggy)
    with pytest.raises(ZeroDivisionError):
        study(write_problem(), tmp_path / "s", strategy="grid")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trials": 0}, "trials must be a positive integer"),
        ({"trials": True}, "trials must be a positive integer"),
        ({"seed": -1}, r"seed must be in \[0, 2\*\*32\)"),
        ({"seed": 2**32}, r"seed must be in \[0, 2\*\*32\)"),
        ({"seed": "1"}, "seed must be an integer"),
        ({"resume": True, "seed": 3}, "a resumed study keeps the strategy it was made with"),
        ({"resume": True, "strategy": "grid"}, "leave out --strategy, --grid and --seed"),
        ({"strategy": "annealing"}, "strategy 'annealing' is none of optuna, grid"),
        ({"strategy": "grid", "seed": 3}, "the grid draws nothing at random"),
        ({"grid": {RADIUS: [200.0]}}, "--grid gives the values of the grid strategy"),
        ({"strategy": "grid"}, "radius is continuous, so the grid needs the values"),
        ({"strategy": "grid", "grid": {"x": [1.0]}}, "x is no parameter the study searches"),
        ({"strategy": "grid", "grid": {RADIUS: [50.0]}}, r"50\.0 is outside the bounds"),
        ({"strategy": "grid", "grid": {RADIUS: [200, 200.0]}}, "takes 200.0 more than once"),
        ({"strategy": "grid", "grid": {RADIUS: []}}, "needs a list of one value or more"),
    ],
)
def test_unusable_arguments_are_refused_before_anything_is_created(
    tmp_path: Path, write_problem: Callable[..., Path], kwargs: dict[str, Any], message: str
) -> None:
    kwargs = {"trials": 1, **kwargs}
    with pytest.raises(OptimizationError, match=message):
        optimize(write_problem(), [trade_off()], tmp_path / "s", **kwargs)
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize("workers", [0, True, "2"])
def test_unusable_workers_are_refused_by_the_session_before_anything_is_created(
    tmp_path: Path, write_problem: Callable[..., Path], workers: object
) -> None:
    with pytest.raises(RunError, match="workers must be a positive integer, 'auto' or 'inline'"):
        optimize(write_problem(), [trade_off()], tmp_path / "s", trials=1, workers=workers)  # type: ignore[arg-type]
    assert not (tmp_path / "s").exists()


def test_an_evaluator_without_a_source_hash_cannot_make_a_study(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    class Inline:  # defined here: its file was imported before anything could hash it
        name, version = "test.trade_off", "1"
        settings = {"x": float}  # the candidate's value, which it sets
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request: Any) -> Any:
            return {("analysis", "a"): 1.0, ("analysis", "b"): 1.0}

    with pytest.raises(OptimizationError, match=r"'test\.trade_off' has no source hash"):
        optimize(write_problem(), [Inline], tmp_path / "s", trials=1)
    assert not (tmp_path / "s").exists()


def test_a_new_study_needs_a_new_or_empty_directory(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem()
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "notes.txt").write_text("mine")
    with pytest.raises(OptimizationError, match="is in use; resume the study in it with --resume"):
        study(problem, tmp_path / "used")
    assert [p.name for p in (tmp_path / "used").iterdir()] == ["notes.txt"]
    (tmp_path / "empty").mkdir()
    assert len(study(problem, tmp_path / "empty", trials=1).trials) == 1
    with pytest.raises(OptimizationError, match="is in use"):  # a study now, so not a new one
        study(problem, tmp_path / "empty")


def test_a_check_mistake_in_the_settings_costs_no_trial(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem(evaluators=[TRADE | {"settings": {"invalid_below": "high"}}])
    with pytest.raises(EvaluationContractError, match="invalid_below"):
        study(problem, tmp_path / "s")
    assert not (tmp_path / "s").exists()


def test_evaluators_that_read_the_simulation_share_one_ddsim_run_per_trial(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    reads = {"stage": "simulation", "collections": {"hits": COLLECTION}}
    problem = write_problem(
        evaluators=[
            {"name": "test.cell_count", "version": "1"} | reads,
            {"name": "test.hit_energy", "version": "1"} | reads,
        ],
        objectives=[
            ("test.cell_count", "simulation", "ecal_barrel.mean_hit_cells", "minimize"),
            ("test.hit_energy", "simulation", "ecal_barrel.mean_hit_energy", "maximize"),
        ],
    )
    document = __import__("yaml").safe_load(problem.read_text())
    document["samples"] = {"gamma": {}}
    document["samples"]["gamma"] = {
        "generation": {
            "executor": "gun",
            "particle": "gamma",
            "energy": {"value": 10, "unit": "GeV"},
            "direction": {
                "distribution": "uniform",
                "theta": [80, 100],
                "phi": [0, 360],
                "unit": "deg",
            },
        },
        "simulation": {"executor": "ddsim", "events": 2, "seed": 7},
    }
    problem.write_text(__import__("yaml").safe_dump(document))
    log = Log()
    report = study(
        problem, tmp_path / "s", trials=3, evaluators=[cell_count(), hit_energy()], **log.kwargs()
    )
    assert len(fake.calls) == 3  # one simulation per trial, for both evaluators
    assert [t.status for t in report.trials] == ["valid"] * 3
    # each unexpected collection is warned about once, not once per trial
    assert len([w for w in log.warnings if "is not expected to write" in w]) == 2


SLASHED = """
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.slashed", "1"
        settings = {"x": float}  # the candidate's value, which it sets
        metrics = {("analysis/x", "y"): "1", ("analysis", "x/y"): "1"}

        def evaluate(self, request):
            x = request.settings["x"]
            return {("analysis/x", "y"): x, ("analysis", "x/y"): -x}
"""


def test_metrics_whose_names_hold_a_slash_are_told_and_reported_apart(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    # joined with "/" as they are, both would be test.slashed/analysis/x/y
    problem = write_problem(
        evaluators=[{"name": "test.slashed", "version": "1"}],
        objectives=[("analysis/x", "y", "minimize"), ("analysis", "x/y", "minimize")],
        constraints=[
            {"stage": "analysis/x", "metric": "y", "max": 1000, "unit": "1", "scale": 1},
            {"stage": "analysis", "metric": "x/y", "max": 1000, "unit": "1", "scale": 1},
        ],
    )
    slashed = evaluator_from(tmp_path / "evaluators", "slashed", SLASHED)
    report = optimize(problem, [slashed], tmp_path / "s", trials=2, seed=1)
    assert tuple(o.label for o in report.problem.objectives) == (
        "test.slashed/analysis%2Fx/y",
        "test.slashed/analysis/x%2Fy",
    )
    assert tuple(limit_names(report.problem.criteria())) == (
        "test.slashed/analysis%2Fx/y:max",
        "test.slashed/analysis/x%2Fy:max",
    )
    for trial in report.trials:
        assert trial.objectives is not None
        (x,) = trial.candidate.values()
        assert list(trial.objectives.values()) == [x, -x]  # neither overwrites the other


def test_a_trial_whose_sample_fails_is_failed_and_keeps_the_other_samples_values(
    tmp_path: Path, sampled_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    simulated = 0

    def second_pions_crash(argv: list[str], **kwargs: Any) -> Any:
        nonlocal simulated
        code = fake(argv, **kwargs)
        simulated += 1
        return 2 if simulated == 4 else code  # the pions of the second trial

    monkeypatch.setattr(simulation.program, "execute", second_pions_crash)
    report = study(sampled_problem, tmp_path / "s", trials=3, evaluators=[cell_count()])
    assert len(fake.calls) == 6  # two samples, one simulation each, every trial
    assert [t.status for t in report.trials] == ["valid", "failed", "valid"]
    assert tuple(o.label for o in report.problem.objectives) == (
        "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
        "pion/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
    )
    statuses = [entry.ending.status for entry in ledger(tmp_path / "s").entries]
    assert statuses == ["valid", "failed", "valid"]
    record = summary(tmp_path / "s", 1)
    assert record["message"].startswith("pion/simulation failed")
    assert [m["value"] for m in record["metrics"]] == [1.5, None]  # gamma's is kept
    assert record["samples"]["gamma"]["evaluations"]["test.cell_count"]["status"] == "valid"


def test_a_combined_evaluator_is_optimized_and_a_trial_without_its_background_fails(
    tmp_path: Path, combined_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    simulated = 0

    def second_background_crashes(argv: list[str], **kwargs: Any) -> Any:
        nonlocal simulated
        code = fake(argv, **kwargs)
        simulated += 1
        return 2 if simulated == 4 else code  # the background of the second trial

    monkeypatch.setattr(simulation.program, "execute", second_background_crashes)
    evaluators = [toy_significance(), yields()]
    report = study(combined_problem, tmp_path / "s", trials=3, evaluators=evaluators)
    assert tuple(o.label for o in report.problem.objectives) == (
        "significance/analysis/expected_significance",
    )
    assert [t.status for t in report.trials] == ["valid", "failed", "valid"]
    statuses = [entry.ending.status for entry in ledger(tmp_path / "s").entries]
    assert statuses == ["valid", "failed", "valid"]
    # 2 fb and 30 fb over 1 ab^-1 = 1000 fb^-1, every event selected
    s, b = 2000.0, 30000.0
    expected = math.sqrt(2 * ((s + b) * math.log1p(s / b) - s))
    first = report.trials[0].objectives
    assert first == {"significance/analysis/expected_significance": pytest.approx(expected)}
    record = summary(tmp_path / "s", 1)
    assert record["message"].startswith("bkg/simulation failed")
    assert record["message"].endswith("; blocked: bkg/test.yields, significance")
    assert record["evaluations"]["significance"]["status"] == "blocked"
    assert record["samples"]["sig"]["evaluations"]["test.yields"]["status"] == "valid"


def test_a_diagnostic_that_fails_leaves_its_trial_complete_and_is_counted(
    tmp_path: Path, diagnosed_problem: Path, strategy: str, told: Told
) -> None:
    log = Log()
    report = study(diagnosed_problem, tmp_path / "s", trials=6, strategy=strategy, **log.kwargs())
    assert (
        "diagnostics (recorded, do not affect the result): check; mark an evaluator "
        "required: true if its result must count"
    ) in log.warnings
    failing = 0
    for trial in report.trials:
        record = summary(tmp_path / "s", trial.number)
        below = trial.candidate[RADIUS] < 350
        failing += below
        assert (told.feedback()[trial.number].status, record["status"]) == ("valid", "valid")
        assert len(trial.diagnostics) == int(below)
        assert record["diagnostics"] == (
            [{"task": "check", "status": "failed", "message": record["diagnostics"][0]["message"]}]
            if below
            else []
        )
        assert ("(1 diagnostic did not succeed)" in trial_line(trial)) is below
    assert 0 < failing < 6  # both ways, to mean much


def test_a_study_of_the_library_evaluates_inline_unless_told_otherwise(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    # as a script or a notebook defines its evaluator: given as it is, evaluated in this process
    optimize(write_problem(), [trade_off()], tmp_path / "s", trials=1, seed=1)
    summary = json.loads((tmp_path / "s" / "trial-000000" / "summary.json").read_text())
    assert summary["environment"]["observed"]["execution"] == {
        "backend": "inline",
    }


def test_a_candidate_a_reconstruction_does_not_take_is_rejected_and_read_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    from tests.support.evaluators import reco_cells
    from tests.support.stack import install_k4run

    job = tmp_path / "CLDConfig"
    job.mkdir()
    (job / "CLDReconstruction.py").write_text("# the job\n")
    path = write_problem(
        parameter={"targets": {"reconstruction": "cms"}, "type": "integer", "bounds": [240, 242]},
        evaluators=[
            {
                "name": "test.reco_cells",
                "version": "1",
                "stage": "reconstruction",
                "collections": {"hits": COLLECTION},
            }
        ],
        objectives=[("reconstruction", "ecal_barrel.mean_hit_cells", "minimize")],
    )
    document = yaml.safe_load(path.read_text())
    document["samples"] = {
        "gamma": {
            "generation": GUN | {"particle": "gamma"},
            "simulation": {"executor": "ddsim", "events": 2, "seed": 1},
            "reconstruction": {
                "executor": "k4run",
                "directory": str(job),
                "options": "CLDReconstruction.py",
            },
        }
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    simulation.install(monkeypatch)
    # the ends of the domain are taken, the value between them is not
    install_k4run(monkeypatch, refuse_args={"--cms=241": "argument --cms: invalid choice: '241'"})
    report = study(path, tmp_path / "s", trials=8, evaluators=[reco_cells()])
    statuses = {t.candidate[RADIUS]: t.status for t in report.trials}
    assert statuses == {240: "valid", 241: "rejected", 242: "valid"}
    (rejected,) = {t.message for t in report.trials if t.status == "rejected"}
    assert rejected.startswith("gamma/reconstruction: the job does not take its arguments")
    assert load_study(tmp_path / "s") == report  # the reader takes a reconstruction's rejection
