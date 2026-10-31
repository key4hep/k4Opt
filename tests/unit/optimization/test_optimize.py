"""``optimize``: studies of the trade-off problem on the mini geometry, with evaluators that read
no stage (and, once, two that share a fake ddsim)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from optuna.exceptions import StorageInternalError
from optuna.study import StudyDirection
from optuna.trial import TrialState

from k4opt.evaluation import EvaluationContractError
from k4opt.optimization import STUDY_NAME, OptimizationError, optimize
from k4opt.problem import Assessment, EvaluationSession, load_problem
from tests.support import simulation
from tests.support.evaluators import (
    candidate_check,
    cell_count,
    hit_energy,
    trade_off,
)
from tests.support.simulation import COLLECTION
from tests.unit.optimization.conftest import (
    RADIUS,
    Log,
    evaluator_from,
    optuna_study,
    summary,
)

TRADE = {"name": "test.trade_off", "version": "1"}


def study(
    problem: Path, output: Path, trials: int = 3, evaluators: list[Any] | None = None, **kwargs: Any
) -> Any:
    kwargs.setdefault("seed", 1)
    return optimize(problem, evaluators or [trade_off()], output, trials=trials, **kwargs)


def test_two_objectives_are_told_in_the_problems_order_and_directions(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem(
        objectives=[("analysis", "b", "maximize"), ("analysis", "a", "minimize")]
    )
    report = study(problem, tmp_path / "s")
    optimizer = optuna_study(tmp_path / "s")
    assert optimizer.directions == [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    for trial in optimizer.trials:
        metrics = {
            m["metric"]: m["value"] for m in summary(tmp_path / "s", trial.number)["metrics"]
        }
        assert trial.values == [metrics["b"], metrics["a"]]
    assert report.objectives == ("test.trade_off/analysis/b", "test.trade_off/analysis/a")
    (first, *_) = report.trials
    assert list(first.objectives) == ["test.trade_off/analysis/b", "test.trade_off/analysis/a"]


def test_one_objective_takes_the_same_path(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    report = study(write_problem(objectives=[("analysis", "a", "minimize")]), tmp_path / "s")
    optimizer = optuna_study(tmp_path / "s")
    assert optimizer.directions == [StudyDirection.MINIMIZE]
    assert [t.values for t in optimizer.trials] == [
        [t.objectives["test.trade_off/analysis/a"]] for t in report.trials
    ]
    # every candidate is feasible without constraints, and the best one is the front
    assert report.pareto == (
        min(report.trials, key=lambda t: t.objectives["test.trade_off/analysis/a"]).number,
    )


def test_every_trial_has_a_directory_with_its_complete_record(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    output = tmp_path / "s"
    report = study(write_problem(), output, trials=4)
    assert sorted(p.name for p in output.iterdir()) == sorted(
        [STUDY_NAME, "study.journal", *[f"trial-{n:06d}" for n in range(4)]]
    )
    document = json.loads((output / STUDY_NAME).read_text())
    optimizer = optuna_study(output)
    assert optimizer.user_attrs["k4opt.study"] == document["id"] == report.id
    assert optimizer.user_attrs["k4opt.identity"] == document["identity"]
    for trial, reported in zip(optimizer.trials, report.trials, strict=True):
        name = f"trial-{trial.number:06d}"
        data = (output / name / "summary.json").read_bytes()
        record = json.loads(data)
        assert record["run"] == {"study": document["id"], "trial": trial.number}
        assert record["assessment"]["feasible"] is True
        assert trial.user_attrs["k4opt.dir"] == name
        assert trial.user_attrs["k4opt.record"] == {
            "id": record["id"],
            "sha256": hashlib.sha256(data).hexdigest(),
            "path": f"{name}/summary.json",
        }
        assert trial.user_attrs["k4opt.fingerprint"] == record["geometry"]["fingerprint"]
        assert isinstance(trial.user_attrs["k4opt.sampler_seed"], int)
        assert (reported.record, reported.directory) == (record["id"], output / name)
        assert trial.params == record["candidate"]


def test_invalid_and_failed_candidates_are_fail_trials_and_the_study_goes_on(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    settings = {"invalid_below": 350, "failed_below": 200}
    problem = write_problem(
        parameter={
            "geometry": {"constant": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": [450, 300, 150],  # valid, invalid, failed; the probe takes 450
        },
        evaluators=[TRADE | {"settings": settings}],
    )
    report = study(problem, tmp_path / "s", trials=12)
    assert len(report.trials) == 12  # nothing stopped it
    optimizer = optuna_study(tmp_path / "s")
    for trial, reported in zip(optimizer.trials, report.trials, strict=True):
        (x,) = trial.params.values()
        expected = "failed" if x < 200 else "invalid" if x < 350 else "valid"
        assert reported.status == trial.user_attrs["k4opt.status"] == expected
        if expected == "valid":
            assert trial.state == TrialState.COMPLETE and reported.feasible is True
        else:
            assert trial.state == TrialState.FAIL and trial.values is None
            assert reported.objectives is None and reported.feasible is None
            assert f"x = {x} is below" in trial.user_attrs["k4opt.message"]
            assert trial.constraints == {}  # never a made-up 0 or inf
    assert {t.status for t in report.trials} == {"valid", "invalid", "failed"}


def test_a_study_of_nothing_but_failures_runs_to_its_end_with_an_empty_front(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem(evaluators=[TRADE | {"settings": {"failed_below": 700}}])
    report = study(problem, tmp_path / "s", trials=5)
    assert [t.status for t in report.trials] == ["failed"] * 5
    assert report.pareto == ()


def test_without_a_feasible_candidate_the_front_is_empty(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    constraints = [{"stage": "analysis", "metric": "a", "max": 50, "unit": "mm"}]
    report = study(write_problem(constraints=constraints), tmp_path / "s", trials=4)
    assert [(t.status, t.feasible) for t in report.trials] == [("valid", False)] * 4
    assert report.pareto == ()
    for trial in optuna_study(tmp_path / "s").trials:  # complete, with their real objectives
        assert trial.state == TrialState.COMPLETE
        (x,) = trial.params.values()
        assert trial.constraints == {"test.trade_off/analysis/a:max": x - 50}


def test_limits_on_several_metrics_are_warned_about(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    constraints = [
        {"stage": "analysis", "metric": "a", "max": 500, "unit": "mm"},
        {"stage": "analysis", "metric": "b", "min": 0.0025},
    ]
    log = Log()
    study(write_problem(constraints=constraints), tmp_path / "s", trials=1, **log.kwargs())
    (warning,) = [w for w in log.warnings if w.startswith("constraints on")]
    assert warning.startswith("constraints on 2 metrics") and "different units" in warning


def test_the_same_seed_proposes_the_same_candidates(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem()
    first = study(problem, tmp_path / "a", trials=4, seed=5)
    second = study(problem, tmp_path / "b", trials=4, seed=5)
    assert [t.candidate for t in first.trials] == [t.candidate for t in second.trials]
    other = study(problem, tmp_path / "c", trials=4, seed=6)
    assert [t.candidate for t in other.trials] != [t.candidate for t in first.trials]


def test_check_and_evaluate_see_each_trials_own_candidate(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem(
        evaluators=[{"name": "test.candidate_check", "version": "1"}],
        objectives=[("analysis", "seen", "minimize")],
    )
    checking = candidate_check()
    checking.instances.clear()  # type: ignore[attr-defined]  # those of earlier tests
    report = study(problem, tmp_path / "s", trials=4, evaluators=[checking])
    proposed = [dict(t.candidate) for t in report.trials]
    # one instance to bind the class when the study opens, used for nothing else, then a new one
    # for each trial, which is checked and evaluated with that trial's values only
    probe, *trials = checking.instances  # type: ignore[attr-defined]
    assert (probe.checked, probe.evaluated) == ([], [])
    assert [(i.checked, i.evaluated) for i in trials] == [([c], [c]) for c in proposed]
    assert len(set(map(str, proposed))) == 4


def test_a_check_that_refuses_a_candidate_stops_the_study_at_it(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem(
        parameter={
            "geometry": {"constant": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": [150, 250],
        },
        evaluators=[{"name": "test.candidate_check", "version": "1", "settings": {"reject": 250}}],
        objectives=[("analysis", "seen", "minimize")],
    )
    with pytest.raises(
        OptimizationError, match=r"trial (\d+) stops the study .*cannot measure 250"
    ):
        study(problem, tmp_path / "s", trials=20, evaluators=[candidate_check()])
    trials = optuna_study(tmp_path / "s").trials
    *before, stopped = trials
    assert all(t.params[RADIUS] == 150 and t.state == TrialState.COMPLETE for t in before)
    assert stopped.state == TrialState.FAIL and stopped.params[RADIUS] == 250
    assert "cannot measure 250" in stopped.user_attrs["k4opt.error"]
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
            metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

            def evaluate(self, request):
                return {("analysis", "a"): 1.0}
    """,
    "unit": """
        from k4opt.evaluation import EvaluationResult, Evaluator, Measurement

        class Probe(Evaluator):
            name, version = "test.trade_off", "1"
            metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

            def evaluate(self, request):
                a = Measurement("analysis", "a", 1.0, "cm")
                return EvaluationResult.valid([a, Measurement("analysis", "b", 1.0, "1/mm")])
    """,
    "bug": """
        from k4opt.evaluation import Evaluator

        class Probe(Evaluator):
            name, version = "test.trade_off", "1"
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
    (trial,) = optuna_study(tmp_path / "s").trials
    assert trial.state == TrialState.FAIL and trial.user_attrs["k4opt.error"].startswith(error)
    assert not (tmp_path / "s" / "trial-000000" / "summary.json").exists()


def test_a_record_that_does_not_hold_up_stops_the_study(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluate = EvaluationSession.evaluate

    def tampering(self: EvaluationSession, *args: Any, **kwargs: Any) -> Any:
        outcome = evaluate(self, *args, **kwargs)
        data = json.loads(outcome.summary_path.read_text())
        data["metrics"][0]["value"] = 1.0  # not what its evaluator measured
        outcome.summary_path.write_text(json.dumps(data))
        return outcome

    monkeypatch.setattr(EvaluationSession, "evaluate", tampering)
    with pytest.raises(
        OptimizationError, match=r"trial 0 stops the study.*metric test\.trade_off/analysis/a"
    ):
        study(write_problem(), tmp_path / "s")
    (trial,) = optuna_study(tmp_path / "s").trials
    assert trial.state == TrialState.FAIL and "k4opt.status" not in trial.user_attrs


def test_a_record_of_other_external_geometry_files_stops_the_study(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluate = EvaluationSession.evaluate

    def tampering(self: EvaluationSession, *args: Any, **kwargs: Any) -> Any:
        outcome = evaluate(self, *args, **kwargs)
        data = json.loads(outcome.summary_path.read_text())
        for entry in data["geometry"]["externals"].values():
            entry["sha256"] = "0" * 64  # not what the study read
        outcome.summary_path.write_text(json.dumps(data))
        return outcome

    monkeypatch.setattr(EvaluationSession, "evaluate", tampering)
    message = r"trial 0 stops the study.*through the geometry's external references"
    with pytest.raises(OptimizationError, match=message):
        study(write_problem(), tmp_path / "s")


def test_the_optimizer_is_told_what_the_record_on_disk_holds(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    constraints = [{"stage": "analysis", "metric": "a", "min": 150, "max": 450, "unit": "mm"}]
    problem = write_problem(constraints=constraints)
    study(problem, tmp_path / "s", trials=6)
    spec = load_problem(problem)
    for trial in optuna_study(tmp_path / "s").trials:
        assessment = Assessment.from_summary(spec, summary(tmp_path / "s", trial.number))
        assert trial.values == [float(m.value) for m in assessment.objectives or ()]
        assert trial.constraints == {limit.name: limit.margin for limit in assessment.limits}
        assert trial.user_attrs["k4opt.feasible"] is assessment.feasible


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            lambda: StorageInternalError("the database is gone"),
            "StorageInternalError: the database",
        ),
        (lambda: OSError(5, "Input/output error"), "OSError: [Errno 5] Input/output error"),
    ],
)
def test_a_failure_of_the_optimizer_or_its_storage_stops_the_study_as_an_error(
    tmp_path: Path,
    write_problem: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: Callable[[], Exception],
    message: str,
) -> None:
    import optuna

    def failing(self: optuna.Study, *args: Any, **kwargs: Any) -> Any:
        raise failure()

    monkeypatch.setattr(optuna.Study, "ask", failing)  # outside any trial
    with pytest.raises(OptimizationError, match="the study cannot go on: ") as info:
        study(write_problem(), tmp_path / "s")
    assert message in str(info.value)
    assert not (tmp_path / "s" / ".lock").exists()


def test_a_bug_of_k4opt_is_not_dressed_up_as_a_study_error(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.optimization._optuna import run

    def buggy(run: Any) -> Any:
        return 1 / 0

    monkeypatch.setattr(run, "_study_report", buggy)
    with pytest.raises(ZeroDivisionError):
        study(write_problem(), tmp_path / "s")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trials": 0}, "trials must be a positive integer"),
        ({"trials": True}, "trials must be a positive integer"),
        ({"seed": -1}, r"seed must be in \[0, 2\*\*32\)"),
        ({"seed": 2**32}, r"seed must be in \[0, 2\*\*32\)"),
        ({"seed": "1"}, "seed must be an integer"),
        ({"resume": True, "seed": 3}, "a resumed study keeps the seed"),
    ],
)
def test_unusable_arguments_are_refused_before_anything_is_created(
    tmp_path: Path, write_problem: Callable[..., Path], kwargs: dict[str, Any], message: str
) -> None:
    kwargs = {"trials": 1, **kwargs}
    with pytest.raises(OptimizationError, match=message):
        optimize(write_problem(), [trade_off()], tmp_path / "s", **kwargs)
    assert not (tmp_path / "s").exists()


def test_an_evaluator_without_a_source_hash_cannot_make_a_study(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    class Inline:  # defined here: its file was imported before anything could hash it
        name, version = "test.trade_off", "1"
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
        metrics = {("analysis/x", "y"): "1", ("analysis", "x/y"): "1"}

        def evaluate(self, request):
            x, *_ = request.candidate.values()
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
            {"stage": "analysis/x", "metric": "y", "max": 1000, "unit": "1"},
            {"stage": "analysis", "metric": "x/y", "max": 1000, "unit": "1"},
        ],
    )
    slashed = evaluator_from(tmp_path / "evaluators", "slashed", SLASHED)
    report = optimize(problem, [slashed], tmp_path / "s", trials=2, seed=1)
    assert report.objectives == ("test.slashed/analysis%2Fx/y", "test.slashed/analysis/x%2Fy")
    assert report.constraints == (
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

    monkeypatch.setattr(simulation.ddsim_run, "_execute", second_pions_crash)
    report = study(sampled_problem, tmp_path / "s", trials=3, evaluators=[cell_count()])
    assert len(fake.calls) == 6  # two samples, one simulation each, every trial
    assert [t.status for t in report.trials] == ["valid", "failed", "valid"]
    assert report.objectives == (
        "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
        "pion/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
    )
    states = [trial.state for trial in optuna_study(tmp_path / "s").trials]
    assert states == [TrialState.COMPLETE, TrialState.FAIL, TrialState.COMPLETE]
    record = summary(tmp_path / "s", 1)
    assert record["message"].startswith("pion/simulation failed")
    assert [m["value"] for m in record["metrics"]] == [1.5, None]  # gamma's is kept
    assert record["samples"]["gamma"]["evaluations"]["test.cell_count"]["status"] == "valid"
