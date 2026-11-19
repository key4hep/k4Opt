"""``load_study``: a study read back as its ledger and records hold it, its Pareto front, and the
records, studies and ledgers it refuses, all without changing anything."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from k4opt.evaluation import Measurement, load_evaluator
from k4opt.execution import EvaluationSession
from k4opt.optimization import OptimizationError, StudyResults, TrialResult, load_study, optimize
from k4opt.optimization._ledger import LedgerWriter
from k4opt.optimization._results import OUTCOMES, pareto_front, recorded, recorded_scope
from k4opt.optimization._study import read_study
from k4opt.problem import parse_problem
from k4opt.record import Assessment
from tests.support import simulation
from tests.support.common import tree_digest
from tests.support.evaluators import TRADE_OFF, cell_count, hit_energy, trade_off
from tests.support.simulation import COLLECTION
from tests.unit.optimization.conftest import (
    OUTCOME,
    RADIUS,
    TRADE,
    endings,
    evaluator_from,
    ledger,
    left_running,
    problem_document,
    summary,
)


def run(
    problem: Path, output: Path, trials: int, evaluators: list[Any] | None = None, **kwargs: Any
) -> Any:
    if "seed" in kwargs:
        pytest.importorskip("optuna")
    return optimize(problem, evaluators or [trade_off()], output, trials=trials, **kwargs)


def optuna_front(study: StudyResults) -> tuple[int, ...]:
    """The trials Optuna's own ``best_trials`` gives for the trials of ``study`` as their records
    hold them: what the front must be."""
    optuna = pytest.importorskip("optuna")
    from optuna.distributions import FloatDistribution
    from optuna.trial import TrialState, create_trial

    held = optuna.create_study(directions=study.scope.directions)
    for trial in study.trials:
        if trial.status == "valid":
            assert trial.assessment is not None and trial.objectives is not None
            held.add_trial(
                create_trial(
                    values=[float(v) for v in trial.objectives.values()],
                    constraints={lim.name: lim.normalized for lim in trial.assessment.limits},
                    params={"n": 0.0},
                    distributions={"n": FloatDistribution(0, 1)},
                )
            )
        else:
            held.add_trial(create_trial(state=TrialState.FAIL))
    return tuple(sorted(t.number for t in held.best_trials))


def record_bytes(output: Path, number: int) -> bytes:
    return (output / f"trial-{number:06d}" / "summary.json").read_bytes()


def test_a_study_reads_back_with_its_problem_and_every_trial_as_recorded(
    tmp_path: Path, mixed_problem: Path
) -> None:
    output = tmp_path / "s"
    report = run(mixed_problem, output, 12, seed=1)
    study = load_study(output)
    assert isinstance(study, StudyResults)
    assert (study.name, study.id, study.directory) == (report.name, report.id, output)
    (parameter,) = study.problem.parameters
    assert (parameter.name, parameter.kind) == (RADIUS, "categorical")
    assert parameter.targets[0].canonical_target == "constants.InnerTracker_Barrel_radius_1"  # type: ignore[union-attr]
    assert parameter.choices == (300, 500, 170, 120)
    assert [(o.metric, o.direction) for o in study.problem.objectives] == [
        ("a", "minimize"),
        ("b", "minimize"),
    ]
    assert [(c.metric, c.max, c.unit) for c in study.problem.constraints] == [("a", 400, "mm")]
    assert [t.number for t in study.trials] == list(range(12))
    assert set(OUTCOME.values()) <= {t.outcome for t in study.trials}  # every way, to mean much
    for trial in study.trials:
        record = summary(output, trial.number)
        assert trial.outcome == OUTCOME[trial.candidate[RADIUS]]
        assert (trial.record, trial.message) == (record["id"], record["message"])
        assert trial.sha256 == hashlib.sha256(record_bytes(output, trial.number)).hexdigest()
        assert trial.assessment.to_dict() == record["assessment"]  # as recorded, nothing made up
        if trial.status != "valid":
            assert trial.assessment.objectives is None and trial.feasible is None
            limits = [(lim.value, lim.margin, lim.satisfied) for lim in trial.assessment.limits]
            assert limits == [(None, None, None)]
    counted = Counter(OUTCOME[t.candidate[RADIUS]] for t in study.trials)
    assert dict(study.outcomes) == {outcome: counted[outcome] for outcome in OUTCOMES}
    # the front: every feasible trial, all at 300 and so tied, as Optuna has it
    feasible = tuple(t.number for t in study.trials if t.outcome == "feasible")
    assert len(feasible) > 1 and study.pareto == feasible == report.pareto
    assert study.pareto == optuna_front(study)
    assert study.trial(3) is study.trials[3]


@pytest.mark.parametrize(
    "directions",
    [("minimize", "minimize"), ("maximize", "minimize"), ("maximize", "maximize")],
)
def test_the_front_follows_every_direction_as_the_optimizer_does(
    tmp_path: Path, write_problem: Callable[..., Path], directions: tuple[str, str]
) -> None:
    problem = write_problem(
        objectives=[("analysis", "a", directions[0]), ("analysis", "b", directions[1])],
        constraints=[{"stage": "analysis", "metric": "a", "max": 450, "unit": "mm"}],
    )
    report = run(problem, tmp_path / "s", 10, seed=4)
    study = load_study(tmp_path / "s")
    assert study.pareto == report.pareto == optuna_front(study)
    assert study.pareto  # something to compare
    assert all(study.trial(n).outcome == "feasible" for n in study.pareto)


def test_one_objective_has_all_its_best_trials_on_the_front(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem(
        parameter={
            "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": [300, 500, 170],
        },
        evaluators=[TRADE | {"settings": {"invalid_below": 200}}],
        objectives=[("analysis", "a", "maximize")],
    )
    run(problem, tmp_path / "s", 10, seed=1)
    study = load_study(tmp_path / "s")
    best = tuple(t.number for t in study.trials if t.candidate[RADIUS] == 500)
    assert len(best) > 1 and study.pareto == best
    assert study.pareto == optuna_front(study)


def _trial(number: int, values: list[int], feasible: bool) -> TrialResult:
    """A valid trial with these objective values, ``analysis/m0``, ``analysis/m1``, ..."""
    measured = tuple(Measurement("analysis", f"m{i}", v, unit="1") for i, v in enumerate(values))
    return TrialResult(number, "valid", {RADIUS: 1.0}, Assessment(measured, (), feasible))


@pytest.mark.parametrize(
    "directions",
    [
        ("minimize",),
        ("maximize",),
        ("minimize", "maximize"),
        ("maximize", "maximize"),
        ("maximize", "minimize", "minimize"),
    ],
)
def test_the_front_is_every_feasible_trial_no_other_dominates(directions: tuple[str, ...]) -> None:
    objectives = [("analysis", f"m{i}", d) for i, d in enumerate(directions)]
    spec = parse_problem(problem_document(objectives=objectives))
    rng = random.Random(len(directions))
    trials = [  # few distinct values, so many ties, and some infeasible trials that would win
        _trial(n, [rng.randrange(5) for _ in directions], rng.random() > 0.2) for n in range(400)
    ]
    signs = [1 if d == "minimize" else -1 for d in directions]
    points = {
        t.number: [s * m.value for s, m in zip(signs, t.assessment.objectives, strict=True)]
        for t in trials
        if t.feasible
    }

    def dominates(a: list[int], b: list[int]) -> bool:
        return all(x <= y for x, y in zip(a, b, strict=True)) and a != b

    expected = tuple(
        n for n, p in points.items() if not any(dominates(q, p) for q in points.values())
    )
    assert len(expected) > 1  # ties stay
    assert pareto_front(trials, spec.criteria()) == expected


def test_a_large_front_takes_a_sort_and_one_pass_not_a_comparison_of_every_pair() -> None:
    objectives = [("analysis", "m0", "minimize"), ("analysis", "m1", "minimize")]
    spec = parse_problem(problem_document(objectives=objectives))
    trials = [_trial(n, [n, -n], True) for n in range(20_000)]  # none dominates another
    assert pareto_front(trials, spec.criteria()) == tuple(
        range(20_000)
    )  # all 2e8 pairs would take minutes
    tied = [_trial(n, [1, 1], True) for n in range(20_000)]
    assert pareto_front(tied, spec.criteria()) == tuple(range(20_000))


def test_values_keep_their_units_and_uncertainties_and_trials_their_recorded_times(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
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
            "direction": {"distribution": "uniform", "theta": [80, 100], "phi": [0, 360]}
            | {"unit": "deg"},
        },
        "simulation": {"executor": "ddsim", "events": 2, "seed": 7},
    }
    problem.write_text(__import__("yaml").safe_dump(document))
    run(problem, tmp_path / "s", 2, seed=1, evaluators=[cell_count(), hit_energy()])
    for trial in load_study(tmp_path / "s").trials:
        record = summary(tmp_path / "s", trial.number)
        measured = [(m.value, m.unit, m.uncertainty) for m in trial.assessment.objectives]
        assert measured == [(m["value"], m["unit"], m["uncertainty"]) for m in record["metrics"]]
        assert measured[0][:2] == (1.5, "1")  # 2 and 1 cells
        assert measured[0][2] == pytest.approx(0.5)  # the standard error of their mean
        assert trial.elapsed_seconds == record["elapsed_seconds"]
        timed = {
            f"gamma/{s}": e["elapsed_seconds"]
            for s, e in record["samples"]["gamma"]["stages"].items()
            if "elapsed_seconds" in e
        }
        assert dict(trial.stage_seconds) == timed and "gamma/simulation" in timed


def test_a_resumed_study_reads_back_with_every_session_and_its_recovered_trial(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 2, seed=1)
    left_running(output, problem)
    report = run(problem, output, 2, resume=True)
    study = load_study(output)
    assert [(t.number, t.status, t.recovered) for t in study.trials] == [
        (0, "valid", False),
        (1, "valid", False),
        (2, "interrupted", True),
        (3, "valid", False),
        (4, "valid", False),
    ]
    interrupted = study.trial(2)
    assert (interrupted.assessment, interrupted.record, interrupted.message) == (None, None, None)
    assert dict(interrupted.candidate) == ledger(output).entries[2].values
    assert [t.record for t in study.trials if t.record] == [
        summary(output, n)["id"] for n in (0, 1, 3, 4)
    ]
    assert (study.outcomes["interrupted"], sum(study.outcomes.values())) == (1, 5)
    assert study.pareto == report.pareto


class Crash(BaseException):
    """A process killed between publishing a record and keeping it in the ledger."""


def test_a_running_trial_is_shown_unchecked_and_left_running(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, output = write_problem(), tmp_path / "s"

    def crashing(*args: Any, **kwargs: Any) -> Any:
        raise Crash

    with monkeypatch.context() as patched:
        patched.setattr(LedgerWriter, "recorded", crashing)
        with pytest.raises(Crash):
            run(problem, output, 1, strategy="grid", grid={RADIUS: [300.0, 350.0]})
    assert (output / "trial-000000" / "summary.json").is_file()  # published, never kept
    # and one proposed, being evaluated
    left_running(output, problem, directory=True, values={RADIUS: 350.0}, note={"point": 1})
    before = tree_digest(output)
    study = load_study(output)
    assert [(t.status, t.record, t.assessment) for t in study.trials] == [
        ("running", None, None)
    ] * 2
    assert all(set(t.candidate) == {RADIUS} for t in study.trials)
    assert (study.outcomes["running"], study.pareto) == (2, ())
    assert tree_digest(output) == before
    assert endings(output) == [None, None]


BROKEN = """
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        settings = {"x": float}  # the candidate's value, which it sets
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            x = request.settings["x"]
            return {("analysis", "a"): x}  # not b, which it owns
"""


def test_a_trial_that_stopped_the_study_is_an_error_without_a_result(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    broken = evaluator_from(tmp_path / "evaluators", "broken", BROKEN)
    with pytest.raises(OptimizationError, match="trial 0 stops the study"):
        run(write_problem(), tmp_path / "s", 1, seed=1, evaluators=[broken])
    (trial,) = load_study(tmp_path / "s").trials
    assert (trial.status, trial.outcome, trial.record, trial.assessment) == (
        "error",
        "error",
        None,
        None,
    )
    assert trial.message == ledger(tmp_path / "s").entries[0].ending.message


def test_a_valid_record_the_ledger_could_not_keep_is_kept_on_resume(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from k4opt.optimization import _ledger

    write = os.write

    def losing(descriptor: int, data: Any) -> int:
        if b'"event":"recorded"' in bytes(data):  # a disk that fills up, say
            raise OSError(28, "No space left on device")
        return write(descriptor, data)

    problem, output = write_problem(), tmp_path / "s"
    monkeypatch.setattr(_ledger.os, "write", losing)
    match = r"trial 0 stops the study \(the ledger could not be written; .*--resume settles"
    with pytest.raises(OptimizationError, match=match):
        run(problem, output, 1, strategy="grid", grid={RADIUS: [300.0, 350.0]})
    monkeypatch.setattr(_ledger.os, "write", write)
    # nothing more was written after the write that failed: the trial is left running
    assert endings(output) == [None]
    assert [t.status for t in load_study(output).trials] == ["running"]
    trial = run(problem, output, 1, resume=True).trials[0]
    assert (trial.status, trial.recovered) == ("valid", True)
    assert trial.record == summary(output, 0)["id"]
    assert endings(output) == ["recorded", "recorded"]


def test_reading_changes_nothing_and_hands_out_nothing_mutable(
    tmp_path: Path, mixed_problem: Path
) -> None:
    output = tmp_path / "s"
    run(mixed_problem, output, 4, seed=1)
    # a session at work on the study, elsewhere
    (output / ".lock").write_text('{"host": "elsewhere", "pid": 1, "started": "now"}\n')
    before = tree_digest(output)
    study = load_study(output)
    assert load_study(output).to_dict() == study.to_dict()
    assert tree_digest(output) == before  # no lock of its own, no recovery, the lock untouched
    with pytest.raises(TypeError):
        study.document["id"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        study.document["problem"]["definition"]["objectives"][0]["direction"] = "maximize"  # type: ignore[index]
    with pytest.raises(TypeError):
        study.trials[0].candidate[RADIUS] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        study.trials[0].status = "valid"  # type: ignore[misc]
    exported = study.to_dict()
    exported["trials"][0]["candidate"][RADIUS] = -1  # type: ignore[index]
    exported["study"]["id"] = "other"  # type: ignore[index]
    again = study.to_dict()
    assert again["trials"][0]["candidate"][RADIUS] != -1  # type: ignore[index]
    assert again["study"]["id"] == study.id  # type: ignore[index]


def test_a_moved_study_reads_without_its_problem_evaluators_geometry_or_stack(
    tmp_path: Path, mixed_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "evaluators" / "trade_off.py"
    copy.parent.mkdir()
    copy.write_text(Path(TRADE_OFF.partition(":")[0]).read_text())
    run(mixed_problem, tmp_path / "s", 4, seed=1, evaluators=[load_evaluator(f"{copy}:TradeOff")])
    before = load_study(tmp_path / "s").to_dict()
    shutil.move(tmp_path / "s", tmp_path / "moved")
    mixed_problem.unlink()
    shutil.rmtree(copy.parent)
    monkeypatch.delenv("K4GEO")
    monkeypatch.delenv("KEY4HEP_STACK")
    after = load_study(tmp_path / "moved")
    assert after.directory == tmp_path / "moved"
    assert after.to_dict() == before  # the JSON form names no location


def test_the_files_of_a_moved_study_are_found_where_its_records_now_are(
    tmp_path: Path, sampled_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    run(sampled_problem, tmp_path / "s", 2, seed=1, evaluators=[cell_count()])
    before = load_study(tmp_path / "s").to_dict()
    shutil.move(tmp_path / "s", tmp_path / "moved")
    after = load_study(tmp_path / "moved")
    assert after.to_dict() == before
    for trial in after.trials:
        summary_path = after.record_path(trial)
        assert summary_path is not None
        record = json.loads(summary_path.read_text())
        assert record["run_dir"] == str(tmp_path / "s" / summary_path.parent.name)  # made there
        for sample in ("gamma", "pion"):
            stage = record["samples"][sample]["stages"]["simulation"]
            for path in (stage["outputs"]["events"]["path"], *stage["logs"].values()):
                assert (summary_path.parent / path).is_file()  # and where it is now


# --- what is refused -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("edit", r"trial 1: trial-000001/summary\.json is not the record the ledger names"),
        ("remove", r"trial 1: trial-000001/summary\.json cannot be read .*nothing shows what"),
    ],
)
def test_a_record_that_changed_or_is_gone_is_refused(
    tmp_path: Path, mixed_problem: Path, change: str, message: str
) -> None:
    run(mixed_problem, tmp_path / "s", 3, seed=1)
    path = tmp_path / "s" / "trial-000001" / "summary.json"
    if change == "remove":
        path.unlink()
    else:
        path.write_text(path.read_text() + " ")  # the same JSON, other bytes
    with pytest.raises(OptimizationError, match=message):
        load_study(tmp_path / "s")


def _told(path: Path) -> dict[str, Any]:
    """The reference the ledger would hold for the record at ``path``."""
    data = path.read_bytes()
    relative = f"{path.parent.name}/{path.name}"
    return {
        "id": json.loads(data)["id"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "path": relative,
    }


def _rewrite(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


RECORD_CHANGES: dict[str, tuple[Callable[[Path, dict[str, Any]], None], str]] = {
    "another trial's": (
        lambda path, told: path.write_bytes(record_bytes(path.parents[1], 1)),
        r"was made for \{'study': .*, 'trial': 1\}, not this trial",
    ),
    "another study's": (
        lambda path, told: _rewrite(path, lambda d: d["run"].update(study="another")),
        r"was made for \{'study': 'another', 'trial': 0\}, not this trial",
    ),
    "another level's": (
        lambda path, told: _rewrite(path, lambda d: d.update(level=1)),
        r"was evaluated for level 1 of a nested optimization, judged by its criteria",
    ),
    "another candidate's": (
        lambda path, told: _rewrite(path, lambda d: d["candidate"].update({RADIUS: 999})),
        r"holds candidate .*999.*, but the trial is",
    ),
    "unassessed": (
        lambda path, told: _rewrite(path, lambda d: d["assessment"]["limits"][0].update(margin=1)),
        "its assessment is not the one its measurements give",
    ),
    "another id": (
        lambda path, told: told.update(id="another"),
        r"has the id .*, but the ledger names record 'another'",
    ),
    "elsewhere": (
        lambda path, told: told.update(path="../elsewhere/summary.json"),
        r"the ledger names a record at '\.\./elsewhere/summary\.json'",
    ),
}


@pytest.mark.parametrize("change", sorted(RECORD_CHANGES))
def test_a_record_must_be_the_trials_own_and_hold_up(
    tmp_path: Path, mixed_problem: Path, change: str
) -> None:
    output = tmp_path / "s"
    run(mixed_problem, output, 2, seed=1)
    document = read_study(output)
    scope = recorded_scope(document)
    params = dict(ledger(output).entries[0].values)
    path = output / "trial-000000" / "summary.json"
    result, _ = recorded(output, document, scope, 0, params, _told(path), {})
    assert result.record == summary(output, 0)["id"]  # as it is, it holds up
    edit, message = RECORD_CHANGES[change]
    told: dict[str, Any] = {}
    edit(path, told)
    told = _told(path) | told  # the bytes on disk, as if the ledger named them
    with pytest.raises(OptimizationError, match=message):
        recorded(output, document, scope, 0, params, told, {})


def test_a_replicate_of_the_trials_candidate_does_not_stand_in_for_its_record(
    tmp_path: Path, sampled_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    output = tmp_path / "s"
    run(sampled_problem, output, 1, seed=1, evaluators=[cell_count()])
    document = read_study(output)
    scope = recorded_scope(document)
    params = dict(ledger(output).entries[0].values)
    # the same candidate, made for the same trial, on other events: a record that holds up
    replicated = tmp_path / "replicate"
    with EvaluationSession.open(sampled_problem, [cell_count()]) as session:
        outcome = session.evaluate(
            params, replicated, run={"study": document["id"], "trial": 0}, replicate=1
        )
        Assessment.from_summary(session.spec, dict(outcome.summary))
    shutil.rmtree(output / "trial-000000")
    shutil.copytree(replicated, output / "trial-000000")
    path = output / "trial-000000" / "summary.json"
    with pytest.raises(OptimizationError, match="is replicate 1 of its candidate"):
        recorded(output, document, scope, 0, params, _told(path), {})


STUDY_CHANGES: dict[str, tuple[Callable[[Path], None], str]] = {
    "missing": (lambda path: path.unlink(), r"is not a study directory: it holds no study\.json"),
    "another schema": (
        lambda path: _rewrite(path, lambda d: d.update(schema="k4opt.study/2")),
        r"schema 'k4opt\.study/2' is not 'k4opt\.study/1'",
    ),
    "another problem": (
        lambda path: _rewrite(
            path, lambda d: d["problem"]["definition"]["objectives"][0].update(direction="maximize")
        ),
        r"study\.json does not hold together: its identity digest is not that of its parts",
    ),
    "no strategy": (
        lambda path: _rewrite(path, lambda d: d.pop("strategy")),
        r"study\.json records no strategy of the study: None",
    ),
    "no name": (
        lambda path: _rewrite(path, lambda d: d.pop("name")),
        r"study\.json lacks the study's id, its creation time or its name",
    ),
    "another scope": (
        lambda path: _rewrite(path, lambda d: d["scope"].update(replicate=1)),
        r"study\.json records the scope .*, but its problem makes it",
    ),
    "no scope": (
        lambda path: _rewrite(path, lambda d: d.pop("scope")),
        r"study\.json records no scope of the study",
    ),
}


@pytest.mark.parametrize("change", sorted(STUDY_CHANGES))
def test_a_study_json_that_is_missing_or_does_not_hold_together_is_refused(
    tmp_path: Path, write_problem: Callable[..., Path], change: str
) -> None:
    run(write_problem(), tmp_path / "s", 1, strategy="grid", grid={RADIUS: [300.0]})
    edit, message = STUDY_CHANGES[change]
    edit(tmp_path / "s" / "study.json")
    before = tree_digest(tmp_path / "s")
    with pytest.raises(OptimizationError, match=message):
        load_study(tmp_path / "s")
    assert tree_digest(tmp_path / "s") == before


def test_a_study_without_its_ledger_is_refused_and_none_is_made(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    output = tmp_path / "s"
    run(write_problem(), output, 1, strategy="grid", grid={RADIUS: [300.0]})
    held = output / "trials.jsonl"
    held.unlink()
    with pytest.raises(OptimizationError, match=r"trials\.jsonl does not exist"):
        load_study(output)
    assert not held.exists()
    held.write_bytes(b'{"seq":0,"event":"st')  # as a process killed while it made it leaves it
    with pytest.raises(OptimizationError, match=r"holds no event: the study was never set up"):
        load_study(output)
    assert held.read_bytes() == b'{"seq":0,"event":"st'


def test_a_trial_directory_of_no_trial_is_warned_about_and_an_unknown_trial_refused(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    output = tmp_path / "s"
    run(write_problem(), output, 2, seed=1)
    (output / "trial-000099").mkdir()
    study = load_study(output)
    assert study.warnings == (
        f"trial-000099 in {output} belongs to no trial of the study, so it is not shown",
    )
    assert [t.number for t in study.trials] == [0, 1]
    with pytest.raises(OptimizationError, match=r"the study has no trial 99: it has trials 0 to 1"):
        study.trial(99)
