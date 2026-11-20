"""Optuna's TPE proposes what a TPE sampler driven by the study's ledger proposes, event by event in
the order they were written: within a session an ask at each proposal and a tell at each ending, in
the order trials end when several run at once; each session a sampler seeded anew from the study's
seed and its trials so far, holding each of them as it ended. Whatever proposes a study's
candidates, a study of Optuna's keeps proposing what it does today."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from k4opt.optimization import load_study, optimize
from k4opt.problem import load_problem
from tests.support.evaluators import trade_off
from tests.support.stack import alive, eventually
from tests.unit.optimization.conftest import evaluator_from

optuna = pytest.importorskip("optuna")

REPOSITORY = Path(__file__).resolve().parents[3]

A_MAX = {"stage": "analysis", "metric": "a", "max": 400, "unit": "mm"}
"""A limit some candidates keep and others do not: TPE is told every constraint value."""

PROBE = """
    import json
    import os
    import time
    from pathlib import Path
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        settings = {"x": float}  # the candidate's value, which it sets
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            told = json.loads(os.environ["PYTEST_K4OPT_PROBE"])
            run = request.workdir.parents[1].name  # <trial>/evaluators/test.trade_off
            marks = Path(told["marks"])
            (marks / f"{run}.started").write_text(str(os.getpid()))
            waiting = told.get("await", {}).get(run)
            deadline = time.monotonic() + 60
            while waiting is not None and not Path(waiting).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{run} still waits")
                time.sleep(0.01)
            if run in told.get("invalid", []):
                self.invalid(f"{run} is invalid")
            x = request.settings["x"]
            return {("analysis", "a"): x, ("analysis", "b"): 1 / x}
"""
"""``test.trade_off`` that marks its start, waits for a file as the environment says (``await``,
by trial directory) and is invalid for the trials it names (``invalid``)."""


def events(output: Path) -> list[dict[str, Any]]:
    """The events of a study's ledger, in the order they were written."""
    lines = (output / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def proposals(output: Path) -> dict[int, dict[str, Any]]:
    """The values the ledger holds proposed for each trial, by number."""
    return {e["trial"]: e["values"] for e in events(output) if e["event"] == "proposed"}


def ended(output: Path) -> dict[int, int]:
    """Where in the ledger each trial ended (its ending's ``seq``), by number."""
    endings = ("recorded", "rejected", "unscored", "interrupted", "error")
    return {e["trial"]: e["seq"] for e in events(output) if e["event"] in endings}


def driven(problem: Path, output: Path, seed: int) -> dict[int, dict[str, Any]]:
    """What a TPE sampler proposes for each trial of the study in ``output``, by number, driven by
    its ledger: a session starts a sampler seeded from ``seed`` and the trials so far, holding each
    as it ended (COMPLETE with its objective values and constraint values, FAIL without values);
    then each proposal of the session is an ask, each ending a tell, in ledger order."""
    from optuna.samplers import TPESampler
    from optuna.trial import TrialState, create_trial

    from k4opt.optimization._optuna import CONSTANT_LIAR, STARTUP_TRIALS, search_space
    from k4opt.optimization._strategy import session_seed

    spec = load_problem(problem)
    space = search_space(spec)
    directions = [objective.direction for objective in spec.objectives]
    trials = {trial.number: trial for trial in load_study(output).trials}

    def outcome(number: int) -> tuple[list[float] | None, dict[str, float]]:
        trial = trials[number]
        if trial.status != "valid":
            return None, {}
        assert trial.assessment is not None and trial.assessment.objectives is not None
        values = [float(objective.value) for objective in trial.assessment.objectives]
        limits = {limit.name: limit.normalized for limit in trial.assessment.limits}
        assert all(value is not None for value in limits.values())
        return values, limits  # type: ignore[return-value]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    proposed: dict[int, dict[str, Any]] = {}
    running: dict[int, Any] = {}
    study: Any = None
    for event in events(output):
        kind = event["event"]
        if kind == "session":
            sampler = TPESampler(
                seed=session_seed(seed, len(proposed)),
                n_startup_trials=STARTUP_TRIALS,
                constant_liar=CONSTANT_LIAR,
            )
            study = optuna.create_study(directions=directions, sampler=sampler)
            running = {}
            for number, params in sorted(proposed.items()):
                values, limits = outcome(number)
                if values is None:
                    frozen = create_trial(state=TrialState.FAIL, params=params, distributions=space)
                else:
                    frozen = create_trial(
                        params=params, distributions=space, values=values, constraints=limits
                    )
                study.add_trial(frozen)
        elif kind == "proposed":
            trial = study.ask(fixed_distributions=space)
            assert trial.number == event["trial"]
            proposed[trial.number] = trial.params
            running[trial.number] = trial
        elif kind != "selected" and event.get("trial") in running:
            trial = running.pop(event["trial"])
            values, limits = outcome(trial.number)
            if values is None:
                study.tell(trial, state=TrialState.FAIL)
            else:
                for key, value in limits.items():
                    trial.set_constraint(key, value)
                study.tell(trial, values)
    return proposed


def test_tpe_proposes_as_its_ledger_drives_it_one_trial_after_another(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    settings = {"invalid_below": 250.0}  # some trials without values: FAIL for TPE
    problem = write_problem(
        evaluators=[{"name": "test.trade_off", "version": "1", "settings": settings}],
        constraints=[A_MAX],
    )
    output, seed = tmp_path / "s", 17
    optimize(problem, [trade_off()], output, trials=12, seed=seed)
    report = optimize(problem, [trade_off()], output, trials=4, resume=True)
    assert proposals(output) == driven(problem, output, seed)
    # every kind of outcome, so that the comparison means much
    assert {t.outcome for t in report.trials} == {"feasible", "infeasible", "invalid"}


def test_tpe_proposes_as_its_ledger_drives_it_with_trials_ending_out_of_order_and_killed(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    marks = tmp_path / "marks"
    marks.mkdir()
    problem = write_problem(constraints=[A_MAX])
    probe = evaluator_from(tmp_path / "evaluators", "probe", PROBE)
    never = str(marks / "never")
    told = {
        "marks": str(marks),
        # trial 0 ends only once trial 2 started: trial 1 ended first and freed its slot
        "await": {
            "trial-000000": str(marks / "trial-000002.started"),
            "trial-000012": never,
            "trial-000013": never,
        },
        "invalid": ["trial-000004"],
    }
    monkeypatch.setenv("PYTEST_K4OPT_PROBE", json.dumps(told))
    output, seed = tmp_path / "s", 5
    reference = f"{tmp_path / 'evaluators' / 'probe.py'}:Probe"
    command = [
        *(sys.executable, "-m", "k4opt", "optimize", str(problem), "--evaluator", reference),
        *("--output", str(output), "--trials", "14", "--seed", str(seed), "--workers", "2"),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # the last two run, held, in both slots: every trial before them ended
        last = (marks / "trial-000012.started", marks / "trial-000013.started")
        assert eventually(lambda: all(path.exists() for path in last), timeout=120)
        process.send_signal(signal.SIGKILL)  # a batch in the middle of its trials
        process.communicate(timeout=60)
    finally:
        process.kill()
    workers = [int(path.read_text()) for path in marks.glob("*.started")]
    assert all(eventually(lambda pid=pid: not alive(pid)) for pid in workers)
    (output / ".lock").unlink()  # nothing works on the study any more
    monkeypatch.setenv("PYTEST_K4OPT_PROBE", json.dumps({"marks": str(marks)}))
    report = optimize(problem, [probe], output, trials=4, resume=True, workers=2)
    assert proposals(output) == driven(problem, output, seed)
    at = ended(output)
    assert at[1] < at[0]  # told in the order they ended, not by number
    statuses = {t.number: (t.status, t.recovered) for t in report.trials}
    assert statuses[4] == ("invalid", False)
    assert statuses[12] == statuses[13] == ("interrupted", True)
    assert len(report.trials) == 18
    assert sum(t.outcome == "feasible" for t in report.trials) > 10  # TPE past its random start
