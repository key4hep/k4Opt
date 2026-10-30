"""``k4opt optimize``: what it prints, its exit codes, and k4opt without Optuna."""

from __future__ import annotations

import io
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from k4opt.cli import main
from k4opt.problem import load_problem
from tests.support.evaluators import TRADE_OFF
from tests.unit.optimization.conftest import optuna_study

REPOSITORY = Path(__file__).resolve().parents[3]


def optimize(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(["optimize", "--evaluator", TRADE_OFF, *argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_the_command_prints_each_trial_and_the_study(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    constraints = [{"stage": "analysis", "metric": "a", "max": 400, "unit": "mm"}]
    problem, output = write_problem(constraints=constraints), tmp_path / "s"
    code, out, err = optimize(str(problem), "--trials", "3", "--seed", "1", "--output", str(output))
    assert code == 0
    # a warning only if this checkout has uncommitted changes
    assert all("has uncommitted changes" in line for line in err.splitlines())
    lines = out.splitlines()
    assert [line.split()[:2] for line in lines[:3]] == [
        ["trial", "0:"],
        ["trial", "1:"],
        ["trial", "2:"],
    ]
    assert all("constants.InnerTracker_Barrel_radius_1=" in line for line in lines[:3])
    assert all("analysis/a=" in line and "analysis/b=" in line for line in lines[:3])
    assert lines[3].startswith("study:    trade-") and lines[3].endswith(f"{output}/study.json")
    assert "2 objective(s), 1 limit(s)" in lines[3]
    assert lines[4].startswith("trials:   3: 3 valid (")
    assert lines[5].startswith("pareto:   ")
    code, out, _ = optimize(str(problem), "--trials", "1", "--resume", "--output", str(output))
    assert code == 0 and out.splitlines()[0].startswith("trial 3:")


def test_failed_candidates_are_reported_and_the_command_succeeds(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    settings = {"failed_below": 700}
    problem = write_problem(
        evaluators=[{"name": "test.trade_off", "version": "1", "settings": settings}]
    )
    code, out, _ = optimize(
        str(problem), "--trials", "2", "--seed", "1", "--output", str(tmp_path / "s")
    )
    assert code == 0  # the budget was spent, whatever the candidates were
    first, _, *summary = out.splitlines()
    assert first.startswith("trial 0:   failed   test.trade_off failed: x = ")
    assert first.endswith("trial-000000/summary.json")
    assert summary[1] == "trials:   2: 0 valid (0 feasible, 0 infeasible), 2 failed"
    assert summary[2] == "pareto:   none feasible yet"


def test_a_resume_prints_the_trials_it_recovers_first(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    from k4opt.optimization._optuna import search_space

    problem, output = write_problem(), tmp_path / "s"
    code, _, _ = optimize(str(problem), "--trials", "1", "--seed", "1", "--output", str(output))
    assert code == 0
    # a trial asked for and left running, as a session killed just after asking leaves it
    optuna_study(output).ask(fixed_distributions=search_space(load_problem(problem)))
    code, out, _ = optimize(str(problem), "--trials", "1", "--resume", "--output", str(output))
    assert code == 0
    recovered, new, *_ = out.splitlines()
    assert recovered == "trial 1:   interrupted  trial-000001  (recovered)"
    assert new.startswith("trial 2:   valid ") and not new.endswith("(recovered)")


def test_a_study_that_cannot_start_or_go_on_exits_1(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    problem, output = write_problem(), tmp_path / "s"
    code, out, err = optimize(str(problem), "--trials", "1", "--resume", "--output", str(output))
    assert (code, out) == (1, "")
    assert err.startswith("k4opt: error: ") and "no study to resume" in err


def test_a_missing_budget_is_a_usage_error(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    with pytest.raises(SystemExit) as info:
        optimize(str(write_problem()), "--output", str(tmp_path / "s"))
    assert info.value.code == 2


def _python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPOSITORY, capture_output=True, text=True, check=False
    )


def test_the_command_line_does_not_import_optuna() -> None:
    run = _python("import sys, k4opt.cli; print('optuna' in sys.modules)")
    assert (run.returncode, run.stdout.strip()) == (0, "False"), run.stderr


def test_without_optuna_only_optimize_needs_it(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem()
    blocked = "import sys; sys.modules['optuna'] = None; from k4opt.cli import main; "
    run = _python(
        blocked
        + f"sys.exit(main(['optimize', '{problem}', '--evaluator', '{TRADE_OFF}', '--trials', '1', "
        f"'--output', '{tmp_path / 's'}']))"
    )
    assert run.returncode == 1
    assert "k4opt optimize needs Optuna, which is not installed: pip install" in run.stderr
    assert not (tmp_path / "s").exists()
    for argv in (["--help"], ["evaluate", "--help"], ["optimize", "--help"]):
        run = _python(blocked + f"sys.exit(main({argv!r}))")
        assert run.returncode == 0, run.stderr
