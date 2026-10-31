"""``k4opt optimize`` and ``k4opt study``: what they print, their exit codes, and k4opt without
Optuna."""

from __future__ import annotations

import io
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from k4opt.cli import main
from k4opt.problem import load_problem
from tests.support.common import tree_digest
from tests.support.evaluators import TRADE_OFF
from tests.unit.optimization.conftest import RADIUS, evaluator_from, optuna_study

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


def test_the_command_refuses_a_timeout_that_is_no_time_limit(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    output = tmp_path / "s"
    code, _, err = optimize(
        str(write_problem()), "--trials", "1", "--timeout", "-1", "--output", str(output)
    )
    assert code == 1 and "timeout must be a positive number of seconds, got -1.0" in err
    assert not output.exists()


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
    from k4opt.optimization._optuna.space import search_space

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


def test_without_optuna_only_optimize_and_study_need_it(
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
    run = _python(blocked + f"sys.exit(main(['study', 'show', '{tmp_path}']))")
    assert run.returncode == 1
    assert "reading a study needs Optuna, which is not installed: pip install" in run.stderr
    for argv in (
        ["--help"],
        ["evaluate", "--help"],
        ["optimize", "--help"],
        ["study", "--help"],
        ["study", "export", "--help"],
    ):
        run = _python(blocked + f"sys.exit(main({argv!r}))")
        assert run.returncode == 0, run.stderr


# --- k4opt study -----------------------------------------------------------------------------


def study(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(["study", *argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


ODD = """
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            x, *_ = request.candidate.values()
            if x < 300:
                self.invalid("\\x1b[31mred\\x1b[0m from a log")
            return {("analysis", "a"): x, ("analysis", "b"): 1 / x}
"""


def test_study_show_trial_and_export_read_a_study_and_change_nothing(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    from k4opt.optimization import optimize as run

    constraints = [{"stage": "analysis", "metric": "a", "max": 400, "unit": "mm"}]
    problem, output = write_problem(constraints=constraints), tmp_path / "s"
    odd = evaluator_from(tmp_path / "evaluators", "odd", ODD)
    run(problem, [odd], output, trials=6, seed=1)
    # a session at work on the study, elsewhere: reading needs no lock and takes none
    (output / ".lock").write_text('{"host": "elsewhere", "pid": 1, "started": "now"}\n')
    before = tree_digest(output)

    code, out, err = study("show", str(output))
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0].startswith("study:      trade-")
    assert lines[1].startswith("id:         ")
    assert lines[2] == f"directory:  {output} (trials in study.journal)"
    assert lines[3] == f"parameters: {RADIUS}  continuous [100.0, 600.0] mm"
    assert lines[4:6] == [
        "objectives: analysis/a  minimize  (mm)",
        "            analysis/b  minimize  (1/mm)",
    ]
    assert lines[6] == "limits:     analysis/a:max  <= 400 mm"
    assert lines[7].startswith("trials:     6: ") and "invalid" in lines[7]
    assert lines[8].startswith("pareto:     ")
    assert all(line.startswith("            trial ") for line in lines[9:])
    assert "\x1b" not in out  # not a terminal: no color

    code, out, err = study("trial", str(output), "0")
    assert (code, err) == (0, "")
    assert out.startswith("trial:      0: ")
    assert f"candidate:  {RADIUS} = " in out and "record:     " in out
    invalid = next(n for n in range(6) if '"status": "invalid"' in _record(output, n))
    code, out, _ = study("trial", str(output), str(invalid))
    assert "message:    test.trade_off invalid: \\x1b[31mred\\x1b[0m from a log" in out
    assert "\x1b" not in out  # a record's escape sequence is shown, not run

    code, out, err = study("export", str(output), "--format", "csv")
    assert (code, err) == (0, "")
    assert (
        len(out.splitlines()) == 7 and out.startswith("study,trial,status,") and "\x1b" not in out
    )
    code, out, err = study(
        "export", str(output), "--format", "json", "--output", str(tmp_path / "s.json")
    )
    assert (code, out, err) == (0, "", "")
    assert (tmp_path / "s.json").read_text().startswith('{\n  "format": "k4opt.study-results/1"')
    assert tree_digest(output) == before


def _record(output: Path, number: int) -> str:
    return (output / f"trial-{number:06d}" / "summary.json").read_text()


def test_a_warning_shows_what_the_study_holds_escaped(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    output = tmp_path / "s"
    code, _, _ = optimize(
        str(write_problem()), "--trials", "1", "--seed", "1", "--output", str(output)
    )
    assert code == 0
    moved = tmp_path / "study\x1b[2J"  # a directory name that would clear the screen
    output.rename(moved)
    (moved / "trial-000099").mkdir()  # warned about, with the study's path
    code, _, err = study("show", str(moved))
    assert code == 0 and err.startswith("k4opt: warning: trial-000099 in ")
    assert "\x1b" not in err and "study\\x1b[2J" in err


def test_study_commands_say_what_they_cannot_read(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    pytest.importorskip("optuna")
    output = tmp_path / "s"
    code, _, _ = optimize(
        str(write_problem()), "--trials", "2", "--seed", "1", "--output", str(output)
    )
    assert code == 0
    code, out, err = study("trial", str(output), "7")
    assert (code, out) == (1, "")
    assert err == "k4opt: error: the study has no trial 7: it has trials 0 to 1\n"
    code, out, err = study("show", str(tmp_path / "nothing"))
    assert (code, out) == (1, "") and "is not a study directory" in err
    code, out, err = study(
        "export", str(output), "--format", "csv", "--output", str(output / "x.csv")
    )
    assert (code, out) == (1, "") and "is in the study directory" in err
    (output / "trial-000099").mkdir()
    code, out, err = study("show", str(output))
    assert code == 0 and err.startswith("k4opt: warning: trial-000099 in ")
    with pytest.raises(SystemExit) as info:
        study("export", str(output))  # no --format
    assert info.value.code == 2
