"""What a study says it will do before it runs, and ``k4opt problem explain``, which says only
that: the levels as loops, what picks and what judges, and at most what runs."""

from __future__ import annotations

import io
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.problem import ProblemSpec, parse_problem
from k4opt.problem.candidate import replicate_seed
from k4opt.problem.outline import outline, render_outline
from tests.support.evaluators import EXAMPLES, TRADE_OFF
from tests.support.geometry import MINI_GEO
from tests.unit.optimization.conftest import problem_document

NESTED = EXAMPLE = EXAMPLES / "nested" / "cld-pi0-nested.yaml"
ORDINARY = EXAMPLES / "problem" / "cld-ecal-pi0-reco.yaml"


def example(
    path: Path = NESTED, change: Callable[[dict[str, Any]], None] | None = None
) -> ProblemSpec:
    document = yaml.safe_load(path.read_text())
    if change is not None:
        change(document)
    return parse_problem(document)


def test_the_levels_read_as_loops_with_what_picks_and_what_judges() -> None:
    lines, warnings = outline(example(), 8)
    assert lines == [
        "For each ECAL cell size (8 trials):",
        "  tune Pandora ECAL hit threshold: 12 trials maximizing π⁰ photons resolved",
        f"  evaluate the best again on new events (seed 42 → {replicate_seed(42, 1)})",
        "compare them by π⁰ photons resolved (maximize) and Readout channels (minimize)",
        "at most, if every stage succeeds: simulation 16 · reconstruction 104 · pfo.pi0 104 · "
        "readout 8",
    ]
    assert warnings == []


def test_each_level_below_is_a_loop_inside_the_one_above_with_events_of_its_own() -> None:
    def three_levels(document: dict[str, Any]) -> None:
        document["parameters"]["hcal_mip_threshold"] = {
            "title": "HCAL hit threshold",
            "type": "continuous",
            "bounds": [0.1, 1.0],
            "targets": {"reconstruction": "Pandora.Parameters[HCalMipThreshold]"},
        }
        level = document["optimization"]["inner"]
        level["trials"] = 2
        level["inner"] = {
            "parameters": ["hcal_mip_threshold"],
            "trials": 3,
            "objective": "pi0.resolved",
        }

    lines, _ = outline(example(change=three_levels), 4)
    assert lines[:6] == [
        "For each ECAL cell size (4 trials):",
        "  for each Pandora ECAL hit threshold (2 trials maximizing π⁰ photons resolved):",
        "    tune HCAL hit threshold: 3 trials maximizing π⁰ photons resolved",
        f"    evaluate the best again on new events (seed 42 → {replicate_seed(42, 1)})",
        f"  evaluate the best again on new events (seed 42 → {replicate_seed(42, 2)})",
        "compare them by π⁰ photons resolved (maximize) and Readout channels (minimize)",
    ]


def test_an_inner_objective_is_called_by_its_own_title() -> None:
    def titled(document: dict[str, Any]) -> None:
        top = document["objectives"][0]
        keys = ("evaluator", "stage", "metric", "direction")
        objective = {key: top[key] for key in keys} | {"title": "Tuned π⁰ photons"}
        document["optimization"]["inner"]["objective"] = objective

    lines, _ = outline(example(change=titled), 8)
    assert lines[1].endswith("12 trials maximizing Tuned π⁰ photons")
    assert lines[3].startswith("compare them by π⁰ photons resolved (maximize)")


def test_an_ordinary_study_is_one_line_and_the_cost_of_its_trials() -> None:
    lines, _ = outline(example(ORDINARY), 1)
    assert lines == [
        "Search ECAL cell size: 1 trial by π⁰ photons resolved (maximize) and Readout channels "
        "(minimize)",
        "at most, if every stage succeeds: simulation 1 · reconstruction 1 · pfo.pi0 1 · readout 1",
    ]


def test_a_fixed_parameter_is_said_and_not_counted() -> None:
    def tuned(document: dict[str, Any]) -> None:
        document["optimization"] = {"parameters": ["ecal_cell_size", "ecal_mip_threshold"]}

    lines, _ = outline(example(change=tuned), 6, {"ecal_cell_size": 5.1})
    assert lines == [
        "Search Pandora ECAL hit threshold, with ECAL cell size = 5.1: 6 trials by π⁰ photons "
        "resolved (maximize) and Readout channels (minimize)",
        "at most, if every stage succeeds: simulation 1 · reconstruction 6 · pfo.pi0 6 · readout 6",
    ]


def test_limits_are_said() -> None:
    def limited(document: dict[str, Any]) -> None:
        document["constraints"] = [
            {
                "evaluator": "readout",
                "stage": "geometry",
                "metric": "ecal_barrel.channels",
                "title": "Channels",
                "max": 1e8,
                "unit": "1",
                "scale": 1e6,
            }
        ]

    lines, _ = outline(example(change=limited), 2)
    assert lines[3].endswith("(minimize), within the limits on Channels")


def test_an_inner_level_that_reaches_the_simulation_is_warned_of() -> None:
    def swapped(document: dict[str, Any]) -> None:
        document["optimization"]["parameters"] = ["ecal_mip_threshold"]
        document["optimization"]["inner"]["parameters"] = ["ecal_cell_size"]

    _, warnings = outline(example(change=swapped), 2)
    assert warnings == [
        "ECAL cell size of an inner level reach the simulation: each of its trials simulates "
        "anew instead of sharing one"
    ]


def test_what_a_problem_says_is_escaped() -> None:
    def hostile(document: dict[str, Any]) -> None:
        document["parameters"]["ecal_cell_size"]["title"] = "cell"
        document["objectives"][1]["title"] = "cost\x1b[2J"

    out = io.StringIO()
    render_outline(example(change=hostile), 2, out)
    assert "\x1b" not in out.getvalue() and "cost\\x1b[2J" in out.getvalue()


def command(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(problem_document()), encoding="utf-8")
    return str(path)


def test_explain_needs_the_problem_and_the_budget_and_makes_nothing(
    tmp_path: Path, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "optuna", None)  # it does not need the optimizer
    monkeypatch.chdir(tmp_path)
    code, out, err = command("problem", "explain", problem, "--trials", "3")
    assert (code, err) == (0, "")
    assert out.startswith("Search ") and "3 trials" in out and "at most, " in out
    assert [path.name for path in tmp_path.iterdir()] == ["problem.yaml"]


def test_explain_needs_no_key4hep(problem: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K4GEO")
    code, out, err = command("problem", "explain", str(EXAMPLE), "--trials", "8")
    assert (code, err) == (0, "")
    assert out.startswith("For each ECAL cell size (8 trials):")


def test_explain_says_what_a_study_with_fixed_parameters_does(tmp_path: Path) -> None:
    document = yaml.safe_load(NESTED.read_text())
    document["optimization"] = {"parameters": ["ecal_cell_size", "ecal_mip_threshold"]}
    path = tmp_path / "one-level.yaml"
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    argv = ("problem", "explain", str(path), "--trials", "6", "--fix", "ecal_cell_size=5.1")
    code, out, err = command(*argv)
    assert (code, err) == (0, "")
    assert "with ECAL cell size = 5.1: 6 trials" in out and "simulation 1 " in out


@pytest.mark.parametrize(
    ("fix", "says"),
    [
        ("ecal_cell_size=5.1", "nothing to search"),
        ("ecal_mip_threshold=1.0", "is searched by a level below this one"),
        ("ecal_cell_size=99", "ecal_cell_size"),
        ("nothing=1", "is no parameter of the problem"),
    ],
)
def test_explain_refuses_what_a_study_would(fix: str, says: str) -> None:
    code, out, err = command("problem", "explain", str(NESTED), "--trials", "6", "--fix", fix)
    assert (code, out) == (1, "")
    assert says in err


@pytest.mark.parametrize("trials", ["0", "-1"])
def test_a_budget_no_study_takes_is_a_usage_error(
    problem: str, trials: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        command("problem", "explain", problem, f"--trials={trials}")
    assert info.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_a_problem_that_does_not_load_exits_1(tmp_path: Path) -> None:
    code, out, err = command("problem", "explain", str(tmp_path / "no.yaml"), "--trials", "3")
    assert (code, out) == (1, "")
    assert "no.yaml" in err


def test_a_new_study_prints_its_outline_and_a_resume_does_not(tmp_path: Path, problem: str) -> None:
    pytest.importorskip("optuna")
    output = str(tmp_path / "s")
    run = ("optimize", problem, "--evaluator", TRADE_OFF, "--output", output)
    code, first, err = command(*run, "--workers", "inline", "--trials", "1", "--seed", "1")
    assert code == 0, err
    assert first.startswith("Search ")
    code, resumed, err = command(*run, "--workers", "inline", "--trials", "1", "--resume")
    assert code == 0, err
    assert "Search " not in resumed


@pytest.mark.parametrize(
    ("argv", "says"),
    [("--trials=0", "trials must be a positive"), ("--fix=radius=300", "nothing to search")],
)
def test_a_study_it_would_refuse_is_refused_before_it_says_anything(
    tmp_path: Path, problem: str, argv: str, says: str
) -> None:
    pytest.importorskip("optuna")
    trials = () if argv.startswith("--trials") else ("--trials", "1")
    run = ("optimize", problem, "--evaluator", TRADE_OFF, "--output", str(tmp_path / "s"))
    code, out, err = command(*run, argv, *trials)
    assert (code, out) == (1, "")
    assert says in err
