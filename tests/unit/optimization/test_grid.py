"""The grid strategy: every combination of the values given for the searched parameters, one
after another in a fixed order, whatever the order in which trials end; it needs nothing of
Optuna, to run a study or to read it back."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.optimization import OptimizationError, load_study, optimize
from k4opt.optimization._grid import LIMIT, Grid, grid_values
from k4opt.optimization._scope import StudyScope
from k4opt.problem import parse_problem
from tests.support.evaluators import TRADE_OFF, trade_off
from tests.unit.optimization.conftest import RADIUS, ledger, left_running, problem_document

REPOSITORY = Path(__file__).resolve().parents[3]


def two_parameters(**world: Any) -> dict[str, Any]:
    """The trade-off problem with a second parameter, ``world``, which the geometry reads."""
    document = problem_document()
    declared = {"targets": {"geometry": "world_side"}, "type": "integer", "bounds": [1, 3]}
    document["parameters"]["world"] = {
        key: value for key, value in (declared | world).items() if value is not None
    }
    return document


def test_the_grid_takes_the_values_given_and_every_choice_or_integer_by_default() -> None:
    spec = parse_problem(two_parameters())
    values = grid_values(StudyScope.of(spec), {RADIUS: [300, 150.5]})
    assert values == {RADIUS: [300.0, 150.5], "world": [1, 2, 3]}  # the given order is kept
    assert [type(v) for v in values[RADIUS]] == [float, float]  # a continuous one's floats
    choices = parse_problem(two_parameters(type="categorical", choices=["b", "a"], bounds=None))
    assert grid_values(StudyScope.of(choices), {RADIUS: [200.0]})["world"] == ["b", "a"]
    numeric = parse_problem(two_parameters(type="categorical", choices=[2, 1.5], bounds=None))
    # a number given for a numeric choice is that choice, as declared: 2, not 2.0
    taken = grid_values(StudyScope.of(numeric), {RADIUS: [200.0], "world": [2.0]})["world"]
    assert taken == [2] and type(taken[0]) is int


@pytest.mark.parametrize(
    ("given", "fixed", "message"),
    [
        ({}, {}, "radius is continuous, so the grid needs the values it takes, as radius=V1"),
        ({RADIUS: [700.0]}, {}, r"700\.0 is outside the bounds \[100\.0, 600\.0\]"),
        ({RADIUS: ["far"]}, {}, "expected a number"),
        ({RADIUS: [300.0], "world": [1.5]}, {}, "expected an integer"),
        ({RADIUS: [300.0, 300]}, {}, "takes 300 more than once"),
        ({RADIUS: "300"}, {}, "needs a list of one value or more"),
        ({RADIUS: [300.0], "other": [1]}, {}, "other is no parameter the study searches"),
        ({RADIUS: [300.0], "world": [1]}, {"world": 1}, "world is fixed, so the grid does not"),
    ],
)
def test_values_that_are_no_grid_of_the_study_are_refused(
    given: dict[str, Any], fixed: dict[str, Any], message: str
) -> None:
    spec = parse_problem(two_parameters())
    with pytest.raises(OptimizationError, match=message):
        grid_values(StudyScope.of(spec, fixed), given)


def test_a_grid_too_large_to_evaluate_is_refused_before_anything_is_made(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    spec = parse_problem(two_parameters(bounds=[0, 10**6]))
    with pytest.raises(OptimizationError, match=r"world has 1000001 integers in \[0, 1000000\]"):
        grid_values(StudyScope.of(spec), {RADIUS: [300.0]})
    wide = [100.0 + n * 0.004 for n in range(LIMIT // 2)]
    with pytest.raises(OptimizationError, match=r"has 15000 points \(radius 5000 by world 3\)"):
        grid_values(StudyScope.of(parse_problem(two_parameters())), {RADIUS: wide})
    problem = tmp_path / "p.yaml"
    problem.write_text(yaml.safe_dump(two_parameters()))
    with pytest.raises(OptimizationError, match="more than 10000"):
        optimize(
            problem, [trade_off()], tmp_path / "s", trials=1, strategy="grid", grid={RADIUS: wide}
        )
    assert not (tmp_path / "s").exists()


def test_the_points_are_in_problem_order_the_last_parameter_varying_fastest(
    tmp_path: Path,
) -> None:
    problem = tmp_path / "p.yaml"
    problem.write_text(yaml.safe_dump(two_parameters(bounds=[1, 2])))
    grid = {RADIUS: [200.0, 300.0, 400.0]}
    report = optimize(problem, [trade_off()], tmp_path / "s", trials=10, strategy="grid", grid=grid)
    assert [(t.candidate[RADIUS], t.candidate["world"]) for t in report.trials] == [
        (200.0, 1),
        (200.0, 2),
        (300.0, 1),
        (300.0, 2),
        (400.0, 1),
        (400.0, 2),
    ]
    assert report.exhausted and report.to_dict()["exhausted"] is True  # asked for a seventh
    document = json.loads((tmp_path / "s" / "study.json").read_text())
    assert document["strategy"] == {
        "name": "grid",
        "values": {RADIUS: [200.0, 300.0, 400.0], "world": [1, 2]},
        "points": 6,
    }
    assert "seed" not in json.dumps(document["strategy"])  # nothing of it is drawn at random


def test_a_resumed_grid_goes_on_from_its_next_point_and_ends_when_exhausted(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    grid = {RADIUS: [150.0, 250.0, 350.0, 450.0, 550.0]}
    first = optimize(problem, [trade_off()], output, trials=2, strategy="grid", grid=grid)
    assert not first.exhausted
    # a session killed just after it proposed the third point
    left_running(output, problem, values={RADIUS: 350.0}, note={"point": 2})
    second = optimize(problem, [trade_off()], output, trials=1, resume=True)
    assert [(t.candidate[RADIUS], t.status, t.recovered) for t in second.trials] == [
        (150.0, "valid", False),
        (250.0, "valid", False),
        (350.0, "interrupted", True),  # not proposed again: its point is spent
        (450.0, "valid", False),
    ]
    third = optimize(problem, [trade_off()], output, trials=5, resume=True)
    assert [t.candidate[RADIUS] for t in third.trials] == grid[RADIUS]  # each once, in order
    assert third.exhausted and load_study(output).exhausted
    fourth = optimize(problem, [trade_off()], output, trials=1, resume=True)
    assert fourth.to_dict() == third.to_dict()  # nothing is left to propose


def test_a_grid_does_not_go_on_from_trials_that_are_not_its_points(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    grid = {RADIUS: [150.0, 250.0, 350.0]}
    optimize(problem, [trade_off()], output, trials=1, strategy="grid", grid=grid)
    left_running(output, problem, values={RADIUS: 300.0}, note={"point": 1})
    with pytest.raises(OptimizationError, match=r"trial 1 of the study is not point 1 of its grid"):
        optimize(problem, [trade_off()], output, trials=1, resume=True)


def test_a_grid_whose_record_was_changed_does_not_resume(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    optimize(problem, [trade_off()], output, trials=1, strategy="grid", grid={RADIUS: [150.0]})
    path = output / "study.json"
    document = json.loads(path.read_text())
    document["strategy"]["points"] = 7
    path.write_text(json.dumps(document))
    with pytest.raises(OptimizationError, match="is not the one its values make for the study"):
        optimize(problem, [trade_off()], output, trials=1, resume=True)


def test_a_grid_with_fixed_values_searches_the_rest(tmp_path: Path) -> None:
    problem = tmp_path / "p.yaml"
    problem.write_text(yaml.safe_dump(two_parameters()))
    report = optimize(
        problem,
        [trade_off()],
        tmp_path / "s",
        trials=5,
        strategy="grid",
        grid={"world": [3, 1]},
        fixed={RADIUS: 250.0},
    )
    assert [dict(t.candidate) for t in report.trials] == [
        {RADIUS: 250.0, "world": 3},
        {RADIUS: 250.0, "world": 1},
    ]
    assert [dict(e.values) for e in ledger(tmp_path / "s").entries] == [{"world": 3}, {"world": 1}]


def test_the_grid_proposes_in_order_whatever_the_order_trials_end_in() -> None:
    spec = parse_problem(two_parameters(bounds=[1, 2]))
    scope = StudyScope.of(spec)
    recorded = {"name": "grid", "values": {RADIUS: [200.0, 300.0], "world": [1, 2]}, "points": 4}
    grid = Grid(scope, recorded, [])
    proposals = [grid.propose(n) for n in range(4)]
    for number in (3, 0, 2):  # ended in another order: it learns nothing
        grid.tell(number, None)  # type: ignore[arg-type]
    assert [p.values for p in proposals] == [  # type: ignore[union-attr]
        {RADIUS: 200.0, "world": 1},
        {RADIUS: 200.0, "world": 2},
        {RADIUS: 300.0, "world": 1},
        {RADIUS: 300.0, "world": 2},
    ]
    from k4opt.optimization._proposal import EXHAUSTED

    assert grid.propose(4) is EXHAUSTED and grid.propose(5) is EXHAUSTED


def _python(code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """``code`` in a new process that cannot import Optuna, with the test environment."""
    from tests.support.geometry import MINI_GEO
    from tests.support.software import VIEW

    blocked = "import sys; sys.modules['optuna'] = None  # importing it fails\n"
    return subprocess.run(
        [sys.executable, "-c", blocked + code],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ
        | {
            "K4GEO": str(MINI_GEO),
            "KEY4HEP_STACK": str(VIEW),
            "PYTHONPATH": os.pathsep.join([str(REPOSITORY), os.environ.get("PYTHONPATH", "")]),
        },
    )


def test_a_grid_study_runs_resumes_and_reads_back_without_optuna(
    tmp_path: Path, mixed_problem: Path
) -> None:
    output = tmp_path / "s"
    code = f"""
from k4opt.cli import main
from k4opt.optimization import load_study
argv = ["optimize", {str(mixed_problem)!r}, "--evaluator", {TRADE_OFF!r}, "--trials", "2",
        "--strategy", "grid", "--output", {str(output)!r}]
assert main(argv) == 0
assert main(["optimize", {str(mixed_problem)!r}, "--evaluator", {TRADE_OFF!r}, "--trials", "3",
             "--resume", "--output", {str(output)!r}]) == 0
study = load_study({str(output)!r})
assert [t.outcome for t in study.trials] == ["feasible", "infeasible", "invalid", "failed"]
assert study.exhausted
for action in (["show"], ["trial", "1"], ["export", "--format", "csv"]):
    assert main(["study", action[0], {str(output)!r}, *action[1:]]) == 0
assert not [m for m, held in sys.modules.items() if m.startswith("optuna") and held]
"""
    run = _python(code, tmp_path)
    assert run.returncode == 0, run.stderr
    assert "strategy:   grid of 4 points, exhausted" in run.stdout
