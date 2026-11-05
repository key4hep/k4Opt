"""Nested studies: each trial of the top study stands for a study of the level below, whose best
trial is evaluated again on new events and is the trial's record."""

from __future__ import annotations

import copy
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("optuna")

from k4opt.evaluation import edm4hep
from k4opt.optimization import OptimizationError, StudyResults, load_study, optimize
from k4opt.optimization._optuna import marks, tree
from k4opt.optimization._show import render_results
from k4opt.problem.candidate import replicate_seed
from tests.support import simulation, stack
from tests.support.evaluators import candidate_check, cell_count, trade_off
from tests.unit.optimization.conftest import GUN, RADIUS, optuna_study

SEED = 1  # the simulation's
CELLS = ("test.cell_count", "simulation", "ecal_barrel.mean_hit_cells", "minimize")
SEEN = {
    "evaluator": "test.candidate_check",
    "stage": "analysis",
    "metric": "seen",
    "direction": "minimize",
}


@pytest.fixture
def nested(write_problem: Callable[..., Path]) -> Path:
    """The radius of the geometry at the top, compared by the cells of the simulation; for each
    radius, three trials of the setting ``x`` of ``test.candidate_check``, tuned by its metric
    ``seen``, which no objective of the problem names, and which reads no stage."""
    path = write_problem(
        evaluators=[
            {
                "name": "test.cell_count",
                "version": "1",
                "stage": "simulation",
                "collections": {"hits": "ECalBarrelCollection"},
            },
            {"name": "test.candidate_check", "version": "1"},
        ],
        objectives=[CELLS],
    )
    document = yaml.safe_load(path.read_text())
    document["parameters"][RADIUS]["targets"] = {"geometry": "InnerTracker_Barrel_radius_1"}
    document["parameters"]["x"] = {
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "targets": {"evaluators": {"test.candidate_check": "x"}},
    }
    document["samples"] = {
        "gamma": {
            "generation": GUN | {"particle": "gamma"},
            "simulation": {"executor": "ddsim", "events": 2, "seed": SEED},
        }
    }
    document["optimization"] = {
        "parameters": [RADIUS],
        "inner": {"parameters": ["x"], "trials": 3, "objective": SEEN},
    }
    path.write_text(yaml.safe_dump(document))
    return path


def run(problem: Path, output: Path, trials: int, **kwargs: Any) -> Any:
    kwargs.setdefault("seed", None if kwargs.get("resume") else 1)
    return optimize(problem, [cell_count(), candidate_check()], output, trials=trials, **kwargs)


def record(output: Path, number: int) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((output / f"trial-{number:06d}" / "summary.json").read_text())
    return data


def seeds(fake: simulation.FakeDdsim) -> list[int]:
    return [
        int(argument.partition("=")[2])
        for argv, _ in fake.calls
        for argument in argv
        if argument.startswith("--random.seed=")
    ]


def test_each_top_trial_is_its_inner_studys_best_trial_on_new_events(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    study = run(nested, output, 2)
    assert [t.status for t in study.trials] == ["valid", "valid"]
    # the inner trials read no stage: only the records of the top trials simulate, on new events
    assert seeds(fake) == [replicate_seed(SEED, 1)] * 2
    for trial in study.trials:
        inner = trial.inner
        assert inner is not None and inner.scope.level == 1
        assert inner.scope.parent == {"study": study.id, "trial": trial.number}
        assert inner.scope.fixed == {RADIUS: trial.candidate[RADIUS]}
        assert len(inner.trials) == 3 and {t.status for t in inner.trials} == {"valid"}
        best = min(inner.trials, key=lambda t: t.candidate["x"])
        assert trial.selected == {
            "study": inner.id,
            "trial": best.number,
            "values": {"x": best.candidate["x"]},
        }
        assert trial.candidate == {RADIUS: trial.candidate[RADIUS], "x": best.candidate["x"]}
        made = record(output, trial.number)
        assert (made["level"], made["replicate"]) == (0, 1)
        assert made["run"] == {
            "study": study.id,
            "trial": trial.number,
            "selected": {"study": inner.id, "trial": best.number},
        }
        # the top is compared by its own objectives: the inner one stays out of its front
        assert [o.metric for o in trial.assessment.objectives] == [CELLS[2]]
        below = record(output / "inner" / f"trial-{trial.number:06d}", best.number)
        assert (below["level"], below["replicate"]) == (1, 0)
        assert below["samples"]["gamma"]["stages"] == {}  # nothing of the sample ran
    read = load_study(output)
    assert read.to_dict() == study.to_dict()


def drop_inner_trial(trial: dict[str, Any], number: int) -> None:
    """The form of ``trial`` whose inner study ran one trial less, which holds together."""
    inner = trial["inner"]
    inner["trials"] = [t for t in inner["trials"] if t["number"] != number]
    inner["outcomes"]["feasible"] -= 1


def test_the_json_form_holds_the_inner_studies_and_checks_their_links(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    study = run(nested, tmp_path / "s", 1)
    data = json.loads(study.to_json())
    assert StudyResults.from_dict(data).to_dict() == data
    other = next(
        t["number"]
        for t in data["trials"][0]["inner"]["trials"]
        if t["number"] != data["trials"][0]["selected"]["trial"]
    )
    for change, message in (
        (lambda t: t["selected"].update(trial=other), "the best trial of its inner study"),
        (lambda t: t["selected"]["values"].update(x=1.5), "selected the values"),
        (lambda t: t["candidate"].update(x=1.5), "selected trial make it"),
        (lambda t: t.update(inner=None), "stands for a study of the level below"),
        (lambda t: drop_inner_trial(t, other), "ran 2 of its 3 trials"),
    ):
        edited = copy.deepcopy(data)
        change(edited["trials"][0])
        with pytest.raises(OptimizationError, match=message):
            StudyResults.from_dict(edited)


def test_a_study_of_a_level_below_is_not_resumed_on_its_own(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    run(nested, tmp_path / "s", 1)
    with pytest.raises(OptimizationError, match="a study of a level below the top"):
        run(nested, tmp_path / "s" / "inner" / "trial-000000", 1, resume=True)


def constrained(problem: Path, **constraint: Any) -> Path:
    document = yaml.safe_load(problem.read_text())
    document["constraints"] = [constraint]
    problem.write_text(yaml.safe_dump(document))
    return problem


def test_a_top_trial_without_a_feasible_inner_trial_is_unscored_and_says_why(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    limited = constrained(
        nested, evaluator="test.candidate_check", stage="analysis", metric="seen", max=0.5
    )
    study = run(limited, tmp_path / "s", 2)
    assert [t.status for t in study.trials] == ["unscored", "unscored"]
    assert study.trials[0].message == "no feasible trial among 3: 3 infeasible"
    assert study.trials[0].record is None and study.trials[0].assessment is None
    assert study.pareto == () and study.outcomes["unscored"] == 2
    assert fake.calls == [] and not (tmp_path / "s" / "trial-000000").exists()
    assert load_study(tmp_path / "s").to_dict() == study.to_dict()
    assert StudyResults.from_dict(json.loads(study.to_json())).to_dict() == study.to_dict()


def test_a_selection_infeasible_on_new_events_makes_its_top_trial_infeasible(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)

    def read(path: Path) -> list[Any]:  # few cells on the tuning events, many on new ones
        tuning = seeds(fake)[-1] == SEED
        return simulation.events_with(*([[1], [2]] if tuning else [[1, 2, 3], [4, 5, 6]]))

    monkeypatch.setattr(edm4hep, "_frames", read)
    limited = constrained(
        nested,
        evaluator="test.cell_count",
        sample="gamma",
        stage="simulation",
        metric="ecal_barrel.mean_hit_cells",
        max=2,
    )
    study = run(limited, tmp_path / "s", 1)
    (trial,) = study.trials
    assert {t.outcome for t in trial.inner.trials} == {"feasible"}
    assert trial.status == "valid" and trial.outcome == "infeasible"
    assert study.pareto == ()


def three_levels(nested: Path) -> Path:
    """The problem with a third level below the second: the setting of ``test.trade_off``."""
    document = yaml.safe_load(nested.read_text())
    document["parameters"]["z"] = {
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "targets": {"evaluators": {"test.trade_off": "x"}},
    }
    document["evaluators"].append({"name": "test.trade_off", "version": "1"})
    document["optimization"]["inner"]["trials"] = 2
    document["optimization"]["inner"]["inner"] = {
        "parameters": ["z"],
        "trials": 2,
        "objective": {
            "evaluator": "test.trade_off",
            "stage": "analysis",
            "metric": "a",
            "direction": "minimize",
        },
    }
    document["objectives"][0]["sample"] = "gamma"
    nested.write_text(yaml.safe_dump(document))
    return nested


def test_three_levels_each_select_on_events_the_level_below_did_not_tune_on(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    implementations = [cell_count(), candidate_check(), trade_off()]
    study = optimize(three_levels(nested), implementations, output, trials=1, seed=1)
    (top,) = study.trials
    assert top.status == "valid" and set(top.candidate) == {RADIUS, "x", "z"}
    middle = top.inner
    assert [t.status for t in middle.trials] == ["valid", "valid"]
    assert record(output, 0)["replicate"] == 2
    assert record(output / "inner" / "trial-000000", 0)["replicate"] == 1
    bottom = middle.trials[0].inner
    assert bottom.scope.level == 2 and len(bottom.trials) == 2
    assert bottom.scope.fixed == {
        RADIUS: top.candidate[RADIUS],
        "x": middle.trials[0].candidate["x"],
    }
    assert record(output / "inner" / "trial-000000" / "inner" / "trial-000000", 0)["replicate"] == 0
    assert load_study(output).to_dict() == study.to_dict()


def test_with_workers_the_inner_studies_run_side_by_side(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch)
    study = optimize(
        nested, [cell_count(), candidate_check()], tmp_path / "s", trials=3, seed=1, workers=2
    )
    assert [t.status for t in study.trials] == ["valid"] * 3
    assert all(len(t.inner.trials) == 3 for t in study.trials)
    assert load_study(tmp_path / "s").to_dict() == study.to_dict()


def test_stages_are_shared_within_an_inner_study_and_across_a_resume(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the inner level judged by the simulation too: its trials share one simulation of the
    # geometry, and the new events are one more
    document = yaml.safe_load(nested.read_text())
    del document["optimization"]["inner"]["objective"]  # the problem's one objective
    nested.write_text(yaml.safe_dump(document))
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_at_settle(1, 2)(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    assert seeds(fake) == [SEED]
    study = run(nested, output, 0, resume=True)
    assert study.trials[0].status == "valid"
    assert seeds(fake) == [SEED, replicate_seed(SEED, 1)]  # none of the tuning simulated again


def test_study_show_lists_the_inner_studies(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    limited = run(nested, tmp_path / "s", 1)
    out = io.StringIO()
    render_results(limited, out)
    best = limited.trials[0].selected["trial"]
    assert f"trial 0:  best trial {best}, " in out.getvalue()
    assert "inner/trial-000000" in out.getvalue()


def test_a_shared_database_holds_the_studies_of_the_tree(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sqlalchemy")
    simulation.install(monkeypatch)
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    study = run(nested, tmp_path / "s", 2, storage=url)
    assert [t.status for t in study.trials] == ["valid", "valid"]
    assert not list((tmp_path / "s").rglob("study.journal"))
    assert load_study(tmp_path / "s", storage=url).to_dict() == study.to_dict()


# --- a run stopped anywhere goes on from where it was ------------------------------------------


class Stop(BaseException):
    """What stops the process at a chosen point, as a kill would: nothing is cleaned up."""


def stop_at_settle(level: int, nth: int = 1) -> Callable[[pytest.MonkeyPatch], None]:
    """Stop when a trial of ``level`` ended, before it is told: its record is written."""

    def install(monkeypatch: pytest.MonkeyPatch) -> None:
        settle, seen = tree._settle, []

        def stopping(run: Any, *args: Any) -> Any:
            if run.scope.level == level:
                seen.append(1)
                if len(seen) == nth:
                    raise Stop
            return settle(run, *args)

        monkeypatch.setattr(tree, "_settle", stopping)

    return install


def stop_in(name: str) -> Callable[[pytest.MonkeyPatch], None]:
    """Stop when ``Tree`` is about to do ``name``."""

    def install(monkeypatch: pytest.MonkeyPatch) -> None:
        def stopping(*args: Any, **kwargs: Any) -> Any:
            raise Stop

        monkeypatch.setattr(tree.Tree, name, stopping)

    return install


def stop_before_the_new_events(monkeypatch: pytest.MonkeyPatch) -> None:
    made = tree.RunRequest

    def stopping(candidate: Any, output: Any, run: Any, *args: Any) -> Any:
        if "selected" in run:
            raise Stop
        return made(candidate, output, run, *args)

    monkeypatch.setattr(tree, "RunRequest", stopping)


@pytest.mark.parametrize(
    "stop",
    [
        pytest.param(stop_in("_open"), id="asked, no inner study yet"),
        pytest.param(stop_at_settle(1, 2), id="inner study under way"),
        pytest.param(stop_in("_score"), id="inner study done, nothing selected"),
        pytest.param(stop_before_the_new_events, id="selected, not evaluated"),
        pytest.param(stop_at_settle(0), id="record written, top trial not told"),
    ],
)
def test_a_run_stopped_anywhere_goes_on_from_where_it_was(
    tmp_path: Path,
    nested: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: Callable[[pytest.MonkeyPatch], None],
) -> None:
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)  # a kill leaves it
    study = run(nested, output, 0, resume=True)  # no new trial: it goes on with the one it has
    (trial,) = study.trials
    assert trial.status == "valid" and trial.selected is not None
    assert len([t for t in trial.inner.trials if t.status == "valid"]) == 3
    assert len(fake.calls) == 1  # the new events, once
    assert load_study(output).to_dict() == study.to_dict()
    again = run(nested, output, 0, resume=True)  # nothing is left to do, and nothing changes
    assert again.to_dict() == study.to_dict() and len(fake.calls) == 1


def test_what_an_interrupted_evaluation_left_is_kept_aside(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_before_the_new_events(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    (output / "trial-000000").mkdir()  # as a kill during its evaluation leaves it
    (output / "trial-000000" / "half").write_text("written")
    study = run(nested, output, 0, resume=True)
    assert study.trials[0].status == "valid"
    (aside,) = (output / "inner").glob("interrupted-trial-000000-*")
    assert (aside / "half").read_text() == "written"


def test_a_selection_that_later_differs_stops_the_run(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_before_the_new_events(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    trial = optuna_study(output).trials[0]
    selected = dict(trial.user_attrs[marks.SELECTED]) | {"trial": 99}
    storage = optuna_study(output)._storage
    storage.set_trial_user_attr(trial._trial_id, marks.SELECTED, selected)
    with pytest.raises(OptimizationError, match="but the best trial of its inner study"):
        run(nested, output, 0, resume=True)


def test_a_trial_stopped_while_it_was_proposed_takes_no_part_of_a_studys_budget(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    problem, output = three_levels(nested), tmp_path / "s"
    implementations = [cell_count(), candidate_check(), trade_off()]
    with monkeypatch.context() as patched:
        stop_at_settle(1)(patched)  # the first middle trial has its record, not yet told
        with pytest.raises(Stop):
            optimize(problem, implementations, output, trials=1, seed=1)
    (output / ".lock").unlink(missing_ok=True)
    optuna_study(output / "inner" / "trial-000000").ask()  # a proposal cut short: no parameters
    study = optimize(problem, implementations, output, trials=0, resume=True)
    middle = study.trials[0].inner
    assert [t.status for t in middle.trials] == ["valid", "interrupted", "valid"]
    assert study.trials[0].status == "valid"
    assert load_study(output).to_dict() == study.to_dict()


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda path: path.unlink(), id="record gone"),
        pytest.param(lambda path: path.write_text(path.read_text() + " "), id="record changed"),
    ],
)
def test_a_resume_refuses_a_tree_that_does_not_hold_up_before_anything_new_runs(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch, corrupt: Callable[[Path], object]
) -> None:
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    run(nested, output, 1)
    corrupt(output / "inner" / "trial-000000" / "trial-000001" / "summary.json")
    calls = len(fake.calls)
    with pytest.raises(OptimizationError, match="trial 1"):
        run(nested, output, 1, resume=True)
    assert len(optuna_study(output).trials) == 1 and len(fake.calls) == calls
    assert not (output / "inner" / "trial-000001").exists()
