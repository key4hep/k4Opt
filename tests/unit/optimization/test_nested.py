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
from k4opt.optimization._study import trial_directory
from k4opt.problem.candidate import replicate_seed
from tests.support import simulation, stack
from tests.support.evaluators import candidate_check, cell_count, trade_off
from tests.unit.optimization.conftest import GUN, RADIUS, optuna_study, summary

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
    told: list[Any] = []
    # no new trial: it goes on with the one it has
    study = run(nested, output, 0, resume=True, progress=told.append)
    (trial,) = study.trials
    assert trial.status == "valid" and trial.selected is not None
    assert trial.recovered and [(t.number, t.recovered) for t in told] == [(0, True)]
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


def test_a_running_trial_does_not_hide_an_inner_study_that_does_not_hold_up(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_at_settle(1, 2)(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    assert load_study(output).trials[0].inner is not None  # as it stands, it reads back
    path = output / "inner" / "trial-000000" / "trial-000000" / "summary.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(OptimizationError, match="is not the record the optimizer was told"):
        load_study(output)


def test_an_inner_study_whose_making_was_cut_short_is_no_error_and_is_made_on_resume(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"

    def stopping(*args: Any) -> Any:
        raise Stop

    with monkeypatch.context() as patched:
        patched.setattr(tree, "open_optuna", stopping)  # its study.json written, no storage
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    assert (output / "inner" / "trial-000000" / "study.json").is_file()
    read = load_study(output)
    assert read.trials[0].status == "running" and read.trials[0].inner is None
    study = run(nested, output, 0, resume=True)
    assert study.trials[0].status == "valid" and len(study.trials[0].inner.trials) == 3


def test_an_inner_study_of_another_problem_or_other_programs_is_refused(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    data = json.loads(run(nested, tmp_path / "a", 1).to_json())
    document = yaml.safe_load(nested.read_text())
    document["samples"]["gamma"]["simulation"]["events"] = 3
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump(document))
    theirs = json.loads(run(other, tmp_path / "b", 1).to_json())["trials"][0]["inner"]
    theirs["study"]["scope"]["parent"]["study"] = data["study"]["id"]  # linked, each valid alone
    edited = copy.deepcopy(data)
    edited["trials"][0]["inner"] = theirs
    with pytest.raises(OptimizationError, match="its inner study is not of the study's problem"):
        StudyResults.from_dict(edited)
    edited = copy.deepcopy(data)
    edited["trials"][0]["inner"]["study"]["programs"] = {"ddsim": None}
    with pytest.raises(OptimizationError, match="its inner study ran other programs"):
        StudyResults.from_dict(edited)


def test_a_run_keeps_only_its_open_studies(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    held: list[int] = []
    close = tree.Tree.close

    def closing(self: Any) -> None:
        held.append(len(self._nodes) + len(self._journals))
        close(self)

    monkeypatch.setattr(tree.Tree, "close", closing)
    run(nested, tmp_path / "s", 3)
    assert held == [1]  # the top study: those of its three trials were let go as they ended


@pytest.mark.parametrize("lost", ["study.json", "study.journal"])
def test_a_selection_needs_its_inner_study_though_its_trial_still_runs(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch, lost: str
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_before_the_new_events(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    assert load_study(output).trials[0].selected is not None  # as it stands, it reads back
    (output / "inner" / "trial-000000" / lost).unlink()
    with pytest.raises(OptimizationError):
        load_study(output)
    with pytest.raises(OptimizationError):
        run(nested, output, 0, resume=True)


def test_a_lost_journal_is_no_study_never_set_up_once_trials_ran(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_at_settle(1, 2)(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    (output / "inner" / "trial-000000" / "study.journal").unlink()  # its trials' records stay
    with pytest.raises(OptimizationError, match=r"study\.journal does not exist"):
        load_study(output)


def test_the_json_form_of_a_running_trial_that_selected_needs_its_inner_study(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_before_the_new_events(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    data = json.loads(load_study(output).to_json())
    assert data["trials"][0]["status"] == "running" and data["trials"][0]["selected"] is not None
    assert StudyResults.from_dict(data).to_dict() == data  # as it stands, it reads back
    data["trials"][0]["inner"] = None
    with pytest.raises(OptimizationError, match="stands for a study of the level below"):
        StudyResults.from_dict(data)


def test_a_running_trial_does_not_hide_an_inner_study_that_lost_its_study_json(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_at_settle(1, 2)(patched)
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    (output / "inner" / "trial-000000" / "study.json").unlink()  # its journal and records stay
    with pytest.raises(OptimizationError, match=r"holds a study's files, but no study\.json"):
        load_study(output)
    with pytest.raises(OptimizationError, match=r"holds a study's files, but no study\.json"):
        run(nested, output, 0, resume=True)  # not a new study over what is there


def test_an_empty_inner_directory_is_a_study_not_made_yet(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        patched.setattr(tree, "write_study", lambda *args: (_ for _ in ()).throw(Stop()))
        with pytest.raises(Stop):
            run(nested, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    assert (output / "inner" / "trial-000000").is_dir()
    assert load_study(output).trials[0].inner is None
    assert run(nested, output, 0, resume=True).trials[0].status == "valid"


@pytest.fixture
def shared(nested: Path) -> Path:
    """:func:`nested`, whose inner studies tune ``x`` of ``test.candidate_check`` (required, so
    that it runs) by the cells of the simulation, which ``x`` does not reach: all the trials of an
    inner study share it."""
    document = yaml.safe_load(nested.read_text())
    document["evaluators"][1]["required"] = True
    cells = dict(zip(("evaluator", "stage", "metric", "direction"), CELLS, strict=True))
    document["optimization"]["inner"]["objective"] = cells
    nested.write_text(yaml.safe_dump(document))
    return nested


def asked(study: Path) -> list[tuple[Any, Any]]:
    """When each trial of ``study`` was asked for and told, in number order."""
    trials = sorted(optuna_study(study).trials, key=lambda t: t.number)
    return [(t.datetime_start, t.datetime_complete) for t in trials]


def test_a_slot_goes_to_another_study_before_one_whose_trials_wait_for_what_they_share(
    tmp_path: Path, shared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch, **{"*": {"sleep": 1}})
    output = tmp_path / "s"
    study = run(shared, output, 2, workers=2)
    assert [t.status for t in study.trials] == ["valid", "valid"]
    first, second = (asked(output / "inner" / trial_directory(n)) for n in (0, 1))
    # while the first study's first trial made what its trials share, the other study began
    assert second[0][0] < first[1][0]
    # one simulation for each inner study, which its other trials reused, and one on new events
    made = [
        summary(directory, number)["samples"]["gamma"]["stages"]["simulation"]
        for directory in (
            output / "inner" / trial_directory(0),
            output / "inner" / trial_directory(1),
        )
        for number in range(3)
    ]
    assert sum("reused" not in stage for stage in made) == 2


def test_a_slot_no_other_study_would_use_goes_to_a_study_held_back(
    tmp_path: Path, shared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch, **{"*": {"sleep": 1}})
    output = tmp_path / "s"
    run(shared, output, 1, workers=2)  # one study below: nothing else to run
    (_, told), (start, _), _ = asked(output / "inner" / trial_directory(0))
    assert start < told


def test_an_inner_study_that_shares_no_stage_asks_for_its_trials_at_once(
    tmp_path: Path, nested: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = yaml.safe_load(nested.read_text())
    document["parameters"]["x"] = {  # the energy of the gun: each trial simulates its own
        "type": "continuous",
        "bounds": [5.0, 10.0],
        "unit": "GeV",
        "targets": {"generation": "energy"},
    }
    del document["samples"]["gamma"]["generation"]["energy"]
    document["evaluators"] = document["evaluators"][:1]  # x reaches no evaluator now
    cells = dict(zip(("evaluator", "stage", "metric", "direction"), CELLS, strict=True))
    document["optimization"]["inner"]["objective"] = cells
    nested.write_text(yaml.safe_dump(document))
    stack.install(monkeypatch, **{"*": {"sleep": 1}})
    output = tmp_path / "s"
    optimize(nested, [cell_count()], output, trials=1, seed=1, workers=2)
    (_, told), (start, _), _ = asked(output / "inner" / trial_directory(0))
    assert start < told  # nothing to wait for: the second ran beside the first


# --- what a resume reuses, at every level --------------------------------------------------------


@pytest.fixture
def repeated(nested: Path) -> Path:
    """:func:`nested`, whose inner studies are judged by the cells of the simulation (which ``x``
    does not reach), with a second parameter at the top, ``y``, which reaches no stage: with the
    radius fixed, every top trial simulates one geometry."""
    document = yaml.safe_load(nested.read_text())
    del document["optimization"]["inner"]["objective"]  # the problem's one objective
    document["evaluators"].append({"name": "test.trade_off", "version": "1", "as": "outer"})
    document["parameters"]["y"] = {
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "targets": {"evaluators": {"outer": "x"}},
    }
    document["optimization"]["parameters"] = [RADIUS, "y"]
    nested.write_text(yaml.safe_dump(document))
    return nested


FIXED = {RADIUS: 450.0}


def optimize_repeated(problem: Path, output: Path, trials: int, **kwargs: Any) -> StudyResults:
    implementations = [cell_count(), candidate_check(), trade_off()]
    kwargs.setdefault("seed", None if kwargs.get("resume") else 1)
    return optimize(problem, implementations, output, trials=trials, fixed=FIXED, **kwargs)


def simulation_of(run_dir: Path) -> dict[str, Any]:
    data = json.loads((run_dir / "summary.json").read_text())
    stage: dict[str, Any] = data["samples"]["gamma"]["stages"]["simulation"]
    return stage


def test_a_resume_reuses_what_the_inner_studies_of_ended_top_trials_made(
    tmp_path: Path, repeated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    optimize_repeated(repeated, output, 1)
    assert seeds(fake) == [SEED, replicate_seed(SEED, 1)]  # tuning, then the new events
    study = optimize_repeated(repeated, output, 1, resume=True)  # a second top trial
    assert [t.status for t in study.trials] == ["valid", "valid"]
    assert seeds(fake) == [SEED, replicate_seed(SEED, 1)]  # nothing simulated again
    first = output / "inner" / trial_directory(0) / trial_directory(0)
    made = json.loads((first / "summary.json").read_text())["id"]
    second = output / "inner" / trial_directory(1)
    assert all(
        simulation_of(second / trial_directory(n))["reused"]["record"] == made for n in range(3)
    )
    assert load_study(output).to_dict() == study.to_dict()


def test_a_resume_of_three_levels_reuses_what_every_level_made(
    tmp_path: Path, repeated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    three = yaml.safe_load(three_levels(repeated).read_text())
    del three["optimization"]["inner"]["inner"]["objective"]  # each level by the cells
    repeated.write_text(yaml.safe_dump(three))
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    optimize_repeated(repeated, output, 1)
    made = sorted(seeds(fake))
    assert made == sorted([SEED, replicate_seed(SEED, 1), replicate_seed(SEED, 2)])
    study = optimize_repeated(repeated, output, 1, resume=True)
    assert [t.status for t in study.trials] == ["valid", "valid"]
    assert sorted(seeds(fake)) == made  # every level of the second top trial reused
    assert load_study(output).to_dict() == study.to_dict()


@pytest.mark.parametrize("change", ["rewritten", "removed"])
def test_a_resume_simulates_again_what_an_inner_trial_made_that_changed_since(
    tmp_path: Path, repeated: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    fake = simulation.install(monkeypatch)
    output = tmp_path / "s"
    optimize_repeated(repeated, output, 1)
    first = output / "inner" / trial_directory(0) / trial_directory(0)
    events = first / simulation_of(first)["outputs"]["events"]["path"]
    if change == "rewritten":
        events.write_text(events.read_text() + " ")
    else:
        events.unlink()
    study = optimize_repeated(repeated, output, 1, resume=True)
    assert [t.status for t in study.trials] == ["valid", "valid"]
    assert seeds(fake) == [SEED, replicate_seed(SEED, 1), SEED]  # the tuning's, once more
    again = output / "inner" / trial_directory(1) / trial_directory(0)
    assert "reused" not in simulation_of(again)
    assert simulation_of(output / trial_directory(1))["reused"]["run_dir"] == str(
        output / trial_directory(0)
    )  # the new events of the first top trial still serve


def test_a_session_given_a_nested_study_reuses_the_outputs_of_its_every_level(
    tmp_path: Path, repeated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.execution import EvaluationSession

    simulation.install(monkeypatch)
    output = tmp_path / "s"
    optimize_repeated(repeated, output, 1)
    with EvaluationSession.open(repeated, [cell_count(), candidate_check(), trade_off()]) as s:
        # the inner study's simulation (its other trials reused it) and the new events
        assert s.reuse([output]) == 2


def test_a_middle_level_shares_what_only_its_own_plan_reads(
    nested: Path,
) -> None:
    """Three levels: the middle one judged by a sample the deepest does not read, whose
    simulation no parameter of the levels below the top reaches."""
    from k4opt.optimization._scope import StudyScope
    from k4opt.problem import load_problem

    document = yaml.safe_load(nested.read_text())
    pion = copy.deepcopy(document["samples"]["gamma"])
    pion["generation"]["particle"] = "pi-"
    pion["simulation"]["seed"] = SEED + 1
    document["samples"]["pion"] = pion
    del document["samples"]["gamma"]["generation"]["energy"]
    document["parameters"]["z"] = {
        "type": "continuous",
        "bounds": [5.0, 10.0],
        "unit": "GeV",
        "targets": {"generation": {"option": "energy", "samples": ["gamma"]}},
    }
    document["evaluators"][0]["samples"] = ["gamma", "pion"]
    cells = dict(zip(("evaluator", "stage", "metric", "direction"), CELLS, strict=True))
    document["objectives"] = [cells | {"sample": "gamma"}]
    document["optimization"]["inner"]["trials"] = 2
    document["optimization"]["inner"]["objective"] = cells | {"sample": "pion"}
    document["optimization"]["inner"]["inner"] = {
        "parameters": ["z"],
        "trials": 2,
        "objective": cells | {"sample": "gamma"},
    }
    nested.write_text(yaml.safe_dump(document))
    spec = load_problem(nested)
    # the deepest level simulates gamma, which z reaches: its trials share nothing
    assert not tree._shares(StudyScope.of(spec, {RADIUS: 450.0, "x": 1.5}, level=2))
    # the middle level's own candidates simulate pion, which neither x nor z reaches
    assert tree._shares(StudyScope.of(spec, {RADIUS: 450.0}, level=1))


def test_a_resumed_inner_study_that_shares_a_stage_is_held_back_again(
    tmp_path: Path, shared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    output = tmp_path / "s"
    with monkeypatch.context() as patched:
        stop_at_settle(1, 1)(patched)  # its first trial's record written
        with pytest.raises(Stop):
            run(shared, output, 1)
    (output / ".lock").unlink(missing_ok=True)
    made, held = tree.Tree._node, []

    def watched(self: tree.Tree, run: Any, *args: Any, **kwargs: Any) -> Any:
        node = made(self, run, *args, **kwargs)
        held.append((run.scope.level, len(run.study.trials), node.held))
        return node

    monkeypatch.setattr(tree.Tree, "_node", watched)
    study = run(shared, output, 0, resume=True)
    assert study.trials[0].status == "valid"
    by_level = {level: (trials, flag) for level, trials, flag in held}
    assert by_level[0] == (1, False)  # the top study is never held back
    trials, inner = by_level[1]
    assert trials >= 1 and inner  # though it has trials, it waits again
