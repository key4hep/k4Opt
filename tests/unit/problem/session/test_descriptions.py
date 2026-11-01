"""The work of a task, described by plain data apart from where it runs."""

from __future__ import annotations

import dataclasses
import os
import pickle
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from k4opt.problem import (
    SUMMARY_NAME,
    EvaluationSession,
    RunError,
    RunRequest,
)
from k4opt.problem._files import Tree, stat_of
from k4opt.problem._work import TaskSpec, asked
from tests.support import simulation
from tests.support.evaluators import (
    cell_count,
    toy_significance,
    yields,
)
from tests.support.geometry import MINI_COMPACT
from tests.unit.problem.session.conftest import (
    RADIUS,
    combined,
)

# --- the work of a task is described by plain data, apart from where it runs ---


def described(monkeypatch: pytest.MonkeyPatch) -> list[tuple[TaskSpec, Any, Any]]:
    """Each task's description, what it left behind and the stats it took, as the coordinator
    hands them to its backend and gets them back."""
    from k4opt.problem import _backends

    handed: list[tuple[TaskSpec, Any, Any]] = []
    submit = _backends.InlineBackend.submit

    def watched(self: Any, spec: TaskSpec, seen: Any) -> Any:
        handle = submit(self, spec, seen)
        _, outcome, observed = self._ended[-1]
        handed.append((spec, outcome, observed))
        return handle

    monkeypatch.setattr(_backends.InlineBackend, "submit", watched)
    return handed


def paths(value: Any) -> list[Path]:
    """Every path in ``value``, a dataclass, mapping or sequence of them, all through."""
    if isinstance(value, Path):
        return [value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [p for f in dataclasses.fields(value) for p in paths(getattr(value, f.name))]
    if isinstance(value, Mapping):
        return [p for item in value.items() for p in paths(item)]
    if isinstance(value, (list, tuple)):
        return [p for item in value for p in paths(item)]
    return []


def test_a_task_crosses_to_where_it_runs_as_plain_data_named_in_its_workspace(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    combined(problem_file)
    handed = described(monkeypatch)
    session = EvaluationSession.open(problem_file, [toy_significance(), yields()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == "valid"
    assert [spec.task.id for spec, _, _ in handed] == [t.id for t in outcome.candidate.plan.tasks]
    run_dir = tmp_path / "a"
    for spec, made, _ in handed:
        assert pickle.loads(pickle.dumps(spec)) == spec
        assert pickle.loads(pickle.dumps(made)) == made
        assert spec.workspace == outcome.summary["id"]
        # a file of the workspace is named in it, so the description holds wherever it is
        for path in paths(spec) + paths(made):
            assert not path.is_absolute() or not path.is_relative_to(run_dir), path
    (significance,) = [spec for spec, _, _ in handed if spec.task.id.label == "significance"]
    for root in (tmp_path / "here", tmp_path / "there"):
        request = asked(significance, root)
        assert request.inputs["geometry"] == root / "geometry" / Path(MINI_COMPACT).name or (
            request.inputs["geometry"].is_relative_to(root)
        )
        for items in request.roles.values():
            for item in items:
                assert all(a.path.is_relative_to(root) for a in item.artifacts.values())


def test_a_task_run_again_in_a_copied_workspace_leaves_the_same_portable_outcome(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.problem._work import Observations, execute_task

    combined(problem_file)
    handed = described(monkeypatch)
    session = EvaluationSession.open(problem_file, [toy_significance(), yields()])
    session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    spec, made, observed = next(h for h in handed if h[0].task.id.label == "zz/test.yields")
    copy_dir = tmp_path / "copy"
    shutil.copytree(tmp_path / "a", copy_dir)
    shutil.rmtree(copy_dir / spec.task.directory)  # its own directory is made again
    # the stats are those of this filesystem's copies, taken anew: never compared across copies
    seen = Observations(
        {a.ref: stat_of(copy_dir / a.path) for a in spec.watched},
        Tree.of(copy_dir / "geometry").stats,
    )
    again, observed_again = execute_task(session._context, spec, copy_dir, seen, yields()())
    untimed = {"elapsed_seconds": None, "call_seconds": None, "memory": None}
    # its status, measurements, files by reference, bytes and hashes; never its times or the
    # peak memory of this process, which depends on what it held before
    assert dataclasses.replace(again, **untimed) == dataclasses.replace(made, **untimed)
    assert observed_again.made.keys() == observed.made.keys()


def test_a_record_reads_back_without_any_optimizer(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    combined(problem_file)
    session = EvaluationSession.open(problem_file, [toy_significance(), yields()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    observed = outcome.summary["environment"]["observed"]  # type: ignore[index]
    assert observed["execution"] == {"backend": "local", "jobs": 1}
    script = (
        "import json, sys\n"
        "sys.modules['optuna'] = None  # importing it fails\n"
        "from k4opt.problem import Assessment, parse_problem\n"
        "summary = json.loads(open(sys.argv[1]).read())\n"
        "spec = parse_problem(summary['problem']['definition'])\n"
        "print(Assessment.from_summary(spec, summary).feasible)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(outcome.summary_path)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(outcome.assessment.feasible)


def test_candidates_are_evaluated_one_after_another_each_given_back_before_the_next_is_taken(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    taken: list[str] = []

    def requests() -> Iterator[RunRequest]:
        for name, radius in (("a", 300.0), ("b", 700.0), ("c", 400.0)):
            taken.append(name)
            yield RunRequest({RADIUS: radius}, tmp_path / name, {"name": name})

    seen = []
    with session.evaluate_each(requests()) as evaluations:
        for request, outcome in evaluations:
            seen.append((request.run, list(taken), type(outcome).__name__))
    assert seen == [
        ({"name": "a"}, ["a"], "RunOutcome"),
        # outside the bounds: refused before anything was created, the next one is still taken
        ({"name": "b"}, ["a", "b"], "CandidateError"),
        ({"name": "c"}, ["a", "b", "c"], "RunOutcome"),
    ]
    assert not (tmp_path / "b").exists()
    assert [f.name for f in tmp_path.iterdir() if f.name in "abc"] != []


def test_evaluations_stopped_early_are_closed_and_the_session_evaluates_again(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    requests = (RunRequest({RADIUS: 300.0 + i}, tmp_path / f"r{i}") for i in range(5))
    evaluations = session.evaluate_each(requests)
    with pytest.raises(RunError, match="evaluating already"):
        session.evaluate({RADIUS: 300.0}, tmp_path / "other")
    next(evaluations)
    evaluations.close()  # kept, not consumed further: closing it is what ends it
    with pytest.raises(StopIteration):
        next(evaluations)
    assert sorted(f.name for f in tmp_path.iterdir() if f.name.startswith("r")) == ["r0"]
    with session:
        session.evaluate({RADIUS: 300.0}, tmp_path / "after")  # the session evaluates again
    assert (tmp_path / "after" / SUMMARY_NAME).is_file()


@pytest.mark.parametrize("how", ["close", "with", "twice"])
def test_evaluations_closed_before_their_first_candidate_give_the_session_back(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, how: str
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    requests = [RunRequest({RADIUS: 300.0}, tmp_path / "never")]
    if how == "with":
        with session.evaluate_each(requests):
            pass
    else:
        evaluations = session.evaluate_each(requests)
        evaluations.close()  # it never started: nothing ran, and nothing holds the session
        if how == "twice":
            evaluations.close()
            with pytest.raises(StopIteration):
                next(evaluations)
    assert not (tmp_path / "never").exists()
    session.evaluate({RADIUS: 300.0}, tmp_path / "after")
    assert (tmp_path / "after" / SUMMARY_NAME).is_file()


def test_evaluations_that_ran_out_give_the_session_back(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    evaluations = session.evaluate_each([RunRequest({RADIUS: 300.0}, tmp_path / "a")])
    assert [type(outcome).__name__ for _, outcome in evaluations] == ["RunOutcome"]
    session.evaluate({RADIUS: 300.0}, tmp_path / "b")  # not closed, but nothing is left of it


@pytest.mark.parametrize("item", [None, {RADIUS: 300.0}])
def test_an_item_that_is_no_request_is_refused_and_ends_nothing_silently(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, item: Any
) -> None:
    session = EvaluationSession.open(problem_file, [cell_count()])
    first = RunRequest({RADIUS: 300.0}, tmp_path / "a")
    after = RunRequest({RADIUS: 300.0}, tmp_path / "b")
    given = []

    def consume() -> None:
        with session.evaluate_each([first, item, after]) as evaluations:
            for request, _ in evaluations:
                given.append(request)

    with pytest.raises(RunError, match="expected a RunRequest"):
        consume()
    assert given == [first] and not (tmp_path / "b").exists()
    session.evaluate({RADIUS: 300.0}, tmp_path / "c")  # the session is free again


def test_the_collections_of_earlier_candidates_files_are_not_kept_for_ever(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.problem import _work

    monkeypatch.setattr(_work, "_LISTED", 2)
    session = EvaluationSession.open(problem_file, [cell_count()])
    for name in "abcd":
        assert session.evaluate({RADIUS: 300.0}, tmp_path / name).candidate.status == "valid"
    assert [path.parent.parent.parent.parent.name for path, _ in session._context.listed] == [
        "c",
        "d",
    ]
