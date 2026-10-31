"""The tasks of a candidate: planned from the problem alone, run after what they read, blocked
when something they read did not succeed (no Key4hep)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from k4opt.evaluation import EvaluationContractError, Status
from k4opt.problem import EvaluatorSpec, Sample, Stage
from k4opt.problem._tasks import (
    GEOMETRY,
    Kind,
    Made,
    Plan,
    Task,
    TaskId,
    TaskResult,
    Tree,
    blocked,
    candidate_message,
    candidate_status,
    changed_files,
    evaluator_task,
    plan_tasks,
    run_tasks,
    stage_task,
)

GUN = Stage("generation", executor="gun", options={"particle": "gamma"})
DDSIM = Stage("simulation", executor="ddsim", events=3, seed=1)
GENERATION, SIMULATION = stage_task("generation", "gamma"), stage_task("simulation", "gamma")
GAMMA = {"gamma": Sample("gamma", {"generation": GUN, "simulation": DDSIM})}


def ev(binding: str, stage: str | None = None, *samples: str) -> EvaluatorSpec:
    """An evaluator ``binding`` that reads ``stage`` of ``samples`` (``gamma`` by default), or
    measures the candidate without a stage."""
    return EvaluatorSpec(
        binding, "1", stage=stage, samples=samples or (("gamma",) if stage else ())
    )


def labels(plan: Plan) -> dict[str, list[str]]:
    """Each task of ``plan`` by label, in plan order, with the labels of what it reads."""
    return {task.id.label: [need.label for need in task.needs] for task in plan.tasks}


def test_a_simulation_reader_reads_the_geometry_and_every_stage_of_its_chain() -> None:
    plan = plan_tasks(GAMMA, [ev("calo", "simulation")])
    assert labels(plan) == {
        "geometry": [],
        "gamma/generation": [],  # a gun is fired by ddsim: it reads no geometry
        "gamma/simulation": ["geometry", "gamma/generation"],
        "gamma/calo": ["geometry", "gamma/generation", "gamma/simulation"],
    }
    directories = {task.id.label: task.directory for task in plan.tasks}
    assert directories == {
        "geometry": "geometry",
        "gamma/generation": None,  # writes nothing
        "gamma/simulation": "samples/gamma/simulation",
        "gamma/calo": "samples/gamma/evaluators/calo",
    }


def test_every_evaluator_reads_the_geometry_even_one_that_reads_no_stage() -> None:
    plan = plan_tasks({}, [ev("cost"), ev("echo")])
    assert labels(plan) == {"geometry": [], "cost": ["geometry"], "echo": ["geometry"]}
    assert all(GEOMETRY in task.needs for task in plan.of(Kind.EVALUATOR))
    assert plan[evaluator_task("cost")].directory == "evaluators/cost"


def test_each_stage_is_planned_once_however_many_evaluators_read_it() -> None:
    evaluators = [ev("a", "simulation"), ev("b", "generation"), ev("c", "simulation")]
    plan = plan_tasks(GAMMA, evaluators)
    assert list(labels(plan)) == [
        "geometry",
        "gamma/generation",
        "gamma/simulation",
        "gamma/a",
        "gamma/b",
        "gamma/c",
    ]
    assert labels(plan)["gamma/b"] == ["geometry", "gamma/generation"]


def test_each_sample_is_planned_on_its_own_and_reads_only_its_own_stages() -> None:
    samples = GAMMA | {"pion": Sample("pion", {"generation": GUN, "simulation": DDSIM})}
    evaluators = [
        ev("calo", "simulation", "gamma", "pion"),
        ev("cost"),
        ev("edge", "generation", "pion"),
    ]
    plan = plan_tasks(samples, evaluators)
    assert labels(plan) == {
        "geometry": [],
        "gamma/generation": [],
        "gamma/simulation": ["geometry", "gamma/generation"],
        "gamma/calo": ["geometry", "gamma/generation", "gamma/simulation"],
        "pion/generation": [],
        "pion/simulation": ["geometry", "pion/generation"],
        "pion/calo": ["geometry", "pion/generation", "pion/simulation"],
        "pion/edge": ["geometry", "pion/generation"],
        "cost": ["geometry"],  # the candidate's after the samples'
    }
    assert plan[evaluator_task("calo", "pion")].directory == "samples/pion/evaluators/calo"
    # a sample that only a generation reader measures does not simulate
    only = plan_tasks(
        samples, [ev("calo", "simulation", "gamma"), ev("edge", "generation", "pion")]
    )
    assert stage_task("simulation", "pion") not in [task.id for task in only.tasks]


def test_a_stage_given_as_a_file_reads_nothing() -> None:
    hepmc = Stage("generation", file="zh.hepmc")
    sample = {"gamma": Sample("gamma", {"generation": hepmc, "simulation": DDSIM})}
    plan = plan_tasks(sample, [ev("calo", "simulation")])
    assert plan[GENERATION] == Task(GENERATION, given="file")
    assert labels(plan)["gamma/simulation"] == ["geometry", "gamma/generation"]


def test_a_supplied_stage_reads_nothing_and_the_stages_only_it_needs_are_not_planned() -> None:
    samples = GAMMA | {"pion": Sample("pion", {"generation": GUN, "simulation": DDSIM})}
    calo = [ev("calo", "simulation", "gamma", "pion")]
    plan = plan_tasks(samples, calo, {SIMULATION})
    assert labels(plan) == {
        "geometry": [],
        "gamma/simulation": [],
        "gamma/calo": ["geometry", "gamma/simulation"],
        # supplied for gamma only
        "pion/generation": [],
        "pion/simulation": ["geometry", "pion/generation"],
        "pion/calo": ["geometry", "pion/generation", "pion/simulation"],
    }
    assert plan[SIMULATION].given == "supplied"
    # supplied wins over a file the problem gives
    filed = {"gamma": Sample("gamma", {"simulation": Stage("simulation", file="events.root")})}
    assert plan_tasks(filed, [ev("calo", "simulation")], {SIMULATION})[SIMULATION].given == (
        "supplied"
    )


def test_the_task_kind_and_sample_tell_tasks_of_the_same_name_apart() -> None:
    assert stage_task("x", "gamma") != evaluator_task("x", "gamma") != TaskId(Kind.GEOMETRY, "x")
    assert evaluator_task("x", "gamma") != evaluator_task("x", "pion") != evaluator_task("x")
    assert (evaluator_task("x", "gamma").label, evaluator_task("x").label) == ("gamma/x", "x")
    with pytest.raises(KeyError):
        plan_tasks({}, [ev("calo")])[stage_task("calo", "gamma")]


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        ((Task(GEOMETRY), Task(GEOMETRY)), "task 'geometry' is planned twice"),
        (
            (Task(evaluator_task("calo"), (SIMULATION,)), Task(SIMULATION)),
            "task 'calo' reads gamma/simulation, which is not planned before it",
        ),
        ((Task(SIMULATION, (SIMULATION,)),), "task 'gamma/simulation' reads gamma/simulation"),
    ],
)
def test_a_plan_runs_each_task_once_and_after_what_it_reads(
    tasks: tuple[Task, ...], message: str
) -> None:
    with pytest.raises(EvaluationContractError, match=message):
        Plan(tasks)


# --- blocking ---


def done(task: TaskId, status: str = "done", message: str | None = None) -> TaskResult:
    return TaskResult(task, status, message)


def test_a_task_runs_only_when_everything_it_reads_succeeded() -> None:
    calo = Task(evaluator_task("calo"), (GEOMETRY, SIMULATION))
    ready = {GEOMETRY: done(GEOMETRY), SIMULATION: done(SIMULATION)}
    assert blocked(calo, ready) is None
    held = blocked(calo, ready | {SIMULATION: done(SIMULATION, "failed", "ddsim crashed")})
    assert held == TaskResult(
        calo.id, Status.BLOCKED, "not run: gamma/simulation failed", blocked_by=(SIMULATION,)
    )
    # an evaluator is read as succeeded only when it is valid (combined evaluators, later)
    rival = evaluator_task("rival")
    reader = Task(evaluator_task("combined"), (rival,))
    assert blocked(reader, {rival: done(rival, "valid")}) is None
    assert blocked(reader, {rival: done(rival, "invalid", "too few events")}) is not None


def test_a_blocked_task_names_where_its_candidate_went_wrong_not_the_blocked_in_between() -> None:
    results = {GEOMETRY: done(GEOMETRY), GENERATION: done(GENERATION, "failed", "unreadable")}
    simulation = Task(SIMULATION, (GEOMETRY, GENERATION))
    results[SIMULATION] = blocked(simulation, results)  # type: ignore[assignment]
    calo = blocked(Task(evaluator_task("calo"), (GEOMETRY, SIMULATION)), results)
    assert calo is not None
    assert (calo.blocked_by, calo.message) == ((GENERATION,), "not run: gamma/generation failed")


def test_the_executor_runs_each_task_once_in_plan_order_with_only_what_it_reads() -> None:
    plan = plan_tasks(GAMMA, [ev("a", "simulation"), ev("b")])
    seen: list[tuple[str, list[str]]] = []

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        seen.append((task.id.label, [need.label for need in reads]))
        return done(task.id, "valid" if task.id.kind is Kind.EVALUATOR else "done")

    results = run_tasks(plan, run)
    assert seen == [
        ("geometry", []),
        ("gamma/generation", []),
        ("gamma/simulation", ["geometry", "gamma/generation"]),
        ("gamma/a", ["geometry", "gamma/generation", "gamma/simulation"]),
        ("b", ["geometry"]),
    ]
    assert list(results) == [task.id for task in plan.tasks]


def test_the_executor_does_not_run_a_blocked_task() -> None:
    plan = plan_tasks(GAMMA, [ev("a", "simulation"), ev("b")])
    ran: list[str] = []

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        ran.append(task.id.label)
        if task.id == SIMULATION:
            return done(task.id, "failed", "ddsim crashed")
        return done(task.id, "valid" if task.id.kind is Kind.EVALUATOR else "done")

    results = run_tasks(plan, run)
    assert ran == ["geometry", "gamma/generation", "gamma/simulation", "b"]
    assert results[evaluator_task("a", "gamma")].blocked_by == (SIMULATION,)
    assert results[evaluator_task("b")].status == "valid"


# --- the candidate's status and message, from its tasks ---


@pytest.mark.parametrize(
    ("statuses", "status"),
    [
        (["done", "valid"], Status.VALID),
        (["done", "valid", "invalid"], Status.INVALID),
        (["failed", "blocked", "invalid"], Status.FAILED),
        (["done", "blocked"], Status.FAILED),  # never alone, but incomplete all the same
        ([], Status.VALID),
    ],
)
def test_a_candidate_is_failed_before_invalid_before_valid(
    statuses: list[str], status: Status
) -> None:
    assert candidate_status(statuses) is status


def test_the_message_names_the_causes_then_what_they_blocked() -> None:
    simulation = done(SIMULATION, "failed", "ddsim crashed")
    results = [
        done(GEOMETRY),
        simulation,
        TaskResult(evaluator_task("calo"), Status.BLOCKED, "x", blocked_by=(SIMULATION,)),
        done(evaluator_task("echo"), "invalid", "too far out"),
        TaskResult(evaluator_task("energy"), Status.BLOCKED, "x", blocked_by=(SIMULATION,)),
        done(evaluator_task("cells"), "valid", "fine"),
    ]
    assert candidate_message(results) == (
        "gamma/simulation failed: ddsim crashed; echo invalid: too far out; blocked: calo, energy"
    )
    assert candidate_message([done(GEOMETRY), done(evaluator_task("a"), "valid")]) is None


# --- the files a task made, and whether they are still what its record says ---


def test_a_made_file_is_named_by_its_task_and_hashed_only_if_asked(tmp_path: Path) -> None:
    path = tmp_path / "events.root"
    path.write_bytes(b"events")
    unhashed = Made.file(SIMULATION, "events", path, hashed=False)
    assert unhashed.artifact.ref == "gamma/simulation#events"
    assert (unhashed.artifact.path, unhashed.artifact.bytes) == (path, 6)
    assert unhashed.artifact.sha256 is None
    hashed = Made.file(evaluator_task("plot", "gamma"), "histogram", path, hashed=True)
    assert hashed.artifact.ref == "gamma/plot#histogram"
    assert hashed.artifact.sha256 is not None and len(hashed.artifact.sha256) == 64
    with pytest.raises(OSError, match="is no file"):
        Made.file(SIMULATION, "events", tmp_path, hashed=False)


def test_a_file_replaced_while_it_is_hashed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.problem import _tasks

    path = tmp_path / "histogram.json"
    path.write_bytes(b"old")
    identity = _tasks.file_identity

    def replaced(file: Path) -> dict[str, object]:
        file.write_bytes(b"new, and longer")  # between the stat and the hash
        return identity(file)

    monkeypatch.setattr(_tasks, "file_identity", replaced)
    # the size of one version with the hash of another, which changed() would then trust
    with pytest.raises(OSError, match="changed while it was hashed"):
        Made.file(evaluator_task("plot", "gamma"), "histogram", path, hashed=True)


@pytest.mark.parametrize("hashed", [False, True])
def test_a_made_file_is_changed_once_its_stat_or_its_bytes_are(
    tmp_path: Path, hashed: bool
) -> None:
    path = tmp_path / "events.root"
    path.write_bytes(b"events")
    made = Made.file(SIMULATION, "events", path, hashed=hashed)
    assert not made.changed()
    os.utime(path, ns=(1, 1))  # another stat, the same bytes: only a hash tells them apart
    assert made.changed() is not hashed
    # other bytes, of another size: a rewrite of the same size within one tick of the kernel's
    # clock keeps every stat (the limit _sha256 documents)
    path.write_bytes(b"other events")
    assert made.changed()
    path.unlink()
    assert made.changed()


def test_the_changed_files_are_those_of_the_results_given(tmp_path: Path) -> None:
    kept, rewritten = tmp_path / "kept.root", tmp_path / "rewritten.root"
    for path in (kept, rewritten):
        path.write_bytes(b"events")
    results = [
        TaskResult(
            SIMULATION, "done", made={"events": Made.file(SIMULATION, "events", kept, hashed=False)}
        ),
        TaskResult(GEOMETRY, "done"),
        TaskResult(
            stage_task("simulation", "pion"),
            "done",
            made={
                "events": Made.file(
                    stage_task("simulation", "pion"), "events", rewritten, hashed=False
                )
            },
        ),
    ]
    rewritten.write_bytes(b"more events")
    assert changed_files(results) == ["pion/simulation#events"]


def test_a_tree_is_changed_by_any_file_in_it_that_changes_goes_or_is_added(
    tmp_path: Path,
) -> None:
    (tmp_path / "compact").mkdir()
    (tmp_path / "compact" / "a.xml").write_text("<a/>")
    (tmp_path / "b.xml").write_text("<b/>")
    tree = Tree.of(tmp_path)
    assert list(tree.stats) == ["b.xml", "compact/a.xml"] and tree.changed() == []
    (tmp_path / "compact" / "a.xml").write_text("<a>edited</a>")
    (tmp_path / "b.xml").unlink()
    (tmp_path / "c.xml").write_text("<c/>")
    assert tree.changed() == ["b.xml", "compact/a.xml", "c.xml"]
    held = TaskResult(GEOMETRY, "done", tree=tree)
    assert changed_files([held]) == ["geometry/b.xml", "geometry/compact/a.xml", "geometry/c.xml"]
