"""The tasks of a candidate: planned from the problem alone, run after what they read, blocked
when something they read did not succeed (no Key4hep)."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from k4opt.evaluation import EvaluationContractError, Status
from k4opt.problem import Stage
from k4opt.problem._tasks import (
    GEOMETRY,
    Kind,
    Plan,
    Task,
    TaskId,
    TaskResult,
    blocked,
    candidate_message,
    candidate_status,
    evaluator_task,
    plan_tasks,
    run_tasks,
    stage_task,
)

GUN = Stage("generation", executor="gun", options={"particle": "gamma"})
DDSIM = Stage("simulation", executor="ddsim", events=3, seed=1)
GENERATION, SIMULATION = stage_task("generation"), stage_task("simulation")


def labels(plan: Plan) -> dict[str, list[str]]:
    """Each task of ``plan`` by label, in plan order, with the labels of what it reads."""
    return {task.id.label: [need.label for need in task.needs] for task in plan.tasks}


def test_a_simulation_reader_reads_the_geometry_and_every_stage_of_its_chain() -> None:
    plan = plan_tasks({"generation": GUN, "simulation": DDSIM}, [("calo", "simulation")])
    assert labels(plan) == {
        "geometry": [],
        "generation": [],  # a gun is fired by ddsim: it reads no geometry
        "simulation": ["geometry", "generation"],
        "calo": ["geometry", "generation", "simulation"],
    }
    directories = {task.id.label: task.directory for task in plan.tasks}
    assert directories == {
        "geometry": "geometry",
        "generation": None,  # writes nothing
        "simulation": "simulation",
        "calo": "evaluators/calo",
    }


def test_every_evaluator_reads_the_geometry_even_one_that_reads_no_stage() -> None:
    plan = plan_tasks({}, [("cost", None), ("echo", None)])
    assert labels(plan) == {"geometry": [], "cost": ["geometry"], "echo": ["geometry"]}
    assert all(GEOMETRY in task.needs for task in plan.of(Kind.EVALUATOR))


def test_each_stage_is_planned_once_however_many_evaluators_read_it() -> None:
    stages = {"generation": GUN, "simulation": DDSIM}
    plan = plan_tasks(stages, [("a", "simulation"), ("b", "generation"), ("c", "simulation")])
    assert list(labels(plan)) == ["geometry", "generation", "simulation", "a", "b", "c"]
    assert labels(plan)["b"] == ["geometry", "generation"]


def test_a_stage_given_as_a_file_reads_nothing() -> None:
    hepmc = Stage("generation", file="zh.hepmc")
    plan = plan_tasks({"generation": hepmc, "simulation": DDSIM}, [("calo", "simulation")])
    assert plan[GENERATION] == Task(GENERATION, given="file")
    assert labels(plan)["simulation"] == ["geometry", "generation"]


def test_a_supplied_stage_reads_nothing_and_the_stages_only_it_needs_are_not_planned() -> None:
    stages = {"generation": GUN, "simulation": DDSIM}
    plan = plan_tasks(stages, [("calo", "simulation")], {"simulation"})
    assert labels(plan) == {"geometry": [], "simulation": [], "calo": ["geometry", "simulation"]}
    assert plan[SIMULATION].given == "supplied"
    # supplied wins over a file the problem gives
    stages["simulation"] = Stage("simulation", file="events.root")
    assert plan_tasks(stages, [("calo", "simulation")], {"simulation"})[SIMULATION].given == (
        "supplied"
    )


def test_the_task_kind_tells_tasks_of_the_same_name_apart() -> None:
    assert stage_task("x") != evaluator_task("x") != TaskId(Kind.GEOMETRY, "x")
    with pytest.raises(KeyError):
        plan_tasks({}, [("calo", None)])[stage_task("calo")]


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        ((Task(GEOMETRY), Task(GEOMETRY)), "task 'geometry' is planned twice"),
        (
            (Task(evaluator_task("calo"), (SIMULATION,)), Task(SIMULATION)),
            "task 'calo' reads simulation, which is not planned before it",
        ),
        ((Task(SIMULATION, (SIMULATION,)),), "task 'simulation' reads simulation"),
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
        calo.id, Status.BLOCKED, "not run: simulation failed", blocked_by=(SIMULATION,)
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
    assert (calo.blocked_by, calo.message) == ((GENERATION,), "not run: generation failed")


def test_the_executor_runs_each_task_once_in_plan_order_with_only_what_it_reads() -> None:
    plan = plan_tasks({"generation": GUN, "simulation": DDSIM}, [("a", "simulation"), ("b", None)])
    seen: list[tuple[str, list[str]]] = []

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        seen.append((task.id.label, [need.label for need in reads]))
        return done(task.id, "valid" if task.id.kind is Kind.EVALUATOR else "done")

    results = run_tasks(plan, run)
    assert seen == [
        ("geometry", []),
        ("generation", []),
        ("simulation", ["geometry", "generation"]),
        ("a", ["geometry", "generation", "simulation"]),
        ("b", ["geometry"]),
    ]
    assert list(results) == [task.id for task in plan.tasks]


def test_the_executor_does_not_run_a_blocked_task() -> None:
    plan = plan_tasks({"generation": GUN, "simulation": DDSIM}, [("a", "simulation"), ("b", None)])
    ran: list[str] = []

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        ran.append(task.id.label)
        if task.id == SIMULATION:
            return done(task.id, "failed", "ddsim crashed")
        return done(task.id, "valid" if task.id.kind is Kind.EVALUATOR else "done")

    results = run_tasks(plan, run)
    assert ran == ["geometry", "generation", "simulation", "b"]
    assert results[evaluator_task("a")].blocked_by == (SIMULATION,)
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
        "simulation failed: ddsim crashed; echo invalid: too far out; blocked: calo, energy"
    )
    assert candidate_message([done(GEOMETRY), done(evaluator_task("a"), "valid")]) is None
