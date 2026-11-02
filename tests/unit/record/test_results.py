"""The result of each task of a candidate: run after what it reads, blocked when something it
reads did not succeed, and the candidate's status and message from the required ones (no
Key4hep)."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from k4opt.evaluation import Status
from k4opt.problem import EvaluatorSpec, Sample, Stage
from k4opt.problem.plan import (
    GEOMETRY,
    Kind,
    Plan,
    Task,
    TaskId,
    evaluator_task,
    plan_tasks,
    stage_task,
)
from k4opt.record.results import (
    TaskResult,
    blocked,
    candidate_message,
    candidate_status,
    diagnostics,
)
from tests.support.problems import run_in_order

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
    plan = plan_tasks(GAMMA, [ev("a", "simulation"), ev("b")], references=())
    seen: list[tuple[str, list[str]]] = []

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        seen.append((task.id.label, [need.label for need in reads]))
        return done(task.id, "valid" if task.id.kind is Kind.EVALUATOR else "done")

    results = run_in_order(plan, run)
    assert seen == [
        ("geometry", []),
        ("gamma/generation", []),
        ("gamma/simulation", ["geometry", "gamma/generation"]),
        ("gamma/a", ["geometry", "gamma/generation", "gamma/simulation"]),
        ("b", ["geometry"]),
    ]
    assert list(results) == [task.id for task in plan.tasks]


def test_the_executor_does_not_run_a_blocked_task() -> None:
    plan = plan_tasks(GAMMA, [ev("a", "simulation"), ev("b")], references=())
    ran: list[str] = []

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        ran.append(task.id.label)
        if task.id == SIMULATION:
            return done(task.id, "failed", "ddsim crashed")
        return done(task.id, "valid" if task.id.kind is Kind.EVALUATOR else "done")

    results = run_in_order(plan, run)
    assert ran == ["geometry", "gamma/generation", "gamma/simulation", "b"]
    assert results[evaluator_task("a", "gamma")].blocked_by == (SIMULATION,)
    assert results[evaluator_task("b")].status == "valid"


# --- the candidate's status and message, from its tasks ---


def required(*tasks: TaskId, diagnostics: tuple[TaskId, ...] = ()) -> Plan:
    """A plan of ``tasks``, each required, then ``diagnostics``, none of them reading another."""
    return Plan(
        tuple(Task(task, required=True) for task in tasks)
        + tuple(Task(task) for task in diagnostics)
    )


ECHO, CELLS = evaluator_task("echo"), evaluator_task("cells")


@pytest.mark.parametrize(
    ("statuses", "status"),
    [
        (["done", "valid"], Status.VALID),
        (["done", "valid", "invalid"], Status.INVALID),
        (["failed", "blocked", "invalid"], Status.FAILED),
        # blocked by an invalid task: nothing failed, and it is no result
        (["done", "invalid", "blocked"], Status.INVALID),
        (["done", "blocked"], Status.INVALID),  # never alone, but no result all the same
    ],
)
def test_a_candidate_is_failed_before_invalid_before_valid(
    statuses: list[str], status: Status
) -> None:
    tasks = [GEOMETRY, SIMULATION, ECHO][: len(statuses)]
    results = {task: done(task, s) for task, s in zip(tasks, statuses, strict=True)}
    assert candidate_status(required(*tasks), results) is status


@pytest.mark.parametrize("diagnostic", ["failed", "invalid", "blocked"])
def test_a_diagnostic_never_changes_the_candidate_status_or_message(diagnostic: str) -> None:
    plan = required(GEOMETRY, ECHO, diagnostics=(CELLS,))
    results = {GEOMETRY: done(GEOMETRY), ECHO: done(ECHO, "valid"), CELLS: done(CELLS, diagnostic)}
    assert candidate_status(plan, results) is Status.VALID
    assert candidate_message(plan, results) is None
    assert diagnostics(plan, results) == [results[CELLS]]
    assert diagnostics(plan, results | {CELLS: done(CELLS, "valid")}) == []
    assert candidate_status(required(), {}) is Status.VALID


def test_the_message_names_the_causes_then_what_they_blocked() -> None:
    calo, energy = evaluator_task("calo"), evaluator_task("energy")
    results = {
        GEOMETRY: done(GEOMETRY),
        SIMULATION: done(SIMULATION, "failed", "ddsim crashed"),
        calo: TaskResult(calo, Status.BLOCKED, "x", blocked_by=(SIMULATION,)),
        ECHO: done(ECHO, "invalid", "too far out"),
        energy: TaskResult(energy, Status.BLOCKED, "x", blocked_by=(SIMULATION,)),
        CELLS: done(CELLS, "valid", "fine"),
    }
    plan = required(*results)
    assert candidate_message(plan, results) == (
        "gamma/simulation failed: ddsim crashed; echo invalid: too far out; blocked: calo, energy"
    )
    # only the required ones: a diagnostic that failed is not the candidate's
    plan = required(GEOMETRY, ECHO, CELLS, diagnostics=(SIMULATION, calo, energy))
    assert candidate_message(plan, results) == "echo invalid: too far out"
    assert candidate_message(required(GEOMETRY, CELLS), results) is None
