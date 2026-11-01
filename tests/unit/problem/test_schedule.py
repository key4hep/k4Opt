"""The scheduler: which tasks of which candidates start, in what order, within the slots."""

from __future__ import annotations

import random

import pytest

from k4opt.evaluation import Status
from k4opt.problem._schedule import Scheduler
from k4opt.problem._tasks import (
    DONE,
    GEOMETRY,
    Plan,
    Task,
    TaskId,
    TaskResult,
    evaluator_task,
    stage_task,
)

GEN, SIM = stage_task("generation", "gamma"), stage_task("simulation", "gamma")
PI_GEN, PI_SIM = stage_task("generation", "pi"), stage_task("simulation", "pi")
SHAPE, PI_SHAPE = evaluator_task("shape", "gamma"), evaluator_task("shape", "pi")
READOUT, SEPARATION = evaluator_task("readout"), evaluator_task("separation")


def two_samples() -> Plan:
    """Two samples simulated from one geometry, an evaluation of each, one of the geometry alone
    and a combined evaluator that reads both evaluations."""
    return Plan(
        (
            Task(GEOMETRY),
            Task(GEN),
            Task(SIM, (GEOMETRY, GEN)),
            Task(SHAPE, (GEOMETRY, GEN, SIM)),
            Task(PI_GEN),
            Task(PI_SIM, (GEOMETRY, PI_GEN)),
            Task(PI_SHAPE, (GEOMETRY, PI_GEN, PI_SIM)),
            Task(READOUT, (GEOMETRY,)),
            Task(SEPARATION, (GEOMETRY, SHAPE, PI_SHAPE)),
        )
    )


def ok(task: TaskId) -> TaskResult:
    return TaskResult(task, DONE if task.kind.value != "evaluator" else Status.VALID)


def run_all(scheduler: Scheduler, *, pick: random.Random | None = None) -> list[tuple[str, TaskId]]:
    """Start whatever is ready, end the running tasks one at a time (in random order with
    ``pick``) and return the order the tasks started in."""
    started: list[tuple[str, TaskId]] = []
    running: list[tuple[str, TaskId]] = []
    while True:
        while (item := scheduler.next()) is not None:
            key, task = item
            started.append((str(key), task.id))
            running.append((str(key), task.id))
        assert scheduler.busy <= scheduler.capacity
        if not running:
            return started
        index = pick.randrange(len(running)) if pick else 0
        key, task = running.pop(index)
        scheduler.finished(key, ok(task))


def test_with_one_slot_every_candidate_runs_in_plan_order_one_after_another() -> None:
    scheduler = Scheduler(1)
    scheduler.add("a", two_samples())
    scheduler.add("b", two_samples())
    planned = [task.id for task in two_samples().tasks]
    assert run_all(scheduler) == [("a", t) for t in planned] + [("b", t) for t in planned]
    assert scheduler.done("a") and scheduler.done("b")


def test_independent_tasks_start_together_once_what_they_read_has_ended() -> None:
    scheduler = Scheduler(8)
    scheduler.add("a", two_samples())
    first = {scheduler.next()[1].id for _ in range(3)}  # type: ignore[index]
    # the geometry and both generations read nothing; the readout waits for the geometry
    assert first == {GEOMETRY, GEN, PI_GEN}
    assert scheduler.next() is None
    scheduler.finished("a", ok(GEOMETRY))
    for task in (GEN, PI_GEN):
        scheduler.finished("a", ok(task))
    ready = set()
    while (item := scheduler.next()) is not None:
        ready.add(item[1].id)
    # both simulations and the readout run side by side
    assert ready == {SIM, PI_SIM, READOUT}


def test_a_combined_evaluator_starts_only_once_every_role_input_ended() -> None:
    scheduler = Scheduler(8)
    scheduler.add("a", two_samples())
    order = run_all(scheduler, pick=random.Random(3))
    started = [task for _, task in order]
    assert started.index(SEPARATION) > max(started.index(SHAPE), started.index(PI_SHAPE))


def test_a_shared_stage_starts_once_however_many_tasks_read_it() -> None:
    scheduler = Scheduler(8)
    scheduler.add("a", two_samples())
    started = [task for _, task in run_all(scheduler, pick=random.Random(5))]
    assert started.count(GEOMETRY) == 1 and len(started) == len(set(started))


@pytest.mark.parametrize("seed", range(20))
def test_the_running_tasks_never_hold_more_slots_than_there_are(seed: int) -> None:
    capacity = 1 + seed % 4
    scheduler = Scheduler(capacity)
    for key in range(1 + seed % 3):
        scheduler.add(f"c{key}", two_samples())
    started = run_all(scheduler, pick=random.Random(seed))
    assert len(started) == len(set(started)) == (1 + seed % 3) * len(two_samples().tasks)


def test_ready_tasks_start_by_candidate_then_plan_order() -> None:
    scheduler = Scheduler(2)
    scheduler.add("first", two_samples())
    scheduler.add("second", two_samples())
    assert [scheduler.next() for _ in range(2)] == [
        ("first", Task(GEOMETRY)),
        ("first", Task(GEN)),
    ]


def test_a_task_whose_need_failed_is_blocked_at_once_takes_no_slot_and_never_starts() -> None:
    scheduler = Scheduler(4)
    scheduler.add("a", two_samples())
    while (item := scheduler.next()) is not None:
        if item[1].id in (GEN, PI_GEN, READOUT):
            scheduler.finished("a", ok(item[1].id))
    scheduler.finished("a", ok(GEOMETRY))
    started = []
    while (item := scheduler.next()) is not None:
        started.append(item[1].id)
    assert set(started) == {SIM, PI_SIM, READOUT}
    decided = scheduler.finished("a", TaskResult(SIM, Status.FAILED, "ddsim exited with 1"))
    # its evaluation is blocked by it, and the combined one by the stage, not the evaluation
    assert [(r.id, r.status, r.blocked_by) for r in decided] == [
        (SHAPE, Status.BLOCKED, (SIM,)),
    ]
    assert scheduler.busy == 2  # the blocked task took no slot
    for task in (PI_SIM, READOUT):
        scheduler.finished("a", ok(task))
    assert scheduler.next() == ("a", Task(PI_SHAPE, (GEOMETRY, PI_GEN, PI_SIM)))
    held = scheduler.finished("a", ok(PI_SHAPE))
    assert [(r.id, r.blocked_by) for r in held] == [(SEPARATION, (SIM,))]
    assert scheduler.next() is None and scheduler.done("a")


def test_idle_means_a_free_slot_and_nothing_ready() -> None:
    scheduler = Scheduler(2)
    assert scheduler.idle
    scheduler.add("a", Plan((Task(GEOMETRY), Task(READOUT, (GEOMETRY,)))))
    assert not scheduler.idle  # the geometry is ready
    scheduler.next()
    assert scheduler.idle  # one slot is free and the readout waits for the geometry
    scheduler.finished("a", ok(GEOMETRY))
    assert not scheduler.idle


def test_a_dropped_candidate_frees_its_slots_and_starts_nothing_more() -> None:
    scheduler = Scheduler(3)
    scheduler.add("a", two_samples())
    scheduler.add("b", two_samples())
    for _ in range(3):
        scheduler.next()
    assert scheduler.drop("a") == {GEOMETRY, GEN, PI_GEN}
    assert scheduler.busy == 0
    assert {str(scheduler.next()[0]) for _ in range(3)} == {"b"}  # type: ignore[index]


def test_a_task_costs_its_slots_but_never_more_than_there_are() -> None:
    plan = Plan((Task(GEOMETRY), Task(GEN), Task(PI_GEN)))
    scheduler = Scheduler(4)
    scheduler.add("a", plan, {GEN: 3, PI_GEN: 9})
    assert scheduler.next() == ("a", Task(GEOMETRY))
    assert scheduler.next() == ("a", Task(GEN))
    assert scheduler.busy == 4 and scheduler.next() is None
    scheduler.finished("a", ok(GEOMETRY))
    scheduler.finished("a", ok(GEN))
    assert scheduler.next() == ("a", Task(PI_GEN)) and scheduler.busy == 4  # all of them


def test_later_tasks_that_fit_start_while_a_heavier_one_waits_but_only_capacity_of_them() -> None:
    scheduler = Scheduler(4)
    scheduler.add("big", Plan((Task(GEOMETRY), Task(SIM, (GEOMETRY,)))), {SIM: 4})  # 4 threads
    small = [f"s{n}" for n in range(1, 10)]
    for key in small:
        scheduler.add(key, Plan((Task(GEOMETRY),)))

    def started() -> list[str]:
        keys = []
        while (item := scheduler.next()) is not None:
            keys.append(str(item[0]))
        return keys

    assert started() == ["big", "s1", "s2", "s3"]
    scheduler.finished("big", ok(GEOMETRY))  # SIM is first now, and needs all four slots
    assert started() == ["s4"]  # one free slot: a later task passes it
    passed = ["s4"]
    for key in ("s1", "s2", "s3"):
        scheduler.finished(key, ok(GEOMETRY))
        passed += started()
    assert passed == ["s4", "s5", "s6", "s7"]  # as many as the capacity, then no more
    scheduler.finished("s4", ok(GEOMETRY))
    assert started() == [] and not scheduler.idle  # it waits for its slots, not idle
    for key in ("s5", "s6"):
        scheduler.finished(key, ok(GEOMETRY))
        assert started() == []
    scheduler.finished("s7", ok(GEOMETRY))
    assert scheduler.next() == ("big", Task(SIM, (GEOMETRY,))) and scheduler.busy == 4
    scheduler.finished("big", ok(SIM))
    assert started() == ["s8", "s9"]


@pytest.mark.parametrize("seed", range(30))
def test_weighted_tasks_never_exceed_the_capacity_and_every_task_starts(seed: int) -> None:
    pick = random.Random(seed)
    capacity = pick.randint(1, 8)
    scheduler = Scheduler(capacity)
    candidates = pick.randint(1, 4)
    for key in range(candidates):
        costs = {task.id: pick.randint(1, 10) for task in two_samples().tasks}
        scheduler.add(f"c{key}", two_samples(), costs)
    started = run_all(scheduler, pick=pick)  # asserts busy <= capacity after every start
    assert len(started) == len(set(started)) == candidates * len(two_samples().tasks)


def test_with_unit_costs_no_task_ever_passes_another() -> None:
    scheduler = Scheduler(3)
    for key in ("a", "b"):
        scheduler.add(key, two_samples())
    started = run_all(scheduler)
    reference = Scheduler(3)
    for key in ("a", "b"):
        reference.add(key, two_samples(), {task.id: 1 for task in two_samples().tasks})
    assert run_all(reference) == started


def test_misuse_is_refused() -> None:
    with pytest.raises(ValueError, match="capacity"):
        Scheduler(0)
    scheduler = Scheduler(1)
    scheduler.add("a", two_samples())
    with pytest.raises(ValueError, match="added twice"):
        scheduler.add("a", two_samples())
    with pytest.raises(ValueError, match="not running"):
        scheduler.finished("a", ok(GEOMETRY))
    with pytest.raises(ValueError, match="without a result"):
        scheduler.remove("a")
