"""The scheduler: which tasks of which candidates start, in what order, within the slots."""

from __future__ import annotations

import random

import pytest

from k4opt.evaluation import Status
from k4opt.execution._schedule import Scheduler
from k4opt.problem.plan import GEOMETRY, Plan, Task, TaskId, evaluator_task, stage_task
from k4opt.record.results import DONE, TaskResult

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
    with pytest.raises(ValueError, match="neither running nor parked"):
        scheduler.finished("a", ok(GEOMETRY))
    with pytest.raises(ValueError, match="not running"):
        scheduler.park("a", GEOMETRY, "token")
    with pytest.raises(ValueError, match="not parked"):
        scheduler.unpark("a", GEOMETRY)
    assert scheduler.next() == ("a", Task(GEOMETRY))
    scheduler.park("a", GEOMETRY, "token")
    with pytest.raises(ValueError, match="not running"):
        scheduler.park("a", GEOMETRY, "token")
    scheduler.finished("a", ok(GEOMETRY))
    with pytest.raises(ValueError, match="neither running nor parked"):
        scheduler.finished("a", ok(GEOMETRY))  # a task ends once
    with pytest.raises(ValueError, match="without a result"):
        scheduler.remove("a")


# --- parked tasks: waiting for something else, holding no slot ----------------------------------


def shared() -> Plan:
    """A simulation, which other candidates make alike, and an evaluation that reads it."""
    return Plan((Task(GEOMETRY), Task(SIM, (GEOMETRY,)), Task(SHAPE, (GEOMETRY, SIM))))


def started_now(scheduler: Scheduler) -> list[tuple[str, TaskId]]:
    started = []
    while (item := scheduler.next()) is not None:
        started.append((str(item[0]), item[1].id))
    return started


def test_a_parked_task_frees_its_slots_and_ends_without_taking_one() -> None:
    scheduler = Scheduler(2)
    for key in ("a", "b"):
        scheduler.add(key, shared(), {SIM: 2})
    assert started_now(scheduler) == [("a", GEOMETRY), ("b", GEOMETRY)]
    for key in ("a", "b"):
        scheduler.finished(key, ok(GEOMETRY))
    assert started_now(scheduler) == [("a", SIM)] and scheduler.busy == 2
    scheduler.finished("a", ok(SIM))
    assert started_now(scheduler) == [("a", SHAPE)]  # b's simulation needs both slots
    scheduler.finished("a", ok(SHAPE))
    assert started_now(scheduler) == [("b", SIM)] and scheduler.busy == 2
    # b's simulation waits for one alike: parked, it holds none of its two slots
    scheduler.park("b", SIM, "made-alike")
    assert scheduler.busy == 0 and not scheduler.running and scheduler.idle
    assert scheduler.parked("made-alike") == [("b", Task(SIM, (GEOMETRY,)))]
    assert scheduler.parked("other") == []
    scheduler.finished("b", ok(SIM))  # it reused what was made alike: no slot taken
    assert scheduler.busy == 0 and scheduler.parked("made-alike") == []
    assert started_now(scheduler) == [("b", SHAPE)]


def test_while_a_task_is_parked_its_slot_goes_to_any_ready_task() -> None:
    """Each candidate simulates a sample that all share and one of its own: the slot a parked
    simulation leaves goes to the simulation of the other sample."""
    plan = Plan((Task(SIM), Task(PI_SIM), Task(SEPARATION, (SIM, PI_SIM))))
    scheduler = Scheduler(2)
    for key in ("a", "b"):
        scheduler.add(key, plan)
    assert started_now(scheduler) == [("a", SIM), ("a", PI_SIM)]
    scheduler.finished("a", ok(PI_SIM))
    assert started_now(scheduler) == [("b", SIM)]
    scheduler.park("b", SIM, "alike")  # a's runs
    assert started_now(scheduler) == [("b", PI_SIM)]  # in the slot it left
    assert scheduler.busy == 2
    scheduler.finished("a", ok(SIM))
    scheduler.finished("b", ok(SIM))  # reused
    assert started_now(scheduler) == [("a", SEPARATION)]
    scheduler.finished("b", ok(PI_SIM))
    assert started_now(scheduler) == [("b", SEPARATION)]


def test_an_unparked_task_is_ready_again_and_starts_in_its_order() -> None:
    plan = Plan((Task(SIM), Task(SHAPE, (SIM,))))
    scheduler = Scheduler(2)
    for key in ("a", "b", "c"):
        scheduler.add(key, plan)
    assert started_now(scheduler) == [("a", SIM), ("b", SIM)]
    scheduler.park("b", SIM, "alike")  # a's runs
    assert started_now(scheduler) == [("c", SIM)]
    scheduler.park("c", SIM, "alike")
    assert scheduler.busy == 1 and scheduler.idle
    assert scheduler.parked("alike") == [("b", Task(SIM)), ("c", Task(SIM))]
    scheduler.unpark("c", SIM)  # what it waited for made nothing: it runs itself
    assert scheduler.parked("alike") == [("b", Task(SIM))]
    assert started_now(scheduler) == [("c", SIM)] and scheduler.busy == 2
    with pytest.raises(ValueError, match="not parked"):
        scheduler.unpark("c", SIM)


def test_a_dropped_candidate_leaves_no_parked_task() -> None:
    scheduler = Scheduler(2)
    for key in ("a", "b"):
        scheduler.add(key, shared())
    started_now(scheduler)
    for key in ("a", "b"):
        scheduler.finished(key, ok(GEOMETRY))
    started_now(scheduler)
    scheduler.park("b", SIM, "alike")
    assert scheduler.drop("b") == set()  # nothing of it runs: nothing to stop
    assert scheduler.parked("alike") == [] and scheduler.busy == 1
    assert scheduler.drop("a") == {SIM} and scheduler.busy == 0


def test_a_task_parked_on_one_of_its_own_candidate_ends_once_and_its_readers_start_once() -> None:
    """Two samples alike in one candidate: the second simulation waits for the first."""
    plan = Plan(
        (
            Task(GEOMETRY),
            Task(SIM, (GEOMETRY,)),
            Task(PI_SIM, (GEOMETRY,)),
            Task(SEPARATION, (SIM, PI_SIM)),
        )
    )
    scheduler = Scheduler(2)
    scheduler.add("a", plan)
    started_now(scheduler)
    scheduler.finished("a", ok(GEOMETRY))
    assert started_now(scheduler) == [("a", SIM), ("a", PI_SIM)]
    scheduler.park("a", PI_SIM, "alike")
    assert scheduler.busy == 1
    scheduler.finished("a", ok(SIM))
    assert started_now(scheduler) == []  # the combined one reads the parked one too
    scheduler.finished("a", ok(PI_SIM))
    assert started_now(scheduler) == [("a", SEPARATION)]
    scheduler.finished("a", ok(SEPARATION))
    assert scheduler.done("a") and len(scheduler.remove("a")) == 4


@pytest.mark.parametrize("seed", range(40))
def test_parking_never_breaks_the_accounts(seed: int) -> None:
    """Tasks started, parked, made ready again and ended in random order: the slots in use are
    always the costs of the running tasks, within the capacity, a parked task never starts, and
    every task ends once."""
    pick = random.Random(seed)
    capacity = pick.randint(1, 6)
    scheduler = Scheduler(capacity)
    costs = {}
    for key in range(pick.randint(1, 4)):
        # each holds at most every slot
        costs[f"c{key}"] = {t.id: min(pick.randint(1, 4), capacity) for t in two_samples().tasks}
        scheduler.add(f"c{key}", two_samples(), costs[f"c{key}"])
    running: set[tuple[str, TaskId]] = set()
    parked: set[tuple[str, TaskId]] = set()
    ended: set[tuple[str, TaskId]] = set()
    while True:
        while (item := scheduler.next()) is not None:
            started = (str(item[0]), item[1].id)
            assert started not in running | parked | ended  # never twice, never while parked
            running.add(started)
        assert scheduler.busy == sum(costs[key][task] for key, task in running) <= capacity
        assert scheduler.running == bool(running)
        if not running and not parked:
            break
        move = pick.random()
        if parked and (not running or move < 0.3):
            key, task = sorted(parked, key=str)[pick.randrange(len(parked))]
            parked.discard((key, task))
            if pick.random() < 0.5:
                scheduler.unpark(key, task)  # to start again
            else:
                ended |= {(key, task)} | {
                    (key, held.id) for held in scheduler.finished(key, ok(task))
                }
            continue
        key, task = sorted(running, key=str)[pick.randrange(len(running))]
        running.discard((key, task))
        if move < 0.6:
            scheduler.park(key, task, task.name)  # tokens shared by the candidates
            parked.add((key, task))
            assert (key, task) in {(str(k), t.id) for k, t in scheduler.parked(task.name)}
        else:
            ended |= {(key, task)} | {(key, held.id) for held in scheduler.finished(key, ok(task))}
    for key in costs:
        assert scheduler.done(key)
        assert len(scheduler.remove(key)) == len(two_samples().tasks)
    assert len(ended) == len(costs) * len(two_samples().tasks)
