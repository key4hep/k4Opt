"""Internal: which tasks of which candidates may run now, within one budget of slots.

The scheduler only keeps account: it runs nothing, starts no thread, reads no file and knows no
clock. A coordinator adds the plan of each candidate (:meth:`Scheduler.add`), asks for the next
task to start (:meth:`Scheduler.next`), hands it to whatever executes tasks (a backend) and
reports each result back (:meth:`Scheduler.finished`). Whatever runs the tasks, the order they
may start in is decided here alone, from the tasks each one reads (``Task.needs``):

* a task is *ready* once every task it reads has a result; one whose needs did not all succeed
  gets its ``blocked`` result (:func:`~k4opt.problem._tasks.blocked`) at once, takes no slot and
  never starts;
* every running task holds its cost in slots, and the costs of the running tasks never sum to
  more than the capacity, whichever candidates they belong to;
* ready tasks start in the order of their candidate (the first added first) and then of their
  plan, so with a capacity of one every candidate runs exactly in plan order, one after another;
* the scheduler is *idle* when a slot is free and no task is ready: the time to add another
  candidate, which keeps the candidates under way as few as the slots need.
"""

from __future__ import annotations

import heapq
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field

from k4opt.problem._tasks import Plan, Task, TaskId, TaskResult, blocked


@dataclass(slots=True)
class _Candidate:
    """The account of one candidate: its plan, the result of each task that ended and how many
    of its needs each other task still waits for."""

    plan: Plan
    order: int  # the candidate's place among those added
    costs: Mapping[TaskId, int]
    waiting: dict[TaskId, int] = field(default_factory=dict)
    readers: dict[TaskId, list[Task]] = field(default_factory=dict)
    results: dict[TaskId, TaskResult] = field(default_factory=dict)
    running: set[TaskId] = field(default_factory=set)


class Scheduler:
    """The tasks of the candidates added, each started at most once and only after what it reads,
    with at most ``capacity`` slots in use at a time."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        self.capacity = capacity
        self._candidates: dict[Hashable, _Candidate] = {}
        self._ready: list[tuple[int, int, Hashable, Task]] = []  # (candidate order, plan index, …)
        self._index: dict[Hashable, dict[TaskId, int]] = {}
        self._added = 0
        self._busy = 0

    def add(self, key: Hashable, plan: Plan, costs: Mapping[TaskId, int] | None = None) -> None:
        """Add the candidate ``key`` with the tasks of ``plan``; each running task holds its cost
        in ``costs`` (1 if not given), at most the capacity."""
        if key in self._candidates:
            raise ValueError(f"candidate {key!r} is added twice")
        given = dict(costs or {})
        candidate = _Candidate(
            plan,
            self._added,
            {task.id: min(max(given.get(task.id, 1), 1), self.capacity) for task in plan.tasks},
        )
        self._added += 1
        self._candidates[key] = candidate
        self._index[key] = {task.id: index for index, task in enumerate(plan.tasks)}
        for task in plan.tasks:
            candidate.waiting[task.id] = len(task.needs)
            for need in task.needs:
                candidate.readers.setdefault(need, []).append(task)
        for task in plan.tasks:
            if not task.needs:
                self._push(key, task)

    def _push(self, key: Hashable, task: Task) -> None:
        candidate = self._candidates[key]
        heapq.heappush(self._ready, (candidate.order, self._index[key][task.id], key, task))

    def next(self) -> tuple[Hashable, Task] | None:
        """The ready task to start now, marked as running, or ``None`` if none is ready or the
        first of them needs more slots than are free."""
        if not self._ready:
            return None
        _, _, key, task = self._ready[0]
        cost = self._candidates[key].costs[task.id]
        if self._busy + cost > self.capacity:
            return None
        heapq.heappop(self._ready)
        self._busy += cost
        self._candidates[key].running.add(task.id)
        return key, task

    def finished(self, key: Hashable, result: TaskResult) -> list[TaskResult]:
        """Record ``result`` of a running task of ``key`` and free its slots; return the
        ``blocked`` results this decides for the tasks that read it (and their readers), which
        are recorded too and never start."""
        candidate = self._candidates[key]
        if result.id not in candidate.running:
            raise ValueError(f"task {result.id.label} of {key!r} is not running")
        candidate.running.discard(result.id)
        self._busy -= candidate.costs[result.id]
        decided: list[TaskResult] = []
        pending = [result]
        while pending:
            done = pending.pop(0)
            candidate.results[done.id] = done
            for reader in candidate.readers.get(done.id, ()):
                candidate.waiting[reader.id] -= 1
                if candidate.waiting[reader.id]:
                    continue
                held = blocked(reader, candidate.results)
                if held is None:
                    self._push(key, reader)
                else:
                    decided.append(held)
                    pending.append(held)
        return decided

    def drop(self, key: Hashable) -> set[TaskId]:
        """Forget the candidate ``key``: none of its tasks starts any more, and the slots of its
        running ones are free. Returns the tasks that were running, which the caller stops."""
        candidate = self._candidates.pop(key)
        self._index.pop(key)
        self._ready = [entry for entry in self._ready if entry[2] != key]
        heapq.heapify(self._ready)
        self._busy -= sum(candidate.costs[task] for task in candidate.running)
        return set(candidate.running)

    def done(self, key: Hashable) -> bool:
        """Whether every task of ``key`` has a result."""
        candidate = self._candidates[key]
        return len(candidate.results) == len(candidate.plan.tasks)

    def results(self, key: Hashable) -> dict[TaskId, TaskResult]:
        """The results of the tasks of ``key`` so far."""
        return dict(self._candidates[key].results)

    def remove(self, key: Hashable) -> dict[TaskId, TaskResult]:
        """Forget the candidate ``key``, whose tasks all have results; return them."""
        if not self.done(key):
            raise ValueError(f"candidate {key!r} still has tasks without a result")
        results = self._candidates.pop(key).results
        self._index.pop(key)
        return results

    @property
    def busy(self) -> int:
        """The slots the running tasks hold."""
        return self._busy

    @property
    def running(self) -> bool:
        """Whether any task runs."""
        return any(candidate.running for candidate in self._candidates.values())

    @property
    def idle(self) -> bool:
        """Whether a slot is free and no task is ready to take it."""
        return self._busy < self.capacity and not self._ready
