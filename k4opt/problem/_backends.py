"""Internal: where and how the tasks of candidates run.

A backend executes task descriptions (:class:`~k4opt.problem._work.TaskSpec`) and returns what
they left behind (:class:`~k4opt.problem._work.TaskOutcome`); which task runs when is the
scheduler's to decide (:mod:`k4opt.problem._schedule`), and what becomes of the results, the
coordinator's. A backend never publishes a record, never touches an optimizer's storage and
holds nothing of a candidate but the place of its workspace, from :meth:`Backend.open` to
:meth:`Backend.close_workspace`.

:class:`InlineBackend` runs each task at once, in the caller's thread, one at a time: exactly what
evaluating a candidate has always done. Other backends (worker processes, later a batch system)
keep the same contract.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from k4opt.evaluation import EvaluatorProtocol
from k4opt.problem._tasks import Kind, TaskId
from k4opt.problem._work import (
    Observations,
    SessionContext,
    TaskOutcome,
    TaskSpec,
    Workspace,
    execute_task,
)


@dataclass(frozen=True, slots=True)
class Handle:
    """A task handed to a backend: the task of the candidate whose workspace is ``workspace``."""

    workspace: str
    task: TaskId


type Completion = tuple[Handle, TaskOutcome, Observations]


class Backend(Protocol):
    """Executes the tasks of candidates.

    ``capacity`` is how many slots the scheduler may keep busy with this backend, and
    :meth:`cost` how many a task holds. A task is submitted only once the workspace of its
    candidate is open; ``seen`` are the stats of the files the task's reads made, taken where
    they were made. :meth:`wait` returns the tasks that ended, with what each left behind and the
    stats of what it made. :meth:`cancel` returns once none of the tasks given runs any more.
    """

    @property
    def capacity(self) -> int: ...

    def cost(self, spec: TaskSpec) -> int: ...

    def open(self, workspace: Workspace) -> None: ...

    def submit(self, spec: TaskSpec, seen: Observations) -> Handle: ...

    def wait(self, timeout: float) -> list[Completion]: ...

    def cancel(self, handles: Collection[Handle]) -> None: ...

    def close_workspace(self, workspace: str) -> None: ...

    def close(self) -> None: ...


type Instances = Callable[[str, TaskId], EvaluatorProtocol]
"""The evaluator instance of an evaluation, by the id of its candidate's workspace."""


class InlineBackend:
    """Runs each task as it is submitted, in the caller's thread, with the evaluator instances the
    candidate's preflight created and checked (``instances``): one task at a time, as k4opt has
    always evaluated a candidate. An exception of a task propagates from :meth:`submit`."""

    capacity = 1

    def __init__(self, context: SessionContext, instances: Instances) -> None:
        self._context = context
        self._instances = instances
        self._roots: dict[str, Path | None] = {}
        self._ended: list[Completion] = []

    def cost(self, spec: TaskSpec) -> int:
        return 1

    def open(self, workspace: Workspace) -> None:
        self._roots[workspace.id] = workspace.root

    def submit(self, spec: TaskSpec, seen: Observations) -> Handle:
        task = spec.task.id
        instance = None
        if task.kind is Kind.EVALUATOR:
            instance = self._instances(spec.workspace, task)
        outcome, observed = execute_task(
            self._context, spec, self._roots[spec.workspace], seen, instance
        )
        handle = Handle(spec.workspace, task)
        self._ended.append((handle, outcome, observed))
        return handle

    def wait(self, timeout: float) -> list[Completion]:
        ended, self._ended = self._ended, []
        return ended

    def cancel(self, handles: Collection[Handle]) -> None:
        """Nothing runs between calls: a task ends within :meth:`submit`."""
        cancelled = set(handles)
        self._ended = [ended for ended in self._ended if ended[0] not in cancelled]

    def close_workspace(self, workspace: str) -> None:
        self._roots.pop(workspace, None)

    def close(self) -> None:
        self._roots.clear()
        self._ended.clear()
