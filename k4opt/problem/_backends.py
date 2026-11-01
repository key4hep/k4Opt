"""Internal: where and how the tasks of candidates run.

A backend executes task descriptions (:class:`~k4opt.problem._work.TaskSpec`) and returns what
they left behind (:class:`~k4opt.problem._work.TaskOutcome`); which task runs when is the
scheduler's to decide (:mod:`k4opt.problem._schedule`), and what becomes of the results, the
coordinator's. A backend never publishes a record, never touches an optimizer's storage and
holds nothing of a candidate but the place of its workspace, from :meth:`Backend.open` to
:meth:`Backend.close_workspace`.

:class:`InlineBackend` runs each task at once, in the caller's thread, one at a time: exactly what
evaluating a candidate has always done. :class:`ProcessBackend` runs up to ``jobs`` tasks at
once, each in a worker process of this machine (:mod:`k4opt.problem._worker`) that the
coordinator can stop at any time. Later backends (a batch system) keep the same contract.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from multiprocessing import connection
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Protocol

import k4opt
from k4opt.evaluation import EvaluationResult, EvaluatorProtocol, Status
from k4opt.problem._tasks import GEOMETRY, Kind, TaskId
from k4opt.problem._work import (
    Observations,
    SessionContext,
    TaskOutcome,
    TaskSpec,
    Workspace,
    execute_task,
)
from k4opt.problem._worker import LOG_NAME, Failure, SessionSpec
from k4opt.problem.errors import RunError


@dataclass(frozen=True, slots=True)
class Handle:
    """A task handed to a backend: the task of the candidate whose workspace is ``workspace``."""

    workspace: str
    task: TaskId


type Completion = tuple[Handle, TaskOutcome | Exception, Observations]
"""A task that ended: what it left behind, or the exception that stops its candidate."""


class Backend(Protocol):
    """Executes the tasks of candidates.

    ``capacity`` is how many slots the scheduler may keep busy with this backend, and
    :meth:`cost` how many a task holds. A task is submitted only once the workspace of its
    candidate is open; ``seen`` are the stats of the files the task's reads made, taken where
    they were made. :meth:`wait` returns the tasks that ended, with what each left behind and the
    stats of what it made, or the exception that stops its candidate. :meth:`cancel` returns
    once none of the tasks given runs any more.
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


_HOME = Path(k4opt.__file__).resolve().parents[1]  # where a worker imports this very k4opt from
# what a worker runs: k4opt from _HOME, first on the path (-P keeps the current directory off it)
_MAIN = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from k4opt.problem._worker import main; sys.exit(main(sys.argv[2:]))"
)
_GRACE = 5.0  # seconds a worker has to end after SIGTERM before it is killed


@dataclass(slots=True, eq=False)
class _Worker:
    """A worker process and the task it runs (``running``, ``None`` while idle), with the
    process groups of the programs that task started (``groups``)."""

    process: subprocess.Popen[bytes]
    connection: Connection
    running: tuple[Handle, TaskSpec] | None = None
    groups: set[int] = field(default_factory=set)
    ready: bool = False


class ProcessBackend:
    """Runs up to ``jobs`` tasks at once, each in a worker process of this machine.

    A worker is started when a task finds none idle, set up once with the session ``spec``
    (:class:`~k4opt.problem._worker.SessionSpec`) and kept for the tasks that follow; there are
    never more than ``jobs``. A worker that dies (a crash, the kernel's OOM killer) fails its task,
    naming the signal, and is replaced by the next task; its programs (ddsim) are killed with it.
    A geometry task that dies, or a task whose name has killed ``jobs + 1`` workers in a row,
    stops its candidate (:class:`~k4opt.problem.RunError`) instead. Cancelled tasks end their
    workers: SIGTERM, then SIGKILL after a grace period.
    """

    def __init__(self, spec: SessionSpec, jobs: int) -> None:
        self._spec = spec
        self._jobs = jobs
        self._workers: list[_Worker] = []
        self._roots: dict[str, Path | None] = {}
        self._deaths: dict[tuple[Kind, str], int] = {}

    @property
    def capacity(self) -> int:
        return self._jobs

    def cost(self, spec: TaskSpec) -> int:
        return min(max(spec.cpus, 1), self._jobs)

    def start(self) -> None:
        """Start a worker and wait until it is set up; :class:`~k4opt.problem.RunError` if it
        refuses the session or dies."""
        worker = self._spawn()
        while not worker.ready:
            if not worker.connection.poll(0.2):
                continue
            try:
                message = worker.connection.recv()
            except (EOFError, OSError):
                status = self._end(worker, terminate=False)
                raise RunError(
                    f"a worker process ended while it was set up ({_ended(status)})"
                ) from None
            if message[0] == "refused":
                self._end(worker)
                raise RunError(str(message[1]))
            worker.ready = message[0] == "ready"

    def _spawn(self) -> _Worker:
        assert len(self._workers) < self._jobs  # the scheduler keeps the tasks within jobs
        ours, theirs = socket.socketpair()
        with ours, theirs:
            argv = [sys.executable, "-P", "-c", _MAIN, str(_HOME)]
            argv += [str(theirs.fileno()), str(os.getpid())]
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=2,  # what it prints outside an evaluation: the coordinator's stderr
                pass_fds=(theirs.fileno(),),
                process_group=0,  # out of reach of the terminal's Ctrl-C, which is ours
            )
            worker = _Worker(process, Connection(ours.detach()))
        self._workers.append(worker)
        with contextlib.suppress(OSError):  # one that died at once is found when read
            worker.connection.send(self._spec)
        return worker

    def open(self, workspace: Workspace) -> None:
        self._roots[workspace.id] = workspace.root

    def submit(self, spec: TaskSpec, seen: Observations) -> Handle:
        worker = self._idle() or self._spawn()
        handle = Handle(spec.workspace, spec.task.id)
        worker.running = (handle, spec)
        with contextlib.suppress(OSError):  # a worker that died is found when read
            worker.connection.send((spec, self._roots[spec.workspace], seen))
        return handle

    def _idle(self) -> _Worker | None:
        """A worker without a task that is still alive; one that died while idle is ended, so
        that no task is charged with its death."""
        for worker in list(self._workers):
            if worker.running is None:
                if worker.process.poll() is None:
                    return worker
                self._end(worker, terminate=False)
        return None

    def wait(self, timeout: float) -> list[Completion]:
        busy = {w.connection: w for w in self._workers if w.running is not None}
        if not busy:
            return []
        ended = []
        for ready in connection.wait(list(busy), timeout):
            assert isinstance(ready, Connection)
            completion = self._read(busy[ready])
            if completion is not None:
                ended.append(completion)
        return ended

    def _read(self, worker: _Worker) -> Completion | None:
        """What ``worker`` sent so far: the end of its task, if it came."""
        assert worker.running is not None
        handle, spec = worker.running
        while True:
            try:
                if not worker.connection.poll():
                    return None
                message = worker.connection.recv()
            except (EOFError, OSError):
                return self._died(worker)
            except Exception as exc:  # an answer that cannot be read: the worker is useless
                self._end(worker)
                return handle, RunError(f"a worker's answer cannot be read: {exc}"), Observations()
            kind = message[0]
            if kind == "ready":
                worker.ready = True
            elif kind == "started":
                worker.groups.add(message[1])
            elif kind == "refused":
                self._end(worker)
                return handle, RunError(str(message[1])), Observations()
            else:
                worker.running = None
                worker.groups.clear()
                self._deaths.pop(_name(spec), None)
                if kind == "failed":
                    assert isinstance(message[1], Failure)
                    return handle, message[1].error(), Observations()
                return handle, message[1], message[2]

    def _died(self, worker: _Worker) -> Completion:
        """The end of the task of ``worker``, which died."""
        assert worker.running is not None
        handle, spec = worker.running
        status = self._end(worker, terminate=False)
        name = _name(spec)
        deaths = self._deaths[name] = self._deaths.get(name, 0) + 1
        task = spec.task.id
        how = f"the worker process that ran {task.label} died ({_ended(status)})"
        if task == GEOMETRY:
            return handle, RunError(f"{how}; the candidate has no geometry"), Observations()
        if deaths > self._jobs:
            return (
                handle,
                RunError(f"{how}, {deaths} times in a row for {task.name}; lower --jobs?"),
                Observations(),
            )
        return handle, _died(spec, self._roots.get(spec.workspace), how), Observations()

    def cancel(self, handles: Collection[Handle]) -> None:
        stopped = set(handles)
        workers = [w for w in self._workers if w.running and w.running[0] in stopped]
        for worker in workers:
            _signal(worker, signal.SIGTERM)
        deadline = time.monotonic() + _GRACE
        for worker in workers:
            self._end(worker, terminate=False, deadline=deadline)

    def _end(self, worker: _Worker, terminate: bool = True, deadline: float | None = None) -> int:
        """End ``worker`` (SIGTERM first if ``terminate``, SIGKILL after the grace period or at
        ``deadline``), then every process group its task started and its own; its exit status."""
        if terminate:
            _signal(worker, signal.SIGTERM)
        left = _GRACE if deadline is None else max(deadline - time.monotonic(), 0.0)
        try:
            worker.process.wait(timeout=left)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait()
        with contextlib.suppress(Exception):  # the groups it reported before it ended
            while worker.connection.poll():
                message = worker.connection.recv()
                if message[0] == "started":
                    worker.groups.add(message[1])
        # a group outlives its leader while a member lives, so its id is not reused meanwhile
        for group in (*worker.groups, worker.process.pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(group, signal.SIGKILL)
        worker.connection.close()
        self._workers.remove(worker)
        return worker.process.returncode

    def close_workspace(self, workspace: str) -> None:
        self._roots.pop(workspace, None)

    def close(self) -> None:
        """Stop every worker; the next task starts new ones."""
        workers = list(self._workers)
        for worker in workers:
            _signal(worker, signal.SIGTERM)
        deadline = time.monotonic() + _GRACE
        for worker in workers:
            self._end(worker, terminate=False, deadline=deadline)
        self._roots.clear()


def _signal(worker: _Worker, signum: int) -> None:
    if worker.process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            worker.process.send_signal(signum)


def _name(spec: TaskSpec) -> tuple[Kind, str]:
    """What tells tasks apart for counting the deaths of their workers: the stage, or binding."""
    return spec.task.id.kind, spec.task.id.name


def _ended(status: int) -> str:
    """How a process with exit ``status`` ended, as :class:`subprocess.Popen` gives it."""
    if status < 0:
        with contextlib.suppress(ValueError):
            return f"killed by {signal.Signals(-status).name}"
        return f"killed by signal {-status}"
    return f"exit status {status}"


def _died(spec: TaskSpec, root: Path | None, how: str) -> TaskOutcome:
    """The ``failed`` outcome of the task ``spec``, whose worker died: what it left in its
    directory stays there, named in the message."""
    task, directory = spec.task.id, spec.task.directory
    own = None if root is None or directory is None else root / directory
    message = how if own is None else f"{how}; what it left is in {own}"
    if task.kind is not Kind.EVALUATOR:
        return TaskOutcome(task, Status.FAILED, message)
    workdir = log = None
    if own is not None and directory is not None and own.is_dir():  # made just before the call
        workdir = Path(directory)
        log = workdir / LOG_NAME if (own / LOG_NAME).is_file() else None
    return TaskOutcome(
        task,
        Status.FAILED,
        message,
        artifacts={} if log is None else {"log": log},
        result=EvaluationResult.failed(message),
        workdir=workdir,
    )
