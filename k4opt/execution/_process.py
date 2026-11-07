"""Internal: the local process backend, which runs the tasks of candidates in worker processes.

:class:`ProcessBackend` keeps up to ``workers`` worker processes of this machine
(:mod:`k4opt.execution._worker`), each running one task at a time, and implements the
:class:`~k4opt.execution._backends.Backend` contract with them: it starts and checks one worker
at once, sends each task to an idle one, reads back what it left behind, and can stop any of them
at any time. A worker that dies fails only the task it ran and is replaced, and so does one whose
task runs longer than its time limit (:attr:`~k4opt.execution._tasks.TaskSpec.timeout`).
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from multiprocessing import connection
from multiprocessing.connection import Connection
from pathlib import Path

import k4opt
from k4opt.evaluation import EvaluationResult, Status
from k4opt.execution._backends import Completion, Handle
from k4opt.execution._files import portable
from k4opt.execution._stages import stage_logs
from k4opt.execution._tasks import Observations, TaskOutcome, TaskSpec, Workspace
from k4opt.execution._worker import LOG_NAME, Failure, SessionSpec
from k4opt.execution.errors import RunError
from k4opt.problem.plan import Kind

_HOME = Path(k4opt.__file__).resolve().parents[1]  # where a worker imports this very k4opt from


# what a worker runs: k4opt from _HOME, first on the path (-P keeps the current directory off it)
_MAIN = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from k4opt.execution._worker import main; sys.exit(main(sys.argv[2:]))"
)


_GRACE = 5.0  # seconds a worker has to end after SIGTERM before it is killed


# seconds a worker has to set up the session (import, load the evaluators) before it is stopped
_SETUP = 300.0


# seconds the rest of a message may take once its first bytes came; a worker writes each whole
_MESSAGE = 60.0


# seconds a stage's worker has beyond the stage's time limit, at which the stage stops the program
# it runs itself, keeping its logs: only k4opt's own code around it hanging reaches this
_BACKSTOP = 30.0


@dataclass(slots=True, eq=False)
class _Worker:
    """A worker process and the task it runs (``running``, ``None`` while idle), with the
    process groups of the programs that task started (``groups``); ``lost`` is why a message
    to it could not be sent whole, after which nothing more is written to it. ``began`` is when
    its task began, once the worker is set up, and ``deadline`` when it must have ended (both
    :func:`time.monotonic`; ``None`` for no limit)."""

    process: subprocess.Popen[bytes]
    connection: Connection
    running: tuple[Handle, TaskSpec] | None = None
    groups: set[int] = field(default_factory=set)
    ready: bool = False
    setup: float = field(default_factory=lambda: time.monotonic() + _SETUP)  # its deadline
    lost: OSError | None = None
    began: float | None = None
    deadline: float | None = None

    def begin(self) -> None:
        """Start the clock of the task it runs, which is set up to run it."""
        assert self.running is not None
        _, spec = self.running
        self.began = time.monotonic()
        limit = _limit(spec)
        self.deadline = None if limit is None else self.began + limit


class ProcessBackend:
    """Runs up to ``workers`` tasks at once, each in a worker process of this machine.

    A worker is started when a task finds none idle, set up once with the session ``spec``
    (:class:`~k4opt.execution._worker.SessionSpec`) and kept for the tasks that follow; there are
    never more than ``workers``. A worker that dies (a crash, the kernel's OOM killer) fails its
    task, naming the signal, and is replaced by the next task; its programs (ddsim) are killed
    with it.
    A task whose name has killed ``workers + 1`` workers in a row stops its candidate
    (:class:`~k4opt.execution.RunError`) instead, and so does a worker that has not set up the
    session after a generous deadline: a hung import never holds a job.
    A task that runs longer than its time limit (``spec.timeout``; a stage gets
    :data:`_BACKSTOP` more, as it stops its own program at the limit) has its worker ended as a
    cancelled one is, fails, naming its limit, and the worker is replaced; a time limit reached is
    no death. Cancelled tasks end their workers: SIGTERM, then SIGKILL after a grace period, then
    every process group of what they started.
    """

    def __init__(self, spec: SessionSpec, workers: int) -> None:
        self._spec = spec
        self._count = workers
        self._workers: list[_Worker] = []
        self._roots: dict[str, Path | None] = {}
        self._deaths: dict[tuple[Kind, str], int] = {}
        self._unsent: list[Completion] = []  # tasks that never reached their worker

    @property
    def capacity(self) -> int:
        return self._count

    def cost(self, cpus: int) -> int:
        """A slot per processor, all of them for a task that keeps more busy (it runs alone)."""
        return min(max(cpus, 1), self._count)

    def start(self) -> None:
        """Start a worker and wait until it is set up; :class:`~k4opt.execution.RunError` if it
        refuses the session, dies or takes too long."""
        worker = self._spawn()
        if worker.lost is not None:
            self._end(worker)
            raise RunError(f"the session could not be sent to a worker process: {worker.lost}")
        while not worker.ready:
            if time.monotonic() > worker.setup:
                self._end(worker)
                raise RunError(_slow())
            if not worker.connection.poll(0.2):
                continue
            try:
                message = worker.connection.recv()
            except BlockingIOError:  # part of a message, and the rest never came
                self._end(worker)
                raise RunError(_partial()) from None
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
        assert len(self._workers) < self._count  # the scheduler keeps the tasks within its workers
        ours, theirs = socket.socketpair()
        # a send to a worker that never reads (hung before it set up) gives up in the end
        ours.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, struct.pack("ll", int(_SETUP), 0))
        # a receive, which starts once the first bytes came, gives up if the rest stalls
        ours.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack("ll", int(_MESSAGE), 0))
        with ours, theirs:
            argv = [sys.executable, "-P", "-c", _MAIN, str(_HOME)]
            argv += [str(theirs.fileno()), str(os.getpid())]
            process = subprocess.Popen(
                argv,
                # what the session identified, not what a library set beneath Python since
                # (ROOT's interpreter sets CLING_PREBUILT_MODULE_PATH with C's setenv)
                env=dict(os.environ),
                stdin=subprocess.DEVNULL,
                stdout=2,  # what it prints outside an evaluation: the coordinator's stderr
                pass_fds=(theirs.fileno(),),
                process_group=0,  # out of reach of the terminal's Ctrl-C, which is ours
            )
            worker = _Worker(process, Connection(ours.detach()))
        self._workers.append(worker)
        self._send(worker, self._spec)
        return worker

    def _send(self, worker: _Worker, message: object) -> None:
        """Send ``message`` to ``worker``, unless a message to it was lost: a stream that may
        hold part of a message is never written to again."""
        if worker.lost is None:
            try:
                worker.connection.send(message)
            except OSError as exc:
                worker.lost = exc

    def open(self, workspace: Workspace) -> None:
        self._roots[workspace.id] = workspace.root

    def submit(self, spec: TaskSpec, seen: Observations) -> Handle:
        worker = self._idle() or self._spawn()
        handle = Handle(spec.workspace, spec.task.id)
        worker.running = (handle, spec)
        self._send(worker, (spec, self._roots[spec.workspace], seen))
        if worker.lost is not None:  # it ends at the next wait, as every task does
            self._unsent.append(self._unreached(worker, worker.lost))
        elif worker.ready:  # a new worker's clock starts once it is set up
            worker.begin()
        return handle

    def _unreached(self, worker: _Worker, error: OSError) -> Completion:
        """The end of the task of ``worker``, which could not be sent to it: a worker that
        closed its end died; one that stopped reading is stopped."""
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return self._died(worker)
        assert worker.running is not None
        handle, spec = worker.running
        self._end(worker)
        failure = RunError(
            f"{spec.task.id.label} could not be sent to its worker process ({error}); the worker "
            "was stopped"
        )
        return handle, failure, Observations()

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
        ended, self._unsent = self._unsent, []
        busy = {w.connection: w for w in self._workers if w.running is not None}
        if not busy or ended:
            return ended
        deadlines = [w.deadline for w in busy.values() if w.deadline is not None]
        if deadlines:  # wake up for the first time limit
            timeout = min(timeout, max(min(deadlines) - time.monotonic(), 0.0))
        for ready in connection.wait(list(busy), timeout):
            assert isinstance(ready, Connection)
            completion = self._read(busy[ready])
            if completion is not None:
                ended.append(completion)
        now = time.monotonic()
        expired = []
        for worker in busy.values():
            if worker not in self._workers or worker.running is None:  # ended meanwhile
                continue
            if not worker.ready and now > worker.setup:  # one that never set up holds no job
                handle, _ = worker.running
                self._end(worker)
                ended.append((handle, RunError(_slow()), Observations()))
            elif worker.deadline is not None and now > worker.deadline:
                expired.append(worker)
        return ended + self._expire(expired)

    def _expire(self, workers: list[_Worker]) -> list[Completion]:
        """End ``workers``, whose tasks ran longer than their time limit, as :meth:`cancel` does:
        each task fails, naming its limit, with what it left."""
        for worker in workers:
            _signal(worker, signal.SIGTERM)
        deadline = time.monotonic() + _GRACE
        ended: list[Completion] = []
        for worker in workers:
            assert worker.running is not None
            assert worker.began is not None
            handle, spec = worker.running
            self._end(worker, terminate=False, deadline=deadline)
            elapsed = time.monotonic() - worker.began
            assert spec.timeout is not None
            how = (
                f"{spec.task.id.label} ran longer than {_LIMITS[spec.task.id.kind]} "
                f"({spec.timeout:g} s) and was stopped"
            )
            root = self._roots.get(spec.workspace)
            ended.append((handle, _died(spec, root, how, elapsed), Observations()))
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
            except BlockingIOError:  # part of a message, and the rest never came
                self._end(worker)
                return handle, RunError(f"{spec.task.id.label}: {_partial()}"), Observations()
            except (EOFError, OSError):
                return self._died(worker)
            except Exception as exc:  # an answer that cannot be read: the worker is useless
                self._end(worker)
                return handle, RunError(f"a worker's answer cannot be read: {exc}"), Observations()
            kind = message[0]
            if kind == "ready":
                worker.ready = True
                worker.begin()
            elif kind == "started":
                worker.groups.add(message[1])
            elif kind == "refused":
                self._end(worker)
                return handle, RunError(str(message[1])), Observations()
            else:
                worker.running = None
                worker.groups.clear()
                worker.began = worker.deadline = None
                self._deaths.pop(_name(spec), None)
                if kind == "failed":
                    assert isinstance(message[1], Failure)
                    return handle, message[1].error(), Observations()
                return handle, message[1], message[2]

    def _died(self, worker: _Worker) -> Completion:
        """The end of the task of ``worker``, which died."""
        assert worker.running is not None
        handle, spec = worker.running
        began = worker.began
        status = self._end(worker, terminate=False)
        name = _name(spec)
        deaths = self._deaths[name] = self._deaths.get(name, 0) + 1
        task = spec.task.id
        how = f"the worker process that ran {task.label} died ({_ended(status)})"
        if deaths > self._count:
            return (
                handle,
                RunError(f"{how}, {deaths} times in a row for {task.name}; fewer --workers?"),
                Observations(),
            )
        elapsed = None if began is None else time.monotonic() - began
        return handle, _died(spec, self._roots.get(spec.workspace), how, elapsed), Observations()

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
        ``deadline``), then every process group its task started and its own; its exit status.
        A worker already ended is left as it is."""
        if worker not in self._workers:
            return worker.process.returncode
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
        self._unsent.clear()


def _partial() -> str:
    return (
        f"a worker process sent part of an answer and nothing more for {_MESSAGE:g} s; it was "
        "stopped"
    )


def _slow() -> str:
    return (
        f"a worker process did not set up the session within {_SETUP:g} s (importing k4opt "
        "or loading the evaluators) and was stopped"
    )


def _signal(worker: _Worker, signum: int) -> None:
    if worker.process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            worker.process.send_signal(signum)


def _name(spec: TaskSpec) -> tuple[Kind, str]:
    """What tells tasks apart for counting the deaths of their workers: the stage, or binding,
    whatever the sample and candidate. A worker dies of what does not stop at a sample (the
    code of the binding's implementation, ddsim, the memory all running tasks take together),
    so its deaths on every sample count as one series, which any task of that name that ends
    resets: a binding that dies on one sample but works on another fails the tasks that die, and
    stops a candidate only once ``workers + 1`` of them died with none ending in between."""
    return spec.task.id.kind, spec.task.id.name


def _ended(status: int) -> str:
    """How a process with exit ``status`` ended, as :class:`subprocess.Popen` gives it."""
    if status < 0:
        with contextlib.suppress(ValueError):
            return f"killed by {signal.Signals(-status).name}"
        return f"killed by signal {-status}"
    return f"exit status {status}"


_LIMITS = {
    Kind.GEOMETRY: "the time limit of a geometry",
    Kind.STAGE: "the stage timeout",
    Kind.EVALUATOR: "the evaluator timeout",
}


def _limit(spec: TaskSpec) -> float | None:
    """How long the worker of the task ``spec`` may run it: its time limit, and for a stage that
    runs a program :data:`_BACKSTOP` more, since the stage stops its program at the limit."""
    if spec.timeout is None:
        return None
    return spec.timeout + (_BACKSTOP if spec.task.id.kind is Kind.STAGE else 0.0)


def _died(spec: TaskSpec, root: Path | None, how: str, elapsed: float | None = None) -> TaskOutcome:
    """The ``failed`` outcome of the task ``spec``, whose worker died or was stopped after
    ``elapsed`` seconds: what it left in its directory stays there, named in the message."""
    task, directory = spec.task.id, spec.task.directory
    own = None if root is None or directory is None else root / directory
    message = how if own is None else f"{how}; what it left is in {own}"
    if task.kind is not Kind.EVALUATOR:  # the logs of what it ran, as far as it got
        found = {} if own is None or task.kind is not Kind.STAGE else stage_logs(task.name, own)
        logs = {role: portable(path, root) for role, path in found.items()}
        return TaskOutcome(task, Status.FAILED, message, artifacts=logs, elapsed_seconds=elapsed)
    workdir = log = None
    if own is not None and directory is not None and own.is_dir():  # made just before the call
        workdir = Path(directory)
        log = workdir / LOG_NAME if (own / LOG_NAME).is_file() else None
    return TaskOutcome(
        task,
        Status.FAILED,
        message,
        artifacts={} if log is None else {"log": log},
        elapsed_seconds=elapsed,
        result=EvaluationResult.failed(message),
        workdir=workdir,
    )
