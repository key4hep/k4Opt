"""Internal: a worker process, which does the tasks of one session's candidates, one at a time.

The local process backend (:class:`~k4opt.problem._backends.ProcessBackend`) starts each worker
in the coordinator's current directory, as a new Python that puts the directory holding the
coordinator's k4opt first on its path and calls :func:`main` with ``FD PARENT``: its end of a
socket pair to the coordinator, and the coordinator's process id. The worker dies with the
coordinator (SIGKILL when it ends, on Linux; otherwise when the socket closes), and ends at
SIGTERM, which reaches it as an
exception wherever it is: a ddsim it runs is then killed with its process group, as Ctrl-C does
in the coordinator's own process.

The two talk by request and response, each message a tuple whose first item says what it is:

* the coordinator first sends a :class:`SessionSpec`; the worker loads the evaluator
  implementations again by their references, checks that they, k4opt's code and the environment
  are those of the session, and answers ``("ready",)`` or ``("refused", message)``;
* then, one at a time, a task: ``(TaskSpec, root, seen)`` (:func:`~k4opt.problem._work.
  execute_task`). The worker answers ``("started", group)`` for each program a stage starts,
  which the coordinator kills if the worker dies, and ends with ``("done", outcome, observed)``
  or ``("failed", Failure)``.

An evaluation gets a new instance of its implementation, whose ``check()`` is called with the
request the preflight made before its ``evaluate()`` (both on that instance); what the evaluator
prints while it is called goes to ``evaluator.log`` in its own directory. The worker ends when
the coordinator closes its end.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import pickle
import signal
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
from types import FrameType

from k4opt.errors import K4OptError
from k4opt.evaluation import evaluator_source, load_evaluator
from k4opt.frozen import FrozenMap, Json
from k4opt.problem._instances import _Bound, _instance
from k4opt.problem._tasks import Kind, TaskId
from k4opt.problem._work import Observations, SessionContext, TaskSpec, execute_task
from k4opt.problem.errors import RunError
from k4opt.revision import code_fingerprint

LOG_NAME = "evaluator.log"
"""The file in an evaluation's own directory that holds what its evaluator printed."""

_IDENTITY = ("path", "bytes", "sha256")  # what tells an implementation's source apart


@dataclass(frozen=True, slots=True)
class Implementation:
    """An evaluator implementation as a worker loads it again: its ``reference``
    (``/abs/path/file.py:Class`` or ``package.module:Class``) and the ``source`` identity the
    session recorded (:func:`~k4opt.evaluation.evaluator_source`)."""

    reference: str
    source: Mapping[str, Json]


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """What every task of a session needs, sent once to each worker.

    ``path`` is the coordinator's ``sys.path``, which a module reference is imported from;
    ``code`` the fingerprint of k4opt's code (:func:`~k4opt.revision.code_fingerprint`) and
    ``environment`` ``KEY4HEP_STACK`` and ``K4GEO``, as the session took them; ``implementations``
    are loaded first, then ``payload`` is read: the :class:`~k4opt.problem._work.SessionContext`
    and what each evaluation is bound to, whose classes it names by their module and name.
    """

    path: tuple[str, ...]
    code: str
    environment: Mapping[str, Json]
    implementations: tuple[Implementation, ...]
    payload: bytes


def session_spec(
    context: SessionContext,
    evaluations: Sequence[tuple[TaskId, _Bound]],
    environment: Mapping[str, Json],
) -> SessionSpec:
    """The :class:`SessionSpec` of a session with ``context`` whose evaluations are bound as
    ``evaluations``. Raises :class:`~k4opt.problem.RunError` for an implementation a worker could
    not load again by its reference and recognize: one :func:`~k4opt.evaluation.load_evaluator`
    did not load from its file, or that comes from ``__main__``."""
    implementations: dict[type, Implementation] = {}
    for _, bound in evaluations:
        cls = bound.implementation
        if cls in implementations:
            continue
        source = evaluator_source(cls)
        reference = source.get("reference")
        if (
            not isinstance(reference, str)
            or source.get("sha256") is None
            or cls.__module__ == "__main__"
        ):
            raise RunError(
                f"evaluator {bound.entry.name!r}: with more than one job each evaluator runs in a "
                "worker process, which loads it again by its reference (FILE.py:Class or "
                f"module:Class); {cls.__qualname__} was not loaded that way by "
                "k4opt.evaluation.load_evaluator (or its module was imported before, or is "
                "__main__), so a worker could not tell it is the same code"
            )
        implementations[cls] = Implementation(reference, FrozenMap(source))
    try:
        payload = pickle.dumps((replace(context, listed={}), dict(evaluations)))
    except Exception as exc:
        raise RunError(
            f"the session cannot be described to a worker process: {type(exc).__name__}: {exc}"
        ) from exc
    return SessionSpec(
        tuple(sys.path),
        code_fingerprint(),
        FrozenMap(environment),
        tuple(implementations.values()),
        payload,
    )


class RemoteTraceback(Exception):  # noqa: N818 - not an error of its own: a cause's traceback
    """The traceback of an exception a task raised in a worker process, as text: the cause of
    the exception raised for it in the coordinator."""

    def __str__(self) -> str:
        return f"\n{self.args[0]}" if self.args else ""


class TaskError(Exception):
    """An exception a task raised in a worker process that cannot be rebuilt in the coordinator:
    its type and message."""


@dataclass(frozen=True, slots=True)
class Failure:
    """An exception a task raised in a worker: its type (``kind``), ``message`` and
    ``traceback`` as text, and the ``exception`` itself pickled, if it can be rebuilt."""

    kind: str
    message: str
    traceback: str
    exception: bytes | None = None

    @classmethod
    def of(cls, error: BaseException) -> Failure:
        text = "".join(traceback.format_exception(error))
        data: bytes | None = None
        with contextlib.suppress(Exception):
            data = pickle.dumps(error)
            pickle.loads(data)  # one that cannot be rebuilt (other arguments) is sent as text
        return cls(f"{type(error).__module__}.{type(error).__qualname__}", str(error), text, data)

    def error(self) -> Exception:
        """The exception to raise in the coordinator: the one the task raised (a
        :class:`~k4opt.errors.K4OptError` as itself), or a :class:`TaskError` if it cannot be
        rebuilt here, with the worker's traceback as its cause."""
        error: Exception | None = None
        if self.exception is not None:
            with contextlib.suppress(Exception):
                loaded = pickle.loads(self.exception)
                if isinstance(loaded, Exception):
                    error = loaded
        if error is None:
            error = TaskError(f"{self.kind}: {self.message}")
        error.__cause__ = RemoteTraceback(self.traceback)
        return error


type Message = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class Session:
    """A session as a worker holds it: what every task needs (``context``, whose stages report
    the programs they start) and the implementation each evaluation is bound to (``bound``)."""

    context: SessionContext
    bound: Mapping[TaskId, _Bound]

    @classmethod
    def of(cls, spec: SessionSpec, started: Callable[[int], None] | None = None) -> Session:
        """Set up the session ``spec`` in this process; ``started`` is told the process group of
        each program a stage starts, which dies with this process. Raises
        :class:`~k4opt.problem.RunError` if k4opt's code, the environment or an implementation
        is not the session's."""
        if code_fingerprint() != spec.code:
            raise RunError(
                "a worker process runs other k4opt code than the session (another checkout "
                "first on the path, or the files changed since the session opened)"
            )
        for name, value in spec.environment.items():
            now = os.environ.get(name)
            if now != value:
                raise RunError(
                    f"{name} is {now!r} in a worker process, not {value!r} as in the session"
                )
        sys.path[:] = list(dict.fromkeys([*spec.path, *sys.path]))
        for implementation in spec.implementations:
            loaded = evaluator_source(load_evaluator(implementation.reference))
            then = {key: implementation.source.get(key) for key in _IDENTITY}
            if {key: loaded.get(key) for key in _IDENTITY} != then:
                raise RunError(
                    f"evaluator {implementation.reference}: {then['path']} changed after the "
                    "session loaded it; open a new session"
                )
        context, bound = pickle.loads(spec.payload)
        assert isinstance(context, SessionContext)
        contexts = {
            sample: replace(stages, started=started, die_with_parent=True)
            for sample, stages in context.contexts.items()
        }
        return cls(replace(context, contexts=FrozenMap(contexts)), FrozenMap(bound))

    def run(self, spec: TaskSpec, root: Path | None, seen: Observations) -> Message:
        """Do the task ``spec`` in the workspace ``root``: the answer to send."""
        task = spec.task.id
        try:
            instance = None
            if task.kind is Kind.EVALUATOR:
                assert spec.request is not None
                instance = _instance(self.bound[task])
                check = getattr(instance, "check", None)
                if check is not None:  # on the instance that evaluates, as in the preflight
                    check(spec.request)
            outcome, observed = execute_task(
                self.context, spec, root, seen, instance, output=_logged
            )
        except Exception as exc:
            return ("failed", Failure.of(exc))
        if task.kind is Kind.EVALUATOR and outcome.workdir is not None:
            outcome = replace(outcome, artifacts=FrozenMap({"log": outcome.workdir / LOG_NAME}))
        return ("done", outcome, observed)


def _flush() -> None:
    """Write out what Python and C (ROOT's ``printf``) still buffer for standard output."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    with contextlib.suppress(Exception):
        ctypes.CDLL(None).fflush(None)


@contextlib.contextmanager
def _logged(workdir: Path) -> Iterator[None]:
    """Standard output and error of this process, Python's and every library's, in
    ``workdir/evaluator.log`` while the context lasts."""
    _flush()
    saved = os.dup(1), os.dup(2)
    try:
        with (workdir / LOG_NAME).open("ab") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                yield
            finally:
                _flush()
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
    finally:
        os.close(saved[0])
        os.close(saved[1])


class Cancelled(BaseException):
    """SIGTERM: the coordinator stops the task this worker runs, and the worker."""


def _cancel(signum: int, frame: FrameType | None) -> None:
    raise Cancelled(f"signal {signum}")


def _die_with(parent: int) -> bool:
    """Have this process get SIGKILL once its parent ends (Linux); ``False`` if the parent
    ``parent`` already ended."""
    if sys.platform.startswith("linux"):
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    return os.getppid() == parent


def _refusal(error: Exception) -> str:
    return str(error) if isinstance(error, K4OptError) else f"{type(error).__name__}: {error}"


def serve(connection: Connection) -> int:
    """Answer the coordinator at the other end of ``connection`` until it closes it."""
    try:
        spec = connection.recv()
    except EOFError:
        return 0
    try:
        session = Session.of(spec, started=lambda group: connection.send(("started", group)))
    except Exception as exc:
        connection.send(("refused", _refusal(exc)))
        return 1
    connection.send(("ready",))
    while True:
        try:
            task, root, seen = connection.recv()
        except EOFError:
            return 0
        answer = session.run(task, root, seen)
        try:
            connection.send(answer)
        except (pickle.PicklingError, TypeError, AttributeError) as exc:  # not sent at all
            connection.send(("failed", Failure.of(exc)))


def main(argv: Sequence[str]) -> int:
    """Serve the coordinator at the end ``argv[0]`` of a socket pair; ``argv[1]`` is its pid."""
    fd, parent = int(argv[0]), int(argv[1])
    signal.signal(signal.SIGTERM, _cancel)
    if not _die_with(parent):
        return 1
    try:
        with Connection(fd) as connection:
            return serve(connection)
    except Cancelled:
        return 128 + signal.SIGTERM
