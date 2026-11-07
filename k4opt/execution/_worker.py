"""Internal: a worker process, which does the tasks of one session's candidates, one at a time.

The local process backend (:class:`~k4opt.execution._process.ProcessBackend`) starts each worker
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
* then, one at a time, a task: ``(TaskSpec, root, seen)`` (:func:`~k4opt.execution._work.
  execute_task`). The worker answers ``("started", group)`` for each program a stage starts,
  which the coordinator kills if the worker dies, and ends with ``("done", outcome, observed)``
  or ``("failed", Failure)``.

An evaluation creates its one instance of its implementation and calls its ``evaluate()`` with the
request the coordinator's preflight made, whose ``check`` it already passed; what the evaluator
prints while it is created and called goes to ``evaluator.log`` in its own directory. The worker
ends when the coordinator closes its end.
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
from k4opt.execution._instances import _Bound
from k4opt.execution._tasks import Observations, SessionContext, TaskSpec
from k4opt.execution._work import execute_task
from k4opt.execution.errors import ProvenanceError, RunError
from k4opt.frozen import FrozenMap, Json
from k4opt.problem.plan import Kind, TaskId
from k4opt.revision import code_fingerprint
from k4opt.software import Software

LOG_NAME = "evaluator.log"
"""The file in an evaluation's own directory that holds what its evaluator printed."""

# what tells an implementation's source apart, and whether those are the bytes that ran
_IDENTITY = ("path", "bytes", "sha256", "helpers", "executed")


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
    ``code`` the fingerprint of k4opt's code (:func:`~k4opt.revision.code_fingerprint`),
    ``environment`` ``KEY4HEP_STACK`` and ``K4GEO`` and ``software`` what it runs on, as the
    session took them; ``implementations`` are loaded first, then ``payload`` is read: the
    :class:`~k4opt.execution._work.SessionContext` and what each evaluation is bound to, whose
    classes it names by their module and name.
    """

    path: tuple[str, ...]
    code: str
    environment: Mapping[str, Json]
    software: Software
    implementations: tuple[Implementation, ...]
    payload: bytes


def session_spec(
    context: SessionContext,
    evaluations: Sequence[tuple[TaskId, _Bound]],
    environment: Mapping[str, Json],
    software: Software,
) -> SessionSpec:
    """The :class:`SessionSpec` of a session with ``context`` whose evaluations are bound as
    ``evaluations``. Raises :class:`~k4opt.execution.RunError` for an implementation a worker could
    not load again by its reference and recognize: one :func:`~k4opt.evaluation.load_evaluator`
    did not load from its file, or that comes from ``__main__``, and for one whose identity is not
    of the code that runs (``executed``): a worker that loads it from its source could run other
    code than this process did."""
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
                f"evaluator {bound.entry.name!r}: with workers each evaluator runs in a "
                "worker process, which loads it again by its reference (FILE.py:Class or "
                f"module:Class); {cls.__qualname__} was not loaded that way by "
                "k4opt.evaluation.load_evaluator (or its module was imported before, or is "
                "__main__), so a worker could not tell it is the same code; evaluate it "
                "inline (workers='inline') instead"
            )
        if source.get("executed") is not True:
            raise RunError(
                f"evaluator {bound.entry.name!r}: the code it runs here is not known to be the "
                "source its hash names (an import hook imported it, or a module it imports, "
                "perhaps from a cached .pyc), so a worker process that loads it again could run "
                "other code; load it from source files Python's path finder imports, or as "
                "FILE.py:Class, or evaluate it inline (workers='inline')"
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
        software,
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
    the programs they start), the implementation each evaluation is bound to (``bound``) and
    what the session runs on (``spec``)."""

    context: SessionContext
    bound: Mapping[TaskId, _Bound]
    spec: SessionSpec

    @classmethod
    def of(cls, spec: SessionSpec, started: Callable[[int], None] | None = None) -> Session:
        """Set up the session ``spec`` in this process; ``started`` is told the process group of
        each program a stage starts, which dies with this process. Raises
        :class:`~k4opt.execution.ProvenanceError` if k4opt's code, the environment or an
        implementation is not the session's."""
        if code_fingerprint() != spec.code:
            raise ProvenanceError(
                "a worker process runs other k4opt code than the session (another checkout "
                "first on the path, or the files changed since the session opened)"
            )
        changed = _changed(spec)
        if changed is not None:
            raise ProvenanceError(f"a worker process does not run as the session: {changed}")
        sys.path[:] = list(dict.fromkeys([*spec.path, *sys.path]))
        for implementation in spec.implementations:
            loaded = evaluator_source(load_evaluator(implementation.reference))
            then = {key: implementation.source.get(key) for key in _IDENTITY}
            if {key: loaded.get(key) for key in _IDENTITY} != then:
                raise ProvenanceError(
                    f"evaluator {implementation.reference}: {then['path']} or a module it "
                    "imports changed after the session loaded it; open a new session"
                )
        changed = _changed(spec)  # an evaluator's module may set variables when imported
        if changed is not None:
            raise ProvenanceError(
                f"a worker process no longer runs as the session once it loaded the evaluators: "
                f"{changed}; an evaluator's module that sets environment variables when it is "
                "imported changes what every task runs on"
            )
        context, bound = pickle.loads(spec.payload)
        assert isinstance(context, SessionContext)
        contexts = {
            sample: replace(stages, started=started, die_with_parent=True)
            for sample, stages in context.contexts.items()
        }
        return cls(replace(context, contexts=FrozenMap(contexts)), FrozenMap(bound), spec)

    def drifted(self, spec: TaskSpec) -> ProvenanceError | None:
        """Why this process no longer runs as the session after the task ``spec``, ``None`` if it
        still does: a task that set a variable (an evaluator writing ``os.environ``) changed what
        every later task here would run on, which no record would say."""
        changed = _changed(self.spec)
        if changed is None:
            return None
        return ProvenanceError(
            f"{spec.task.id.label}: the worker process that ran it no longer runs as the session: "
            f"{changed}; a task that sets environment variables changes what every later task "
            "runs on, so the worker is ended"
        )

    def run(self, spec: TaskSpec, root: Path | None, seen: Observations) -> Message:
        """Do the task ``spec`` in the workspace ``root``: the answer to send."""
        task = spec.task.id
        try:
            # the instance is made within the task, so that its time and memory count
            bound = self.bound.get(task) if task.kind is Kind.EVALUATOR else None
            outcome, observed = execute_task(self.context, spec, root, seen, bound, _logged)
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


_PR_SET_PDEATHSIG = 1  # prctl(2): the signal a process gets when its parent thread ends


def _changed(spec: SessionSpec) -> str | None:
    """How this process differs from the session ``spec`` in ``KEY4HEP_STACK``, ``K4GEO`` and
    the software, for a person; ``None`` if it does not."""
    for name, value in spec.environment.items():
        now = os.environ.get(name)
        if now != value:
            return f"{name} is {now!r}, not {value!r} as in the session"
    changed = spec.software.differences(Software.now())
    return None if not changed else f"its software differs: {'; '.join(changed)}"


def _die_with(parent: int) -> str | None:
    """Have this process get SIGKILL once its parent ``parent`` ends (Linux); why it cannot,
    ``None`` if it does."""
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            error = os.strerror(ctypes.get_errno())
            return f"a worker process cannot be made to end with the session (prctl: {error})"
    if os.getppid() != parent:
        return "the session ended before its worker process started"
    return None


def _refusal(error: Exception) -> str:
    return str(error) if isinstance(error, K4OptError) else f"{type(error).__name__}: {error}"


def serve(connection: Connection, refused: str | None = None) -> int:
    """Answer the coordinator at the other end of ``connection`` until it closes it; one that
    must be ``refused`` is told why once it sent the session."""
    try:
        spec = connection.recv()
    except EOFError:
        return 0
    if refused is not None:
        connection.send(("refused", refused))
        return 1
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
        drifted = session.drifted(task)
        if drifted is not None:  # the coordinator ends it before it gives it another task
            connection.send(("ended", Failure.of(drifted)))
            return 1
        try:
            connection.send(answer)
        except (pickle.PicklingError, TypeError, AttributeError) as exc:  # not sent at all
            connection.send(("failed", Failure.of(exc)))


def main(argv: Sequence[str]) -> int:
    """Serve the coordinator at the end ``argv[0]`` of a socket pair; ``argv[1]`` is its pid."""
    fd, parent = int(argv[0]), int(argv[1])
    signal.signal(signal.SIGTERM, _cancel)
    refused = _die_with(parent)
    try:
        with Connection(fd) as connection:
            return serve(connection, refused)
    except Cancelled:
        return 128 + signal.SIGTERM
