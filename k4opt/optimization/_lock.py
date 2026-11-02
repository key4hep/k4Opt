"""Internal: the lock of a study directory, held while a process works on the study."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import TracebackType

from k4opt.optimization._study import LOCK_NAME, now
from k4opt.optimization.errors import OptimizationError


class Lock:
    """``.lock`` in a study directory while a process works on the study.

    It is created exclusively, holds the host, process and start time of its owner and is
    removed when the owner ends. A lock that a killed process left behind is never taken over:
    the error says whose it is and which file to remove once nothing works on the study.
    """

    def __init__(self, directory: Path) -> None:
        self.path = directory / LOCK_NAME
        self._held = False

    def acquire(self) -> None:
        owner = {"host": socket.gethostname(), "pid": os.getpid(), "started": now()}
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            raise OptimizationError(_in_use(self.path)) from None
        except OSError as exc:
            raise OptimizationError(f"cannot create {self.path}: {exc}") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(owner, handle)
                handle.write("\n")
        except BaseException as exc:
            # a lock without its owner would refuse every later start, as an unknown process's
            self.path.unlink(missing_ok=True)
            if isinstance(exc, OSError):
                raise OptimizationError(f"cannot write {self.path}: {exc}") from exc
            raise
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> Lock:
        self.acquire()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it runs, as another user
        return True
    return True


def _in_use(path: Path) -> str:
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        host, pid, started = owner["host"], owner["pid"], owner["started"]
    except (OSError, ValueError, KeyError, TypeError):
        who, state = "an unknown process (the file cannot be read)", ""
    else:
        who, state = f"process {pid} on {host}, since {started}", ""
        if host == socket.gethostname() and isinstance(pid, int) and not isinstance(pid, bool):
            state = ", which is still running" if _running(pid) else ", which no longer runs"
    return (
        f"the study is in use: {path} names {who}{state}. Once nothing works on the study any "
        f"more, remove {path} and start again"
    )
