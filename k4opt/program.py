"""Running one program of the Key4hep stack (``ddsim``, ``k4run``) as a stage of a candidate runs
it, and how a parameter's value is written into one of its options.

Shared by the components that run a program (:mod:`k4opt.ddsim`, :mod:`k4opt.k4run`): each
builds its command, says where it keeps the logs and checks what the program wrote; :func:`run`
starts the program found on ``PATH`` in the stage's directory, without a shell, in a process group
of its own, and keeps its standard output and error and a record of the command (``argv``, the
directory, the exit status, the time limit and the peak memory). A program that fails is reported
by :attr:`ProgramRun.error`, never by an exception: settings that cannot be run are refused before
(:class:`ProgramError`).
"""

from __future__ import annotations

import contextlib
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

from k4opt.errors import K4OptError
from k4opt.jsonio import write_json

LOG_TAIL = 15
"""Lines of standard error quoted in the message of a program that failed."""

LOG_ROLES = ("stdout", "stderr", "command")
"""The files :func:`run` keeps, by role: the program's standard output and error and the record of
its command, written once it ended."""


class ProgramError(K4OptError):
    """The settings of a stage cannot be turned into a command of the program that runs it, or set
    what k4opt manages. Raised before anything runs; each program's component has its own
    (:class:`~k4opt.ddsim.DdsimError`, :class:`~k4opt.k4run.K4runError`)."""


@dataclass(frozen=True, slots=True)
class Settable:
    """How a parameter's value is written into an option of an executor that it sets: as
    ``text`` (a particle name), as a ``quantity`` ``{value, unit}`` with a unit of ``units``, as an
    ``expression``, the text ddsim's command line reads (a number, ``<value>*<unit>`` for a
    number with a unit, a text or a boolean), or as the ``value`` itself, a number, a text or a
    boolean without a unit, whose unit the program decides (a k4run argument).

    Whether a value is valid is decided by the check of that option alone, a test of its one
    value that an interval or a set decides (positive, finite, a unit of a list, a nonblank
    name): the ends of a parameter's domain decide it for every value between them, so a
    problem is checked at those ends, before any candidate. An option whose check is not of
    that kind is not settable.
    """

    form: Literal["text", "quantity", "expression", "value"]
    units: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProgramRun:
    """What one run of a program left behind.

    ``error`` is ``None`` only if the program exited with status 0; otherwise it says what went
    wrong, with the end of its standard error. ``returncode`` is ``None`` if the process never ran,
    or was killed because it ran out of time. ``stdout``, ``stderr`` and ``command`` exist only if
    the process got that far. ``peak_memory_bytes`` is the largest peak resident memory of the
    program and of the processes it started and waited for, each on its own (``None`` if it is
    not known: the program did not end by itself, or the system does not say).
    """

    argv: tuple[str, ...]
    cwd: Path
    returncode: int | None
    stdout: Path
    stderr: Path
    command: Path
    error: str | None
    peak_memory_bytes: int | None = None

    @property
    def files(self) -> dict[str, Path]:
        """The existing files of the run, by role: ``stdout``, ``stderr`` and ``command``."""
        roles = {"stdout": self.stdout, "stderr": self.stderr, "command": self.command}
        return {role: path for role, path in roles.items() if path.is_file()}


def tail(path: Path) -> str:
    """The last :data:`LOG_TAIL` lines of the text file ``path``; empty if it cannot be read."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=LOG_TAIL)
    except OSError:
        return ""
    return "\n".join(line.rstrip("\r\n") for line in lines)


def timeout_problem(timeout: float | None) -> str | None:
    """Why ``timeout`` is no time limit, ``None`` if it is one: ``None`` (no limit) or a positive,
    finite number of seconds; ``nan`` would never run out, a negative one at once."""
    if timeout is None:
        return None
    number = isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    if not number or not 0 < timeout < math.inf:
        return f"timeout must be a positive number of seconds, got {timeout!r}"
    return None


_LAUNCHER = Path(__file__).with_name("_launch.py")


def _launched(argv: list[str], die_with_parent: bool) -> list[str]:
    """``argv``, started so that it dies with the calling thread if ``die_with_parent`` (where
    the system can do that: Linux)."""
    if not die_with_parent or not sys.platform.startswith("linux"):
        return argv
    return [sys.executable, "-I", str(_LAUNCHER), str(os.getpid()), *argv]


def _peak(usage: resource.struct_rusage | None) -> int | None:
    """The peak resident memory in ``usage``, in bytes (Linux counts it in KiB)."""
    if usage is None or usage.ru_maxrss <= 0:
        return None
    return usage.ru_maxrss * 1024 if sys.platform.startswith("linux") else usage.ru_maxrss


def _reap(process: subprocess.Popen[bytes], timeout: float | None) -> tuple[int, int | None]:
    """Wait for ``process`` to end, as ``process.wait(timeout)`` does: its exit status and its
    peak resident memory in bytes, the largest of it and of each process it waited for, as the
    kernel kept it (``wait4``)."""
    deadline = None if timeout is None else time.monotonic() + timeout
    delay = 0.0005
    while True:
        try:
            pid, status, usage = os.wait4(process.pid, 0 if deadline is None else os.WNOHANG)
        except ChildProcessError:  # reaped elsewhere: its status is Popen's to say
            return process.wait(), None
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return process.returncode, _peak(usage)
        assert deadline is not None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout or 0)
        delay = min(delay * 2, remaining, 0.05)
        time.sleep(delay)


def execute(
    argv: list[str],
    *,
    cwd: Path,
    stdout: IO[bytes],
    stderr: IO[bytes],
    timeout: float | None,
    started: Callable[[int], None] | None = None,
    measured: Callable[[int], None] | None = None,
    die_with_parent: bool = False,
) -> int | None:
    """Run ``argv`` to its end: its exit status, ``None`` if it ran longer than ``timeout``.

    It runs in a process group of its own, which is killed as a whole when its time runs out or
    this process is interrupted (Ctrl-C, SIGTERM), so nothing a wrapper script started (a shell
    script without ``exec``, a container) outlives it; a process that leaves the group does.
    ``started`` is told the id of that group as soon as it exists, and ``measured`` the peak
    resident memory in bytes of the program, and of the processes it started and waited for, once
    it ended by itself (where the system says). With ``die_with_parent`` the
    program is killed when this thread ends, even by SIGKILL (on Linux; through a launcher that
    fails the run, with the reason in ``stderr``, if the system refuses that).
    """
    with subprocess.Popen(
        _launched(argv, die_with_parent),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        process_group=0,
    ) as process:
        try:
            if started is not None:
                started(process.pid)  # the leader of its own group: the group's id
            returncode, peak = _reap(process, timeout)
            if measured is not None and peak is not None:
                measured(peak)
            return returncode
        except BaseException as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                return None
            raise


def run(
    name: str,
    arguments: Sequence[str],
    *,
    workdir: Path,
    logs: Mapping[str, str],
    cwd: Path | None = None,
    timeout: float | None = None,
    started: Callable[[int], None] | None = None,
    die_with_parent: bool = False,
) -> ProgramRun:
    """Run the program ``name`` found on ``PATH`` with ``arguments`` in the existing directory
    ``cwd`` (``workdir`` if ``None``), keeping the files ``logs`` names by role
    (:data:`LOG_ROLES`) in the existing directory ``workdir``.

    The program is started by its absolute path, so that a relative ``PATH`` entry still resolves
    from inside ``cwd``. One that runs longer than ``timeout`` seconds is killed with every
    process it started, and the run fails with its logs kept. ``started`` is told the id of the
    process group the program runs in once it started, and ``die_with_parent`` kills it when the
    calling thread ends, even by SIGKILL (see :func:`execute`). The caller checks ``timeout``
    (:func:`timeout_problem`) and what the program should have written.
    """
    stdout, stderr, command = (workdir / logs[role] for role in LOG_ROLES)
    here = workdir if cwd is None else cwd

    def made(argv: tuple[str, ...], returncode: int | None, error: str | None) -> ProgramRun:
        return ProgramRun(argv, here, returncode, stdout, stderr, command, error)

    found = shutil.which(name)
    if found is None:
        return made((), None, f"{name} not found on PATH (is the Key4hep environment set up?)")
    argv = (os.path.abspath(found), *arguments)
    peaks: list[int] = []
    try:
        with stdout.open("wb") as out, stderr.open("wb") as err:
            # None: killed when its time ran out
            returncode = execute(
                list(argv),
                cwd=here,
                stdout=out,
                stderr=err,
                timeout=timeout,
                started=started,
                measured=peaks.append,
                die_with_parent=die_with_parent,
            )
    except OSError as exc:
        return made(argv, None, f"{name} could not be started: {exc}")
    error = None
    if returncode is None:
        error = f"{name} ran longer than {timeout:g} s and was killed (logs in {workdir}):\n"
        error += tail(stderr)
    elif returncode != 0:
        error = f"{name} exited with status {returncode} (logs in {workdir}):\n" + tail(stderr)
    peak = peaks[0] if peaks else None
    record = {
        "argv": argv,
        "cwd": str(here),
        "returncode": returncode,
        "timeout": timeout,
        "peak_memory_bytes": peak,
    }
    try:
        write_json(command, record)
    except OSError as exc:
        failure = f"cannot record {name} command in {command}: {exc}"
        error = failure if error is None else f"{error}\n{failure}"
    return ProgramRun(argv, here, returncode, stdout, stderr, command, error, peak)
