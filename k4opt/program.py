"""Running one program of the Key4hep stack (``ddsim``, ``k4run``) as a stage of a candidate runs
it, and how a parameter's value is written into one of its options.

Shared by the components that run a program (:mod:`k4opt.ddsim`, :mod:`k4opt.k4run`): each
builds its command, says where it keeps the logs and checks what the program wrote; :func:`run`
starts the program (the executable it is given, or the one found on ``PATH``) in the stage's
directory, without a shell, in a process group of its own, and keeps its standard output and
error and a record of the command (``argv``, the directory, the exit status, the time limit and
the peak memory). A program that fails is reported by :attr:`ProgramRun.error`, never by an
exception: settings that cannot be run are refused before (:class:`ProgramError`).

What a program runs is identified by :func:`locate` and :func:`identify` (its executable, where
its links lead, with its size and SHA-256). ddsim and k4run are Python scripts
(``#!/usr/bin/env python3``): the same script with another ``PYTHONPATH`` runs other code, so what
they load is identified with the build and the variables they run with
(:class:`k4opt.software.Software`).

What a program does while it runs can be *observed* (:data:`Observation`): ``files`` runs it under
``strace``, which keeps each file any of its processes opened or failed to open, each program they
started and each connection they made; ``libraries`` has the loader say which libraries it loaded
(``LD_DEBUG``); ``none`` observes nothing. The raw traces are left in :data:`TRACE` in its working
directory, for :mod:`k4opt.software.observe` to read and remove. Neither changes what the program
does nor how it ends: its exit status, its signal and its peak memory are its own.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
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

type Observation = Literal["files", "libraries", "none"]
"""How closely a program is watched while it runs: the files it opens (``strace``), the libraries
it loads (``LD_DEBUG``), or nothing."""

OBSERVATIONS: tuple[Observation, ...] = ("files", "libraries", "none")
"""Every :data:`Observation`, from the closest to none."""

TRACE = "trace"
"""The directory in a program's working directory where :func:`run` leaves the raw traces of an
observed run, one file by process: ``t.<id>`` (``files``) or ``ld.<id>`` (``libraries``)."""

STRACE_OPTIONS = (
    "-f",
    "-ff",
    "-qq",
    "-y",
    "--kill-on-exit",
    "--seccomp-bpf",
    "-e",
    "signal=none",
    "-e",
    "trace=open,openat,openat2,execve,execveat,connect,socket",
)
"""How ``strace`` observes a program at the ``files`` level: every process and thread it starts
(``-f``), each into a file of its own (``-ff``), with the path of every descriptor (``-y``: a
relative path is then known from the directory it was opened in), the opens, the programs started,
the connections and the sockets made only (a datagram needs no connection), stopped in the kernel
for nothing else (``--seccomp-bpf``: a few percent of a short job), and the program killed if
``strace`` is (``--kill-on-exit``)."""


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
    observed: Observation = "none"
    trace: Path | None = None

    @property
    def files(self) -> dict[str, Path]:
        """The existing files of the run, by role: ``stdout``, ``stderr`` and ``command``."""
        roles = {"stdout": self.stdout, "stderr": self.stderr, "command": self.command}
        return {role: path for role, path in roles.items() if path.is_file()}


def locate(name: str) -> Path | None:
    """The executable ``name`` found on ``PATH``, where its links lead; ``None`` if there is
    none."""
    found = shutil.which(name)
    return None if found is None else Path(os.path.realpath(found))


def identify(executable: Path) -> dict[str, object] | None:
    """What identifies ``executable``: its ``path``, ``bytes`` and ``sha256``, and the
    ``interpreter`` that runs it, identified alike (``None`` for a program that is no script);
    ``None`` if it cannot be read, or its interpreter cannot be found or read: what would run
    is then not known.

    ddsim and k4run are scripts that ``#!/usr/bin/env python3`` runs in the first interpreter on
    ``PATH``: that interpreter, not ``PATH`` itself, is what ``PATH`` decides for them. It keeps
    the path ``PATH`` gave, links not followed, since the interpreter of a virtual environment
    finds its packages from where it is."""
    found = _file_identity(executable)
    if found is None:
        return None
    try:
        interpreter = _interpreter(executable)
    except OSError:
        return None
    if interpreter is None:
        return {**found, "interpreter": None}
    identified = _file_identity(interpreter)
    return None if identified is None else {**found, "interpreter": identified}


def _file_identity(path: Path) -> dict[str, object] | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            size = 0
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
                size += len(block)
    except OSError:
        return None
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _interpreter(script: Path) -> Path | None:
    """The interpreter the ``#!`` line of ``script`` names, as ``PATH`` finds it now for
    ``#!/usr/bin/env NAME``; ``None`` for a file without one. Raises :class:`OSError` if
    ``script`` cannot be read, or if its ``#!`` line names no interpreter ``PATH`` finds."""
    with script.open("rb") as handle:
        line = handle.readline(4096)
    if not line.startswith(b"#!"):
        return None
    words = line[2:].decode(errors="replace").split()
    if not words:
        raise OSError(f"the #! line of {script} names no interpreter")
    if Path(words[0]).name != "env":
        return Path(words[0])
    name = next((word for word in words[1:] if not word.startswith("-")), None)
    found = None if name is None else shutil.which(name)
    if found is None:
        raise OSError(f"PATH finds no {name}, which the #! line of {script} names")
    return Path(found)


def tail(path: Path) -> str:
    """The last :data:`LOG_TAIL` lines of the text file ``path``; empty if it cannot be read."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=LOG_TAIL)
    except OSError:
        return ""
    return "\n".join(line.rstrip("\r\n") for line in lines)


def ended(returncode: int) -> str:
    """How a program that did not succeed ended, for a person: ``exited with status 3``, or
    ``was killed by SIGKILL`` for a negative status (a signal); a SIGKILL that k4opt did not
    send is most often the kernel's, out of memory."""
    if returncode >= 0:
        return f"exited with status {returncode}"
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        return f"was killed by signal {-returncode}"
    if name == "SIGKILL":
        return "was killed by SIGKILL (out of memory? the record holds its peak memory)"
    return f"was killed by {name}"


def timeout_problem(timeout: float | None) -> str | None:
    """Why ``timeout`` is no time limit, ``None`` if it is one: ``None`` (no limit) or a positive,
    finite number of seconds; ``nan`` would never run out, a negative one at once."""
    if timeout is None:
        return None
    number = isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    if not number or not 0 < timeout < math.inf:
        return f"timeout must be a positive number of seconds, got {timeout!r}"
    return None


def unobservable(observation: Observation) -> str | None:
    """Why programs cannot be observed at ``observation`` here, ``None`` if they can. ``files``
    needs an ``strace`` on ``PATH`` that knows its :data:`STRACE_OPTIONS` and may trace a process
    (some containers and batch systems forbid it): tried on ``true``, in milliseconds. The loader
    of the C library says what it loads (``libraries``) wherever programs run."""
    if observation != "files":
        return None
    strace = shutil.which("strace")
    if strace is None:
        return "strace is not installed (none on PATH)"
    true = shutil.which("true") or "/bin/true"
    with tempfile.TemporaryDirectory(prefix="k4opt-strace-") as scratch:
        try:
            done = subprocess.run(
                [strace, *STRACE_OPTIONS, "-o", os.path.join(scratch, "t"), "--", true],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"{strace} cannot be run: {exc}"
        said = " ".join(done.stderr.decode(errors="replace").split())
        if done.returncode != 0:
            return f"{strace} cannot trace a program here (exit status {done.returncode}): {said}"
        if not os.listdir(scratch):
            return f"{strace} traced a program but wrote no trace"
    return None


def _observed(
    argv: list[str], observation: Observation, trace: Path | None
) -> tuple[list[str], dict[str, str], Observation]:
    """``argv`` as it runs observed at ``observation``, with the variables that adds, and how it
    is observed then: not at all if ``trace`` is ``None`` or ``strace`` went from ``PATH``."""
    if trace is None or observation == "none":
        return argv, {}, "none"
    if observation == "libraries":
        return argv, {"LD_DEBUG": "files", "LD_DEBUG_OUTPUT": str(trace / "ld")}, "libraries"
    strace = shutil.which("strace")
    if strace is None:
        return argv, {}, "none"
    return [strace, *STRACE_OPTIONS, "-o", str(trace / "t"), "--", *argv], {}, "files"


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
    added: Mapping[str, str] | None = None,
) -> int | None:
    """Run ``argv`` to its end: its exit status, ``None`` if it ran longer than ``timeout``.

    It runs with exactly this process's ``os.environ`` and the variables ``added`` (those of an
    observation, never compared), which is what k4opt identified
    (:class:`k4opt.software.Software`), not with what a library set beneath Python (ROOT's
    interpreter sets ``CLING_PREBUILT_MODULE_PATH`` with C's ``setenv``), which a child would
    otherwise inherit unseen. It runs in a process group of its own, which is killed as a whole
    when its time runs out or this process is interrupted (Ctrl-C, SIGTERM), so nothing a wrapper
    script started (a shell script without ``exec``, a container) outlives it; a process that
    leaves the group does.
    ``started`` is told the id of that group as soon as it exists, and ``measured`` the peak
    resident memory in bytes of the program, and of the processes it started and waited for, once
    it ended by itself (where the system says). With ``die_with_parent`` the
    program is killed when this thread ends, even by SIGKILL (on Linux; through a launcher that
    fails the run, with the reason in ``stderr``, if the system refuses that).
    """
    with subprocess.Popen(
        _launched(argv, die_with_parent),
        cwd=cwd,
        env={**os.environ, **(added or {})},
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
    executable: Path | None = None,
    observe: Observation = "none",
) -> ProgramRun:
    """Run the program ``name`` with ``arguments`` in the existing directory ``cwd``
    (``workdir`` if ``None``), keeping the files ``logs`` names by role (:data:`LOG_ROLES`) in
    the existing directory ``workdir``.

    The program is ``executable`` if it is given (as a session pins it, :func:`locate`), else the
    one found on ``PATH`` now. It is started by its absolute path, so that a relative ``PATH``
    entry still resolves from inside ``cwd``. One that runs longer than ``timeout`` seconds is
    killed with every process it started, and the run fails with its logs kept. ``started`` is
    told the id of the process group the program runs in once it started, and
    ``die_with_parent`` kills it when the calling thread ends, even by SIGKILL (see
    :func:`execute`). It is observed at ``observe``, its raw traces left in :data:`TRACE` in
    ``workdir`` (not at all if that directory cannot be made, or ``strace`` is gone from ``PATH``:
    :attr:`ProgramRun.observed` says). The caller checks ``timeout`` (:func:`timeout_problem`), and
    whether programs can be observed here (:func:`unobservable`), and what the program should have
    written.
    """
    stdout, stderr, command = (workdir / logs[role] for role in LOG_ROLES)
    here = workdir if cwd is None else cwd

    def made(argv: tuple[str, ...], returncode: int | None, error: str | None) -> ProgramRun:
        return ProgramRun(argv, here, returncode, stdout, stderr, command, error)

    found = str(executable) if executable is not None else shutil.which(name)
    if found is None:
        return made((), None, f"{name} not found on PATH (is the Key4hep environment set up?)")
    argv = (os.path.abspath(found), *arguments)
    trace: Path | None = None
    if observe != "none":
        trace = workdir / TRACE
        try:
            trace.mkdir()
        except OSError:
            trace = None
    launched, added, observed = _observed(list(argv), observe, trace)
    if observed == "none" and trace is not None:
        trace.rmdir()
        trace = None
    peaks: list[int] = []
    try:
        with stdout.open("wb") as out, stderr.open("wb") as err:
            # None: killed when its time ran out
            returncode = execute(
                launched,
                cwd=here,
                stdout=out,
                stderr=err,
                timeout=timeout,
                started=started,
                measured=peaks.append,
                die_with_parent=die_with_parent,
                added=added,
            )
    except OSError as exc:
        if trace is not None:
            shutil.rmtree(trace, ignore_errors=True)
        return made(argv, None, f"{name} could not be started: {exc}")
    error = None
    if returncode is None:
        error = f"{name} ran longer than {timeout:g} s and was killed (logs in {workdir}):\n"
        error += tail(stderr)
    elif returncode != 0:
        error = f"{name} {ended(returncode)} (logs in {workdir}):\n" + tail(stderr)
        if observed == "files" and returncode < 0 and returncode != -signal.SIGKILL:
            error += (
                "\n(observed at the files level, under strace, which keeps ROOT from printing "
                "the backtrace of a crash: observe it at the libraries level to see it)"
            )
    peak = peaks[0] if peaks else None
    record = {
        "argv": argv,
        "cwd": str(here),
        "returncode": returncode,
        "timeout": timeout,
        "peak_memory_bytes": peak,
        "observed": observed,
    }
    try:
        write_json(command, record)
    except OSError as exc:
        failure = f"cannot record {name} command in {command}: {exc}"
        error = failure if error is None else f"{error}\n{failure}"
    return ProgramRun(argv, here, returncode, stdout, stderr, command, error, peak, observed, trace)
