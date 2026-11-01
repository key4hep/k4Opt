"""Run ``ddsim`` once, with its command, exit code and logs kept in the run directory."""

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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO

from k4opt.ddsim.command import ddsim_arguments, option_args, particle_gun_args
from k4opt.ddsim.errors import DdsimError
from k4opt.frozen import Json
from k4opt.jsonio import write_json

_LOG_TAIL = 15  # lines of standard error quoted in a failure message

LOG_FILES: Mapping[str, str] = {
    "stdout": "ddsim.stdout.log",
    "stderr": "ddsim.stderr.log",
    "command": "ddsim.json",
}
"""The files besides its output that a run keeps in its directory, by role (the command record
is written once ddsim ended)."""


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """What one ``ddsim`` invocation left behind.

    ``error`` is ``None`` only if ``ddsim`` exited with status 0 and wrote ``output``; otherwise
    it says what went wrong. ``returncode`` is ``None`` if the process never ran, or was killed
    because it ran out of time. The files ``stdout``, ``stderr``, ``command`` and ``output`` exist
    only if the process got that far. ``peak_memory_bytes`` is the largest peak resident memory of
    ddsim and of the processes it started and waited for, each on its own (``None`` if it is not
    known: ddsim did not end by itself, or the system does not say).
    """

    argv: tuple[str, ...]
    cwd: Path
    returncode: int | None
    output: Path
    stdout: Path
    stderr: Path
    command: Path
    error: str | None
    peak_memory_bytes: int | None = None

    @property
    def files(self) -> dict[str, Path]:
        """The existing files of the run, by role: stdout, stderr, command and edm4hep."""
        roles = {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command": self.command,
            "edm4hep": self.output,
        }
        return {role: path for role, path in roles.items() if path.is_file()}


def check_particle_gun(particle_gun: Json) -> None:
    """Raise :class:`DdsimError` if ``particle_gun`` does not describe a gun ddsim can fire."""
    particle_gun_args(particle_gun)


def check_options(options: Mapping[str, Json]) -> None:
    """Raise :class:`DdsimError` if ``options`` (ddsim option names mapped to values) cannot be
    passed to ddsim faithfully, or set what k4opt manages."""
    option_args(options)


def check_arguments(
    *,
    events: int,
    seed: int | None,
    particle_gun: Json = None,
    input_file: Path | None = None,
    options: Mapping[str, Json],
) -> None:
    """Raise :class:`DdsimError` if these settings cannot be turned into a ``ddsim`` command.

    Needs no files and starts nothing, so it can run before anything is prepared.
    """
    ddsim_arguments(
        compact=Path("compact.xml"),
        output=Path("events.edm4hep.root"),
        events=events,
        seed=seed,
        particle_gun=particle_gun,
        input_file=input_file,
        options=options,
    )


def _tail(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=_LOG_TAIL)
    except OSError:
        return ""
    return "\n".join(line.rstrip("\r\n") for line in lines)


def _check_timeout(timeout: float | None) -> None:
    """Raise :class:`DdsimError` unless ``timeout`` is ``None`` (no limit) or a positive, finite
    number of seconds: ``nan`` would never run out, a negative one at once."""
    if timeout is None:
        return
    number = isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    if not number or not 0 < timeout < math.inf:
        raise DdsimError(f"timeout must be a positive number of seconds, got {timeout!r}")


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


def _execute(
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


def simulate(
    *,
    compact: Path,
    workdir: Path,
    events: int,
    seed: int | None,
    particle_gun: Json = None,
    input_file: Path | None = None,
    options: Mapping[str, Json],
    timeout: float | None = None,
    started: Callable[[int], None] | None = None,
    die_with_parent: bool = False,
) -> SimulationRun:
    """Run the ``ddsim`` found on ``PATH`` on ``compact``, in the new directory ``workdir``.

    The command is an argument list without a shell: the options k4opt manages (geometry,
    output, event count, seed, particle gun or input file) followed by ``options`` in order
    (see :func:`~k4opt.ddsim.command.option_args`). Relative ``compact`` and ``input_file``
    paths are taken from the current directory; option values are passed unchanged and ddsim runs in
    ``workdir``, so a file named in an option (``physics.pdgfile``) must be absolute.
    ``workdir`` receives ``events.edm4hep.root``, ``ddsim.stdout.log``, ``ddsim.stderr.log``
    and ``ddsim.json`` (command, directory, exit code, time limit, peak memory). A ``ddsim`` that
    runs longer
    than ``timeout`` seconds is killed with every process it started, and the run fails with its
    logs kept. ``started`` is told the id of the process group ddsim runs in once it started, and
    ``die_with_parent`` kills ddsim when the calling thread ends, even by SIGKILL (see
    :func:`_execute`). Raises :class:`DdsimError` for unsupported
    settings or options, a ``timeout`` that
    is not a positive number of seconds, or an existing ``workdir``.
    """
    # ddsim runs inside workdir, so every path in its command line must be absolute
    compact = Path(os.path.abspath(compact))
    input_file = None if input_file is None else Path(os.path.abspath(input_file))
    workdir = Path(os.path.abspath(workdir))
    output = workdir / "events.edm4hep.root"
    stdout, stderr, command = (
        workdir / LOG_FILES[role] for role in ("stdout", "stderr", "command")
    )
    arguments = ddsim_arguments(
        compact=compact,
        output=output,
        events=events,
        seed=seed,
        particle_gun=particle_gun,
        input_file=input_file,
        options=options,
    )
    _check_timeout(timeout)

    def run(argv: tuple[str, ...], returncode: int | None, error: str | None) -> SimulationRun:
        return SimulationRun(argv, workdir, returncode, output, stdout, stderr, command, error)

    try:
        workdir.mkdir(parents=True)
    except FileExistsError as exc:
        raise DdsimError(f"{workdir} already exists; use a new directory") from exc
    except OSError as exc:
        return run((), None, f"cannot create simulation directory {workdir}: {exc}")

    found = shutil.which("ddsim")
    if found is None:
        return run((), None, "ddsim not found on PATH (is the Key4hep environment set up?)")
    # a relative PATH entry would no longer resolve from inside workdir
    argv = (os.path.abspath(found), *arguments)
    peaks: list[int] = []
    try:
        with stdout.open("wb") as out, stderr.open("wb") as err:
            # None: killed when its time ran out
            returncode = _execute(
                list(argv),
                cwd=workdir,
                stdout=out,
                stderr=err,
                timeout=timeout,
                started=started,
                measured=peaks.append,
                die_with_parent=die_with_parent,
            )
    except OSError as exc:
        return run(argv, None, f"ddsim could not be started: {exc}")
    error = None
    if returncode is None:
        error = f"ddsim ran longer than {timeout:g} s and was killed (logs in {workdir}):\n"
        error += _tail(stderr)
    elif returncode != 0:
        error = f"ddsim exited with status {returncode} (logs in {workdir}):\n" + _tail(stderr)
    elif not output.is_file():
        error = f"ddsim finished but wrote no {output.name} (logs in {workdir})"
    peak = peaks[0] if peaks else None
    record = {
        "argv": argv,
        "cwd": str(workdir),
        "returncode": returncode,
        "timeout": timeout,
        "peak_memory_bytes": peak,
    }
    try:
        write_json(command, record)
    except OSError as exc:
        failure = f"cannot record ddsim command in {command}: {exc}"
        error = failure if error is None else f"{error}\n{failure}"
    return replace(run(argv, returncode, error), peak_memory_bytes=peak)
