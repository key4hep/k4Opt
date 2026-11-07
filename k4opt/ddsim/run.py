"""Run ``ddsim`` once, with its command, exit code and logs kept in the run directory."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from k4opt import program
from k4opt.ddsim.command import ddsim_arguments
from k4opt.ddsim.errors import DdsimError
from k4opt.frozen import Json

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
    known: ddsim did not end by itself, or the system does not say). ``observed`` is how it was
    observed and ``trace`` where its raw traces are (:attr:`k4opt.program.ProgramRun.trace`).
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
    observed: program.Observation = "none"
    trace: Path | None = None

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
    executable: Path | None = None,
    observe: program.Observation = "none",
) -> SimulationRun:
    """Run ``ddsim`` (``executable``, or the one found on ``PATH``) on ``compact``, in the new
    directory ``workdir``.

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
    :func:`k4opt.program.execute`); ``observe`` is as for :func:`k4opt.program.run`. Raises
    :class:`DdsimError` for unsupported settings or options, a ``timeout`` that is not a positive
    number of seconds, or an existing ``workdir``.
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
    problem = program.timeout_problem(timeout)
    if problem is not None:
        raise DdsimError(problem)

    try:
        workdir.mkdir(parents=True)
    except FileExistsError as exc:
        raise DdsimError(f"{workdir} already exists; use a new directory") from exc
    except OSError as exc:
        failed = f"cannot create simulation directory {workdir}: {exc}"
        return SimulationRun((), workdir, None, output, stdout, stderr, command, failed)
    run = program.run(
        "ddsim",
        arguments,
        workdir=workdir,
        logs=LOG_FILES,
        timeout=timeout,
        started=started,
        die_with_parent=die_with_parent,
        executable=executable,
        observe=observe,
    )
    error = run.error
    if error is None and not output.is_file():
        error = f"ddsim finished but wrote no {output.name} (logs in {workdir})"
    return SimulationRun(
        run.argv,
        workdir,
        run.returncode,
        output,
        stdout,
        stderr,
        command,
        error,
        run.peak_memory_bytes,
        run.observed,
        run.trace,
    )
