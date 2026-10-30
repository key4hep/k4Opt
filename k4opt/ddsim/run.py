"""Run ``ddsim`` once, with its command, exit code and logs kept in the run directory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from k4opt.ddsim.command import ddsim_arguments, option_args, particle_gun_args
from k4opt.ddsim.errors import DdsimError
from k4opt.frozen import Json

_LOG_TAIL = 15  # lines of standard error quoted in a failure message


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """What one ``ddsim`` invocation left behind.

    ``error`` is ``None`` only if ``ddsim`` exited with status 0 and wrote ``output``; otherwise
    it says what went wrong. ``returncode`` is ``None`` if the process never ran. The files
    ``stdout``, ``stderr``, ``command`` and ``output`` exist only if the process got that far.
    """

    argv: tuple[str, ...]
    cwd: Path
    returncode: int | None
    output: Path
    stdout: Path
    stderr: Path
    command: Path
    error: str | None

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
    sample: Path | None = None,
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
        sample=sample,
        options=options,
    )


def _tail(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-_LOG_TAIL:])


def simulate(
    *,
    compact: Path,
    workdir: Path,
    events: int,
    seed: int | None,
    particle_gun: Json = None,
    sample: Path | None = None,
    options: Mapping[str, Json],
) -> SimulationRun:
    """Run the ``ddsim`` found on ``PATH`` on ``compact``, in the new directory ``workdir``.

    The command is an argument list without a shell: the options k4opt manages (geometry,
    output, event count, seed, particle gun or input sample) followed by ``options`` in order
    (see :func:`~k4opt.ddsim.command.option_args`). Relative ``compact`` and ``sample`` paths
    are taken from the current directory; option values are passed unchanged and ddsim runs in
    ``workdir``, so a file named in an option (``physics.pdgfile``) must be absolute.
    ``workdir`` receives ``events.edm4hep.root``, ``ddsim.stdout.log``, ``ddsim.stderr.log``
    and ``ddsim.json`` (command, directory, exit code). Raises :class:`DdsimError` for
    unsupported settings or options or an existing ``workdir``.
    """
    # ddsim runs inside workdir, so every path in its command line must be absolute
    compact = Path(os.path.abspath(compact))
    sample = None if sample is None else Path(os.path.abspath(sample))
    workdir = Path(os.path.abspath(workdir))
    output = workdir / "events.edm4hep.root"
    stdout, stderr, command = (
        workdir / name for name in ("ddsim.stdout.log", "ddsim.stderr.log", "ddsim.json")
    )
    arguments = ddsim_arguments(
        compact=compact,
        output=output,
        events=events,
        seed=seed,
        particle_gun=particle_gun,
        sample=sample,
        options=options,
    )
    try:
        workdir.mkdir(parents=True)
    except FileExistsError as exc:
        raise DdsimError(f"{workdir} already exists; use a new directory") from exc

    def run(argv: tuple[str, ...], returncode: int | None, error: str | None) -> SimulationRun:
        return SimulationRun(argv, workdir, returncode, output, stdout, stderr, command, error)

    found = shutil.which("ddsim")
    if found is None:
        return run((), None, "ddsim not found on PATH (is the Key4hep environment set up?)")
    # a relative PATH entry would no longer resolve from inside workdir
    argv = (os.path.abspath(found), *arguments)
    try:
        with stdout.open("wb") as out, stderr.open("wb") as err:
            completed = subprocess.run(
                list(argv),
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                shell=False,
                check=False,
            )
    except OSError as exc:
        return run(argv, None, f"ddsim could not be started: {exc}")
    command.write_text(
        json.dumps(
            {"argv": argv, "cwd": str(workdir), "returncode": completed.returncode}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        error = f"ddsim exited with status {completed.returncode} (logs in {workdir}):\n" + _tail(
            stderr
        )
        return run(argv, completed.returncode, error)
    if not output.is_file():
        return run(argv, 0, f"ddsim finished but wrote no {output.name} (logs in {workdir})")
    return run(argv, 0, None)
