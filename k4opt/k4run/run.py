"""Run a ``k4run`` job once, from a copy of its directory, with its command, exit code, logs and
resolved options kept in the stage's directory.

The stage's directory holds what k4opt keeps of the run (the output ``events.edm4hep.root``, the
logs, ``k4run.json``, ``joboptions.txt`` and k4opt's own options file ``k4opt.opts.py``) and the
job's working directory ``job/``: a writable copy of the stage's ``directory`` (a reconstruction
such as CLD's runs only from its own directory: its Pandora settings name files relative to it),
or an empty directory, where whatever else the job writes (histograms, ``cdb.log``) stays.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from k4opt import program
from k4opt.frozen import Json
from k4opt.k4run.command import check_options, k4run_arguments, wiring
from k4opt.k4run.errors import K4runError
from k4opt.k4run.joboptions import verify

OUTPUT = "events.edm4hep.root"
"""The file the job writes (``IOSvc.Output``), in the stage's directory."""

JOB = "job"
"""The job's working directory, inside the stage's."""

LOG_FILES: Mapping[str, str] = {
    "stdout": "k4run.stdout.log",
    "stderr": "k4run.stderr.log",
    "command": "k4run.json",
    "joboptions": "joboptions.txt",
    "options": "k4opt.opts.py",
}
"""The files besides its output that a run keeps in the stage's directory, by role: the logs, the
command record (written once k4run ended), the job's resolved options (written by Gaudi as the job
starts) and k4opt's options file, which has Gaudi write them."""

DRY_RUN_TIMEOUT = 300.0
"""Seconds a dry run (:func:`dry_run`) may take: the job's options files run, nothing else."""


@dataclass(frozen=True, slots=True)
class ReconstructionRun:
    """What one ``k4run`` invocation left behind.

    ``error`` is ``None`` only if ``k4run`` exited with status 0, wrote ``output`` and its
    resolved options show that it read the geometry and the input k4opt gave it and wrote
    ``output``; otherwise it says what went wrong. ``returncode`` is ``None`` if the process never
    ran, or was killed because it ran out of time. ``files`` are those of :data:`LOG_FILES` that
    exist, by role, and ``edm4hep``, the output, if it exists. ``peak_memory_bytes`` is that of
    k4run (:attr:`k4opt.program.ProgramRun.peak_memory_bytes`).
    """

    argv: tuple[str, ...]
    cwd: Path
    returncode: int | None
    output: Path
    files: Mapping[str, Path]
    error: str | None
    peak_memory_bytes: int | None = None


def job_files(directory: Path) -> dict[str, Path]:
    """Every file below ``directory`` by its path relative to it (``PandoraSettingsCLD/...``),
    sorted, as the job's copy of it holds them: links followed, as copying does."""
    found: dict[str, Path] = {}
    for root, dirs, names in os.walk(directory, followlinks=True):
        dirs.sort()
        for name in names:
            path = Path(root) / name
            found[path.relative_to(directory).as_posix()] = path
    return dict(sorted(found.items()))


def _make_writable(root: Path) -> None:
    """Give the owner write permission throughout ``root``: a copy keeps the modes of its source,
    so a copy of a read-only tree (CVMFS) would not let the job write next to its own files."""
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _stage(workdir: Path, directory: Path | None) -> Path:
    """Make the new directory ``workdir`` with the job's working directory in it; raises
    :class:`K4runError` if ``workdir`` exists, :class:`OSError` if it cannot be made."""
    workdir = Path(os.path.abspath(workdir))
    try:
        workdir.mkdir(parents=True)
    except FileExistsError as exc:
        raise K4runError(f"{workdir} already exists; use a new directory") from exc
    job = workdir / JOB
    if directory is None:
        job.mkdir()
    else:
        shutil.copytree(directory, job)
        _make_writable(job)
    return job


def _run_files(files: Sequence[Path], directory: Path | None) -> list[str]:
    """The options files as k4run is given them: inside the job's copy of ``directory`` by their
    path relative to it, otherwise by their absolute path."""
    if directory is None:
        return [os.path.abspath(file) for file in files]
    return [
        Path(os.path.abspath(file)).relative_to(os.path.abspath(directory)).as_posix()
        for file in files
    ]


def _dump_options(dump: Path) -> str:
    """k4opt's options file: Gaudi writes the job's resolved options to ``dump``."""
    return (
        "# written by k4opt, loaded after the job's options files: Gaudi writes the job's\n"
        "# resolved options, which k4opt checks and keeps\n"
        "from Configurables import JobOptionsSvc\n\n"
        f"JobOptionsSvc().DUMPFILE = {str(dump)!r}\n"
    )


def _placeholder_options(placeholder: Path) -> str:
    """k4opt's options file for a dry run: it writes ``placeholder``, an EDM4hep file of one empty
    event, before k4FWCore checks the input (it must exist, and a multithreaded job reads its
    first event to learn the collections), unless it exists."""
    if placeholder.is_file():
        return "# written by k4opt for a dry run, whose input of one empty event exists\n"
    return (
        "# written by k4opt for a dry run: an input of one empty event, which k4FWCore checks\n"
        "import podio\n"
        "from podio import root_io\n\n"
        f"_writer = root_io.Writer({str(placeholder)!r})\n"
        '_writer.write_frame(podio.Frame(), "events")\n'
        "del _writer\n"
    )


def reconstruct(
    *,
    files: Sequence[Path],
    directory: Path | None,
    compact: Path,
    input_file: Path,
    workdir: Path,
    options: Mapping[str, Json],
    timeout: float | None = None,
    started: Callable[[int], None] | None = None,
    die_with_parent: bool = False,
) -> ReconstructionRun:
    """Run the ``k4run`` found on ``PATH`` on the job's options ``files`` (absolute, inside
    ``directory`` if it is given), reconstructing every event of ``input_file`` with the geometry
    ``compact``, in the new directory ``workdir``.

    ``options`` are the stage's (:data:`~k4opt.k4run.command.EXECUTOR_KEYS` and the job's
    arguments, see :func:`~k4opt.k4run.command.k4run_arguments`). The job runs in ``workdir/job``,
    a writable copy of ``directory`` or an empty directory; ``workdir`` receives
    ``events.edm4hep.root`` and the files of :data:`LOG_FILES`. A ``k4run`` that runs longer than
    ``timeout`` seconds is killed with every process it started, and the run fails with its logs
    kept; ``started`` and ``die_with_parent`` are as for :func:`k4opt.program.run`. After the run,
    the job's resolved options must show that it read ``compact`` and ``input_file`` and wrote
    the output (:func:`~k4opt.k4run.verify`). Raises :class:`K4runError` for unsupported options,
    a ``timeout`` that is not a positive number of seconds, or an existing ``workdir``.
    """
    check_options(options)
    problem = program.timeout_problem(timeout)
    if problem is not None:
        raise K4runError(problem)
    workdir = Path(os.path.abspath(workdir))
    compact, input_file = Path(os.path.abspath(compact)), Path(os.path.abspath(input_file))
    output = workdir / OUTPUT
    paths = {role: workdir / name for role, name in LOG_FILES.items()}

    def made(
        argv: tuple[str, ...], returncode: int | None, error: str | None, peak: int | None = None
    ) -> ReconstructionRun:
        present = {role: path for role, path in paths.items() if path.is_file()}
        if output.is_file():
            present["edm4hep"] = output
        return ReconstructionRun(argv, workdir / JOB, returncode, output, present, error, peak)

    try:
        job = _stage(workdir, directory)
        paths["options"].write_text(_dump_options(paths["joboptions"]))
    except OSError as exc:
        return made((), None, f"cannot prepare the reconstruction in {workdir}: {exc}")
    arguments = k4run_arguments(
        files=_run_files(files, directory),
        k4opt_options=paths["options"],
        compact=compact,
        input_file=input_file,
        output=output,
        options=options,
    )
    run = program.run(
        "k4run",
        arguments,
        workdir=workdir,
        cwd=job,
        logs=LOG_FILES,
        timeout=timeout,
        started=started,
        die_with_parent=die_with_parent,
    )
    error = run.error
    if error is None:
        geometry, input_argument = wiring(options)
        try:
            dumped = paths["joboptions"].read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            error = f"k4run finished but its resolved options cannot be read: {exc}"
        else:
            error = verify(
                dumped,
                compact=compact,
                input_file=input_file,
                output=output,
                geometry=geometry,
                input_argument=input_argument,
            )
        if error is None and not output.is_file():
            error = f"k4run finished but wrote no {output.name} (logs in {workdir})"
    return made(run.argv, run.returncode, error, run.peak_memory_bytes)


def dry_run(
    *,
    files: Sequence[Path],
    directory: Path | None,
    compact: Path,
    workdir: Path,
    options: Mapping[str, Json],
    placeholder: Path | None = None,
    timeout: float = DRY_RUN_TIMEOUT,
) -> str | None:
    """Why the job does not take its arguments, ``None`` if it does: ``k4run --dry-run`` with the
    arguments :func:`reconstruct` would give it, from a copy of ``directory`` in the new
    directory ``workdir``, on the geometry ``compact`` and the input ``placeholder``
    (``workdir/placeholder.edm4hep.root`` if ``None``), an EDM4hep file of one empty event that
    the job writes first unless it exists: dry runs that share one write it once. The job's
    options files run and every argument is parsed, strictly; nothing is reconstructed. Raises
    :class:`K4runError` as :func:`reconstruct` does."""
    check_options(options)
    workdir = Path(os.path.abspath(workdir))
    if placeholder is None:
        placeholder = workdir / "placeholder.edm4hep.root"
    placeholder = Path(os.path.abspath(placeholder))
    try:
        job = _stage(workdir, directory)
        mine = workdir / LOG_FILES["options"]
        mine.write_text(_placeholder_options(placeholder))
    except OSError as exc:
        return f"cannot prepare the dry run in {workdir}: {exc}"
    arguments = k4run_arguments(
        files=_run_files(files, directory),
        k4opt_options=mine,
        compact=Path(os.path.abspath(compact)),
        input_file=placeholder,
        output=workdir / OUTPUT,
        options=options,
        dry_run=True,
    )
    run = program.run("k4run", arguments, workdir=workdir, cwd=job, logs=LOG_FILES, timeout=timeout)
    if run.error is None:
        return None
    if run.returncode == 2:  # argparse: the last line says which argument
        lines = program.tail(run.stderr).splitlines()
        said = lines[-1] if lines else run.error
        if "unrecognized arguments: --IOSvc." in said:
            said += (
                " (the job does not use k4FWCore's IOSvc, through which k4opt gives it its input "
                "and output)"
            )
        elif "unrecognized arguments: --GeoSvc." in said:
            said += (
                " (the job has no GeoSvc; name the argument it takes the geometry by with "
                "geometry:)"
            )
        return f"the job does not take its arguments: {said}"
    return f"the job's options files fail to run: {run.error}"
