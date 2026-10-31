"""Running ddsim: paths, logs and failure reporting, with a fake process."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import DdsimError, simulate
from k4opt.ddsim import run as ddsim_run
from k4opt.frozen import freeze
from tests.support import simulation
from tests.unit.ddsim.test_command import GENERATOR


def test_an_unwritable_workdir_is_a_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    run = simulate(
        compact=tmp_path / "c.xml",
        workdir=blocker / "run",
        events=1,
        seed=None,
        input_file=tmp_path / "events.hepmc",
        options={},
    )
    assert run.returncode is None and "cannot create simulation directory" in run.error
    assert fake.calls == []


@pytest.mark.parametrize("returncode", [0, 2])
def test_a_command_record_write_failure_keeps_the_process_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    simulation.install(monkeypatch, returncode=returncode)

    def denied(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("command is not writable")

    monkeypatch.setattr(ddsim_run, "write_json", denied)
    run = simulate(
        compact=tmp_path / "c.xml",
        workdir=tmp_path / "run",
        events=1,
        seed=None,
        input_file=tmp_path / "events.hepmc",
        options={},
    )
    assert run.returncode == returncode
    assert "cannot record ddsim command" in run.error
    if returncode:
        assert "ddsim exited with status 2" in run.error
    assert run.files["edm4hep"].is_file()
    assert "command" not in run.files


def test_error_tail_streams_the_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "stderr.log"
    path.write_bytes(b"discarded\n" * 100_000 + b"\xff\nlast line\n")

    def no_read_text(*args: Any, **kwargs: Any) -> str:
        pytest.fail("the whole log must not be read into memory")

    monkeypatch.setattr(Path, "read_text", no_read_text)
    lines = ddsim_run._tail(path).splitlines()
    assert len(lines) == 15 and lines[-2:] == ["\ufffd", "last line"]


def generator() -> object:
    return freeze(GENERATOR, "generator", DdsimError)


def test_relative_paths_are_made_absolute_before_ddsim_runs_in_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    monkeypatch.chdir(tmp_path)
    Path("compact.xml").write_text("<lccdd/>")

    run = simulate(
        compact=Path("compact.xml"),
        workdir=Path("run"),
        events=1,
        seed=1,
        particle_gun=generator(),  # type: ignore[arg-type]
        options={},
    )

    ((argv, kwargs),) = fake.calls
    assert f"--compactFile={tmp_path / 'compact.xml'}" in argv
    assert f"--outputFile={tmp_path / 'run' / 'events.edm4hep.root'}" in argv
    assert not any(a.startswith(("--compactFile=compact", "--outputFile=run")) for a in argv)
    assert kwargs["cwd"] == tmp_path / "run"
    assert run.error is None
    assert run.output.is_file()
    assert os.path.isabs(run.output)


def test_a_relative_executable_found_on_path_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    monkeypatch.setattr(ddsim_run.shutil, "which", lambda name: "bin/ddsim")  # PATH=bin:...
    monkeypatch.chdir(tmp_path)
    simulate(
        compact=tmp_path / "c.xml",
        workdir=Path("run"),
        events=1,
        seed=None,
        particle_gun=generator(),  # type: ignore[arg-type]
        options={},
    )
    ((argv, _),) = fake.calls
    assert argv[0] == str(tmp_path / "bin" / "ddsim")


def test_an_existing_workdir_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = simulation.install(monkeypatch)
    with pytest.raises(DdsimError, match="already exists"):
        simulate(
            compact=tmp_path / "c.xml",
            workdir=tmp_path,
            events=1,
            seed=None,
            particle_gun=generator(),  # type: ignore[arg-type]
            options={},
        )
    assert fake.calls == []


def test_an_input_file_is_passed_as_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run = simulate(
        compact=tmp_path / "c.xml",
        workdir=Path("run"),
        events=1,
        seed=None,
        input_file=Path("zh.hepmc"),
        options={},
    )
    ((argv, _),) = fake.calls
    assert f"--inputFiles={tmp_path / 'zh.hepmc'}" in argv
    assert not any(a.startswith("--gun") for a in argv)
    assert run.error is None


# what a wrapper script around ddsim does when it does not ``exec``: run it as its own child
WRAPPER = """\
#!/bin/sh
echo 'G4 stuck in event 3' >&2
sleep 60 &
echo $! > child.pid
wait
"""


@pytest.fixture
def wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A ``ddsim`` wrapper that never ends; the file its child writes its process ID to."""
    ddsim = tmp_path / "bin" / "ddsim"
    ddsim.parent.mkdir()
    ddsim.write_text(WRAPPER)
    ddsim.chmod(0o755)
    monkeypatch.setattr(ddsim_run.shutil, "which", lambda name: str(ddsim))
    child = tmp_path / "run" / "child.pid"
    yield child
    if child.is_file():  # never left behind, whatever the test found
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(child.read_text()), signal.SIGKILL)


def _ended(pid: int) -> bool:
    """Whether process ``pid`` ends within 10 s; a zombie its new parent has not reaped has."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


def _simulate(tmp_path: Path, timeout: float | None) -> ddsim_run.SimulationRun:
    return simulate(
        compact=tmp_path / "c.xml",
        workdir=tmp_path / "run",
        events=1,
        seed=None,
        input_file=tmp_path / "events.hepmc",
        options={},
        timeout=timeout,
    )


def test_a_ddsim_that_runs_out_of_time_is_killed_with_what_it_started(
    tmp_path: Path, wrapped: Path
) -> None:
    started = time.monotonic()
    run = _simulate(tmp_path, timeout=0.5)
    assert time.monotonic() - started < 30
    assert run.returncode is None and run.error is not None
    assert run.error.startswith("ddsim ran longer than 0.5 s and was killed")
    assert run.error.endswith("G4 stuck in event 3")
    assert json.loads(run.files["command"].read_text())["timeout"] == 0.5
    assert {"stdout", "stderr", "command"} <= run.files.keys()
    assert _ended(int(wrapped.read_text())), "the wrapper's child outlived it"


def test_an_interrupted_ddsim_is_killed_with_what_it_started(tmp_path: Path, wrapped: Path) -> None:
    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt  # what Ctrl-C, or SIGTERM in k4opt optimize, raises

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        with pytest.raises(KeyboardInterrupt):
            _simulate(tmp_path, timeout=None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert _ended(int(wrapped.read_text())), "the wrapper's child outlived it"


@pytest.mark.parametrize("timeout", [0, -1.0, float("nan"), float("inf"), True])
def test_a_timeout_that_is_no_time_limit_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: float
) -> None:
    fake = simulation.install(monkeypatch)
    with pytest.raises(DdsimError, match="timeout must be a positive number of seconds"):
        _simulate(tmp_path, timeout=timeout)
    assert not (tmp_path / "run").exists() and fake.calls == []
