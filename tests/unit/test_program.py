"""What identifies a program of the stack (``k4opt.program.identify``): its executable, and the
interpreter that runs it, as ``PATH`` finds it."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from k4opt import program
from k4opt.program import identify, run
from tests.support.common import set_beneath_python


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def test_a_script_run_by_env_names_the_interpreter_path_finds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _executable(tmp_path / "bin" / "ddsim", "#!/usr/bin/env python3\nprint()\n")
    first = _executable(tmp_path / "one" / "python3", "#!/bin/sh\necho one\n")
    other = _executable(tmp_path / "two" / "python3", "#!/bin/sh\necho two\n")
    monkeypatch.setenv("PATH", f"{first.parent}{os.pathsep}{other.parent}")
    found = identify(script)
    assert found is not None and found["path"] == str(script)
    interpreter = found["interpreter"]
    assert isinstance(interpreter, dict) and interpreter["path"] == str(first)
    # what else PATH holds plays no part; which python3 it finds first does
    monkeypatch.setenv("PATH", f"{first.parent}{os.pathsep}{other.parent}{os.pathsep}/nowhere")
    assert identify(script) == found
    monkeypatch.setenv("PATH", f"{other.parent}{os.pathsep}{first.parent}")
    moved = identify(script)
    assert moved is not None and moved["interpreter"] != interpreter


def test_a_link_to_the_interpreter_keeps_the_path_path_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a virtual environment's python3 is a link, and finds its packages from where it is
    real = _executable(tmp_path / "python" / "bin" / "python3.13", "#!/bin/sh\n")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python3").symlink_to(real)
    script = _executable(tmp_path / "bin" / "k4run", "#!/usr/bin/env python3\n")
    monkeypatch.setenv("PATH", str(venv))
    found = identify(script)
    assert found is not None and found["interpreter"]["path"] == str(venv / "python3")  # type: ignore[index]


def test_an_absolute_interpreter_and_a_program_that_is_no_script(tmp_path: Path) -> None:
    shell = _executable(tmp_path / "sh", "binary")
    script = _executable(tmp_path / "run", f"#!{shell} -e\n")
    found = identify(script)
    assert found is not None and found["interpreter"]["path"] == str(shell)  # type: ignore[index]
    binary = tmp_path / "binary"
    binary.write_bytes(b"\x7fELF\x02\x01")
    assert identify(binary) == {**_bytes(binary), "interpreter": None}
    assert identify(tmp_path / "missing") is None


def _bytes(path: Path) -> dict[str, object]:
    import hashlib

    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_a_script_whose_interpreter_is_not_found_is_not_identified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _executable(tmp_path / "bin" / "ddsim", "#!/usr/bin/env python3\n")
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert identify(script) is None  # what would run is not known
    assert identify(_executable(tmp_path / "empty", "#!\n")) is None


def test_a_program_runs_with_what_k4opt_identified_not_what_a_library_set_beneath_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "show"
    script.write_text('#!/bin/sh\necho "${K4OPT_BENEATH-unset} ${K4OPT_ABOVE-unset}"\n')
    script.chmod(0o755)
    monkeypatch.setenv("K4OPT_ABOVE", "seen")
    logs = {"stdout": "out", "stderr": "err", "command": "command.json"}
    with set_beneath_python("K4OPT_BENEATH", "unseen"):
        ran = run("show", [], workdir=tmp_path, logs=logs, executable=script)
    assert ran.error is None
    assert (tmp_path / "out").read_text() == "unset seen\n"


LOGS = {"stdout": "out.log", "stderr": "err.log", "command": "command.json"}
traceable = pytest.mark.skipif(
    program.unobservable("files") is not None, reason="strace cannot trace here"
)


@traceable
@pytest.mark.parametrize("observe", ["none", "libraries", "files"])
@pytest.mark.parametrize(("script", "status"), [("exit 3", 3), ("kill -SEGV $$", -11)])
def test_a_program_ends_as_it_ends_however_it_is_observed(
    tmp_path: Path, observe: program.Observation, script: str, status: int
) -> None:
    done = run("sh", ["-c", script], workdir=tmp_path, logs=LOGS, observe=observe)
    assert done.returncode == status and done.error is not None
    assert done.observed == observe
    assert json.loads(done.command.read_text())["observed"] == observe


@traceable
def test_a_crash_under_strace_says_how_to_get_its_backtrace(tmp_path: Path) -> None:
    done = run("sh", ["-c", "kill -SEGV $$"], workdir=tmp_path, logs=LOGS, observe="files")
    assert done.error is not None and "observe it at the libraries level" in done.error


@traceable
def test_an_observed_program_out_of_time_leaves_nothing_running(tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    script = f"sleep 300 & echo $! > {pids}; wait"
    done = run("sh", ["-c", script], workdir=tmp_path, logs=LOGS, timeout=1.0, observe="files")
    assert done.returncode is None and done.error is not None
    child = int(pids.read_text())
    for _ in range(200):
        if not Path(f"/proc/{child}").exists():
            break
        time.sleep(0.01)
    assert not Path(f"/proc/{child}").exists()


def test_the_loader_says_what_it_loaded_into_the_trace_directory(tmp_path: Path) -> None:
    done = run("true", [], workdir=tmp_path, logs=LOGS, observe="libraries")
    assert done.trace == tmp_path / program.TRACE
    assert any(path.name.startswith("ld.") for path in done.trace.iterdir())


def test_a_run_unobserved_leaves_no_trace_directory(tmp_path: Path) -> None:
    done = run("true", [], workdir=tmp_path, logs=LOGS)
    assert done.trace is None and done.observed == "none"
    assert not (tmp_path / program.TRACE).exists()


def test_a_trace_directory_that_cannot_be_made_runs_the_program_unobserved(tmp_path: Path) -> None:
    (tmp_path / program.TRACE).write_text("in the way")
    done = run("true", [], workdir=tmp_path, logs=LOGS, observe="libraries")
    assert done.error is None and done.observed == "none" and done.trace is None


def test_strace_gone_from_path_runs_the_program_unobserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    true = _executable(tmp_path / "bin" / "true", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", str(true.parent))
    work = tmp_path / "work"
    work.mkdir()
    done = run("true", [], workdir=work, logs=LOGS, observe="files")
    assert done.error is None and done.observed == "none"
    assert not (work / program.TRACE).exists()


def test_no_strace_is_why_files_cannot_be_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert "strace is not installed" in str(program.unobservable("files"))
    assert program.unobservable("libraries") is None
    assert program.unobservable("none") is None


def test_a_strace_that_may_not_trace_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _executable(
        tmp_path / "strace",
        "#!/bin/sh\necho 'PTRACE_TRACEME: Operation not permitted' >&2\nexit 1\n",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert "Operation not permitted" in str(program.unobservable("files"))


def test_a_strace_that_traced_nothing_is_no_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _executable(tmp_path / "strace", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert "wrote no trace" in str(program.unobservable("files"))
