"""What identifies a program of the stack (``k4opt.program.identify``): its executable, and the
interpreter that runs it, as ``PATH`` finds it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from k4opt.program import identify


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
