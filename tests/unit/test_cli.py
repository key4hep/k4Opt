"""Tests for the top-level ``k4opt`` command (dispatch, version, exit codes)."""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from k4opt import __version__
from k4opt.cli import build_parser, main


def test_help_lists_component_command_groups() -> None:
    help_text = build_parser().format_help()
    assert "geometry" in help_text


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as info:
        main([], stdout=io.StringIO(), stderr=io.StringIO())
    assert info.value.code == 2


def test_expected_errors_exit_1_with_a_prefixed_message() -> None:
    err = io.StringIO()
    code = main(["geometry", "inspect", "/does/not/exist.xml"], stdout=io.StringIO(), stderr=err)
    assert code == 1
    assert err.getvalue().startswith("k4opt: error: ")


def test_unexpected_errors_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("a genuine bug")

    monkeypatch.setattr("k4opt.geometry.cli.GeometrySource", boom)
    with pytest.raises(RuntimeError, match="a genuine bug"):
        main(["geometry", "inspect", "x.xml"], stdout=io.StringIO(), stderr=io.StringIO())


def test_module_entry_point_and_version() -> None:
    help_run = subprocess.run(
        [sys.executable, "-m", "k4opt", "--help"], capture_output=True, text=True, check=False
    )
    assert help_run.returncode == 0
    assert "geometry" in help_run.stdout
    version_run = subprocess.run(
        [sys.executable, "-m", "k4opt", "--version"], capture_output=True, text=True, check=False
    )
    assert version_run.returncode == 0
    assert version_run.stdout.strip() == f"k4opt {__version__}"
