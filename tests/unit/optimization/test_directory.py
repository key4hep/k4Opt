"""Where a new study is made: a directory that does not exist or is empty, also through a symlink;
one in use, or a symlink to nothing, is refused before anything is made."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from k4opt.optimization import OptimizationError, optimize
from tests.support.evaluators import trade_off
from tests.unit.optimization.conftest import GRID


def test_a_directory_in_use_is_refused_and_left_as_it_is(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "notes.txt").write_text("mine")
    with pytest.raises(OptimizationError, match="is in use; resume the study in it"):
        optimize(
            write_problem(), [trade_off()], tmp_path / "used", trials=1, strategy="grid", grid=GRID
        )
    assert [p.name for p in (tmp_path / "used").iterdir()] == ["notes.txt"]


def test_a_dangling_symlink_is_refused(tmp_path: Path, write_problem: Callable[..., Path]) -> None:
    (tmp_path / "s").symlink_to(tmp_path / "gone")
    with pytest.raises(OptimizationError, match="is in use; resume the study in it"):
        optimize(
            write_problem(), [trade_off()], tmp_path / "s", trials=1, strategy="grid", grid=GRID
        )
    assert (tmp_path / "s").is_symlink() and not (tmp_path / "gone").exists()


def test_a_symlink_to_an_empty_directory_holds_a_new_study(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "s").symlink_to(tmp_path / "real")
    report = optimize(
        write_problem(), [trade_off()], tmp_path / "s", trials=1, strategy="grid", grid=GRID
    )
    assert [t.status for t in report.trials] == ["valid"]
    assert (tmp_path / "real" / "study.json").is_file()
    assert (tmp_path / "real" / "trials.jsonl").is_file()
