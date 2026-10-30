"""What ``k4opt optimize`` prints: a line per trial, the study, and a progress bar on a terminal."""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from k4opt.cli import main
from k4opt.optimization import TrialReport
from k4opt.optimization._render import BAR_WIDTH, Progress, trial_line
from tests.support.evaluators import TRADE_OFF

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def screen(text: str) -> list[str]:
    """The lines a terminal shows for ``text``: a cleared line is gone, colors are left out."""
    return [line.rsplit("\r\x1b[2K", 1)[-1] for line in ANSI.sub("", text).split("\n")]


def trial(number: int, status: str = "valid", **fields: Any) -> TrialReport:
    if status == "valid":
        fields = {"feasible": True, "objectives": {"analysis/a": 1 / 3}, **fields}
    return TrialReport(
        number, Path(f"trial-{number:06d}"), {"constants.x": 498.78582224598927}, status, **fields
    )


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def progress(
    trials: int, clock: Clock, out: io.StringIO | None = None, *, color: bool = False
) -> Progress:
    out = out if out is not None else Terminal()
    return Progress(trials, out, out, bar=True, color=color, interval=3600, clock=clock)


def test_a_trial_is_one_line_and_its_values_are_shown_to_6_significant_digits() -> None:
    assert trial_line(trial(0)) == (
        "trial 0:   valid    feasible    constants.x=498.786  analysis/a=0.333333"
    )
    failed = trial(12, "failed", record="r", message="ddsim exited with status 1")
    assert trial_line(failed) == (
        "trial 12:  failed   ddsim exited with status 1  trial-000012/summary.json"
    )
    for line in (trial(0), trial(1, feasible=False), failed, trial(3, "interrupted")):
        colored = trial_line(line, color=True)
        assert colored != trial_line(line) and ANSI.sub("", colored) == trial_line(line)


def test_without_the_bar_only_the_lines_are_printed() -> None:
    out, err = io.StringIO(), io.StringIO()
    with Progress(2, out, err, bar=False, color=False) as shown:
        shown.warning("careful")
        shown.trial(trial(0))
    assert (out.getvalue(), err.getvalue()) == (
        trial_line(trial(0)) + "\n",
        "k4opt: warning: careful\n",
    )


def test_the_bar_shows_the_share_of_trials_that_ended_and_the_time() -> None:
    clock = Clock()
    shown = progress(30, clock)
    assert shown.line(140).split()[2:] == [
        "0%",
        "0/30",
        "trials",
        "0:00",
        "elapsed",
        "--:--",
        "left",
    ]
    clock.now = 312.0  # 12 s of setup and a trial of 300 s
    shown.trial(trial(0))
    line = shown.line(140)
    assert "  3%  1/30 trials  5:12 elapsed  ~2:30:48 left  5:12/trial  1 feasible" in line
    clock.now = 3600 + 312.0
    assert "1:05:12 elapsed" in shown.line(140)


def test_a_trial_takes_the_mean_time_without_the_setup_and_the_time_left_counts_down() -> None:
    clock = Clock()
    shown = progress(10, clock)
    for clock.now in (100.0, 160.0, 240.0):  # the first trial holds the setup
        shown.trial(trial(0))
    assert shown.per_trial == 70.0
    assert shown.left == 7 * 70.0
    clock.now = 260.0  # the running trial has had 20 s
    assert shown.left == 50.0 + 6 * 70.0 and "~7:50 left" in shown.line(140)
    clock.now = 400.0  # it runs longer than a trial takes: the others are still to come
    assert shown.left == 6 * 70.0
    for clock.now in (410.0, 480.0, 550.0, 620.0, 690.0, 760.0, 830.0):
        shown.trial(trial(0))
    assert shown.left == 0.0 and " 100%  10/10 trials" in shown.line(140)


def test_the_percentage_reaches_100_only_with_the_last_trial() -> None:
    clock = Clock()
    shown = progress(200, clock)
    for _ in range(199):
        shown.trial(trial(0))
    assert "  99%  199/200 trials" in shown.line(140)


def test_recovered_trials_are_printed_but_do_not_advance_the_bar() -> None:
    clock, out = Clock(), Terminal()
    with progress(3, clock, out) as shown:
        clock.now = 5.0
        shown.trial(trial(7, "interrupted", recovered=True))
        clock.now = 65.0
        shown.trial(trial(8))
    assert shown.durations == [60.0] and "1/3 trials" in shown.line(140)
    assert screen(out.getvalue())[0].startswith("trial 7:   interrupted")


def test_the_bar_fits_the_terminal_and_keeps_its_width_as_parts_come_and_go() -> None:
    clock = Clock()
    shown = progress(30, clock, color=True)
    widths = set()
    for status in ("valid", "valid", "invalid", "failed", "interrupted", "error"):
        clock.now += 299.0
        shown.trial(trial(0, status, feasible=status == "valid" and clock.now < 500))
        for columns in range(10, 200):
            assert len(ANSI.sub("", shown.line(columns))) <= columns - 1
        line = ANSI.sub("", shown.line(120))
        widths.add(len(line.split()[1]))
    assert widths == {BAR_WIDTH}
    narrow = ANSI.sub("", shown.line(40)).split()
    assert narrow[1:] == ["20%", "6/30", "trials", "~1:59:36", "left"]  # no room for a bar


def test_the_bar_is_left_marked_when_the_study_ends_and_lines_go_above_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "150")
    clock, out = Clock(), Terminal()
    with progress(2, clock, out, color=True) as shown:
        shown.warning("careful")
        shown.trial(trial(0))
        shown.trial(trial(1, "failed", message="no"))
    lines = screen(out.getvalue())
    assert lines[0] == "k4opt: warning: careful"
    assert lines[1].startswith("trial 0:") and lines[2].startswith("trial 1:")
    assert lines[3].startswith("✔ ━━━") and "100%  2/2 trials" in lines[3]
    assert "left" not in lines[3] and "1 feasible  1 failed" in lines[3] and lines[4] == ""

    out = Terminal()

    def interrupted() -> None:
        with progress(2, clock, out) as shown:
            shown.trial(trial(0))
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    last = screen(out.getvalue())[-2]
    assert last.startswith("✘ ━━━━━━━━━━━━━━━━━━━━─") and "50%  1/2 trials" in last


def test_on_a_terminal_the_command_draws_the_bar_and_colors_the_output(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("optuna")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "150")
    monkeypatch.delenv("NO_COLOR", raising=False)
    problem = write_problem()

    def optimize(output: str) -> tuple[int, str]:
        out = Terminal()
        argv = ["optimize", str(problem), "--evaluator", TRADE_OFF, "--trials", "2"]
        code = main([*argv, "--output", str(tmp_path / output)], stdout=out, stderr=io.StringIO())
        return code, out.getvalue()

    code, text = optimize("color")
    assert code == 0 and "\x1b[32mvalid" in text
    lines = screen(text)
    assert lines[2].startswith("✔ ") and "100%  2/2 trials" in lines[2]
    assert lines[3].startswith("study:    trade-")

    monkeypatch.setenv("NO_COLOR", "1")
    code, text = optimize("plain")
    assert code == 0 and not ANSI.search(text) and "\r\x1b[2K" in text

    monkeypatch.setenv("TERM", "dumb")
    code, text = optimize("dumb")
    assert code == 0 and "\x1b" not in text and "\r" not in text and len(text.splitlines()) == 5
