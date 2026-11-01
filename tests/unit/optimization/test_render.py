"""What ``k4opt optimize`` prints: a line per trial, the study, and a progress bar on a terminal."""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from k4opt.cli import main
from k4opt.frozen import FrozenMap
from k4opt.optimization import StudyResults, TrialReport, TrialResult
from k4opt.optimization._render import BAR_WIDTH, Progress, trial_line
from k4opt.optimization._show import render_results, render_trial
from k4opt.problem import Activity, Assessment, Limit, ObjectiveValue, parse_problem
from tests.support.evaluators import TRADE_OFF
from tests.unit.optimization.conftest import RADIUS, problem_document

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def screen(text: str) -> list[str]:
    """The lines a terminal shows for ``text``: a cleared line is gone, colors are left out."""
    return [line.rsplit("\r\x1b[2K", 1)[-1] for line in ANSI.sub("", text).split("\n")]


def trial(number: int, status: str = "valid", **fields: Any) -> TrialReport:
    if status == "valid":
        fields = {"feasible": True, "objectives": {"test.trade_off/analysis/a": 1 / 3}, **fields}
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
        "trial 0:   valid    feasible    constants.x=498.786  test.trade_off/analysis/a=0.333333"
    )
    failed = trial(12, "failed", record="r", message="ddsim exited with status 1")
    assert trial_line(failed) == (
        "trial 12:  failed   ddsim exited with status 1  trial-000012/summary.json"
    )
    recovered = trial(4, "interrupted", recovered=True)
    assert trial_line(recovered) == "trial 4:   interrupted  trial-000004  (recovered)"
    assert trial_line(trial(5, recovered=True)).endswith(
        "test.trade_off/analysis/a=0.333333  (recovered)"
    )
    for line in (trial(0), trial(1, feasible=False), failed, trial(3, "interrupted"), recovered):
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


def test_a_trial_takes_the_time_so_far_per_trial_ended_and_the_time_left_counts_down() -> None:
    clock = Clock()
    shown = progress(10, clock)
    for clock.now in (100.0, 160.0, 240.0):
        shown.trial(trial(0))
    assert shown.per_trial == 80.0
    assert shown.left == 7 * 80.0
    clock.now = 260.0  # 20 s since the last trial ended
    assert shown.left == 7 * 80.0 - 20 and "~9:00 left" in shown.line(140)
    clock.now = 900.0  # longer than all the rest should take: nothing left to count
    assert shown.left == 0.0
    for clock.now in (910.0, 980.0, 1050.0, 1120.0, 1190.0, 1260.0, 1330.0):
        shown.trial(trial(0))
    assert shown.left == 0.0 and " 100%  10/10 trials" in shown.line(140)


def test_trials_that_end_together_count_as_often_as_they_end() -> None:
    clock = Clock()
    shown = progress(20, clock)
    shown.activity(Activity(candidates=4, busy=8, capacity=8))
    for clock.now in (600.0, 601.0, 602.0, 603.0):  # four jobs, each trial 10 min
        shown.trial(trial(0))
    assert shown.per_trial == pytest.approx(150.75)
    assert shown.left == pytest.approx(16 * 150.75)  # 40 min, not 16 s apart each
    line = shown.line(160)
    assert "4 running  8/8 slots  2:30/trial" in line


def test_with_one_job_the_bar_shows_no_running_trials_nor_slots() -> None:
    shown = progress(5, Clock())
    shown.activity(Activity(candidates=1, busy=1, capacity=1))
    assert "running" not in shown.line(200) and "slots" not in shown.line(200)


def test_running_trials_and_slots_give_way_to_the_times_but_not_to_the_rest() -> None:
    clock = Clock()
    shown = progress(30, clock)
    shown.activity(Activity(candidates=3, busy=4, capacity=4))
    clock.now = 100.0
    shown.trial(trial(0))
    assert ANSI.sub("", shown.line(120)).endswith("~48:20 left  3 running  4/4 slots")
    assert ANSI.sub("", shown.line(60)).endswith("~48:20 left  3 running")  # and no bar


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
    assert shown.ended == 1 and shown.per_trial == 65.0 and "1/3 trials" in shown.line(140)
    assert screen(out.getvalue())[0] == "trial 7:   interrupted  trial-000007  (recovered)"


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


def test_a_study_that_stops_before_a_trial_ended_leaves_no_bar() -> None:
    clock, out = Clock(), Terminal()

    def stopped(recovered: bool) -> None:
        with progress(30, clock, out) as shown:
            if recovered:
                shown.trial(trial(3, "interrupted", recovered=True))
            raise KeyboardInterrupt

    for recovered in (False, True):
        with pytest.raises(KeyboardInterrupt):
            stopped(recovered)
    assert "0/30 trials" in out.getvalue()  # drawn while it ran
    assert screen(out.getvalue()) == ["trial 3:   interrupted  trial-000003  (recovered)", ""]


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

    out, err = Terminal(), io.StringIO()  # a study that cannot start: its error, and no bar
    argv = ["optimize", str(problem), "--evaluator", TRADE_OFF, "--trials", "2", "--resume"]
    code = main([*argv, "--output", str(tmp_path / "none")], stdout=out, stderr=err)
    assert code == 1 and "no study to resume" in err.getvalue()
    assert screen(out.getvalue()) == [""]

    monkeypatch.setenv("NO_COLOR", "1")
    code, text = optimize("plain")
    assert code == 0 and not ANSI.search(text) and "\r\x1b[2K" in text

    monkeypatch.setenv("TERM", "dumb")
    code, text = optimize("dumb")
    assert code == 0 and "\x1b" not in text and "\r" not in text and len(text.splitlines()) == 5


def test_a_running_study_cannot_send_the_terminal_a_control_sequence() -> None:
    esc = "\x1b[2J\nfake"  # clears the screen and starts a line, were it printed as it is
    out, err = io.StringIO(), io.StringIO()
    with Progress(3, out, err, bar=False, color=False) as shown:
        candidate, objectives = {f"x{esc}": f"c{esc}"}, {f"a{esc}": 1.0}
        shown.trial(
            TrialReport(0, Path("t"), candidate, "valid", feasible=True, objectives=objectives)
        )
        shown.trial(trial(1, "failed", record="r", message=f"ddsim{esc}"))
        shown.trial(TrialReport(2, Path(f"trial{esc}"), {}, "interrupted"))
        shown.warning(f"trial 1: careful{esc}")
    lines = out.getvalue().splitlines()
    assert "\x1b" not in out.getvalue() + err.getvalue() and len(lines) == 3
    assert "".join(lines).count("\\x1b[2J\\nfake") == 5
    assert err.getvalue() == "k4opt: warning: trial 1: careful\\x1b[2J\\nfake\n"


def test_a_study_read_back_cannot_send_the_terminal_a_control_sequence() -> None:
    esc = "\x1b[2J"  # clears the screen, were it printed as it is
    evaluator = "test.trade_off"  # an evaluator name holds no control character
    spec = parse_problem(
        problem_document(
            objectives=[("analysis", f"a{esc}", "minimize")],
            constraints=[{"stage": "analysis", "metric": f"a{esc}", "max": 1, "unit": f"mm{esc}"}],
            parameter={
                "geometry": {"constant": "InnerTracker_Barrel_radius_1"},
                "type": "categorical",
                "choices": [f"x{esc}"],
            },
        )
    )
    measured = (ObjectiveValue(evaluator, "analysis", f"a{esc}", "minimize", 1.0, f"mm{esc}"),)
    limit = Limit(evaluator, "analysis", f"a{esc}", "max", 1, f"mm{esc}", 1.0, None, 0.0, True)
    trial = TrialResult(
        0,
        "valid",
        {RADIUS: f"x{esc}"},
        Path(f"trial{esc}"),
        Assessment(measured, (limit,), True),
        record=f"id{esc}",
        sha256="0" * 64,
        message=f"note{esc}",
        elapsed_seconds=1.0,
        stage_seconds={f"simulation{esc}": 1.0},
    )
    metric = FrozenMap({"stage": "analysis", "metric": f"a{esc}", "unit": f"mm{esc}"})
    document = FrozenMap(
        {
            "optimizer": FrozenMap({"storage": FrozenMap({"journal": f"study{esc}.journal"})}),
            "implementations": FrozenMap({evaluator: FrozenMap({"metrics": (metric,)})}),
        }
    )
    study = StudyResults(
        f"name{esc}", f"id{esc}", f"now{esc}", Path(f"dir{esc}"), spec, document, (trial,), (0,)
    )
    out = io.StringIO()
    render_results(study, out)
    render_trial(study, trial, out)
    shown = out.getvalue()
    assert "\x1b" not in shown
    assert shown.count("\\x1b[2J") >= 12  # each shown escaped instead
