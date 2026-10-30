"""Human-readable output of a study: a line per trial as it ends, and the study at the end.

On an interactive terminal the lines are colored and a progress bar of the trials stays on the
last line, redrawn while the study runs; piped or written to a file, each trial is one plain
line, fit for a log. A parameter value is shown to 6 significant digits; its record holds all of
them. A trial a resume recovered is marked ``(recovered)``.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable
from types import TracebackType
from typing import Any, TextIO

from k4opt.optimization._study import StudyReport, TrialReport
from k4opt.terminal import Style, width

BAR_WIDTH = 40
MIN_BAR_WIDTH = 20
BAR_PRIORITY = 5  # the parts above it are shown before the bar, the others where it leaves room
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
OUTCOMES = ("feasible", "infeasible", "invalid", "failed", "interrupted", "error")


def _paint(style: Style, outcome: str, text: str) -> str:
    """``text`` in the color of a trial's status or feasibility."""
    if outcome in ("valid", "feasible"):
        return style.green(text)
    if outcome in ("invalid", "infeasible"):
        return style.yellow(text)
    return style.red(text)


def _value(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _outcome(trial: TrialReport) -> str:
    """``feasible`` or ``infeasible`` for a valid trial, its status otherwise."""
    if trial.status == "valid":
        return "feasible" if trial.feasible else "infeasible"
    return trial.status


def trial_line(trial: TrialReport, *, color: bool = False) -> str:
    """One line for a trial: its status, and its candidate and objectives or what went wrong."""
    style = Style(color)
    head = f"trial {trial.number}:".ljust(11) + _paint(style, trial.status, trial.status.ljust(9))
    if trial.objectives is not None:
        outcome = _outcome(trial)
        candidate = "  ".join(
            style.dim(f"{name}=") + style.cyan(_value(value))
            for name, value in trial.candidate.items()
        )
        values = "  ".join(
            style.dim(f"{name}=") + style.bold(f"{value:.6g}")
            for name, value in trial.objectives.items()
        )
        line = f"{head}{_paint(style, outcome, outcome.ljust(10))}  {candidate}  {values}"
    else:
        where = f"{trial.directory.name}/summary.json" if trial.record else trial.directory.name
        line = f"{head}{trial.message or ''}  {style.dim(where)}"
    return f"{line}  {style.dim('(recovered)')}" if trial.recovered else line


def render_study(report: StudyReport, out: TextIO, *, color: bool = False) -> None:
    """Print the study: its name, how many trials ended how, and its Pareto front."""
    style = Style(color)
    counts = Counter(_outcome(trial) for trial in report.trials)
    valid = counts["feasible"] + counts["infeasible"]
    parts = [
        f"{style.green(str(valid))} valid "
        f"({_paint(style, 'feasible', str(counts['feasible']))} feasible, "
        f"{_paint(style, 'infeasible', str(counts['infeasible']))} infeasible)"
    ]
    parts += [
        _paint(style, outcome, f"{counts[outcome]} {outcome}")
        for outcome in OUTCOMES[2:]
        if counts[outcome]
    ]
    print(
        f"{style.dim('study:')}    {style.bold(report.name)}, {len(report.objectives)} "
        f"objective(s), {len(report.constraints)} limit(s), {report.study_path}",
        file=out,
    )
    print(f"{style.dim('trials:')}   {len(report.trials)}: {', '.join(parts)}", file=out)
    pareto = ", ".join(map(str, report.pareto)) if report.pareto else "none feasible yet"
    print(f"{style.dim('pareto:')}   {style.green(pareto) if report.pareto else pareto}", file=out)


def _duration(seconds: float) -> str:
    """``M:SS``, or ``H:MM:SS`` from an hour on."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes}:{seconds:02}"


class Progress:
    """Prints the trials of a study as they end and, when ``bar``, a progress bar under them.

    The bar shows the new trials that ended against ``trials``, as a bar and a percentage, the
    time since the study began, the time left, the time a trial takes and how the trials ended.
    It is redrawn every ``interval`` seconds and after each line printed above it, and left in
    place when the study ends, marked ``✔``, or ``✘`` when it was stopped; a study stopped
    before any of its new trials ended (one that could not start) leaves none. Trials a resume
    recovers are printed but do not advance it.

    A trial takes the mean time from the end of one trial to the end of the next. The first
    trial of a session counts only until another has ended, since its time holds the study's
    setup. The time left is that of the trials still to run, less the time the running one has
    had, so it counts down between the ends of trials.
    """

    def __init__(
        self,
        trials: int,
        out: TextIO,
        err: TextIO,
        *,
        bar: bool,
        color: bool,
        interval: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.trials = trials
        self.out = out
        self.err = err
        self.bar = bar
        self.color = color
        self.interval = interval
        self.clock = clock
        self.outcomes: Counter[str] = Counter()
        self.durations: list[float] = []  # of the new trials, each since the one before ended
        self._style = Style(color)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._tick, daemon=True)
        self._start = self._last = clock()
        self._ended: bool | None = None  # whether the study ran all its trials, once it ended

    def __enter__(self) -> Progress:
        self._start = self._last = self.clock()
        if self.bar:
            self._draw()
            self._ticker.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.bar:
            self._stop.set()
            self._ticker.join()
            with self._lock:
                self._ended = kind is None
                if kind is None or self.durations:
                    self._draw()
                    self.out.write("\n")
                else:  # nothing ran that the bar could tell of: the error alone says why
                    self.out.write("\r\x1b[2K")
                self.out.flush()

    def trial(self, trial: TrialReport) -> None:
        """Print the line of a trial that ended."""
        with self._lock:
            now = self.clock()
            if not trial.recovered:
                self.outcomes[_outcome(trial)] += 1
                self.durations.append(now - self._last)
            self._last = now
            self._print(trial_line(trial, color=self.color), self.out)

    def warning(self, message: str) -> None:
        with self._lock:
            self._print(f"k4opt: warning: {message}", self.err)

    @property
    def per_trial(self) -> float | None:
        """The time a trial takes, once one has ended."""
        measured = self.durations[1:] or self.durations
        return sum(measured) / len(measured) if measured else None

    @property
    def left(self) -> float | None:
        """The time until the last trial ends, once one has ended."""
        remaining = self.trials - len(self.durations)
        if self.per_trial is None or remaining <= 0:
            return None if self.per_trial is None else 0.0
        running = self.clock() - self._last
        return max(self.per_trial - running, 0.0) + (remaining - 1) * self.per_trial

    def _print(self, text: str, stream: TextIO) -> None:
        if self.bar:
            self.out.write("\r\x1b[2K")
            self.out.flush()
        print(text, file=stream, flush=True)
        if self.bar:
            self._draw()

    def _tick(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                self._draw()

    def _draw(self) -> None:
        self.out.write("\r\x1b[2K" + self.line(width(self.out)))
        self.out.flush()

    def _cells(self, cells: int, fraction: float) -> str:
        """The bar in ``cells`` characters, to half a cell."""
        style = self._style
        halves = round(fraction * cells * 2)
        done = "━" * (halves // 2) + "╸" * (halves % 2)
        rest = ("━" if self.color else "─") * (cells - len(done))
        paint = {None: style.cyan, True: style.green, False: style.red}[self._ended]
        return paint(done) + style.gray(rest)

    def line(self, columns: int | None) -> str:
        """The bar, fit to ``columns``: the least telling parts are left out first."""
        style = self._style
        done = len(self.durations)
        fraction = min(done / max(self.trials, 1), 1.0)
        elapsed = self.clock() - self._start
        if self._ended is None:
            mark = style.cyan(SPINNER[int(elapsed / 0.1) % len(SPINNER)])
        else:
            mark = style.green("✔") if self._ended else style.red("✘")
        # the parts after the bar in the order shown, each with a priority: the lowest goes first
        percent, count = f"{int(fraction * 100):3}%", f"{done}/{self.trials} trials"
        parts = [(8, percent, style.bold(percent)), (9, count, count)]
        parts.append((6, f"{_duration(elapsed)} elapsed", f"{_duration(elapsed)} elapsed"))
        if self._ended is None:
            left = "--:-- left" if self.left is None else f"~{_duration(self.left)} left"
            parts.append((7, left, style.bold(left)))
        if self.per_trial is not None:
            each = f"{_duration(self.per_trial)}/trial"
            parts.append((4, each, style.dim(each)))
        counts = [(o, f"{self.outcomes[o]} {o}") for o in OUTCOMES if self.outcomes[o]]
        if counts:  # all of them or none
            plain = "  ".join(text for _, text in counts)
            parts.append((3, plain, "  ".join(_paint(style, o, text) for o, text in counts)))

        # the parts above BAR_PRIORITY take their room first, then the bar, then the others in
        # what is left, so the bar keeps its width as they come and go
        space = (columns or 80) - 3  # the mark and a space; writing the last column can wrap
        cells = min(BAR_WIDTH, space - sum(len(p[1]) + 2 for p in parts if p[0] > BAR_PRIORITY))
        room = space - cells if cells >= MIN_BAR_WIDTH else space + 2
        kept = set()
        for index in sorted(range(len(parts)), key=lambda index: -parts[index][0]):
            if len(parts[index][1]) + 2 <= room:
                kept.add(index)
                room -= len(parts[index][1]) + 2
        shown = [styled for index, (_, _, styled) in enumerate(parts) if index in kept]
        if cells >= MIN_BAR_WIDTH:
            shown.insert(0, self._cells(cells, fraction))
        return f"{mark} {'  '.join(shown)}".rstrip()
