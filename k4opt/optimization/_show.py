"""Human-readable output of a study read back: ``k4opt study show`` and ``k4opt study trial``.

``show`` says what the study optimizes, how its trials ended and which are on the Pareto front,
those with their values to 6 significant digits; ``trial`` shows one trial with every value as
recorded. On an interactive terminal the labels and outcomes are colored. Every text that comes
from the study (its name and times, parameter, stage and metric names, units, paths, ids and
messages) is printed with its control characters escaped, so that a study cannot move the cursor
or change colors: only the labels and colors are k4opt's own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TextIO

from k4opt.optimization._render import _paint, _value
from k4opt.optimization._results import StudyResults, TrialResult
from k4opt.problem import MetricRef, Parameter
from k4opt.terminal import Style

LABEL = 12  # the width of a line's label


def printable(text: object) -> str:
    """``text`` with its control characters escaped (``\\x1b``, ``\\n``)."""
    return "".join(c if c.isprintable() else c.encode("unicode_escape").decode() for c in str(text))


def _measured(value: object, unit: str | None, uncertainty: object = None, *, exact: bool) -> str:
    """A value with its uncertainty and unit; ``exact``: every digit, strings quoted."""
    shown = repr if exact else _value
    text = shown(value)
    if uncertainty is not None:
        text += f" ± {shown(uncertainty)}"
    return printable(f"{text} {unit}" if unit else text)


def _domain(parameter: Parameter) -> str:
    if parameter.choices is not None:
        text = f"{parameter.kind} {{{', '.join(map(repr, parameter.choices))}}}"
    else:
        assert parameter.bounds is not None
        low, high = parameter.bounds
        text = f"{parameter.kind} [{low!r}, {high!r}]"
    return printable(f"{text} {parameter.unit}" if parameter.unit else text)


def _units(results: StudyResults) -> dict[MetricRef, str]:
    """The unit each binding of the study measures each metric in, as its implementation
    declares it."""
    units: dict[MetricRef, str] = {}
    implementations = results.document.get("implementations")
    if not isinstance(implementations, Mapping):
        return units
    for entry in results.problem.evaluators:
        implementation = implementations.get(entry.name)
        metrics = implementation.get("metrics") if isinstance(implementation, Mapping) else None
        for metric in metrics if isinstance(metrics, tuple) else ():
            if isinstance(metric, Mapping):
                stage, name, unit = metric.get("stage"), metric.get("metric"), metric.get("unit")
                if isinstance(stage, str) and isinstance(name, str) and isinstance(unit, str):
                    units[MetricRef(entry.binding, stage, name)] = unit
    return units


def _limits(results: StudyResults) -> list[str]:
    """Each limit of the problem: its name and its bound, ``e/analysis/a:max  <= 450 mm``."""
    limits = []
    for c in results.problem.constraints:
        for bound, sign in (("min", ">="), ("max", "<=")):
            limit = getattr(c, bound)
            if limit is not None:
                name = printable(f"{c.label}:{bound}")
                limits.append(f"{name}  {sign} {_measured(limit, c.unit, exact=True)}")
    return limits


def _print(out: TextIO, style: Style, label: str, lines: list[str]) -> None:
    """``lines`` under one label, the label on the first."""
    for index, line in enumerate(lines or ["none"]):
        head = f"{label}:" if index == 0 else ""
        print(f"{style.dim(head.ljust(LABEL))}{line}", file=out)


def _counts(results: StudyResults, style: Style) -> str:
    outcomes = results.outcomes
    if not results.trials:
        return "none yet"
    valid = outcomes["feasible"] + outcomes["infeasible"]
    parts = [
        f"{style.green(str(valid))} valid ({_paint(style, 'feasible', str(outcomes['feasible']))} "
        f"feasible, {_paint(style, 'infeasible', str(outcomes['infeasible']))} infeasible)"
    ]
    parts += [
        _paint(style, outcome, f"{count} {outcome}")
        for outcome, count in outcomes.items()
        if outcome not in ("feasible", "infeasible") and count
    ]
    return f"{len(results.trials)}: {', '.join(parts)}"


def _front_line(trial: TrialResult, results: StudyResults, style: Style) -> str:
    """A trial of the Pareto front: its candidate and objectives, to 6 significant digits."""
    assert trial.assessment is not None
    assert trial.assessment.objectives is not None
    candidate = [
        style.dim(f"{printable(p.geometry.canonical_target)}=")
        + style.cyan(_measured(trial.candidate[p.geometry.canonical_target], p.unit, exact=False))
        for p in results.problem.parameters
    ]
    objectives = [
        style.dim(f"{printable(o.label)}=") + style.bold(_measured(o.value, o.unit, exact=False))
        for o in trial.assessment.objectives
    ]
    return f"trial {trial.number}:  " + "  ".join(candidate + objectives)


def render_results(results: StudyResults, out: TextIO, *, color: bool = False) -> None:
    """Print what the study optimizes, how its trials ended and its Pareto front."""
    style = Style(color)
    spec, units = results.problem, _units(results)
    storage = results.document.get("optimizer")
    kept = storage.get("storage") if isinstance(storage, Mapping) else None
    where = next(iter(kept.values()), None) if isinstance(kept, Mapping) else None
    name, created = printable(results.name), printable(results.created)
    _print(out, style, "study", [f"{style.bold(name)}, created {created}"])
    _print(out, style, "id", [printable(results.id)])
    directory = f"{printable(results.directory)} (trials in {printable(where)})"
    _print(out, style, "directory", [directory])
    parameters = [
        f"{printable(p.geometry.canonical_target)}  {_domain(p)}" for p in spec.parameters
    ]
    _print(out, style, "parameters", parameters)
    objectives = [
        f"{printable(o.label)}  {o.direction}"
        + (f"  ({printable(units[o.ref])})" if o.ref in units else "")
        for o in spec.objectives
    ]
    _print(out, style, "objectives", objectives)
    _print(out, style, "limits", _limits(results))
    _print(out, style, "trials", [_counts(results, style)])
    if results.pareto:
        front = [results.trial(number) for number in results.pareto]
        numbers = style.green(", ".join(map(str, results.pareto)))
        _print(out, style, "pareto", [numbers, *(_front_line(t, results, style) for t in front)])
    else:
        _print(out, style, "pareto", ["none yet" if not results.trials else "none feasible yet"])


_NO_RECORD = {
    "running": "none yet: the optimizer has not been told the trial's result",
    "interrupted": "none: the trial ended before its record was written",
    "error": "none: the trial stopped the study before its record was written",
}


def render_trial(
    results: StudyResults, trial: TrialResult, out: TextIO, *, color: bool = False
) -> None:
    """Print one trial with every value as recorded: its candidate, objectives, limits, times,
    record and message."""
    style = Style(color)
    spec = results.problem
    state = [_paint(style, trial.status, trial.status)]
    if trial.status == "valid":
        state.append(_paint(style, trial.outcome, trial.outcome))
    if trial.number in results.pareto:
        state.append(style.green("on the Pareto front"))
    if trial.recovered:
        state.append(style.dim("recovered by a resume"))
    _print(out, style, "trial", [f"{style.bold(str(trial.number))}: {', '.join(state)}"])
    _print(out, style, "study", [printable(f"{results.name}, {results.study_path}")])
    candidate = [
        f"{printable(p.geometry.canonical_target)} = "
        + (
            _measured(trial.candidate[p.geometry.canonical_target], p.unit, exact=True)
            if p.geometry.canonical_target in trial.candidate
            else "not proposed yet"
        )
        for p in spec.parameters
    ]
    _print(out, style, "candidate", candidate)
    assessment = trial.assessment
    if assessment is None or assessment.objectives is None:
        _print(out, style, "objectives", ["none: the trial has no result"])
    else:
        objectives = [
            f"{printable(o.label)} = "
            + style.bold(_measured(o.value, o.unit, o.uncertainty, exact=True))
            + f", {o.direction}"
            for o in assessment.objectives
        ]
        _print(out, style, "objectives", objectives)
    if assessment is not None and assessment.feasible is not None:
        limits = []
        for limit in assessment.limits:
            value = _measured(limit.value, limit.unit, limit.uncertainty, exact=True)
            bound = _measured(limit.limit, limit.unit, exact=True)
            sign = "<=" if limit.bound == "max" else ">="
            verdict = "satisfied" if limit.satisfied else "violated"
            painted = _paint(style, "feasible" if limit.satisfied else "infeasible", verdict)
            limits.append(
                f"{printable(limit.name)}  {value} {sign} {bound}: {painted}, "
                f"margin {limit.margin!r}"
            )
        _print(out, style, "limits", limits)
    elif spec.constraints:
        _print(out, style, "limits", ["not assessed"])
    if trial.elapsed_seconds is None:
        times = "not recorded"
    else:
        stages = ", ".join(
            f"{printable(stage)} {seconds!r} s" for stage, seconds in trial.stage_seconds.items()
        )
        times = f"{trial.elapsed_seconds!r} s in all" + (f"; {stages}" if stages else "")
    _print(out, style, "time", [times])
    if trial.record is None:
        _print(out, style, "record", [_NO_RECORD.get(trial.status, "none")])
    else:
        record = [
            printable(trial.summary_path),
            printable(f"id {trial.record}, sha256 {trial.sha256}"),
        ]
        if trial.status == "error":
            record.append("not taken: its values never reached the optimizer")
        _print(out, style, "record", record)
    if trial.message is not None:
        _print(out, style, "message", [printable(trial.message)])
