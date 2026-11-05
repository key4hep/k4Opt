"""The rows of the dashboard's tables, from a catalog and a study alone: no Streamlit here.

Each function returns a list of rows, one ``dict`` each, column by column, with every value as the
study records it (``None`` where it has none: an empty cell, never 0), headed by what
:func:`~k4opt.visualization.names` calls each (each column's whole label is in :func:`helps`).
What a value means is the study's to say: outcomes, feasibility and the Pareto front are read,
never judged again; a best value is the best the study's feasible trials reached of one
objective, by its direction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from k4opt.optimization import StudyResults, TradeOffs, TrialResult
from k4opt.problem import Parameter
from k4opt.record import Limit, ObjectiveValue
from k4opt.visualization import GraphNode, names

import labels

type Row = dict[str, object]

WITHOUT_RESULT = ("invalid", "failed", "rejected", "unscored", "interrupted", "error")
"""The outcomes of trials without objective values, but for the running ones."""


def domain(parameter: Parameter) -> str:
    """The values ``parameter`` may take: its choices, or its bounds."""
    if parameter.choices is not None:
        return ", ".join(map(str, parameter.choices))
    assert parameter.bounds is not None
    low, high = parameter.bounds
    return f"[{low}, {high}]"


def units(results: StudyResults) -> dict[str, str | None]:
    """The unit of each objective and constrained metric, by label: the first its trials recorded
    (a declared metric has one; :func:`~k4opt.visualization.trials_figure` refuses an axis of
    several)."""
    found: dict[str, str | None] = {}
    for trial in results.trials:
        assessment = trial.assessment
        if assessment is None:
            continue
        values: list[ObjectiveValue | Limit] = [*(assessment.objectives or ()), *assessment.limits]
        for value in values:
            if found.get(value.label) is None:
                found[value.label] = value.unit
    return found


def _heading(name: str, unit: str | None) -> str:
    shown = labels.unit(unit)
    return f"{name} [{shown}]" if shown else name


type _Kind = Literal["fixed", "parameter", "objective", "uncertainty", "margin", "change"]


@dataclass(frozen=True, slots=True)
class _Column:
    """A column of a table: its heading, what it shows (a parameter's name, an objective's label,
    a limit's ``label:bound``; the heading of a fixed column) and what kind of value that is."""

    heading: str
    key: str
    kind: _Kind


_TRIAL_FIXED = (("Trial", "Outcome", "Pareto"), ("Time [s]", "Message"))
_TRADE_FIXED = (("Trial", "Reference", "Pareto"), ())


def _columns(
    results: StudyResults, fixed: tuple[tuple[str, ...], tuple[str, ...]], *, trade_off: bool
) -> list[_Column]:
    """Every column of a table, in order, each headed by what it is called, and no two alike:
    the ``fixed`` columns before and after the rest; each parameter; each objective, with its
    uncertainty if any trial recorded one and each limit's margin (of the trials table) or how it
    changed against the reference (``trade_off``). A heading another column has too is replaced
    by one with the whole name or label of what it shows (``Size (ecal_cell_size) [mm]``), and
    one still alike then by a number after it (``a (2)``), so that no column ever takes the place
    of another; a fixed one is never replaced."""
    shown, called = units(results), names(results.problem)
    uncertain = {
        value.label
        for trial in results.trials
        if trial.assessment is not None
        for value in trial.assessment.objectives or ()
        if value.uncertainty is not None
    }
    options: dict[tuple[_Kind, str], list[str]] = {}
    before, after = fixed
    for name in before:
        options["fixed", name] = [name]
    for p in results.problem.parameters:
        said = called.parameter(p.name)
        options["parameter", p.name] = [
            _heading(said, p.unit),
            _heading(f"{said} ({p.name})", p.unit),
        ]
    for o in results.problem.objectives:
        unit = shown.get(o.label)
        choices = [_heading(called.objective(o.label), unit), _heading(o.label, unit)]
        options["objective", o.label] = choices
        if trade_off:
            options["change", o.label] = [f"Δ {called.objective(o.label)} [%]", f"Δ {o.label} [%]"]
        elif o.label in uncertain:
            options["uncertainty", o.label] = [f"{choice} ±" for choice in choices]
    for c in () if trade_off else results.problem.constraints:
        unit = c.unit or shown.get(c.label)
        for bound in ("min", "max"):
            if getattr(c, bound) is not None:
                options["margin", f"{c.label}:{bound}"] = [
                    _heading(f"margin {called.constraint(c.label)} {bound}", unit),
                    _heading(f"margin {c.label} {bound}", unit),
                ]
    for name in after:
        options["fixed", name] = [name]
    chosen = _distinct({key: list(dict.fromkeys(each)) for key, each in options.items()})
    # the rare heading still alike (a title written as another column's whole label) numbered,
    # the fixed columns claiming theirs first
    taken: set[str] = set()
    headings: dict[tuple[_Kind, str], str] = {}
    for kind, key in sorted(options, key=lambda column: column[0] != "fixed"):
        heading, number = chosen[kind, key], 2
        while heading in taken:
            heading, number = f"{chosen[kind, key]} ({number})", number + 1
        taken.add(heading)
        headings[kind, key] = heading
    return [_Column(headings[kind, key], key, kind) for kind, key in options]


def _distinct[K](options: Mapping[K, Sequence[str]]) -> dict[K, str]:
    """For each key, the first of its ``options`` no other key has: each key alike with another
    takes its next option, while it has one."""
    level = dict.fromkeys(options, 0)
    while True:
        chosen = {key: options[key][level[key]] for key in options}
        alike = Counter(chosen.values())
        clash = [k for k in options if alike[chosen[k]] > 1 and level[k] < len(options[k]) - 1]
        if not clash:
            return chosen
        for key in clash:
            level[key] += 1


def helps(results: StudyResults) -> dict[str, str]:
    """What each measurement column of :func:`trials` shows, by heading, in whole."""
    said = {
        "objective": "{}",
        "uncertainty": "the uncertainty of {}",
        "margin": "the margin of the limit {}: zero or below when satisfied",
    }
    return {
        column.heading: said[column.kind].format(column.key)
        for column in _columns(results, _TRIAL_FIXED, trade_off=False)
        if column.kind in said
    }


def trials(results: StudyResults, numbers: Iterable[int] | None = None) -> list[Row]:
    """A row for each trial, or for those numbered ``numbers``: its outcome, its candidate, each
    objective's value (and its uncertainty, if any trial recorded one), each limit's margin (zero
    or below when satisfied), its time and message."""
    columns = _columns(results, _TRIAL_FIXED, trade_off=False)
    chosen = None if numbers is None else set(numbers)
    front = set(results.pareto)
    rows = []
    for trial in results.trials:
        if chosen is not None and trial.number not in chosen:
            continue
        fixed = {
            "Trial": trial.number,
            "Outcome": trial.outcome,
            "Pareto": trial.number in front,
            "Time [s]": trial.elapsed_seconds,
            "Message": trial.message,
        }
        rows.append(_row(trial, columns, fixed))
    return rows


def _row(trial: TrialResult, columns: Sequence[_Column], fixed: Mapping[str, object]) -> Row:
    """The row of ``trial``: each column's value, ``fixed`` that of each fixed column."""
    assessment = trial.assessment
    measured = {} if assessment is None else {o.label: o for o in assessment.objectives or ()}
    limits = {} if assessment is None else {limit.name: limit for limit in assessment.limits}
    row: Row = {}
    for column in columns:
        value: object = None
        if column.kind == "fixed":
            value = fixed[column.key]
        elif column.kind == "parameter":
            value = trial.candidate.get(column.key)
        elif column.kind == "margin":
            limit = limits.get(column.key)
            value = None if limit is None else limit.margin
        elif column.kind in ("objective", "uncertainty"):
            found = measured.get(column.key)
            if found is not None:
                value = found.value if column.kind == "objective" else found.uncertainty
        row[column.heading] = value
    return row


@dataclass(frozen=True, slots=True)
class Best:
    """The best value the feasible trials reached of one objective, and by which trial: the first
    to reach it."""

    label: str
    name: str
    direction: str
    value: float
    uncertainty: float | None
    unit: str | None
    trial: int


def bests(results: StudyResults) -> list[Best]:
    """The best of each objective that a feasible trial measured; none of one that none did."""
    called = names(results.problem)
    found = []
    for index, objective in enumerate(results.problem.objectives):
        sign = -1 if objective.direction == "maximize" else 1
        best: ObjectiveValue | None = None
        number = -1
        for trial in results.trials:
            if trial.outcome != "feasible":
                continue
            assert trial.assessment is not None
            assert trial.assessment.objectives is not None
            value = trial.assessment.objectives[index]
            if best is None or sign * value.value < sign * best.value:
                best, number = value, trial.number
        if best is not None:
            found.append(
                Best(
                    objective.label,
                    called.objective(objective.label),
                    objective.direction,
                    best.value,
                    best.uncertainty,
                    best.unit,
                    number,
                )
            )
    return found


def provenance(results: StudyResults) -> list[Row]:
    """What the study was made with, as its ``study.json`` records it: k4opt, the Key4hep stack,
    the geometry and the optimizer."""
    document = results.document
    k4opt, environment = _get(document, "k4opt"), _get(document, "environment")
    geometry, optimizer = _get(document, "geometry"), _get(document, "optimizer")
    sampler = _get(optimizer, "sampler")
    stack = results.problem.environment.key4hep
    rows = [
        ("Study id", results.id),
        ("Created", results.created),
        ("k4opt", f"{_get(k4opt, 'version')} (code SHA-256 {_get(k4opt, 'sha256')})"),
        ("k4opt commit", _get(_get(k4opt, "git"), "commit")),
        ("Key4hep stack (problem)", f"{stack.channel} {stack.version}"),
        ("KEY4HEP_STACK", _get(environment, "KEY4HEP_STACK")),
        ("Geometry", _get(geometry, "compact")),
        ("Geometry fingerprint", _get(geometry, "fingerprint")),
        ("Optimizer", f"{_get(optimizer, 'backend')} {_get(optimizer, 'version')}"),
        ("Sampler", f"{_get(sampler, 'name')}, seed {_get(sampler, 'seed')}"),
        ("Identity digest", _get(document, "identity")),
    ]
    return [{"": key, "Value": value} for key, value in rows]


def evaluators(results: StudyResults) -> list[Row]:
    """Each evaluator implementation the study ran: its version, source hash and commit."""
    implementations = _get(results.document, "implementations")
    rows = []
    for name, entry in (implementations or {}).items():
        source = _get(entry, "source")
        revision = _get(source, "revision")
        rows.append(
            {
                "Evaluator": name,
                "Version": _get(entry, "version"),
                "Source SHA-256": _get(source, "sha256"),
                "Commit": _get(revision, "commit"),
                "Uncommitted changes": _get(revision, "dirty"),
            }
        )
    return rows


def _get(data: object, key: str) -> Any:
    return data.get(key) if isinstance(data, Mapping) else None


def facts(node: GraphNode) -> list[Row]:
    """What the problem says of a node of the task graph, a row per fact."""
    return [{"": name, "Value": value} for name, value in node.facts]


def task_trials(results: StudyResults, node: GraphNode) -> list[Row]:
    """A row for each trial with a record, of a stage or an evaluation: how long the stage took (if
    it records its own time), and how the task ended when it is a diagnostic that did not succeed.
    Empty when there is neither, and for any other node, whose values in each trial the table of
    every trial holds."""
    if node.kind not in ("geometry", "stage", "evaluator"):
        return []
    rows = []
    for trial in results.trials:
        if trial.record is None:
            continue
        row: Row = {"Trial": trial.number, "Outcome": trial.outcome}
        if node.seconds:
            row["Time [s]"] = node.seconds.get(trial.number)
        if not node.required:
            ended = next((d for d in trial.diagnostics if d.get("task") == node.id), None)
            row["Did not succeed"] = None if ended is None else ended.get("status")
            row["Message"] = None if ended is None else ended.get("message")
        rows.append(row)
    return rows if any(len(row) > 2 for row in rows) else []


def trial_record(trial: TrialResult) -> list[Row]:
    """Where ``trial``'s record is, a row per fact."""
    rows = [("Record", trial.record), ("Record SHA-256", trial.sha256)]
    return [{"": name, "Value": value} for name, value in rows]


def trial_limits(results: StudyResults, trial: TrialResult) -> list[Row]:
    """Each limit of ``trial``: the value, the bound, its margin (zero or below when satisfied)."""
    assessment = trial.assessment
    called = names(results.problem)
    return [
        {
            "Constraint": called.constraint(limit.label),
            "Bound": limit.bound,
            "Limit": limit.limit,
            "Value": limit.value,
            "Margin": limit.margin,
            "Unit": labels.unit(limit.unit),
            "Satisfied": limit.satisfied,
        }
        for limit in (() if assessment is None else assessment.limits)
    ]


def trial_stages(trial: TrialResult) -> list[Row]:
    """How long each stage of ``trial`` took, as its record holds it, longest first."""
    ordered = sorted(trial.stage_seconds.items(), key=lambda item: -item[1])
    return [
        {"Stage": stage, "Time": labels.duration(seconds), "Seconds": seconds}
        for stage, seconds in ordered
    ]


def trial_diagnostics(trial: TrialResult) -> list[Row]:
    """Each diagnostic of ``trial`` that did not succeed: how it ended, and why."""
    return [
        {"Diagnostic": d.get("task"), "Status": d.get("status"), "Message": d.get("message")}
        for d in trial.diagnostics
    ]


def trade_offs(results: StudyResults, found: TradeOffs) -> list[Row]:
    """A row for the reference and each trial that meets the trade-off: its candidate, each
    objective's value and how it changed against the reference, in percent of the reference's
    value (``None`` where that is zero)."""
    columns = _columns(results, _TRADE_FIXED, trade_off=True)
    front = set(results.pareto)
    reference = found.reference
    assert reference.objectives is not None
    rows = []
    for number in (reference.number, *found.matches):
        trial = results.trial(number)
        fixed = {
            "Trial": number,
            "Reference": number == reference.number,
            "Pareto": number in front,
        }
        row = _row(trial, columns, fixed)
        for column in columns:
            if column.kind != "change":
                continue
            base = reference.objectives[column.key]
            change = 0.0 if number == reference.number else found.change[number][column.key]
            row[column.heading] = None if base == 0 else 100 * change / abs(base)
        rows.append(row)
    return rows
