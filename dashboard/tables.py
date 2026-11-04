"""The rows of the dashboard's tables, from a catalog and a study alone: no Streamlit here.

Each function returns a list of rows, one ``dict`` each, column by column, with every value as the
study records it (``None`` where it has none: an empty cell, never 0), headed by what
:func:`~k4opt.visualization.names` calls each (each column's whole label is in :func:`helps`).
What a value means is the study's to say: outcomes, feasibility and the Pareto front are read,
never judged again; a best value is the best the study's feasible trials reached of one
objective, by its direction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from k4opt.optimization import StudyResults, TradeOffs, TrialResult
from k4opt.problem import Parameter
from k4opt.record import Limit, ObjectiveValue
from k4opt.visualization import GraphNode, names

import labels

type Row = dict[str, object]

WITHOUT_RESULT = ("invalid", "failed", "rejected", "interrupted", "error")
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


def _columns(results: StudyResults) -> list[tuple[str, str, str]]:
    """Each column of a trial's measurements: its heading, the whole label it shows (``± `` for
    an uncertainty, ``margin `` for a limit's margin) and what it is."""
    shown, called = units(results), names(results.problem)
    uncertain = {
        value.label
        for trial in results.trials
        if trial.assessment is not None
        for value in trial.assessment.objectives or ()
        if value.uncertainty is not None
    }
    columns = []
    for o in results.problem.objectives:
        heading = _heading(called.metric(o.label), shown.get(o.label))
        columns.append((heading, o.label, "objective"))
        if o.label in uncertain:
            columns.append((f"{heading} ±", o.label, "uncertainty"))
    for c in results.problem.constraints:
        for bound in ("min", "max"):
            if getattr(c, bound) is not None:
                name = f"margin {called.metric(c.label)} {bound}"
                heading = _heading(name, c.unit or shown.get(c.label))
                columns.append((heading, f"{c.label}:{bound}", "margin"))
    return columns


def helps(results: StudyResults) -> dict[str, str]:
    """What each measurement column of :func:`trials` shows, by heading, in whole."""
    said = {
        "objective": "{}",
        "uncertainty": "the uncertainty of {}",
        "margin": "the margin of the limit {}: zero or below when satisfied",
    }
    return {heading: said[kind].format(label) for heading, label, kind in _columns(results)}


def trials(results: StudyResults, numbers: Iterable[int] | None = None) -> list[Row]:
    """A row for each trial, or for those numbered ``numbers``: its outcome, its candidate, each
    objective's value (and its uncertainty, if any trial recorded one), each limit's margin (zero
    or below when satisfied), its time and message."""
    columns = _columns(results)
    chosen = None if numbers is None else set(numbers)
    front = set(results.pareto)
    called = names(results.problem)
    rows = []
    for trial in results.trials:
        if chosen is not None and trial.number not in chosen:
            continue
        row: Row = {
            "Trial": trial.number,
            "Outcome": trial.outcome,
            "Pareto": trial.number in front,
        }
        for p in results.problem.parameters:
            row[_heading(called.parameter(p.name), p.unit)] = trial.candidate.get(p.name)
        row |= _measured(trial, columns)
        row["Time [s]"] = trial.elapsed_seconds
        row["Message"] = trial.message
        rows.append(row)
    return rows


def _measured(trial: TrialResult, columns: list[tuple[str, str, str]]) -> Row:
    assessment = trial.assessment
    measured = {} if assessment is None else {o.label: o for o in assessment.objectives or ()}
    limits = {} if assessment is None else {limit.name: limit for limit in assessment.limits}
    row: Row = {}
    for heading, label, kind in columns:
        if kind == "margin":
            limit = limits.get(label)
            row[heading] = None if limit is None else limit.margin
            continue
        value = measured.get(label)
        if value is None:
            row[heading] = None
        else:
            row[heading] = value.value if kind == "objective" else value.uncertainty
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
                    called.metric(objective.label),
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
            "Constraint": called.metric(limit.label),
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
    shown, called = units(results), names(results.problem)
    front = set(results.pareto)
    reference = found.reference
    assert reference.objectives is not None
    rows = []
    for number in (reference.number, *found.matches):
        trial = results.trial(number)
        assert trial.objectives is not None
        row: Row = {
            "Trial": number,
            "Reference": number == reference.number,
            "Pareto": number in front,
        }
        for p in results.problem.parameters:
            row[_heading(called.parameter(p.name), p.unit)] = trial.candidate.get(p.name)
        for o in results.problem.objectives:
            row[_heading(called.metric(o.label), shown.get(o.label))] = trial.objectives[o.label]
            base = reference.objectives[o.label]
            change = 0.0 if number == reference.number else found.change[number][o.label]
            row[f"Δ {called.metric(o.label)} [%]"] = None if base == 0 else 100 * change / abs(base)
        rows.append(row)
    return rows
