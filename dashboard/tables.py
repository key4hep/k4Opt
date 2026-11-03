"""The rows of the dashboard's tables, from a catalog and a study alone: no Streamlit here.

Each function returns a list of rows, one ``dict`` each, column by column, with every value as the
study records it (``None`` where it has none: an empty cell, never 0). What a value means is the
study's to say: outcomes, feasibility and the Pareto front are read, never judged again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from k4opt.optimization import StudyResults, TradeOffs, TrialResult
from k4opt.problem import Parameter
from k4opt.publication import Catalog
from k4opt.record import Limit, ObjectiveValue
from k4opt.visualization import GraphNode

type Row = dict[str, object]

WITHOUT_RESULT = ("invalid", "failed", "rejected", "interrupted", "error")
"""The outcomes of trials without objective values, but for the running ones."""


def studies(catalog: Catalog) -> list[Row]:
    """A row for each study of ``catalog``, the latest published first."""
    entries = sorted(catalog.studies, key=lambda entry: entry.published, reverse=True)
    return [
        {
            "Study": entry.name,
            "Problem": entry.problem,
            "Created": entry.created,
            "Published": entry.published,
            "Trials": entry.trials,
            "Feasible share": entry.outcomes["feasible"] / entry.trials if entry.trials else None,
            "Feasible": entry.outcomes["feasible"],
            "Infeasible": entry.outcomes["infeasible"],
            "Without a result": sum(entry.outcomes[o] for o in WITHOUT_RESULT),
            "Running at publication": entry.outcomes["running"],
            "Parameters": entry.parameters,
            "Objectives": entry.objectives,
            "Constraints": entry.constraints,
            "Id": entry.id,
        }
        for entry in entries
    ]


def _domain(parameter: Parameter) -> str:
    if parameter.choices is not None:
        return ", ".join(map(str, parameter.choices))
    assert parameter.bounds is not None
    low, high = parameter.bounds
    return f"[{low}, {high}]"


def parameters(results: StudyResults) -> list[Row]:
    return [
        {
            "Parameter": p.name,
            "Type": p.kind,
            "Domain": _domain(p),
            "Unit": p.unit,
            "Sets": "; ".join(destination.label for destination in p.targets),
        }
        for p in results.problem.parameters
    ]


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


def objectives(results: StudyResults) -> list[Row]:
    shown = units(results)
    return [
        {"Objective": o.label, "Direction": o.direction, "Unit": shown.get(o.label)}
        for o in results.problem.objectives
    ]


def constraints(results: StudyResults) -> list[Row]:
    shown = units(results)
    return [
        {
            "Constraint": c.label,
            "Min": c.min,
            "Max": c.max,
            "Unit": c.unit or shown.get(c.label),
            "Scale": c.scale,
        }
        for c in results.problem.constraints
    ]


def _heading(name: str, unit: str | None) -> str:
    return f"{name} [{unit}]" if unit else name


def trials(results: StudyResults, numbers: Iterable[int] | None = None) -> list[Row]:
    """A row for each trial, or for those numbered ``numbers``: its outcome, its candidate, each
    objective's value and uncertainty, each limit's margin (zero or below when satisfied), its
    time, record and message."""
    shown = units(results)
    chosen = None if numbers is None else set(numbers)
    front = set(results.pareto)
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
            row[_heading(p.name, p.unit)] = trial.candidate.get(p.name)
        row |= _measured(results, trial, shown)
        row["Time [s]"] = trial.elapsed_seconds
        row["Record"] = trial.record
        row["Record SHA-256"] = trial.sha256
        row["Message"] = trial.message
        rows.append(row)
    return rows


def _measured(results: StudyResults, trial: TrialResult, shown: Mapping[str, Any]) -> Row:
    row: Row = {}
    assessment = trial.assessment
    measured = {} if assessment is None else {o.label: o for o in assessment.objectives or ()}
    for objective in results.problem.objectives:
        value = measured.get(objective.label)
        heading = _heading(objective.label, shown.get(objective.label))
        row[heading] = None if value is None else value.value
        row[f"{heading} ±"] = None if value is None else value.uncertainty
    limits = {} if assessment is None else {limit.name: limit for limit in assessment.limits}
    for constraint in results.problem.constraints:
        for bound in ("min", "max"):
            if getattr(constraint, bound) is None:
                continue
            name = f"{constraint.label}:{bound}"
            limit = limits.get(name)
            row[_heading(f"margin {name}", constraint.unit or shown.get(constraint.label))] = (
                None if limit is None else limit.margin
            )
    return row


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


def trial_facts(results: StudyResults, trial: TrialResult) -> list[Row]:
    """How ``trial`` ended and where its record is, a row per fact."""
    rows: list[tuple[str, object]] = [
        ("Outcome", trial.outcome),
        ("On the Pareto front", trial.number in results.pareto),
        ("Time [s]", None if trial.elapsed_seconds is None else round(trial.elapsed_seconds, 1)),
        ("Finished by a resume", trial.recovered),
        ("Record", trial.record),
        ("Record SHA-256", trial.sha256),
        ("Message", trial.message),
    ]
    return [{"": name, "Value": value} for name, value in rows]


def trial_candidate(results: StudyResults, trial: TrialResult) -> list[Row]:
    """Each parameter's value in ``trial``, ``None`` for one it was not proposed (yet)."""
    return [
        {"Parameter": p.name, "Value": trial.candidate.get(p.name), "Unit": p.unit}
        for p in results.problem.parameters
    ]


def trial_objectives(trial: TrialResult) -> list[Row]:
    """Each objective ``trial`` measured, with its uncertainty and unit; none without a record."""
    assessment = trial.assessment
    return [
        {
            "Objective": o.label,
            "Direction": o.direction,
            "Value": o.value,
            "±": o.uncertainty,
            "Unit": o.unit,
        }
        for o in (() if assessment is None else assessment.objectives or ())
    ]


def trial_limits(trial: TrialResult) -> list[Row]:
    """Each limit of ``trial``: the value, the bound, its margin (zero or below when satisfied)."""
    assessment = trial.assessment
    return [
        {
            "Constraint": limit.label,
            "Bound": limit.bound,
            "Limit": limit.limit,
            "Value": limit.value,
            "Margin": limit.margin,
            "Unit": limit.unit,
            "Satisfied": limit.satisfied,
        }
        for limit in (() if assessment is None else assessment.limits)
    ]


def trial_stages(trial: TrialResult) -> list[Row]:
    """How long each stage of ``trial`` took, as its record holds it."""
    return [{"Stage": stage, "Time [s]": seconds} for stage, seconds in trial.stage_seconds.items()]


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
    shown = units(results)
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
            row[_heading(p.name, p.unit)] = trial.candidate.get(p.name)
        for o in results.problem.objectives:
            row[_heading(o.label, shown.get(o.label))] = trial.objectives[o.label]
            base = reference.objectives[o.label]
            change = 0.0 if number == reference.number else found.change[number][o.label]
            row[f"Δ {o.metric} [%]"] = None if base == 0 else 100 * change / abs(base)
        rows.append(row)
    return rows
