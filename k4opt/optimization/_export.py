"""Internal: the results of a study as JSON or CSV, for plotting and analysis elsewhere.

``k4opt study export`` writes one of them; ``docs/optimization.md`` describes both. JSON is the
typed form, :meth:`~k4opt.optimization.StudyResults.to_dict`, with every value as recorded. CSV
is one row per trial, by number, with columns that only the problem decides, so that the exports
of a study line up however many trials it has; a missing value is an empty cell, never 0 or NaN,
a float is written with all its digits, and text with its control characters escaped. Either is
made whole in memory before anything is written.
"""

from __future__ import annotations

import csv
import io
import json
import os
import unicodedata
from pathlib import Path
from typing import TextIO

from k4opt.jsonio import publish
from k4opt.optimization._results import StudyResults, limit_names
from k4opt.optimization._study import trial_directory
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import SUMMARY_NAME

FORMATS = ("csv", "json")


def to_json(results: StudyResults) -> str:
    return json.dumps(results.to_dict(), indent=2, allow_nan=False) + "\n"


def _stages(results: StudyResults) -> list[str]:
    """The stages with a time column: the problem's, in its order, then any other a record
    times, by name."""
    declared = list(results.problem.stages)
    timed = {stage for trial in results.trials for stage in trial.stage_seconds}
    return declared + sorted(timed - set(declared))


def columns(results: StudyResults) -> list[str]:
    """The CSV header: the same for every export of a study with the same stages."""
    return _columns(results, _stages(results))


def _columns(results: StudyResults, stages: list[str]) -> list[str]:
    spec = results.problem
    return [
        "study",
        "trial",
        "status",
        "feasible",
        "pareto",
        "recovered",
        *(f"parameter:{p.geometry.canonical_target}" for p in spec.parameters),
        *(
            f"objective:{o.stage}/{o.metric}{part}"
            for o in spec.objectives
            for part in ("", ":unit", ":uncertainty")
        ),
        *(
            f"limit:{name}:{part}"
            for name in limit_names(spec)
            for part in ("value", "margin", "satisfied")
        ),
        "elapsed_seconds",
        *(f"stage:{stage}:elapsed_seconds" for stage in stages),
        "record",
        "record_sha256",
        "record_path",
        "message",
    ]


def _cell(value: object) -> str:
    """A value as a CSV cell: empty for none, ``true``/``false``, a float with all its digits,
    text with its control characters escaped (``\\x1b``) but for tab and newline, which a quoted
    cell holds: a terminal that shows the file runs none of them. JSON keeps text as it is."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return "".join(
            c.encode("unicode_escape").decode()
            if unicodedata.category(c) == "Cc" and c not in "\t\n"
            else c
            for c in value
        )
    return str(value)


def to_csv(results: StudyResults) -> str:
    spec, front = results.problem, set(results.pareto)
    stages, names = _stages(results), limit_names(spec)  # once, for the header and every row
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_columns(results, stages))
    for trial in results.trials:
        assessment = trial.assessment
        measured = assessment.objectives if assessment is not None else None
        limits = {limit.name: limit for limit in assessment.limits} if assessment else {}
        row: list[object] = [
            results.id,
            trial.number,
            trial.status,
            trial.feasible,
            trial.number in front,
            trial.recovered,
        ]
        row += [trial.candidate.get(p.geometry.canonical_target) for p in spec.parameters]
        for index in range(len(spec.objectives)):
            m = None if measured is None else measured[index]
            row += [None, None, None] if m is None else [m.value, m.unit, m.uncertainty]
        for name in names:
            limit = limits.get(name)
            row += [None] * 3 if limit is None else [limit.value, limit.margin, limit.satisfied]
        row.append(trial.elapsed_seconds)
        row += [trial.stage_seconds.get(stage) for stage in stages]
        path = None if trial.record is None else f"{trial_directory(trial.number)}/{SUMMARY_NAME}"
        row += [trial.record, trial.sha256, path, trial.message]
        writer.writerow([_cell(value) for value in row])
    return buffer.getvalue()


def export(
    results: StudyResults, form: str, out: TextIO, output: str | os.PathLike[str] | None
) -> None:
    """Write the ``form`` (``csv`` or ``json``) export of ``results`` to ``out``, or to the new
    file ``output``: published whole (:func:`k4opt.jsonio.publish`), never over an existing file
    and never into the study directory, which reading leaves as it is."""
    text = to_json(results) if form == "json" else to_csv(results)
    if output is None:
        out.write(text)
        return
    path = Path(os.path.abspath(Path(output).expanduser()))
    if Path(os.path.realpath(path)).is_relative_to(os.path.realpath(results.directory)):
        raise OptimizationError(
            f"{path} is in the study directory {results.directory}; reading a study adds nothing "
            "to it, so export somewhere else"
        )
    try:
        publish(path, text)
    except FileExistsError:
        raise OptimizationError(f"{path} already exists; choose a new --output") from None
    except OSError as exc:
        raise OptimizationError(f"cannot write {path}: {exc}") from exc
