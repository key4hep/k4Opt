"""Internal: a study as its records hold it, read back and checked; nothing of Optuna.

:func:`k4opt.optimization.load_study` returns a :class:`StudyResults`: the problem as
``study.json`` records it, and every trial of the study with what its ``summary.json`` holds,
once that record has been checked against the study and against what the optimizer was told. The
optimizer's adapter says which trials there are and in which state; how a record is read, the
Pareto front and the exported form are decided here, the same for any optimizer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from k4opt.evaluation import Scalar
from k4opt.frozen import FrozenMap, Json, thaw
from k4opt.optimization._study import (
    STUDY_NAME,
    check_candidate_files,
    check_record,
    digest,
    trial_directory,
)
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import (
    SUMMARY_NAME,
    Assessment,
    ProblemConfigError,
    ProblemSpec,
    RecordError,
    parse_problem,
)

RESULTS_FORMAT = "k4opt.study-results/1"
OUTCOMES = ("feasible", "infeasible", "invalid", "failed", "interrupted", "error", "running")


def limit_names(spec: ProblemSpec) -> list[str]:
    """The name of each limit of the problem, as the optimizer knows its constraint value."""
    names = [
        f"{c.stage}/{c.metric}:{bound}"
        for c in spec.constraints
        for bound in ("min", "max")
        if getattr(c, bound) is not None
    ]
    assert len(set(names)) == len(names)  # constraints are unique by stage and metric
    return names


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One trial of a study, as its record holds it.

    ``status`` is the candidate's (``valid``, ``invalid``, ``failed``) as its record says, or
    ``interrupted`` for a trial that ended without a record, ``error`` for one that stopped the
    study (``message`` says why), and ``running`` for one the optimizer still holds as running:
    being evaluated, or left by a session that was killed, until a resume closes it. A running
    trial is not checked and has no measurements; its ``candidate`` holds the values the
    optimizer has proposed so far.

    ``assessment`` (objectives with value, unit and uncertainty; limits with margin and whether
    each is satisfied; feasibility) is that of the record, ``None`` without one, and also for an
    ``error`` trial whose record the optimizer never took. ``record`` and ``sha256`` identify the
    ``summary.json`` read; ``elapsed_seconds`` (the whole evaluation) and ``stage_seconds`` (each
    stage that ran) are the wall-clock times it records, no estimate. ``recovered`` marks a trial
    a resume finished.
    """

    number: int
    status: str
    candidate: Mapping[str, Scalar]
    directory: Path
    assessment: Assessment | None = None
    record: str | None = None
    sha256: str | None = None
    message: str | None = None
    elapsed_seconds: float | None = None
    stage_seconds: Mapping[str, float] = field(default_factory=lambda: FrozenMap({}))
    recovered: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate", FrozenMap(self.candidate))
        object.__setattr__(self, "stage_seconds", FrozenMap(self.stage_seconds))

    @property
    def feasible(self) -> bool | None:
        """Whether every limit is satisfied; ``None`` unless the trial was assessed."""
        return None if self.assessment is None else self.assessment.feasible

    @property
    def outcome(self) -> str:
        """``feasible`` or ``infeasible`` for a valid trial, its status otherwise."""
        if self.status == "valid":
            return "feasible" if self.feasible else "infeasible"
        return self.status

    @property
    def summary_path(self) -> Path | None:
        """The ``summary.json`` the trial was read from, ``None`` without a record."""
        return None if self.record is None else self.directory / SUMMARY_NAME

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-compatible data, as the JSON export holds each trial."""
        return {
            "number": self.number,
            "status": self.status,
            "outcome": self.outcome,
            "recovered": self.recovered,
            "candidate": dict(self.candidate),
            "assessment": None if self.assessment is None else self.assessment.to_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "stage_seconds": dict(self.stage_seconds),
            "message": self.message,
            "record": None
            if self.record is None
            else {
                "id": self.record,
                "path": f"{trial_directory(self.number)}/{SUMMARY_NAME}",
                "sha256": self.sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class StudyResults:
    """A study read back from its directory, as :func:`k4opt.optimization.load_study` gives it.

    ``problem`` is the problem as ``study.json`` records it (parameters with their domains and
    units, objectives with their directions, constraints), ``document`` all of ``study.json``
    as a read-only mapping (what the study is made with, and its optimizer; a database URL in it
    has no password). ``trials`` holds every trial of the study, from all its sessions, by
    number; ``pareto`` the numbers of the trials on the Pareto front of the feasible ``valid``
    ones. ``warnings`` says what the records leave open (evaluator checkouts at another revision)
    and what the study directory holds besides its trials. Nothing in it is mutable or shared
    with the study's storage.
    """

    name: str
    id: str
    created: str
    directory: Path
    problem: ProblemSpec
    document: Mapping[str, Json]
    trials: tuple[TrialResult, ...]
    pareto: tuple[int, ...]
    warnings: tuple[str, ...] = ()

    @property
    def study_path(self) -> Path:
        return self.directory / STUDY_NAME

    @property
    def outcomes(self) -> Mapping[str, int]:
        """How many trials ended how, by :attr:`TrialResult.outcome`; every outcome is a key."""
        return FrozenMap({o: sum(t.outcome == o for t in self.trials) for o in OUTCOMES})

    def trial(self, number: int) -> TrialResult:
        """The trial numbered ``number``; :class:`OptimizationError` if the study has none."""
        for trial in self.trials:
            if trial.number == number:
                return trial
        if self.trials:
            held = f"trials {self.trials[0].number} to {self.trials[-1].number}"
        else:
            held = "no trials yet"
        raise OptimizationError(f"the study has no trial {number}: it has {held}")

    def to_dict(self) -> dict[str, object]:
        """The JSON export (``docs/optimization.md``): plain JSON-compatible data, a fresh copy
        on every call."""
        front = set(self.pareto)
        return {
            "format": RESULTS_FORMAT,
            "directory": str(self.directory),
            "study": thaw(self.document),
            "outcomes": dict(self.outcomes),
            "pareto": list(self.pareto),
            "warnings": list(self.warnings),
            "trials": [t.to_dict() | {"pareto": t.number in front} for t in self.trials],
        }


# --- reading ---------------------------------------------------------------------------------


def recorded_problem(document: Mapping[str, Any]) -> ProblemSpec:
    """The problem ``study.json`` records, once the document holds together: its identity digest
    is that of its parts, and the optimizer it records has the problem's directions and limits.
    No geometry, file or evaluator the problem names is needed."""
    if digest(document) != document.get("identity"):
        raise OptimizationError(
            f"{STUDY_NAME} does not hold together: its identity digest is not that of its parts"
        )
    try:
        spec = parse_problem(document["problem"]["definition"])
    except ProblemConfigError as exc:
        raise OptimizationError(
            f"{STUDY_NAME}: its problem is no problem k4opt reads: {exc}"
        ) from None
    optimizer = document.get("optimizer")
    optimizer = optimizer if isinstance(optimizer, Mapping) else {}
    named = [document.get("id"), document.get("created"), optimizer.get("study")]
    if not all(isinstance(value, str) for value in named):
        raise OptimizationError(
            f"{STUDY_NAME} lacks the study's id, its creation time or the optimizer's study name"
        )
    directions = [o.direction for o in spec.objectives]
    if optimizer.get("directions") != directions:
        raise OptimizationError(
            f"{STUDY_NAME} records the directions {optimizer.get('directions')!r}, but its "
            f"problem's objectives are {directions!r}"
        )
    if optimizer.get("constraints") != limit_names(spec):
        raise OptimizationError(
            f"{STUDY_NAME} records the limits {optimizer.get('constraints')!r}, but its problem's "
            f"are {limit_names(spec)!r}"
        )
    return spec


def recorded(
    directory: Path,
    document: Mapping[str, Any],
    spec: ProblemSpec,
    number: int,
    params: Mapping[str, Any],
    told: object,
    read: dict[str, tuple[int, object]],
) -> tuple[TrialResult, list[str]]:
    """Trial ``number`` as the record the optimizer was told holds it, with the warnings of
    :func:`~k4opt.optimization._study.check_record`.

    ``told`` is the reference the optimizer holds (``id``, ``sha256``, and ``path`` in the
    study), ``params`` the candidate it proposed, and ``read`` what the external references only
    candidates name read for the earlier trials
    (:func:`~k4opt.optimization._study.check_candidate_files`). The record must be
    ``trial-NNNNNN/summary.json`` with exactly those bytes and that id, made for this trial of
    this study with its evaluators, stack and code, and its assessment must recompute from its
    measurements; otherwise :class:`OptimizationError`.
    """
    name = trial_directory(number)
    relative = f"{name}/{SUMMARY_NAME}"
    where = f"trial {number}: {relative}"
    reference = told if isinstance(told, Mapping) else {}
    if reference.get("path") != relative:
        raise OptimizationError(
            f"trial {number}: the optimizer was told a record at {reference.get('path')!r}, not "
            f"at {relative}"
        )
    try:
        data = (directory / relative).read_bytes()
    except OSError as exc:
        raise OptimizationError(
            f"{where} cannot be read ({exc.strerror or exc}), though the optimizer was told it: "
            "nothing shows what the trial's values rest on"
        ) from None
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != reference.get("sha256"):
        raise OptimizationError(f"{where} is not the record the optimizer was told (other bytes)")
    try:
        summary = json.loads(data)
    except ValueError as exc:
        raise OptimizationError(f"{where} is not JSON: {exc}") from None
    warnings = [f"trial {number}: {w}" for w in check_record(document, summary, number, params)]
    if not isinstance(summary.get("id"), str) or summary["id"] != reference.get("id"):
        raise OptimizationError(
            f"{where} has the id {summary.get('id')!r}, but the optimizer was told record "
            f"{reference.get('id')!r}"
        )
    check_candidate_files(read, summary, number)
    try:
        assessment = Assessment.from_summary(spec, summary)
    except RecordError as exc:
        raise OptimizationError(f"{where}: {exc}") from None
    stages = summary.get("stages")
    if not isinstance(stages, Mapping) or not all(isinstance(e, Mapping) for e in stages.values()):
        raise OptimizationError(f"{where}: stages must map each stage to its record")
    timed = {
        stage: _seconds(entry.get("elapsed_seconds"), f"{where}: stages.{stage}.elapsed_seconds")
        for stage, entry in stages.items()
    }
    references = [parameter.geometry.canonical_target for parameter in spec.parameters]
    result = TrialResult(
        number=number,
        status=summary["status"],
        candidate={key: summary["candidate"][key] for key in references},
        directory=directory / name,
        assessment=assessment,
        record=summary["id"],
        sha256=sha256,
        message=summary.get("message"),
        elapsed_seconds=_seconds(summary.get("elapsed_seconds"), f"{where}: elapsed_seconds"),
        stage_seconds={stage: seconds for stage, seconds in timed.items() if seconds is not None},
    )
    return result, warnings


def _seconds(value: object, where: str) -> float | None:
    """A time a record holds: ``None``, or a finite number of seconds, zero or more."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise OptimizationError(f"{where} is {value!r}, not a time in seconds")
    return value


def pareto_front(trials: Sequence[TrialResult], spec: ProblemSpec) -> tuple[int, ...]:
    """The numbers of the feasible ``valid`` trials that no other one dominates.

    A trial dominates another when it is at least as good in every objective, in that
    objective's direction, and better in one; trials with equal values do not dominate each
    other, so all of them stay on the front. The values compared are the recorded ones,
    uncertainties play no part.
    """
    signs = [1 if o.direction == "minimize" else -1 for o in spec.objectives]
    points = [
        (t.number, [sign * m.value for sign, m in zip(signs, t.assessment.objectives, strict=True)])
        for t in trials
        if t.status == "valid"
        and t.feasible is True
        and t.assessment is not None
        and t.assessment.objectives is not None
    ]

    def dominates(a: list[int | float], b: list[int | float]) -> bool:
        return all(x <= y for x, y in zip(a, b, strict=True)) and any(
            x < y for x, y in zip(a, b, strict=True)
        )

    return tuple(
        sorted(
            number
            for number, values in points
            if not any(dominates(other, values) for _, other in points)
        )
    )
