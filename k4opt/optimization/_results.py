"""Internal: a study as its records hold it, read back and checked; nothing of Optuna.

:func:`k4opt.optimization.load_study` returns a :class:`StudyResults`: the problem as
``study.json`` records it, and every trial of the study with what its ``summary.json`` holds,
once that record has been checked against the study and against what the optimizer was told. The
optimizer's adapter says which trials there are and in which state; how a record is read, the
Pareto front and the JSON form are decided here, the same for any optimizer. The JSON form
(:meth:`StudyResults.to_dict`) holds no location, so it can leave the machine (an export, a
publication), and :meth:`StudyResults.from_dict` reads it back, checked, without Optuna.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, NoReturn

from k4opt.evaluation import Scalar
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.optimization._identity import digest
from k4opt.optimization._study import (
    STUDY_NAME,
    check_candidate_files,
    check_record,
    portable,
    trial_directory,
)
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import Constraint, Objective, ProblemConfigError, ProblemSpec, parse_problem
from k4opt.record import SUMMARY_NAME, Assessment, Limit, ObjectiveValue, RecordError

RESULTS_FORMAT = "k4opt.study-results/1"
OUTCOMES = (
    "feasible",
    "infeasible",
    "invalid",
    "failed",
    "rejected",
    "interrupted",
    "error",
    "running",
)
"""How a trial ended (:attr:`TrialResult.outcome`), in the order reports list them."""


def limit_names(spec: ProblemSpec) -> list[str]:
    """The name of each limit of the problem, as the optimizer knows its constraint value."""
    names = [
        f"{c.label}:{bound}"
        for c in spec.constraints
        for bound in ("min", "max")
        if getattr(c, bound) is not None
    ]
    assert len(set(names)) == len(names)  # constraints are unique by metric reference
    return names


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One trial of a study, as its record holds it.

    ``status`` is the candidate's (``valid``, ``invalid``, ``failed``) as its record says, or
    ``rejected`` for one whose candidate an evaluator's ``check`` rejected before anything of it
    was created (no record, no directory; ``message`` names the evaluation and says why),
    ``interrupted`` for a trial that ended without a record, ``error`` for one that stopped the
    study (``message`` says why), and ``running`` for one the optimizer still holds as running:
    being evaluated, or left by a session that was killed, until a resume closes it. A running
    trial is not checked and has no measurements; its ``candidate`` holds the values the
    optimizer has proposed so far.

    ``assessment`` (objectives with value, unit and uncertainty; limits with margin and whether
    each is satisfied; feasibility) is that of the record, ``None`` without one, and also for an
    ``error`` trial whose record the optimizer never took. ``record`` and ``sha256`` identify the
    ``summary.json`` read, which is ``trial-NNNNNN/summary.json`` in the study directory
    (:meth:`StudyResults.record_path`); ``elapsed_seconds`` (the whole evaluation) and
    ``stage_seconds`` (each stage that ran, by label: ``gamma/simulation``) are the wall-clock
    times it records, no estimate. ``diagnostics`` holds, as the record does, each diagnostic that
    did not succeed (``task``, ``status``, ``message``): what the status leaves out. ``recovered``
    marks a trial a resume finished.
    """

    number: int
    status: str
    candidate: Mapping[str, Scalar]
    assessment: Assessment | None = None
    record: str | None = None
    sha256: str | None = None
    message: str | None = None
    elapsed_seconds: float | None = None
    stage_seconds: Mapping[str, float] = field(default_factory=lambda: FrozenMap({}))
    recovered: bool = False
    diagnostics: tuple[Mapping[str, str | None], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate", FrozenMap(self.candidate))
        object.__setattr__(self, "stage_seconds", FrozenMap(self.stage_seconds))
        diagnostics = tuple(FrozenMap(entry) for entry in self.diagnostics)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def feasible(self) -> bool | None:
        """Whether every limit is satisfied; ``None`` unless the trial was assessed."""
        return None if self.assessment is None else self.assessment.feasible

    @property
    def objectives(self) -> Mapping[str, int | float] | None:
        """The objective values of a ``valid`` trial, by label, in the problem's order; ``None``
        for any other."""
        if self.assessment is None or self.assessment.objectives is None:
            return None
        return FrozenMap({o.label: o.value for o in self.assessment.objectives})

    @property
    def outcome(self) -> str:
        """``feasible`` or ``infeasible`` for a valid trial, its status otherwise."""
        if self.status == "valid":
            return "feasible" if self.feasible else "infeasible"
        return self.status

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
            "diagnostics": [dict(entry) for entry in self.diagnostics],
            "record": None if self.record is None else {"id": self.record, "sha256": self.sha256},
        }


@dataclass(frozen=True, slots=True)
class StudyResults:
    """A study as :func:`k4opt.optimization.load_study` reads it back from its directory, or as
    :meth:`from_dict` reads its JSON form.

    ``problem`` is the problem as ``study.json`` records it (parameters with their domains and
    units, objectives with their directions, constraints), ``document`` ``study.json`` as a
    read-only mapping (what the study is made with, and its optimizer; a database URL in it has
    no password), all of it when read from the directory, without its locations when read from
    the JSON form. ``trials`` holds every trial of the study, from all its sessions, by number;
    ``pareto`` the numbers of the trials on the Pareto front of the feasible ``valid`` ones.
    ``directory`` is the study directory it was read from, ``None`` for a study read from its JSON
    form; ``warnings`` says what the records leave open (evaluator checkouts at another revision)
    and what the study directory holds besides its trials. Nothing in it is mutable or shared
    with the study's storage.
    """

    name: str
    id: str
    created: str
    problem: ProblemSpec
    document: Mapping[str, Json]
    trials: tuple[TrialResult, ...]
    pareto: tuple[int, ...]
    directory: Path | None = None
    warnings: tuple[str, ...] = ()
    _numbered: Mapping[int, TrialResult] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_numbered", FrozenMap({t.number: t for t in self.trials}))

    @property
    def study_path(self) -> Path | None:
        """Its ``study.json``, ``None`` for a study read from its JSON form."""
        return None if self.directory is None else self.directory / STUDY_NAME

    def record_path(self, trial: TrialResult) -> Path | None:
        """The ``summary.json`` ``trial`` was read from, ``None`` without a record or without the
        study directory."""
        if trial.record is None or self.directory is None:
            return None
        return self.directory / trial_directory(trial.number) / SUMMARY_NAME

    @property
    def outcomes(self) -> Mapping[str, int]:
        """How many trials ended how, by :attr:`TrialResult.outcome`; every outcome is a key."""
        return FrozenMap({o: sum(t.outcome == o for t in self.trials) for o in OUTCOMES})

    def trial(self, number: int) -> TrialResult:
        """The trial numbered ``number``; :class:`OptimizationError` if the study has none."""
        found = self._numbered.get(number)
        if found is not None:
            return found
        if self.trials:
            held = f"trials {self.trials[0].number} to {self.trials[-1].number}"
        else:
            held = "no trials yet"
        raise OptimizationError(f"the study has no trial {number}: it has {held}")

    def to_dict(self) -> dict[str, object]:
        """The JSON form of the study (``docs/optimization.md``): plain JSON-compatible data
        without any location, a fresh copy on every call. :meth:`from_dict` reads it back."""
        front = set(self.pareto)
        return {
            "format": RESULTS_FORMAT,
            "study": portable(self.document),
            "outcomes": dict(self.outcomes),
            "pareto": list(self.pareto),
            "trials": [t.to_dict() | {"pareto": t.number in front} for t in self.trials],
        }

    def to_json(self) -> str:
        """:meth:`to_dict` as indented JSON text: the same study gives the same text every time;
        what ``k4opt study export --format json`` writes and a publication holds."""
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, data: object) -> StudyResults:
        """The study whose JSON form (:meth:`to_dict`) ``data`` is, checked.

        What is checked is the form, not the measurements, whose records it does not hold: the
        identity digest of its ``study.json`` (so its problem definition is the study's), each
        trial's status, outcome, candidate, assessment and record against the problem and each
        other, and the outcome counts and the Pareto front, recomputed. Raises
        :class:`OptimizationError` naming what does not hold up. The study has no ``directory``
        and no ``warnings``.
        """
        return _from_dict(data)


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


def record_of(
    directory: Path,
    document: Mapping[str, Any],
    spec: ProblemSpec,
    number: int,
    params: Mapping[str, Any],
    read: dict[str, tuple[int, object]],
) -> tuple[TrialResult, list[str]]:
    """Trial ``number`` as its record ``trial-NNNNNN/summary.json`` holds it, checked before
    anything trusts it, with the warnings of :func:`~k4opt.optimization._study.check_record`.

    ``params`` is the candidate the optimizer proposed, and ``read`` what the external references
    only candidates name read for the earlier trials
    (:func:`~k4opt.optimization._study.check_candidate_files`). The record must be made for this
    trial of this study with its evaluators, stack and code, and its assessment must recompute
    from its measurements; otherwise :class:`OptimizationError`. The one reading of a trial's
    record, live (before the optimizer is told it) and when a study is read back
    (:func:`recorded`)."""
    name = trial_directory(number)
    where = f"trial {number}: {name}/{SUMMARY_NAME}"
    try:
        data = (directory / name / SUMMARY_NAME).read_bytes()
    except OSError as exc:
        raise OptimizationError(f"{where} cannot be read ({exc.strerror or exc})") from None
    try:
        summary = json.loads(data)
    except ValueError as exc:
        raise OptimizationError(f"{where} is not JSON: {exc}") from None
    warnings = [f"trial {number}: {w}" for w in check_record(document, summary, number, params)]
    if not isinstance(summary.get("id"), str):
        raise OptimizationError(f"{where} has no record id")
    check_candidate_files(read, summary, number)
    try:
        assessment = Assessment.from_summary(spec, summary)
    except RecordError as exc:
        raise OptimizationError(f"{where}: {exc}") from None
    timed = {}
    for sample, part in summary["samples"].items():  # checked by from_summary
        for stage, entry in part["stages"].items():
            at = f"{where}: samples.{sample}.stages.{stage}.elapsed_seconds"
            timed[f"{sample}/{stage}"] = _seconds(entry.get("elapsed_seconds"), at)
    references = [parameter.name for parameter in spec.parameters]
    result = TrialResult(
        number=number,
        status=summary["status"],
        candidate={key: summary["candidate"][key] for key in references},
        assessment=assessment,
        record=summary["id"],
        sha256=hashlib.sha256(data).hexdigest(),
        message=summary.get("message"),
        elapsed_seconds=_seconds(summary.get("elapsed_seconds"), f"{where}: elapsed_seconds"),
        stage_seconds={stage: seconds for stage, seconds in timed.items() if seconds is not None},
        diagnostics=summary["diagnostics"],  # checked by from_summary
    )
    return result, warnings


def recorded(
    directory: Path,
    document: Mapping[str, Any],
    spec: ProblemSpec,
    number: int,
    params: Mapping[str, Any],
    told: object,
    read: dict[str, tuple[int, object]],
) -> tuple[TrialResult, list[str]]:
    """Trial ``number`` as the record the optimizer was told holds it (:func:`record_of`).

    ``told`` is the reference the optimizer holds (``id``, ``sha256``, and ``path`` in the
    study): the record must be ``trial-NNNNNN/summary.json`` with exactly those bytes and that
    id; otherwise :class:`OptimizationError`.
    """
    relative = f"{trial_directory(number)}/{SUMMARY_NAME}"
    where = f"trial {number}: {relative}"
    reference = told if isinstance(told, Mapping) else {}
    if reference.get("path") != relative:
        raise OptimizationError(
            f"trial {number}: the optimizer was told a record at {reference.get('path')!r}, not "
            f"at {relative}"
        )
    if not (directory / relative).is_file():
        raise OptimizationError(
            f"{where} cannot be read (it is gone), though the optimizer was told it: nothing "
            "shows what the trial's values rest on"
        )
    result, warnings = record_of(directory, document, spec, number, params, read)
    if result.sha256 != reference.get("sha256"):
        raise OptimizationError(f"{where} is not the record the optimizer was told (other bytes)")
    if result.record != reference.get("id"):
        raise OptimizationError(
            f"{where} has the id {result.record!r}, but the optimizer was told record "
            f"{reference.get('id')!r}"
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

    Each distinct point (the values, each to be minimized) is judged once, in sorted order: a
    point can only be dominated by one before it, and if it is, then by one on the front found
    so far. With two objectives that is the last one on the front, so the front takes a sort
    and one pass; with one, the first point alone is on it.
    """
    signs = [1 if o.direction == "minimize" else -1 for o in spec.objectives]
    points: dict[tuple[int | float, ...], list[int]] = {}
    for t in trials:
        if t.status != "valid" or t.feasible is not True or t.assessment is None:
            continue
        measured = t.assessment.objectives
        if measured is not None:
            point = tuple(sign * m.value for sign, m in zip(signs, measured, strict=True))
            points.setdefault(point, []).append(t.number)
    front: list[tuple[int | float, ...]] = []
    for point in sorted(points):
        if len(point) == 2:  # the front so far has falling second values: the last is least
            dominated = bool(front) and front[-1][1] <= point[1]
        else:
            dominated = any(all(x <= y for x, y in zip(o, point, strict=True)) for o in front)
        if not dominated:
            front.append(point)
    return tuple(sorted(number for point in front for number in points[point]))


# --- the JSON form, read back ----------------------------------------------------------------

_KEYS = ("format", "study", "outcomes", "pareto", "trials")
_TRIAL = (
    "number",
    "status",
    "outcome",
    "recovered",
    "candidate",
    "assessment",
    "elapsed_seconds",
    "stage_seconds",
    "message",
    "diagnostics",
    "record",
    "pareto",
)
_RECORDED = ("valid", "invalid", "failed")  # with a record, always; an error trial may have one
_UNRECORDED = ("rejected", "interrupted", "running")  # never
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OBJECTIVE = tuple(f.name for f in fields(ObjectiveValue))
_LIMIT = tuple(f.name for f in fields(Limit))
_JUDGED = ("value", "uncertainty", "margin", "normalized", "satisfied")  # of a valid result only
_BOUNDS: tuple[Literal["min", "max"], ...] = ("min", "max")  # in the order of limit_names


def _refuse(where: str, problem: str) -> NoReturn:
    raise OptimizationError(f"not the JSON form of a study: {where} {problem}")


def _mapping(value: object, where: str, keys: Sequence[str] | None) -> Mapping[str, Json]:
    """``value`` as an object, with exactly ``keys`` unless ``None``."""
    if not isinstance(value, Mapping):
        _refuse(where, "is not an object")
    if keys is not None and set(value) != set(keys):
        missing, extra = sorted(set(keys) - set(value)), sorted(set(value) - set(keys))
        _refuse(where, f"lacks {missing} and has {extra}, other than the keys {list(keys)}")
    return value


def _number(value: object, where: str, *, optional: bool = False) -> int | float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(where, f"is {value!r}, not a number")
    return value


def _text(value: object, where: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        _refuse(where, f"is {value!r}, not text")
    return value


def _from_dict(data: object) -> StudyResults:
    """:meth:`StudyResults.from_dict`."""
    form = _mapping(freeze(data, "the JSON form of a study", OptimizationError), "it", _KEYS)
    if form["format"] != RESULTS_FORMAT:
        _refuse("its format", f"is {form['format']!r}, not {RESULTS_FORMAT!r}")
    study = _mapping(form["study"], "study", None)
    document = thaw(study)
    assert isinstance(document, dict)
    spec = recorded_problem(document)  # its identity digest is that of its parts
    if portable(document) != document:
        _refuse("study", "holds more of study.json than its JSON form keeps")
    entries = form["trials"]
    if not isinstance(entries, tuple):
        _refuse("trials", "is not a list")
    trials: list[TrialResult] = []
    flags: list[object] = []
    for index, entry in enumerate(entries):
        trial, flag = _trial(entry, spec, f"trials[{index}]")
        if trials and trial.number <= trials[-1].number:
            _refuse(f"trials[{index}]", f"is trial {trial.number}, after trial {trials[-1].number}")
        trials.append(trial)
        flags.append(flag)
    results = StudyResults(
        name=document["optimizer"]["study"],  # checked by recorded_problem
        id=document["id"],
        created=document["created"],
        problem=spec,
        document=study,
        trials=tuple(trials),
        pareto=pareto_front(trials, spec),
    )
    if form["pareto"] != results.pareto:
        _refuse("pareto", f"is {form['pareto']!r}, but the front of its trials is {results.pareto}")
    for trial, flag in zip(trials, flags, strict=True):
        if flag != (trial.number in results.pareto):
            _refuse(f"trial {trial.number}'s pareto", f"is {flag!r}, other than its front")
    outcomes = dict(_mapping(form["outcomes"], "outcomes", OUTCOMES))
    if outcomes != dict(results.outcomes):
        _refuse("outcomes", f"are {outcomes}, but its trials {dict(results.outcomes)}")
    return results


def _trial(entry: object, spec: ProblemSpec, where: str) -> tuple[TrialResult, object]:
    """One trial of the JSON form, as :func:`record_of` and the optimizer's adapter make it, and
    its ``pareto`` flag."""
    data = _mapping(entry, where, _TRIAL)
    number, status = data["number"], data["status"]
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        _refuse(f"{where}.number", f"is {number!r}, not a trial number")
    where = f"trial {number}"
    if status not in (*_RECORDED, *_UNRECORDED, "error"):
        _refuse(f"{where}'s status", f"is {status!r}, not a status of a trial")
    assert isinstance(status, str)
    record_id, sha256 = _record(data["record"], where)
    held = record_id is not None
    if (status in _RECORDED and not held) or (status in _UNRECORDED and held):
        _refuse(where, f"is {status} {'with' if held else 'without'} a record")
    stage_seconds: dict[str, float] = {}
    for stage, value in _mapping(data["stage_seconds"], f"{where}'s stage_seconds", None).items():
        seconds = _seconds(value, f"{where}'s stage_seconds.{stage}")
        if seconds is None:
            _refuse(f"{where}'s stage_seconds.{stage}", "is null")
        stage_seconds[stage] = seconds
    diagnostics = _diagnostics(data["diagnostics"], where)
    elapsed = _seconds(data["elapsed_seconds"], f"{where}'s elapsed_seconds")
    if not held and (elapsed is not None or stage_seconds or diagnostics):
        _refuse(where, "has times or diagnostics of a record, without a record")
    if not isinstance(data["recovered"], bool):
        _refuse(f"{where}'s recovered", f"is {data['recovered']!r}, not true or false")
    trial = TrialResult(
        number=number,
        status=status,
        candidate=_candidate(data["candidate"], spec, held, where),
        assessment=_assessment(data["assessment"], spec, status, where),
        record=record_id,
        sha256=sha256,
        message=_text(data["message"], f"{where}'s message", optional=True),
        elapsed_seconds=elapsed,
        stage_seconds=stage_seconds,
        recovered=data["recovered"],
        diagnostics=diagnostics,
    )
    if data["outcome"] != trial.outcome:
        _refuse(f"{where}'s outcome", f"is {data['outcome']!r}, its status gives {trial.outcome!r}")
    return trial, data["pareto"]


def _record(value: object, where: str) -> tuple[str | None, str | None]:
    """The id and SHA-256 of the record a trial was read from; ``(None, None)`` without one."""
    if value is None:
        return None, None
    named = _mapping(value, f"{where}'s record", ("id", "sha256"))
    record_id, sha256 = named["id"], named["sha256"]
    if not isinstance(record_id, str):
        _refuse(f"{where}'s record id", f"is {record_id!r}, not text")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        _refuse(f"{where}'s record sha256", f"is {sha256!r}, not a SHA-256")
    return record_id, sha256


def _diagnostics(value: object, where: str) -> tuple[Mapping[str, str | None], ...]:
    """Each diagnostic of a trial's record that did not succeed: its task, status and message."""
    if not isinstance(value, tuple):
        _refuse(f"{where}'s diagnostics", "is not a list")
    found: list[Mapping[str, str | None]] = []
    for index, entry in enumerate(value):
        at = f"{where}'s diagnostics[{index}]"
        data = _mapping(entry, at, ("task", "status", "message"))
        found.append(
            {
                "task": _text(data["task"], f"{at}.task"),
                "status": _text(data["status"], f"{at}.status"),
                "message": _text(data["message"], f"{at}.message", optional=True),
            }
        )
    return tuple(found)


def _candidate(value: object, spec: ProblemSpec, held: bool, where: str) -> dict[str, Scalar]:
    """The candidate of a trial: with a record, a value of every parameter, in the problem's
    order; without one (running, rejected, interrupted, an error), what the optimizer proposed so
    far, in that order."""
    candidate = _mapping(value, f"{where}'s candidate", None)
    names = [parameter.name for parameter in spec.parameters]
    expected = names if held else [name for name in names if name in candidate]
    if list(candidate) != expected:
        _refuse(f"{where}'s candidate", f"names {list(candidate)}, not the parameters {expected}")
    for name, item in candidate.items():
        if item is None or isinstance(item, (Mapping, tuple)):
            _refuse(f"{where}'s candidate.{name}", f"is {item!r}, not a value")
    return dict(candidate)  # type: ignore[arg-type]  # scalars, checked above


def _assessment(value: object, spec: ProblemSpec, status: str, where: str) -> Assessment | None:
    """The assessment of a trial with a valid, invalid or failed record, against the problem:
    the objectives of a valid one, each bound of each constraint, and whether all are kept."""
    if status not in _RECORDED:
        if value is not None:
            _refuse(f"{where}'s assessment", f"is given, but the trial is {status}")
        return None
    data = _mapping(value, f"{where}'s assessment", ("feasible", "objectives", "limits"))
    valid = status == "valid"
    objectives: tuple[ObjectiveValue, ...] | None = None
    measured = data["objectives"]
    if valid:
        if not isinstance(measured, tuple) or len(measured) != len(spec.objectives):
            _refuse(f"{where}'s objectives", f"are not one value of each of {len(spec.objectives)}")
        objectives = tuple(
            _objective(entry, objective, f"{where}'s objectives[{index}]")
            for index, (entry, objective) in enumerate(zip(measured, spec.objectives, strict=True))
        )
    elif measured is not None:
        _refuse(f"{where}'s objectives", f"are given, but the trial is {status}")
    bounds: list[tuple[Constraint, Literal["min", "max"]]] = [
        (constraint, bound)
        for constraint in spec.constraints
        for bound in _BOUNDS
        if getattr(constraint, bound) is not None
    ]
    entries = data["limits"]
    if not isinstance(entries, tuple) or len(entries) != len(bounds):
        _refuse(f"{where}'s limits", f"are not one of each of the limits {limit_names(spec)}")
    limits = tuple(
        _limit(entry, constraint, bound, valid, f"{where}'s limits[{index}]")
        for index, (entry, (constraint, bound)) in enumerate(zip(entries, bounds, strict=True))
    )
    feasible = all(limit.satisfied for limit in limits) if valid else None
    if data["feasible"] != feasible:
        _refuse(f"{where}'s feasible", f"is {data['feasible']!r}, its limits give {feasible!r}")
    return Assessment(objectives, limits, feasible)


def _objective(entry: object, objective: Objective, where: str) -> ObjectiveValue:
    data = _mapping(entry, where, _OBJECTIVE)
    named = (data["evaluator"], data["sample"], data["stage"], data["metric"], data["direction"])
    if named != (
        objective.evaluator,
        objective.sample,
        objective.stage,
        objective.metric,
        objective.direction,
    ):
        _refuse(where, f"is not the objective {objective.label} ({objective.direction})")
    value = _number(data["value"], f"{where}.value")
    assert value is not None
    return ObjectiveValue(
        objective.evaluator,
        objective.stage,
        objective.metric,
        objective.direction,
        value,
        _text(data["unit"], f"{where}.unit", optional=True),
        _number(data["uncertainty"], f"{where}.uncertainty", optional=True),
        objective.sample,
    )


def _limit(
    entry: object, constraint: Constraint, bound: Literal["min", "max"], valid: bool, where: str
) -> Limit:
    data = _mapping(entry, where, _LIMIT)
    scale = 1 if constraint.scale is None else constraint.scale
    named = (data["evaluator"], data["sample"], data["stage"], data["metric"], data["bound"])
    expected = (constraint.evaluator, constraint.sample, constraint.stage, constraint.metric, bound)
    if named != expected or (data["limit"], data["scale"]) != (getattr(constraint, bound), scale):
        _refuse(where, f"is not the limit {constraint.label}:{bound} of the problem")
    unit = _text(data["unit"], f"{where}.unit", optional=True)
    limit = getattr(constraint, bound)
    if not valid:
        if any(data[key] is not None for key in _JUDGED):
            _refuse(where, "judges a value, but the trial has no valid result")
        return Limit(
            constraint.evaluator,
            constraint.stage,
            constraint.metric,
            bound,
            limit,
            unit,
            sample=constraint.sample,
            scale=scale,
        )
    if not isinstance(data["satisfied"], bool):
        _refuse(f"{where}.satisfied", f"is {data['satisfied']!r}, not true or false")
    value = _number(data["value"], f"{where}.value")
    margin = _number(data["margin"], f"{where}.margin", optional=True)
    normalized = _number(data["normalized"], f"{where}.normalized", optional=True)
    return Limit(
        constraint.evaluator,
        constraint.stage,
        constraint.metric,
        bound,
        limit,
        unit,
        value,
        _number(data["uncertainty"], f"{where}.uncertainty", optional=True),
        None if margin is None else float(margin),
        data["satisfied"],
        constraint.sample,
        scale,
        None if normalized is None else float(normalized),
    )
