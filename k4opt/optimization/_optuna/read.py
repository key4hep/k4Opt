"""Internal: a study read back by :func:`k4opt.optimization.load_study`, without changing it.

The storage is opened so that it refuses every write
(:func:`~k4opt.optimization._optuna.storage.existing_storage`), and nothing that makes, resumes or
recovers a study is called. Each trial's state comes from Optuna; its record is read and checked
by :func:`k4opt.optimization._results.recorded`, and must agree with what Optuna was told.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import BaseStorage
from optuna.trial import FrozenTrial, TrialState

from k4opt.frozen import freeze
from k4opt.optimization._optuna.optimizer import check_optuna_study, study_attributes
from k4opt.optimization._optuna.space import constraint_values, objective_values
from k4opt.optimization._optuna.storage import close, existing_storage, leaving
from k4opt.optimization._results import (
    StudyResults,
    TrialResult,
    pareto_front,
    recorded,
    recorded_problem,
)
from k4opt.optimization._study import STUDY_NAME, read_study, trial_directories, trial_directory
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import ProblemSpec


def _existing_study(document: Mapping[str, Any], storage: BaseStorage) -> optuna.Study:
    """The Optuna study ``document`` describes, as ``storage`` holds it; one that was never set
    up, or only in part, is refused rather than set up."""
    name = document["optimizer"]["study"]
    try:
        study = optuna.load_study(study_name=name, storage=storage)
    except KeyError:
        raise OptimizationError(
            f"the optimizer's storage holds no study {name!r}: the process that made the study "
            "stopped before setting it up; k4opt optimize --resume sets it up"
        ) from None
    missing = [key for key in study_attributes(document) if key not in study.user_attrs]
    if missing:
        raise OptimizationError(
            f"the optimizer's study {name!r} lacks {', '.join(missing)}: the process that made it "
            "stopped while setting it up; k4opt optimize --resume finishes that"
        )
    check_optuna_study(study, document)
    return study


def _check_told_values(frozen: FrozenTrial, result: TrialResult) -> None:
    """A COMPLETE trial holds what its record gives: the objective values, the feasibility and a
    constraint value for each limit."""
    assessment = result.assessment
    assert assessment is not None
    where = f"trial {frozen.number}: the optimizer holds"
    values, constraints = objective_values(assessment), constraint_values(assessment)
    if frozen.values != values:
        raise OptimizationError(f"{where} the values {frozen.values}, its record gives {values}")
    if frozen.user_attrs.get("k4opt.feasible") != assessment.feasible:
        raise OptimizationError(
            f"{where} feasible={frozen.user_attrs.get('k4opt.feasible')!r}, its record gives "
            f"{assessment.feasible!r}"
        )
    if frozen.constraints != constraints:
        raise OptimizationError(
            f"{where} the constraint values {frozen.constraints}, its record gives {constraints}"
        )


def _trial_result(
    directory: Path,
    document: Mapping[str, Any],
    spec: ProblemSpec,
    frozen: FrozenTrial,
    read: dict[str, tuple[int, object]],
) -> tuple[TrialResult, list[str]]:
    """Trial ``frozen`` as its record holds it, checked against what the optimizer was told, with
    the warnings of its record.

    A RUNNING trial is ``running``, unchecked and without a record, even if one is already on
    disk: the optimizer has not taken it. A trial told a record shows that record, which must
    agree with the trial's state (COMPLETE for ``valid``, FAIL otherwise) and values, except
    that a FAIL trial with a ``valid`` record and ``k4opt.error`` is ``error``, without a
    result: its values never reached the optimizer. A FAIL trial without a record is ``error``
    with ``k4opt.error`` and ``interrupted`` without; any other trial was not made by k4opt.
    """
    number, attrs = frozen.number, frozen.user_attrs
    references = [parameter.geometry.canonical_target for parameter in spec.parameters]
    unknown = [key for key in frozen.params if key not in references]
    if unknown:
        raise OptimizationError(
            f"trial {number} has the parameter(s) {', '.join(unknown)}, which the problem does "
            "not have"
        )
    candidate = {key: frozen.params[key] for key in references if key in frozen.params}
    recovered = attrs.get("k4opt.recovered") is True
    own = directory / trial_directory(number)
    if frozen.state == TrialState.RUNNING:
        return TrialResult(number, "running", candidate, own, recovered=recovered), []
    told, error = attrs.get("k4opt.record"), attrs.get("k4opt.error")
    if told is None:
        ours = "k4opt.dir" in attrs or recovered
        if frozen.state != TrialState.FAIL or "k4opt.status" in attrs or not ours:
            raise OptimizationError(
                f"trial {number} is {frozen.state.name} without the record k4opt tells a trial "
                "with, or any mark of k4opt, so it was not made by k4opt optimize"
            )
        status, message = ("interrupted", None) if error is None else ("error", str(error))
        return TrialResult(number, status, candidate, own, message=message, recovered=recovered), []
    result, warnings = recorded(directory, document, spec, number, frozen.params, told, read)
    if attrs.get("k4opt.status") != result.status:
        raise OptimizationError(
            f"trial {number}: the optimizer was told the status {attrs.get('k4opt.status')!r}, "
            f"but its record is {result.status!r}"
        )
    if frozen.state == TrialState.COMPLETE and result.status == "valid":
        _check_told_values(frozen, result)
    elif frozen.state == TrialState.FAIL and result.status == "valid" and error is not None:
        result = replace(result, status="error", assessment=None, message=str(error))
    elif frozen.state != TrialState.FAIL or result.status == "valid":
        raise OptimizationError(
            f"trial {number} is {frozen.state.name}, but its record is {result.status!r}"
        )
    return replace(result, recovered=recovered), warnings


def load(directory: str | os.PathLike[str], storage: str | None) -> StudyResults:
    """The study in ``directory`` as its records hold it, read without changing anything; see
    :func:`k4opt.optimization.load_study`."""
    path = Path(os.path.abspath(Path(directory).expanduser()))
    if not (path / STUDY_NAME).is_file():
        raise OptimizationError(f"{path} is not a study directory: it holds no {STUDY_NAME}")
    document = read_study(path)
    spec = recorded_problem(document)
    trials: list[TrialResult] = []
    warnings: list[str] = []
    database: BaseStorage | None = None
    try:
        database = existing_storage(path, document, storage)
        study = _existing_study(document, database)
        read: dict[str, tuple[int, object]] = {}
        held = sorted(study.get_trials(deepcopy=False), key=lambda frozen: frozen.number)
        for frozen in held:
            result, notes = _trial_result(path, document, spec, frozen, read)
            trials.append(result)
            warnings += notes
    except Exception as exc:
        replacement, secret = leaving(exc, storage, what="the study cannot be read")
        if replacement is None:
            raise
        if secret:  # nothing of the original, which names it, is kept for a traceback
            raise replacement from None
        raise replacement from exc
    finally:
        close(database)  # everything is read: the snapshot holds nothing of the storage
    known = {frozen.user_attrs.get("k4opt.dir", trial_directory(frozen.number)) for frozen in held}
    warnings += [
        f"{name} in {path} belongs to no trial of the study, so it is not shown"
        for name in trial_directories(path)
        if name not in known
    ]
    document_ = freeze(document, STUDY_NAME, OptimizationError)
    assert isinstance(document_, Mapping)
    return StudyResults(
        name=document["optimizer"]["study"],
        id=document["id"],
        created=document["created"],
        directory=path,
        problem=spec,
        document=document_,
        trials=tuple(trials),
        pareto=pareto_front(trials, spec),
        warnings=tuple(warnings),
    )
