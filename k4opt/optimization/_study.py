"""Internal: the directory of a study, its ``study.json`` and the records of its trials; nothing of
Optuna.

A study directory holds ``study.json``, written once with what the study is made with, the
optimizer's storage, a ``.lock`` while a process works on the study (:mod:`._lock`), and a
``trial-NNNNNN`` directory for each trial that got as far as its evaluation, laid out as
``k4opt evaluate`` lays out its output; a trial whose candidate an evaluator rejected has none.
:func:`check_record` refuses the record of a trial that is not its study's, by the comparison of
:mod:`._identity`. A file that only candidates name (a ``ref`` parameter) is not known before they
do: each record must read it as every earlier record of the study did
(:func:`check_candidate_files`).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from k4opt.frozen import freeze, thaw
from k4opt.jsonio import write_json
from k4opt.optimization._identity import (
    _contents,
    compared,
    program_differences,
    revision_warnings,
)
from k4opt.optimization.errors import OptimizationError

STUDY_NAME = "study.json"
STUDY_SCHEMA = "k4opt.study/1"
LOCK_NAME = ".lock"
JOURNAL_NAME = "study.journal"
INNER_NAME = "inner"
CONSTRAINT_VALUES = "margin / scale"  # how a limit's margin becomes the optimizer's value

_TRIAL = re.compile(r"trial-[0-9]{6,}")


def trial_directory(number: int) -> str:
    """The directory of trial ``number``, relative to the study: ``trial-000007``."""
    return f"trial-{number:06d}"


def inner_directory(directory: Path, number: int) -> Path:
    """The directory of the study of the level below that trial ``number`` of the study in
    ``directory`` stands for: beside the trial directories, each of them as any study keeps."""
    return directory / INNER_NAME / trial_directory(number)


def trial_directories(directory: Path) -> list[str]:
    """The trial directories in a study directory, by name."""
    return sorted(path.name for path in directory.iterdir() if _TRIAL.fullmatch(path.name))


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def session_seed(study_seed: int, trials: int) -> int:
    """The sampler's seed for a session that starts when the study has ``trials`` trials.

    Derived, so that a resumed session does not replay the proposals the first session started
    with; the same history and seed give the same proposals again.
    """
    digest = hashlib.sha256(f"{study_seed}:{trials}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# --- study.json ------------------------------------------------------------------------------


type _Kept = bool | Mapping[str, _Kept]

_SOURCE: _Kept = {
    "bytes": True,
    "sha256": True,
    "helpers": True,
    "executed": True,
    "revision": True,
}
PORTABLE: Mapping[str, _Kept] = {
    "schema": True,
    "id": True,
    "created": True,
    "identity": True,
    "problem": {"definition": True},
    "inputs": {"*": {"bytes": True, "sha256": True}},
    "implementations": {
        "*": {
            "version": True,
            "stage": True,
            "metrics": True,
            "artifacts": True,
            "roles": True,
            "source": _SOURCE,
        }
    },
    "geometry": {
        "compact": True,
        "fingerprint": True,
        "targets": True,
        "externals": {"*": {"sha256": True, "files": True, "unresolved": True}},
    },
    "environment": {"KEY4HEP_STACK": True},
    # by content: where a program and its interpreter are is a location, as any other
    "programs": {
        "*": {"bytes": True, "sha256": True, "interpreter": {"bytes": True, "sha256": True}}
    },
    "search": True,
    "k4opt": {"version": True, "sha256": True, "git": True},
    # what the study searches and fixes, and the level and replicate of its trials
    "scope": {"level": True, "searched": True, "fixed": True, "replicate": True, "parent": True},
    "optimizer": {
        "backend": True,
        "version": True,
        "study": True,
        "sampler": True,
        "objectives": True,
        "directions": True,
        "parameters": True,
        "constraints": True,
        "constraint_values": True,
    },
}
"""What of ``study.json`` leaves the study directory (:func:`portable`), key by key; ``*`` stands
for every key of a mapping keyed by name. Left out: every location (the problem file, the files
the problem names, the evaluators' files, the geometry's source root and the files its external
references resolve to, ``K4GEO``, the programs of the stack and their interpreters, which are
kept by size and SHA-256) and the optimizer's storage (a journal name, or a database URL
with its host and user), and any key not named here."""


def portable(document: Mapping[str, Any]) -> dict[str, Any]:
    """``study.json`` without its locations, as the JSON form of a study holds it
    (:data:`PORTABLE`): a fresh plain copy. Locations are no part of a study's identity, so its
    ``identity`` digest is that of the portable document too."""
    kept = _portable(document, PORTABLE, STUDY_NAME)
    assert isinstance(kept, dict)
    return kept


def _portable(value: object, kept: _Kept, where: str) -> object:
    if kept is True:
        return thaw(freeze(value, where, OptimizationError))
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(kept, Mapping):
        raise OptimizationError(f"{where} is not an object")
    if "*" in kept:
        return {key: _portable(item, kept["*"], f"{where}.{key}") for key, item in value.items()}
    return {
        key: _portable(value[key], rule, f"{where}.{key}")
        for key, rule in kept.items()
        if rule and key in value
    }


def write_study(directory: Path, document: Mapping[str, object]) -> None:
    try:
        write_json(directory / STUDY_NAME, document)
    except OSError as exc:
        raise OptimizationError(f"cannot write {directory / STUDY_NAME}: {exc}") from exc


def read_study(directory: Path) -> dict[str, Any]:
    """The ``study.json`` of ``directory``; raises :class:`OptimizationError` if there is none or
    it is not one this k4opt reads."""
    path = directory / STUDY_NAME
    if not path.is_file():
        raise OptimizationError(
            f"{directory} holds no {STUDY_NAME}, so there is no study to resume; start one "
            "without --resume"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OptimizationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != STUDY_SCHEMA:
        schema = document.get("schema") if isinstance(document, dict) else None
        raise OptimizationError(f"{path}: schema {schema!r} is not {STUDY_SCHEMA!r}")
    compared(document)  # every part a resume compares is there
    return document


# --- the record of a trial -------------------------------------------------------------------


def _get(data: object, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
    return data


def _recorded_input(summary: Mapping[str, Any], label: str) -> Mapping[str, Any] | None:
    """The size and SHA-256 a record holds for the file ``label`` names
    (``samples.<sample>.<stage>.file`` or ``.<option>``), ``None`` if its stage did not get as
    far as reading it."""
    parts = label.split(".", 3)
    if len(parts) != 4 or parts[0] != "samples":
        return None  # a supplied file: a study supplies none
    _, sample, stage, rest = parts
    if rest == "file":
        entry = _get(summary, "samples", sample, "stages", stage)
    else:
        entry = _get(summary, "samples", sample, "stages", stage, "files", rest)
    if not isinstance(entry, Mapping) or "sha256" not in entry:
        return None
    return entry


# what says which attribute of the geometry a change is of, and by which parameter
_RESOLVED = ("parameter", "target", "file", "locator", "attribute")


def check_record(
    study: Mapping[str, Any],
    summary: object,
    number: int,
    params: Mapping[str, Any],
    replicate: int = 0,
    level: int = 0,
    selected: Mapping[str, Any] | None = None,
) -> list[str]:
    """Refuse a record that is not the one trial ``number`` of ``study`` asked for with the
    complete candidate ``params``, as its ``replicate`` and judged by the criteria of its
    ``level`` (the study's scope), or was made with something the study does not compare equal;
    a trial that stands for a study of the level below names the trial ``selected`` of it. Return
    the warnings of :func:`revision_warnings`. Its assessment is checked apart, by
    :meth:`k4opt.record.Assessment.from_summary`."""
    where = f"trial {number}: {trial_directory(number)}/summary.json"
    if not isinstance(summary, Mapping):
        raise OptimizationError(f"{where} is not a run record")
    made_for: dict[str, Any] = {"study": study["id"], "trial": number}
    if selected is not None:
        made_for["selected"] = {"study": selected["study"], "trial": selected["trial"]}
    if summary.get("run") != made_for:
        raise OptimizationError(f"{where} was made for {summary.get('run')!r}, not this trial")
    candidate = summary.get("candidate")
    # as given, type and value: 1, 1.0 and True are different candidates
    given = list(candidate.items()) if isinstance(candidate, Mapping) else None
    if given is None or [(k, type(v), v) for k, v in given] != [
        (k, type(v), v) for k, v in params.items()
    ]:
        raise OptimizationError(f"{where} holds candidate {candidate!r}, but the trial is {params}")
    made = summary.get("replicate")
    if made != replicate or isinstance(made, bool):  # the events the trial asked for
        raise OptimizationError(
            f"{where} is replicate {made!r} of its candidate, evaluated on other events than the "
            f"trial asked for (replicate {replicate})"
        )
    judged = summary.get("level")
    if judged != level or isinstance(judged, bool):  # by the criteria the study tells
        raise OptimizationError(
            f"{where} was evaluated for level {judged!r} of a nested optimization, judged by its "
            f"criteria; the trial asked for level {level}"
        )
    if summary.get("represents_candidate") is not True or summary.get("supplied") != []:
        raise OptimizationError(f"{where} was not made from its candidate alone (a supplied stage)")
    implementations = summary.get("implementations")
    if not isinstance(implementations, Mapping) or list(implementations) != list(
        study["implementations"]
    ):
        raise OptimizationError(f"{where} does not hold the study's evaluators")
    for name, old in study["implementations"].items():
        new = implementations[name]
        if _get(new, "source", "sha256") != old["source"].get("sha256") or _get(
            new, "source", "helpers"
        ) != old["source"].get("helpers"):
            raise OptimizationError(f"{where}: evaluator {name!r} ran other code than the study's")
        if _get(new, "metrics") != old.get("metrics"):
            raise OptimizationError(
                f"{where}: evaluator {name!r} declared other metrics or units than in the study"
            )
        if _get(new, "artifacts") != old.get("artifacts"):
            raise OptimizationError(
                f"{where}: evaluator {name!r} declared other artifacts than in the study"
            )
        if _get(new, "roles") != old.get("roles"):
            raise OptimizationError(
                f"{where}: evaluator {name!r} declared other roles than in the study"
            )
    observed = _get(summary, "environment", "observed")
    if _get(observed, "KEY4HEP_STACK") != study["environment"]["KEY4HEP_STACK"]:
        raise OptimizationError(f"{where} ran in another KEY4HEP_STACK than the study")
    if _get(observed, "k4opt_source", "sha256") != study["k4opt"]["sha256"]:
        raise OptimizationError(f"{where} was made by other k4opt code than the study")
    ran = program_differences(study, {"programs": _get(observed, "programs")})
    if ran:
        raise OptimizationError(f"{where} ran other programs than the study: {'; '.join(ran)}")
    # a geometry that was not made (it ran out of time, its worker died) changed and read nothing
    if _get(summary, "geometry", "status") == "done":
        changes = _get(summary, "geometry", "changes")
        resolved = [
            {key: change.get(key) for key in (*_RESOLVED, "old_value")}
            if isinstance(change, Mapping)
            else None
            for change in (changes if isinstance(changes, list) else [None])
        ]
        targets = [
            {**{key: target[key] for key in _RESOLVED}, "old_value": target["value"]}
            for target in study["geometry"]["targets"]
        ]
        if resolved != targets:
            raise OptimizationError(
                f"{where} changed other attributes of the geometry than the study's parameters set"
            )
        externals = _get(summary, "geometry", "externals")
        if externals is None or _contents(externals) != _contents(study["geometry"]["externals"]):
            raise OptimizationError(
                f"{where} read other files through the geometry's external references than the "
                "study"
            )
    for label, entry in study["inputs"].items():
        recorded = _recorded_input(summary, label)
        if recorded is not None and (recorded.get("bytes"), recorded.get("sha256")) != (
            entry.get("bytes"),
            entry.get("sha256"),
        ):
            raise OptimizationError(f"{where} read another {label} than the study")
    return revision_warnings(study["implementations"], implementations)


def check_candidate_files(
    read: dict[str, tuple[int, object]], summary: object, number: int
) -> None:
    """Refuse the record of trial ``number`` if an external reference that only candidates name
    (``geometry.candidate_externals``: a ``ref`` parameter choosing ``${CAD}/other.stl``) reads
    other files than for an earlier trial of the study; then add what it read to ``read``, by
    reference: the first trial that read it and the files' content."""
    where = f"trial {number}: {trial_directory(number)}/summary.json"
    if _get(summary, "geometry", "status") != "done":  # not made: its references read nothing
        return
    contents = _contents(_get(summary, "geometry", "candidate_externals"))
    if not isinstance(contents, dict):
        raise OptimizationError(f"{where} does not say what its candidate's own references read")
    for ref, content in contents.items():
        if ref in read and read[ref][1] != content:
            raise OptimizationError(
                f"{where}: {ref} reads other files than for trial {read[ref][0]}, so the two "
                "trials did not evaluate one geometry"
            )
    for ref, content in contents.items():
        read.setdefault(ref, (number, content))
