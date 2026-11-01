"""Internal: the optimizer ``study.json`` records, and the Optuna study that holds its trials.

:func:`new_document` writes down how a new study is optimized, :func:`check_resume` refuses to
resume one with anything else, :func:`open_optuna` makes or loads its Optuna study, and
:func:`check_optuna_study` refuses an Optuna study that is not the one ``study.json`` describes.
"""

from __future__ import annotations

import json
import secrets
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import BaseDistribution, distribution_to_json
from optuna.storages import BaseStorage

from k4opt.optimization._optuna.storage import storage_record
from k4opt.optimization._study import (
    CONSTRAINT_VALUES,
    STUDY_SCHEMA,
    differences,
    digest,
    now,
    revision_warnings,
    trial_directories,
)
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import MetricRef, ProblemSpec

STARTUP_TRIALS = 10  # random proposals until this many trials are COMPLETE
CONSTANT_LIAR = True  # trials still running count as bad ones, so trials at once spread out


def new_document(
    spec: ProblemSpec,
    identity: Mapping[str, Any],
    space: Mapping[str, BaseDistribution],
    names: Sequence[str],
    seed: int | None,
    storage: str | None,
) -> dict[str, Any]:
    study_id = str(uuid.uuid4())
    document: dict[str, Any] = {"schema": STUDY_SCHEMA, "id": study_id, "created": now()}
    document |= dict(identity)
    document["identity"] = digest(identity)
    seed = secrets.randbelow(2**32) if seed is None else seed
    name = f"{spec.name}-{study_id[:8]}"
    document["optimizer"] = _optimizer(spec, space, names, name=name, seed=seed, storage=storage)
    return document


def _optimizer(
    spec: ProblemSpec,
    space: Mapping[str, BaseDistribution],
    names: Sequence[str],
    *,
    name: str,
    seed: int,
    storage: str | None,
) -> dict[str, Any]:
    """How the study is optimized, as ``study.json`` records it under ``optimizer``."""
    return {
        "backend": "optuna",
        "version": optuna.__version__,
        "study": name,
        "storage": storage_record(storage),
        "sampler": {
            "name": "TPESampler",
            "seed": seed,
            "n_startup_trials": STARTUP_TRIALS,
            "constant_liar": CONSTANT_LIAR,
            "seeding": "per session, from the seed and the number of trials",
        },
        "directions": [o.direction for o in spec.objectives],
        "parameters": {
            key: json.loads(distribution_to_json(distribution))
            for key, distribution in space.items()
        },
        "constraints": list(names),
        "constraint_values": CONSTRAINT_VALUES,
    }


def check_resume(
    document: Mapping[str, Any],
    identity: Mapping[str, Any],
    spec: ProblemSpec,
    space: Mapping[str, BaseDistribution],
    names: Sequence[str],
    storage: str | None,
    warn: Callable[[str], None],
) -> None:
    """Refuse to resume a study with anything it compares otherwise than it was made with, and
    one whose recorded optimizer is not the one this k4opt would run it with."""
    optimizer = document["optimizer"]
    found = differences(document, identity)
    seed = (optimizer.get("sampler") or {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        found.append(f"its sampler seed {seed!r} is no seed in [0, 2**32)")
    else:
        expected = _optimizer(
            spec, space, names, name=str(optimizer.get("study")), seed=seed, storage=storage
        )
        found += [
            f"its recorded optimizer {key} is {optimizer.get(key)!r}, not {value!r}"
            for key, value in expected.items()
            if key not in ("version", "storage") and optimizer.get(key) != value
        ]
    if optimizer.get("version") != optuna.__version__:
        found.append(
            f"the study was made with Optuna {optimizer.get('version')}, this is Optuna "
            f"{optuna.__version__}: install optuna=={optimizer.get('version')} to resume it"
        )
    if optimizer.get("storage") != storage_record(storage):
        found.append(
            f"the study's storage is {optimizer.get('storage')}, not {storage_record(storage)}; "
            "give the same --storage as when it was made"
        )
    if digest(document) != document.get("identity"):
        found.append("its study.json does not hold together (its identity digest differs)")
    if found:
        raise OptimizationError(
            f"cannot resume {document['optimizer']['study']}: {'; '.join(found)}. Start a new "
            "study for this"
        )
    for warning in revision_warnings(document["implementations"], identity["implementations"]):
        warn(warning)


def open_optuna(
    directory: Path, document: Mapping[str, Any], storage: BaseStorage, resume: bool
) -> optuna.Study:
    """The Optuna study of ``document``: made for a new study; loaded for a resumed one, and its
    setup finished if a killed process cut it short before the first trial."""
    optimizer = document["optimizer"]
    name = optimizer["study"]
    attributes = study_attributes(document)
    if resume:
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            if trial_directories(directory):
                raise OptimizationError(
                    f"the optimizer's storage holds no study {name!r}, though {directory} has "
                    "trials"
                ) from None
        else:
            if not study.trials:  # made, but perhaps not all its attributes written
                for key, value in attributes.items():
                    if key not in study.user_attrs:
                        study.set_user_attr(key, value)
            return study  # whatever it holds is checked by check_optuna_study
    study = optuna.create_study(
        study_name=name, storage=storage, directions=list(optimizer["directions"])
    )
    for key, value in attributes.items():
        study.set_user_attr(key, value)
    return study


def study_attributes(document: Mapping[str, Any]) -> dict[str, Any]:
    """The ``k4opt.*`` attributes of the Optuna study ``document`` describes."""
    optimizer = document["optimizer"]
    objectives = document["problem"]["definition"]["objectives"]
    return {
        "k4opt.study": document["id"],
        "k4opt.identity": document["identity"],
        "k4opt.parameters": list(optimizer["parameters"]),
        "k4opt.objectives": [
            MetricRef(o["evaluator"], o["stage"], o["metric"], o.get("sample")).label
            for o in objectives
        ],
        "k4opt.constraints": list(optimizer["constraints"]),
    }


def check_optuna_study(study: optuna.Study, document: Mapping[str, Any]) -> None:
    """The Optuna study is the one ``study.json`` describes."""
    optimizer = document["optimizer"]
    directions = [direction.name.lower() for direction in study.directions]
    expected = {
        "directions": optimizer["directions"],
        "k4opt.study": document["id"],
        "k4opt.identity": document["identity"],
        "k4opt.constraints": optimizer["constraints"],
    }
    found = {key: study.user_attrs.get(key) for key in expected} | {"directions": directions}
    differ = [key for key in expected if found[key] != expected[key]]
    if differ:
        raise OptimizationError(
            f"the optimizer's study {study.study_name!r} is not the one {document['id']} "
            f"describes: its {', '.join(differ)} differ"
        )
