"""Internal: the instances of an evaluator implementation, each created anew for one evaluation.

The session binds an implementation to the problem through a probe instance, whose declaration
(:class:`_Declaration`) is what every later instance must declare too: whatever process creates
an instance for an evaluation (the coordinator, or a worker that loaded the implementation again)
creates it from the :class:`_Bound` entry and checks it against the probe (:func:`_instance`).
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple

from k4opt.evaluation import (
    EvaluationContractError,
    EvaluatorProtocol,
    Role,
    declared_artifacts,
    declared_collections,
    declared_roles,
    metric_units,
)
from k4opt.problem._work import MetricKey
from k4opt.problem.models import EvaluatorSpec


def _stage(evaluator: EvaluatorProtocol) -> str | None:
    """The pipeline stage the evaluator reads, ``None`` if it reads none."""
    stage: str | None = getattr(evaluator, "stage", None)
    return stage


def _create(implementation: type[EvaluatorProtocol]) -> EvaluatorProtocol:
    """A new instance of the class ``implementation``, which must have ``name``, ``version`` and
    ``evaluate``; :class:`EvaluationContractError` if it is no class, cannot be created or lacks
    one."""
    if not isinstance(implementation, type):
        # a function that builds an instance could configure it in ways the class's source,
        # which the records identify as the code that ran, does not say
        raise EvaluationContractError(
            f"got a {type(implementation).__name__} object, not an evaluator class: give the "
            "class, whose source is recorded and of which k4opt creates an instance per candidate"
        )
    label = implementation.__qualname__
    try:
        evaluator = implementation()
    except Exception as exc:
        raise EvaluationContractError(
            f"creating {label}() failed: {type(exc).__name__}: {exc}"
        ) from exc
    missing = [
        member for member in ("name", "version", "evaluate") if not hasattr(evaluator, member)
    ]
    if missing or not callable(evaluator.evaluate):
        raise EvaluationContractError(
            f"{label} is not an evaluator: needs name, version and an evaluate() method"
            + (f" (missing {', '.join(missing)})" if missing else "")
        )
    return evaluator


class _Declaration(NamedTuple):
    """What every instance of an implementation must say of itself as its probe did: the session
    records the probe's class as the code that runs, the stage and metrics decided what each
    evaluator reads and measures, the settings and collections (their defaults included)
    make the request it gets, which must not change from one candidate to the next, the artifacts
    are what its results are checked against, and the roles what it reads."""

    cls: type
    name: str
    version: str
    stage: str | None
    settings: Mapping[str, object]
    collections: Mapping[str, object]
    metrics: Mapping[MetricKey, str] | None
    artifacts: Mapping[str, str] | None
    roles: Mapping[str, Role] | None


def _declaration(evaluator: EvaluatorProtocol) -> _Declaration:
    return _Declaration(
        type(evaluator),
        evaluator.name,
        evaluator.version,
        _stage(evaluator),
        # a copy all through: an instance that changes a nested default changes its class's, and
        # the next instance would declare the changed one unnoticed
        deepcopy(dict(getattr(evaluator, "settings", None) or {})),
        declared_collections(evaluator),
        metric_units(evaluator),
        declared_artifacts(evaluator),
        declared_roles(evaluator),
    )


@dataclass(frozen=True, slots=True)
class _Bound:
    """A binding of the problem: its entry, the implementation given for it and what the probe
    instance that bound them declared."""

    entry: EvaluatorSpec
    implementation: type[EvaluatorProtocol]
    declaration: _Declaration


def _instance(bound: _Bound) -> EvaluatorProtocol:
    """A new instance of a bound implementation; :class:`EvaluationContractError` unless it
    declares what the probe declared."""
    evaluator = _create(bound.implementation)
    labels = (
        "class",
        "name",
        "version",
        "stage",
        "settings",
        "collections",
        "metrics",
        "artifacts",
        "roles",
    )
    changed = [
        f"{label} {new!r}, not {old!r}"
        for label, new, old in zip(labels, _declaration(evaluator), bound.declaration, strict=True)
        if new != old
    ]
    if changed:
        raise EvaluationContractError(
            f"evaluator {bound.entry.binding!r}: a new instance of "
            f"{bound.implementation.__qualname__} has {'; '.join(changed)}; every instance must "
            "declare what the first did"
        )
    return evaluator
