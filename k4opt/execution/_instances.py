"""Internal: the instances of an evaluator implementation, each created anew for one evaluation.

The session binds an implementation to the problem through a probe instance, whose declaration
(:class:`_Probe`) is what every later instance must declare too: whatever process creates
an instance for an evaluation (the coordinator, or a worker that loaded the implementation again)
creates it from the :class:`_Bound` entry and checks it against the probe (:func:`_instance`).
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from k4opt.evaluation import (
    Declaration,
    EvaluationContractError,
    EvaluatorProtocol,
    declared_artifacts,
    declared_collections,
)
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


@dataclass(frozen=True, slots=True)
class _Probe:
    """What every instance of an implementation must say of itself as its probe did: the session
    records the probe's class as the code that runs, and its ``declaration`` as what the
    implementation declares (the stage and metrics decided what each evaluator reads and
    measures, the artifacts and roles what its results are checked against and what it reads);
    the settings and collections (their defaults included) make the request it gets, which must
    not change from one candidate to the next, and ``artifacts`` say what each file holds. Only
    the declaration is plain data a record keeps: the rest is this process's alone."""

    cls: type
    declaration: Declaration
    settings: Mapping[str, object]
    collections: Mapping[str, object]
    artifacts: Mapping[str, str] | None


def _probe(evaluator: EvaluatorProtocol) -> _Probe:
    return _Probe(
        type(evaluator),
        Declaration.of(evaluator),
        # a copy all through: an instance that changes a nested default changes its class's, and
        # the next instance would declare the changed one unnoticed
        deepcopy(dict(getattr(evaluator, "settings", None) or {})),
        declared_collections(evaluator),
        declared_artifacts(evaluator),
    )


def _identity(probe: _Probe) -> dict[str, object]:
    """What an instance says of itself, by what an error calls it, in the order it says it."""
    declaration = probe.declaration
    return {
        "class": probe.cls,
        "name": declaration.name,
        "version": declaration.version,
        "stage": declaration.stage,
        "settings": probe.settings,
        "collections": probe.collections,
        "metrics": declaration.metrics,
        "artifacts": probe.artifacts,
        "roles": declaration.roles,
    }


@dataclass(frozen=True, slots=True)
class _Bound:
    """A binding of the problem: its entry, the implementation given for it and what the probe
    instance that bound them declared."""

    entry: EvaluatorSpec
    implementation: type[EvaluatorProtocol]
    probe: _Probe


def _instance(bound: _Bound) -> EvaluatorProtocol:
    """A new instance of a bound implementation; :class:`EvaluationContractError` unless it
    declares what the probe declared."""
    evaluator = _create(bound.implementation)
    new, old = _identity(_probe(evaluator)), _identity(bound.probe)
    changed = [
        f"{label} {new[label]!r}, not {old[label]!r}" for label in old if new[label] != old[label]
    ]
    if changed:
        raise EvaluationContractError(
            f"evaluator {bound.entry.binding!r}: a new instance of "
            f"{bound.implementation.__qualname__} has {'; '.join(changed)}; every instance must "
            "declare what the first did"
        )
    return evaluator
