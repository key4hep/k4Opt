"""Internal: an evaluator implementation bound to a problem, and the instance of an evaluation.

The session binds an implementation by what its class declares (:class:`_Declared`), read without
creating it or running its code (:func:`~k4opt.evaluation.check_class`). Every request is made from
that, in the session's own process. An instance is created only where an evaluation runs (the
session's process, or a worker that loaded the implementation again), once per evaluation, right
before its ``evaluate()`` (:func:`_instance`), and must declare what the class declared when the
session bound it.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from k4opt.evaluation import (
    CandidateRejected,
    Declaration,
    EvaluationContractError,
    EvaluatorProtocol,
    declared_artifacts,
    declared_collections,
)
from k4opt.evaluation.declaration import DECLARED
from k4opt.problem.models import EvaluatorSpec


def _stage(evaluator: object) -> str | None:
    """The pipeline stage an evaluator class reads, ``None`` if it reads none."""
    stage: str | None = getattr(evaluator, "stage", None)
    return stage


@dataclass(frozen=True, slots=True)
class _Declared:
    """What an implementation's class declares: the session records the class as the code that
    runs, and its ``declaration`` as what the implementation declares (the stage and metrics
    decide what each evaluator reads and measures, the artifacts and roles what its results are
    checked against and what it reads); the settings and collections (their defaults included)
    make the request it gets, which must not change from one candidate to the next, and
    ``artifacts`` say what each file holds. Only the declaration is plain data a record keeps: the
    rest is this process's alone."""

    cls: type[EvaluatorProtocol]
    declaration: Declaration
    settings: Mapping[str, object]
    collections: Mapping[str, object]
    artifacts: Mapping[str, str] | None


def _declared(cls: object) -> _Declared:
    """What the class ``cls`` declares, read without creating it or running its code;
    :class:`EvaluationContractError` for a class that is no evaluator."""
    declaration = Declaration.of(cls)  # checks the class first
    assert isinstance(cls, type)
    return _Declared(
        cls,
        declaration,
        # a copy all through: an instance that changes a nested default changes its class's, and
        # the next request would hold the changed one unnoticed
        deepcopy(dict(getattr(cls, "settings", None) or {})),
        declared_collections(cls),
        declared_artifacts(cls),
    )


def _identity(declared: _Declared) -> dict[str, object]:
    """What a class declares, by what an error calls it, in the order it says it."""
    declaration = declared.declaration
    return {
        "class": declared.cls,
        "name": declaration.name,
        "version": declaration.version,
        "stage": declaration.stage,
        "settings": declared.settings,
        "collections": declared.collections,
        "metrics": declaration.metrics,
        "artifacts": declared.artifacts,
        "roles": declaration.roles,
    }


@dataclass(frozen=True, slots=True)
class _Bound:
    """A binding of the problem: its entry and what the class given for it declared when the
    session bound them."""

    entry: EvaluatorSpec
    declared: _Declared

    @property
    def implementation(self) -> type[EvaluatorProtocol]:
        return self.declared.cls


def _instance(bound: _Bound) -> EvaluatorProtocol:
    """The one new instance of an evaluation; :class:`EvaluationContractError` if it cannot be
    created, sets a declaration of its own, or its class declares otherwise than when the
    session bound it (its ``__init__`` changed a default, say)."""
    cls = bound.implementation
    label = cls.__qualname__
    try:
        evaluator = cls()
    except CandidateRejected as exc:
        raise EvaluationContractError(
            f"evaluator {bound.entry.binding!r}: creating {label}() rejected the candidate "
            f"({exc}); a candidate is rejected in check() alone, before anything of it runs"
        ) from exc
    except Exception as exc:
        raise EvaluationContractError(
            f"evaluator {bound.entry.binding!r}: creating {label}() failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    own = [name for name in DECLARED if name in getattr(evaluator, "__dict__", {})]
    if own:
        raise EvaluationContractError(
            f"evaluator {bound.entry.binding!r}: an instance of {label} sets {', '.join(own)} "
            "itself; declare them on the class, where k4opt reads them before any instance exists"
        )
    _check_unchanged(bound)
    return evaluator


def _check_unchanged(bound: _Bound) -> None:
    """Raise :class:`EvaluationContractError` if the class of ``bound`` declares otherwise than
    when the session bound it: an instance that changed a nested default changed its class's, and
    later requests would be made from what it changed."""
    cls = bound.implementation
    new, old = _identity(_declared(cls)), _identity(bound.declared)
    changed = [f"{key} {new[key]!r}, not {old[key]!r}" for key in old if new[key] != old[key]]
    if changed:
        raise EvaluationContractError(
            f"evaluator {bound.entry.binding!r}: {cls.__qualname__} now declares "
            f"{'; '.join(changed)}; what a class declares must not change once the session bound it"
        )
