"""Internal: the ``evaluators`` of a problem document, parsed: each binding of an implementation,
its settings and collections, the samples it measures and the roles of a combined evaluator."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from k4opt.frozen import Json
from k4opt.problem._fields import (
    _choice,
    _fail,
    _mapping,
    _seconds,
    _sequence,
    _show,
    _text,
)
from k4opt.problem.models import (
    EvaluatorSpec,
    Reference,
    freeze,
)
from k4opt.problem.plan import RESERVED_NAMES
from k4opt.problem.stages import STAGE_TYPES

_EVALUATOR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _name(value: object, path: str) -> str:
    """An evaluator's ``name`` or ``as``: letters, digits, ``.``, ``_`` and ``-``."""
    name = _text(value, path)
    if not _EVALUATOR_NAME.fullmatch(name):
        raise _fail(
            path,
            f"{_show(name)} must start with a letter or digit and hold only letters, digits, '.', "
            "'_' and '-'",
        )
    return name


def _evaluators(raw: object, samples: list[str]) -> tuple[EvaluatorSpec, ...]:
    """The ``evaluators``, each measuring the ``samples`` it lists (or the problem's only sample)
    if it reads a stage, the candidate otherwise.

    A binding may be named by several entries if each binds the same implementation, version and
    stage on other samples (with other settings, say); otherwise it names one entry."""
    items = _sequence(raw, "evaluators")
    if not items:
        raise _fail("evaluators", "must list at least one evaluator")
    evaluators: list[EvaluatorSpec] = []
    bound: dict[str, list[int]] = {}  # the entries of each binding, by binding case folded
    for index, item in enumerate(items):
        path = f"evaluators[{index}]"
        m = _mapping(
            item,
            path,
            (
                "name",
                "as",
                "version",
                "stage",
                "samples",
                "collections",
                "settings",
                "roles",
                "required",
                "timeout",
            ),
            ("name", "version"),
        )
        if "roles" in m:
            others = [key for key in ("stage", "samples", "collections") if key in m]
            if others:
                raise _fail(
                    path,
                    f"a combined evaluator (roles:) reads no stage of its own, so it takes no "
                    f"{', '.join(others)}; its roles name what it reads",
                )
        name = _name(m["name"], f"{path}.name")
        # the binding names the evaluator's task, its directory in a run and its metrics
        where = f"{path}.as" if "as" in m else f"{path}.name"
        binding = _name(m["as"], where) if "as" in m else name
        # records name the work of a candidate, its geometry, stages and evaluators, by name alone
        if binding in RESERVED_NAMES:
            reserved = ", ".join(RESERVED_NAMES)
            raise _fail(where, f"{binding!r} names a step k4opt runs itself; reserved: {reserved}")
        settings: Mapping[str, Json] = {}
        if "settings" in m:
            if not isinstance(m["settings"], Mapping):
                raise _fail(
                    f"{path}.settings", f"must be a mapping, got {type(m['settings']).__name__}"
                )
            frozen = freeze(m["settings"], f"{path}.settings")
            assert isinstance(frozen, Mapping)
            settings = frozen
        stage = _choice(m["stage"], f"{path}.stage", STAGE_TYPES) if "stage" in m else None
        collections: dict[str, str] = {}
        if "collections" in m:
            where, raw = f"{path}.collections", m["collections"]
            roles = _mapping(raw, where, tuple(raw) if isinstance(raw, Mapping) else ())
            collections = {role: _text(name, f"{where}.{role}") for role, name in roles.items()}
        version = _text(m["version"], f"{path}.version")
        measured = _measured(m, path, stage, samples)
        roles = _roles(m["roles"], f"{path}.roles") if "roles" in m else {}
        required = _required(m["required"], f"{path}.required") if "required" in m else False
        timeout = _seconds(m["timeout"], f"{path}.timeout") if "timeout" in m else None
        entry = EvaluatorSpec(
            name, version, settings, stage, collections, binding, measured, roles, required, timeout
        )
        folded = binding.casefold()  # run directories may be on a file system that folds case
        for earlier in bound.get(folded, []):
            _check_repeated(entry, evaluators[earlier], f"evaluators[{earlier}]", path)
        bound.setdefault(folded, []).append(index)
        evaluators.append(entry)
    for index, entry in enumerate(evaluators):  # once all are known: a role may read a later one
        _check_roles(entry, f"evaluators[{index}].roles", evaluators, samples)
    return tuple(evaluators)


def _required(value: object, path: str) -> bool:
    """``required: true``, the one value it takes: an evaluator is required already when an
    objective or a constraint, or a required evaluator, reads it, and a diagnostic otherwise."""
    if value is not True:
        raise _fail(
            path,
            f"must be true, got {_show(value)}; an evaluator that no objective, constraint or "
            "required evaluator reads is a diagnostic without it",
        )
    return True


_REFERENCE_KEYS = ("sample", "samples", "evaluator", "stage")


def _roles(raw: object, path: str) -> dict[str, tuple[Reference, ...]]:
    """The items each role of a combined evaluator reads, by role: each role is given one
    reference, ``{sample, evaluator}`` (an evaluation) or ``{sample, stage}`` (a stage's file),
    with ``samples:`` for one such item per sample, or a list of references."""
    m = _mapping(raw, path, tuple(raw) if isinstance(raw, Mapping) else ())
    if not m:
        raise _fail(
            path, "must name at least one role; an evaluator without roles measures the candidate"
        )
    roles: dict[str, tuple[Reference, ...]] = {}
    for role, value in m.items():
        where = f"{path}.{role}"
        _text(role, path)
        given = value if isinstance(value, (list, tuple)) else [value]
        if not given:
            raise _fail(where, "must reference at least one evaluation or stage")
        items: list[Reference] = []
        for index, reference in enumerate(given):
            at = f"{where}[{index}]" if isinstance(value, (list, tuple)) else where
            for item in _references(reference, at):
                if item in items:
                    raise _fail(
                        at,
                        f"{item.label} is given twice in this role, so the evaluator would read "
                        "it twice; give it once",
                    )
                items.append(item)
        roles[role] = tuple(items)
    return roles


def _references(raw: object, path: str) -> list[Reference]:
    """The items one reference of a role stands for: one per sample it names."""
    m = _mapping(raw, path, _REFERENCE_KEYS)
    if ("sample" in m) == ("samples" in m):
        raise _fail(path, "needs exactly one of 'sample' and 'samples'")
    if ("evaluator" in m) == ("stage" in m):
        raise _fail(
            path, "needs exactly one of 'evaluator' (its evaluation) and 'stage' (its file)"
        )
    if "sample" in m:
        samples = [_text(m["sample"], f"{path}.sample")]
    else:
        listed = _sequence(m["samples"], f"{path}.samples")
        if not listed:
            raise _fail(f"{path}.samples", "must list at least one sample")
        samples = [_text(s, f"{path}.samples[{i}]") for i, s in enumerate(listed)]
    if "evaluator" in m:
        evaluator = _text(m["evaluator"], f"{path}.evaluator")
        return [Reference(sample, evaluator=evaluator) for sample in samples]
    stage = _choice(m["stage"], f"{path}.stage", STAGE_TYPES)
    return [Reference(sample, stage=stage) for sample in samples]


def _check_roles(
    entry: EvaluatorSpec, path: str, evaluators: list[EvaluatorSpec], samples: list[str]
) -> None:
    """Each item a role of ``entry`` reads is of a sample of the problem, and an evaluation of it
    by a binding that measures it; the stage of a stage item is checked with the samples
    (:func:`_check_pipeline`)."""
    measures: dict[str, list[str]] = {}
    for other in evaluators:
        measures.setdefault(other.binding, []).extend(other.samples)
    for role, items in entry.roles.items():
        for item in items:
            where = f"{path}.{role}"
            if item.sample not in samples:
                known = f"its samples are {', '.join(samples)}" if samples else "it has none"
                raise _fail(where, f"the problem has no sample {item.sample!r}; {known}")
            if item.evaluator is None:
                continue
            if item.evaluator not in measures:
                raise _fail(
                    where,
                    f"the problem has no evaluator {item.evaluator!r}; its evaluators are "
                    f"{', '.join(measures)}",
                )
            measured = measures[item.evaluator]
            if not measured:
                raise _fail(
                    where,
                    f"{item.evaluator!r} measures the candidate, not a sample; chained combined "
                    "evaluators are not supported yet",
                )
            if item.sample not in measured:
                raise _fail(
                    where,
                    f"evaluator {item.evaluator!r} does not measure {item.sample!r}; it measures "
                    f"{', '.join(measured)}",
                )


def _measured(
    m: Mapping[str, Any], path: str, stage: str | None, samples: list[str]
) -> tuple[str, ...]:
    """The samples an evaluator entry measures: those it lists, the problem's only sample if it
    lists none, and none if it reads no stage (it measures the candidate)."""
    if "samples" in m:
        where = f"{path}.samples"
        if stage is None:
            raise _fail(
                where,
                "an evaluator measures samples at the stage it reads; add that stage, or remove "
                "samples to measure the candidate",
            )
        items = _sequence(m["samples"], where)
        if not items:
            raise _fail(where, "must list at least one sample")
        for index, item in enumerate(items):
            at = f"{where}[{index}]"
            sample = _text(item, at)
            if sample not in samples:
                known = f"its samples are {', '.join(samples)}" if samples else "it has none"
                raise _fail(at, f"the problem has no sample {sample!r}; {known}")
            if sample in items[:index]:
                raise _fail(at, f"{sample!r} is listed twice")
        return tuple(items)
    if stage is None:
        return ()
    if len(samples) == 1:
        return (samples[0],)
    if not samples:
        raise _fail(path, f"it reads the {stage} stage of a sample, but the problem has no samples")
    raise _fail(
        path,
        f"the problem has the samples {', '.join(samples)}; add samples: with those this "
        "evaluator measures",
    )


def _check_repeated(entry: EvaluatorSpec, other: EvaluatorSpec, at: str, path: str) -> None:
    """``entry`` may share its binding name with the earlier entry ``other`` (at ``at``) only to
    bind the same implementation, version and stage on other samples."""
    binding = entry.binding
    if other.binding != binding:
        why = (
            f"{binding!r} differs only in case from {other.binding!r} of {at}, which run "
            "directories may not tell apart"
        )
    elif entry.stage is None or other.stage is None:
        why = f"{binding!r} is already bound by {at}"
    elif shared := [sample for sample in entry.samples if sample in other.samples]:
        why = f"{binding!r} is already bound to {', '.join(shared)} by {at}"
    elif (entry.name, entry.version, entry.stage) != (other.name, other.version, other.stage):
        why = (
            f"{binding!r} binds {other.name} version {other.version!r} at the {other.stage} stage "
            f"in {at}, and one binding name binds one implementation, version and stage"
        )
    else:
        return
    raise _fail(path, f"{why}; name this one with as:")
