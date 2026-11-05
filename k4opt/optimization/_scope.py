"""Internal: what one study searches and how its trials are judged, its *scope*; nothing of Optuna.

A study searches some of the problem's parameters (``searched``) and gives others one value for
every trial (``fixed``): it searches all of them in an ordinary study, those of its level of a
nested optimization otherwise, less any the caller fixed (``k4opt optimize --fix``); a study of a
level below the top is started for a candidate of the levels above, whose values it fixes. Its
trials are judged by the criteria of its ``level`` (:meth:`~k4opt.problem.ProblemSpec.criteria`)
and evaluated as its ``replicate`` (:func:`~k4opt.problem.candidate.configure`), and a study
started for a trial of another names it (``parent``). ``study.json`` records the scope under
``scope``; the optimizer knows only the searched parameters, while every record holds the
complete candidate (:meth:`StudyScope.candidate`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from k4opt.evaluation import Scalar
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import CandidateError, Criteria, ProblemSpec
from k4opt.problem.candidate import validate_value


@dataclass(frozen=True, slots=True)
class StudyScope:
    """One study of ``spec``: at ``level`` (0 for an ordinary study), its parameters but those
    ``fixed`` searched, started for the trial ``parent`` (``{"study": id, "trial": n}``) of a
    study of the level above, if any. Made by :meth:`of`, which checks it."""

    spec: ProblemSpec
    level: int = 0
    fixed: Mapping[str, Scalar] = field(default_factory=lambda: FrozenMap({}))
    parent: Mapping[str, Json] | None = None

    @classmethod
    def of(
        cls,
        spec: ProblemSpec,
        fixed: Mapping[str, Any] | None = None,
        *,
        level: int = 0,
        parent: Mapping[str, Any] | None = None,
    ) -> StudyScope:
        """The scope of a study of ``spec`` at ``level`` that gives the parameters ``fixed`` their
        value: every parameter of the levels above (the candidate of the level above it was
        started for: inherited, so required), and any of its own level it is not to search
        (``k4opt optimize --fix``). Raises :class:`OptimizationError` for a level the problem
        lacks, a value of the levels above missing, a parameter of a level below or none of the
        problem's, a value outside its domain, and when nothing is left to search."""
        try:
            spec.check_level(level)
        except ValueError as exc:
            raise OptimizationError(str(exc)) from None
        given = dict(fixed or {})
        own = _own(spec, level)
        above = [name for each in spec.levels[:level] for name in each.parameters]
        missing = [name for name in above if name not in given]
        if missing:
            raise OptimizationError(
                f"fixed: a study of level {level} is started for a candidate of the levels "
                f"above, so it needs the value of {', '.join(missing)}"
            )
        below = [name for each in spec.levels[level + 1 :] for name in each.parameters]
        for name in given:
            if name in below:
                raise OptimizationError(
                    f"fixed: {name} is searched by a level below this one (by the study of each "
                    "of its candidates), so this study does not give it a value"
                )
            if name not in own and name not in above:
                raise OptimizationError(
                    f"fixed: {name} is no parameter of the problem; this study searches "
                    f"{', '.join(own)}"
                )
        by_name = {parameter.name: parameter for parameter in spec.parameters}
        for name, value in given.items():
            try:
                validate_value(by_name[name], value)
            except CandidateError as exc:
                raise OptimizationError(f"fixed: {exc}") from None
        if all(name in given for name in own):
            raise OptimizationError(
                "fixed: every parameter the study searches is fixed, so there is nothing to "
                "search; evaluate the candidate with k4opt evaluate"
            )
        ordered = {p.name: given[p.name] for p in spec.parameters if p.name in given}
        plain = freeze(ordered, "fixed", OptimizationError)
        parent_json = None if parent is None else freeze(parent, "parent", OptimizationError)
        assert isinstance(plain, Mapping)
        assert parent_json is None or isinstance(parent_json, Mapping)
        return cls(spec, level, plain, parent_json)  # type: ignore[arg-type]  # scalars, checked

    @property
    def searched(self) -> tuple[str, ...]:
        """The parameters the optimizer proposes, in problem order."""
        return tuple(name for name in _own(self.spec, self.level) if name not in self.fixed)

    @property
    def criteria(self) -> Criteria:
        """What the study's trials are judged by and the optimizer is told."""
        return self.spec.criteria(self.level)

    @property
    def replicate(self) -> int:
        """The replicate of its trials' records: 0 for a study whose trials are evaluated as
        they are, the levels below for one whose trials stand for a study of the next level."""
        return 0 if self.level == self.spec.depth else self.spec.depth - self.level

    @property
    def directions(self) -> list[str]:
        """The direction of each objective the optimizer is told, in order."""
        return [objective.direction for objective in self.criteria.objectives]

    @property
    def limit_names(self) -> list[str]:
        """The name of each limit, as the optimizer knows its constraint value."""
        return limit_names(self.criteria)

    def candidate(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The candidate of a trial whose proposed values are ``params``: with the values the
        study fixes, in problem order. It is complete for a study of the last level (or an
        ordinary one); a study whose trials stand for studies of the level below lacks the
        values those choose. Of a trial interrupted before it was proposed everything, what it
        has."""
        values = {**self.fixed, **params}
        return {p.name: values[p.name] for p in self.spec.parameters if p.name in values}

    def to_dict(self) -> dict[str, object]:
        """What ``study.json`` records under ``scope``."""
        return {
            "level": self.level,
            "searched": list(self.searched),
            "fixed": thaw(self.fixed),
            "replicate": self.replicate,
            "parent": None if self.parent is None else thaw(self.parent),
        }

    @classmethod
    def recorded(cls, spec: ProblemSpec, document: Mapping[str, Any]) -> StudyScope:
        """The scope ``study.json`` (``document``) records for a study of ``spec``, checked: it
        must be the one :meth:`of` makes of its ``level``, ``fixed`` and ``parent``."""
        recorded = document.get("scope")
        if not isinstance(recorded, Mapping):
            raise OptimizationError("study.json records no scope of the study")
        fixed, parent = recorded.get("fixed"), recorded.get("parent")
        if not isinstance(fixed, Mapping) or not (parent is None or isinstance(parent, Mapping)):
            raise OptimizationError(f"study.json records no scope k4opt reads: {recorded!r}")
        scope = cls.of(spec, fixed, level=recorded.get("level"), parent=parent)  # type: ignore[arg-type]
        if json.dumps(thaw(recorded)) != json.dumps(scope.to_dict()):  # type and value
            raise OptimizationError(
                f"study.json records the scope {thaw(recorded)!r}, but its problem makes it "
                f"{scope.to_dict()!r}"
            )
        return scope


def _own(spec: ProblemSpec, level: int) -> tuple[str, ...]:
    """The parameters a study at ``level`` searches before any is fixed: those of its level, or
    every parameter of an ordinary problem."""
    if not spec.levels:
        return tuple(parameter.name for parameter in spec.parameters)
    return spec.levels[level].parameters


def limit_names(criteria: Criteria) -> list[str]:
    """The name of each limit of ``criteria``, as the optimizer knows its constraint value."""
    names = [
        f"{c.label}:{bound}"
        for c in criteria.constraints
        for bound in ("min", "max")
        if getattr(c, bound) is not None
    ]
    assert len(set(names)) == len(names)  # constraints are unique by metric reference
    return names
