"""Internal: the grid, a strategy that proposes the points of a Cartesian grid one after another.

The grid of a study is every combination of the values of the parameters it searches, in problem
order, the last parameter varying fastest; it learns nothing from the trials, so what it
proposes next depends on how many points it proposed already, never on how or in which order
trials ended. Its values are given for every parameter the studies of an optimization search,
at every level (:func:`grid_values`): those given, or every choice of a categorical parameter
and every integer of an integer one; a continuous parameter needs them given. ``study.json``
records them (:func:`config`), so that every study of a nested optimization takes the grid of
its own parameters from them. A grid of more than :data:`LIMIT` points is refused before
anything is made: each point is a candidate to evaluate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from k4opt.evaluation import Scalar
from k4opt.frozen import same_json
from k4opt.optimization._proposal import EXHAUSTED, Answer, Feedback, Past, Proposal
from k4opt.optimization._scope import StudyScope
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import CandidateError, Parameter, ProblemSpec
from k4opt.problem.candidate import validate_value

NAME = "grid"
LIMIT = 10_000
"""The most points the grid of one study may have."""


def _searched(scope: StudyScope) -> list[Parameter]:
    """The parameters the studies of ``scope`` search, at its level and every level below, in
    problem order."""
    spec = scope.spec
    names = {
        name for level in range(scope.level, spec.depth + 1) for name in spec.parameters_of(level)
    }
    return [p for p in spec.parameters if p.name in names and p.name not in scope.fixed]


def grid_values(
    scope: StudyScope, given: Mapping[str, Sequence[object]] | None
) -> dict[str, list[Scalar]]:
    """The values the grid takes for each parameter the studies of ``scope`` search, in problem
    order: those ``given`` (each of its parameter's type and in its domain, none twice; a
    continuous parameter's integers as the floats they are), or every choice of a categorical
    parameter, every integer of an integer one; a continuous parameter needs them given. Raises
    :class:`OptimizationError` for values of a parameter the studies do not search, or a grid of
    a study of more than :data:`LIMIT` points."""
    searched = _searched(scope)
    names = [p.name for p in searched]
    given = dict(given or {})
    for name in given:
        if name in scope.fixed:
            raise OptimizationError(f"grid: {name} is fixed, so the grid does not search it")
        if name not in names:
            raise OptimizationError(
                f"grid: {name} is no parameter the study searches ({', '.join(names)})"
            )
    values: dict[str, list[Scalar]] = {}
    for parameter in searched:
        name = parameter.name
        if name in given:
            values[name] = _given(parameter, given[name])
        elif parameter.kind == "categorical":
            assert parameter.choices is not None
            values[name] = list(parameter.choices)
        elif parameter.kind == "integer":
            assert parameter.bounds is not None
            low, high = (int(bound) for bound in parameter.bounds)
            if high - low + 1 > LIMIT:
                raise OptimizationError(
                    f"grid: {name} has {high - low + 1} integers in [{low}, {high}], more than a "
                    f"grid takes ({LIMIT}); give the values it takes, as {name}=V1,V2,..."
                )
            values[name] = list(range(low, high + 1))
        else:
            raise OptimizationError(
                f"grid: {name} is continuous, so the grid needs the values it takes, as "
                f"{name}=V1,V2,..."
            )
    spec = scope.spec
    for level in range(scope.level, spec.depth + 1):
        own = [n for n in names if n in spec.parameters_of(level)]
        size = math.prod(len(values[n]) for n in own)
        if size > LIMIT:
            sizes = " by ".join(f"{n} {len(values[n])}" for n in own)
            where = "" if spec.depth == 0 else f" of level {level}"
            raise OptimizationError(
                f"grid: the grid{where} has {size} points ({sizes}), more than {LIMIT}; give "
                "fewer values"
            )
    return values


def _given(parameter: Parameter, items: object) -> list[Scalar]:
    name = parameter.name
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or not items:
        raise OptimizationError(f"grid: {name} needs a list of one value or more, not {items!r}")
    values: list[Scalar] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        try:
            key = validate_value(parameter, item)
        except CandidateError as exc:
            raise OptimizationError(f"grid: {exc}") from None
        if key in seen:
            raise OptimizationError(f"grid: {name} takes {item!r} more than once")
        seen.add(key)
        if parameter.kind == "continuous" and isinstance(item, int):
            if float(item) != item:
                raise OptimizationError(f"grid: {name}: no float holds {item} exactly")
            item = float(item)
        elif parameter.kind == "categorical" and not isinstance(item, (str, bool)):
            assert parameter.choices is not None  # the choice itself, as declared: 1, not 1.0
            item = next(c for c in parameter.choices if not isinstance(c, bool) and c == item)
        values.append(item)
    return values


def config(scope: StudyScope, values: Mapping[str, Sequence[Scalar]]) -> dict[str, Any]:
    """What ``study.json`` records of the grid of the study of ``scope``: the values of every
    parameter it and the studies below it search, and how many points its own grid has."""
    names = [p.name for p in _searched(scope)]
    return {
        "name": NAME,
        "values": {name: list(values[name]) for name in names},
        "points": math.prod(len(values[name]) for name in scope.searched),
    }


def recorded_values(recorded: Mapping[str, Any], scope: StudyScope) -> dict[str, list[Scalar]]:
    """The values a study's recorded grid gives, checked against the problem as
    :func:`grid_values` checks them; :class:`OptimizationError` for a grid that is not one of
    ``scope``."""
    values = recorded.get("values")
    if not isinstance(values, Mapping):
        raise OptimizationError(f"the recorded grid has no values: {recorded!r}")
    checked = grid_values(scope, {name: list(items) for name, items in values.items()})
    if not same_json(dict(recorded), config(scope, checked)):
        raise OptimizationError(
            f"the recorded grid {dict(recorded)!r} is not the one its values make for the study"
        )
    return checked


def differences(recorded: Mapping[str, Any], scope: StudyScope) -> list[str]:
    """Why the recorded grid cannot run the study of ``scope``; none for one it can."""
    try:
        recorded_values(recorded, scope)
    except OptimizationError as exc:
        return [str(exc)]
    return []


def child(recorded: Mapping[str, Any], scope: StudyScope) -> dict[str, Any]:
    """The grid of the study ``scope`` of the level below, started for a trial of the study
    whose grid is ``recorded``: the values of the parameters it and the levels below search."""
    return config(scope, recorded["values"])


def size(spec: ProblemSpec, values: Mapping[str, Sequence[Scalar]], level: int) -> int:
    """How many points the grid of a study of ``level`` has, of the parameters of its level
    that ``values`` gives (those it does not give are fixed)."""
    return math.prod(len(values[n]) for n in spec.parameters_of(level) if n in values)


class Grid:
    """The grid of one study, made from its recorded ``config`` and its trials so far, which must
    be its first points, in order: it goes on from the next one."""

    def __init__(
        self, scope: StudyScope, recorded: Mapping[str, Any], history: Sequence[Past]
    ) -> None:
        values = recorded_values(recorded, scope)
        self._names = scope.searched
        self._axes = [values[name] for name in self._names]
        self._points = math.prod(len(axis) for axis in self._axes)
        self._next = 0
        for past in history:
            if past.note != {"point": self._next} or not _same(past.values, self.point(self._next)):
                raise OptimizationError(
                    f"trial {past.number} of the study is not point {self._next} of its grid "
                    f"({self.point(self._next) if self._next < self._points else 'none left'}), "
                    "so the grid cannot go on from it"
                )
            self._next += 1

    def point(self, index: int) -> dict[str, Scalar]:
        """The values of point ``index`` (from 0), the last parameter varying fastest."""
        values: dict[str, Scalar] = {}
        for name, axis in reversed(list(zip(self._names, self._axes, strict=True))):
            index, position = divmod(index, len(axis))
            values[name] = axis[position]
        return {name: values[name] for name in self._names}

    @property
    def exhausted(self) -> bool:
        """Whether every point was proposed."""
        return self._next >= self._points

    def propose(self, number: int) -> Proposal | Answer:
        if self._next >= self._points:
            return EXHAUSTED
        index, self._next = self._next, self._next + 1
        return Proposal(self.point(index), {"point": index})

    def tell(self, number: int, feedback: Feedback) -> None:
        """The grid learns nothing from how a trial ended."""


def _same(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Equal values, type and value: 1, 1.0 and True are other points."""
    return [(k, type(v), v) for k, v in a.items()] == [(k, type(v), v) for k, v in b.items()]
