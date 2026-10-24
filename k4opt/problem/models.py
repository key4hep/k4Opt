"""Immutable domain models of a parsed optimization problem.

The models describe the draft ``k4opt.problem/v1alpha1`` format of ``docs/problem.md``.
They hold declarative data only: nothing here resolves geometry targets, looks up an
evaluator or metric, or touches the Key4hep stack. Evaluator-owned content (``settings``
and backend arguments) is stored as an immutable copy, so neither the caller's YAML data
nor the object handed out by :meth:`ProblemSpec.to_dict` aliases the stored state.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

from k4opt.problem.errors import ProblemConfigError

SCHEMA_VERSION = "k4opt.problem/v1alpha1"

type Json = bool | int | float | str | tuple[Json, ...] | Mapping[str, Json] | None
"""A JSON-compatible value as stored in a model: sequences are tuples, mappings read-only."""

type Number = int | float

Channel = Literal["release", "nightly"]
ParameterKind = Literal["continuous", "integer", "categorical"]
Direction = Literal["minimize", "maximize"]

_MAX_DEPTH = 64
_MAX_VALUES = 1_000_000


class _FrozenMap[V](Mapping[str, V]):
    """A read-only, hashable mapping that keeps insertion order."""

    __slots__ = ("_items",)

    def __init__(self, items: Mapping[str, V]) -> None:
        self._items = dict(items)

    def __getitem__(self, key: str) -> V:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        # equal mappings compare equal whatever their key order, so they must hash alike
        return hash(frozenset(self._items.items()))

    def __repr__(self) -> str:
        return f"{self._items!r}"


def freeze(value: object, path: str) -> Json:
    """Return an immutable copy of a JSON-compatible ``value``.

    Accepts ``None``, booleans, integers, finite floats, strings, lists (or tuples) and
    mappings with string keys, nested to a bounded depth. Anything else, including
    non-finite numbers, dates, bytes, sets and self-referencing structures, raises
    :class:`ProblemConfigError` naming ``path``. A value shared by several YAML aliases is
    frozen once; a document whose aliases expand to more than ``_MAX_VALUES`` values is
    refused, so that serializing or hashing the result stays cheap.
    """
    return _freeze(value, path, _Walk(), 0)[0]


class _Walk:
    """Bookkeeping of one :func:`freeze` call."""

    def __init__(self) -> None:
        self.active: set[int] = set()  # containers being frozen now: a hit is a cycle
        self.done: dict[int, tuple[Json, int]] = {}  # frozen container, size once expanded


def _freeze(value: object, path: str, walk: _Walk, depth: int) -> tuple[Json, int]:
    """Freeze ``value``; the second item is the number of values it expands to."""
    if value is None or isinstance(value, (str, bool, int)):
        return value, 1
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProblemConfigError(f"{path}: non-finite number {value!r} is not allowed")
        return value, 1
    if isinstance(value, (list, tuple, Mapping)):
        if id(value) in walk.done:  # an alias of a container that was already frozen
            return walk.done[id(value)]
        if depth >= _MAX_DEPTH:
            raise ProblemConfigError(f"{path}: nested deeper than {_MAX_DEPTH} levels")
        if id(value) in walk.active:
            raise ProblemConfigError(f"{path}: contains itself (recursive YAML alias)")
        walk.active.add(id(value))
        try:
            size = 1
            frozen: Json
            if isinstance(value, Mapping):
                items: dict[str, Json] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ProblemConfigError(
                            f"{path}: mapping keys must be strings, got {key!r}"
                        )
                    items[key], count = _freeze(item, f"{path}.{key}", walk, depth + 1)
                    size += count
                frozen = _FrozenMap(items)
            else:
                elements: list[Json] = []
                for index, item in enumerate(value):
                    element, count = _freeze(item, f"{path}[{index}]", walk, depth + 1)
                    elements.append(element)
                    size += count
                frozen = tuple(elements)
            if size > _MAX_VALUES:
                raise ProblemConfigError(
                    f"{path}: aliases expand to more than {_MAX_VALUES} values"
                )
        finally:
            walk.active.discard(id(value))
        walk.done[id(value)] = (frozen, size)
        return frozen, size
    raise ProblemConfigError(
        f"{path}: unsupported value of type {type(value).__name__}; "
        "allowed: null, booleans, numbers, strings, lists and mappings with string keys"
    )


def thaw(value: Json) -> object:
    """Return a plain, mutable ``dict``/``list`` copy of a value produced by :func:`freeze`."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Key4hepStack:
    """The pinned Key4hep software environment (``environment.key4hep``).

    The pin identifies an environment; it does not guarantee that it is installed.
    """

    channel: Channel
    version: str

    def to_dict(self) -> dict[str, object]:
        return {"channel": self.channel, "version": self.version}


@dataclass(frozen=True, slots=True)
class Environment:
    """Where the problem is meant to be evaluated."""

    key4hep: Key4hepStack

    def to_dict(self) -> dict[str, object]:
        return {"key4hep": self.key4hep.to_dict()}


@dataclass(frozen=True, slots=True)
class GeometrySpec:
    """The starting detector description: a compact file name, relative to the k4geo tree."""

    entry: str

    def to_dict(self) -> dict[str, object]:
        return {"entry": self.entry}


@dataclass(frozen=True, slots=True)
class GeometrySelector:
    """Which geometry quantity a parameter varies.

    Exactly one of ``constant`` (shorthand for the target ``constants.<name>``) and
    ``target`` (any canonical target reference of :mod:`k4opt.geometry`) is set. The
    reference is not checked against a geometry here.
    """

    constant: str | None = None
    target: str | None = None

    @property
    def canonical_target(self) -> str:
        """The target reference both spellings stand for."""
        if self.constant is not None:
            return f"constants.{self.constant}"
        assert self.target is not None
        return self.target

    def to_dict(self) -> dict[str, object]:
        if self.constant is not None:
            return {"constant": self.constant}
        return {"target": self.target}


@dataclass(frozen=True, slots=True)
class Parameter:
    """A quantity that varies between candidates, with its search domain.

    ``continuous`` and ``integer`` parameters have inclusive ``bounds``; ``categorical``
    ones have ``choices``. ``unit`` is a declared label, never interpreted.
    """

    geometry: GeometrySelector
    kind: ParameterKind
    bounds: tuple[Number, Number] | None = None
    choices: tuple[str | bool | Number, ...] | None = None
    unit: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"geometry": self.geometry.to_dict(), "type": self.kind}
        if self.bounds is not None:
            data["bounds"] = list(self.bounds)
        if self.choices is not None:
            data["choices"] = list(self.choices)
        if self.unit is not None:
            data["unit"] = self.unit
        return data


@dataclass(frozen=True, slots=True)
class Evaluation:
    """How a candidate is measured: a proposed standard evaluator plus its inputs.

    ``settings`` is owned by the evaluator's future contract and ``backend`` maps a tool
    name to its ordered, uninterpreted argument list.
    """

    evaluator: str
    version: str
    events: int | None = None
    seed: int | None = None
    settings: Mapping[str, Json] = field(default_factory=lambda: _FrozenMap({}))
    backend: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: _FrozenMap({}))

    def __post_init__(self) -> None:
        # Direct construction gets the same defensive copy as parsing does.
        settings = freeze(self.settings, "settings")
        assert isinstance(settings, Mapping)
        object.__setattr__(self, "settings", settings)
        object.__setattr__(
            self,
            "backend",
            _FrozenMap({name: tuple(args) for name, args in self.backend.items()}),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"evaluator": self.evaluator, "version": self.version}
        if self.events is not None:
            data["events"] = self.events
        if self.seed is not None:
            data["seed"] = self.seed
        if self.settings:
            data["settings"] = thaw(self.settings)
        if self.backend:
            data["backend"] = {name: {"args": list(args)} for name, args in self.backend.items()}
        return data


@dataclass(frozen=True, slots=True)
class Objective:
    """A metric to minimize or maximize. ``stage`` classifies the metric."""

    stage: str
    metric: str
    direction: Direction

    def to_dict(self) -> dict[str, object]:
        return {"stage": self.stage, "metric": self.metric, "direction": self.direction}


@dataclass(frozen=True, slots=True)
class Constraint:
    """A hard, inclusive limit on a measured metric: ``min`` and/or ``max``."""

    stage: str
    metric: str
    min: Number | None = None
    max: Number | None = None
    unit: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"stage": self.stage, "metric": self.metric}
        if self.min is not None:
            data["min"] = self.min
        if self.max is not None:
            data["max"] = self.max
        if self.unit is not None:
            data["unit"] = self.unit
        return data


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """A validated problem definition (``k4opt.problem/v1alpha1``).

    Built by :func:`k4opt.problem.load_problem` or :func:`k4opt.problem.parse_problem`.
    Validation is limited to what can be known without Key4hep: nothing here claims that a
    constant exists, that an evaluator or metric is defined, or that the ranges make
    scientific sense.
    """

    environment: Environment
    geometry: GeometrySpec
    parameters: tuple[Parameter, ...]
    evaluation: Evaluation
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...] = ()
    name: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-compatible data in schema order; a fresh copy on every call.

        ``parse_problem(spec.to_dict())`` gives back an equal :class:`ProblemSpec`.
        """
        data: dict[str, object] = {"schema_version": SCHEMA_VERSION}
        if self.name is not None:
            data["name"] = self.name
        data["environment"] = self.environment.to_dict()
        data["geometry"] = self.geometry.to_dict()
        data["parameters"] = [parameter.to_dict() for parameter in self.parameters]
        data["evaluation"] = self.evaluation.to_dict()
        data["objectives"] = [objective.to_dict() for objective in self.objectives]
        if self.constraints:
            data["constraints"] = [constraint.to_dict() for constraint in self.constraints]
        return data
