"""Immutable copies of JSON-compatible data, shared by the components that store it.

``settings`` of a problem and of an evaluation request, backend arguments and result
provenance all hold caller-owned nested data. :func:`freeze` copies it into read-only
containers, so that neither later changes by the caller nor by a reader reach the stored
state, and :func:`thaw` hands out a plain mutable copy. Each component says which error
:func:`freeze` raises, so a bad value is reported in that component's terms.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from typing import NamedTuple

type Json = bool | int | float | str | tuple[Json, ...] | Mapping[str, Json] | None
"""A JSON-compatible value as stored in a model: sequences are tuples, mappings read-only."""

_MAX_DEPTH = 64
_MAX_VALUES = 1_000_000


class FrozenMap[V](Mapping[str, V]):
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


def freeze(value: object, path: str, error: Callable[[str], Exception]) -> Json:
    """Return an immutable copy of a JSON-compatible ``value``.

    Accepts ``None``, booleans, integers, finite floats, strings, lists (or tuples) and
    mappings with string keys, nested to a bounded depth. Anything else, including
    non-finite numbers, dates, bytes, sets and self-referencing structures, raises
    ``error(message)`` with a message naming ``path``. A value shared by several YAML aliases
    is frozen once, yet counts with its full size and depth at every place it is used; a
    document whose aliases expand to more than ``_MAX_VALUES`` values is refused, so that
    serializing or hashing the result stays cheap.
    """
    return _freeze(value, path, _Walk(error), 0).value


class _Frozen(NamedTuple):
    value: Json
    size: int  # number of values it expands to
    height: int  # levels of containers it spans: 0 for a scalar


class _Walk:
    """Bookkeeping of one :func:`freeze` call."""

    def __init__(self, error: Callable[[str], Exception]) -> None:
        self.error = error
        self.active: set[int] = set()  # containers being frozen now: a hit is a cycle
        self.done: dict[int, _Frozen] = {}  # containers already frozen, by identity


def _freeze(value: object, path: str, walk: _Walk, depth: int) -> _Frozen:
    if value is None or isinstance(value, (str, bool, int)):
        return _Frozen(value, 1, 0)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise walk.error(f"{path}: non-finite number {value!r} is not allowed")
        return _Frozen(value, 1, 0)
    if isinstance(value, (list, tuple, Mapping)):
        if id(value) in walk.done:  # an alias of a container that was already frozen
            done = walk.done[id(value)]
            if depth + done.height > _MAX_DEPTH:
                raise walk.error(f"{path}: nested deeper than {_MAX_DEPTH} levels")
            return done
        if depth >= _MAX_DEPTH:
            raise walk.error(f"{path}: nested deeper than {_MAX_DEPTH} levels")
        if id(value) in walk.active:
            raise walk.error(f"{path}: contains itself (recursive YAML alias)")
        walk.active.add(id(value))
        try:
            size, height = 1, 1
            frozen: Json
            if isinstance(value, Mapping):
                items: dict[str, Json] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise walk.error(f"{path}: mapping keys must be strings, got {key!r}")
                    child = _freeze(item, f"{path}.{key}", walk, depth + 1)
                    items[key] = child.value
                    size += child.size
                    height = max(height, child.height + 1)
                frozen = FrozenMap(items)
            else:
                elements: list[Json] = []
                for index, item in enumerate(value):
                    child = _freeze(item, f"{path}[{index}]", walk, depth + 1)
                    elements.append(child.value)
                    size += child.size
                    height = max(height, child.height + 1)
                frozen = tuple(elements)
            if size > _MAX_VALUES:
                raise walk.error(f"{path}: aliases expand to more than {_MAX_VALUES} values")
        finally:
            walk.active.discard(id(value))
        result = _Frozen(frozen, size, height)
        walk.done[id(value)] = result
        return result
    raise walk.error(
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
