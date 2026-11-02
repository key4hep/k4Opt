"""Internal: the readers of the values of a problem document, each failing with where it is."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from k4opt.problem.errors import ProblemConfigError
from k4opt.problem.models import (
    Number,
)


def _fail(path: str, message: str) -> ProblemConfigError:
    return ProblemConfigError(f"{path or 'top level'}: {message}")


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _show(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _mapping(
    value: object, path: str, allowed: tuple[str, ...], required: tuple[str, ...] = ()
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, f"must be a mapping, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise _fail(path, f"keys must be strings, got {_show(key)}")
    unknown = [key for key in value if key not in allowed]
    if unknown:
        raise _fail(path, f"unknown key(s) {unknown}; allowed: {sorted(allowed)}")
    missing = [key for key in required if key not in value]
    if missing:
        raise _fail(path, f"missing required key(s) {missing}")
    return value


def _sequence(value: object, path: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise _fail(path, f"must be a list, got {type(value).__name__}")
    return list(value)


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        got = "a blank string" if isinstance(value, str) else f"{type(value).__name__}"
        quote = ""
        if value is not None and not isinstance(value, (str, list, tuple, Mapping)):
            got += f" {_show(value)}"
            quote = "; quote it if it is text"
        raise _fail(path, f"must be a non-empty string, got {got}{quote}")
    return value


def _optional_text(m: Mapping[str, Any], key: str, path: str) -> str | None:
    return _text(m[key], _join(path, key)) if key in m else None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: object, path: str) -> Number:
    if not _is_number(value):
        raise _fail(path, f"must be a number, got {type(value).__name__} {_show(value)}")
    assert isinstance(value, (int, float))
    if isinstance(value, float) and not math.isfinite(value):  # an int is always finite
        raise _fail(path, f"must be finite, got {value!r}")
    return value


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _fail(path, f"must be an integer, got {type(value).__name__} {_show(value)}")
    if minimum is not None and value < minimum:
        raise _fail(path, f"must be >= {minimum}, got {value}")
    return value


def _choice(value: object, path: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _fail(path, f"must be one of {list(allowed)}, got {_show(value)}")
    return value
