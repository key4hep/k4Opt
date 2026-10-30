"""Override records and their resolution against a catalog.

An :class:`Override` says *which* attribute to change (by target reference and
optional file discriminator) and the literal replacement text. Resolution turns
it into a :class:`ResolvedOverride` bound to exactly one catalogued attribute.
Values are never evaluated, converted or type-inferred.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from k4opt.geometry.discovery import Catalog, Parameter
from k4opt.geometry.errors import DuplicateTargetError, InvalidTargetError, TargetError
from k4opt.geometry.references import parse_reference

#: Characters that cannot appear in an XML 1.0 document (control characters, lone surrogates).
_XML_INVALID = re.compile("[^\t\n\r\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


@dataclass(frozen=True, slots=True)
class Override:
    """A requested change: set the attribute addressed by ``target`` to ``value``."""

    target: str
    value: str
    file: str | None = None
    """Optional root-relative path disambiguating a target that exists in several files."""

    def __post_init__(self) -> None:
        if not isinstance(self.target, str):
            raise InvalidTargetError(
                f"override target must be a string, got {type(self.target).__name__}"
            )
        parse_reference(self.target)
        if not isinstance(self.value, str):
            raise InvalidTargetError(
                f"override value for {self.target!r} must be a string, got "
                f"{type(self.value).__name__} ({self.value!r}); "
                f'pass it as text, e.g. "{self.value}"'
            )
        if _XML_INVALID.search(self.value):
            raise InvalidTargetError(
                f"override value for {self.target!r} contains characters that are not valid in XML"
            )
        if self.file is not None and not isinstance(self.file, str):
            raise InvalidTargetError(f"override file for {self.target!r} must be a string")


@dataclass(frozen=True, slots=True)
class ResolvedOverride:
    """An override bound to the single catalogued attribute it modifies."""

    override: Override
    parameter: Parameter

    @property
    def target(self) -> str:
        return self.parameter.target

    @property
    def file(self) -> str:
        return self.parameter.file

    @property
    def attribute(self) -> str:
        return self.parameter.attribute

    @property
    def locator(self) -> str:
        return self.parameter.locator

    @property
    def old_value(self) -> str:
        return self.parameter.value

    @property
    def new_value(self) -> str:
        return self.override.value

    @property
    def changed(self) -> bool:
        return self.old_value != self.new_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "file": self.file,
            "attribute": self.attribute,
            "locator": self.locator,
            "line": self.parameter.line,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


def resolve_overrides(
    catalog: Catalog, overrides: Iterable[Override]
) -> tuple[ResolvedOverride, ...]:
    """Bind every override to exactly one attribute, reporting all failures at once.

    Raises the specific :class:`~k4opt.errors.TargetError` subclass when a single
    override fails, or a :class:`~k4opt.errors.TargetError` listing all failures
    in ``errors`` when several do.
    """
    errors: list[TargetError] = []
    resolved: list[ResolvedOverride] = []
    seen: dict[tuple[str, str, str], ResolvedOverride] = {}
    for override in overrides:
        try:
            if not isinstance(override, Override):
                raise InvalidTargetError(f"expected an Override, got {type(override).__name__}")
            parameter = catalog.lookup(override.target, override.file)
        except TargetError as exc:
            errors.append(exc)
            continue
        key = (parameter.file, parameter.locator, parameter.attribute)
        if key in seen:
            first = seen[key]
            errors.append(
                DuplicateTargetError(
                    f"overrides {first.override.target!r} and {override.target!r} "
                    "both select attribute "
                    f"{parameter.attribute!r} of {parameter.locator} in {parameter.file}"
                )
            )
            continue
        bound = ResolvedOverride(override, parameter)
        seen[key] = bound
        resolved.append(bound)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        summary = "\n".join("  - " + str(error).replace("\n", "\n    ") for error in errors)
        raise TargetError(f"{len(errors)} overrides could not be resolved:\n{summary}", errors)
    return tuple(resolved)
