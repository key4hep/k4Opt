"""What an evaluator implementation declares of itself, as plain data a record keeps.

:class:`Declaration` is the scientific contract of an implementation in a normalized form:
its name and version, the stage it reads, its metrics with their units, the names of its
artifacts and its roles. :meth:`Declaration.of` reads it from an instance, :meth:`to_dict`
writes it as a record holds it and :meth:`Declaration.from_dict` reads it back, so a record
is checked against what its implementation declared without that implementation. Nothing in
it is Python-specific: the defaults of settings (a :class:`~k4opt.evaluation.Quantity`), the
:class:`~k4opt.evaluation.Collection` objects and what each artifact holds stay with the
instance that declared them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from k4opt.errors import K4OptError
from k4opt.evaluation.base import Role, declared_artifacts, declared_roles, metric_units
from k4opt.evaluation.errors import EvaluationContractError
from k4opt.evaluation.models import ARTIFACT_NAME
from k4opt.frozen import FrozenMap, Json


@dataclass(frozen=True, slots=True)
class Declaration:
    """What an implementation declares: ``name`` and ``version``, the ``stage`` it reads
    (``None``: none), its ``metrics`` by ``(stage, metric)`` with the unit every measurement of
    one carries, the names of the ``artifacts`` a ``valid`` result of it returns and its
    ``roles``; ``None`` for metrics, artifacts or roles it does not declare at all."""

    name: str
    version: str
    stage: str | None = None
    metrics: Mapping[tuple[str, str], str] | None = None
    artifacts: tuple[str, ...] | None = None
    roles: Mapping[str, Role] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.metrics is not None:
            object.__setattr__(self, "metrics", FrozenMap(dict(self.metrics)))
        if self.artifacts is not None:
            object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.roles is not None:
            object.__setattr__(self, "roles", FrozenMap(dict(self.roles)))

    @classmethod
    def of(cls, evaluator: Any) -> Declaration:
        """What ``evaluator`` declares; raises :class:`EvaluationContractError` for a
        declaration that cannot be read."""
        artifacts = declared_artifacts(evaluator)
        return cls(
            evaluator.name,
            evaluator.version,
            getattr(evaluator, "stage", None),
            metric_units(evaluator),
            None if artifacts is None else tuple(artifacts),
            declared_roles(evaluator),
        )

    def to_dict(self) -> dict[str, Json]:
        """The declaration as a record holds it, by its implementation's name: all but the
        name, each metric as ``{stage, metric, unit}`` and each role as ``{many, stage, metrics,
        artifacts}``."""
        metrics: tuple[Json, ...] | None = None
        if self.metrics is not None:
            metrics = tuple(
                {"stage": s, "metric": m, "unit": u} for (s, m), u in self.metrics.items()
            )
        roles: dict[str, Json] | None = None
        if self.roles is not None:
            roles = {
                name: {
                    "many": role.many,
                    "stage": role.stage,
                    "metrics": tuple(role.metrics),
                    "artifacts": role.artifacts,
                }
                for name, role in self.roles.items()
            }
        return {
            "version": self.version,
            "stage": self.stage,
            "metrics": metrics,
            "artifacts": self.artifacts,
            "roles": roles,
        }

    @classmethod
    def from_dict(cls, name: str, data: object, error: type[K4OptError]) -> Declaration:
        """The declaration of the implementation ``name`` that a record holds as ``data``
        (:meth:`to_dict`); raises ``error`` for data that is no such declaration."""
        where = f"implementation {name!r}"
        entry = _mapping(data, where, error)
        version, stage = entry.get("version"), entry.get("stage")
        if not isinstance(version, str) or not version.strip():
            raise error(f"{where}: version must be a nonblank string, got {version!r}")
        if stage is not None and (not isinstance(stage, str) or not stage.strip()):
            raise error(f"{where}: stage must be null or a nonblank string, got {stage!r}")
        return cls(
            name,
            version,
            stage,
            _metrics(entry.get("metrics"), where, error),
            _names(entry.get("artifacts"), f"{where}: artifacts", error),
            _roles(entry.get("roles"), where, error),
        )


def _mapping(value: object, where: str, error: type[K4OptError]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _list(value: object, where: str, error: type[K4OptError]) -> list[object]:
    """A list as JSON reads it back, or a tuple as :meth:`Declaration.to_dict` writes it."""
    if not isinstance(value, (list, tuple)):
        raise error(f"{where} must be a list, got {type(value).__name__}")
    return list(value)


def _text(value: object, where: str, error: type[K4OptError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{where} must be a nonblank string, got {value!r}")
    return value


def _metrics(
    value: object, where: str, error: type[K4OptError]
) -> dict[tuple[str, str], str] | None:
    if value is None:
        return None
    metrics: dict[tuple[str, str], str] = {}
    for index, item in enumerate(_list(value, f"{where}: metrics", error)):
        at = f"{where}: metrics[{index}]"
        entry = _mapping(item, at, error)
        stage, metric, unit = (_text(entry.get(key), f"{at}.{key}", error) for key in _METRIC)
        if (stage, metric) in metrics:
            raise error(f"{where}: {stage}/{metric} is declared twice")
        metrics[(stage, metric)] = unit
    return metrics


_METRIC = ("stage", "metric", "unit")


def _names(value: object, where: str, error: type[K4OptError]) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = _list(value, where, error)
    for name in names:
        if not isinstance(name, str) or not ARTIFACT_NAME.fullmatch(name):
            raise error(f"{where}: {name!r} is no artifact name")
    if len(set(names)) != len(names):
        raise error(f"{where}: {names} names one twice")
    return tuple(str(name) for name in names)


def _roles(value: object, where: str, error: type[K4OptError]) -> dict[str, Role] | None:
    if value is None:
        return None
    roles: dict[str, Role] = {}
    for name, item in _mapping(value, f"{where}: roles", error).items():
        at = f"{where}: roles.{name}"
        entry = _mapping(item, at, error)
        metrics = _list(entry.get("metrics"), f"{at}.metrics", error)
        pairs = [tuple(key) if isinstance(key, (list, tuple)) else key for key in metrics]
        try:
            roles[name] = Role(  # which checks each field as a declared role's
                many=cast(bool, entry.get("many")),
                metrics=cast(tuple[tuple[str, str], ...], tuple(pairs)),
                artifacts=_names(entry.get("artifacts"), f"{at}.artifacts", error) or (),
                stage=cast(str | None, entry.get("stage")),
            )
        except EvaluationContractError as exc:
            raise error(f"{at}: {exc}") from None
    return roles
