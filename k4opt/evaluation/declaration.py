"""What an evaluator implementation declares of itself: read from its class, kept as plain data.

An evaluator declares, as plain data on its class: ``name`` and ``version``, the ``stage`` it
reads, the ``collections`` it reads from that stage's file (:class:`Collection`, by role), its
``settings`` (:mod:`k4opt.evaluation.settings`), its ``metrics`` with their units, the
``artifacts`` a ``valid`` result returns and, for a combined evaluator, its ``roles``
(:class:`Role`). k4opt reads them from the class, never from an instance, and without running any
of the class's code (:func:`check_class`): binding an evaluator to a problem creates nothing, and
an instance, created only where an evaluation runs, cannot declare otherwise.

:class:`Declaration` is the scientific part of that contract in a normalized form: the name and
version, the stage, the metrics with their units, the names of the artifacts and the roles.
:meth:`Declaration.of` reads it from a class, :meth:`~Declaration.to_dict` writes it as a record
holds it and :meth:`Declaration.from_dict` reads it back, so a record is checked against what its
implementation declared without that implementation. Nothing in it is Python-specific: the
defaults of settings (a :class:`~k4opt.evaluation.Quantity`), the :class:`Collection` objects and
what each artifact holds stay with the class that declared them.
"""

from __future__ import annotations

import inspect
import keyword
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from k4opt.errors import K4OptError
from k4opt.evaluation.errors import EvaluationContractError
from k4opt.evaluation.models import ARTIFACT_NAME
from k4opt.frozen import FrozenMap, Json

DECLARED = ("name", "version", "stage", "collections", "settings", "metrics", "artifacts", "roles")
"""What an evaluator class declares, as plain data on the class."""


_TYPE = re.compile(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)?")


@dataclass(frozen=True, slots=True)
class Collection:
    """An EDM4hep collection an evaluator reads from the file of its stage.

    ``type`` is the type of its elements, ``"SimCalorimeterHit"`` (the ``edm4hep::`` prefix is
    implied). The problem names the collection in the evaluator's ``collections``, unless
    ``default`` names it (``"MCParticles"``); the problem may still name another.
    """

    type: str
    default: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not _TYPE.fullmatch(self.type):
            raise EvaluationContractError(
                f"Collection: type must be an EDM4hep type such as 'SimCalorimeterHit', "
                f"got {self.type!r}"
            )
        if self.default is not None and (not isinstance(self.default, str) or not self.default):
            raise EvaluationContractError(
                f"Collection: default must be a collection name, got {self.default!r}"
            )

    @property
    def podio_type(self) -> str:
        """``type`` as podio names it, ``edm4hep::SimCalorimeterHit``."""
        return self.type if "::" in self.type else f"edm4hep::{self.type}"


@dataclass(frozen=True, slots=True)
class Role:
    """An input of a combined evaluator: what each item the problem gives it must provide.

    An item is the evaluation of a sample by another evaluator (``{sample: zz, evaluator:
    zh.selection}``), whose ``metrics`` (``(stage, metric)`` pairs) and ``artifacts`` (names) this
    role reads: that evaluator must declare them, and a ``valid`` result of it carries them. A role
    with a ``stage`` reads instead the file of that stage of a sample (``{sample: zz, stage:
    simulation}``). A role takes one item, or with ``many`` one or more; ``request.roles`` holds a
    tuple of :class:`~k4opt.evaluation.RoleInput` either way.
    """

    many: bool = False
    metrics: tuple[tuple[str, str], ...] = ()
    artifacts: tuple[str, ...] = ()
    stage: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.many, bool):
            raise EvaluationContractError(f"Role: many must be True or False, got {self.many!r}")
        if not isinstance(self.metrics, (list, tuple)) or not all(
            isinstance(key, tuple)
            and len(key) == 2
            and all(isinstance(part, str) and part.strip() for part in key)
            for key in self.metrics
        ):
            raise EvaluationContractError(
                "Role: metrics must be (stage, metric) pairs, such as ('analysis', 'yield'); "
                f"got {self.metrics!r}"
            )
        if len(set(self.metrics)) != len(self.metrics):
            raise EvaluationContractError(f"Role: metrics {list(self.metrics)} name one twice")
        if not isinstance(self.artifacts, (list, tuple)) or not all(
            isinstance(name, str) and ARTIFACT_NAME.fullmatch(name) for name in self.artifacts
        ):
            raise EvaluationContractError(
                f"Role: artifacts must be artifact names, such as 'histogram'; got "
                f"{self.artifacts!r}"
            )
        if len(set(self.artifacts)) != len(self.artifacts):
            raise EvaluationContractError(f"Role: artifacts {list(self.artifacts)} name one twice")
        if self.stage is not None:
            if not isinstance(self.stage, str) or not self.stage.strip():
                raise EvaluationContractError(
                    f"Role: stage must name a pipeline stage, got {self.stage!r}"
                )
            if self.metrics or self.artifacts:
                raise EvaluationContractError(
                    "Role: a role that reads a stage's file reads no metrics or artifacts"
                )
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


type MetricKey = str | tuple[str, str]


def _qualified(evaluator: object, key: MetricKey) -> tuple[str, str]:
    """``(stage, metric)`` of a metric written as ``metric`` (stage: ``stage``) or a pair."""
    if isinstance(key, tuple) and len(key) == 2 and all(isinstance(part, str) for part in key):
        return key
    if not isinstance(key, str):
        raise EvaluationContractError(
            f"metric {key!r}: write it as 'metric' or as a pair of strings (stage, metric)"
        )
    stage = getattr(evaluator, "stage", None)
    if stage is None:
        raise EvaluationContractError(
            f"metric {key!r}: write it as (stage, metric) or set stage = <stage>"
        )
    return stage, key


def _role_name(role: object) -> bool:
    return (
        isinstance(role, str)
        and role.isidentifier()
        and not keyword.iskeyword(role)
        and not role.startswith("_")
    )


def declared_roles(evaluator: object) -> dict[str, Role] | None:
    """The ``roles`` ``evaluator`` declares, by name; ``None`` if it declares none (it is no
    combined evaluator). Raises :class:`EvaluationContractError` for a declaration that cannot be
    read, or roles of an evaluator that reads a stage."""
    declared = getattr(evaluator, "roles", None)
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise EvaluationContractError(
            "roles: declare a mapping of name to Role(...), such as "
            f"{{'signal': Role(metrics=[('analysis', 'yield')])}}; got {declared!r}"
        )
    if getattr(evaluator, "stage", None) is not None:
        raise EvaluationContractError(
            "roles: an evaluator that reads a stage measures each sample on its own; a combined "
            "evaluator reads other evaluations instead, so set no stage"
        )
    for role, declaration in declared.items():
        if not _role_name(role):
            raise EvaluationContractError(
                f"roles: {role!r} must be a Python identifier such as signal, not a keyword"
            )
        if not isinstance(declaration, Role):
            raise EvaluationContractError(
                f"roles.{role}: declare it as Role(...), got {declaration!r}"
            )
    return dict(declared)


def declared_collections(evaluator: object) -> dict[str, Collection]:
    """The ``collections`` ``evaluator`` declares, by role; raises
    :class:`EvaluationContractError` for a declaration that cannot be read."""
    declared = getattr(evaluator, "collections", None)
    if declared is None:
        return {}
    if not isinstance(declared, Mapping):
        raise EvaluationContractError(
            "collections: declare a mapping of role to Collection(type), such as "
            f"{{'hits': Collection('SimCalorimeterHit')}}; got {declared!r}"
        )
    if declared and getattr(evaluator, "stage", None) is None:
        raise EvaluationContractError(
            "collections are read from the file of the evaluator's stage; set stage = <stage>"
        )
    for role, collection in declared.items():
        if not _role_name(role):
            raise EvaluationContractError(
                f"collections: role {role!r} must be a Python identifier such as hits, "
                "not a keyword"
            )
        if not isinstance(collection, Collection):
            raise EvaluationContractError(
                f"collections.{role}: declare it as Collection(type), got {collection!r}"
            )
    return dict(declared)


def collection_names(evaluator: object, given: Mapping[str, Json]) -> dict[str, str]:
    """The collection each role of ``evaluator`` reads: the one the problem names in ``given``,
    or the declared default. Raises :class:`EvaluationContractError` for an unknown or missing
    role or a name that is not text."""
    declared = declared_collections(evaluator)
    unknown = [role for role in given if role not in declared]
    missing = [r for r, c in declared.items() if c.default is None and r not in given]
    if unknown or missing:
        raise EvaluationContractError(
            f"collections: expected the roles {sorted(declared)}"
            + (f", unknown {unknown}" if unknown else "")
            + (f", missing {missing}" if missing else "")
        )
    names: dict[str, str] = {}
    for role, collection in declared.items():
        name = given.get(role, collection.default)
        if not isinstance(name, str) or not name.strip():
            raise EvaluationContractError(
                f"collections.{role}: must be the name of a collection, got {name!r}"
            )
        names[role] = name
    return names


def metric_units(evaluator: object) -> dict[tuple[str, str], str] | None:
    """The declared ``{(stage, metric): unit}`` of ``evaluator``, ``None`` if it declares none."""
    declared = getattr(evaluator, "metrics", None)
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise EvaluationContractError(
            "metrics: declare a mapping of metric to its unit, such as "
            f"{{'ecal.cells': '1', ('analysis', 'rms'): 'mm'}}; got {declared!r}"
        )
    units: dict[tuple[str, str], str] = {}
    for key, unit in declared.items():
        qualified = _qualified(evaluator, key)
        if not isinstance(unit, str) or not unit.strip():
            raise EvaluationContractError(
                f"metric {qualified[0]}/{qualified[1]}: unit must be a nonblank string such as "
                f"'mm' or '1' (dimensionless), got {unit!r}"
            )
        if qualified in units:
            raise EvaluationContractError(
                f"metric {qualified[0]}/{qualified[1]} is declared twice (as {key!r} too)"
            )
        units[qualified] = unit
    return units


def declared_artifacts(evaluator: object) -> dict[str, str] | None:
    """The ``artifacts`` ``evaluator`` declares, each name with what the file holds; ``None`` if
    it declares none. Raises :class:`EvaluationContractError` for a declaration that cannot be
    read."""
    declared = getattr(evaluator, "artifacts", None)
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise EvaluationContractError(
            "artifacts: declare a mapping of name to what the file holds, such as "
            f"{{'histogram': 'shower widths, JSON {{edges, counts}}'}}; got {declared!r}"
        )
    for name, described in declared.items():
        if not isinstance(name, str) or not ARTIFACT_NAME.fullmatch(name):
            raise EvaluationContractError(
                f"artifacts: name {name!r} must start with a letter or digit and hold only "
                "letters, digits, '.', '_' and '-'"
            )
        if not isinstance(described, str) or not described.strip():
            raise EvaluationContractError(
                f"artifacts.{name}: say what the file holds, got {described!r}"
            )
    return dict(declared)


def check_class(cls: object) -> type:
    """``cls`` if it is an evaluator class whose declarations can be read without running its
    code; :class:`EvaluationContractError` otherwise.

    It must be a class with ``name``, ``version`` and an ``evaluate`` method of its own (not only
    the abstract one of :class:`~k4opt.evaluation.Evaluator`). Each of
    :data:`DECLARED` it has must be plain data on the class (a string, a mapping, ``None``), not
    a property, a method or another descriptor, whose value only code could give. Its ``check``,
    if it has one, must be a classmethod (or a staticmethod): k4opt calls it with each candidate's
    request without creating an instance, so that refusing a candidate costs no initialization.
    """
    if not isinstance(cls, type):
        # a function that builds an instance could configure it in ways the class's source,
        # which the records identify as the code that ran, does not say
        raise EvaluationContractError(
            f"got a {type(cls).__name__} object, not an evaluator class: give the class, whose "
            "source is recorded and of which k4opt creates an instance per evaluation"
        )
    label = cls.__qualname__
    missing = [
        member
        for member in ("name", "version", "evaluate")
        if inspect.getattr_static(cls, member, None) is None
    ]
    evaluate = inspect.getattr_static(cls, "evaluate", None)
    if missing or not callable(evaluate):
        raise EvaluationContractError(
            f"{label} is not an evaluator: needs name, version and an evaluate() method"
            + (f" (missing {', '.join(missing)})" if missing else "")
        )
    if getattr(evaluate, "__isabstractmethod__", False):
        raise EvaluationContractError(
            f"{label} does not implement evaluate(): it inherits the abstract one of its base"
        )
    for attribute in DECLARED:
        value = inspect.getattr_static(cls, attribute, None)
        if hasattr(type(value), "__get__"):
            raise EvaluationContractError(
                f"{label}.{attribute} is a {type(value).__name__}: declare it as plain data on the "
                "class, which k4opt reads without creating an instance or running its code"
            )
    check = inspect.getattr_static(cls, "check", None)
    if check is not None and not isinstance(check, (classmethod, staticmethod)):
        raise EvaluationContractError(
            f"{label}.check must be a classmethod: k4opt calls it with each candidate's request "
            "without creating the evaluator, before anything of the candidate runs. Write "
            "@classmethod def check(cls, request), and keep it to inexpensive checks of the "
            "request (no files, geometry or physics: those belong in evaluate(), which runs "
            "isolated and time-limited)"
        )
    return cls


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
    def of(cls, implementation: object) -> Declaration:
        """What the evaluator class ``implementation`` declares, read without running its code
        (:func:`check_class`); raises :class:`EvaluationContractError` for a class that is no
        evaluator or a declaration that cannot be read."""
        evaluator: Any = check_class(implementation)
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
        if not _role_name(name):  # as declared_roles() refuses it of a live evaluator
            raise error(
                f"{where}: roles: {name!r} must be a Python identifier such as signal, not a "
                "keyword"
            )
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
