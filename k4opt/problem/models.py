"""Immutable domain models of a parsed optimization problem.

The models describe the draft ``k4opt.problem/v1alpha1`` format of ``docs/problem.md``.
They hold declarative data only: nothing here looks up an evaluator or metric or touches
the Key4hep stack; :func:`k4opt.problem.load_problem` attaches the resolved geometry targets
to the :class:`ProblemSpec` as private runtime state. Executor- and evaluator-owned content
(each stage's options, each evaluator's ``settings``) is stored as an immutable copy, so
neither the caller's YAML data nor the object handed out by :meth:`ProblemSpec.to_dict`
aliases the stored state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from k4opt.frozen import FrozenMap, Json, thaw
from k4opt.frozen import freeze as _freeze_json
from k4opt.geometry import Catalog, GeometrySource
from k4opt.geometry import Parameter as GeometryParameter
from k4opt.problem.errors import ProblemConfigError

SCHEMA_VERSION = "k4opt.problem/v1alpha1"

type Number = int | float

Channel = Literal["release", "nightly"]
ParameterKind = Literal["continuous", "integer", "categorical"]
Direction = Literal["minimize", "maximize"]


def freeze(value: object, path: str) -> Json:
    """Return an immutable copy of a JSON-compatible ``value``.

    The shared :func:`k4opt.frozen.freeze`, reporting a :class:`ProblemConfigError` that
    names ``path``.
    """
    return _freeze_json(value, path, ProblemConfigError)


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
    ones have ``choices``. ``unit`` is the unit of the numbers in ``bounds``; it is never
    converted (``k4opt evaluate`` writes a value into the geometry as ``<value>*<unit>``).
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
class EvaluatorSpec:
    """One evaluator of a problem: a binding of an implementation, with its settings.

    ``name`` and ``version`` must equal those of the implementation that measures, and so must
    ``stage``, the pipeline stage it reads (``None``: it reads none). ``binding`` names this use
    of the implementation (``as:`` in the problem, its ``name`` when not given): objectives,
    constraints, records and run directories refer to it, so one implementation can be bound
    several times with other settings. An evaluator that reads a stage measures each of its
    ``samples`` on its own, from the stage of that sample; one that reads none measures the
    candidate once, and has no samples. ``collections`` names the collection of that stage's file
    for each role the evaluator declares; ``settings`` is owned by that evaluator's contract.
    """

    name: str
    version: str
    settings: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    stage: str | None = None
    collections: Mapping[str, str] = field(default_factory=lambda: FrozenMap({}))
    binding: str = ""
    samples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Direct construction gets the same defensive copy as parsing does.
        settings = freeze(self.settings, "settings")
        assert isinstance(settings, Mapping)
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "collections", FrozenMap(self.collections))
        object.__setattr__(self, "samples", tuple(self.samples))
        if not self.binding:
            object.__setattr__(self, "binding", self.name)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name}
        if self.binding != self.name:
            data["as"] = self.binding
        data["version"] = self.version
        if self.stage is not None:
            data["stage"] = self.stage
        if self.samples:
            data["samples"] = list(self.samples)
        if self.collections:
            data["collections"] = dict(self.collections)
        if self.settings:
            data["settings"] = thaw(self.settings)
        return data


@dataclass(frozen=True, slots=True)
class Stage:
    """One stage of a sample: a pipeline stage, run by an ``executor`` or given as a ``file``.

    ``type`` says what the stage does (``generation``, ``simulation``); the entry is named after
    it. A stage that runs has its executor's ``options``: the particle gun's keys for ``gun``,
    ddsim options for ``ddsim``, which also has ``events`` (how many to simulate) and ``seed``.
    A stage given as a ``file`` (a path; a relative one is read from the problem file's
    directory) is not run: the file is its output. Which executors and options are valid is
    checked by :func:`k4opt.problem.parse_problem`.
    """

    type: str
    executor: str | None = None
    file: str | None = None
    events: int | None = None
    seed: int | None = None
    options: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))

    def __post_init__(self) -> None:
        if (self.executor is None) == (self.file is None):
            raise ProblemConfigError(f"{self.type}: needs exactly one of 'executor' and 'file'")
        # Direct construction gets the same defensive copy as parsing does.
        options = freeze(self.options, self.type)
        assert isinstance(options, Mapping)
        object.__setattr__(self, "options", options)

    def to_dict(self) -> dict[str, object]:
        if self.file is not None:
            return {"file": self.file}
        data: dict[str, object] = {"executor": self.executor}
        if self.events is not None:
            data["events"] = self.events
        if self.seed is not None:
            data["seed"] = self.seed
        options = thaw(self.options)
        assert isinstance(options, dict)
        return data | options


@dataclass(frozen=True, slots=True)
class Sample:
    """One entry of ``samples``: the events a candidate is measured on, named ``name``.

    ``stages`` says how each stage of the sample is made, by stage type in pipeline order. The
    name is a label that evaluators, objectives and constraints refer to the sample by; what its
    events are is said by its stages alone.
    """

    name: str
    stages: Mapping[str, Stage] = field(default_factory=lambda: FrozenMap({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", FrozenMap(self.stages))

    def to_dict(self) -> dict[str, object]:
        return {name: stage.to_dict() for name, stage in self.stages.items()}


def _escaped(part: str) -> str:
    """A part of a label, with ``%`` and ``/`` percent-escaped."""
    return part.replace("%", "%25").replace("/", "%2F")


@dataclass(frozen=True, slots=True)
class MetricRef:
    """A metric of one evaluation: the identity of every value an objective or a constraint names.

    ``evaluator`` is the binding that measures it (:attr:`EvaluatorSpec.binding`) and ``sample``
    the sample it measures it on, ``None`` for a binding that measures the candidate; ``stage``
    and ``metric`` name the measurement as the evaluator reports it. Two bindings, or one binding
    on two samples, may report the same metric, and their references differ.
    """

    evaluator: str
    stage: str
    metric: str
    sample: str | None = None

    @property
    def invocation(self) -> str:
        """``sample/evaluator``, or ``evaluator`` without a sample: the evaluation that measures
        it, as records name it."""
        return self.evaluator if self.sample is None else f"{self.sample}/{self.evaluator}"

    @property
    def label(self) -> str:
        """``sample/evaluator/stage/metric``, such as
        ``gamma/calo.photon/simulation/ecal_barrel.fired_cells``, or ``evaluator/stage/metric``
        without a sample; records, exports and the optimizer name the value by it.

        A stage or metric name may hold ``/``, so each part has its ``%`` and ``/``
        percent-escaped (``analysis%2Fx/y``; ``urllib.parse.unquote`` reverses it): two
        references never share a label, and its last two parts are always stage and metric.
        """
        parts = [self.evaluator, self.stage, self.metric]
        if self.sample is not None:
            parts.insert(0, self.sample)
        return "/".join(_escaped(part) for part in parts)


def _owner(sample: str | None, evaluator: str) -> dict[str, object]:
    """Where an objective or constraint takes its metric from, as the problem writes it."""
    owner: dict[str, object] = {} if sample is None else {"sample": sample}
    return owner | {"evaluator": evaluator}


@dataclass(frozen=True, slots=True)
class Objective:
    """A metric of an evaluator to minimize or maximize, measured on ``sample`` (``None`` for an
    evaluator that measures the candidate). ``stage`` classifies the metric."""

    evaluator: str
    stage: str
    metric: str
    direction: Direction
    sample: str | None = None

    @property
    def ref(self) -> MetricRef:
        return MetricRef(self.evaluator, self.stage, self.metric, self.sample)

    @property
    def label(self) -> str:
        return self.ref.label

    def to_dict(self) -> dict[str, object]:
        return _owner(self.sample, self.evaluator) | {
            "stage": self.stage,
            "metric": self.metric,
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class Constraint:
    """A hard, inclusive limit on a metric of an evaluator, measured on ``sample`` (``None`` for
    an evaluator that measures the candidate): ``min`` and/or ``max``."""

    evaluator: str
    stage: str
    metric: str
    min: Number | None = None
    max: Number | None = None
    unit: str | None = None
    sample: str | None = None

    @property
    def ref(self) -> MetricRef:
        return MetricRef(self.evaluator, self.stage, self.metric, self.sample)

    @property
    def label(self) -> str:
        return self.ref.label

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = _owner(self.sample, self.evaluator) | {
            "stage": self.stage,
            "metric": self.metric,
        }
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

    Built by :func:`k4opt.problem.load_problem`, which also resolves the geometry targets,
    or by :func:`k4opt.problem.parse_problem`, which only validates the document. Nothing
    here claims that an evaluator or metric is defined, or that the ranges make scientific
    sense.
    """

    environment: Environment
    # the starting detector description: a compact file, relative to the k4geo tree
    geometry: str
    parameters: tuple[Parameter, ...]
    evaluators: tuple[EvaluatorSpec, ...]
    objectives: tuple[Objective, ...]
    samples: Mapping[str, Sample] = field(default_factory=lambda: FrozenMap({}))
    constraints: tuple[Constraint, ...] = ()
    name: str | None = None
    # Runtime state of a loaded problem (see load_problem): the geometry discovered once and
    # the attribute each parameter targets, in the order of ``parameters``. Private, left out
    # of equality, hashing, repr and to_dict(); empty for a problem from parse_problem.
    _source: GeometrySource | None = field(default=None, kw_only=True, repr=False, compare=False)
    _catalog: Catalog | None = field(default=None, kw_only=True, repr=False, compare=False)
    _targets: tuple[GeometryParameter, ...] = field(
        default=(), kw_only=True, repr=False, compare=False
    )
    # the directory of the problem file, against which a relative stage ``file`` is read
    _directory: Path | None = field(default=None, kw_only=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluators", tuple(self.evaluators))
        object.__setattr__(self, "samples", FrozenMap(self.samples))

    @property
    def references(self) -> tuple[MetricRef, ...]:
        """The metrics the objectives and constraints name, each once, objectives first."""
        refs = [o.ref for o in self.objectives] + [c.ref for c in self.constraints]
        return tuple(dict.fromkeys(refs))

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-compatible data in schema order; a fresh copy on every call.

        Every evaluator that reads a stage lists its samples, and every objective and constraint
        names its evaluator and the sample it is measured on, also where the problem file leaves
        them out. ``parse_problem(spec.to_dict())`` gives back an equal :class:`ProblemSpec`.
        """
        data: dict[str, object] = {"schema_version": SCHEMA_VERSION}
        if self.name is not None:
            data["name"] = self.name
        data["environment"] = self.environment.to_dict()
        data["geometry"] = self.geometry
        data["parameters"] = [parameter.to_dict() for parameter in self.parameters]
        if self.samples:
            data["samples"] = {name: sample.to_dict() for name, sample in self.samples.items()}
        data["evaluators"] = [evaluator.to_dict() for evaluator in self.evaluators]
        data["objectives"] = [objective.to_dict() for objective in self.objectives]
        if self.constraints:
            data["constraints"] = [constraint.to_dict() for constraint in self.constraints]
        return data
