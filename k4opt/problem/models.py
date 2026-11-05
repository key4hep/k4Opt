"""Immutable domain models of a parsed optimization problem.

The models describe the draft ``k4opt.problem/v1alpha1`` format of ``docs/problem.md``.
:class:`ProblemSpec` and its parts hold declarative data only: nothing here looks up an
evaluator or metric or touches the Key4hep stack. A problem loaded from its file, with its
geometry discovered and every geometry destination resolved, is a :class:`ResolvedProblem`
(:func:`k4opt.problem.config.resolve_problem`). Executor- and evaluator-owned content
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
class GeometryDestination:
    """The attribute of the geometry a parameter sets, by ``reference``: a constant by its name
    (``ECal_cell_size``), any other attribute by its canonical target reference of
    :mod:`k4opt.geometry` (``InnerTrackerBarrel.layer[id=1].rphi_layout.nphi``). The reference is
    not looked up in a geometry here."""

    reference: str

    @property
    def canonical_target(self) -> str:
        """The target reference it stands for: ``constants.<name>`` for a constant's name."""
        return self.reference if "." in self.reference else f"constants.{self.reference}"

    @property
    def label(self) -> str:
        """``geometry constants.ECal_cell_size``: the destination, for a person."""
        return f"geometry {self.canonical_target}"


@dataclass(frozen=True, slots=True)
class StageDestination:
    """The option ``option`` of the stage ``stage`` of each of ``samples`` that a parameter sets:
    a key of the particle gun (``energy``) or a ddsim option (``physics.rangecut``), run by an
    executor that lets a parameter set it (:attr:`~k4opt.problem.stages.Executor.settable`)."""

    stage: str
    option: str
    samples: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))

    @property
    def label(self) -> str:
        """``simulation physics.rangecut of gamma, pion``: the destination, for a person."""
        return f"{self.stage} {self.option} of {', '.join(self.samples)}"


@dataclass(frozen=True, slots=True)
class SettingDestination:
    """The setting ``setting`` that a parameter sets in every entry of ``evaluators`` bound as
    ``evaluator`` (:attr:`EvaluatorSpec.binding`); the implementation must declare it."""

    evaluator: str
    setting: str

    @property
    def label(self) -> str:
        """``evaluator readout: cell_size``: the destination, for a person."""
        return f"evaluator {self.evaluator}: {self.setting}"


type Destination = GeometryDestination | StageDestination | SettingDestination
"""Where a parameter's value goes: an attribute of the geometry, an option of a stage of some
samples or a setting of an evaluator binding."""


@dataclass(frozen=True, slots=True)
class Parameter:
    """A quantity that varies between candidates: its search domain, and where its value goes.

    ``name`` identifies it in the problem: candidates, records and optimizers key its value by
    it. ``continuous`` and ``integer`` parameters have inclusive ``bounds``; ``categorical``
    ones have ``choices``. ``unit`` is the unit of the numbers in ``bounds``. ``targets`` holds
    every destination its value is written into, at least one, grouped by kind in the order the
    problem gives them: attributes of the geometry (``<value>*<unit>``), options of a stage,
    settings of evaluators (``{value, unit}`` with a unit). The domain says nothing about a
    destination, and the destinations nothing about the domain. ``title`` is what a reader is
    shown instead of the name (``ECAL cell size``), if given; nothing else reads it.
    """

    name: str
    kind: ParameterKind
    targets: tuple[Destination, ...]
    bounds: tuple[Number, Number] | None = None
    choices: tuple[str | bool | Number, ...] | None = None
    unit: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))

    def to_dict(self) -> dict[str, object]:
        """Everything but the name, which keys it in the problem's ``parameters``; ``targets`` in
        one form of each target: a geometry reference, ``{option, samples}`` for a stage option
        (its samples listed), the setting of an evaluator binding; a list where a kind has
        several."""
        data = _titled(self.title) | {"type": self.kind}
        if self.bounds is not None:
            data["bounds"] = list(self.bounds)
        if self.choices is not None:
            data["choices"] = list(self.choices)
        if self.unit is not None:
            data["unit"] = self.unit
        grouped: dict[str, list[object]] = {}
        settings: dict[str, list[object]] = {}
        for destination in self.targets:
            if isinstance(destination, GeometryDestination):
                grouped.setdefault("geometry", []).append(destination.reference)
            elif isinstance(destination, StageDestination):
                option = {"option": destination.option, "samples": list(destination.samples)}
                grouped.setdefault(destination.stage, []).append(option)
            else:
                grouped.setdefault("evaluators", [])
                settings.setdefault(destination.evaluator, []).append(destination.setting)
        data["targets"] = {
            kind: {binding: _one(items) for binding, items in settings.items()}
            if kind == "evaluators"
            else _one(items)
            for kind, items in grouped.items()
        }
        return data


def _one(items: list[object]) -> object:
    """A list of targets of one kind as a problem writes it: the target alone if it is one."""
    return items[0] if len(items) == 1 else items


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
    A combined evaluator reads no stage either, but ``roles``: the items each of its roles reads,
    by role, in the order the problem gives them. ``required`` makes the candidate's status
    depend on each of its evaluations, as on one an objective or a constraint names; an
    evaluation that none of them needs is a diagnostic, recorded but not deciding the status.
    ``timeout`` is how many seconds each of its evaluations may take, its instance's creation
    included (``None``: the problem's ``timeout``, if any): one that takes longer is stopped, and
    fails.
    """

    name: str
    version: str
    settings: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    stage: str | None = None
    collections: Mapping[str, str] = field(default_factory=lambda: FrozenMap({}))
    binding: str = ""
    samples: tuple[str, ...] = ()
    roles: Mapping[str, tuple[Reference, ...]] = field(default_factory=lambda: FrozenMap({}))
    required: bool = False
    timeout: Number | None = None

    def __post_init__(self) -> None:
        # Direct construction gets the same defensive copy as parsing does.
        settings = freeze(self.settings, "settings")
        assert isinstance(settings, Mapping)
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "collections", FrozenMap(self.collections))
        object.__setattr__(self, "samples", tuple(self.samples))
        roles = {role: tuple(items) for role, items in self.roles.items()}
        object.__setattr__(self, "roles", FrozenMap(roles))
        if not self.binding:
            object.__setattr__(self, "binding", self.name)

    @property
    def reads(self) -> tuple[Reference, ...]:
        """Every item its roles read, each once, in the order the problem gives them."""
        return tuple(dict.fromkeys(item for items in self.roles.values() for item in items))

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
        if self.roles:
            data["roles"] = {
                role: [item.to_dict() for item in items] for role, items in self.roles.items()
            }
        if self.settings:
            data["settings"] = thaw(self.settings)
        if self.required:
            data["required"] = True
        if self.timeout is not None:
            data["timeout"] = self.timeout
        return data


@dataclass(frozen=True, slots=True)
class Reference:
    """An item a role of a combined evaluator reads: the evaluation of ``sample`` by the binding
    ``evaluator``, or the ``stage`` of ``sample`` (its file); exactly one of them is set."""

    sample: str
    evaluator: str | None = None
    stage: str | None = None

    def __post_init__(self) -> None:
        if (self.evaluator is None) == (self.stage is None):
            raise ProblemConfigError(
                f"a reference to {self.sample} names an evaluator or a stage, exactly one of them"
            )

    @property
    def label(self) -> str:
        """``sample/evaluator`` or ``sample/stage``: the task it reads, as records name it."""
        return f"{self.sample}/{self.evaluator or self.stage}"

    def to_dict(self) -> dict[str, object]:
        if self.evaluator is not None:
            return {"sample": self.sample, "evaluator": self.evaluator}
        return {"sample": self.sample, "stage": self.stage}


@dataclass(frozen=True, slots=True)
class Stage:
    """One stage of a sample: a pipeline stage, run by an ``executor`` or given as a ``file``.

    ``type`` says what the stage does (``generation``, ``simulation``); the entry is named after
    it. A stage that runs has its executor's ``options``: the particle gun's keys for ``gun``,
    ddsim options for ``ddsim``, which also has ``events`` (how many to simulate) and ``seed``.
    A stage given as a ``file`` (a path; a relative one is read from the problem file's
    directory) is not run: the file is its output. Which executors and options are valid is
    checked by :func:`k4opt.problem.parse_problem`. ``timeout`` is how many seconds a stage that
    runs may take (``None``: the problem's ``timeout``, if any): one that takes longer is stopped,
    and fails.
    """

    type: str
    executor: str | None = None
    file: str | None = None
    events: int | None = None
    seed: int | None = None
    options: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    timeout: Number | None = None

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
        if self.timeout is not None:
            data["timeout"] = self.timeout
        options = thaw(self.options)
        assert isinstance(options, dict)
        return data | options


@dataclass(frozen=True, slots=True)
class Sample:
    """One entry of ``samples``: the events a candidate is measured on, named ``name``.

    ``stages`` says how each stage of the sample is made, by stage type in pipeline order. The
    name is a label that evaluators, objectives and constraints refer to the sample by; what its
    events are is said by its stages alone. ``properties`` are facts about the sample that its
    evaluators may read (a process, a cross section), any JSON: k4opt records and delivers them,
    and never reads them itself. ``title`` is what a reader is shown instead of the name
    (``10 GeV γ``), if given.
    """

    name: str
    stages: Mapping[str, Stage] = field(default_factory=lambda: FrozenMap({}))
    properties: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    title: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", FrozenMap(self.stages))
        properties = freeze(self.properties, f"samples.{self.name}.properties")
        assert isinstance(properties, Mapping)
        object.__setattr__(self, "properties", properties)

    def to_dict(self) -> dict[str, object]:
        data = _titled(self.title)
        if self.properties:
            data["properties"] = thaw(self.properties)
        return data | {name: stage.to_dict() for name, stage in self.stages.items()}


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


def _titled(title: str | None) -> dict[str, object]:
    return {} if title is None else {"title": title}


@dataclass(frozen=True, slots=True)
class Objective:
    """A metric of an evaluator to minimize or maximize, measured on ``sample`` (``None`` for an
    evaluator that measures the candidate). ``stage`` classifies the metric. ``title`` is what a
    reader is shown instead of the metric (``Angular residual``), if given: it names what is
    measured, not the sample it is measured on."""

    evaluator: str
    stage: str
    metric: str
    direction: Direction
    sample: str | None = None
    title: str | None = None

    @property
    def ref(self) -> MetricRef:
        return MetricRef(self.evaluator, self.stage, self.metric, self.sample)

    @property
    def label(self) -> str:
        return self.ref.label

    def to_dict(self) -> dict[str, object]:
        return (
            _owner(self.sample, self.evaluator)
            | _titled(self.title)
            | {
                "stage": self.stage,
                "metric": self.metric,
                "direction": self.direction,
            }
        )


@dataclass(frozen=True, slots=True)
class Constraint:
    """A hard, inclusive limit on a metric of an evaluator, measured on ``sample`` (``None`` for
    an evaluator that measures the candidate): ``min`` and/or ``max``.

    ``scale`` is a positive characteristic size of the metric, in the unit of its limits: how far
    beyond a limit counts as far. An optimizer is told how far a candidate is beyond each limit in
    units of it (:attr:`~k4opt.record.Limit.normalized`), so that limits on metrics of different
    units or magnitudes add up meaningfully. The problem requires one on every constraint once
    there are constraints on two or more metrics; ``None`` otherwise, which counts as 1.
    ``title`` is what a reader is shown instead of the metric, as of an :class:`Objective`.
    """

    evaluator: str
    stage: str
    metric: str
    min: Number | None = None
    max: Number | None = None
    unit: str | None = None
    sample: str | None = None
    scale: Number | None = None
    title: str | None = None

    @property
    def ref(self) -> MetricRef:
        return MetricRef(self.evaluator, self.stage, self.metric, self.sample)

    @property
    def label(self) -> str:
        return self.ref.label

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = (
            _owner(self.sample, self.evaluator)
            | _titled(self.title)
            | {
                "stage": self.stage,
                "metric": self.metric,
            }
        )
        if self.min is not None:
            data["min"] = self.min
        if self.max is not None:
            data["max"] = self.max
        if self.unit is not None:
            data["unit"] = self.unit
        if self.scale is not None:
            data["scale"] = self.scale
        return data


@dataclass(frozen=True, slots=True)
class Criteria:
    """How a candidate is judged at one level of an optimization: the ``objectives`` it is
    scored by and the ``constraints`` its feasibility is judged by (:meth:`ProblemSpec.criteria`).
    """

    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...] = ()

    @property
    def references(self) -> tuple[MetricRef, ...]:
        """The metrics the objectives and constraints name, each once, objectives first."""
        refs = [o.ref for o in self.objectives] + [c.ref for c in self.constraints]
        return tuple(dict.fromkeys(refs))


@dataclass(frozen=True, slots=True)
class Level:
    """One level of a nested optimization (the problem's ``optimization`` block): the
    ``parameters`` it searches, and, for each candidate of the level above, a study of them of
    ``trials`` trials whose best trial by ``objective`` stands for that candidate. The top level
    has neither ``trials`` (its budget is the run's) nor ``objective`` (the problem's objectives
    judge it); ``inner`` is the level below, if any."""

    parameters: tuple[str, ...]
    trials: int | None = None
    objective: Objective | None = None
    inner: Level | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"parameters": list(self.parameters)}
        if self.trials is not None:
            data["trials"] = self.trials
        if self.objective is not None:
            data["objective"] = self.objective.to_dict()
        if self.inner is not None:
            data["inner"] = self.inner.to_dict()
        return data


def fixed_events(samples: Mapping[str, Sample]) -> str | None:
    """Why the candidates of a problem with ``samples`` cannot be evaluated on other events
    (:func:`k4opt.problem.candidate.check_replicate`), ``None`` if they can: a sample reads a
    stage from a file (another seed would see the same collisions again), or no stage has a seed
    (nothing would differ)."""
    for name, sample in samples.items():
        given = [stage for stage, entry in sample.stages.items() if entry.file is not None]
        if given:
            return (
                f"samples.{name} reads its {', '.join(given)} from a file, so another seed would "
                "not make new events; only samples that make their events from a seed (the "
                "particle gun, fired by ddsim) have them"
            )
    if not any(stage.seed is not None for s in samples.values() for stage in s.stages.values()):
        return "the problem makes no events from a seed (no sample simulates any)"
    return None


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """A validated problem definition (``k4opt.problem/v1alpha1``).

    Built by :func:`k4opt.problem.load_problem`, which also checks that its geometry targets
    resolve, or by :func:`k4opt.problem.parse_problem`, which only validates the document. Nothing
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
    # seconds each stage that runs and each evaluation may take, unless it sets its own
    timeout: Number | None = None
    # the levels of a nested optimization, the top one first; None for an ordinary one
    optimization: Level | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluators", tuple(self.evaluators))
        object.__setattr__(self, "samples", FrozenMap(self.samples))

    @property
    def levels(self) -> tuple[Level, ...]:
        """The levels of the optimization, the top one first; none without ``optimization``."""
        found, level = [], self.optimization
        while level is not None:
            found.append(level)
            level = level.inner
        return tuple(found)

    @property
    def depth(self) -> int:
        """How many levels lie below the top one: 0 for an ordinary optimization."""
        return max(len(self.levels) - 1, 0)

    def parameters_of(self, level: int = 0) -> tuple[str, ...]:
        """The parameters a study at ``level`` searches: those of the level, or every parameter
        of a problem without levels."""
        if not self.levels:
            return tuple(parameter.name for parameter in self.parameters)
        return self.levels[level].parameters

    def check_level(self, level: object) -> int:
        """``level`` as a level of the problem: an integer from 0 (the top) to :attr:`depth`.
        Raises :class:`ValueError` for anything else (``True`` is no level)."""
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= self.depth:
            raise ValueError(f"level {level!r}: the problem's levels are 0 to {self.depth}")
        return level

    def criteria(self, level: int = 0) -> Criteria:
        """How a candidate is judged at ``level``: by the problem's objectives at the top
        (``0``), by the level's ``objective`` below; by the problem's constraints at every level.
        Raises :class:`ValueError` for a level the problem does not have (:meth:`check_level`)."""
        if self.check_level(level) == 0:
            return Criteria(self.objectives, self.constraints)
        objective = self.levels[level].objective
        assert objective is not None  # parse_problem: every inner level has one
        return Criteria((objective,), self.constraints)

    @property
    def references(self) -> tuple[MetricRef, ...]:
        """The metrics the objectives and constraints name, each once, objectives first: those
        the top level judges by (:meth:`criteria`)."""
        return self.criteria(0).references

    @property
    def measured(self) -> tuple[MetricRef, ...]:
        """The metrics any level judges by, each once: every one an evaluation must carry when
        it is valid, whatever level it was made for."""
        refs = [ref for level in range(self.depth + 1) for ref in self.criteria(level).references]
        return tuple(dict.fromkeys(refs))

    @property
    def destinations(self) -> tuple[tuple[Parameter, Destination], ...]:
        """Every destination of every parameter, with its parameter, in problem order."""
        return tuple((p, d) for p in self.parameters for d in p.targets)

    @property
    def geometry_destinations(self) -> tuple[tuple[Parameter, GeometryDestination], ...]:
        """The destinations in the geometry, with their parameters, in problem order: one
        attribute each, and one override each per candidate."""
        return tuple((p, d) for p, d in self.destinations if isinstance(d, GeometryDestination))

    def pending(self, sample: str) -> dict[str, frozenset[str]]:
        """The options of each stage of ``sample`` that parameters set, by stage: the stage does
        not give them, and every candidate does."""
        return pending(self.parameters, sample)

    def time_limit(self, own: Number | None) -> float | None:
        """How many seconds a stage or an evaluation whose own ``timeout`` is ``own`` may take:
        its own, or the problem's; ``None`` for no limit."""
        limit = self.timeout if own is None else own
        return None if limit is None else float(limit)

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-compatible data in schema order; a fresh copy on every call.

        Every sample holds each of its stages in full, also where the problem file shares them in
        ``stages``; every evaluator that reads a stage lists its samples, and every objective and
        constraint names its evaluator and the sample it is measured on, also where the problem
        file leaves them out. ``parse_problem(spec.to_dict())`` gives back an equal
        :class:`ProblemSpec`.
        """
        data: dict[str, object] = {"schema_version": SCHEMA_VERSION}
        if self.name is not None:
            data["name"] = self.name
        data["environment"] = self.environment.to_dict()
        if self.timeout is not None:
            data["timeout"] = self.timeout
        data["geometry"] = self.geometry
        data["parameters"] = {parameter.name: parameter.to_dict() for parameter in self.parameters}
        if self.samples:
            data["samples"] = {name: sample.to_dict() for name, sample in self.samples.items()}
        data["evaluators"] = [evaluator.to_dict() for evaluator in self.evaluators]
        data["objectives"] = [objective.to_dict() for objective in self.objectives]
        if self.constraints:
            data["constraints"] = [constraint.to_dict() for constraint in self.constraints]
        if self.optimization is not None:
            data["optimization"] = self.optimization.to_dict()
        return data


# the sections of a problem that map names in an order it defines: candidates, samplers,
# records and reports follow it
_ORDERED = ("parameters", "samples")


def ordered(definition: Mapping[str, object]) -> dict[str, object]:
    """``definition`` (:meth:`ProblemSpec.to_dict` data) with each section whose order the problem
    defines (``parameters``, ``samples``) as a list of ``[name, value]`` pairs, so that comparing
    two, or a digest of one, sees that order too."""
    return {
        key: [[name, item] for name, item in value.items()]
        if key in _ORDERED and isinstance(value, Mapping)
        else value
        for key, value in definition.items()
    }


_TITLED = ("parameters", "samples", "objectives", "constraints")


def _untitled(item: object) -> object:
    if isinstance(item, Mapping):
        return {key: value for key, value in item.items() if key != "title"}
    return item


def _untitled_level(level: object) -> object:
    """A level of ``optimization`` without the ``title`` of its objective, and so each below."""
    if not isinstance(level, Mapping):
        return level
    found = dict(level)
    if "objective" in found:
        found["objective"] = _untitled(found["objective"])
    if "inner" in found:
        found["inner"] = _untitled_level(found["inner"])
    return found


def essential(definition: Mapping[str, object]) -> dict[str, object]:
    """``definition`` (:meth:`ProblemSpec.to_dict` data) as two are compared: :func:`ordered`,
    without what only names things for a reader, its ``name`` and every ``title``. Two problems
    that differ only in those are one problem: a study goes on, and keeps its records."""
    found = ordered(definition)
    found.pop("name", None)
    if "optimization" in found:
        found["optimization"] = _untitled_level(found["optimization"])
    for key in _TITLED:
        value = found.get(key)
        if not isinstance(value, list):
            continue
        found[key] = [
            [entry[0], _untitled(entry[1])] if key in _ORDERED else _untitled(entry)
            for entry in value
        ]
    return found


def pending(parameters: tuple[Parameter, ...], sample: str) -> dict[str, frozenset[str]]:
    """The options of each stage of ``sample`` that ``parameters`` set, by stage."""
    found: dict[str, set[str]] = {}
    for parameter in parameters:
        for destination in parameter.targets:
            if isinstance(destination, StageDestination) and sample in destination.samples:
                found.setdefault(destination.stage, set()).add(destination.option)
    return {stage: frozenset(options) for stage, options in found.items()}


@dataclass(frozen=True, slots=True)
class ResolvedProblem:
    """A problem as loaded from its file: its definition (``spec``), the file (``path``,
    absolute), the geometry it starts from, discovered once (``source``, ``catalog``), and the
    attribute each geometry destination sets, in the order of
    :attr:`ProblemSpec.geometry_destinations` (``targets``)."""

    spec: ProblemSpec
    path: Path
    source: GeometrySource
    catalog: Catalog
    targets: tuple[GeometryParameter, ...]

    @property
    def directory(self) -> Path:
        """The directory of the problem file, against which a relative stage ``file`` is
        read."""
        return self.path.parent
