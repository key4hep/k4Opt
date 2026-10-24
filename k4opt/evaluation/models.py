"""Immutable request and result models of the evaluation contract.

An evaluator receives one :class:`EvaluationRequest` and returns one
:class:`EvaluationResult` made of :class:`Measurement` objects and named artifacts. The
models check what the contract promises (finite numbers, nonnegative uncertainty,
nonblank names, unique metrics) and nothing about scientific meaning, files on disk or
the search domain. Nested caller-owned data is copied on construction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from k4opt.evaluation.errors import EvaluationContractError
from k4opt.frozen import FrozenMap, Json, freeze

type Scalar = bool | int | float | str
"""A parameter value of a candidate."""

type Status = Literal["valid", "invalid", "failed"]

_STATUSES = ("valid", "invalid", "failed")


def _name(value: object, what: str) -> str:
    """Return ``value`` unchanged if it is a nonblank string, else raise."""
    if not isinstance(value, str):
        raise EvaluationContractError(f"{what} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise EvaluationContractError(f"{what} must not be blank")
    return value


def _real(value: object, what: str) -> int | float:
    """Return ``value`` if it is a finite real number; booleans are not numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationContractError(f"{what} must be a number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationContractError(f"{what} must be finite, got {value!r}")
    return value


def _paths(value: Mapping[str, Path], what: str) -> FrozenMap[Path]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"{what} must be a mapping of names to paths")
    for name, path in value.items():
        _name(name, f"{what} name")
        if not isinstance(path, Path):
            raise EvaluationContractError(
                f"{what}[{name!r}] must be a pathlib.Path, got {type(path).__name__}"
            )
    return FrozenMap(value)


def _json_mapping(value: Mapping[str, Json], what: str) -> FrozenMap[Json]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"{what} must be a mapping")
    frozen = freeze(value, what, EvaluationContractError)
    assert isinstance(frozen, FrozenMap)
    return frozen


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Everything an evaluator is given for one candidate.

    ``candidate`` holds the parameter values, keyed by the problem's canonical parameter
    identity (for a geometry parameter the canonical target, ``constants.ECal_cell_size``);
    the request does not check them against any search domain. ``settings`` is the
    evaluator-defined content of ``evaluation.settings``. ``inputs`` names already prepared
    local files (a compact XML snapshot, an EDM4hep file); which names an evaluator
    accepts is part of its documentation. ``workdir`` is where the evaluator may write; it
    is neither created nor checked here. ``backend_args`` maps a tool name to its raw
    arguments, kept exactly as given, including order and repetition, and never parsed.
    ``events`` (positive) and ``seed`` are the optional controls of ``evaluation``.

    Every field is optional so that a toy evaluator needs little setup.
    """

    candidate: Mapping[str, Scalar] = field(default_factory=lambda: FrozenMap({}))
    settings: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    inputs: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    workdir: Path | None = None
    events: int | None = None
    seed: int | None = None
    backend_args: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: FrozenMap({}))

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Mapping):
            raise EvaluationContractError("candidate must be a mapping of parameter to value")
        for key, value in self.candidate.items():
            _name(key, "candidate parameter")
            if isinstance(value, float):
                _real(value, f"candidate[{key!r}]")
            elif not isinstance(value, (bool, int, str)):
                raise EvaluationContractError(
                    f"candidate[{key!r}] must be a boolean, integer, finite float or string, "
                    f"got {value!r}"
                )
        object.__setattr__(self, "candidate", FrozenMap(self.candidate))
        object.__setattr__(self, "settings", _json_mapping(self.settings, "settings"))
        object.__setattr__(self, "inputs", _paths(self.inputs, "inputs"))
        if self.workdir is not None and not isinstance(self.workdir, Path):
            raise EvaluationContractError("workdir must be a pathlib.Path")
        if self.events is not None and (
            isinstance(self.events, bool) or not isinstance(self.events, int) or self.events < 1
        ):
            raise EvaluationContractError(f"events must be a positive integer, got {self.events!r}")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise EvaluationContractError(f"seed must be an integer, got {self.seed!r}")
        if not isinstance(self.backend_args, Mapping):
            raise EvaluationContractError("backend_args must be a mapping of tool to arguments")
        backend: dict[str, tuple[str, ...]] = {}
        for tool, args in self.backend_args.items():
            _name(tool, "backend_args tool")
            # a bare string would silently become a tuple of characters
            if isinstance(args, str) or not isinstance(args, Iterable):
                raise EvaluationContractError(
                    f"backend_args[{tool!r}] must be a sequence of strings"
                )
            args = tuple(args)
            for index, arg in enumerate(args):
                if not isinstance(arg, str):
                    raise EvaluationContractError(
                        f"backend_args[{tool!r}][{index}] must be a string, got {arg!r}"
                    )
            backend[tool] = args
        object.__setattr__(self, "backend_args", FrozenMap(backend))


@dataclass(frozen=True, slots=True)
class Measurement:
    """One numerical result of an evaluation.

    ``stage`` classifies the metric (``simulation``, ``reconstruction``, ``analysis``, ...);
    it does not say how or where it was produced. ``metric`` names it within the stage, like
    ``ecal_barrel.position_rms``. ``value`` is a finite number. ``unit`` is a descriptive
    label and is never converted. ``uncertainty`` is a symmetric, nonnegative uncertainty of
    ``value``; ``None`` means that none was provided, not that it is zero. What a metric
    means, and how value and uncertainty are estimated, is defined by the evaluator's
    documentation for its version.
    """

    stage: str
    metric: str
    value: int | float
    unit: str | None = None
    uncertainty: int | float | None = None

    def __post_init__(self) -> None:
        _name(self.stage, "stage")
        _name(self.metric, "metric")
        where = f"measurement {self.stage}/{self.metric}"
        _real(self.value, f"{where}: value")
        if self.unit is not None:
            _name(self.unit, f"{where}: unit")
        if self.uncertainty is not None and _real(self.uncertainty, f"{where}: uncertainty") < 0:
            raise EvaluationContractError(
                f"{where}: uncertainty must not be negative, got {self.uncertainty!r}"
            )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """What an evaluator returns for one request.

    ``status`` separates three outcomes that are never collapsed into a number: ``valid``
    (the evaluation ran and its result is scientifically usable), ``invalid`` (it ran, but
    the result is not usable, for example the evaluator's own checks failed) and ``failed``
    (the execution failed). A missing measurement is simply absent: never ``0`` or
    ``inf``. ``measurements`` may be empty, for instance for a simulation that only produces
    files, and may mix stages; each ``(stage, metric)`` occurs once. ``artifacts`` names
    local output files that can be inputs of a later evaluation; whether they exist is not
    checked here. ``message`` explains the status. ``provenance`` is JSON-compatible
    metadata that the evaluator chose to supply; nothing is collected automatically.
    """

    status: Status
    measurements: tuple[Measurement, ...] = ()
    artifacts: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    message: str | None = None
    provenance: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise EvaluationContractError(
                f"status must be one of {', '.join(_STATUSES)}, got {self.status!r}"
            )
        if isinstance(self.measurements, (str, Mapping)) or not isinstance(
            self.measurements, Iterable
        ):
            raise EvaluationContractError("measurements must be a sequence of Measurement")
        measurements = tuple(self.measurements)
        seen: set[tuple[str, str]] = set()
        for index, measurement in enumerate(measurements):
            if not isinstance(measurement, Measurement):
                raise EvaluationContractError(
                    f"measurements[{index}] must be a Measurement, got {measurement!r}"
                )
            key = (measurement.stage, measurement.metric)
            if key in seen:
                raise EvaluationContractError(
                    f"duplicate measurement of metric {measurement.metric!r} "
                    f"in stage {measurement.stage!r}"
                )
            seen.add(key)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "artifacts", _paths(self.artifacts, "artifacts"))
        if self.message is not None and not isinstance(self.message, str):
            raise EvaluationContractError("message must be a string")
        object.__setattr__(self, "provenance", _json_mapping(self.provenance, "provenance"))
