"""Immutable request and result models of the evaluation contract.

An evaluator receives one :class:`EvaluationRequest` and returns one
:class:`EvaluationResult` made of :class:`Measurement` objects and named artifacts. The
models check what the contract promises (finite numbers, nonnegative uncertainty,
nonblank names, unique metrics) and nothing about scientific meaning, files on disk or
the search domain. Nested caller-owned data is copied on construction.
"""

from __future__ import annotations

import math
import numbers
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from k4opt.evaluation.edm4hep import read_collection
from k4opt.evaluation.errors import EvaluationContractError
from k4opt.frozen import FrozenMap, Json, freeze

type Scalar = bool | int | float | str
"""A parameter value of a candidate."""


class Status(StrEnum):
    """How an evaluation ended; the values are the strings written to ``summary.json``."""

    VALID = "valid"
    INVALID = "invalid"
    FAILED = "failed"


def _name(value: object, what: str) -> str:
    """Return ``value`` unchanged if it is a nonblank string, else raise."""
    if not isinstance(value, str):
        raise EvaluationContractError(f"{what} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise EvaluationContractError(f"{what} must not be blank")
    return value


def _real(value: object, what: str) -> int | float:
    """Return ``value`` as a finite ``int`` or ``float``; booleans are not numbers.

    Any other real number (``numpy.int64``, ``numpy.float32``) becomes the Python number it
    stands for, so values computed with numpy need no conversion.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise EvaluationContractError(f"{what} must be a number, got {value!r}")
    number = int(value) if isinstance(value, numbers.Integral) else float(value)
    if isinstance(number, float) and not math.isfinite(number):
        raise EvaluationContractError(f"{what} must be finite, got {value!r}")
    return number


def _paths(value: Mapping[str, Path], what: str) -> FrozenMap[str, Path]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"{what} must be a mapping of names to paths")
    for name, path in value.items():
        _name(name, f"{what} name")
        if not isinstance(path, Path):
            raise EvaluationContractError(
                f"{what}[{name!r}] must be a pathlib.Path, got {type(path).__name__}"
            )
    return FrozenMap(value)


def _json_mapping(value: Mapping[str, Json], what: str) -> FrozenMap[str, Json]:
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
    evaluator-defined content of its entry in ``evaluation.evaluators``. ``inputs`` names already
    prepared local files (a compact XML snapshot, an EDM4hep file); which names an evaluator
    accepts is part of its documentation. ``workdir`` is where the evaluator may write; it
    is neither created nor checked here. ``backend_args`` maps a tool name to its raw
    arguments, kept exactly as given, including order and repetition, and never parsed.
    ``events`` (positive) and ``seed`` are the optional controls of ``evaluation``.
    ``generation`` is the problem's ``evaluation.generation``, a particle gun or
    ``{"sample": path}`` exactly as written (a relative path is not resolved here); an
    evaluator that runs its own workflow can read the conditions of the measurement from it.

    Every field is optional so that a toy evaluator needs little setup.
    """

    candidate: Mapping[str, Scalar] = field(default_factory=lambda: FrozenMap({}))
    settings: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    inputs: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    workdir: Path | None = None
    events: int | None = None
    seed: int | None = None
    backend_args: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: FrozenMap({}))
    generation: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))

    def read(self, stage: str, collection: str, value_type: str) -> Iterator[Any]:
        """The collection ``collection`` of each event of the file that ``stage`` produced.

        See :func:`~k4opt.evaluation.edm4hep.read_collection`.
        """
        if stage not in self.inputs:
            raise EvaluationContractError(f"inputs[{stage!r}] is missing")
        return read_collection(self.inputs[stage], collection, value_type)

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
        object.__setattr__(self, "generation", _json_mapping(self.generation, "generation"))
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
            # only an ordered sequence keeps the arguments: a str would become characters, a
            # dict its keys, a set an arbitrary order
            if not isinstance(args, (list, tuple)):
                raise EvaluationContractError(
                    f"backend_args[{tool!r}] must be a list or tuple of strings"
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

    @classmethod
    def from_samples(
        cls, stage: str, metric: str, samples: Sequence[int | float], unit: str | None = None
    ) -> Measurement:
        """The mean of ``samples`` (one per event, say) with its statistical uncertainty.

        The uncertainty is the standard error of the mean, ``s / sqrt(N)`` with the sample
        standard deviation ``s`` (``ddof=1``); a single sample has none.
        """
        where = f"measurement {stage}/{metric}"
        if len(samples) == 0:
            raise EvaluationContractError(f"{where}: no samples to average")
        values = [_real(sample, f"{where}: sample") for sample in samples]
        try:
            mean = statistics.fmean(values)
            error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
        except OverflowError:
            raise EvaluationContractError(f"{where}: samples too large to average") from None
        return cls(stage, metric, mean, unit, error)

    def __post_init__(self) -> None:
        _name(self.stage, "stage")
        _name(self.metric, "metric")
        where = f"measurement {self.stage}/{self.metric}"
        object.__setattr__(self, "value", _real(self.value, f"{where}: value"))
        if self.unit is not None:
            _name(self.unit, f"{where}: unit")
        if self.uncertainty is not None:
            uncertainty = _real(self.uncertainty, f"{where}: uncertainty")
            if uncertainty < 0:
                raise EvaluationContractError(
                    f"{where}: uncertainty must not be negative, got {self.uncertainty!r}"
                )
            object.__setattr__(self, "uncertainty", uncertainty)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """What an evaluator returns for one request.

    Build one with :meth:`valid`, :meth:`invalid` or :meth:`failed`. ``status`` separates
    three outcomes that are never collapsed into a number: ``valid`` (the evaluation ran and
    its result is scientifically usable), ``invalid`` (it ran, but the result is not usable, for
    example the evaluator's own checks failed) and ``failed`` (the execution failed); the last
    two need a ``message``. Measurements are independent of the status: an ``invalid`` result
    may keep diagnostic values (hit counts of a configuration that violates a validity
    condition), which an optimizer must not use as objectives, and a ``valid`` result may have
    none (it only produced ``artifacts``; a problem that asks for a metric still requires it).
    A missing measurement is simply absent: never ``0`` or ``inf``. ``measurements`` may mix
    stages; each ``(stage, metric)`` occurs once. ``artifacts`` names
    local output files that can be inputs of a later evaluation; whether they exist is not
    checked here. ``message`` explains the status. ``provenance`` is JSON-compatible
    metadata that the evaluator chose to supply.
    """

    status: Status
    measurements: tuple[Measurement, ...] = ()
    artifacts: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    message: str | None = None
    provenance: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))

    @classmethod
    def valid(
        cls,
        measurements: Sequence[Measurement] = (),
        *,
        artifacts: Mapping[str, Path] | None = None,
        provenance: Mapping[str, Json] | None = None,
        message: str | None = None,
    ) -> EvaluationResult:
        """A scientifically usable result."""
        return cls(Status.VALID, tuple(measurements), artifacts or {}, message, provenance or {})

    @classmethod
    def invalid(
        cls,
        message: str,
        measurements: Sequence[Measurement] = (),
        *,
        artifacts: Mapping[str, Path] | None = None,
        provenance: Mapping[str, Json] | None = None,
    ) -> EvaluationResult:
        """The evaluation ran but its result is not usable; say why. Diagnostics may be kept."""
        return cls(Status.INVALID, tuple(measurements), artifacts or {}, message, provenance or {})

    @classmethod
    def failed(
        cls,
        message: str,
        measurements: Sequence[Measurement] = (),
        *,
        artifacts: Mapping[str, Path] | None = None,
        provenance: Mapping[str, Json] | None = None,
    ) -> EvaluationResult:
        """The execution failed (crash, missing output); say what happened."""
        return cls(Status.FAILED, tuple(measurements), artifacts or {}, message, provenance or {})

    def __post_init__(self) -> None:
        try:
            status = Status(self.status)
        except ValueError:
            raise EvaluationContractError(
                f"status must be one of {', '.join(s.value for s in Status)}, got {self.status!r}"
            ) from None
        object.__setattr__(self, "status", status)
        if not isinstance(self.measurements, (list, tuple)):
            raise EvaluationContractError("measurements must be a list or tuple of Measurement")
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
        if status is not Status.VALID and (
            not isinstance(self.message, str) or not self.message.strip()
        ):
            raise EvaluationContractError(f"a {status.value} result needs a message")
        object.__setattr__(self, "artifacts", _paths(self.artifacts, "artifacts"))
        if self.message is not None and not isinstance(self.message, str):
            raise EvaluationContractError("message must be a string")
        object.__setattr__(self, "provenance", _json_mapping(self.provenance, "provenance"))


type CandidateKey = tuple[tuple[str, str, str, str], ...]
"""Identity of a candidate within one problem: per parameter ``(reference, kind, tag, value)``."""


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One evaluator's attempt at one candidate, as observed by the framework that ran it.

    ``request`` is the object the evaluator received and ``result`` the one it returned, or the
    ``failed`` result that stands in for it when a pipeline stage it reads could not be read. It
    holds only the evaluator's own work: the stages it read, which other evaluators may share,
    are recorded once by the caller, and ``request.inputs`` names their files.
    ``evaluator_name`` and ``evaluator_version`` are those of the evaluator object that ran (the
    caller verified that they match the problem). ``candidate_key`` identifies the candidate
    within the problem that defined it: it is scoped to that problem definition and is neither a
    cross-problem hash nor a complete cache key. ``elapsed_seconds`` is the wall-clock duration
    of the ``evaluate()`` call alone, measured with ``time.perf_counter()``, and ``None`` if the
    evaluator was not called; it is not CPU time, memory use or an estimate.

    ``result.provenance`` is what the evaluator chose to report; ``candidate_key`` and
    ``elapsed_seconds`` are what the framework observed. Neither is proof of the software
    environment actually used.
    """

    request: EvaluationRequest
    result: EvaluationResult
    evaluator_name: str
    evaluator_version: str
    candidate_key: CandidateKey
    elapsed_seconds: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, EvaluationRequest):
            raise EvaluationContractError("request must be an EvaluationRequest")
        if not isinstance(self.result, EvaluationResult):
            raise EvaluationContractError("result must be an EvaluationResult")
        _name(self.evaluator_name, "evaluator_name")
        _name(self.evaluator_version, "evaluator_version")
        if not isinstance(self.candidate_key, tuple) or not all(
            isinstance(part, tuple) and len(part) == 4 and all(isinstance(x, str) for x in part)
            for part in self.candidate_key
        ):
            raise EvaluationContractError(
                "candidate_key must be a tuple of (reference, kind, tag, value) string tuples"
            )
        if self.elapsed_seconds is None:
            return
        if _real(self.elapsed_seconds, "elapsed_seconds") < 0:
            raise EvaluationContractError(
                f"elapsed_seconds must not be negative, got {self.elapsed_seconds!r}"
            )
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
