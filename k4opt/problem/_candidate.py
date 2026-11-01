"""Internal: a candidate's values, and the record of what its evaluation found.

A candidate maps each canonical parameter reference to a value. :func:`_validate_candidate`
checks it against the parameters' domains and keys it (:data:`~k4opt.evaluation.CandidateKey`),
problem-aware, so that ``True``, ``1`` and ``1.0`` are told apart where the parameter does;
:func:`_parse_candidate` reads it from ``REFERENCE=VALUE`` strings (``k4opt evaluate --param``)
exactly, and :func:`_overrides` writes it into the geometry. :class:`CandidateRecord` holds the
result of every task of its evaluation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from k4opt.evaluation import (
    CandidateKey,
    EvaluationContractError,
    EvaluationRecord,
    Measurement,
    Scalar,
    Status,
)
from k4opt.frozen import FrozenMap
from k4opt.geometry import Override
from k4opt.problem._tasks import (
    Kind,
    Plan,
    TaskId,
    TaskResult,
    candidate_message,
    candidate_status,
    changed_files,
    diagnostics,
    stage_task,
)
from k4opt.problem.errors import CandidateError
from k4opt.problem.models import MetricRef, Parameter, ProblemSpec


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _canonical_number(value: int | float) -> str:
    """A stable text for a number, equal for ``1``, ``1.0`` and ``1e0``; exact for any value."""
    # Both conversions are exact, so equal text means equal numerical value (10**23 differs
    # from 1e23, which is rounded). normalize() is avoided because it rounds to the context
    # precision.
    number = Decimal(value) if isinstance(value, int) else Decimal.from_float(value)
    sign, digits, exponent = number.as_tuple()
    assert isinstance(exponent, int)
    text = "".join(map(str, digits)).lstrip("0")
    if not text:
        return "0"  # also -0.0
    stripped = text.rstrip("0")
    exponent += len(text) - len(stripped)
    return f"{'-' if sign else ''}{stripped}e{exponent}"


def _check_number(value: object, reference: str) -> int | float:
    if not _is_number(value):
        raise CandidateError(f"{reference}: expected a number, got {value!r}")
    assert isinstance(value, (int, float))
    if isinstance(value, float) and not math.isfinite(value):
        raise CandidateError(f"{reference}: value must be finite, got {value!r}")
    return value


def _validate_value(parameter: Parameter, value: object) -> tuple[str, str]:
    """Return ``(tag, canonical value)`` of ``value`` for ``parameter`` or raise."""
    reference = parameter.geometry.canonical_target
    if parameter.kind == "continuous":
        number = _check_number(value, reference)
    elif parameter.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CandidateError(f"{reference}: expected an integer, got {value!r}")
        number = value
    else:
        choices = parameter.choices or ()
        if all(isinstance(choice, str) for choice in choices):
            if not isinstance(value, str) or value not in choices:
                raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
            return "str", value
        if all(isinstance(choice, bool) for choice in choices):
            if not isinstance(value, bool) or value not in choices:
                raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
            return "bool", "true" if value else "false"
        number = _check_number(value, reference)
        if not any(number == choice for choice in choices):
            raise CandidateError(f"{reference}: {value!r} is not one of {list(choices)!r}")
        return "num", _canonical_number(number)
    assert parameter.bounds is not None
    low, high = parameter.bounds
    if not low <= number <= high:
        raise CandidateError(f"{reference}: {value!r} is outside the bounds [{low}, {high}]")
    return ("int" if parameter.kind == "integer" else "num"), _canonical_number(number)


def _validate_candidate(
    spec: ProblemSpec, candidate: Mapping[str, Scalar]
) -> tuple[dict[str, Scalar], CandidateKey]:
    """Check ``candidate`` against every parameter; return it in problem order with its key."""
    if not isinstance(candidate, Mapping):
        raise CandidateError(f"candidate must be a mapping, got {type(candidate).__name__}")
    declared = [parameter.geometry.canonical_target for parameter in spec.parameters]
    bad_keys = [key for key in candidate if not isinstance(key, str)]
    if bad_keys:
        raise CandidateError(f"candidate keys must be strings, got {bad_keys!r}")
    unknown = [key for key in candidate if key not in declared]
    if unknown:
        raise CandidateError(f"candidate has unknown parameter(s): {', '.join(unknown)}")
    missing = [reference for reference in declared if reference not in candidate]
    if missing:
        raise CandidateError(f"candidate is missing parameter(s): {', '.join(missing)}")
    values: dict[str, Scalar] = {}
    key: list[tuple[str, str, str, str]] = []
    for parameter, reference in zip(spec.parameters, declared, strict=True):
        value = candidate[reference]
        tag, canonical = _validate_value(parameter, value)
        values[reference] = value
        key.append((reference, parameter.kind, tag, canonical))
    return values, tuple(key)


_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")


def _int(reference: str, text: str) -> int:
    """``int(text)`` for a text that matches ``_INTEGER``."""
    try:
        return int(text)
    except ValueError:  # more digits than Python converts (sys.get_int_max_str_digits())
        raise CandidateError(f"{reference}: {text[:20]}... has too many digits") from None


def _parse_value(parameter: Parameter, text: str) -> Scalar:
    """The value ``text`` stands for, read according to the kind of ``parameter``."""
    reference = parameter.geometry.canonical_target
    choices = parameter.choices or ()
    if parameter.kind == "categorical" and all(isinstance(c, str) for c in choices):
        return text
    if parameter.kind == "categorical" and all(isinstance(c, bool) for c in choices):
        if text not in ("true", "false"):
            raise CandidateError(f"{reference}: expected true or false, got {text!r}")
        return text == "true"
    if parameter.kind == "integer":
        if not _INTEGER.fullmatch(text):
            raise CandidateError(f"{reference}: expected an integer, got {text!r}")
        return _int(reference, text)
    if not _DECIMAL.fullmatch(text):
        raise CandidateError(f"{reference}: expected a number, got {text!r}")
    if parameter.kind == "categorical":
        # Compare with the declared choices exactly (Decimal vs int or float is exact), and
        # return the choice itself: float(text) would merge 2**53 and 2**53+1.
        exact = Decimal(text)
        numeric = [c for c in choices if not isinstance(c, bool)]
        for choice in numeric:
            if exact == choice:
                return choice
        for choice in numeric:  # a float choice such as 1e23 is not the decimal 10**23 ...
            # ... so also accept its shortest decimal spelling, never a rounded neighbor
            if (
                isinstance(choice, float)
                and math.isfinite(choice)
                and exact == Decimal(repr(choice))
            ):
                return choice
        # never round an unmatched value into a neighboring choice
        raise CandidateError(f"{reference}: {text!r} is not one of {list(choices)!r}")
    if _INTEGER.fullmatch(text):
        # an integer literal stays exact where float would change it (above 2**53) or overflow
        number = _int(reference, text)
        try:
            return float(number) if float(number) == number else number
        except OverflowError:
            return number
    return float(text)


def _split_assignment(assignment: str) -> tuple[str, str]:
    """Split ``REFERENCE=VALUE`` at the first ``=`` outside ``[...]`` (``layer[id=1].x=90``)."""
    depth = 0
    for index, char in enumerate(assignment):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(depth - 1, 0)
        elif char == "=" and depth == 0:
            return assignment[:index], assignment[index + 1 :]
    raise CandidateError(f"--param {assignment!r}: expected REFERENCE=VALUE")


def _parse_candidate(spec: ProblemSpec, assignments: Sequence[str]) -> dict[str, Scalar]:
    by_reference = {p.geometry.canonical_target: p for p in spec.parameters}
    candidate: dict[str, Scalar] = {}
    for assignment in assignments:
        reference, text = _split_assignment(assignment)
        if reference in candidate:
            raise CandidateError(f"--param {reference} is given more than once")
        parameter = by_reference.get(reference)
        # an unknown reference is reported by the candidate validation, with the known ones
        candidate[reference] = text if parameter is None else _parse_value(parameter, text)
    return candidate


# Units that are written into a DD4hep expression (``6.0*mm``); anything else is refused
# rather than passed on unchecked.
_UNITS = ("um", "mm", "cm", "m", "mrad", "rad", "deg", "MeV", "GeV", "TeV")


def _xml_value(parameter: Parameter, value: Scalar) -> str:
    """The literal written into the geometry XML for a validated ``value``."""
    reference = parameter.geometry.canonical_target
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if parameter.unit is None:
        return repr(value)
    if parameter.unit not in _UNITS:
        raise CandidateError(
            f"{reference}: unit {parameter.unit!r} cannot be written into the geometry; "
            f"supported: {', '.join(_UNITS)}"
        )
    return f"{value!r}*{parameter.unit}"


def _overrides(spec: ProblemSpec, values: Mapping[str, Scalar]) -> list[Override]:
    assert spec._source is not None
    assert len(spec._targets) == len(spec.parameters)
    overrides = []
    for parameter, target in zip(spec.parameters, spec._targets, strict=True):
        value = values[parameter.geometry.canonical_target]
        overrides.append(
            Override(target=target.target, value=_xml_value(parameter, value), file=target.file)
        )
    return overrides


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One candidate, measured by every evaluator of its problem.

    ``plan`` holds its tasks, and ``results`` the result of each, in plan order: its geometry,
    the pipeline stages of each sample that its evaluators read, each recorded once, and every
    evaluation, with only its own work. ``elapsed_seconds`` is the wall time of the whole
    evaluation: the tasks and k4opt's checks. Its mappings are read-only copies.

    ``represents_candidate`` is ``False`` when the caller supplied a required stage that the
    parameters change (``k4opt evaluate --input gamma/simulation=FILE``): that file was not made
    from the candidate, so the record must never count as an evaluation of it, whatever its
    status.
    ``warnings`` are the preflight's: collections the simulation was not expected to write, and
    what is measured or simulated twice.
    """

    candidate_key: CandidateKey
    plan: Plan
    results: Mapping[TaskId, TaskResult]
    elapsed_seconds: float
    represents_candidate: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = [task.id.label for task in self.plan.tasks if task.id not in self.results]
        if missing or len(self.results) != len(self.plan.tasks):
            raise EvaluationContractError(
                f"the results do not match the plan's tasks (missing: {', '.join(missing)})"
            )
        results = {task.id: self.results[task.id] for task in self.plan.tasks}
        object.__setattr__(self, "results", FrozenMap(results))

    @property
    def evaluations(self) -> Mapping[str, EvaluationRecord]:
        """The record of each evaluation, by its label (``gamma/calo.photon``, ``readout``), in
        plan order; a blocked one's evaluator was never called."""
        return FrozenMap(
            {
                task.id.label: record
                for task in self.plan.of(Kind.EVALUATOR)
                if (record := self.results[task.id].record) is not None
            }
        )

    @property
    def records(self) -> tuple[EvaluationRecord, ...]:
        """The record of each evaluation, in plan order."""
        return tuple(self.evaluations.values())

    @property
    def supplied(self) -> frozenset[str]:
        """The stages the caller supplied as files, by label (``gamma/simulation``)."""
        return frozenset(t.id.label for t in self.plan.of(Kind.STAGE) if t.given == "supplied")

    def stage(self, name: str, sample: str) -> TaskResult | None:
        """The result of the stage ``name`` of ``sample``, ``None`` if it was not planned."""
        return self.results.get(stage_task(name, sample))

    @property
    def status(self) -> Status:
        """From the required tasks alone: ``failed`` if one failed (the evaluation is
        incomplete), otherwise ``invalid`` if an evaluator found the result unusable (and the
        tasks that read it were not run), otherwise ``valid``. A diagnostic changes nothing."""
        return candidate_status(self.plan, self.results)

    @property
    def message(self) -> str | None:
        """Every required task that failed or is invalid, with its message, then the required
        ones they blocked."""
        return candidate_message(self.plan, self.results)

    @property
    def diagnostics(self) -> tuple[TaskResult, ...]:
        """The result of each task that is not required and did not succeed, in plan order."""
        return tuple(diagnostics(self.plan, self.results))

    def changed(self) -> list[str]:
        """The references of the files its tasks made that are no longer what its record says
        (:meth:`~k4opt.problem._tasks.Made.changed`)."""
        return changed_files(self.results.values())

    def record(self, label: str) -> EvaluationRecord:
        """The record of the evaluation ``label`` (``gamma/calo.photon``, ``readout``);
        :class:`KeyError` if there is none."""
        return self.evaluations[label]

    def value(self, ref: MetricRef) -> Measurement | None:
        """The measurement of the metric ``ref`` by the evaluation it names.

        ``None`` unless that evaluation's result is ``valid``: the diagnostic values of an
        ``invalid`` or ``failed`` result never stand in for an objective.
        """
        try:
            result = self.record(ref.invocation).result
        except KeyError:
            return None
        if result.status is not Status.VALID:
            return None
        return next(
            (m for m in result.measurements if (m.stage, m.metric) == (ref.stage, ref.metric)),
            None,
        )
