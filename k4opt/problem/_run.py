"""Internal: evaluate one candidate of a problem file end to end (behind ``k4opt evaluate``).

Not a workflow API. In order it loads the problem, parses and validates the candidate, binds
and checks the evaluators (nothing is created before this succeeds), creates the new run
directory, materializes the candidate's geometry into it, measures the candidate with every
evaluator through :func:`k4opt.problem._evaluate._evaluate_candidate`, which runs each pipeline
stage once, and writes ``summary.json``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from k4opt.evaluation import EvaluationRecord, EvaluatorProtocol, Measurement, Scalar
from k4opt.frozen import Json, thaw
from k4opt.geometry import Override
from k4opt.problem._evaluate import (
    CandidateRecord,
    _evaluate_candidate,
    _preflight,
    _validate_candidate,
)
from k4opt.problem.config import load_problem
from k4opt.problem.errors import CandidateError, RunError
from k4opt.problem.models import Parameter, ProblemSpec

SUMMARY_NAME = "summary.json"
SUMMARY_SCHEMA = "k4opt.run/1"

# Units that are written into a DD4hep expression (``6.0*mm``); anything else is refused
# rather than passed on unchecked.
_UNITS = ("um", "mm", "cm", "m", "mrad", "rad", "deg", "MeV", "GeV", "TeV")

_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What one run produced: the candidate's record, the written summary and where it is."""

    candidate: CandidateRecord
    summary: Mapping[str, object]
    run_dir: Path
    fingerprint: str

    @property
    def summary_path(self) -> Path:
        return self.run_dir / SUMMARY_NAME

    @property
    def events_read(self) -> object:
        """The events found in the simulation output, ``None`` if it was not inspected."""
        return self.candidate.stages.provenance.get("simulation", {}).get("events_read")


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


def _measurement(m: Measurement | None) -> dict[str, object]:
    return {
        "value": None if m is None else m.value,
        "unit": None if m is None else m.unit,
        "uncertainty": None if m is None else m.uncertainty,
    }


def _evaluator(record: EvaluationRecord, reads: Sequence[str]) -> dict[str, object]:
    """One evaluator's own part of the summary: nothing of the stages it shares."""
    result, request = record.result, record.request
    return {
        "name": record.evaluator_name,
        "version": record.evaluator_version,
        "status": result.status,
        "message": result.message,
        "settings": thaw(request.settings),
        "stages": list(reads),
        "measurements": [
            {"stage": m.stage, "metric": m.metric} | _measurement(m) for m in result.measurements
        ],
        "artifacts": {role: str(path) for role, path in result.artifacts.items()},
        "provenance": thaw(result.provenance),
        "workdir": None if request.workdir is None else str(request.workdir),
        "elapsed_seconds": record.elapsed_seconds,
    }


def _summary(
    problem_path: Path,
    spec: ProblemSpec,
    values: Mapping[str, Scalar],
    candidate: CandidateRecord,
    run_dir: Path,
    geometry: object,
) -> dict[str, object]:
    stages = candidate.stages
    simulation = stages.provenance.get("simulation", {})
    argv = simulation.get("argv")
    generation: Json = stages.provenance.get("generation") or spec.evaluation.generation
    stage_records: dict[str, object] = {}
    for stage, provenance in stages.provenance.items():  # pipeline order
        entry = thaw(provenance)
        assert isinstance(entry, dict)
        if stage in stages.errors:
            entry["error"] = stages.errors[stage]
        if stage in stages.seconds:
            entry["elapsed_seconds"] = stages.seconds[stage]
        stage_records[stage] = entry
    return {
        "schema": SUMMARY_SCHEMA,
        "problem": {"path": str(problem_path), "name": spec.name},
        "candidate": dict(values),
        "candidate_key": [list(part) for part in candidate.candidate_key],
        "status": candidate.status,
        "message": candidate.message,
        # the objective and constraint metrics, each from the evaluator that owns it
        "metrics": [
            {"stage": stage, "metric": metric, "evaluator": owner}
            | _measurement(candidate.value(stage, metric))
            for (stage, metric), owner in candidate.owners.items()
        ],
        "evaluators": [
            _evaluator(record, candidate.reads[record.evaluator_name])
            for record in candidate.records
        ],
        "stages": stage_records,
        "events": {"requested": spec.evaluation.events, "observed": simulation.get("events_read")},
        "elapsed_seconds": candidate.elapsed_seconds,
        "geometry": geometry,
        "generation": thaw(generation),
        "paths": {"run_dir": str(run_dir)}
        | {
            f"{stage}.{role}": str(path)
            for stage, files in stages.artifacts.items()
            for role, path in files.items()
        },
        "environment": {
            # declared by the problem: a pin, not a check of what ran
            "declared": spec.environment.to_dict()["key4hep"],
            # observed in this process; ddsim is the executable that was started
            "observed": {
                "KEY4HEP_STACK": os.environ.get("KEY4HEP_STACK"),
                "K4GEO": os.environ.get("K4GEO"),
                "ddsim": argv[0] if isinstance(argv, (list, tuple)) and argv else None,
            },
        },
    }


def run_candidate(
    problem_path: str | os.PathLike[str],
    assignments: Sequence[str],
    output: str | os.PathLike[str],
    *,
    evaluators: Sequence[EvaluatorProtocol],
) -> RunOutcome:
    """Evaluate the candidate given as ``REFERENCE=VALUE`` strings; ``output`` must not exist.

    ``evaluators`` implement the problem's ``evaluation.evaluators``, in any order. Raises a
    :class:`~k4opt.errors.K4OptError` for an invalid problem, candidate, evaluator, settings or
    output location before anything is created. Execution problems are reported as ``failed``
    or ``invalid`` results in the outcome.
    """
    problem_path = Path(problem_path).expanduser()  # expanded once: loaded and recorded alike
    spec = load_problem(problem_path)
    assert spec._source is not None
    assert spec._catalog is not None

    values, _ = _validate_candidate(spec, _parse_candidate(spec, assignments))
    _preflight(spec, values, evaluators)
    overrides = _overrides(spec, values)

    run_dir = Path(os.path.abspath(output))
    if run_dir.exists() or run_dir.is_symlink():
        raise RunError(f"output {run_dir} already exists; choose a new directory")
    # materialize() validates the location (not inside the geometry source) before it creates
    # run_dir, so a rejected location leaves nothing behind
    try:
        snapshot = spec._source.materialize(overrides, run_dir / "geometry", catalog=spec._catalog)
    except BaseException:
        with contextlib.suppress(OSError):
            run_dir.rmdir()  # created by the failed attempt; only removed while empty
        raise

    candidate = _evaluate_candidate(
        spec, values, evaluators, inputs={"geometry": snapshot.compact_path}, workdir=run_dir
    )
    geometry = {
        "fingerprint": snapshot.fingerprint,
        "compact": str(snapshot.compact_path),
        "manifest": str(snapshot.manifest_path),
        "source_root": str(snapshot.source_root),
        "changes": [
            {"target": c.target, "old_value": c.old_value, "new_value": c.new_value}
            for c in snapshot.changes
        ],
    }
    summary = _summary(problem_path.absolute(), spec, values, candidate, run_dir, geometry)
    with (run_dir / SUMMARY_NAME).open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return RunOutcome(candidate, summary, run_dir, snapshot.fingerprint)
