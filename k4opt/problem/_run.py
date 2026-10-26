"""Internal: evaluate one candidate of a problem file end to end (behind ``k4opt evaluate``).

Not a workflow API. In order it loads the problem, parses and validates the candidate and
the evaluator settings (nothing is created before this succeeds), creates the new run
directory, materializes the candidate's geometry into it, runs the one evaluator through
:func:`k4opt.problem._evaluate._evaluate_candidate` and writes ``summary.json``.
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

from k4opt.evaluation import EvaluationContractError, EvaluationRecord, EvaluationRequest
from k4opt.evaluation.models import Scalar
from k4opt.evaluators import CaloHitMultiplicity
from k4opt.frozen import thaw
from k4opt.geometry import Override
from k4opt.problem._evaluate import _check_identity, _evaluate_candidate, _validate_candidate
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
    """What one run produced: the record, the written summary and where it is."""

    record: EvaluationRecord
    summary: Mapping[str, object]
    run_dir: Path
    fingerprint: str

    @property
    def summary_path(self) -> Path:
        return self.run_dir / SUMMARY_NAME


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
        return int(text)
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
        number = int(text)
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


def _summary(
    problem_path: Path, spec: ProblemSpec, record: EvaluationRecord, run_dir: Path, geometry: object
) -> dict[str, object]:
    result, request = record.result, record.request
    provenance = thaw(result.provenance)
    assert isinstance(provenance, dict)
    argv = provenance.get("argv")
    return {
        "schema": SUMMARY_SCHEMA,
        "problem": {"path": str(problem_path), "name": spec.name},
        "candidate": dict(request.candidate),
        "candidate_key": [list(part) for part in record.candidate_key],
        "evaluator": {"name": record.evaluator_name, "version": record.evaluator_version},
        "status": result.status,
        "message": result.message,
        "measurements": [
            {
                "stage": m.stage,
                "metric": m.metric,
                "value": m.value,
                "unit": m.unit,
                "uncertainty": m.uncertainty,
            }
            for m in result.measurements
        ],
        "events": {"requested": request.events, "observed": provenance.get("events_read")},
        "elapsed_seconds": record.elapsed_seconds,
        "geometry": geometry,
        "simulation": {
            "command": argv,
            "returncode": provenance.get("returncode"),
            "collection": provenance.get("collection"),
        },
        "paths": {"run_dir": str(run_dir)} | {k: str(v) for k, v in result.artifacts.items()},
        "environment": {
            # declared by the problem: a pin, not a check of what ran
            "declared": spec.environment.to_dict()["key4hep"],
            # observed in this process; ddsim is the executable that was started
            "observed": {
                "KEY4HEP_STACK": os.environ.get("KEY4HEP_STACK"),
                "K4GEO": os.environ.get("K4GEO"),
                "ddsim": argv[0] if isinstance(argv, list) and argv else None,
            },
        },
    }


def _check_metrics(spec: ProblemSpec, evaluator: CaloHitMultiplicity) -> None:
    """Reject objectives and constraints the evaluator cannot measure, before any simulation."""
    declared = [(o.stage, o.metric) for o in spec.objectives]
    declared += [(c.stage, c.metric) for c in spec.constraints]
    unsupported = [
        f"{s}/{m}" for s, m in dict.fromkeys(declared) if (s, m) not in evaluator.metrics
    ]
    if unsupported:
        supported = sorted(f"{s}/{m}" for s, m in evaluator.metrics)
        raise EvaluationContractError(
            f"evaluator {evaluator.name!r} does not measure {', '.join(unsupported)}; "
            f"supported: {', '.join(supported)}"
        )
    for constraint in spec.constraints:
        unit = evaluator.metrics[(constraint.stage, constraint.metric)]
        if constraint.unit is not None and constraint.unit != unit:
            raise EvaluationContractError(
                f"constraint on {constraint.stage}/{constraint.metric} declares unit "
                f"{constraint.unit!r} but the evaluator measures it in {unit!r}"
            )


def run_candidate(
    problem_path: str | os.PathLike[str],
    assignments: Sequence[str],
    output: str | os.PathLike[str],
    *,
    evaluator: CaloHitMultiplicity | None = None,
) -> RunOutcome:
    """Evaluate the candidate given as ``REFERENCE=VALUE`` strings; ``output`` must not exist.

    Raises a :class:`~k4opt.errors.K4OptError` for an invalid problem, candidate, settings or
    output location before anything is created. Execution problems are reported as a
    ``failed`` or ``invalid`` result in the outcome.
    """
    evaluator = evaluator or CaloHitMultiplicity()
    problem_path = Path(problem_path).expanduser()  # expanded once: loaded and recorded alike
    spec = load_problem(problem_path)
    assert spec._source is not None
    assert spec._catalog is not None

    values, _ = _validate_candidate(spec, _parse_candidate(spec, assignments))
    _check_identity(spec, evaluator)
    evaluator.check(
        EvaluationRequest(
            candidate=values,
            settings=spec.evaluation.settings,
            events=spec.evaluation.events,
            seed=spec.evaluation.seed,
            backend_args=spec.evaluation.backend,
        )
    )
    _check_metrics(spec, evaluator)
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

    record = _evaluate_candidate(
        spec, values, evaluator, inputs={"geometry": snapshot.compact_path}, workdir=run_dir
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
    summary = _summary(problem_path.absolute(), spec, record, run_dir, geometry)
    with (run_dir / SUMMARY_NAME).open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return RunOutcome(record, summary, run_dir, snapshot.fingerprint)
