"""Candidates of a problem file, evaluated end to end (behind ``k4opt evaluate``).

:class:`EvaluationSession` loads the problem once, binds and checks the evaluator
implementations and records the identity of every file the problem names. For each candidate it
then validates the candidate, creates a new instance of each evaluator and checks its request
for the candidate, checks that no such file changed (nothing is created before this succeeds),
creates the new run directory, materializes the candidate's geometry into it, measures the
candidate with those instances through
:func:`k4opt.problem._evaluate._execute_plan`, which runs each pipeline stage once, assesses it
against the objectives and constraints (:mod:`k4opt.problem._assess`) and writes
``summary.json``. :func:`run_candidate` evaluates one candidate given as ``REFERENCE=VALUE``
strings.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from k4opt import __version__
from k4opt.evaluation import (
    CandidateKey,
    EvaluationRecord,
    EvaluatorFactory,
    Measurement,
    Scalar,
    evaluator_source,
)
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.geometry import GeometryError, GeometrySource, MaterializationResult, Override
from k4opt.jsonio import write_json
from k4opt.problem._assess import SUMMARY_SCHEMA, Assessment, assess
from k4opt.problem._evaluate import (
    CandidateRecord,
    MetricKey,
    _bind_problem,
    _Binding,
    _execute_plan,
    _Plan,
    _preflight,
    _validate_candidate,
)
from k4opt.problem._stages import STAGE_TYPES, file_identity, input_files, plan_stages
from k4opt.problem.config import load_problem
from k4opt.problem.errors import CandidateError, RunError
from k4opt.problem.models import EvaluatorSpec, Parameter, ProblemSpec
from k4opt.revision import code_identity, code_unchanged

SUMMARY_NAME = "summary.json"

# Units that are written into a DD4hep expression (``6.0*mm``); anything else is refused
# rather than passed on unchecked.
_UNITS = ("um", "mm", "cm", "m", "mrad", "rad", "deg", "MeV", "GeV", "TeV")

_INTEGER = re.compile(r"[+-]?[0-9]+")
_DECIMAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What one run produced: the candidate's record, the written summary and where it is, and
    the candidate's assessment against the objectives and constraints."""

    candidate: CandidateRecord
    summary: Mapping[str, object]
    run_dir: Path
    fingerprint: str
    assessment: Assessment

    @property
    def summary_path(self) -> Path:
        return self.run_dir / SUMMARY_NAME

    @property
    def events_read(self) -> object:
        """The events found in the simulation output, ``None`` if it was not inspected."""
        return self.candidate.stages.provenance.get("simulation", {}).get("events_read")

    @property
    def events_requested(self) -> object:
        """The events the simulation stage asks for, ``None`` if it asks for none."""
        events = self.summary.get("events")
        return events.get("requested") if isinstance(events, Mapping) else None


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


def _parse_inputs(assignments: Sequence[str]) -> dict[str, str]:
    """``STAGE=FILE`` strings as files by stage; the session checks the stages and files."""
    inputs: dict[str, str] = {}
    for assignment in assignments:
        stage, separator, text = assignment.partition("=")
        if not separator or not stage or not text:
            raise RunError(f"--input {assignment!r}: expected STAGE=FILE")
        if stage in inputs:
            raise RunError(f"--input {stage} is given more than once")
        inputs[stage] = text
    return inputs


def _supplied(inputs: Mapping[str, str | os.PathLike[str]]) -> dict[str, Path]:
    """The files that stand in for stages, as absolute paths; each must name a stage and a file."""
    supplied: dict[str, Path] = {}
    for stage, given in inputs.items():
        if stage not in STAGE_TYPES:
            raise RunError(f"input {stage!r}: unknown stage; use {', '.join(STAGE_TYPES)}")
        path = Path(os.path.abspath(Path(given).expanduser()))
        if not path.is_file():
            raise RunError(f"input {stage}={os.fspath(given)}: {path} is not a file")
        supplied[stage] = path
    return supplied


def _readable_identity(path: Path) -> dict[str, Json] | None:
    """``bytes`` and ``sha256`` of a file, ``None`` if it cannot be read (its stage's to report)."""
    try:
        return file_identity(path)
    except OSError:
        return None


def _timeout(value: float | None) -> float | None:
    """``value`` as a time limit in seconds; :class:`RunError` unless it is positive and finite."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < math.inf:
        raise RunError(f"timeout must be a positive number of seconds, got {value!r}")
    return float(value)


def _check_code() -> None:
    """Raise :class:`RunError` if k4opt's files are no longer those this process identified."""
    if not code_unchanged():
        raise RunError(
            "k4opt's code changed on disk after this process first identified it, so its "
            "records could not say which code ran; start a new process"
        )


def _run_reference(run: Mapping[str, Json] | None) -> object:
    """``run`` as the summary records it: a plain copy of JSON-compatible data, or ``None``."""
    if run is None:
        return None
    if not isinstance(run, Mapping):
        raise RunError(f"run must be a mapping, got {type(run).__name__}")
    return thaw(freeze(run, "run", RunError))


def _declared(units: Mapping[MetricKey, str] | None) -> list[dict[str, str]] | None:
    """The metrics an evaluator declares (``units``), each with its unit; ``None`` if it declares
    none."""
    if units is None:
        return None
    return [{"stage": s, "metric": m, "unit": unit} for (s, m), unit in units.items()]


def _evaluator(
    record: EvaluationRecord, reads: Sequence[str], entry: EvaluatorSpec, taken: Mapping[str, Json]
) -> dict[str, object]:
    """One evaluator's own part of the summary: nothing of the stages it shares. ``taken`` is
    what the session recorded of the evaluator when it opened: its source and its metrics."""
    result, request = record.result, record.request
    return {
        "name": record.evaluator_name,
        "version": record.evaluator_version,
        "stage": entry.stage,
        "source": thaw(taken["source"]),
        # each with the unit every measurement of it carries, whatever the status
        "metrics": thaw(taken["metrics"]),
        "status": result.status,
        "message": result.message,
        # as the problem wrote them, and as the evaluator received them (defaults filled in,
        # quantities in the declared unit)
        "settings": {"declared": thaw(entry.settings), "used": thaw(request.settings)},
        "collections": dict(request.collections),
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
    assessment: Assessment,
    run_dir: Path,
    geometry: object,
    taken: Mapping[str, Mapping[str, Json]],
    run: object,
    process: Mapping[str, Json],
) -> dict[str, object]:
    stages = candidate.stages
    simulation = stages.provenance.get("simulation", {})
    environment = thaw(process["environment"])
    assert isinstance(environment, dict)
    argv = simulation.get("argv")
    simulate = spec.stages.get("simulation")
    entries = {entry.name: entry for entry in spec.evaluators}
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
        "id": str(uuid.uuid4()),  # the record's identity, wherever it is copied to
        "run": run,  # what the caller evaluates it for (a study and trial, say), null if nothing
        "problem": {"path": str(problem_path), "name": spec.name, "definition": spec.to_dict()},
        "candidate": dict(values),
        "candidate_key": [list(part) for part in candidate.candidate_key],
        "status": candidate.status,
        "message": candidate.message,
        # False if a supplied stage file was not made from this candidate (--input)
        "represents_candidate": candidate.represents_candidate,
        "supplied": sorted(stages.supplied),
        "warnings": list(candidate.warnings),
        # the objective and constraint metrics, each from the evaluator that owns it
        "metrics": [
            {"stage": stage, "metric": metric, "evaluator": owner}
            | _measurement(candidate.value(stage, metric))
            for (stage, metric), owner in candidate.owners.items()
        ],
        # the objectives and the constraints judged, from those metrics
        "assessment": assessment.to_dict(),
        "evaluators": [
            _evaluator(
                record,
                candidate.reads[record.evaluator_name],
                entries[record.evaluator_name],
                taken[record.evaluator_name],
            )
            for record in candidate.records
        ],
        "stages": stage_records,
        "events": {
            "requested": None if simulate is None else simulate.events,
            "observed": simulation.get("events_read"),
        },
        "elapsed_seconds": candidate.elapsed_seconds,
        "geometry": geometry,
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
                "k4opt": __version__,
                # what the version only labels: the content of its code, and its checkout
                "k4opt_source": thaw(process["k4opt"]),
                # the session's, which the process still had when this candidate started
                **environment,
                "ddsim": argv[0] if isinstance(argv, (list, tuple)) and argv else None,
            },
        },
    }


class EvaluationSession:
    """Candidates of one problem, evaluated one after another with the same evaluators.

    :meth:`open` loads the problem file once, so later edits to it do not reach the session,
    binds the evaluator implementations to it and checks what does not depend on a candidate.
    It also records the identity (size and SHA-256) of every file the stages read: a stage given
    as a ``file``, a file a ddsim option names, a file the caller supplies, and what the
    geometry's external references read. Every candidate gets a new instance of each evaluator,
    so no state of one reaches another candidate; it must declare what the first instance,
    created when the session opened, declared (its class, name, version, stage, settings,
    collections and metrics).
    Each candidate is then checked afresh, by :meth:`check` and again by :meth:`evaluate`: its
    values against the parameter domains, the ``check()`` of each new instance with a request
    made for this candidate, and that none of those files, the environment or k4opt's
    code changed, which :meth:`evaluate` checks again before it writes the record. A candidate
    whose values change a ``ref`` can name an external file of its own; what it reads is
    recorded, and must not change while the candidate runs nor between the candidates that name
    it. One candidate at a time; a session is not thread-safe.
    """

    __slots__ = (
        "_binding",
        "_externals",
        "_files",
        "_inputs",
        "_path",
        "_process",
        "_read",
        "_spec",
        "_taken",
        "_timeout",
        "_warnings",
    )

    def __init__(
        self,
        path: Path,
        spec: ProblemSpec,
        binding: _Binding,
        inputs: Mapping[str, Path],
        files: Mapping[str, tuple[Path, Mapping[str, Json] | None]],
        externals: Mapping[str, Json],
        taken: Mapping[str, Mapping[str, Json]],
        warnings: tuple[str, ...],
        process: Mapping[str, Json],
        timeout: float | None,
    ) -> None:
        """Use :meth:`open`."""
        self._path = path
        self._spec = spec
        self._binding = binding
        self._inputs = FrozenMap(inputs)
        self._files = FrozenMap(files)
        self._externals = externals
        self._taken = FrozenMap(taken)
        self._warnings = warnings
        self._process = process
        self._timeout = timeout
        # what the external references only candidates have read, by reference, without the path
        self._read: dict[str, Json] = {}

    @classmethod
    def open(
        cls,
        problem_path: str | os.PathLike[str],
        implementations: Sequence[EvaluatorFactory],
        *,
        inputs: Mapping[str, str | os.PathLike[str]] | None = None,
        timeout: float | None = None,
    ) -> EvaluationSession:
        """Load the problem file ``problem_path`` to evaluate its candidates with
        ``implementations``.

        ``implementations`` (evaluator classes, as :func:`~k4opt.evaluation.load_evaluator`
        returns them) implement the problem's ``evaluators``, in any order; one instance of each
        is created now to bind it to the problem, and a new one for every candidate. ``inputs``
        maps a stage to a file that stands in for it (``k4opt evaluate --input``): that stage
        does not run, nor do the stages only it needs, and a supplied stage that the parameters
        change was not made from the candidate, which every outcome says
        (``represents_candidate``). ``timeout`` is how many seconds a stage that runs a program
        (ddsim) may take: one that runs longer is killed, and its candidate fails with the logs
        kept. Raises a :class:`~k4opt.errors.K4OptError` for an invalid problem, evaluator,
        stage or input (an :class:`~k4opt.evaluation.EvaluationContractError` for an
        implementation that cannot be created, too), and :class:`~k4opt.problem.RunError` if
        k4opt's own files changed since this process first identified them
        (:func:`k4opt.revision.code_unchanged`).
        """
        path = Path(problem_path).expanduser()  # expanded once: loaded and recorded alike
        limit = _timeout(timeout)
        spec = load_problem(path)
        supplied = _supplied(inputs or {})
        binding = _bind_problem(spec, implementations, supplied)
        for stage in supplied:  # read by an evaluator, given the other supplied stages
            if stage not in plan_stages(*binding.targets, provided=supplied.keys() - {stage}):
                raise RunError(f"input {stage}: no evaluator of the problem reads that stage")
        named = input_files(binding.context, *binding.targets, provided=supplied)
        named |= {f"supplied.{stage}": file for stage, file in supplied.items()}
        # an unreadable file has no identity: its stage reports it, and a record still names it
        files = {label: (file, _readable_identity(file)) for label, file in named.items()}
        assert spec._source is not None
        externals = freeze(spec._source.external_files(spec._catalog), "geometry", RunError)
        assert isinstance(externals, Mapping)
        taken = {}
        for bound in binding.evaluators:  # what the first instances declared
            declaration = bound.declaration
            data = {
                "source": evaluator_source(declaration.cls),
                "metrics": _declared(declaration.metrics),
            }
            frozen = freeze(data, f"evaluator {declaration.name}", RunError)
            assert isinstance(frozen, Mapping)
            taken[declaration.name] = frozen
        # before anything runs, so that an edit to the checkout during a long run is not
        # recorded as the code that ran, nor a later change of the environment as the session's
        environment = {name: os.environ.get(name) for name in ("KEY4HEP_STACK", "K4GEO")}
        process = freeze(
            {"k4opt": code_identity(), "environment": environment}, "process", RunError
        )
        _check_code()
        assert isinstance(process, Mapping)
        return cls(
            path.absolute(),
            spec,
            binding,
            supplied,
            files,
            externals,
            taken,
            binding.warnings,
            process,
            limit,
        )

    @property
    def spec(self) -> ProblemSpec:
        """The problem, as loaded when the session was opened."""
        return self._spec

    @property
    def problem_path(self) -> Path:
        """The absolute path of the problem file."""
        return self._path

    @property
    def warnings(self) -> tuple[str, ...]:
        """Collections an evaluator reads that the simulation is not expected to write; the
        simulation's output decides, so they do not stop an evaluation. Each summary keeps
        them too."""
        return self._warnings

    def identity(self) -> dict[str, object]:
        """What this session evaluates with, as plain JSON-compatible data.

        ``problem``: the path and normalized definition; ``inputs``: the path, size and SHA-256
        of each file the stages read, by where it is named (``stages.generation.file``) or
        supplied (``supplied.simulation``), a file that could not be read with its path only;
        ``evaluators``: the name, version, source identity and declared metrics of each, in
        problem order; ``geometry``: the compact file, the source root, the fingerprint of the
        unmodified geometry and what its external references read
        (:meth:`~k4opt.geometry.GeometrySource.external_files`); ``environment``:
        ``KEY4HEP_STACK`` and ``K4GEO``; ``k4opt``: its version, the SHA-256 of its code and its
        git checkout, as every summary records them. The environment, k4opt and the external
        files are taken when the session opened. A fresh copy on every call.
        """
        source, catalog = self._spec._source, self._spec._catalog
        assert source is not None
        assert catalog is not None
        process = thaw(self._process)
        assert isinstance(process, dict)
        return {
            "problem": {"path": str(self._path), "definition": self._spec.to_dict()},
            "inputs": {
                label: {"path": str(file), **(identity or {})}
                for label, (file, identity) in self._files.items()
            },
            "evaluators": [
                {"name": entry.name, "version": entry.version}
                | {key: thaw(value) for key, value in self._taken[entry.name].items()}
                for entry in self._spec.evaluators
            ],
            "geometry": {
                "compact": source.compact,
                "source_root": str(source.root),
                "fingerprint": source.fingerprint([], catalog=catalog),
                "externals": thaw(self._externals),
            },
            "environment": process["environment"],
            "k4opt": {"version": __version__, **process["k4opt"]},
        }

    def verify_inputs(self) -> None:
        """Raise :class:`~k4opt.problem.RunError` if a file the stages read (named by the problem
        or supplied) is not the file it was when the session opened: other bytes, or now readable
        or unreadable.

        A file is hashed again only when its size, times or inode changed.
        """
        for label, (file, opened) in self._files.items():
            now = _readable_identity(file)
            if now != opened:
                raise RunError(
                    f"{label}: {file} changed after the session opened ({_described(opened)} "
                    f"then, {_described(now)} now); evaluate the candidates of a problem with the "
                    "same files"
                )

    def _verify_externals(self) -> None:
        """Raise :class:`~k4opt.problem.RunError` if an external reference of the geometry
        (``${DD4hepINSTALL}/...``) does not read the files it read when the session opened."""
        source = self._spec._source
        assert source is not None
        now = source.external_files(self._spec._catalog)
        for ref, opened in self._externals.items():
            if now.get(ref) != thaw(opened):
                path = (now.get(ref) or {}).get("path") or "no file"
                raise RunError(
                    f"geometry: {ref} does not read the files it read when the session opened "
                    f"(now {path}); the candidates of a session start from one geometry"
                )

    def _candidate_externals(self, snapshot: MaterializationResult) -> dict[str, dict[str, Json]]:
        """What the external references of a candidate's geometry read that those of the
        session's do not: a ``ref`` the candidate changes can name another file
        (``${CAD}/other.stl``)."""
        if not any(change.attribute == "ref" and change.changed for change in snapshot.changes):
            return {}
        assert self._spec._source is not None
        try:
            now = GeometrySource(
                self._spec._source.compact, root=snapshot.output_root
            ).external_files()
        except GeometryError as exc:
            raise RunError(f"geometry: {exc}") from exc
        opened = thaw(self._externals)
        assert isinstance(opened, dict)
        return {
            ref: entry for ref, entry in now.items() if _content(entry) != _content(opened.get(ref))
        }

    def _check_read(self, externals: Mapping[str, Mapping[str, Json]]) -> None:
        """Raise :class:`~k4opt.problem.RunError` if a reference only candidates name reads other
        files than it did for an earlier candidate of the session."""
        for ref, entry in externals.items():
            if ref in self._read and self._read[ref] != _content(entry):
                raise RunError(
                    f"geometry: {ref} does not read the files it read for an earlier candidate "
                    f"(now {entry.get('path') or 'no file'}); the candidates of a session start "
                    "from one geometry"
                )

    def _verify(self) -> None:
        """Raise :class:`~k4opt.problem.RunError` if the environment, k4opt's code, a file the
        stages read or what the geometry's external references read changed since the session
        opened."""
        self._verify_environment()
        _check_code()
        self.verify_inputs()
        self._verify_externals()

    def _verify_environment(self) -> None:
        """Raise :class:`~k4opt.problem.RunError` if ``KEY4HEP_STACK`` or ``K4GEO`` is not what
        it was when the session opened: the stages would run in another environment than the
        geometry was discovered in, and than the records say."""
        opened = self._process["environment"]
        assert isinstance(opened, Mapping)
        for name, then in opened.items():
            now = os.environ.get(name)
            if now != then:
                raise RunError(
                    f"{name} changed after the session opened ({then!r} then, {now!r} now); the "
                    "candidates of a session run in one environment"
                )

    def check(self, candidate: Mapping[str, Scalar]) -> None:
        """Check ``candidate`` as :meth:`evaluate` does before it creates anything, and stop there.

        ``candidate`` maps each canonical parameter reference to its value. Raises
        :class:`~k4opt.problem.CandidateError` for a value outside its domain, what creating an
        evaluator or its ``check()`` raises for the request this candidate gives it, and
        :class:`~k4opt.problem.RunError` for a file the stages or the geometry read, the
        environment or k4opt's code that changed since the session opened.
        """
        self._prepare(candidate)

    def evaluate(
        self,
        candidate: Mapping[str, Scalar],
        output: str | os.PathLike[str],
        *,
        run: Mapping[str, Json] | None = None,
    ) -> RunOutcome:
        """Evaluate ``candidate`` into the new directory ``output`` and write its summary there.

        ``candidate`` maps each canonical parameter reference to its value. ``run`` is recorded
        as the summary's ``run``: JSON-compatible data that says what the record was made for
        (a study and a trial, say). Raises a :class:`~k4opt.errors.K4OptError` before anything
        is created for what :meth:`check` refuses, an existing ``output`` or an unusable
        location. Execution problems are ``failed`` or ``invalid`` results in the outcome; a
        result that breaks the evaluation contract raises an
        :class:`~k4opt.evaluation.EvaluationContractError` after the stages ran, and no summary
        is written. Nor is one written if what :meth:`check` verifies changed while the
        candidate ran: that raises a :class:`~k4opt.problem.RunError`, since the record could not
        say which files or code the candidate was evaluated with.
        """
        spec = self._spec
        assert spec._source is not None
        assert spec._catalog is not None
        reference = _run_reference(run)
        started = time.perf_counter()
        values, key, plan, overrides = self._prepare(candidate)
        # the checks are part of the candidate's time; materializing is not
        checked = time.perf_counter() - started

        run_dir = Path(os.path.abspath(output))
        if run_dir.exists() or run_dir.is_symlink():
            raise RunError(f"output {run_dir} already exists; choose a new directory")
        # materialize() validates the location (not inside the geometry source) before it creates
        # run_dir, so a rejected location leaves nothing behind
        try:
            snapshot = spec._source.materialize(
                overrides, run_dir / "geometry", catalog=spec._catalog
            )
        except BaseException:
            with contextlib.suppress(OSError):
                run_dir.rmdir()  # created by the failed attempt; only removed while empty
            raise
        try:
            externals = self._candidate_externals(snapshot)
            self._check_read(externals)
        except RunError as exc:
            raise RunError(f"{exc} (in {run_dir}, which has no {SUMMARY_NAME})") from None

        record = _execute_plan(
            spec,
            plan,
            key,
            inputs={"geometry": snapshot.compact_path, **self._inputs},
            workdir=run_dir,
            checked_seconds=checked,
            timeout=self._timeout,
        )
        try:
            self._verify()  # the record names what the stages and evaluators read
            now = self._candidate_externals(snapshot)
            if now != externals:
                refs = dict.fromkeys([*externals, *now])
                changed = [ref for ref in refs if now.get(ref) != externals.get(ref)]
                raise RunError(
                    f"geometry: {', '.join(changed)} does not read the files it read when the "
                    "candidate started"
                )
        except RunError as exc:
            raise RunError(
                f"{exc} (while the candidate in {run_dir} ran, which has no {SUMMARY_NAME})"
            ) from None
        self._read |= {ref: _content(entry) for ref, entry in externals.items()}
        assessment = assess(spec, record)
        geometry = {
            "fingerprint": snapshot.fingerprint,
            "compact": str(snapshot.compact_path),
            "manifest": str(snapshot.manifest_path),
            "source_root": str(snapshot.source_root),
            "changes": [
                {"target": c.target, "old_value": c.old_value, "new_value": c.new_value}
                for c in snapshot.changes
            ],
            "externals": thaw(self._externals),
            "candidate_externals": externals,
        }
        summary = _summary(
            self._path,
            spec,
            values,
            record,
            assessment,
            run_dir,
            geometry,
            self._taken,
            reference,
            self._process,
        )
        try:
            write_json(run_dir / SUMMARY_NAME, summary)
        except OSError as exc:
            raise RunError(f"cannot write summary in {run_dir}: {exc}") from exc
        return RunOutcome(record, summary, run_dir, snapshot.fingerprint, assessment)

    def _prepare(
        self, candidate: Mapping[str, Scalar]
    ) -> tuple[dict[str, Scalar], CandidateKey, _Plan, list[Override]]:
        """The checked candidate, its key, its plan (its own evaluator instances and the requests
        made for it) and its overrides."""
        values, key = _validate_candidate(self._spec, candidate)
        self._verify()  # first: a file that is gone changed, more than it is missing
        plan = _preflight(self._binding, values)
        overrides = _overrides(self._spec, values)
        return values, key, plan, overrides


def _content(entry: object) -> Json:
    """What an external reference reads (:func:`~k4opt.geometry.includes.external_files`),
    without where the files are: the same files in another snapshot read alike."""
    if not isinstance(entry, Mapping):
        return None
    return {key: value for key, value in entry.items() if key != "path"}


def _described(identity: Mapping[str, Json] | None) -> str:
    if identity is None:
        return "unreadable"
    return f"{identity['bytes']} bytes, SHA-256 {str(identity['sha256'])[:12]}"


def run_candidate(
    problem_path: str | os.PathLike[str],
    assignments: Sequence[str],
    output: str | os.PathLike[str],
    *,
    implementations: Sequence[EvaluatorFactory],
    inputs: Sequence[str] = (),
    timeout: float | None = None,
    warn: Callable[[str], None] | None = None,
) -> RunOutcome:
    """Evaluate the candidate given as ``REFERENCE=VALUE`` strings; ``output`` must not exist.

    :meth:`EvaluationSession.evaluate` in a session of its own: ``inputs`` are ``STAGE=FILE``
    strings, ``timeout`` that of :meth:`EvaluationSession.open`. ``warn`` is called with each of
    the session's warnings once the candidate is known to be valid, before anything is created;
    the warnings are also kept in the summary.
    """
    session = EvaluationSession.open(
        problem_path, implementations, inputs=_parse_inputs(inputs), timeout=timeout
    )
    values = _parse_candidate(session.spec, assignments)
    _validate_candidate(session.spec, values)  # a wrong candidate is reported before the warnings
    if warn is not None:
        for warning in session.warnings:
            warn(warning)
    return session.evaluate(values, output)
