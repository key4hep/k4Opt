"""Candidates of a problem file, evaluated end to end (behind ``k4opt evaluate``).

:class:`EvaluationSession` loads the problem once, binds and checks the evaluator
implementations and records the identity of every file the problem names. For each candidate it
then validates the candidate, creates a new instance of each evaluator and checks its request
for the candidate, checks that no such file changed (nothing is created before this succeeds),
and runs the candidate's tasks in a new run directory through
:func:`k4opt.problem._evaluate._execute_plan`: the geometry task materializes the candidate's
geometry into it, each pipeline stage of each sample runs once and each evaluation measures its
sample, or the candidate, with its new instance. It then assesses the candidate against the
objectives and constraints (:mod:`k4opt.problem._assess`) and writes ``summary.json``, which
:func:`assemble` makes from the task results alone. Every path the record holds of a file in the
run directory is relative to it, so a record stays true wherever the directory is moved; the
run directory itself is recorded once, as ``run_dir``. :func:`run_candidate` evaluates one candidate
given as ``REFERENCE=VALUE`` strings.
"""

from __future__ import annotations

import math
import os
import re
import time
import uuid
import weakref
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import TracebackType

from k4opt import __version__
from k4opt.evaluation import (
    CandidateKey,
    EvaluatorProtocol,
    Measurement,
    Role,
    Scalar,
    evaluator_source,
)
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.geometry import Override
from k4opt.jsonio import write_json
from k4opt.problem._assess import SUMMARY_SCHEMA, Assessment, assess
from k4opt.problem._backends import Backend, InlineBackend, ProcessBackend
from k4opt.problem._evaluate import (
    CandidateRecord,
    _bind_problem,
    _Binding,
    _context,
    _drive,
    _finish_plan,
    _preflight,
    _Prepared,
    _Run,
    _validate_candidate,
)
from k4opt.problem._schedule import Scheduler
from k4opt.problem._stages import STAGE_TYPES, file_identity, input_files, plan_stages
from k4opt.problem._tasks import (
    GEOMETRY,
    Kind,
    Made,
    Task,
    TaskId,
    TaskResult,
    evaluations,
    stage_task,
)
from k4opt.problem._work import (
    SUMMARY_NAME,
    GeometryWork,
    MetricKey,
    TaskOutcome,
    Workspace,
    candidate_externals,
    content,
)
from k4opt.problem._worker import session_spec
from k4opt.problem.config import load_problem
from k4opt.problem.errors import CandidateError, RunError
from k4opt.problem.models import EvaluatorSpec, Parameter, ProblemSpec
from k4opt.revision import code_identity, code_unchanged

LOCAL: Mapping[str, Json] = FrozenMap({"backend": "local", "jobs": 1})
"""How a session executes its tasks by default: one after another, in its own process."""

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
    def events(self) -> dict[str, tuple[object, object]]:
        """``(found, requested)``: the events found in the simulation output of each sample whose
        output was inspected, and those its simulation stage asks for (``None`` for none)."""
        definition = self.summary["problem"]
        assert isinstance(definition, Mapping)
        samples = definition["definition"].get("samples", {})
        events: dict[str, tuple[object, object]] = {}
        for task in self.candidate.plan.of(Kind.STAGE):
            found = self.candidate.results[task.id].provenance.get("events_read")
            if task.id.name == "simulation" and found is not None:
                assert task.id.sample is not None
                requested = samples[task.id.sample]["simulation"].get("events")
                events[task.id.sample] = (found, requested)
        return events


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
    """``SAMPLE/STAGE=FILE`` strings as files by stage label; the session checks the samples,
    stages and files."""
    inputs: dict[str, str] = {}
    for assignment in assignments:
        label, separator, text = assignment.partition("=")
        if not separator or not label or not text:
            raise RunError(f"--input {assignment!r}: expected SAMPLE/STAGE=FILE")
        if label in inputs:
            raise RunError(f"--input {label} is given more than once")
        inputs[label] = text
    return inputs


def _supplied(
    spec: ProblemSpec, inputs: Mapping[str, str | os.PathLike[str]]
) -> dict[TaskId, Path]:
    """The files that stand in for stages, by stage task, as absolute paths; each must be named
    ``SAMPLE/STAGE`` after a sample of ``spec`` and a stage, and be a file."""
    supplied: dict[TaskId, Path] = {}
    samples = ", ".join(spec.samples) or "none"
    for label, given in inputs.items():
        sample, separator, stage = label.partition("/")
        if not separator:
            raise RunError(
                f"input {label!r}: name the sample too, as SAMPLE/{label}; the problem's samples "
                f"are {samples}"
            )
        if sample not in spec.samples:
            raise RunError(f"input {label!r}: unknown sample {sample!r}; the samples are {samples}")
        if stage not in STAGE_TYPES:
            raise RunError(f"input {label!r}: unknown stage; use {', '.join(STAGE_TYPES)}")
        path = Path(os.path.abspath(Path(given).expanduser()))
        if not path.is_file():
            raise RunError(f"input {label}={os.fspath(given)}: {path} is not a file")
        supplied[stage_task(stage, sample)] = path
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


def _roles(roles: Mapping[str, Role] | None) -> dict[str, dict[str, Json]] | None:
    """The roles a combined evaluator declares, each with what it reads; ``None`` if it declares
    none."""
    if roles is None:
        return None
    return {
        name: {
            "many": role.many,
            "stage": role.stage,
            "metrics": tuple(role.metrics),
            "artifacts": tuple(role.artifacts),
        }
        for name, role in roles.items()
    }


def _relative(path: Path, run_dir: Path) -> str:
    """How the record names ``path``, a file in the run directory ``run_dir``: relative to it."""
    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def _artifact(made: Made, run_dir: Path) -> dict[str, object]:
    """A file a task made, as the record names it; ``sha256`` is ``null`` unless it was hashed."""
    artifact = made.artifact
    return {
        "ref": artifact.ref,
        "path": _relative(artifact.path, run_dir),
        "bytes": artifact.bytes,
        "sha256": artifact.sha256,
    }


def _fragment(result: TaskResult) -> dict[str, object]:
    """What every task's part of the summary says first: how it ended, and why."""
    return {"status": result.status, "message": result.message}


def _reads(task: Task, result: TaskResult) -> dict[str, object]:
    """What every task's part of the summary says of the tasks before it: those it reads, and
    those of them that kept it from running."""
    return {
        "reads": [need.label for need in task.needs],
        "blocked_by": [cause.label for cause in result.blocked_by],
    }


def _stage(task: Task, result: TaskResult, run_dir: Path) -> dict[str, object]:
    """One stage's part of the summary: how it ended, how it ran, where, and what it made."""
    provenance = thaw(result.provenance)
    assert isinstance(provenance, dict)
    return (
        _fragment(result)
        | provenance
        | {
            "directory": task.directory,
            "outputs": {name: _artifact(made, run_dir) for name, made in result.made.items()},
            "logs": {role: _relative(path, run_dir) for role, path in result.artifacts.items()},
        }
        | _reads(task, result)
        | {"elapsed_seconds": result.elapsed_seconds}
    )


def _evaluation(
    task: Task, result: TaskResult, entry: EvaluatorSpec, run_dir: Path
) -> dict[str, object]:
    """One evaluation's own part of the summary: nothing of the stages it shares, nor of its
    implementation, which ``implementations`` records once, nor of what its roles read, which is
    recorded by the tasks that made it."""
    record = result.record
    assert record is not None  # every binding has one, a blocked one too
    outcome, request = record.result, record.request
    # what each role of an evaluation of the candidate reads, by the task that made it
    roles = {role: [item.label for item in items] for role, items in entry.roles.items()}
    return {
        "implementation": entry.name,
        **_fragment(result),
        # as the problem wrote them, and as the evaluator received them (defaults filled in,
        # quantities in the declared unit)
        "settings": {"declared": thaw(entry.settings), "used": thaw(request.settings)},
        "collections": dict(request.collections),
        **({} if task.id.sample is not None else {"roles": roles}),
        **_reads(task, result),
        "measurements": [
            {"stage": m.stage, "metric": m.metric} | _measurement(m) for m in outcome.measurements
        ],
        "artifacts": {name: _artifact(made, run_dir) for name, made in result.made.items()},
        "provenance": thaw(outcome.provenance),
        # its own directory, made just before it was called; none if it was not
        "workdir": None if request.workdir is None else task.directory,
        # what it printed while it was called in a worker process; none in the caller's process
        "log": _relative(log, run_dir) if (log := result.artifacts.get("log")) else None,
        "elapsed_seconds": record.elapsed_seconds,
    }


def assemble(
    record_id: str,
    problem_path: Path,
    spec: ProblemSpec,
    values: Mapping[str, Scalar],
    candidate: CandidateRecord,
    assessment: Assessment,
    run_dir: Path,
    taken: Mapping[str, Mapping[str, Json]],
    run: object,
    process: Mapping[str, Json],
    execution: Mapping[str, Json] = LOCAL,
) -> dict[str, object]:
    """The summary of a candidate, made from its task results and nothing else that ran.

    Each task's part (``geometry``; each sample's ``stages`` and ``evaluations``; the
    ``evaluations`` of the candidate) is made from that task's result alone; the candidate's
    ``status``, ``message``, ``required``, ``diagnostics``, ``metrics`` and ``assessment`` derive
    from them and the plan, as
    :meth:`Assessment.from_summary` derives them again.
    ``record_id`` is the record's identity; the rest is what the session holds: the problem,
    what it recorded of each implementation (``taken``: its version, stage, source and metrics)
    and of the process (``process``), and how it executed the tasks (``execution``: provenance,
    which no identity or comparison of records reads).
    """
    plan, results = candidate.plan, candidate.results
    environment = thaw(process["environment"])
    assert isinstance(environment, dict)
    started = [
        argv
        for task in plan.of(Kind.STAGE)
        if isinstance(argv := results[task.id].provenance.get("argv"), (list, tuple)) and argv
    ]
    entries = dict(evaluations(spec.samples, spec.evaluators))
    geometry = results[GEOMETRY]

    def measured(sample: str | None) -> dict[str, object]:
        """The evaluations of ``sample``, or of the candidate, by binding."""
        return {
            task.id.name: _evaluation(task, results[task.id], entries[task.id], run_dir)
            for task in plan.of(Kind.EVALUATOR)
            if task.id.sample == sample
        }

    made = thaw(geometry.provenance)
    assert isinstance(made, dict)
    return {
        "schema": SUMMARY_SCHEMA,
        "id": record_id,  # the record's identity, wherever it is copied to
        "run": run,  # what the caller evaluates it for (a study and trial, say), null if nothing
        # where it was made, which the paths of its files are relative to; it may move since
        "run_dir": str(run_dir),
        "problem": {"path": str(problem_path), "name": spec.name, "definition": spec.to_dict()},
        "candidate": dict(values),
        "candidate_key": [list(part) for part in candidate.candidate_key],
        "status": candidate.status,
        "message": candidate.message,
        # the tasks the status is decided by: what the objectives, constraints and the evaluators
        # marked required read; the others are diagnostics
        "required": [task.id.label for task in plan.tasks if task.required],
        # each diagnostic that did not succeed, which the status leaves out
        "diagnostics": [
            {"task": result.id.label, "status": result.status, "message": result.message}
            for result in candidate.diagnostics
        ],
        # False if a required stage file was supplied that was not made from this candidate
        "represents_candidate": candidate.represents_candidate,
        "supplied": sorted(candidate.supplied),
        "warnings": list(candidate.warnings),
        # the objective and constraint metrics, each from the binding it names
        "metrics": [
            {"evaluator": ref.evaluator, "sample": ref.sample, "stage": ref.stage}
            | {"metric": ref.metric, "label": ref.label}
            | _measurement(candidate.value(ref))
            for ref in spec.references
        ],
        # the objectives and the constraints judged, from those metrics
        "assessment": assessment.to_dict(),
        # each implementation once, however many bindings it has
        "implementations": {name: thaw(entry) for name, entry in taken.items()},
        # each sample's stages, each recorded once, and the evaluations of the sample
        "samples": {
            sample: {
                "stages": {
                    task.id.name: _stage(task, results[task.id], run_dir)
                    for task in plan.of(Kind.STAGE)
                    if task.id.sample == sample
                },
                "evaluations": measured(sample),
            }
            for sample in spec.samples
        },
        # the evaluations of the candidate, of no sample
        "evaluations": measured(None),
        "elapsed_seconds": candidate.elapsed_seconds,
        "geometry": {"status": geometry.status}
        | made
        | {"elapsed_seconds": geometry.elapsed_seconds},
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
                "ddsim": started[0][0] if started else None,
                # where and how the tasks ran: provenance, never part of what was evaluated
                "execution": thaw(execution),
            },
        },
    }


@dataclass(frozen=True, slots=True)
class RunRequest:
    """A candidate to evaluate into the new directory ``output``, as
    :meth:`EvaluationSession.evaluate` takes it; ``run`` is recorded as the summary's ``run``."""

    candidate: Mapping[str, Scalar]
    output: str | os.PathLike[str]
    run: Mapping[str, Json] | None = None


class Evaluations:
    """The candidates of :meth:`EvaluationSession.evaluate_each`, each ``(request, outcome)`` as it
    ends. :meth:`close` (or leaving its ``with``) stops whatever still runs; it is never left to
    the garbage collector."""

    def __init__(
        self,
        session: EvaluationSession,
        generator: Generator[tuple[RunRequest, RunOutcome | Exception]],
    ) -> None:
        self._session: EvaluationSession | None = session
        self._generator = generator

    def __iter__(self) -> Evaluations:
        return self

    def __next__(self) -> tuple[RunRequest, RunOutcome | Exception]:
        try:
            return next(self._generator)
        except BaseException:  # it ended: nothing of it runs any more
            self._release()
            raise

    def close(self) -> None:
        """Stop whatever still runs and give the session back; closing again does nothing."""
        try:
            self._generator.close()
        finally:
            self._release()

    def _release(self) -> None:
        # here, not only in the generator: one closed before its first item never ran its body
        if self._session is not None and self._session._open is self:
            self._session._open = None
        self._session = None

    def __enter__(self) -> Evaluations:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class EvaluationSession:
    """Candidates of one problem, evaluated with the same evaluators: one task after another in
    the caller's thread, or up to ``jobs`` tasks at once in worker processes (:meth:`open`).

    :meth:`open` loads the problem file once, so later edits to it do not reach the session,
    binds the evaluator implementations to it and checks what does not depend on a candidate.
    It also records the identity (size and SHA-256) of every file the stages read: a stage given
    as a ``file``, a file a ddsim option names, a file the caller supplies, and what the
    geometry's external references read. Every candidate gets a new instance of each evaluator,
    so no state of one reaches another candidate; it must declare what the first instance,
    created when the session opened, declared (its class, name, version, stage, settings,
    collections, metrics, artifacts and roles).
    Each candidate is then checked afresh, by :meth:`check` and again by :meth:`evaluate`: its
    values against the parameter domains, the ``check()`` of each new instance with a request
    made for this candidate, and that none of those files, the environment or k4opt's
    code changed, which :meth:`evaluate` checks again before it writes the record. A candidate
    whose values change a ``ref`` can name an external file of its own; what it reads is
    recorded, and must not change while the candidate runs nor between the candidates that name
    it. A session coordinates its candidates from one thread: it is not thread-safe. Close it
    (:meth:`close`, or ``with``) to stop its worker processes.
    """

    __slots__ = (
        "__weakref__",
        "_backend",
        "_binding",
        "_context",
        "_execution",
        "_externals",
        "_files",
        "_inputs",
        "_open",
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
        inputs: Mapping[TaskId, Path],
        files: Mapping[str, tuple[Path, Mapping[str, Json] | None]],
        externals: Mapping[str, Json],
        taken: Mapping[str, Mapping[str, Json]],
        warnings: tuple[str, ...],
        process: Mapping[str, Json],
        timeout: float | None,
        jobs: int = 1,
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
        # what every task needs, wherever it runs
        self._context = _context(spec, binding, timeout, externals)
        self._execution: Mapping[str, Json] = FrozenMap({"backend": "local", "jobs": jobs})
        self._open: Evaluations | None = None
        self._backend: ProcessBackend | None = None
        if jobs > 1:  # one worker started at once: one that cannot work refuses here
            environment = process["environment"]
            assert isinstance(environment, Mapping)
            described = session_spec(self._context, binding.evaluations, environment)
            backend = self._backend = ProcessBackend(described, jobs)
            weakref.finalize(self, backend.close)  # a backstop: close() stops the workers
            try:
                backend.start()
            except BaseException:
                backend.close()
                raise

    @classmethod
    def open(
        cls,
        problem_path: str | os.PathLike[str],
        implementations: Sequence[type[EvaluatorProtocol]],
        *,
        inputs: Mapping[str, str | os.PathLike[str]] | None = None,
        timeout: float | None = None,
        jobs: int = 1,
    ) -> EvaluationSession:
        """Load the problem file ``problem_path`` to evaluate its candidates with
        ``implementations``.

        ``implementations`` (evaluator classes, as :func:`~k4opt.evaluation.load_evaluator`
        returns them) implement the problem's ``evaluators``, in any order; one instance of each
        is created now to bind it to the problem, and a new one for every evaluation of every
        candidate. ``inputs`` maps a stage of a sample, ``SAMPLE/STAGE``, to a file that stands
        in for it (``k4opt evaluate --input``): that stage does not run, nor do the stages of the
        sample only it needs, and a supplied stage that the parameters change was not made from
        the candidate, which every outcome says
        (``represents_candidate``). ``timeout`` is how many seconds a stage that runs a program
        (ddsim) may take: one that runs longer is killed, and its candidate fails with the logs
        kept. ``jobs`` is how many tasks (simulations, evaluations, geometries) may run at once,
        of one candidate and of several (:meth:`evaluate_each`); with more than one, each runs
        in a worker process that loads the implementations again by their references, so they
        must be loaded by :func:`~k4opt.evaluation.load_evaluator`, and one worker is started
        and checked now. Close a session (:meth:`close`, or ``with``) to stop its workers.
        Raises a :class:`~k4opt.errors.K4OptError` for an invalid problem, evaluator,
        stage or input (an :class:`~k4opt.evaluation.EvaluationContractError` for an
        implementation that cannot be created, too), and :class:`~k4opt.problem.RunError` if
        k4opt's own files changed since this process first identified them
        (:func:`k4opt.revision.code_unchanged`).
        """
        path = Path(problem_path).expanduser()  # expanded once: loaded and recorded alike
        limit = _timeout(timeout)
        _jobs(jobs)
        spec = load_problem(path)
        supplied = _supplied(spec, inputs or {})
        binding = _bind_problem(spec, implementations, supplied)
        named: dict[str, Path] = {}
        for sample, context in binding.contexts.items():
            given = {task.name for task in supplied if task.sample == sample}
            targets = binding.targets[sample]
            for stage in given:  # read by an evaluator, given the other supplied stages
                if stage not in plan_stages(*targets, provided=given - {stage}):
                    raise RunError(
                        f"input {sample}/{stage}: no evaluator of the problem reads that stage of "
                        f"{sample}"
                    )
            named |= input_files(context, *targets, provided=given)
        named |= {f"supplied.{task.sample}.{task.name}": file for task, file in supplied.items()}
        # an unreadable file has no identity: its stage reports it, and a record still names it
        files = {label: (file, _readable_identity(file)) for label, file in named.items()}
        assert spec._source is not None
        externals = freeze(spec._source.external_files(spec._catalog), "geometry", RunError)
        assert isinstance(externals, Mapping)
        taken = {}
        for bound in binding.evaluators:  # what the first instance of each declared
            declaration = bound.declaration
            if declaration.name in taken:
                continue
            data = {
                "version": declaration.version,
                "stage": declaration.stage,
                "source": evaluator_source(declaration.cls),
                # each with the unit every measurement of it carries, whatever the status
                "metrics": _declared(declaration.metrics),
                # the names of the files a valid result of it returns; null if it declares none
                "artifacts": None if declaration.artifacts is None else list(declaration.artifacts),
                # what each role of a combined evaluator reads; null for an evaluator of no roles
                "roles": _roles(declaration.roles),
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
            jobs,
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

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """The evaluations that no objective, constraint or required evaluator reads, by label
        (``zz/calo.occupancy``), in plan order: each is run and recorded, but whatever its
        result, the candidate's status does not depend on it."""
        return tuple(
            task.id.label for task in self._binding.plan.of(Kind.EVALUATOR) if not task.required
        )

    @property
    def notice(self) -> str | None:
        """What to tell someone who starts the session of its :attr:`diagnostics`, ``None``
        without one: a check meant to veto candidates that is not marked ``required: true``
        would otherwise veto nothing, unnoticed."""
        if not self.diagnostics:
            return None
        return (
            f"diagnostics (recorded, do not affect the result): {', '.join(self.diagnostics)}; "
            "mark an evaluator required: true if its result must count"
        )

    def identity(self) -> dict[str, object]:
        """What this session evaluates with, as plain JSON-compatible data.

        ``problem``: the path and normalized definition; ``inputs``: the path, size and SHA-256
        of each file the stages read, by where it is named (``samples.gamma.generation.file``) or
        supplied (``supplied.gamma.simulation``), a file that could not be read with its path
        only; a file that several samples read is named for each, and hashed once;
        ``implementations``: the version, stage, source identity and declared metrics, artifacts
        and roles of each implementation the problem binds, by name, in first-bound order;
        ``geometry``: the compact file, the source root, the fingerprint of the unmodified
        geometry and what its external references read
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
            "implementations": {name: thaw(entry) for name, entry in self._taken.items()},
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

    def _check_read(self, externals: Mapping[str, Mapping[str, Json]]) -> None:
        """Raise :class:`~k4opt.problem.RunError` if a reference only candidates name reads other
        files than it did for an earlier candidate of the session."""
        for ref, entry in externals.items():
            if ref in self._read and self._read[ref] != content(entry):
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
        with self.evaluate_each([RunRequest(candidate, output, run)]) as evaluations:
            for _, result in evaluations:
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError("the candidate was never evaluated")  # pragma: no cover

    def evaluate_each(self, requests: Iterable[RunRequest]) -> Evaluations:
        """Evaluate the candidate of each of ``requests``, as :meth:`evaluate` does, and give
        each ``(request, outcome)`` as its candidate ends; the outcome is the exception that
        stopped it instead, if one did (nothing else is then written for it).

        The next request is taken from ``requests`` only once the candidates under way leave a
        job idle (none of their tasks is ready to take it), and every candidate that ended was
        yielded before, so whatever makes the requests (an optimizer) learns from every outcome
        it can before it proposes again. With one job, that is one candidate at a time, each
        yielded before the next is taken; with more, the candidates end in any order, each
        with its own record. Close what this returns (``with``) when done with it: that stops
        whatever still runs, as leaving the loop early does not by itself. One at a time per
        session.
        """
        if self._open is not None:
            raise RunError("the session is evaluating already; close that evaluation first")
        evaluations = Evaluations(self, self._evaluations(iter(requests)))
        self._open = evaluations
        return evaluations

    def close(self) -> None:
        """Stop whatever an open :meth:`evaluate_each` still runs, and the session's worker
        processes; a later evaluation starts new ones."""
        try:
            if self._open is not None:
                self._open.close()
        finally:
            if self._backend is not None:
                self._backend.close()

    def __enter__(self) -> EvaluationSession:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _evaluations(
        self, requests: Iterator[RunRequest]
    ) -> Generator[tuple[RunRequest, RunOutcome | Exception]]:
        runs: dict[str, _Run] = {}
        backend: Backend = self._backend or InlineBackend(
            self._context, lambda workspace, task: _prepared(runs[workspace]).evaluators[task][1]
        )

        def pull() -> _Run | None:
            request = next(requests, _END)
            if request is _END:
                return None
            if not isinstance(request, RunRequest):  # the caller's mistake, not a candidate's
                raise RunError(f"expected a RunRequest, got {type(request).__name__}")
            try:
                return self._start(request)
            except Exception as exc:  # before anything was created
                return _Run(Workspace("", None), None, error=exc, request=request)

        driven = _drive(Scheduler(backend.capacity), backend, runs, pull, self._checked)
        try:
            for run in driven:
                request = run.request
                assert isinstance(request, RunRequest)
                if run.error is not None:
                    yield request, run.error
                    continue
                try:
                    outcome = self._finish(run)
                except Exception as exc:
                    yield request, exc
                    continue
                yield request, outcome
        finally:
            driven.close()
            if backend is not self._backend:  # the session's workers serve its next evaluation
                backend.close()

    def _start(self, request: RunRequest) -> _Run:
        """A candidate checked and prepared as :meth:`check` does, with its own new run directory
        (not yet created) as its workspace."""
        reference = _run_reference(request.run)
        started = time.perf_counter()
        values, key, prepared, overrides = self._prepare(request.candidate)
        run_dir = Path(os.path.abspath(request.output))
        if run_dir.exists() or run_dir.is_symlink():
            raise RunError(f"output {run_dir} already exists; choose a new directory")
        return _Run(
            Workspace(str(uuid.uuid4()), run_dir),
            prepared,
            key,
            self._inputs,
            GeometryWork(tuple(overrides)),
            started,
            request=request,
            values=values,
            reference=reference,
        )

    def _checked(self, run: _Run, outcome: TaskOutcome, result: TaskResult) -> None:
        """What is checked of each result as it comes: that the references only the candidate's
        geometry names read what they read for the candidates already recorded."""
        if result.id != GEOMETRY:
            return
        run.own_refs = outcome.own_refs
        try:
            self._check_read(_externals(result))
        except RunError as exc:
            run_dir = run.workspace.root
            raise RunError(f"{exc} (in {run_dir}, which has no {SUMMARY_NAME})") from None

    def _finish(self, run: _Run) -> RunOutcome:
        """The record of a candidate whose tasks all ended, checked and published.

        Nothing is published, and nothing is remembered of the candidate, if what the session
        verifies changed while it ran, a file a task made is no longer what its record says, or
        a reference only its geometry names reads other files than for a candidate recorded
        meanwhile. Those checks, the record and what the session remembers of it come one
        candidate after another."""
        spec = self._spec
        source = spec._source
        assert source is not None
        run_dir = run.workspace.root
        assert run_dir is not None
        record = _finish_plan(run)
        geometry = record.results[GEOMETRY]
        externals = _externals(geometry)
        try:
            self._verify()  # the record names what the stages and evaluators read
            changed = record.changed()
            if changed:
                raise RunError(
                    f"{', '.join(changed)} changed after it was made, so the record could not "
                    "say what it holds"
                )
            now = candidate_externals(
                source.compact, run_dir / GEOMETRY.name, run.own_refs, self._externals
            )
            if _plain(now) != _plain(externals):
                refs = dict.fromkeys([*externals, *now])
                changed = [ref for ref in refs if now.get(ref) != externals.get(ref)]
                raise RunError(
                    f"geometry: {', '.join(changed)} does not read the files it read when the "
                    "candidate started"
                )
            # again, now: a candidate recorded while this one ran may have read other files
            self._check_read(externals)
        except RunError as exc:
            raise RunError(
                f"{exc} (while the candidate in {run_dir} ran, which has no {SUMMARY_NAME})"
            ) from None
        assessment = assess(spec, record)
        summary = assemble(
            run.workspace.id,
            self._path,
            spec,
            run.values,
            record,
            assessment,
            run_dir,
            self._taken,
            run.reference,
            self._process,
            self._execution,
        )
        try:
            write_json(run_dir / SUMMARY_NAME, summary)
        except OSError as exc:
            raise RunError(f"cannot write summary in {run_dir}: {exc}") from exc
        self._read |= {ref: content(entry) for ref, entry in externals.items()}
        fingerprint = geometry.provenance["fingerprint"]
        assert isinstance(fingerprint, str)
        return RunOutcome(record, summary, run_dir, fingerprint, assessment)

    def _prepare(
        self, candidate: Mapping[str, Scalar]
    ) -> tuple[dict[str, Scalar], CandidateKey, _Prepared, list[Override]]:
        """The checked candidate, its key, what its preflight prepared (its own evaluator
        instances and the requests made for it) and its overrides."""
        values, key = _validate_candidate(self._spec, candidate)
        self._verify()  # first: a file that is gone changed, more than it is missing
        prepared = _preflight(self._binding, values)
        overrides = _overrides(self._spec, values)
        return values, key, prepared, overrides


_END = object()  # what the requests give once there are none left; no request is it


def _prepared(run: _Run) -> _Prepared:
    assert run.prepared is not None
    return run.prepared


def _externals(geometry: TaskResult) -> dict[str, dict[str, Json]]:
    """What the references only the candidate's geometry names read, as its geometry task
    recorded it."""
    externals = thaw(geometry.provenance.get("candidate_externals") or {})
    assert isinstance(externals, dict)
    return externals


def _plain(value: object) -> object:
    """``value`` as plain JSON data, as a record holds it."""
    return thaw(freeze(value, "geometry", RunError))


def _jobs(value: int) -> None:
    """:class:`RunError` unless ``value`` is a positive number of jobs."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunError(f"jobs must be a positive integer, got {value!r}")


def _described(identity: Mapping[str, Json] | None) -> str:
    if identity is None:
        return "unreadable"
    return f"{identity['bytes']} bytes, SHA-256 {str(identity['sha256'])[:12]}"


def run_candidate(
    problem_path: str | os.PathLike[str],
    assignments: Sequence[str],
    output: str | os.PathLike[str],
    *,
    implementations: Sequence[type[EvaluatorProtocol]],
    inputs: Sequence[str] = (),
    timeout: float | None = None,
    jobs: int = 1,
    warn: Callable[[str], None] | None = None,
) -> RunOutcome:
    """Evaluate the candidate given as ``REFERENCE=VALUE`` strings; ``output`` must not exist.

    :meth:`EvaluationSession.evaluate` in a session of its own: ``inputs`` are
    ``SAMPLE/STAGE=FILE`` strings, ``timeout`` and ``jobs`` those of
    :meth:`EvaluationSession.open`. ``warn``
    is called with each of the session's warnings once the candidate is known to be valid, before
    anything is created, and with its :attr:`~EvaluationSession.notice` of diagnostics; the
    warnings are also kept in the summary.
    """
    with EvaluationSession.open(
        problem_path, implementations, inputs=_parse_inputs(inputs), timeout=timeout, jobs=jobs
    ) as session:
        values = _parse_candidate(session.spec, assignments)
        # a wrong candidate is reported before the warnings
        _validate_candidate(session.spec, values)
        if warn is not None:
            for warning in session.warnings:
                warn(warning)
            if session.notice is not None:
                warn(session.notice)
        return session.evaluate(values, output)
