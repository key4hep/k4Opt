"""Candidates of a problem file, evaluated end to end (behind ``k4opt evaluate``).

:class:`EvaluationSession` loads the problem once, binds and checks the evaluator
implementations (:mod:`k4opt.problem._binding`) and records the identity of every file the
problem names. For each candidate it then validates the candidate, creates a new instance of each
evaluator and checks its request for the candidate, checks that no such file changed (nothing is
created before this succeeds), and runs the candidate's tasks in a new run directory
(:mod:`k4opt.problem._evaluate`): the geometry task materializes the candidate's geometry into
it, each pipeline stage of each sample runs once and each evaluation measures its sample, or the
candidate, with its new instance. It then assesses the candidate against the objectives and
constraints (:mod:`k4opt.problem._assess`) and writes its ``summary.json``
(:mod:`k4opt.problem._summary`). :func:`run_candidate` evaluates one candidate given as
``REFERENCE=VALUE`` strings.
"""

from __future__ import annotations

import math
import os
import time
import uuid
import weakref
from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal

from k4opt import __version__
from k4opt.evaluation import (
    CandidateKey,
    EvaluatorProtocol,
    Scalar,
)
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.geometry import Override
from k4opt.jsonio import write_json
from k4opt.problem._assess import Assessment, assess
from k4opt.problem._backends import Backend, InlineBackend, ProcessBackend
from k4opt.problem._binding import (
    _bind_problem,
    _Binding,
    _context,
    _implementations,
    _preflight,
    _Prepared,
    task_cpus,
)
from k4opt.problem._candidate import (
    CandidateRecord,
    _overrides,
    _parse_candidate,
    _validate_candidate,
)
from k4opt.problem._cpus import AUTO, usable_cpus
from k4opt.problem._evaluate import Activity, _drive, _finish_plan, _Run
from k4opt.problem._files import readable_identity
from k4opt.problem._schedule import Scheduler
from k4opt.problem._stages import STAGE_TYPES, input_files, plan_stages
from k4opt.problem._summary import assemble
from k4opt.problem._tasks import (
    GEOMETRY,
    Kind,
    TaskId,
    TaskResult,
    stage_task,
)
from k4opt.problem._work import (
    SUMMARY_NAME,
    GeometryWork,
    TaskOutcome,
    Workspace,
    candidate_externals,
    content,
)
from k4opt.problem._worker import session_spec
from k4opt.problem.config import load_problem
from k4opt.problem.errors import RunError
from k4opt.problem.models import ProblemSpec
from k4opt.revision import code_identity, code_unchanged


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
        jobs: int | Literal["auto"] = 1,
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
        and checked now. ``"auto"`` is as many jobs as the processors this process may use (its
        affinity, at most its cgroup's quota). A simulation on several threads
        (``numberOfThreads``) holds as many jobs while it runs. Close a session
        (:meth:`close`, or ``with``) to stop its workers.
        Raises a :class:`~k4opt.errors.K4OptError` for an invalid problem, evaluator,
        stage or input (an :class:`~k4opt.evaluation.EvaluationContractError` for an
        implementation that cannot be created, too), and :class:`~k4opt.problem.RunError` if
        k4opt's own files changed since this process first identified them
        (:func:`k4opt.revision.code_unchanged`).
        """
        path = Path(problem_path).expanduser()  # expanded once: loaded and recorded alike
        limit = _timeout(timeout)
        jobs, usable = _jobs(jobs), usable_cpus()
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
        files = {label: (file, readable_identity(file)) for label, file in named.items()}
        assert spec._source is not None
        externals = freeze(spec._source.external_files(spec._catalog), "geometry", RunError)
        assert isinstance(externals, Mapping)
        taken = _implementations(binding)
        # before anything runs, so that an edit to the checkout during a long run is not
        # recorded as the code that ran, nor a later change of the environment as the session's
        environment = {name: os.environ.get(name) for name in ("KEY4HEP_STACK", "K4GEO")}
        process = freeze(
            {"k4opt": code_identity(), "environment": environment}, "process", RunError
        )
        _check_code()
        assert isinstance(process, Mapping)
        crowded = []
        for task in binding.plan.tasks:
            cpus = task_cpus(binding, task)
            beyond = f"more than the {usable} processors this process may use"
            if cpus > jobs:
                crowded.append(
                    f"{task.id.label} simulates with {cpus} threads, more than the {jobs} "
                    f"job(s) of the session: it runs alone, on {cpus} processors"
                    + ("" if cpus <= usable else f", {beyond}")
                )
            elif cpus > usable:  # within the jobs, which are then too many themselves
                crowded.append(f"{task.id.label} simulates with {cpus} threads, {beyond}")
        if jobs > usable:
            crowded.append(
                f"{jobs} jobs, more than the {usable} processors this process may use: tasks "
                "will share processors"
            )
        return cls(
            path.absolute(),
            spec,
            binding,
            supplied,
            files,
            externals,
            taken,
            (*binding.warnings, *crowded),
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
        simulation's output decides, so they do not stop an evaluation (each summary keeps them
        too). Then how the session's jobs suit what it runs: a simulation on more threads than
        there are jobs, more jobs than processors (how it ran: no summary keeps these)."""
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
            now = readable_identity(file)
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

    def evaluate_each(
        self,
        requests: Iterable[RunRequest],
        *,
        activity: Callable[[Activity], None] | None = None,
    ) -> Evaluations:
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
        session. ``activity`` is told what runs (:class:`~k4opt.problem.Activity`) whenever that
        changes, in this thread.
        """
        if self._open is not None:
            raise RunError("the session is evaluating already; close that evaluation first")
        evaluations = Evaluations(self, self._evaluations(iter(requests), activity))
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
        self, requests: Iterator[RunRequest], activity: Callable[[Activity], None] | None
    ) -> Generator[tuple[RunRequest, RunOutcome | Exception]]:
        runs: dict[str, _Run] = {}
        backend: Backend = self._backend or InlineBackend(
            self._context, lambda workspace, task: _prepared(runs[workspace]).instances[task]
        )

        def pull() -> _Run | None:
            request = next(requests, _END)
            if request is _END:
                return None
            if not isinstance(request, RunRequest):  # the caller's mistake, not a candidate's
                raise RunError(f"expected a RunRequest, got {type(request).__name__}")
            try:
                return self._start(request, {run.workspace.root for run in runs.values()})
            except Exception as exc:  # before anything was created
                return _Run(Workspace("", None), None, error=exc, request=request)

        driven = _drive(Scheduler(backend.capacity), backend, runs, pull, self._checked, activity)
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

    def _start(self, request: RunRequest, under_way: Collection[Path | None] = ()) -> _Run:
        """A candidate checked and prepared as :meth:`check` does, with its own new run directory
        (not yet created) as its workspace; the directories of the candidates ``under_way``,
        which their geometry may not have made yet, count as existing, also when named through
        a symlink."""
        reference = _run_reference(request.run)
        started = time.perf_counter()
        values, key, prepared, overrides = self._prepare(request.candidate)
        if self._backend is not None:  # its workers check and evaluate instances of their own
            prepared = prepared.released()
        run_dir = Path(os.path.abspath(request.output))
        taken = {root.resolve() for root in under_way if root is not None}
        if run_dir.exists() or run_dir.is_symlink() or run_dir.resolve() in taken:
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


def _jobs(value: int | Literal["auto"]) -> int:
    """The number of jobs ``value`` says: itself, or for ``"auto"`` the processors this process
    may use. :class:`RunError` unless it is a positive integer or ``"auto"``."""
    if value == AUTO:
        return usable_cpus()
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunError(f"jobs must be a positive integer or 'auto', got {value!r}")
    return value


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
    jobs: int | Literal["auto"] = 1,
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
