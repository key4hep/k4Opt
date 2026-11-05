"""Candidates of a problem file, evaluated end to end.

:class:`EvaluationSession` loads the problem once, binds the evaluator classes by what they
declare (:mod:`k4opt.execution._binding`) and takes what it evaluates with, the identity of every
file the problem names included (:mod:`k4opt.execution._baseline`). For each candidate it then
validates the candidate, makes the request of each evaluation and asks each evaluator class's
``check`` about it (:mod:`k4opt.execution._preflight`), which may reject the candidate
(:class:`~k4opt.execution.Rejected`), and checks that none of what it evaluates with changed;
nothing is created before this succeeds. It then admits the candidate into a new run directory of
its own and runs its tasks there (:mod:`k4opt.execution._drive`): the geometry task materializes
the candidate's geometry into it, each pipeline stage of each sample runs once and each
evaluation measures its sample, or the candidate, with a new instance of its evaluator created
where it runs. Its record is then checked and published (:mod:`k4opt.execution._publish`).
:mod:`k4opt.execution._command` evaluates one candidate given as ``NAME=VALUE`` strings
(``k4opt evaluate``).
"""

from __future__ import annotations

import json
import os
import time
import uuid
import weakref
from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal

from k4opt.evaluation import (
    CandidateKey,
    CandidateRejected,
    EvaluatorProtocol,
    Scalar,
)
from k4opt.execution._backends import Backend, InlineBackend
from k4opt.execution._baseline import Baseline, candidate_reads
from k4opt.execution._binding import _bind_problem, _Binding, _context, task_cpus
from k4opt.execution._cpus import AUTO, INLINE, usable_cpus, usable_memory
from k4opt.execution._drive import WAIT, Activity, Wait, _drive, _Run
from k4opt.execution._preflight import Rejected, _preflight, _Prepared
from k4opt.execution._process import ProcessBackend
from k4opt.execution._publish import RunOutcome, publish
from k4opt.execution._reuse import Reuse
from k4opt.execution._schedule import Scheduler
from k4opt.execution._stages import DryRuns, input_files, try_reconstruction
from k4opt.execution._tasks import GeometryWork, TaskOutcome, Workspace
from k4opt.execution._worker import session_spec
from k4opt.execution.errors import ProvenanceError, RunError
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.geometry import Override
from k4opt.geometry import Parameter as GeometryParameter
from k4opt.problem.candidate import (
    Configuration,
    configure,
    geometry_overrides,
    validate_candidate,
)
from k4opt.problem.config import resolve_problem
from k4opt.problem.errors import CandidateError
from k4opt.problem.models import ProblemSpec, ResolvedProblem
from k4opt.problem.plan import GEOMETRY, Kind, Task, TaskId, stage_task
from k4opt.problem.stages import STAGE_TYPES, executors, plan_stages
from k4opt.record.read import SUMMARY_SCHEMA
from k4opt.record.results import DONE, TaskResult
from k4opt.record.signature import (
    SIGNATURE,
    geometry_identity,
    given_identity,
    signed,
    stage_document,
)
from k4opt.record.write import SUMMARY_NAME


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


def _named(binding: _Binding, supplied: Mapping[TaskId, Path]) -> dict[str, Path]:
    """Every file the stages of the session read, by where it is named
    (:func:`~k4opt.execution._stages.input_files`, ``supplied.<sample>.<stage>``), as the
    directories the problem names hold them now."""
    named: dict[str, Path] = {}
    for sample, context in binding.contexts.items():
        given = {task.name for task in supplied if task.sample == sample}
        named |= input_files(context, *binding.targets[sample], provided=given)
    named |= {f"supplied.{task.sample}.{task.name}": file for task, file in supplied.items()}
    return named


def _try_reconstructions(problem: ResolvedProblem, binding: _Binding, runs: DryRuns) -> None:
    """Raise :class:`~k4opt.evaluation.EvaluationContractError` if the job of a reconstruction
    that runs does not take its arguments (:func:`~k4opt.execution._stages.try_reconstruction`):
    before anything is simulated, so that a misspelt argument costs a second, not a simulation."""
    for task in binding.plan.of(Kind.STAGE):
        if task.id.name == "reconstruction" and task.given is None and task.id.sample is not None:
            try_reconstruction(binding.contexts[task.id.sample], problem.spec.parameters, runs)


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
    # the same candidate on other events: every seed derived (:func:`k4opt.problem.configure`)
    replicate: int = 0
    # the level of a nested optimization it is evaluated for, whose criteria judge it
    level: int = 0


class Evaluations:
    """The candidates of :meth:`EvaluationSession.evaluate_each`, each ``(request, outcome)`` as it
    ends. :meth:`close` (or leaving its ``with``) stops whatever still runs; it is never left to
    the garbage collector."""

    def __init__(
        self,
        session: EvaluationSession,
        generator: Generator[tuple[RunRequest, RunOutcome | Rejected | Exception]],
    ) -> None:
        self._session: EvaluationSession | None = session
        self._generator = generator

    def __iter__(self) -> Evaluations:
        return self

    def __next__(self) -> tuple[RunRequest, RunOutcome | Rejected | Exception]:
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
    the caller's thread, or as many at once as it has worker processes (:meth:`open`).

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
        "_baseline",
        "_binding",
        "_context",
        "_dry_runs",
        "_execution",
        "_inputs",
        "_open",
        "_reuse",
        "_warnings",
    )

    def __init__(
        self,
        baseline: Baseline,
        binding: _Binding,
        inputs: Mapping[TaskId, Path],
        warnings: tuple[str, ...],
        workers: int | None = None,
        dry_runs: DryRuns | None = None,
    ) -> None:
        """Use :meth:`open`."""
        self._baseline = baseline
        self._binding = binding
        problem = baseline.problem
        self._dry_runs = dry_runs or DryRuns(problem.source.root / problem.source.compact)
        weakref.finalize(self, self._dry_runs.close)  # a backstop: close() removes its scratch
        self._inputs = FrozenMap(inputs)
        self._warnings = warnings
        # what every task needs, wherever it runs
        self._context = _context(
            problem.spec, binding, baseline.externals, problem.source, problem.catalog
        )
        execution: dict[str, Json] = {"backend": "inline"}
        if workers is not None:
            execution = {"backend": "process", "workers": workers}
        self._execution: Mapping[str, Json] = FrozenMap(execution)
        self._open: Evaluations | None = None
        self._reuse = Reuse()
        self._backend: ProcessBackend | None = None
        if workers is not None:  # one worker started at once: one that cannot work refuses here
            environment = baseline.process["environment"]
            assert isinstance(environment, Mapping)
            described = session_spec(self._context, binding.evaluations, environment)
            backend = self._backend = ProcessBackend(described, workers)
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
        workers: int | Literal["auto", "inline"] = "inline",
    ) -> EvaluationSession:
        """Load the problem file ``problem_path`` to evaluate its candidates with
        ``implementations``.

        ``implementations`` (evaluator classes, as :func:`~k4opt.evaluation.load_evaluator`
        returns them) implement the problem's ``evaluators``, in any order; each is bound to the
        problem by what its class declares, without creating it, and every evaluation of every
        candidate creates one new instance where it runs. ``inputs`` maps a stage of a sample,
        ``SAMPLE/STAGE``, to a file that stands in for it (``k4opt evaluate --input``): that stage
        does not run, nor do the stages of the sample only it needs, and a supplied stage that the
        parameters change was not made from the candidate, which every outcome says
        (``represents_candidate``). A stage or an evaluation that runs longer than the problem's
        ``timeout`` for it is stopped and fails, its candidate with it, and its logs are kept.
        ``workers`` says where the tasks (geometries, simulations, evaluations) run: ``"inline"``,
        in this process, one after another, with the implementations as given (classes defined in a
        script or a notebook included); or a number of worker processes, each running one task at a
        time, of one candidate and of several (:meth:`evaluate_each`), so that a crash or an OOM
        kill of an evaluator fails its task, not the session. Workers load the implementations again
        by their references, so they must be loaded by :func:`~k4opt.evaluation.load_evaluator` (a
        :class:`~k4opt.execution.RunError` otherwise, never a fallback to ``"inline"``); one worker
        is started and checked now. ``"auto"`` is as many workers as the processors this process may
        use (its affinity, at most its cgroup's quota). A simulation on several threads
        (``numberOfThreads``) holds as many workers while it runs. Each evaluator class's ``check``
        is still called in this process, without a time limit, before anything of its candidate is
        created. Inline, nothing stops the Python of a task, so a problem that limits its
        evaluations (its ``timeout``) needs workers to evaluate candidates, though not to check
        them (:meth:`check_limits`); a stage's limit stops the program the stage runs. Close a
        session (:meth:`close`, or ``with``) to stop its workers.
        Raises a :class:`~k4opt.errors.K4OptError` for an invalid problem, evaluator,
        stage or input (an :class:`~k4opt.evaluation.EvaluationContractError` for an evaluator
        class that declares anything but plain data, too), and
        :class:`~k4opt.execution.ProvenanceError` if k4opt's own files changed since this process
        first identified them (:func:`k4opt.revision.code_unchanged`).
        """
        path = Path(problem_path).expanduser()  # expanded once: loaded and recorded alike
        count, usable = _workers(workers), usable_cpus()
        problem = resolve_problem(path)
        spec = problem.spec
        supplied = _supplied(spec, inputs or {})
        binding = _bind_problem(
            spec, implementations, supplied, directory=problem.directory, catalog=problem.catalog
        )
        for sample in binding.contexts:
            given = {task.name for task in supplied if task.sample == sample}
            targets = binding.targets[sample]
            for stage in given:  # read by an evaluator, given the other supplied stages
                if stage not in plan_stages(*targets, provided=given - {stage}):
                    raise RunError(
                        f"input {sample}/{stage}: no evaluator of the problem reads that stage of "
                        f"{sample}"
                    )
        runs = DryRuns(problem.source.root / problem.source.compact)
        try:
            _try_reconstructions(problem, binding, runs)
            baseline = Baseline.take(problem, binding, lambda: _named(binding, supplied))
        except BaseException:
            runs.close()
            raise
        crowded = []  # of how the session runs, or what it is given: no summary keeps these
        slots = 1 if count is None else count
        for task in binding.plan.tasks:
            cpus = task_cpus(binding.contexts, task)
            beyond = f"more than the {usable} processors this process may use"
            if count is not None and cpus > count:
                crowded.append(
                    f"{task.id.label} simulates with {cpus} threads, more than the {count} "
                    f"worker(s) of the session: it runs alone, on {cpus} processors"
                    + ("" if cpus <= usable else f", {beyond}")
                )
            elif cpus > usable:  # within the workers, which are then too many themselves
                crowded.append(f"{task.id.label} simulates with {cpus} threads, {beyond}")
        if slots > usable:
            crowded.append(
                f"{slots} workers, more than the {usable} processors this process may use: tasks "
                "will share processors"
            )
        crowded += _memory_warning(binding, count)
        if supplied:
            crowded.append(
                f"supplied {', '.join(sorted(task.label for task in supplied))}: each file is "
                "identified (size and SHA-256), not verified to be the output of the stage the "
                "problem defines; the records list it as supplied"
            )
        return cls(baseline, binding, supplied, (*binding.warnings, *crowded), count, runs)

    @property
    def spec(self) -> ProblemSpec:
        """The problem, as loaded when the session was opened."""
        return self._baseline.spec

    @property
    def targets(self) -> tuple[GeometryParameter, ...]:
        """The attribute each geometry destination of the parameters sets, with its value in the
        unmodified geometry, in the order of
        :attr:`~k4opt.problem.ProblemSpec.geometry_destinations`."""
        return self._baseline.problem.targets

    @property
    def problem_path(self) -> Path:
        """The absolute path of the problem file."""
        return self._baseline.path

    @property
    def warnings(self) -> tuple[str, ...]:
        """Collections an evaluator reads that the simulation is not expected to write; the
        simulation's output decides, so they do not stop an evaluation (each summary keeps them
        too). Then how the session's workers suit what it runs: a simulation on more threads than
        there are workers, more workers than processors; and that a file the caller supplies is
        identified, not verified to be the stage it stands for (no summary keeps these)."""
        return self._warnings

    def diagnostics(self, level: int = 0) -> tuple[str, ...]:
        """The evaluations that no objective, constraint or required evaluator of ``level`` (of
        a nested optimization; 0, the top, for any other) reads, by label (``zz/calo.occupancy``),
        in plan order: each is run and recorded, but whatever its result, the status of a
        candidate evaluated for that level does not depend on it."""
        return tuple(
            task.id.label
            for task in self._binding.plans[level].of(Kind.EVALUATOR)
            if not task.required
        )

    def notice(self, level: int = 0) -> str | None:
        """What to tell someone who starts the session of its :meth:`diagnostics` at ``level``,
        ``None`` without one: a check meant to veto candidates that is not marked
        ``required: true`` would otherwise veto nothing, unnoticed."""
        diagnostics = self.diagnostics(level)
        if not diagnostics:
            return None
        return (
            f"diagnostics (recorded, do not affect the result): {', '.join(diagnostics)}; "
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
        ``KEY4HEP_STACK`` and ``K4GEO``; ``programs``: each program of the stack the stages run,
        by name, as the executable the session runs (its path, size and SHA-256, ``None`` if it
        cannot be read); ``search``: the SHA-256 of each variable that decides what those
        programs load (:func:`k4opt.program.search_environment`); ``k4opt``: its version, the
        SHA-256 of its code and its git checkout, as every summary records them. The
        environment, the programs, k4opt and the external files are taken when the session
        opened. A fresh copy on every call.
        """
        return self._baseline.identity()

    def reuse(self, directories: Iterable[str | os.PathLike[str]]) -> int:
        """Let the stages of this session's candidates reuse what the stages of the records in
        ``directories`` made; how many outputs that adds.

        Each directory is a run directory (it holds a ``summary.json``: ``k4opt evaluate``'s
        output, a trial of a study) or holds run directories (a study). A stage of a candidate
        is not run when one with its signature (:mod:`k4opt.record.signature`: the same
        definition, inputs, geometry, files, programs, environment and k4opt code) made an
        output that is still there with the SHA-256 its record holds: its record then says
        ``reused``, with where the output comes from. The session's own stages are reused the
        same way, without being named here. Raises :class:`~k4opt.execution.RunError` for a
        directory that does not exist."""
        runs: list[Path] = []
        for given in directories:
            path = Path(os.path.abspath(Path(given).expanduser()))
            if (path / SUMMARY_NAME).is_file():
                runs.append(path)
            elif path.is_dir():
                runs += sorted(each for each in path.iterdir() if (each / SUMMARY_NAME).is_file())
            else:
                raise RunError(f"reuse {os.fspath(given)}: {path} is no directory")
        return self._reuse.index(runs)

    def check_limits(self) -> None:
        """Raise :class:`~k4opt.execution.RunError` if this session could not keep the time
        limits of the problem's evaluations: inline, nothing stops an evaluation that runs in
        this process when its time runs out, and no limit is claimed that is not kept. Checking a
        candidate (:meth:`check`) needs none of that; evaluating one does, so :meth:`evaluate`
        and :meth:`evaluate_each` refuse such a session before they create anything."""
        if self._backend is not None:
            return
        spec = self.spec
        limited = [f"evaluators[{i}].timeout" for i, e in enumerate(spec.evaluators) if e.timeout]
        if limited or spec.timeout is not None:
            raise RunError(
                f"{', '.join(limited) or 'timeout'}: an evaluation that runs in the session's own "
                "process cannot be stopped when its time runs out; evaluate with worker processes "
                "(--workers 1, or more)"
            )

    def verify_inputs(self) -> None:
        """Raise :class:`~k4opt.execution.RunError` if a file the stages read (named by the problem
        or supplied) is not the file it was when the session opened: other bytes, or now readable
        or unreadable.

        A file is hashed again only when its size, times or inode changed.
        """
        self._baseline.verify_inputs()

    def check(self, candidate: Mapping[str, Scalar], replicate: int = 0) -> Configuration:
        """Check ``candidate`` as :meth:`evaluate` does before it creates anything, and stop there;
        what the candidate configures, each value written where its parameter sends it, with the
        seeds of ``replicate``.

        ``candidate`` maps the name of each parameter to its value. Raises
        :class:`~k4opt.problem.CandidateError` for a value outside its domain, what creating an
        evaluator or its ``check()`` raises for the request this candidate gives it, and
        :class:`~k4opt.execution.RunError` for a file the stages or the geometry read, the
        environment or k4opt's code that changed since the session opened.
        """
        _, _, prepared, _ = self._prepare(candidate, replicate)
        if isinstance(prepared, Rejected):
            raise CandidateRejected(str(prepared))
        return prepared.configuration

    def evaluate(
        self,
        candidate: Mapping[str, Scalar],
        output: str | os.PathLike[str],
        *,
        run: Mapping[str, Json] | None = None,
        replicate: int = 0,
        level: int = 0,
    ) -> RunOutcome | Rejected:
        """Evaluate ``candidate`` into the new directory ``output`` and write its summary there;
        :class:`Rejected`, with nothing created, if an evaluator's ``check`` rejects it.

        ``candidate`` maps the name of each parameter to its value. ``run`` is recorded
        as the summary's ``run``: JSON-compatible data that says what the record was made for
        (a study and a trial, say). ``replicate`` above 0 evaluates the candidate on other events:
        each stage's seed derived from the problem's
        (:func:`k4opt.problem.candidate.replicate_seed`), for a problem whose samples make their
        events from their seeds and with no stage supplied. ``level`` is the level of a nested
        optimization it is evaluated for (:attr:`~k4opt.problem.ProblemSpec.levels`), whose
        criteria judge it (:meth:`~k4opt.problem.ProblemSpec.criteria`). Raises a
        :class:`~k4opt.errors.K4OptError` before anything is created for what :meth:`check`
        refuses, an existing ``output`` or an unusable location. Execution problems are
        ``failed`` or ``invalid`` results in the outcome; a result that breaks the evaluation
        contract raises an
        :class:`~k4opt.evaluation.EvaluationContractError` after the stages ran, and no summary
        is written. Nor is one written if what :meth:`check` verifies changed while the
        candidate ran: that raises a :class:`~k4opt.execution.RunError`, since the record could not
        say which files or code the candidate was evaluated with.
        """
        request = RunRequest(candidate, output, run, replicate, level)
        with self.evaluate_each([request]) as evaluations:
            for _, result in evaluations:
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError("the candidate was never evaluated")  # pragma: no cover

    def evaluate_each(
        self,
        requests: Iterable[RunRequest | Wait],
        *,
        activity: Callable[[Activity], None] | None = None,
    ) -> Evaluations:
        """Evaluate the candidate of each of ``requests``, as :meth:`evaluate` does, and give
        each ``(request, outcome)`` as its candidate ends: its :class:`RunOutcome`, the
        :class:`Rejected` of a candidate an evaluator rejected (nothing of it was created), or the
        exception that stopped it, if one did (nothing else is then written for it).

        The next request is taken from ``requests`` only once the candidates under way leave a
        slot idle (none of their tasks is ready to take it), and every candidate that ended was
        yielded before; :data:`~k4opt.execution.WAIT` in place of a request says there is none
        now, and the next is taken only once a candidate under way was yielded (it raises a
        :class:`~k4opt.execution.RunError` if none is under way), so whatever makes the
        requests (an optimizer) learns from every outcome it can before it proposes again. With
        one slot, that is one candidate at a time, each yielded before the next is taken; with
        more, the candidates end in any order, each with its own record. Close what this
        returns (``with``) when done with it: that stops whatever still runs, as leaving the
        loop early does not by itself. One at a time per session. ``activity`` is told what runs
        (:class:`~k4opt.execution.Activity`) whenever that changes, in this thread.
        """
        if self._open is not None:
            raise RunError("the session is evaluating already; close that evaluation first")
        self.check_limits()  # before anything is created
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
            self._dry_runs.close()
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
        self, requests: Iterator[RunRequest | Wait], activity: Callable[[Activity], None] | None
    ) -> Generator[tuple[RunRequest, RunOutcome | Rejected | Exception]]:
        runs: dict[str, _Run] = {}
        backend: Backend = self._backend or InlineBackend(
            self._context, dict(self._binding.evaluations)
        )

        def pull() -> _Run | Wait | None:
            request = next(requests, _END)
            if request is _END:
                return None
            if request is WAIT:
                return WAIT
            if not isinstance(request, RunRequest):  # the caller's mistake, not a candidate's
                raise RunError(f"expected a RunRequest, got {type(request).__name__}")
            try:
                return self._start(request, {run.workspace.root for run in runs.values()})
            except Exception as exc:  # before anything was created
                return _Run(Workspace("", None), None, error=exc, request=request)

        driven = _drive(
            Scheduler(backend.capacity),
            backend,
            runs,
            pull,
            self._checked,
            activity,
            sign=self._sign,
            reuse=self._reuse,
        )
        try:
            for run in driven:
                request = run.request
                assert isinstance(request, RunRequest)
                if run.error is not None:
                    yield request, run.error
                    continue
                if run.rejected is not None:
                    yield request, run.rejected
                    continue
                try:
                    self._check_relied(run)
                    outcome = publish(run, self._baseline, self._execution)
                except Exception as exc:
                    self._reuse.withdraw(run.workspace.id)  # no record says where they come from
                    yield request, exc
                    continue
                self._reuse.published(run.workspace.id)
                yield request, outcome
        finally:
            driven.close()
            if backend is not self._backend:  # the session's workers serve its next evaluation
                backend.close()

    def _check_relied(self, run: _Run) -> None:
        """Raise :class:`ProvenanceError` if ``run`` reused the output of another candidate of
        this session that was not published: its record would name one that does not exist."""
        missing = sorted(record for record in run.relies if not self._reuse.is_published(record))
        if missing:
            raise ProvenanceError(
                f"a stage reused the output of the candidate(s) {', '.join(missing)}, which were "
                f"not published, so the record could not say where it comes from (in "
                f"{run.workspace.root}, which has no {SUMMARY_NAME})"
            )

    def _start(self, request: RunRequest, under_way: Collection[Path | None] = ()) -> _Run:
        """A candidate checked and prepared as :meth:`check` does, with its own new run directory
        (not yet created) as its workspace (:func:`_check_output`); one an evaluator rejected,
        with no workspace."""
        reference = _run_reference(request.run)
        started = time.perf_counter()
        values, key, prepared, overrides = self._prepare(
            request.candidate, request.replicate, request.level
        )
        if isinstance(prepared, Rejected):
            return _Run(
                Workspace("", None), None, key, rejected=prepared, request=request, values=values
            )
        parameters = tuple(p.name for p, _ in self._baseline.spec.geometry_destinations)
        run_dir = Path(os.path.abspath(request.output))
        _check_output(run_dir, [root for root in under_way if root is not None])
        return _Run(
            Workspace(str(uuid.uuid4()), run_dir),
            prepared,
            key,
            self._inputs,
            GeometryWork(tuple(overrides), parameters),
            started,
            request=request,
            values=values,
            reference=reference,
            replicate=request.replicate,
            level=request.level,
        )

    def _checked(self, run: _Run, outcome: TaskOutcome | None, result: TaskResult) -> None:
        """What is checked of each result as it comes: that the geometry made changes the
        attributes the session resolved, to the values the candidate's configuration gives them,
        and that the references only the candidate's geometry names read what they read for the
        candidates already recorded. A geometry that was not made (it ran out of time, its worker
        died) has none to check."""
        if result.id != GEOMETRY or result.status != DONE or outcome is None:
            return
        run.own_refs = outcome.own_refs
        assert run.prepared is not None
        try:
            self._baseline.check_changes(result, run.prepared.configuration)
            self._baseline.check_read(candidate_reads(result))
        except RunError as exc:  # of the kind it is: a ProvenanceError stays one
            run_dir = run.workspace.root
            raise type(exc)(f"{exc} (in {run_dir}, which has no {SUMMARY_NAME})") from None

    def _sign(
        self, run: _Run, task: Task, results: Mapping[TaskId, TaskResult]
    ) -> Mapping[str, Json] | None:
        """The signature of ``task`` of ``run`` (:mod:`k4opt.record.signature`), from the results
        of what it reads: for a stage that runs, what it is made from; ``None`` for anything
        else, and for a stage that reads a geometry without a fingerprint."""
        if task.id.kind is not Kind.STAGE or task.given is not None:
            return None
        sample, name = task.id.sample, task.id.name
        assert sample is not None
        assert run.prepared is not None
        stage = run.prepared.contexts[sample].stages[name]
        reads: dict[str, object] = {}
        for need in task.needs:
            provenance = results[need].provenance
            if need == GEOMETRY:
                reads[need.name] = geometry_identity(provenance)
            elif self._binding.plan[need].given is not None:
                reads[need.name] = given_identity(provenance)
            else:
                upstream = provenance.get(SIGNATURE)
                reads[need.name] = upstream.get("sha256") if isinstance(upstream, Mapping) else None
        if any(identity is None for identity in reads.values()):
            return None  # what it reads is not identified: neither is what it makes
        prefix = f"samples.{sample}.{name}."
        files = {
            label.removeprefix(prefix): identity
            for label, (_, identity) in self._baseline.files.items()
            if label.startswith(prefix)
        }
        process = self._baseline.process
        programs, environment, k4opt = process["programs"], process["environment"], process["k4opt"]
        assert isinstance(programs, Mapping)
        assert isinstance(environment, Mapping)
        assert isinstance(k4opt, Mapping)
        program = None if stage.executor is None else executors(name)[stage.executor].program
        document = stage_document(
            name,
            stage,
            reads=reads,
            files=files,
            program=None if program is None else programs.get(program),
            environment={
                "KEY4HEP_STACK": environment.get("KEY4HEP_STACK"),
                "search": process["search"],
            },
            k4opt=k4opt.get("sha256"),
        )
        signature = freeze(signed(document), SIGNATURE, RunError)
        assert isinstance(signature, Mapping)
        return signature

    def _prepare(
        self, candidate: Mapping[str, Scalar], replicate: int = 0, level: int = 0
    ) -> tuple[dict[str, Scalar], CandidateKey, _Prepared | Rejected, list[Override]]:
        """The checked candidate, its key, what its preflight prepared (its configuration and
        the requests made for it, or why an evaluator rejected it) and its overrides; with the
        seeds of ``replicate``, which a stage the caller supplies would not follow, judged by the
        criteria of ``level``."""
        spec = self._baseline.spec
        values, key = validate_candidate(spec, candidate)
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= spec.depth:
            raise CandidateError(f"level {level!r}: the problem's levels are 0 to {spec.depth}")
        if replicate and self._inputs:
            supplied = ", ".join(sorted(task.label for task in self._inputs))
            raise CandidateError(
                f"replicate {replicate}: the stages {supplied} are supplied, so another seed would "
                "not make new events for them; a replicate makes every stage from its seeds"
            )
        configuration = configure(spec, values, replicate)
        self._baseline.verify()  # first: a file that is gone changed, more than it is missing
        prepared = _preflight(self._binding, configuration, self._dry_runs, level)
        if isinstance(prepared, Rejected):
            return values, key, prepared, []
        overrides = geometry_overrides(self._baseline.problem, configuration)
        return values, key, prepared, overrides


_END = object()  # what the requests give once there are none left; no request is it

PROGRAM_MEMORY = 2 * 1024**3
"""What k4opt expects one task that runs a program of the stack (ddsim, k4run) to take at its
peak, in bytes, for :func:`_memory_warning`: CLD's simulation and reconstruction peak at
1.3 to 1.6 GiB."""


def _memory_warning(binding: _Binding, workers: int | None) -> list[str]:
    """Why the session's workers may not fit in the memory this process may use, if they may
    not: as many programs of the stack (ddsim, k4run) at once as there are workers, each of
    :data:`PROGRAM_MEMORY`. k4opt reserves and limits no memory; a program the kernel kills for
    want of it fails its stage."""
    runs = any(task.given is None for task in binding.plan.of(Kind.STAGE))
    memory = usable_memory()
    if workers is None or not runs or memory is None:
        return []
    expected = workers * PROGRAM_MEMORY
    if expected <= memory:
        return []
    return [
        f"{workers} workers may run {workers} programs of the stack (ddsim, k4run) at once, about "
        f"{expected / 1024**3:.0f} GiB at {PROGRAM_MEMORY / 1024**3:.0f} GiB each, more than the "
        f"{memory / 1024**3:.1f} GiB this process may use: a program the kernel kills for "
        "memory fails its stage; fewer --workers?"
    ]


def _check_output(run_dir: Path, under_way: Iterable[Path]) -> None:
    """Raise :class:`RunError` unless ``run_dir`` can be a candidate's own new run directory: it
    does not exist (nor is it a symlink, a dangling one included), it is neither the directory of
    a candidate ``under_way`` (which its geometry may not have made yet), nor inside or around
    one, and it is not inside the run directory of a recorded candidate. Each is compared where
    its symlinks lead (:func:`os.path.realpath`); ``run_dir`` itself is created as given."""
    if run_dir.exists() or run_dir.is_symlink():
        raise RunError(f"output {run_dir} already exists; choose a new directory")
    real = Path(os.path.realpath(run_dir))
    for root in under_way:
        other = Path(os.path.realpath(root))
        if real == other:
            raise RunError(f"output {run_dir} already exists; choose a new directory")
        if other in real.parents or real in other.parents:
            raise RunError(
                f"output {run_dir} overlaps {root}, the run directory of a candidate under way; "
                "choose a separate directory"
            )
    for parent in real.parents:
        if _is_run(parent / SUMMARY_NAME):
            raise RunError(
                f"output {run_dir} is inside {parent}, the run directory of a recorded "
                "candidate; choose a directory outside it"
            )


def _is_run(path: Path) -> bool:
    """Whether ``path`` is the summary of a k4opt run: a JSON object of its schema. A file of
    that name that is not one (unreadable, another program's) makes no run directory."""
    try:
        if not path.is_file():
            return False
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return False
    return isinstance(data, Mapping) and data.get("schema") == SUMMARY_SCHEMA


def _workers(value: int | Literal["auto", "inline"]) -> int | None:
    """The number of worker processes ``value`` says, ``None`` for none (``"inline"``): itself,
    or for ``"auto"`` the processors this process may use. :class:`RunError` unless it is a
    positive integer, ``"auto"`` or ``"inline"``."""
    if value == INLINE:
        return None
    if value == AUTO:
        return usable_cpus()
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunError(f"workers must be a positive integer, 'auto' or 'inline', got {value!r}")
    return value
