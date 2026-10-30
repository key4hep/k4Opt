"""Internal: Optuna behind :func:`k4opt.optimization.optimize`, and the only module importing it.

Optuna's TPE sampler proposes each candidate. k4opt evaluates it with an
:class:`~k4opt.problem.EvaluationSession` into the trial's directory, and Optuna is told the
result as the trial's ``summary.json`` says it, read back from disk and validated against the
study (:func:`_report`): what the optimizer learns is always what the record holds. A valid
candidate becomes a COMPLETE trial with its objective values and one constraint value per limit
(:func:`constraint_values`); an invalid or failed one a FAIL trial, and the study goes on. A
trial that breaks the evaluation contract, or whose record does not hold up, is FAIL and stops
the study.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import signal
import sys
import threading
import urllib.parse
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    distribution_to_json,
)
from optuna.storages import BaseStorage, JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import FrozenTrial, Trial, TrialState

from k4opt.errors import K4OptError
from k4opt.evaluation import EvaluatorProtocol
from k4opt.optimization._study import (
    CONSTRAINT_VALUES,
    JOURNAL_NAME,
    LOCK_NAME,
    STUDY_SCHEMA,
    Lock,
    StudyReport,
    TrialReport,
    check_record,
    differences,
    digest,
    dirty_checkouts,
    now,
    read_study,
    revision_warnings,
    session_seed,
    trial_directories,
    trial_directory,
    unhashed,
    write_study,
)
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import SUMMARY_NAME, Assessment, EvaluationSession, ProblemSpec, RecordError

_EXACT_INTEGERS = 2**53  # TPE samples integers through floats, exact up to here
_STARTUP_TRIALS = 10  # random proposals before TPE models the trials
_MESSAGE = 1000  # characters of a message kept in the optimizer's storage; the record has all

type Warn = Callable[[str], None]
type Progress = Callable[[TrialReport], None]


# --- what Optuna is told ---------------------------------------------------------------------


def _float_bound(name: str, value: int | float) -> float:
    try:
        bound = float(value)
    except OverflowError:
        bound = None
    if bound is None or bound != value:
        text = str(value)
        shown = text if len(text) <= 24 else f"{text[:12]}... ({len(text)} digits)"
        raise OptimizationError(f"{name}: the bound {shown} is no float, which the sampler draws")
    return bound


def search_space(spec: ProblemSpec) -> dict[str, BaseDistribution]:
    """The distribution of each parameter, by canonical reference, in problem order.

    Continuous parameters are sampled uniformly between their bounds, with no log scale and no
    step; their bounds must be floats exactly, and so must their range. Integer bounds must stay
    within 2**53, below which the sampler's floats hold every integer. A categorical parameter
    takes one of its declared choices, as declared.
    """
    space: dict[str, BaseDistribution] = {}
    for parameter in spec.parameters:
        name = parameter.geometry.canonical_target
        if parameter.kind == "categorical":
            assert parameter.choices is not None
            space[name] = CategoricalDistribution(list(parameter.choices))
            continue
        assert parameter.bounds is not None
        low, high = parameter.bounds
        if parameter.kind == "integer":
            if max(abs(low), abs(high)) > _EXACT_INTEGERS:
                raise OptimizationError(
                    f"{name}: integer bounds beyond 2**53 in magnitude cannot be sampled exactly"
                )
            space[name] = IntDistribution(int(low), int(high))
            continue
        lower, upper = _float_bound(name, low), _float_bound(name, high)
        if upper - lower == float("inf"):
            raise OptimizationError(f"{name}: the range [{low}, {high}] is too wide to sample")
        space[name] = FloatDistribution(lower, upper)
    return space


def objective_values(assessment: Assessment) -> list[float]:
    """The objective values of a valid candidate, in the problem's order, as the floats the
    optimizer is told; a value no float holds exactly (an integer beyond 2**53) is refused, as
    it would be told as another value."""
    assert assessment.objectives is not None
    values = []
    for measurement in assessment.objectives:
        try:
            value = float(measurement.value)
        except OverflowError:
            value = None
        if value is None or value != measurement.value:
            raise OptimizationError(
                f"{measurement.stage}/{measurement.metric}: no float holds {measurement.value} "
                "exactly, so the optimizer cannot be told it"
            )
        values.append(value)
    return values


def constraint_values(assessment: Assessment) -> dict[str, float]:
    """The value the optimizer is told for each limit of an assessed candidate, by limit name.

    The physical margin as it is (``study.json`` records this as ``physical margins``), zero or
    below when the limit is satisfied. The sampler ranks infeasible trials by the sum of their
    positive values, so with limits on several metrics it adds values in different units.
    """
    values = {}
    for limit in assessment.limits:
        assert limit.satisfied is not None  # assessed
        if limit.margin is None:
            raise OptimizationError(
                f"{limit.name}: the margin of {limit.value} to {limit.limit} does not fit in a "
                "float, which the optimizer is told"
            )
        values[limit.name] = limit.margin
    return values


def limit_names(spec: ProblemSpec) -> list[str]:
    """The name of each limit of the problem, as the optimizer knows its constraint value."""
    names = [
        f"{c.stage}/{c.metric}:{bound}"
        for c in spec.constraints
        for bound in ("min", "max")
        if getattr(c, bound) is not None
    ]
    assert len(set(names)) == len(names)  # constraints are unique by stage and metric
    return names


# --- storage and setup -----------------------------------------------------------------------


def _redacted(url: str) -> str:
    """``url`` as the study records it and messages show it: its password and the value of
    every query parameter (``?password=...``, ``sslkey=...``) are ``***``, and it has no
    fragment."""
    parts = urllib.parse.urlsplit(url)
    text = url.partition("#")[0]  # edited in place: rebuilding it would drop sqlite:///'s //
    if parts.password is not None:
        userinfo, _, host = parts.netloc.rpartition("@")
        text = text.replace(parts.netloc, f"{userinfo.partition(':')[0]}:***@{host}", 1)
    if parts.query:
        keys = [key for key, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)]
        text = text.partition("?")[0] + "?" + "&".join(f"{key}=***" for key in keys)
    return text


def _scrubbed(text: str, url: str) -> str:
    """``text`` (a database error, say) with ``url`` and every secret in it redacted."""
    parts = urllib.parse.urlsplit(url)
    secrets_ = [parts.password, parts.fragment]
    secrets_ += [value for _, value in urllib.parse.parse_qsl(parts.query)]
    text = text.replace(url, _redacted(url))
    spellings = {
        spelling
        for secret in secrets_
        if secret
        for spelling in (secret, urllib.parse.unquote(secret), urllib.parse.quote(secret, safe=""))
    }
    for secret in sorted(spellings, key=len, reverse=True):
        text = text.replace(secret, "***")
    return text


def _storage(directory: Path, url: str | None) -> BaseStorage:
    if url is None:
        path = str(directory / JOURNAL_NAME)
        return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    try:
        return RDBStorage(url)
    except ImportError as exc:
        raise OptimizationError(
            f"--storage {_redacted(url)} needs SQLAlchemy ({exc}); install it, or leave out "
            "--storage to keep the study's storage in its directory"
        ) from None
    except Exception as exc:
        raise OptimizationError(f"--storage {_redacted(url)}: {_scrubbed(str(exc), url)}") from None


def _storage_record(url: str | None) -> dict[str, str]:
    return {"journal": JOURNAL_NAME} if url is None else {"url": _redacted(url)}


@contextlib.contextmanager
def _terminate_as_interrupt() -> Iterator[None]:
    """SIGTERM (a batch system ending the job) stops the study as Ctrl-C does, so the trial is
    closed and the lock removed; in the main thread only, where signals are handled."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Optuna logs every trial at INFO; k4opt reports them itself."""
    level = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        yield
    finally:
        optuna.logging.set_verbosity(level)


# --- one study -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Study:
    directory: Path
    document: Mapping[str, Any]  # study.json
    spec: ProblemSpec
    study: optuna.Study
    warn: Warn


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _report(
    run: _Study, trial: Trial, existing: Mapping[str, float], *, recovered: bool = False
) -> FrozenTrial:
    """Tell the optimizer the result of ``trial`` as its ``summary.json`` holds it.

    The one way a trial is told, live or when a resume recovers it: the record is read back from
    disk, checked against the study (:func:`~k4opt.optimization._study.check_record`) and its
    assessment recomputed from its measurements. ``existing`` are the constraint values the
    trial already has, which must equal the recomputed ones. Raises
    :class:`OptimizationError`, before anything is told, for a record that does not hold up.
    """
    number, name = trial.number, trial_directory(trial.number)
    path = run.directory / name / SUMMARY_NAME
    try:
        data = path.read_bytes()
        summary = json.loads(data)
    except (OSError, ValueError) as exc:
        raise OptimizationError(f"trial {number}: cannot read {path}: {exc}") from exc
    warnings = check_record(run.document, summary, number, trial.params)
    try:
        assessment = Assessment.from_summary(run.spec, summary)
    except RecordError as exc:
        raise OptimizationError(f"trial {number}: {path}: {exc}") from None
    valid = summary["status"] == "valid"
    values = objective_values(assessment) if valid else None
    constraints = constraint_values(assessment) if valid else {}
    for key, value in constraints.items():
        if key in existing and existing[key] != value:
            raise OptimizationError(
                f"trial {number}: the optimizer holds {existing[key]} for {key}, but the "
                f"record gives {value}"
            )
    for warning in warnings:
        run.warn(f"trial {number}: {warning}")
    trial.set_user_attr("k4opt.status", summary["status"])
    record = {"id": summary.get("id"), "sha256": _sha256(data), "path": f"{name}/{SUMMARY_NAME}"}
    trial.set_user_attr("k4opt.record", record)
    geometry = summary.get("geometry")
    if isinstance(geometry, Mapping):
        trial.set_user_attr("k4opt.fingerprint", geometry.get("fingerprint"))
    if isinstance(summary.get("message"), str):
        trial.set_user_attr("k4opt.message", summary["message"][:_MESSAGE])
    if recovered:
        trial.set_user_attr("k4opt.recovered", True)
    if values is None:
        return run.study.tell(trial, state=TrialState.FAIL)
    for key, value in constraints.items():
        if key not in existing:
            trial.set_constraint(key, value)
    trial.set_user_attr("k4opt.feasible", assessment.feasible)
    return run.study.tell(trial, values)


def _fail(run: _Study, trial: Trial, error: str | None = None) -> FrozenTrial:
    if error is not None:
        trial.set_user_attr("k4opt.error", error[:_MESSAGE])
    return run.study.tell(trial, state=TrialState.FAIL)


def _evaluate(
    run: _Study, session: EvaluationSession, space: Mapping[str, BaseDistribution], seed: int
) -> FrozenTrial:
    """Ask for one candidate, evaluate it and tell the optimizer; raise to stop the study."""
    trial = run.study.ask(fixed_distributions=dict(space))
    name = trial_directory(trial.number)
    trial.set_user_attr("k4opt.dir", name)
    trial.set_user_attr("k4opt.sampler_seed", seed)
    candidate = {key: trial.params[key] for key in space}
    output = run.directory / name
    reference = {"study": run.document["id"], "trial": trial.number}
    try:
        session.evaluate(candidate, output, run=reference)
        return _report(run, trial, {})
    except KeyboardInterrupt:
        _close_interrupted(run, trial, output)
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _fail(run, trial, error)
        raise OptimizationError(
            f"trial {trial.number} stops the study ({name} is FAIL): {error}"
        ) from exc


def _close_interrupted(run: _Study, trial: Trial, output: Path) -> None:
    """A trial that Ctrl-C or SIGTERM ended: told its record if that was published just before,
    FAIL otherwise, its directory left as it is."""
    if (output / SUMMARY_NAME).is_file():
        # the interruption may have come within _report, after the constraints were set
        (frozen,) = [t for t in run.study.get_trials(deepcopy=False) if t.number == trial.number]
        try:
            _report(run, trial, frozen.constraints)
            return
        except K4OptError:
            pass  # a record that does not hold up is no result
    _fail(run, trial)


def _recover(run: _Study, progress: Progress) -> None:
    """Finish the trials a session left running, before any new one: one without a record is
    FAIL, its directory left as it is, and one with a record is told it; ``progress`` is called
    with each as it is finished. Before anything changes, the resume refuses if a trial
    directory belongs to no trial, or if the record a finished trial was told is gone or was
    changed (its SHA-256)."""
    trials = run.study.get_trials(deepcopy=False)
    known = {trial.user_attrs.get("k4opt.dir", trial_directory(trial.number)) for trial in trials}
    stray = [name for name in trial_directories(run.directory) if name not in known]
    if stray:
        raise OptimizationError(
            f"{', '.join(stray)} in {run.directory} belong(s) to no trial of the study; move "
            "them out of the study directory to resume it"
        )
    for frozen in trials:  # the evidence of what the optimizer was told is still there
        _check_told_record(run, frozen)
    for frozen in trials:
        if frozen.state != TrialState.RUNNING:
            continue
        trial = Trial(run.study, frozen._trial_id)  # the one private access of Optuna
        if (run.directory / trial_directory(frozen.number) / SUMMARY_NAME).is_file():
            finished = _report(run, trial, frozen.constraints, recovered=True)
        else:
            trial.set_user_attr("k4opt.recovered", True)
            finished = run.study.tell(trial, state=TrialState.FAIL)
        progress(_trial_report(run, finished))


def _check_told_record(run: _Study, frozen: FrozenTrial) -> None:
    """A trial the optimizer was told a record of still has that record, byte for byte, and
    every other finished trial is one k4opt made: interrupted, or one that stopped the study.

    k4opt marks each trial it asks for with ``k4opt.dir`` at once, and recovery marks one that
    a killed session left without it (``k4opt.recovered``). A finished trial with a result but
    no record, or with no mark of k4opt at all, was added by another client of the storage: it
    would count among the study's trials (and so change the next session's seed) without being
    one of its evaluations.
    """
    attrs = frozen.user_attrs
    told = attrs.get("k4opt.record")
    if frozen.state == TrialState.RUNNING:
        return  # recovery decides about it
    if not isinstance(told, Mapping):
        ours = "k4opt.dir" in attrs or attrs.get("k4opt.recovered") is True
        if frozen.state != TrialState.FAIL or "k4opt.status" in attrs or not ours:
            raise OptimizationError(
                f"trial {frozen.number} is {frozen.state.name} without the record k4opt tells a "
                "trial with, or any mark of k4opt, so it was not made by k4opt optimize; resume "
                "needs every trial to be one of the study's own"
            )
        return
    path = run.directory / str(told.get("path"))
    try:
        found = _sha256(path.read_bytes())
    except OSError:
        found = None
    if found != told.get("sha256"):
        state = "is gone" if found is None else "is not the record the optimizer was told"
        raise OptimizationError(
            f"trial {frozen.number}: {path} {state}, so nothing shows what its values rest on; "
            "put the record back to resume the study"
        )


def _trial_report(run: _Study, frozen: FrozenTrial) -> TrialReport:
    attrs = frozen.user_attrs
    status = attrs.get("k4opt.status")
    if status is None:
        status = "error" if "k4opt.error" in attrs else "interrupted"
        if frozen.state == TrialState.RUNNING:
            status = "running"
    names = [f"{o.stage}/{o.metric}" for o in run.spec.objectives]
    values = frozen.values if frozen.state == TrialState.COMPLETE else None
    return TrialReport(
        number=frozen.number,
        directory=run.directory / trial_directory(frozen.number),
        candidate=dict(frozen.params),
        status=status,
        feasible=attrs.get("k4opt.feasible"),
        objectives=None if values is None else dict(zip(names, values, strict=True)),
        record=(attrs.get("k4opt.record") or {}).get("id"),
        message=attrs.get("k4opt.message", attrs.get("k4opt.error")),
        recovered=bool(attrs.get("k4opt.recovered", False)),
    )


def _study_report(run: _Study) -> StudyReport:
    trials = tuple(_trial_report(run, t) for t in run.study.get_trials(deepcopy=False))
    return StudyReport(
        name=run.study.study_name,
        directory=run.directory,
        id=run.document["id"],
        trials=trials,
        pareto=tuple(sorted(t.number for t in run.study.best_trials)),
        objectives=tuple(f"{o.stage}/{o.metric}" for o in run.spec.objectives),
        constraints=tuple(limit_names(run.spec)),
    )


def _check_optuna_study(study: optuna.Study, document: Mapping[str, Any]) -> None:
    """The Optuna study is the one ``study.json`` describes."""
    optimizer = document["optimizer"]
    directions = [direction.name.lower() for direction in study.directions]
    expected = {
        "directions": optimizer["directions"],
        "k4opt.study": document["id"],
        "k4opt.identity": document["identity"],
        "k4opt.constraints": optimizer["constraints"],
    }
    found = {key: study.user_attrs.get(key) for key in expected} | {"directions": directions}
    differ = [key for key in expected if found[key] != expected[key]]
    if differ:
        raise OptimizationError(
            f"the optimizer's study {study.study_name!r} is not the one {document['id']} "
            f"describes: its {', '.join(differ)} differ"
        )


def optimize(
    problem: str | os.PathLike[str],
    evaluators: Sequence[EvaluatorProtocol],
    output: str | os.PathLike[str],
    *,
    trials: int,
    seed: int | None,
    resume: bool,
    storage: str | None,
    warn: Warn,
    progress: Progress,
) -> StudyReport:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise OptimizationError(f"trials must be a positive integer, got {trials!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise OptimizationError(f"seed must be an integer, got {seed!r}")
    if seed is not None and not 0 <= seed < 2**32:
        raise OptimizationError(f"seed must be in [0, 2**32), got {seed}")
    if resume and seed is not None:
        raise OptimizationError("a resumed study keeps the seed it was made with; leave out --seed")
    directory = Path(os.path.abspath(Path(output).expanduser()))

    # everything that can be checked before a file is created
    session = EvaluationSession.open(problem, evaluators)
    spec = session.spec
    space = search_space(spec)  # the evaluators' settings were checked when the session opened
    identity: dict[str, Any] = session.identity()
    reasons = unhashed(identity)
    if reasons:
        raise OptimizationError(f"the study could not identify what it runs: {'; '.join(reasons)}")
    names = limit_names(spec)
    for warning in session.warnings:
        warn(warning)
    if not resume:
        for warning in dirty_checkouts(identity["evaluators"]):
            warn(warning)
    metrics = {(c.stage, c.metric) for c in spec.constraints}
    if len(metrics) > 1:
        warn(
            f"constraints on {len(metrics)} metrics: while feasible trials are scarce, the sampler "
            "ranks infeasible ones by the sum of their violations, which adds values in different "
            "units (docs/optimization.md)"
        )
    database: BaseStorage | None = None  # made in the directory, or reached once below
    if resume:
        document = read_study(directory)
    else:
        document = _new_document(spec, identity, space, names, seed, storage)
        # a database is reached, and set up once, before a file is made: a wrong URL makes none
        database = None if storage is None else _storage(directory, storage)
        _refuse_used(directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OptimizationError(f"cannot create {directory}: {exc}") from exc

    try:
        with _quiet(), _terminate_as_interrupt(), Lock(directory):
            if resume:
                _check_resume(document, identity, spec, space, names, storage, warn)
            else:
                _refuse_used(directory, LOCK_NAME)  # again, now that no one else can start here
                write_study(directory, document)
            storage_ = database or _storage(directory, storage)
            study = _open_optuna(directory, document, storage_, resume)
            run = _Study(directory, document, spec, study, warn)
            if resume:
                _check_optuna_study(study, document)
                _recover(run, progress)
            seed = document["optimizer"]["sampler"]["seed"]
            seed_now = session_seed(seed, len(study.trials))
            study.sampler = optuna.samplers.TPESampler(
                seed=seed_now, n_startup_trials=_STARTUP_TRIALS
            )
            for _ in range(trials):
                progress(_trial_report(run, _evaluate(run, session, space, seed_now)))
            return _study_report(run)
    except Exception as exc:
        replacement, secret = _leaving(exc, storage)
        if replacement is None:
            raise
        if secret:  # nothing of the original, which names it, is kept for a traceback
            raise replacement from None
        raise replacement from exc


def _chain(exc: BaseException) -> list[BaseException]:
    """``exc`` and the exceptions it was raised from or during, as a traceback shows them."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return chain


def _storage_errors() -> tuple[type[BaseException], ...]:
    """What Optuna and the storage raise when they fail, as opposed to a bug of k4opt."""
    errors: tuple[type[BaseException], ...] = (optuna.exceptions.OptunaError, OSError)
    database = sys.modules.get("sqlalchemy.exc")  # imported only if a database was used
    if database is not None:
        errors += (database.SQLAlchemyError,)
    return errors


def _leaving(exc: Exception, url: str | None) -> tuple[Exception | None, bool]:
    """What leaves :func:`optimize` in place of ``exc`` (``None``: ``exc`` itself), and whether it
    must leave without ``exc``, which then names a secret of the storage URL ``url``.

    A k4opt error leaves as it is, and so does a bug of k4opt; a failure of Optuna or the
    storage becomes an :class:`OptimizationError`. Each is redacted when anything in its chain
    names a secret of ``url``.
    """
    text = "\n".join(str(e) for e in _chain(exc))
    secret = url is not None and _scrubbed(text, url) != text
    if isinstance(exc, K4OptError) or not isinstance(exc, _storage_errors()):
        message = str(exc) if isinstance(exc, K4OptError) else f"{type(exc).__name__}: {exc}"
        if not secret:
            return None, False
        assert url is not None
        return OptimizationError(_scrubbed(message, url)), True
    message = f"the study cannot go on: {type(exc).__name__}: {exc}"
    return OptimizationError(message if url is None else _scrubbed(message, url)), secret


def _refuse_used(directory: Path, *ours: str) -> None:
    """A new study starts in a directory that does not exist or holds nothing (but ``ours``)."""
    held = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    if [name for name in held if name not in ours] or (
        directory.exists() and not directory.is_dir()
    ):
        raise OptimizationError(
            f"{directory} is in use; resume the study in it with --resume, or choose a new "
            "directory"
        )


def _new_document(
    spec: ProblemSpec,
    identity: Mapping[str, Any],
    space: Mapping[str, BaseDistribution],
    names: Sequence[str],
    seed: int | None,
    storage: str | None,
) -> dict[str, Any]:
    study_id = str(uuid.uuid4())
    document: dict[str, Any] = {"schema": STUDY_SCHEMA, "id": study_id, "created": now()}
    document |= dict(identity)
    document["identity"] = digest(identity)
    seed = secrets.randbelow(2**32) if seed is None else seed
    name = f"{spec.name}-{study_id[:8]}"
    document["optimizer"] = _optimizer(spec, space, names, name=name, seed=seed, storage=storage)
    return document


def _optimizer(
    spec: ProblemSpec,
    space: Mapping[str, BaseDistribution],
    names: Sequence[str],
    *,
    name: str,
    seed: int,
    storage: str | None,
) -> dict[str, Any]:
    """How the study is optimized, as ``study.json`` records it under ``optimizer``."""
    return {
        "backend": "optuna",
        "version": optuna.__version__,
        "study": name,
        "storage": _storage_record(storage),
        "sampler": {
            "name": "TPESampler",
            "seed": seed,
            "n_startup_trials": _STARTUP_TRIALS,
            "seeding": "per session, from the seed and the number of trials",
        },
        "directions": [o.direction for o in spec.objectives],
        "parameters": {
            key: json.loads(distribution_to_json(distribution))
            for key, distribution in space.items()
        },
        "constraints": list(names),
        "constraint_values": CONSTRAINT_VALUES,
    }


def _check_resume(
    document: Mapping[str, Any],
    identity: Mapping[str, Any],
    spec: ProblemSpec,
    space: Mapping[str, BaseDistribution],
    names: Sequence[str],
    storage: str | None,
    warn: Warn,
) -> None:
    """Refuse to resume a study with anything it compares otherwise than it was made with, and
    one whose recorded optimizer is not the one this k4opt would run it with."""
    optimizer = document["optimizer"]
    found = differences(document, identity)
    seed = (optimizer.get("sampler") or {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        found.append(f"its sampler seed {seed!r} is no seed in [0, 2**32)")
    else:
        expected = _optimizer(
            spec, space, names, name=str(optimizer.get("study")), seed=seed, storage=storage
        )
        found += [
            f"its recorded optimizer {key} is {optimizer.get(key)!r}, not {value!r}"
            for key, value in expected.items()
            if key not in ("version", "storage") and optimizer.get(key) != value
        ]
    if optimizer.get("version") != optuna.__version__:
        found.append(
            f"the study was made with Optuna {optimizer.get('version')}, this is Optuna "
            f"{optuna.__version__}: install optuna=={optimizer.get('version')} to resume it"
        )
    if optimizer.get("storage") != _storage_record(storage):
        found.append(
            f"the study's storage is {optimizer.get('storage')}, not {_storage_record(storage)}; "
            "give the same --storage as when it was made"
        )
    if digest(document) != document.get("identity"):
        found.append("its study.json does not hold together (its identity digest differs)")
    if found:
        raise OptimizationError(
            f"cannot resume {document['optimizer']['study']}: {'; '.join(found)}. Start a new "
            "study for this"
        )
    for warning in revision_warnings(document["evaluators"], identity["evaluators"]):
        warn(warning)


def _open_optuna(
    directory: Path, document: Mapping[str, Any], storage: BaseStorage, resume: bool
) -> optuna.Study:
    """The Optuna study of ``document``: made for a new study; loaded for a resumed one, and its
    setup finished if a killed process cut it short before the first trial."""
    optimizer = document["optimizer"]
    name = optimizer["study"]
    attributes = _study_attributes(document)
    if resume:
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            if trial_directories(directory):
                raise OptimizationError(
                    f"the optimizer's storage holds no study {name!r}, though {directory} has "
                    "trials"
                ) from None
        else:
            if not study.trials:  # made, but perhaps not all its attributes written
                for key, value in attributes.items():
                    if key not in study.user_attrs:
                        study.set_user_attr(key, value)
            return study  # whatever it holds is checked by _check_optuna_study
    study = optuna.create_study(
        study_name=name, storage=storage, directions=list(optimizer["directions"])
    )
    for key, value in attributes.items():
        study.set_user_attr(key, value)
    return study


def _study_attributes(document: Mapping[str, Any]) -> dict[str, Any]:
    """The ``k4opt.*`` attributes of the Optuna study ``document`` describes."""
    optimizer = document["optimizer"]
    objectives = document["problem"]["definition"]["objectives"]
    return {
        "k4opt.study": document["id"],
        "k4opt.identity": document["identity"],
        "k4opt.parameters": list(optimizer["parameters"]),
        "k4opt.objectives": [f"{o['stage']}/{o['metric']}" for o in objectives],
        "k4opt.constraints": list(optimizer["constraints"]),
    }
