"""Internal: one task of a candidate, described as plain data, and what it left behind.

Whatever runs a task (:func:`~k4opt.execution._work.execute_task`) gets it in three parts:

* a :class:`SessionContext`, the same for every task of a session: the problem, the stage
  conditions of each sample and what each evaluation must measure;
* a :class:`TaskSpec`, the task itself, plain data: the planned task, the files it reads by
  their place in the candidate's workspace, and the request of an evaluation;
* the root of the candidate's workspace (:class:`Workspace`) on the machine that runs it, and
  the :class:`Observations` taken of the files its reads made.

Paths come in two kinds. A file a task made is named relative to its candidate's workspace, so
that a description stays true wherever the workspace is; an input from outside it (a stage file
of the problem, a file the caller supplied, the geometry's source) keeps its absolute path,
which the session identified by its content. The ``stat`` of a file (its size, times and inode)
tells a rewritten file apart only on the filesystem that took it: it is an
:class:`Observations`, handed along with the description by an executor that shares the
filesystem, never part of what a task returns (:class:`TaskOutcome`).

What a task returns is portable in the same way; the coordinator turns it into the
:class:`~k4opt.record.TaskResult` its records are assembled from
(:func:`~k4opt.execution._drive.to_result`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

from k4opt.evaluation import (
    Artifact,
    Declaration,
    EvaluationRequest,
    EvaluationResult,
    RoleInput,
)
from k4opt.execution._stages import StageContext
from k4opt.frozen import FrozenMap, Json
from k4opt.geometry import Catalog, GeometrySource, Override
from k4opt.hashing import Stat
from k4opt.problem.contract import MetricKey
from k4opt.problem.models import EvaluatorSpec, ProblemSpec, Stage
from k4opt.problem.plan import Task, TaskId
from k4opt.program import Observation
from k4opt.record.results import PeakMemory
from k4opt.software import Local

type Output = Callable[[Path], AbstractContextManager[object]]
"""Where an evaluator's output goes while it is called, given its own directory."""


GEOMETRY_TIMEOUT = 600.0
"""How many seconds the task that makes a candidate's geometry may run in a worker process. It is
k4opt's own work on files (seconds, normally), which no physics makes long: only a stalled
filesystem would, and then the geometry fails, recorded, rather than hold its worker forever. The
problem's ``timeout`` limits its stages and evaluations, not this."""


@dataclass(frozen=True, slots=True)
class Workspace:
    """Where the tasks of one candidate write: ``id`` names it wherever it is (the record id of
    the candidate), ``root`` is its directory on the coordinator's machine, ``None`` for a
    candidate none of whose tasks writes a file. A task description names only the ``id``;
    whoever runs the task knows where the workspace is."""

    id: str
    root: Path | None


@dataclass(frozen=True, slots=True)
class Measuring:
    """What the work of an evaluation needs besides its request: the ``entry`` that binds it,
    what its implementation declared when bound (``declaration``: its results are checked
    against that, which the record holds, not against what it declares now) and the metrics the
    roles of combined evaluators ``demanded`` of it."""

    entry: EvaluatorSpec
    declaration: Declaration
    demanded: tuple[MetricKey, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionContext:
    """What every task of a session needs: the problem ``spec``, the stage conditions of each
    sample (``contexts``, without stages: those of a task are its candidate's, in its
    :class:`TaskSpec`, as its time limit is), what each evaluation must measure
    (``evaluations``), what the external references of the session's geometry read
    (``externals``), which a candidate's own references are compared with, and the geometry
    candidates start from (``source``, discovered as ``catalog``; none where every candidate is
    given its geometry). ``observe`` is how closely the programs of stages are observed, as the
    session found it can (:data:`k4opt.program.Observation`), and ``local`` the local entries of
    the software (:attr:`k4opt.software.Software.local`), which a footprint names rather than
    identifies again."""

    spec: ProblemSpec
    contexts: Mapping[str, StageContext]
    evaluations: Mapping[TaskId, Measuring]
    externals: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    source: GeometrySource | None = None
    # what discovering ``source`` gives, kept to discover it once: not compared
    catalog: Catalog | None = field(default=None, compare=False)
    # the collection names of each stage file, by path and stat, so that a file several
    # evaluations read is listed once; a cache of this process, never part of a description,
    # which keeps the files of the last few candidates only
    listed: dict[tuple[Path, Stat | None], tuple[str, ...] | str] = field(
        default_factory=dict, compare=False
    )
    observe: Observation = "none"
    local: tuple[Local, ...] = ()


@dataclass(frozen=True, slots=True)
class GeometryWork:
    """How the geometry task makes the candidate's geometry: by applying ``overrides`` to the
    session's geometry, each set by the parameter of the same place in ``parameters``, or by
    taking the compact file ``given``, already prepared (named by its absolute path: it is an
    input from outside the workspace)."""

    overrides: tuple[Override, ...] = ()
    parameters: tuple[str, ...] = ()
    given: Path | None = None


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One task of a candidate, as plain data.

    ``workspace`` is the id of the candidate's :class:`Workspace`, ``task`` the planned task.
    ``inputs`` are the files the task is given, by name (``geometry``, a stage); ``watched`` the
    files its reads made, and ``tree`` the directory of the geometry it reads (``None`` if it
    reads none made in the workspace): what it checks before and after it runs. ``stages`` are
    the stages of its sample as its candidate configures them, for a stage. ``geometry`` says
    how the geometry task works (``None``: there is no geometry), ``supplied`` is a file the
    caller gives for a stage, and ``request`` and ``roles`` are what an evaluation is asked, its
    request without inputs or a directory, and ``declined`` why its evaluator's ``check``
    rejected the candidate, for a diagnostic, which is then not run. ``cpus`` is how many
    processors the task keeps busy,
    and ``timeout`` how many seconds it may run, ``None`` for no limit (the problem's
    ``timeout`` of a stage or an evaluation, :data:`GEOMETRY_TIMEOUT` for the geometry):
    whatever runs the task stops it then, and a stage passes it on to the program it runs.
    """

    workspace: str
    task: Task
    inputs: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    watched: tuple[Artifact, ...] = ()
    tree: Path | None = None
    stages: Mapping[str, Stage] = field(default_factory=lambda: FrozenMap({}))
    geometry: GeometryWork | None = None
    supplied: Path | None = None
    request: EvaluationRequest | None = None
    roles: Mapping[str, tuple[RoleInput, ...]] = field(default_factory=lambda: FrozenMap({}))
    cpus: int = 1
    timeout: float | None = None
    declined: str | None = None


@dataclass(frozen=True, slots=True)
class Observations:
    """The ``stat`` of files, valid only on the filesystem that took them: of what a task made, by
    the reference of each file (``made``), and of every file of the geometry it made, by its path
    below the geometry's directory (``tree``)."""

    made: Mapping[str, Stat] = field(default_factory=lambda: FrozenMap({}))
    tree: Mapping[str, Stat] | None = None


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """What a task left behind, each file in it named relative to its workspace (or by its
    absolute path, for an input from outside).

    As :class:`~k4opt.record.TaskResult` describes it, without the record of an
    evaluation, whose request the coordinator made: ``result`` is what the evaluator returned (its
    artifacts relative to the workspace), ``workdir`` the evaluation's own directory if it was
    called, and ``call_seconds`` the time of its call. ``own_refs`` says whether the geometry a
    geometry task made has ``ref`` attributes the candidate changed. ``memory`` is its peak memory
    where it ran.

    Only the files are named so that they hold wherever the workspace is. ``provenance`` says how
    the task ran where it ran (the command of a stage, with the absolute paths it was given) and
    is kept as it is: it records what ran, not where to find a file.
    """

    task: TaskId
    status: str
    message: str | None = None
    output: Path | None = None
    artifacts: Mapping[str, Path] = field(default_factory=lambda: FrozenMap({}))
    provenance: Mapping[str, Json] = field(default_factory=lambda: FrozenMap({}))
    elapsed_seconds: float | None = None
    made: Mapping[str, Artifact] = field(default_factory=lambda: FrozenMap({}))
    result: EvaluationResult | None = None
    workdir: Path | None = None
    call_seconds: float | None = None
    own_refs: bool = False
    memory: PeakMemory | None = None
