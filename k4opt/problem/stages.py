"""The pipeline stage types a problem's samples are made of, described once.

Each sample of a problem says how each of its stages is made: run by an executor (``gun`` for
``generation``, ``ddsim`` for ``simulation``, ``k4run`` for ``reconstruction``) or given as a
file. An evaluator names the stage it
reads with ``stage`` (``"simulation"``); the framework makes that stage of each sample it
measures, and the stages it depends on, available.

* ``generation``: where the primary particles come from. The ``gun`` executor has no file of
  its own (``ddsim`` fires it); a generation given as a file is an event file such as HepMC.
* ``simulation``: ``ddsim`` on the prepared geometry; reads ``generation``.
* ``reconstruction``: a Gaudi job (``k4run``) on the prepared geometry; reads every event of
  ``simulation``.

This module says what each stage type is: the stages it reads, its executors, the options each takes
and which of them a parameter can set, whether it reads the candidate's geometry or writes files,
the name of its main output and of its logs. :mod:`k4opt.problem.config` checks a problem against
it, the plan of a candidate's tasks (:mod:`k4opt.problem.plan`) follows it, and a record is read
back against it. How a stage runs is the execution's (:mod:`k4opt.execution._stages`): a new stage
type is one entry here and one runner there.

Pipeline stages are steps that run. The ``stage`` of a measurement, objective or constraint
only classifies a metric: an evaluator may report ``analysis`` metrics without an ``analysis``
pipeline stage, and nothing here reads metric labels.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from k4opt import ddsim, k4run
from k4opt.evaluation import EvaluationContractError
from k4opt.frozen import Json
from k4opt.program import Settable


@dataclass(frozen=True, slots=True)
class Executor:
    """What runs a stage: whether it takes ``events`` (required) and ``seed``, or a ``seed`` alone
    (optional: k4run's, of the job's random numbers), what checks its
    own options (raising a :class:`~k4opt.program.ProgramError`) given those that parameters
    set, which it then does not have (the pending ones), how a parameter sets one of its options
    (``settable``, raising for one it cannot set) and the stage whose process runs it, if another
    one does (``inside``: the gun, fired by ddsim, which sets its events and seed). An executor
    without ``events`` that runs in a process of its own processes every event of the stage it
    reads (k4run). ``asks`` says whether k4opt asks the program itself whether it takes the
    options, before anything runs (k4run's dry run): then a required stage whose option a
    parameter sets is asked for each candidate, and rejects one whose values it does not take.
    ``program`` names the executable of the stack it runs (``ddsim``), ``None`` for one that runs
    none of its own."""

    events: bool
    options: Callable[[Mapping[str, Json], Collection[str]], None]
    settable: Callable[[str], Settable]
    inside: str | None = None
    asks: bool = False
    program: str | None = None
    seed: bool = False


@dataclass(frozen=True, slots=True)
class StageType:
    """One stage type: the stage types it reads (``needs``), its ``executors`` by name, whether
    it reads the prepared geometry (``geometry``) and writes files, so that it needs a directory
    of its own (``writes``), the name of its main ``output`` and of the files besides it that it
    leaves in its directory, by role (``logs``)."""

    needs: tuple[str, ...]
    executors: Mapping[str, Executor]
    geometry: bool
    writes: bool
    output: str = "events"
    logs: Mapping[str, str] = field(default_factory=dict)


_STAGES: Mapping[str, StageType] = {
    "generation": StageType(
        (),
        {
            "gun": Executor(
                events=False,
                options=ddsim.check_particle_gun,
                settable=ddsim.settable_gun,
                inside="simulation",
            )
        },
        geometry=False,
        writes=False,
    ),
    "simulation": StageType(
        ("generation",),
        {
            "ddsim": Executor(
                events=True,
                options=ddsim.check_options,
                settable=ddsim.settable_option,
                program="ddsim",
            )
        },
        geometry=True,
        writes=True,
        logs=ddsim.LOG_FILES,
    ),
    "reconstruction": StageType(
        ("simulation",),
        {
            "k4run": Executor(
                events=False,
                options=k4run.check_options,
                settable=k4run.settable_option,
                asks=True,
                program="k4run",
                seed=True,
            )
        },
        geometry=True,
        writes=True,
        logs=k4run.LOG_FILES,
    ),
}

STAGE_TYPES = tuple(_STAGES)
"""The stage types, in pipeline order."""


def stage_type(stage: str) -> StageType:
    """The stage type ``stage``."""
    return _STAGES[stage]


def stage_reads(stage: str) -> tuple[str, ...]:
    """The stage types that ``stage`` reads."""
    return _STAGES[stage].needs


def reads_geometry(stage: str) -> bool:
    """Whether ``stage``, when it runs, reads the prepared geometry."""
    return _STAGES[stage].geometry


def writes_files(stage: str) -> bool:
    """Whether ``stage``, when it runs, writes files, and so needs a directory of its own."""
    return _STAGES[stage].writes


def stage_output(stage: str) -> str:
    """The name of the main output of ``stage`` (``events``)."""
    return _STAGES[stage].output


def executors(stage: str) -> Mapping[str, Executor]:
    """The executors that can run ``stage``, by name."""
    return _STAGES[stage].executors


def graph(targets: Collection[str], given: Collection[str]) -> list[tuple[str, tuple[str, ...]]]:
    """The nodes that make ``targets`` available, each with the nodes it reads, dependencies
    first. A node in ``given`` is a leaf: it is not run, so it reads nothing."""
    order: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        reads = () if name in given else _STAGES[name].needs
        for need in reads:
            visit(need)
        order.append((name, reads))

    for target in targets:
        if target not in _STAGES:
            raise EvaluationContractError(
                f"unknown stage {target!r}; available: {', '.join(sorted(_STAGES))}"
            )
        visit(target)
    return order


def plan_stages(*targets: str, provided: Collection[str] = ()) -> list[str]:
    """The stages to run, dependencies first, to make every stage of ``targets`` available.

    Each stage appears once, however many targets need it. A stage whose name is in
    ``provided`` (a given file) is not run, and neither are the stages only it would need.
    """
    return [name for name, _ in graph(targets, provided) if name not in provided]
