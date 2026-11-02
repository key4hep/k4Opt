"""The pipeline stage types a problem's samples are made of, described once.

Each sample of a problem says how each of its stages is made: run by an executor (``gun`` for
``generation``, ``ddsim`` for ``simulation``) or given as a file. An evaluator names the stage it
reads with ``stage`` (``"simulation"``); the framework makes that stage of each sample it
measures, and the stages it depends on, available.

* ``generation``: where the primary particles come from. The ``gun`` executor has no file of
  its own (``ddsim`` fires it); a generation given as a file is an event file such as HepMC.
* ``simulation``: ``ddsim`` on the prepared geometry; reads ``generation``.

This module says what each stage type is: the stages it reads, its executors and the options
each takes, whether it reads the candidate's geometry or writes files, the name of its main
output and of its logs. :mod:`k4opt.problem.config` checks a problem against it, the plan of a
candidate's tasks (:mod:`k4opt.problem._tasks`) follows it, and a record is read back against
it. How a stage runs is the execution's (:mod:`k4opt.problem._stages`): a new stage type is one
entry here and one runner there.

Pipeline stages are steps that run. The ``stage`` of a measurement, objective or constraint
only classifies a metric: an evaluator may report ``analysis`` metrics without an ``analysis``
pipeline stage, and nothing here reads metric labels.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from k4opt.ddsim import LOG_FILES, check_options, check_particle_gun
from k4opt.evaluation import EvaluationContractError
from k4opt.frozen import Json


@dataclass(frozen=True, slots=True)
class Executor:
    """What runs a stage: whether it takes ``events`` (required) and ``seed``, what checks its
    own options (raising :class:`~k4opt.ddsim.DdsimError`) and the stage whose process runs it,
    if another one does (``inside``: the gun, fired by ddsim, which sets its events and seed)."""

    events: bool
    options: Callable[[Mapping[str, Json]], None]
    inside: str | None = None


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
        {"gun": Executor(events=False, options=check_particle_gun, inside="simulation")},
        geometry=False,
        writes=False,
    ),
    "simulation": StageType(
        ("generation",),
        {"ddsim": Executor(events=True, options=check_options)},
        geometry=True,
        writes=True,
        logs=LOG_FILES,
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


def changed_by_geometry(stage: str) -> bool:
    """Whether ``stage`` depends on the candidate geometry: it reads it, or reads a stage that
    does. A file given for such a stage was not made from the candidate."""
    return _STAGES[stage].geometry or any(changed_by_geometry(need) for need in stage_reads(stage))


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
