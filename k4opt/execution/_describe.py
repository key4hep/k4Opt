"""Internal: each task of a candidate described as plain data, from the results of what it reads.

:func:`describe` turns a planned task of a prepared candidate into the
:class:`~k4opt.execution._tasks.TaskSpec` a backend executes: the files the task is given, named in
the candidate's workspace, what it checks of the files its reads made, the stages of its sample as
the candidate configures them, the request of an evaluation (with what each role of a combined
evaluator reads), how many processors it keeps busy and how long it may run. :func:`seen` is the
stats taken of those files where they were made, handed along with the description.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from k4opt.evaluation import Artifact, RoleInput
from k4opt.execution._binding import task_cpus
from k4opt.execution._files import NOTHING, Observed, portable, portable_artifact
from k4opt.execution._preflight import _Prepared
from k4opt.execution._reuse import REUSED
from k4opt.execution._tasks import (
    GEOMETRY_TIMEOUT,
    GeometryWork,
    Observations,
    TaskSpec,
    Workspace,
)
from k4opt.frozen import FrozenMap
from k4opt.problem.models import EvaluatorSpec, Reference
from k4opt.problem.plan import GEOMETRY, Kind, Task, TaskId, reference_task
from k4opt.problem.stages import stage_output
from k4opt.record.results import TaskResult


def _inputs(reads: Mapping[TaskId, TaskResult], root: Path | None) -> dict[str, Path]:
    """What a task is given of the tasks it reads: the main output of each, by its name, named
    in the candidate's workspace (:func:`~k4opt.execution._files.portable`)."""
    return {
        task.name: portable(result.output, root)
        for task, result in reads.items()
        if result.output is not None
    }


def _role_inputs(
    prepared: _Prepared,
    entry: EvaluatorSpec,
    reads: Mapping[TaskId, TaskResult],
    root: Path | None,
) -> dict[str, tuple[RoleInput, ...]]:
    """What each role of a combined evaluator reads, from the results of the tasks it reads, its
    files named in the candidate's workspace."""
    return {
        role: tuple(
            _role_input(prepared, item, reads[reference_task(item)], root) for item in items
        )
        for role, items in entry.roles.items()
    }


def _role_input(
    prepared: _Prepared, item: Reference, result: TaskResult, root: Path | None
) -> RoleInput:
    """One item of a role, from the result of its task, which succeeded: an evaluation with what
    it measured and made, or a stage with its file."""
    properties = prepared.binding.properties[item.sample]
    if item.evaluator is not None:
        record = result.record
        assert record is not None  # every evaluation has one
        return RoleInput(
            item.sample,
            binding=item.evaluator,
            measurements=record.result.measurements,
            artifacts={
                name: portable_artifact(artifact, root) for name, artifact in result.made.items()
            },
            events=record.request.events,
            seed=record.request.seed,
            properties=properties,
            settings=record.request.settings,
            provenance=record.result.provenance,
        )
    assert item.stage is not None
    assert result.output is not None  # a stage that makes a file
    name = stage_output(item.stage)
    made = result.made.get(name)
    if made is not None:  # made in its directory, hashed when it was made
        artifact = made
    else:  # a file of the problem or the caller's, hashed when the session opened, or reused
        given = result.provenance.get(REUSED)
        identity = given if isinstance(given, Mapping) else result.provenance
        size, sha256 = identity.get("bytes"), identity.get("sha256")
        artifact = Artifact(
            f"{result.id.label}#{name}",
            result.output,
            size if isinstance(size, int) else result.output.stat().st_size,
            sha256 if isinstance(sha256, str) else None,
        )
    events, seed = prepared.contexts[item.sample].conditions(item.stage)
    return RoleInput(
        item.sample,
        stage=item.stage,
        artifacts={item.stage: portable_artifact(artifact, root)},
        events=events,
        seed=seed,
        properties=properties,
        provenance=result.provenance,
    )


def describe(
    task: Task,
    prepared: _Prepared,
    workspace: Workspace,
    results: Mapping[TaskId, TaskResult],
    observed: Mapping[TaskId, Observed],
    *,
    supplied: Path | None = None,
    geometry: GeometryWork | None = None,
) -> TaskSpec:
    """The description of ``task`` of a candidate ``prepared`` to run in ``workspace``, from the
    ``results`` of the tasks it reads and what is ``observed`` of the files they made; a stage
    the caller ``supplied`` a file for, and how the ``geometry`` is made. It may run as long as
    the problem says: the ``timeout`` of its stage or evaluator, or the problem's
    (:data:`~k4opt.execution._tasks.GEOMETRY_TIMEOUT` for the geometry)."""
    root = workspace.root
    reads = {need: results[need] for need in task.needs}
    watching = _watching(task, observed)
    watched = tuple(
        portable_artifact(made.artifact, root) for each in watching for made in each.made.values()
    )
    trees = [each.tree for each in watching if each.tree is not None]
    tree = None if not trees else portable(trees[0].root, root)
    default = prepared.binding.timeout
    if task.id.kind is Kind.GEOMETRY:
        return TaskSpec(
            workspace.id,
            task,
            watched=watched,
            tree=tree,
            geometry=geometry,
            timeout=GEOMETRY_TIMEOUT,
        )
    if task.id.kind is Kind.STAGE:
        assert task.id.sample is not None
        stages = prepared.contexts[task.id.sample].stages  # as the candidate configures them
        stage = stages[task.id.name]
        return TaskSpec(
            workspace.id,
            task,
            _inputs(reads, root),
            watched,
            tree,
            stages,
            supplied=supplied,
            cpus=task_cpus(prepared.contexts, task),
            # a given file runs nothing
            timeout=None if task.given is not None else _limit(stage.timeout, default),
        )
    entry = prepared.entries[task.id]
    if task.id.sample is None:  # the geometry, and what its roles read: no stage of its own
        given = _inputs({need: result for need, result in reads.items() if need == GEOMETRY}, root)
        roles = _role_inputs(prepared, entry, reads, root)
    else:
        given, roles = _inputs(reads, root), {}
    request = prepared.requests[task.id]
    timeout = _limit(entry.timeout, default)
    return TaskSpec(
        workspace.id,
        task,
        given,
        watched,
        tree,
        request=request,
        roles=roles,
        timeout=timeout,
        declined=prepared.declined.get(task.id),
    )


def _limit(own: float | None, default: float | None) -> float | None:
    """The time limit of a stage or an evaluation: its own ``timeout``, or the problem's."""
    limit = default if own is None else own
    return None if limit is None else float(limit)


def _watching(task: Task, observed: Mapping[TaskId, Observed]) -> list[Observed]:
    """What is watched of the files the tasks ``task`` reads made, in the order it reads them."""
    return [observed.get(need, NOTHING) for need in task.needs]


def seen(task: Task, observed: Mapping[TaskId, Observed]) -> Observations:
    """The stats taken of the files the tasks ``task`` reads made, where they were made."""
    watching = _watching(task, observed)
    made = {m.artifact.ref: m.stat for each in watching for m in each.made.values()}
    trees = [each.tree.stats for each in watching if each.tree is not None]
    return Observations(FrozenMap(made), trees[0] if trees else None)
