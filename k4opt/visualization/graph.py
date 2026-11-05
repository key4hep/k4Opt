"""The tasks that evaluate a candidate, as a graph: what each parameter sets, what each task reads,
and which evaluations the objectives and constraints are measured by.

:func:`task_graph` reads it from the plan of the problem (:func:`~k4opt.problem.plan.plan_tasks`),
the one every candidate is evaluated by. Its nodes are the parameters, the geometry, the stages and
evaluations of each sample, the evaluations of the candidate, and the objectives and constraints;
its edges are of three kinds:

* ``sets``: a parameter to each task it sets something of (the geometry, an option of a stage of
  some samples, a setting of an evaluator), labeled with what it sets; the task says what is
  varied of it (``varies ECal_cell_size``), and by which parameter only where several vary it;
* ``reads``: a task to a task that reads it, but only where no other path implies it (the
  geometry reaches an evaluation of a sample through its simulation); an item of a role of a
  combined evaluator is labeled with the role;
* ``measures``: an evaluation to each objective and constraint it measures.

A diagnostic, a task no objective, constraint or ``required: true`` needs, is not ``required``.
Of a study, each node also holds the recorded times of its stage, one per trial that recorded one,
and the trials it did not succeed in as a diagnostic.

Samples alike but for the options of their stages (an energy scan: the same stages, evaluators
and objectives, each sample another energy) are **stacked**: drawn as one box, each task in it
standing for that task of every sample, unless the stack is ``expanded``. How it is drawn is the
figures': :func:`~k4opt.visualization.pipeline_graph`, :func:`~k4opt.visualization.pipeline_figure`.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Literal

from k4opt.evaluation import Scalar
from k4opt.frozen import Json
from k4opt.optimization import StudyResults
from k4opt.problem import (
    Constraint,
    GeometryDestination,
    Names,
    Objective,
    Parameter,
    ProblemSpec,
    SettingDestination,
    Stage,
    StageDestination,
    names,
)
from k4opt.problem.plan import GEOMETRY, Kind, Plan, Task, TaskId, level_plan, reference_task
from k4opt.problem.stages import executors
from k4opt.visualization.errors import VisualizationError

type NodeKind = Literal["parameter", "geometry", "stage", "evaluator", "objective", "constraint"]
type EdgeKind = Literal["sets", "reads", "measures"]

_LONGEST = 32  # characters of a value shown in a detail; the facts hold all of it


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One node: its ``id`` (a task's label, ``gamma/simulation``; ``parameter:<name>``,
    ``objective:<label>``, ``constraint:<label>``), what it is, its ``title`` and the short
    ``details`` a figure shows under it, and its ``facts``, everything the problem says of it, as
    (name, value) pairs.

    ``sample`` is the box (:attr:`TaskGraph.samples`) of a task of a sample, or the sample an
    objective or constraint is measured on; ``None`` for a node of no sample. ``group`` is what it
    is in every sample alike (``stage:simulation``, ``evaluator:calo.photon``), ``samples`` the
    samples a node of a stack stands for (empty for any other). ``required`` is ``False`` for a
    diagnostic, ``given`` ``True`` for a stage given as a file. Of a study, ``seconds`` holds the
    recorded times of a stage, by trial number (of a stack, each trial's mean over its samples),
    and ``failed`` the numbers of the trials a diagnostic did not succeed in. ``loop`` is, of a
    study with levels below its own, the innermost level whose trials it runs once for: the deepest
    level whose searched parameters reach it (a parameter: its own level), ``None`` outside every
    loop.
    """

    id: str
    kind: NodeKind
    title: str
    details: tuple[str, ...] = ()
    facts: tuple[tuple[str, str], ...] = ()
    sample: str | None = None
    group: str | None = None
    required: bool = True
    given: bool = False
    seconds: Mapping[int, float] | None = None
    failed: tuple[int, ...] = ()
    samples: tuple[str, ...] = ()
    loop: int | None = None

    @property
    def median_seconds(self) -> float | None:
        """The median of the recorded times, ``None`` without any."""
        return statistics.median(self.seconds.values()) if self.seconds else None


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """An edge from ``source`` to ``target``, by node id, of one ``kind``, with what it carries
    (``label``: what a parameter sets, the role an item is read as)."""

    source: str
    target: str
    kind: EdgeKind
    label: str | None = None


@dataclass(frozen=True, slots=True)
class SampleBox:
    """The box the tasks of a sample are drawn in, or of a stack of samples alike: its ``title``,
    the ``lines`` under it (a sample's properties; what tells the samples of a stack apart), and
    the ``members``, the samples it holds."""

    title: str
    lines: tuple[str, ...]
    members: tuple[str, ...]

    @property
    def is_stack(self) -> bool:
        return len(self.members) > 1


@dataclass(frozen=True, slots=True)
class LoopBox:
    """The frame of one level of a nested study, around what runs once for each of its trials:
    its ``level``, its ``title`` (``for each ECAL cell size``) and the ``lines`` under it (how many
    trials, what judges them; below the study's level, that its best is evaluated again)."""

    level: int
    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskGraph:
    """The nodes, in order: the parameters, the tasks in plan order, the objectives and the
    constraints; the edges; the boxes of the samples, by key (a sample's name; a stack's is that
    of its first sample); ``recorded``, the number of trials with a record (of a problem: 0); and
    of a study with levels below its own, the frame of each level from its own down
    (:class:`LoopBox`), each drawn in the one before, a node in the frame of its
    :attr:`GraphNode.loop`.
    """

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    samples: Mapping[str, SampleBox]
    recorded: int = 0
    loops: tuple[LoopBox, ...] = ()

    def node(self, id: str) -> GraphNode:
        """The node ``id``; :class:`VisualizationError` if the graph has none."""
        for node in self.nodes:
            if node.id == id:
                return node
        raise VisualizationError(f"the graph has no node {id!r}")


def task_graph(
    source: StudyResults | ProblemSpec, *, stack: bool = True, expanded: Collection[str] = ()
) -> TaskGraph:
    """The tasks of a candidate of ``source``; of a study, with the recorded times of each stage and
    the trials each diagnostic did not succeed in, and of a study of a level below the top, the
    tasks that level runs, judged by its criteria, the parameters it fixes shown with their value.
    Samples alike are stacked unless ``stack`` is false or the stack's key (its first sample's
    name) is in ``expanded``."""
    results = source if isinstance(source, StudyResults) else None
    spec = source.problem if isinstance(source, StudyResults) else source
    level = 0 if results is None else results.scope.level
    fixed = {} if results is None else results.scope.fixed
    plan = level_plan(spec, level)
    times: dict[str, dict[int, float]] = {}
    failures: dict[str, list[int]] = {}
    for trial in results.trials if results is not None else ():
        for label, seconds in trial.stage_seconds.items():
            times.setdefault(label, {})[trial.number] = seconds
        for label in dict.fromkeys(str(entry["task"]) for entry in trial.diagnostics):
            failures.setdefault(label, []).append(trial.number)
    called = names(spec)
    sets: dict[TaskId, list[tuple[str, str]]] = {}  # (what, parameter name) by task
    edges: list[GraphEdge] = []
    nodes = [
        _parameter(parameter, called, fixed.get(parameter.name)) for parameter in spec.parameters
    ]
    for parameter in spec.parameters:
        for target, name in _sets(plan, parameter):
            sets.setdefault(target, []).append((name, parameter.name))
            edges.append(GraphEdge(f"parameter:{parameter.name}", target.label, "sets", name))
    nodes += [
        _task(spec, task, sets.get(task.id, []), called, times.get(task.id.label), failures)
        for task in plan.tasks
    ]
    roles = _roles(spec)
    edges += [
        GraphEdge(need.label, task.id.label, "reads", roles.get((need, task.id)))
        for task in plan.tasks
        for need in _reduced(plan, task)
    ]
    criteria = spec.criteria(level)
    measured: list[Objective | Constraint] = [*criteria.objectives, *criteria.constraints]
    # in its sample's box a measurement is called by its title or metric alone, the box saying
    # the sample, unless another of its kind in that box is called alike
    alike = Counter((type(item), item.sample, item.title or item.metric) for item in measured)
    for item in measured:
        node = _measured(item, alike[type(item), item.sample, item.title or item.metric] > 1)
        nodes.append(node)
        edges.append(
            GraphEdge(
                TaskId(Kind.EVALUATOR, item.evaluator, item.sample).label, node.id, "measures"
            )
        )
    samples = {
        name: SampleBox(
            called.sample(name),
            tuple(f"{key}: {_value(value)}" for key, value in sample.properties.items()),
            (name,),
        )
        for name, sample in spec.samples.items()
        if any(task.id.sample == name for task in plan.tasks)
    }
    recorded = 0 if results is None else sum(t.record is not None for t in results.trials)
    loops: tuple[LoopBox, ...] = ()
    if spec.depth > level:
        nodes, edges = _looped(spec, plan, nodes, edges, level, fixed)
        loops = tuple(
            _loop(spec, at, level, called, results) for at in range(level, spec.depth + 1)
        )
    graph = TaskGraph(tuple(nodes), tuple(edges), samples, recorded, loops)
    return _stacked(graph, expanded) if stack else graph


def _parameter(parameter: Parameter, called: Names, value: Scalar | None = None) -> GraphNode:
    """A parameter's node: its domain, or the ``value`` the study gives it for every trial."""
    facts = [("parameter", parameter.name)] if parameter.title is not None else []
    facts += [("type", parameter.kind), ("domain", _domain(parameter, short=False))]
    if value is not None:
        facts.append(("fixed", _fixed(parameter, value)))
    facts += [("sets", destination.label) for destination in parameter.targets]
    return GraphNode(
        f"parameter:{parameter.name}",
        "parameter",
        called.parameter(parameter.name),
        (_domain(parameter) if value is None else f"fixed: {_fixed(parameter, value)}",),
        tuple(facts),
    )


def _looped(
    spec: ProblemSpec,
    plan: Plan,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    level: int,
    fixed: Mapping[str, Scalar],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """``nodes`` each in its loop (:attr:`GraphNode.loop`), from the study's ``level`` down, and
    ``edges``, what an evaluation of a level below measures for this one labeled: its best,
    measured again on new events. A task that does no work of its own (the particle gun, fired
    by ddsim) runs in the outermost loop of the working tasks of its sample."""
    searched = {
        name: at
        for at in range(level, spec.depth + 1)
        for name in spec.parameters_of(at)
        if name not in fixed
    }
    loops: dict[str, int | None] = {}
    for task in plan.tasks:
        reached = [searched[name] for name in task.varies if name in searched]
        loops[task.id.label] = max(reached) if reached else None
    for task in plan.tasks:
        if task.directory is None and task.id.sample is not None:
            working = [
                loops[other.id.label]
                for other in plan.tasks
                if other.id.sample == task.id.sample and other.directory is not None
            ]
            known = [at for at in working if at is not None]
            loops[task.id.label] = min(known) if known and len(known) == len(working) else None
    for name, at in searched.items():
        loops[f"parameter:{name}"] = at
    looped = [replace(node, loop=loops.get(node.id)) for node in nodes]
    deeper = {id for id, at in loops.items() if at is not None and at > level}
    relabeled = [
        replace(edge, label="best, on new events")
        if edge.kind == "measures" and edge.source in deeper
        else edge
        for edge in edges
    ]
    return looped, relabeled


def _loop(
    spec: ProblemSpec, at: int, level: int, called: Names, results: StudyResults | None
) -> LoopBox:
    """The frame of the trials of level ``at``: for each of what it searches, how many trials and
    what judges them; below the study's ``level``, that its best is evaluated again."""
    searched = ", ".join(called.parameter(name) for name in spec.parameters_of(at))
    criteria = spec.criteria(at)
    if len(criteria.objectives) == 1:
        (objective,) = criteria.objectives
        arrow = "▲" if objective.direction == "maximize" else "▼"
        judged = f"{arrow} {called.objective(objective.label)}"
    else:
        judged = f"compared by its {len(criteria.objectives)} objectives"
    if at == level:
        trials = [] if results is None else [f"{len(results.trials)} trials"]
        return LoopBox(at, f"for each {searched}", (" · ".join([*trials, judged]),))
    above = ", ".join(called.parameter(name) for name in spec.parameters_of(at - 1))
    budget = spec.levels[at].trials
    return LoopBox(
        at,
        f"for each {searched} trial",
        (f"{budget} per {above} · {judged}", "best one evaluated again on new events"),
    )


def _fixed(parameter: Parameter, value: Scalar) -> str:
    shown = f"{value:.4g}" if isinstance(value, float) else str(value)
    return f"{shown} {parameter.unit}" if parameter.unit else shown


def _measured(item: Objective | Constraint, alike: bool) -> GraphNode:
    """An objective or a constraint, titled by its title or metric, by what it measures in its
    sample (``group``, as every sample alike) where another in its box is ``alike``."""
    group = f"{item.evaluator}/{item.stage}/{item.metric}"
    title = group if alike else item.title or item.metric
    if isinstance(item, Objective):
        arrow = "▲" if item.direction == "maximize" else "▼"
        facts = [("objective", item.label), ("direction", item.direction)]
        return GraphNode(
            f"objective:{item.label}",
            "objective",
            title,
            (f"{arrow} {item.direction}",),
            tuple(facts),
            item.sample,
            f"objective:{group}",
        )
    facts = [("constraint", item.label)]
    facts += [
        (bound, f"{getattr(item, bound):g}")
        for bound in ("min", "max")
        if getattr(item, bound) is not None
    ]
    if item.unit:
        facts.append(("unit", item.unit))
    if item.scale is not None:
        facts.append(("scale", f"{item.scale:g}"))
    return GraphNode(
        f"constraint:{item.label}",
        "constraint",
        title,
        (_bounds(item),),
        tuple(facts),
        item.sample,
        f"constraint:{group}",
    )


def _task(
    spec: ProblemSpec,
    task: Task,
    sets: Sequence[tuple[str, str]],
    called: Names,
    seconds: Mapping[int, float] | None,
    failures: Mapping[str, list[int]],
) -> GraphNode:
    """One task: what it is, how it is made and what parameters set of it (``sets``: what, and
    the name of the parameter that sets it)."""
    task_id = task.id
    details: list[str] = []
    facts: list[tuple[str, str]] = [("task", task_id.label)]
    group = None
    if task_id.kind is Kind.GEOMETRY:
        kind: NodeKind = "geometry"
        title = "geometry"
        details.append(PurePosixPath(spec.geometry).name)
        facts.append(("compact file", spec.geometry))
    elif task_id.kind is Kind.STAGE:
        assert task_id.sample is not None
        kind, title, group = "stage", task_id.name, f"stage:{task_id.name}"
        stage = spec.samples[task_id.sample].stages[task_id.name]
        details.append(_making(spec, task_id.sample, stage))
        if stage.executor is not None and executors(stage.type)[stage.executor].inside:
            seconds = None  # its work is that of the stage that runs it
        details += [f"{key}: {_short(value)}" for key, value in _differing(spec, stage)]
        facts += _stage_facts(stage)
    else:
        entry = next(e for e in spec.evaluators if e.binding == task_id.name)
        kind, title = "evaluator", entry.binding
        group = None if task_id.sample is None else f"evaluator:{entry.binding}"
        same = entry.name == entry.binding
        details.append(f"v{entry.version}" if same else f"{entry.name} v{entry.version}")
        facts += [("evaluator", entry.name), ("version", entry.version)]
        facts += [(f"setting {key}", _value(value)) for key, value in entry.settings.items()]
        facts += [(f"collection {role}", name) for role, name in entry.collections.items()]
    details += _varied(sets, called)
    if not task.required:
        details.append("diagnostic")
    facts.append(("reads", ", ".join(need.label for need in task.needs) or "nothing"))
    if task.varies:
        facts.append(("varied by", ", ".join(task.varies)))
    if task.directory is not None:
        facts.append(("directory", task.directory))
    facts.append(("required", "yes" if task.required else "no: a diagnostic"))
    return GraphNode(
        task_id.label,
        kind,
        title,
        tuple(details),
        tuple(facts),
        task_id.sample,
        group,
        task.required,
        task.given is not None,
        seconds,
        tuple(failures.get(task_id.label, ())),
    )


def _varied(sets: Sequence[tuple[str, str]], called: Names) -> list[str]:
    """What parameters vary of a task, each ``(what, parameter name)``: what alone where one
    parameter varies it all, as the edge from that parameter says which (``varies
    ECal_cell_size``); with what each parameter is called where several do (``physics.rangecut ←
    Range cut``)."""
    if not sets:
        return []
    if len({parameter for _, parameter in sets}) == 1:
        return ["varies " + ", ".join(what for what, _ in sets)]
    return [f"{what} ← {called.parameter(parameter)}" for what, parameter in sets]


def _making(spec: ProblemSpec, sample: str, stage: Stage) -> str:
    """How a stage is made: from a file, or by an executor (``ddsim · 1000 events``)."""
    if stage.file is not None:
        return f"file {PurePosixPath(stage.file).name}"
    assert stage.executor is not None
    inside = executors(stage.type)[stage.executor].inside
    if inside is not None:
        runner = spec.samples[sample].stages.get(inside)
        by = inside if runner is None or runner.executor is None else runner.executor
        return f"{stage.executor}, run by {by}"
    if stage.events is not None:
        return f"{stage.executor} · {stage.events} events"
    return stage.executor


def _differing(spec: ProblemSpec, stage: Stage) -> list[tuple[str, Json]]:
    """The options of ``stage`` that the same stage of another sample sets otherwise: what tells
    the samples apart (``particle``, ``energy``)."""
    others = [
        sample.stages[stage.type]
        for sample in spec.samples.values()
        if stage.type in sample.stages and sample.stages[stage.type] is not stage
    ]
    if not others:
        return []
    return [
        (key, value)
        for key, value in stage.options.items()
        if any(other.options.get(key) != value for other in others)
    ]


def _stage_facts(stage: Stage) -> list[tuple[str, str]]:
    facts = [
        (name, str(value))
        for name, value in (
            ("file", stage.file),
            ("executor", stage.executor),
            ("events", stage.events),
            ("seed", stage.seed),
        )
        if value is not None
    ]
    return facts + [(f"option {key}", _value(value)) for key, value in stage.options.items()]


def _reduced(plan: Plan, task: Task) -> list[TaskId]:
    """The needs of ``task`` that no other need of it reaches: the edges no path implies."""
    reach: dict[TaskId, set[TaskId]] = {}
    for each in plan.tasks:  # each after its needs
        reach[each.id] = set(each.needs).union(*(reach[need] for need in each.needs))
    implied = set().union(*(reach[need] for need in task.needs))
    return [need for need in task.needs if need not in implied]


def _roles(spec: ProblemSpec) -> dict[tuple[TaskId, TaskId], str]:
    """The role each item of a combined evaluator is read as, by (item, evaluation) edge."""
    found: dict[tuple[TaskId, TaskId], list[str]] = {}
    for entry in spec.evaluators:
        evaluation = TaskId(Kind.EVALUATOR, entry.binding)
        for role, items in entry.roles.items():
            for item in items:
                found.setdefault((reference_task(item), evaluation), []).append(role)
    return {edge: ", ".join(names) for edge, names in found.items()}


def _sets(plan: Plan, parameter: Parameter) -> list[tuple[TaskId, str]]:
    """The tasks ``parameter`` sets something of, each with the name it sets."""
    planned = [task.id for task in plan.tasks]
    found: list[tuple[TaskId, str]] = []
    for destination in parameter.targets:
        if isinstance(destination, GeometryDestination):
            found.append((GEOMETRY, destination.reference))
        elif isinstance(destination, StageDestination):
            found += [
                (TaskId(Kind.STAGE, destination.stage, sample), destination.option)
                for sample in destination.samples
                if TaskId(Kind.STAGE, destination.stage, sample) in planned
            ]
        elif isinstance(destination, SettingDestination):
            found += [
                (task, destination.setting)
                for task in planned
                if task.kind is Kind.EVALUATOR and task.name == destination.evaluator
            ]
    return found


def _domain(parameter: Parameter, *, short: bool = True) -> str:
    unit = f" {parameter.unit}" if parameter.unit else ""
    if parameter.choices is not None:
        choices = ", ".join(map(str, parameter.choices))
        return (_short(choices) if short else choices) + unit
    assert parameter.bounds is not None
    low, high = parameter.bounds
    return f"[{low:g}, {high:g}]{unit}"


def _bounds(constraint: Constraint) -> str:
    unit = f" {constraint.unit}" if constraint.unit and constraint.unit != "1" else ""
    if constraint.min is not None and constraint.max is not None:
        return f"{constraint.min:g} ≤ value ≤ {constraint.max:g}{unit}"
    if constraint.min is not None:
        return f"≥ {constraint.min:g}{unit}"
    assert constraint.max is not None
    return f"≤ {constraint.max:g}{unit}"


def duration(seconds: float) -> str:
    """``seconds`` as a person reads a duration: in s, min or h, three digits."""
    if seconds < 60:
        return f"{seconds:.3g} s"
    if seconds < 3600:
        return f"{seconds / 60:.3g} min"
    return f"{seconds / 3600:.3g} h"


def _value(value: Json) -> str:
    """A JSON value as a person reads it: a quantity as ``20 GeV``, a list in brackets."""
    if isinstance(value, Mapping):
        if set(value) == {"value", "unit"}:
            return f"{_value(value['value'])} {value['unit']}"
        return "{" + ", ".join(f"{key}: {_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, Sequence) and not isinstance(value, str):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _short(value: Json) -> str:
    text = _value(value)
    return text if len(text) <= _LONGEST else text[: _LONGEST - 1] + "…"


def status_lines(node: GraphNode, recorded: int) -> list[str]:
    """What a study says of ``node``: the median time of its stage and in how many trials it did
    not succeed as a diagnostic, as lines a figure shows under its details."""
    lines = []
    median = node.median_seconds
    if median is not None:
        assert node.seconds is not None
        each = " per sample" if node.samples else ""
        lines.append(f"median {duration(median)}{each} · {len(node.seconds)} trials")
    if node.failed:
        lines.append(f"⚠ did not succeed in {len(node.failed)} of {recorded} trials")
    return lines


# --- stacks ------------------------------------------------------------------------------------


_MEASURED = ("objective", "constraint")


def _stacked(graph: TaskGraph, expanded: Collection[str]) -> TaskGraph:
    """``graph`` with each set of samples alike drawn as one stack, but those in ``expanded``.

    Samples are alike when their tasks, objectives and constraints stand for the same things
    (by group), alike required and given, joined alike to each other and to the rest of the graph:
    the same parameters set the same of them, they are read in the same roles. What differs is
    only what the details say, which the stack's box then lists."""
    by_sample: dict[str, list[GraphNode]] = {}
    for node in graph.nodes:
        if node.sample is not None:
            by_sample.setdefault(node.sample, []).append(node)
    alike: dict[object, list[str]] = {}
    for sample in graph.samples:
        alike.setdefault(_signature(graph, sample, by_sample[sample]), []).append(sample)
    stacks = [members for members in alike.values() if len(members) > 1]
    stacks = [members for members in stacks if members[0] not in expanded]
    if not stacks:
        return graph
    renamed: dict[str, str] = {}
    merged: dict[str, GraphNode] = {}
    boxes = dict(graph.samples)
    for members in stacks:
        key = members[0]
        for node in by_sample[key]:
            group = [next(n for n in by_sample[s] if n.group == node.group) for s in members]
            stacked = _merge(node, group, key, members)
            merged[stacked.id] = stacked
            renamed |= {member.id: stacked.id for member in group}
        boxes[key] = SampleBox(
            f"{len(members)} samples",
            (
                _names([graph.samples[member].title for member in members]),
                *_differences(graph, members, by_sample),
            ),
            tuple(members),
        )
        for sample in members[1:]:
            del boxes[sample]
    nodes: list[GraphNode] = []
    for node in graph.nodes:
        shown = merged.get(renamed.get(node.id, node.id), node)
        if shown not in nodes:
            nodes.append(shown)
    edges = dict.fromkeys(
        replace(
            edge,
            source=renamed.get(edge.source, edge.source),
            target=renamed.get(edge.target, edge.target),
        )
        for edge in graph.edges
    )
    return TaskGraph(tuple(nodes), tuple(edges), boxes, graph.recorded, graph.loops)


def _signature(graph: TaskGraph, sample: str, nodes: Sequence[GraphNode]) -> object:
    """What ``sample`` is but for its name and its details: its nodes by group (and title: alike
    samples' objectives are titled alike), and its edges with each of its nodes named by group."""
    groups = {node.id: node.group for node in nodes}
    kinds = sorted((n.group or "", n.title, n.required, n.given) for n in nodes)

    def named(id: str) -> str:
        group = groups.get(id)
        return f"own:{group}" if group is not None else id

    edges = sorted(
        (named(e.source), named(e.target), e.kind, e.label or "")
        for e in graph.edges
        if e.source in groups or e.target in groups
    )
    return (tuple(kinds), tuple(edges))


def _merge(
    node: GraphNode, group: Sequence[GraphNode], key: str, members: Sequence[str]
) -> GraphNode:
    """One node of a stack standing for ``group``, the node of each of its samples: the details
    they share, each trial's mean time and every trial one of them did not succeed in."""
    details = tuple(d for d in node.details if all(d in other.details for other in group))
    kept = [(name, value) for name, value in node.facts if name not in ("task", "directory")]
    facts = (("samples", ", ".join(members)), *kept)
    seconds = None
    if all(member.seconds for member in group):
        trials = set.intersection(*(set(member.seconds or {}) for member in group))
        seconds = {t: statistics.fmean((m.seconds or {})[t] for m in group) for t in sorted(trials)}
    failed = tuple(sorted({number for member in group for number in member.failed}))
    measured = node.kind in _MEASURED
    if measured:
        details = (*details, f"one for each of {len(members)} samples")
    return replace(
        node,
        id=f"stack:{key}|{node.group}",
        details=details,
        facts=facts,
        sample=key,
        seconds=seconds or None,
        failed=failed,
        samples=tuple(members),
    )


def _names(members: Sequence[str], shown: int = 4) -> str:
    if len(members) <= shown:
        return ", ".join(members)
    return f"{', '.join(members[: shown - 1])}, … {members[-1]}"


def _differences(
    graph: TaskGraph, members: Sequence[str], by_sample: Mapping[str, Sequence[GraphNode]]
) -> list[str]:
    """What tells the samples of a stack apart: each detail ``name: value`` that differs between
    them, with the value of each, and each property that does."""
    lines: dict[str, list[str]] = {}
    for sample in members:
        named = [d for node in by_sample[sample] for d in node.details]
        named += list(graph.samples[sample].lines)
        for detail in named:
            name, colon, value = detail.partition(": ")
            if colon:
                lines.setdefault(name, []).append(value)
    return [
        f"{name}: {_short(', '.join(values))}"
        for name, values in lines.items()
        if len(values) == len(members) and len(set(values)) > 1
    ]


# --- collapsed samples -------------------------------------------------------------------------


COLLAPSED = "sample:"
"""The prefix of the id of a sample's box drawn as one node (``sample:<key>``)."""


def _outermost(loops: Sequence[int | None]) -> int | None:
    """The loop a node standing for nodes in ``loops`` is drawn in: the outermost of them, none if
    one of them is in none."""
    known = [at for at in loops if at is not None]
    return min(known) if known and len(known) == len(loops) else None


def collapsed(graph: TaskGraph, keys: Collection[str]) -> TaskGraph:
    """``graph`` with the box of each sample (or stack) in ``keys`` drawn as one node, so that a
    large graph shows what joins what before what each sample does.

    The node (``sample:<key>``, a stage) is titled as the box, lists its tasks in order and the
    box's lines, holds each trial's time summed over the box's stages and every trial a
    diagnostic in it did not succeed in, and is joined to whatever the box's tasks were joined to
    outside it, once each way and kind."""
    folded = {key: graph.samples[key] for key in keys if key in graph.samples}
    if not folded:
        return graph
    into = {
        node.id: f"{COLLAPSED}{node.sample}"
        for node in graph.nodes
        if node.sample in folded and node.kind not in _MEASURED
    }
    nodes: list[GraphNode] = []
    for node in graph.nodes:
        target = into.get(node.id)
        if target is None:
            nodes.append(node)
            continue
        if any(n.id == target for n in nodes):
            continue
        key = target.removeprefix(COLLAPSED)
        box = folded[key]
        members = [n for n in graph.nodes if into.get(n.id) == target]
        seconds: dict[int, float] = {}
        for member in members:
            for number, each in (member.seconds or {}).items():
                seconds[number] = seconds.get(number, 0.0) + each
        tasks = " → ".join(member.title for member in members)
        nodes.append(
            GraphNode(
                id=target,
                kind="stage",
                title=box.title,
                details=(tasks, *box.lines),
                facts=(("tasks", tasks), *(("sample", line) for line in box.lines)),
                required=any(member.required for member in members),
                seconds=seconds or None,
                failed=tuple(sorted({n for member in members for n in member.failed})),
                samples=box.members if box.is_stack else (),
                loop=_outermost([member.loop for member in members]),
            )
        )
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in graph.edges:
        source, target = into.get(edge.source, edge.source), into.get(edge.target, edge.target)
        if source == target or (source, target, edge.kind) in seen:
            continue
        seen.add((source, target, edge.kind))
        edges.append(replace(edge, source=source, target=target))
    samples = {key: box for key, box in graph.samples.items() if key not in folded}
    return TaskGraph(tuple(nodes), tuple(edges), samples, graph.recorded, graph.loops)
