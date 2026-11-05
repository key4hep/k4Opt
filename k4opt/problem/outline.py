"""What a study of a problem will do, in words, before anything of it runs.

``k4opt problem explain`` prints it, and ``k4opt optimize`` when a new study starts. It follows
from the problem alone: the levels as loops around one another, each with its budget and
what picks its best trial, what judges the top level, and at most how many stages and evaluations
run (:func:`~k4opt.problem.plan.level_costs`)::

    For each ECAL cell size (8 trials):
      tune Pandora ECAL hit threshold: 12 trials maximizing π⁰ photons resolved
      evaluate the best again on new events (seed 42 → 506952155)
    compare them by π⁰ photons resolved (maximize) and Readout channels (minimize)
    at most, if every stage succeeds: simulation 16 · reconstruction 104 · pfo.pi0 104 · readout 8

Everything is called as the problem titles it (:func:`~k4opt.problem.names`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import TextIO

from k4opt.evaluation import Scalar
from k4opt.problem.candidate import check_fixed, replicate_seed
from k4opt.problem.models import ProblemSpec
from k4opt.problem.names import Names, names
from k4opt.problem.plan import Kind, level_costs, level_plan
from k4opt.terminal import Style, printable


def _listed(items: Iterable[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    *head, last = items
    return f"{', '.join(head)} and {last}" if head else last


def _trials(count: int) -> str:
    return f"{count} trial" if count == 1 else f"{count} trials"


def _seeds(spec: ProblemSpec, replicate: int) -> str:
    """Each seed of the problem's samples and what it becomes for ``replicate``, each once."""
    seeds = [s.seed for sample in spec.samples.values() for s in sample.stages.values()]
    found = dict.fromkeys(seed for seed in seeds if seed is not None)
    pairs = [f"{seed} → {replicate_seed(seed, replicate)}" for seed in found]
    return f"seed {', '.join(pairs)}" if len(pairs) == 1 else f"seeds {', '.join(pairs)}"


def _inner(spec: ProblemSpec, called: Names, level: int) -> list[str]:
    """The lines of the studies of ``level``, below the top, and of the levels below it."""
    indent = "  " * level
    searched = _listed(called.parameter(name) for name in spec.levels[level].parameters)
    trials = spec.levels[level].trials
    assert trials is not None  # parse_problem: every inner level has them
    (objective,) = spec.criteria(level).objectives
    verb = "maximizing" if objective.direction == "maximize" else "minimizing"
    budget = f"{_trials(trials)} {verb} {called.objective(objective.label)}"
    if level == spec.depth:
        lines = [f"{indent}tune {searched}: {budget}"]
    else:
        lines = [f"{indent}for each {searched} ({budget}):", *_inner(spec, called, level + 1)]
    # the replicate of the level above, whose trial the best stands for
    seeds = _seeds(spec, spec.depth - level + 1)
    return [*lines, f"{indent}evaluate the best again on new events ({seeds})"]


def _judged(spec: ProblemSpec, called: Names) -> str:
    """What the top level is judged by."""
    goals = _listed(f"{called.objective(o.label)} ({o.direction})" for o in spec.objectives)
    if not spec.constraints:
        return goals
    limits = _listed(called.constraint(c.label) for c in spec.constraints)
    return f"{goals}, within the limits on {limits}"


def _resimulated(spec: ProblemSpec, called: Names) -> list[str]:
    """A warning for each inner level with parameters that reach a simulation: its trials cannot
    share it, which is rarely what a level below the geometry is meant for."""
    warnings = []
    for level in range(1, spec.depth + 1):
        own = spec.levels[level].parameters
        reach = {
            name
            for task in level_plan(spec, level).of(Kind.STAGE)
            if task.id.name == "simulation"
            for name in task.varies
            if name in own
        }
        if reach:
            found = _listed(called.parameter(name) for name in own if name in reach)
            warnings.append(
                f"{found} of an inner level reach the simulation: each of its trials simulates "
                "anew instead of sharing one"
            )
    return warnings


def outline(
    spec: ProblemSpec, trials: int, fixed: Mapping[str, Scalar] | None = None
) -> tuple[list[str], list[str]]:
    """What a new study of ``spec`` of ``trials`` trials with the parameters ``fixed`` does, line
    by line, and the warnings about it. Raises :class:`~k4opt.problem.CandidateError` for values
    such a study refuses."""
    given = dict(fixed or {})
    check_fixed(spec, given)
    called = names(spec)
    searched = _listed(called.parameter(n) for n in spec.parameters_of() if n not in given)
    held_at = [p.name for p in spec.parameters if p.name in given]
    values = [f"{called.parameter(name)} = {given[name]}" for name in held_at]
    held = f", with {_listed(values)}" if values else ""
    if spec.depth:
        lines = [
            f"For each {searched} ({_trials(trials)}){held}:",
            *_inner(spec, called, 1),
            f"compare them by {_judged(spec, called)}",
        ]
    else:
        lines = [f"Search {searched}{held}: {_trials(trials)} by {_judged(spec, called)}"]
    costs: Counter[str] = Counter()
    for task, count in level_costs(spec, trials, given).items():
        costs[task.name] += count
    counted = " · ".join(f"{name} {count}" for name, count in costs.items())
    lines.append(f"at most, if every stage succeeds: {counted}")
    return lines, _resimulated(spec, called)


def render_outline(
    spec: ProblemSpec,
    trials: int,
    out: TextIO,
    *,
    fixed: Mapping[str, Scalar] | None = None,
    color: bool = False,
) -> None:
    """Print the :func:`outline` of a study of ``spec``."""
    style = Style(color)
    lines, warnings = outline(spec, trials, fixed)
    for line in lines:
        print(printable(line), file=out)
    for warning in warnings:
        print(style.yellow(f"warning: {printable(warning)}"), file=out)
