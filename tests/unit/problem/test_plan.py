"""The tasks of a candidate, planned from the problem alone: what each reads, where it writes and
whether the candidate's status depends on it (no Key4hep)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from k4opt.evaluation import EvaluationContractError
from k4opt.problem import EvaluatorSpec, MetricRef, Reference, Sample, Stage
from k4opt.problem.plan import (
    GEOMETRY,
    Kind,
    Plan,
    Task,
    TaskId,
    evaluator_task,
    plan_tasks,
    stage_task,
)

GUN = Stage("generation", executor="gun", options={"particle": "gamma"})
DDSIM = Stage("simulation", executor="ddsim", events=3, seed=1)
GENERATION, SIMULATION = stage_task("generation", "gamma"), stage_task("simulation", "gamma")
GAMMA = {"gamma": Sample("gamma", {"generation": GUN, "simulation": DDSIM})}


def ev(binding: str, stage: str | None = None, *samples: str) -> EvaluatorSpec:
    """An evaluator ``binding`` that reads ``stage`` of ``samples`` (``gamma`` by default), or
    measures the candidate without a stage."""
    return EvaluatorSpec(
        binding, "1", stage=stage, samples=samples or (("gamma",) if stage else ())
    )


def labels(plan: Plan) -> dict[str, list[str]]:
    """Each task of ``plan`` by label, in plan order, with the labels of what it reads."""
    return {task.id.label: [need.label for need in task.needs] for task in plan.tasks}


def test_a_simulation_reader_reads_the_geometry_and_every_stage_of_its_chain() -> None:
    plan = plan_tasks(GAMMA, [ev("calo", "simulation")], references=())
    assert labels(plan) == {
        "geometry": [],
        "gamma/generation": [],  # a gun is fired by ddsim: it reads no geometry
        "gamma/simulation": ["geometry", "gamma/generation"],
        "gamma/calo": ["geometry", "gamma/generation", "gamma/simulation"],
    }
    directories = {task.id.label: task.directory for task in plan.tasks}
    assert directories == {
        "geometry": "geometry",
        "gamma/generation": None,  # writes nothing
        "gamma/simulation": "samples/gamma/simulation",
        "gamma/calo": "samples/gamma/evaluators/calo",
    }


def test_every_evaluator_reads_the_geometry_even_one_that_reads_no_stage() -> None:
    plan = plan_tasks({}, [ev("cost"), ev("echo")], references=())
    assert labels(plan) == {"geometry": [], "cost": ["geometry"], "echo": ["geometry"]}
    assert all(GEOMETRY in task.needs for task in plan.of(Kind.EVALUATOR))
    assert plan[evaluator_task("cost")].directory == "evaluators/cost"


def test_each_stage_is_planned_once_however_many_evaluators_read_it() -> None:
    evaluators = [ev("a", "simulation"), ev("b", "generation"), ev("c", "simulation")]
    plan = plan_tasks(GAMMA, evaluators, references=())
    assert list(labels(plan)) == [
        "geometry",
        "gamma/generation",
        "gamma/simulation",
        "gamma/a",
        "gamma/b",
        "gamma/c",
    ]
    assert labels(plan)["gamma/b"] == ["geometry", "gamma/generation"]


def test_each_sample_is_planned_on_its_own_and_reads_only_its_own_stages() -> None:
    samples = GAMMA | {"pion": Sample("pion", {"generation": GUN, "simulation": DDSIM})}
    evaluators = [
        ev("calo", "simulation", "gamma", "pion"),
        ev("cost"),
        ev("edge", "generation", "pion"),
    ]
    plan = plan_tasks(samples, evaluators, references=())
    assert labels(plan) == {
        "geometry": [],
        "gamma/generation": [],
        "gamma/simulation": ["geometry", "gamma/generation"],
        "gamma/calo": ["geometry", "gamma/generation", "gamma/simulation"],
        "pion/generation": [],
        "pion/simulation": ["geometry", "pion/generation"],
        "pion/calo": ["geometry", "pion/generation", "pion/simulation"],
        "pion/edge": ["geometry", "pion/generation"],
        "cost": ["geometry"],  # the candidate's after the samples'
    }
    assert plan[evaluator_task("calo", "pion")].directory == "samples/pion/evaluators/calo"
    # a sample that only a generation reader measures does not simulate
    only = plan_tasks(
        samples,
        [ev("calo", "simulation", "gamma"), ev("edge", "generation", "pion")],
        references=(),
    )
    assert stage_task("simulation", "pion") not in [task.id for task in only.tasks]


def test_a_combined_evaluator_reads_the_geometry_and_each_item_of_its_roles() -> None:
    samples = GAMMA | {"pion": Sample("pion", {"generation": GUN, "simulation": DDSIM})}
    calo = ev("calo", "simulation", "gamma")
    roles = {
        "signal": (Reference("gamma", evaluator="calo"),),
        # a stage no evaluator of the sample reads is planned for the role that reads it
        "events": (Reference("pion", stage="simulation"), Reference("gamma", evaluator="calo")),
    }
    sep = EvaluatorSpec("sep", "1", roles=roles)
    plan = plan_tasks(samples, [sep, calo], references=())
    assert labels(plan) == {
        "geometry": [],
        "gamma/generation": [],
        "gamma/simulation": ["geometry", "gamma/generation"],
        "gamma/calo": ["geometry", "gamma/generation", "gamma/simulation"],
        "pion/generation": [],
        "pion/simulation": ["geometry", "pion/generation"],
        # each item once, in the order the roles name them; the stages before them are theirs
        "sep": ["geometry", "gamma/calo", "pion/simulation"],
    }
    assert plan[evaluator_task("sep")].directory == "evaluators/sep"


def test_a_stage_given_as_a_file_reads_nothing() -> None:
    hepmc = Stage("generation", file="zh.hepmc")
    sample = {"gamma": Sample("gamma", {"generation": hepmc, "simulation": DDSIM})}
    plan = plan_tasks(sample, [ev("calo", "simulation")], references=())
    assert plan[GENERATION] == Task(GENERATION, given="file")
    assert labels(plan)["gamma/simulation"] == ["geometry", "gamma/generation"]


def test_a_supplied_stage_reads_nothing_and_the_stages_only_it_needs_are_not_planned() -> None:
    samples = GAMMA | {"pion": Sample("pion", {"generation": GUN, "simulation": DDSIM})}
    calo = [ev("calo", "simulation", "gamma", "pion")]
    plan = plan_tasks(samples, calo, {SIMULATION}, references=())
    assert labels(plan) == {
        "geometry": [],
        "gamma/simulation": [],
        "gamma/calo": ["geometry", "gamma/simulation"],
        # supplied for gamma only
        "pion/generation": [],
        "pion/simulation": ["geometry", "pion/generation"],
        "pion/calo": ["geometry", "pion/generation", "pion/simulation"],
    }
    assert plan[SIMULATION].given == "supplied"
    # supplied wins over a file the problem gives
    filed = {"gamma": Sample("gamma", {"simulation": Stage("simulation", file="events.root")})}
    assert plan_tasks(filed, [ev("calo", "simulation")], {SIMULATION}, references=())[
        SIMULATION
    ].given == ("supplied")


def test_the_task_kind_and_sample_tell_tasks_of_the_same_name_apart() -> None:
    assert stage_task("x", "gamma") != evaluator_task("x", "gamma") != TaskId(Kind.GEOMETRY, "x")
    assert evaluator_task("x", "gamma") != evaluator_task("x", "pion") != evaluator_task("x")
    assert (evaluator_task("x", "gamma").label, evaluator_task("x").label) == ("gamma/x", "x")
    with pytest.raises(KeyError):
        plan_tasks({}, [ev("calo")], references=())[stage_task("calo", "gamma")]


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        ((Task(GEOMETRY), Task(GEOMETRY)), "task 'geometry' is planned twice"),
        (
            (Task(evaluator_task("calo"), (SIMULATION,)), Task(SIMULATION)),
            "task 'calo' reads gamma/simulation, which is not planned before it",
        ),
        ((Task(SIMULATION, (SIMULATION,)),), "task 'gamma/simulation' reads gamma/simulation"),
    ],
)
def test_a_plan_runs_each_task_once_and_after_what_it_reads(
    tasks: tuple[Task, ...], message: str
) -> None:
    with pytest.raises(EvaluationContractError, match=message):
        Plan(tasks)


# --- which tasks are required ---


def needed(plan: Plan) -> list[str]:
    return [task.id.label for task in plan.tasks if task.required]


def test_the_tasks_required_are_those_an_objective_or_constraint_names_and_what_they_read() -> None:
    evaluators = [ev("calo", "simulation"), ev("occupancy", "simulation"), ev("cost")]
    calo = MetricRef("calo", "simulation", "x", "gamma")
    plan = plan_tasks(GAMMA, evaluators, references=[calo])
    assert needed(plan) == ["geometry", "gamma/generation", "gamma/simulation", "gamma/calo"]
    assert [task.id.label for task in plan.diagnostics] == ["gamma/occupancy", "cost"]
    # several metrics of one evaluation, or of the candidate: each task once
    cost = MetricRef("cost", "geometry", "channels")
    plan = plan_tasks(GAMMA, evaluators, references=[calo, replace(calo, metric="y"), cost])
    assert [task.id.label for task in plan.diagnostics] == ["gamma/occupancy"]


def test_an_evaluator_marked_required_is_required_with_what_it_reads_on_every_sample() -> None:
    samples = GAMMA | {"pion": Sample("pion", {"generation": GUN, "simulation": DDSIM})}
    check = replace(ev("check", "simulation", "gamma", "pion"), required=True)
    plan = plan_tasks(samples, [ev("cost"), check], references=[MetricRef("cost", "g", "c")])
    assert needed(plan) == [
        "geometry",
        "gamma/generation",
        "gamma/simulation",
        "gamma/check",
        "pion/generation",
        "pion/simulation",
        "pion/check",
        "cost",
    ]
    assert plan.diagnostics == ()


def test_what_a_required_combined_evaluator_reads_is_required_and_only_that() -> None:
    samples = {name: Sample(name, {"generation": GUN, "simulation": DDSIM}) for name in "ab"}
    shape = ev("shape", "simulation", "a", "b")
    occupancy = ev("occupancy", "simulation", "b")
    separation = EvaluatorSpec(
        "separation", "1", roles={"signal": (Reference("a", evaluator="shape"),)}
    )
    plan = plan_tasks(
        samples,
        [shape, occupancy, separation],
        references=[MetricRef("separation", "analysis", "rejection")],
    )
    # b is simulated for its diagnostics alone: no objective needs its events
    assert [task.id.label for task in plan.diagnostics] == [
        "b/generation",
        "b/simulation",
        "b/shape",
        "b/occupancy",
    ]
    assert needed(plan) == ["geometry", "a/generation", "a/simulation", "a/shape", "separation"]
