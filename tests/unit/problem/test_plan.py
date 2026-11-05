"""The tasks of a candidate, planned from the problem alone: what each reads, where it writes and
whether the candidate's status depends on it (no Key4hep)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from k4opt.evaluation import EvaluationContractError
from k4opt.problem import (
    EvaluatorSpec,
    GeometryDestination,
    Level,
    MetricRef,
    Objective,
    Parameter,
    ProblemSpec,
    Reference,
    Sample,
    SettingDestination,
    Stage,
    StageDestination,
)
from k4opt.problem.plan import (
    GEOMETRY,
    Kind,
    Plan,
    Task,
    TaskId,
    evaluator_task,
    level_costs,
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


def test_without_diagnostics_a_plan_holds_what_is_required_alone() -> None:
    # a level below the top: the cost of the candidate alone, so no stage of the sample runs
    evaluators = [ev("calo", "simulation"), ev("cost")]
    cost = MetricRef("cost", "geometry", "channels")
    assert [t.id.label for t in plan_tasks(GAMMA, evaluators, references=[cost]).tasks] == [
        "geometry",
        "gamma/generation",
        "gamma/simulation",
        "gamma/calo",
        "cost",
    ]
    plan = plan_tasks(GAMMA, evaluators, references=[cost], diagnostics=False)
    assert [task.id.label for task in plan.tasks] == ["geometry", "cost"]
    assert plan.diagnostics == ()
    # a required entry runs all the same, with what it reads
    check = replace(ev("calo", "simulation"), required=True)
    plan = plan_tasks(GAMMA, [check, ev("cost")], references=[cost], diagnostics=False)
    assert needed(plan) == [task.id.label for task in plan.tasks]
    assert "gamma/calo" in needed(plan)


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


def continuous(name: str, *targets: object) -> Parameter:
    return Parameter(name, "continuous", targets, bounds=(0.0, 1.0))  # type: ignore[arg-type]


def costs_of(spec: ProblemSpec, trials: int, fixed: tuple[str, ...] = ()) -> dict[str, int]:
    return {task.label: count for task, count in level_costs(spec, trials, fixed).items()}


CELL = continuous("cell", GeometryDestination("cell"))
GUN_ENERGY = continuous("energy", StageDestination("generation", "energy", ("gamma",)))
CUT = continuous("cut", SettingDestination("tuned", "cut"))
TUNED = Objective("tuned", "simulation", "m", "maximize", sample="gamma")
COST = Objective("cost", "geometry", "c", "minimize")


def problem(*parameters: Parameter, optimization: Level | None = None) -> ProblemSpec:
    """A simulation of ``gamma`` with an evaluator ``tuned`` of it, and ``cost`` of the candidate,
    each the evaluation of an objective."""
    return ProblemSpec(
        environment=None,  # type: ignore[arg-type]  # nothing here reads it
        geometry="detector.xml",
        parameters=parameters,
        evaluators=(ev("tuned", "simulation"), ev("cost")),
        objectives=(TUNED, COST),
        samples=GAMMA,
        optimization=optimization,
    )


def test_an_ordinary_study_makes_a_stage_for_what_reaches_it_and_evaluates_each_trial() -> None:
    assert costs_of(problem(CELL, CUT), 5) == {
        "gamma/simulation": 5,
        "gamma/tuned": 5,
        "cost": 5,
    }
    # no parameter reaches the simulation: every trial reuses one
    assert costs_of(problem(CUT), 5)["gamma/simulation"] == 1
    # nor does a parameter fixed for the whole study
    assert costs_of(problem(CELL, CUT), 5, ("cell",))["gamma/simulation"] == 1


def test_a_nested_study_simulates_once_per_candidate_and_level_and_evaluates_every_trial() -> None:
    inner = Level(("cut",), trials=20, objective=TUNED)
    spec = problem(CELL, CUT, optimization=Level(("cell",), inner=inner))
    assert costs_of(spec, 8) == {
        "gamma/simulation": 8 + 8,  # for the tuning, shared by its trials, and on new events
        "gamma/tuned": 8 * 20 + 8,
        "cost": 8,  # only the top level reads it
    }


def test_a_stage_an_inner_level_reaches_runs_for_every_evaluation() -> None:
    inner = Level(("energy",), trials=3, objective=TUNED)
    spec = problem(CELL, GUN_ENERGY, optimization=Level(("cell",), inner=inner))
    assert costs_of(spec, 2)["gamma/simulation"] == 2 * 3 + 2


def test_three_levels_multiply_their_budgets() -> None:
    last = Level(("cut",), trials=4, objective=TUNED)
    middle = Level(("energy",), trials=3, objective=TUNED, inner=last)
    spec = problem(CELL, GUN_ENERGY, CUT, optimization=Level(("cell",), inner=middle))
    assert costs_of(spec, 2) == {
        "gamma/simulation": 2 * 3 + 2 * 3 + 2,  # the last level shares the middle's candidate
        "gamma/tuned": 2 * 3 * 4 + 2 * 3 + 2,
        "cost": 2,
    }


def test_each_study_proposes_its_own_values_where_the_level_above_reaches_nothing() -> None:
    inner = Level(("energy",), trials=3, objective=TUNED)
    spec = problem(CUT, GUN_ENERGY, optimization=Level(("cut",), inner=inner))
    assert costs_of(spec, 5)["gamma/simulation"] == 5 * 3 + 5


def test_a_level_between_two_that_reach_a_stage_multiplies_it_too() -> None:
    other = continuous("other", SettingDestination("tuned", "other"))
    last = Level(("energy",), trials=4, objective=TUNED)
    middle = Level(("other",), trials=3, objective=TUNED, inner=last)
    spec = problem(CELL, other, GUN_ENERGY, optimization=Level(("cell",), inner=middle))
    assert costs_of(spec, 2)["gamma/simulation"] == 2 * 3 * 4 + 2 * 3 + 2


def test_a_fixed_top_parameter_leaves_the_studies_that_tune_below_it() -> None:
    other = continuous("other", SettingDestination("cost", "other"))
    inner = Level(("cut",), trials=3, objective=TUNED)
    spec = problem(CELL, other, CUT, optimization=Level(("cell", "other"), inner=inner))
    # the cell is fixed: one simulation for the tuning, shared by all, one on new events
    assert costs_of(spec, 4, ("cell",)) == {
        "gamma/simulation": 1 + 1,
        "gamma/tuned": 4 * 3 + 4,
        "cost": 4,
    }


def test_each_sample_is_counted_by_what_reaches_it() -> None:
    samples = {**GAMMA, "pion": Sample("pion", {"generation": GUN, "simulation": DDSIM})}
    energy = continuous("energy", StageDestination("generation", "energy", ("pion",)))
    spec = replace(
        problem(energy),
        samples=samples,
        evaluators=(ev("tuned", "simulation", "gamma", "pion"), ev("cost")),
    )
    costs = costs_of(spec, 4)
    assert (costs["gamma/simulation"], costs["pion/simulation"]) == (1, 4)
    assert (costs["gamma/tuned"], costs["pion/tuned"]) == (4, 4)
