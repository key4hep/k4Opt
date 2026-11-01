"""The tasks of a candidate: planned from the problem alone, run after what they read, blocked
when something they read did not succeed (no Key4hep)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from k4opt.evaluation import EvaluationContractError, Status
from k4opt.problem import EvaluatorSpec, MetricRef, Reference, Sample, Stage
from k4opt.problem._files import Made, Tree
from k4opt.problem._tasks import (
    GEOMETRY,
    Kind,
    Plan,
    Task,
    TaskId,
    TaskResult,
    blocked,
    candidate_message,
    candidate_status,
    changed_files,
    diagnostics,
    evaluator_task,
    plan_tasks,
    run_tasks,
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


# --- blocking ---


def done(task: TaskId, status: str = "done", message: str | None = None) -> TaskResult:
    return TaskResult(task, status, message)


def test_a_task_runs_only_when_everything_it_reads_succeeded() -> None:
    calo = Task(evaluator_task("calo"), (GEOMETRY, SIMULATION))
    ready = {GEOMETRY: done(GEOMETRY), SIMULATION: done(SIMULATION)}
    assert blocked(calo, ready) is None
    held = blocked(calo, ready | {SIMULATION: done(SIMULATION, "failed", "ddsim crashed")})
    assert held == TaskResult(
        calo.id, Status.BLOCKED, "not run: gamma/simulation failed", blocked_by=(SIMULATION,)
    )
    # an evaluator is read as succeeded only when it is valid (combined evaluators, later)
    rival = evaluator_task("rival")
    reader = Task(evaluator_task("combined"), (rival,))
    assert blocked(reader, {rival: done(rival, "valid")}) is None
    assert blocked(reader, {rival: done(rival, "invalid", "too few events")}) is not None


def test_a_blocked_task_names_where_its_candidate_went_wrong_not_the_blocked_in_between() -> None:
    results = {GEOMETRY: done(GEOMETRY), GENERATION: done(GENERATION, "failed", "unreadable")}
    simulation = Task(SIMULATION, (GEOMETRY, GENERATION))
    results[SIMULATION] = blocked(simulation, results)  # type: ignore[assignment]
    calo = blocked(Task(evaluator_task("calo"), (GEOMETRY, SIMULATION)), results)
    assert calo is not None
    assert (calo.blocked_by, calo.message) == ((GENERATION,), "not run: gamma/generation failed")


def test_the_executor_runs_each_task_once_in_plan_order_with_only_what_it_reads() -> None:
    plan = plan_tasks(GAMMA, [ev("a", "simulation"), ev("b")], references=())
    seen: list[tuple[str, list[str]]] = []

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        seen.append((task.id.label, [need.label for need in reads]))
        return done(task.id, "valid" if task.id.kind is Kind.EVALUATOR else "done")

    results = run_tasks(plan, run)
    assert seen == [
        ("geometry", []),
        ("gamma/generation", []),
        ("gamma/simulation", ["geometry", "gamma/generation"]),
        ("gamma/a", ["geometry", "gamma/generation", "gamma/simulation"]),
        ("b", ["geometry"]),
    ]
    assert list(results) == [task.id for task in plan.tasks]


def test_the_executor_does_not_run_a_blocked_task() -> None:
    plan = plan_tasks(GAMMA, [ev("a", "simulation"), ev("b")], references=())
    ran: list[str] = []

    def run(task: Task, reads: Mapping[TaskId, TaskResult]) -> TaskResult:
        ran.append(task.id.label)
        if task.id == SIMULATION:
            return done(task.id, "failed", "ddsim crashed")
        return done(task.id, "valid" if task.id.kind is Kind.EVALUATOR else "done")

    results = run_tasks(plan, run)
    assert ran == ["geometry", "gamma/generation", "gamma/simulation", "b"]
    assert results[evaluator_task("a", "gamma")].blocked_by == (SIMULATION,)
    assert results[evaluator_task("b")].status == "valid"


# --- the candidate's status and message, from its tasks ---


def required(*tasks: TaskId, diagnostics: tuple[TaskId, ...] = ()) -> Plan:
    """A plan of ``tasks``, each required, then ``diagnostics``, none of them reading another."""
    return Plan(
        tuple(Task(task, required=True) for task in tasks)
        + tuple(Task(task) for task in diagnostics)
    )


ECHO, CELLS = evaluator_task("echo"), evaluator_task("cells")


@pytest.mark.parametrize(
    ("statuses", "status"),
    [
        (["done", "valid"], Status.VALID),
        (["done", "valid", "invalid"], Status.INVALID),
        (["failed", "blocked", "invalid"], Status.FAILED),
        # blocked by an invalid task: nothing failed, and it is no result
        (["done", "invalid", "blocked"], Status.INVALID),
        (["done", "blocked"], Status.INVALID),  # never alone, but no result all the same
    ],
)
def test_a_candidate_is_failed_before_invalid_before_valid(
    statuses: list[str], status: Status
) -> None:
    tasks = [GEOMETRY, SIMULATION, ECHO][: len(statuses)]
    results = {task: done(task, s) for task, s in zip(tasks, statuses, strict=True)}
    assert candidate_status(required(*tasks), results) is status


@pytest.mark.parametrize("diagnostic", ["failed", "invalid", "blocked"])
def test_a_diagnostic_never_changes_the_candidate_status_or_message(diagnostic: str) -> None:
    plan = required(GEOMETRY, ECHO, diagnostics=(CELLS,))
    results = {GEOMETRY: done(GEOMETRY), ECHO: done(ECHO, "valid"), CELLS: done(CELLS, diagnostic)}
    assert candidate_status(plan, results) is Status.VALID
    assert candidate_message(plan, results) is None
    assert diagnostics(plan, results) == [results[CELLS]]
    assert diagnostics(plan, results | {CELLS: done(CELLS, "valid")}) == []
    assert candidate_status(required(), {}) is Status.VALID


def test_the_message_names_the_causes_then_what_they_blocked() -> None:
    calo, energy = evaluator_task("calo"), evaluator_task("energy")
    results = {
        GEOMETRY: done(GEOMETRY),
        SIMULATION: done(SIMULATION, "failed", "ddsim crashed"),
        calo: TaskResult(calo, Status.BLOCKED, "x", blocked_by=(SIMULATION,)),
        ECHO: done(ECHO, "invalid", "too far out"),
        energy: TaskResult(energy, Status.BLOCKED, "x", blocked_by=(SIMULATION,)),
        CELLS: done(CELLS, "valid", "fine"),
    }
    plan = required(*results)
    assert candidate_message(plan, results) == (
        "gamma/simulation failed: ddsim crashed; echo invalid: too far out; blocked: calo, energy"
    )
    # only the required ones: a diagnostic that failed is not the candidate's
    plan = required(GEOMETRY, ECHO, CELLS, diagnostics=(SIMULATION, calo, energy))
    assert candidate_message(plan, results) == "echo invalid: too far out"
    assert candidate_message(required(GEOMETRY, CELLS), results) is None


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


# --- the files a task made, and whether they are still what its record says ---


def test_a_made_file_is_named_by_its_task_and_hashed_only_if_asked(tmp_path: Path) -> None:
    path = tmp_path / "events.root"
    path.write_bytes(b"events")
    unhashed = Made.file(SIMULATION, "events", path, hashed=False)
    assert unhashed.artifact.ref == "gamma/simulation#events"
    assert (unhashed.artifact.path, unhashed.artifact.bytes) == (path, 6)
    assert unhashed.artifact.sha256 is None
    hashed = Made.file(evaluator_task("plot", "gamma"), "histogram", path, hashed=True)
    assert hashed.artifact.ref == "gamma/plot#histogram"
    assert hashed.artifact.sha256 is not None and len(hashed.artifact.sha256) == 64
    with pytest.raises(OSError, match="is no file"):
        Made.file(SIMULATION, "events", tmp_path, hashed=False)


def test_a_file_replaced_while_it_is_hashed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.problem import _files

    path = tmp_path / "histogram.json"
    path.write_bytes(b"old")
    identity = _files.file_identity

    def replaced(file: Path) -> dict[str, object]:
        file.write_bytes(b"new, and longer")  # between the stat and the hash
        return identity(file)

    monkeypatch.setattr(_files, "file_identity", replaced)
    # the size of one version with the hash of another, which changed() would then trust
    with pytest.raises(OSError, match="changed while it was hashed"):
        Made.file(evaluator_task("plot", "gamma"), "histogram", path, hashed=True)


@pytest.mark.parametrize("hashed", [False, True])
def test_a_made_file_is_changed_once_its_stat_or_its_bytes_are(
    tmp_path: Path, hashed: bool
) -> None:
    path = tmp_path / "events.root"
    path.write_bytes(b"events")
    made = Made.file(SIMULATION, "events", path, hashed=hashed)
    assert not made.changed()
    os.utime(path, ns=(1, 1))  # another stat, the same bytes: only a hash tells them apart
    assert made.changed() is not hashed
    # other bytes, of another size: a rewrite of the same size within one tick of the kernel's
    # clock keeps every stat (the limit _sha256 documents)
    path.write_bytes(b"other events")
    assert made.changed()
    path.unlink()
    assert made.changed()


def test_the_changed_files_are_those_of_the_results_given(tmp_path: Path) -> None:
    kept, rewritten = tmp_path / "kept.root", tmp_path / "rewritten.root"
    for path in (kept, rewritten):
        path.write_bytes(b"events")
    results = [
        TaskResult(
            SIMULATION, "done", made={"events": Made.file(SIMULATION, "events", kept, hashed=False)}
        ),
        TaskResult(GEOMETRY, "done"),
        TaskResult(
            stage_task("simulation", "pion"),
            "done",
            made={
                "events": Made.file(
                    stage_task("simulation", "pion"), "events", rewritten, hashed=False
                )
            },
        ),
    ]
    rewritten.write_bytes(b"more events")
    assert changed_files(results) == ["pion/simulation#events"]


def test_a_tree_is_changed_by_any_file_in_it_that_changes_goes_or_is_added(
    tmp_path: Path,
) -> None:
    (tmp_path / "compact").mkdir()
    (tmp_path / "compact" / "a.xml").write_text("<a/>")
    (tmp_path / "b.xml").write_text("<b/>")
    tree = Tree.of(tmp_path)
    assert list(tree.stats) == ["b.xml", "compact/a.xml"] and tree.changed() == []
    (tmp_path / "compact" / "a.xml").write_text("<a>edited</a>")
    (tmp_path / "b.xml").unlink()
    (tmp_path / "c.xml").write_text("<c/>")
    assert tree.changed() == ["b.xml", "compact/a.xml", "c.xml"]
    held = TaskResult(GEOMETRY, "done", tree=tree)
    assert changed_files([held]) == ["geometry/b.xml", "geometry/compact/a.xml", "geometry/c.xml"]
