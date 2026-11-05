"""The task graph of a candidate and its three drawings: the graph read from the problem's plan,
each edge one that no other path implies, what each parameter sets, which tasks are diagnostics,
of a study its stage times and the diagnostics that did not succeed; as DOT text, laid out by
Graphviz in a Plotly figure, samples alike stacked into one box until expanded."""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("plotly")

from k4opt.frozen import FrozenMap
from k4opt.optimization import StudyResults, TrialResult
from k4opt.problem import ProblemSpec, names, parse_problem
from k4opt.problem.plan import level_costs, stage_task
from k4opt.visualization import (
    COLLAPSED,
    TaskGraph,
    VisualizationError,
    collapsed,
    pipeline_figure,
    pipeline_graph,
    task_graph,
)
from k4opt.visualization.graph import _loop

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "problem"
GRAPHVIZ = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz is not installed")
SAMPLES = ("gamma_50gev", "pi0_50gev")
RESIDUAL = "calo.photon/simulation/ecal_barrel.angular_residual"


# a Graphviz built without fonts, as the Key4hep stack's, estimates text widths and says so
FONTLESS = ("no hard-coded metrics for", "no value for width of non-ASCII character")


def complaints(stderr: str) -> list[str]:
    """What Graphviz wrote to stderr, but that it lays out without fonts."""
    return [line for line in stderr.splitlines() if not any(f in line for f in FONTLESS)]


def example(name: str = "cld-ecal-pi0-rejection") -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load((EXAMPLES / f"{name}.yaml").read_text())
    return document


def edges(graph: TaskGraph, kind: str) -> dict[tuple[str, str], str | None]:
    return {(e.source, e.target): e.label for e in graph.edges if e.kind == kind}


def study(spec: ProblemSpec, trials: list[TrialResult]) -> StudyResults:
    return StudyResults("pi0", "id", "now", spec, FrozenMap({}), tuple(trials), ())


def trial(number: int, **fields: Any) -> TrialResult:
    return TrialResult(number, "valid", {"ecal_cell_size": 5.0}, record="r", **fields)


def with_constraint() -> ProblemSpec:
    """The example with its channel count a constraint, not an objective."""
    document = example()
    document["objectives"] = document["objectives"][:1]
    document["constraints"] = [
        {"evaluator": "readout", "stage": "geometry", "metric": "ecal_barrel.channels"}
        | {"max": 5.0e7, "unit": "1"}
    ]
    return parse_problem(document)


def without_constraint() -> ProblemSpec:
    document = example()
    document["objectives"] = document["objectives"][:1]  # nothing needs the readout: a diagnostic
    return parse_problem(document)


def test_each_task_parameter_objective_and_constraint_is_one_node_in_order() -> None:
    graph = task_graph(with_constraint())
    assert [node.id for node in graph.nodes] == [
        "parameter:ecal_cell_size",
        "geometry",
        *(f"{s}/{t}" for s in SAMPLES for t in ("generation", "simulation", "calo.shower_shape")),
        "pi0_rejection",
        "readout",
        "objective:pi0_rejection/analysis/background_rejection",
        "constraint:readout/geometry/ecal_barrel.channels",
    ]
    objective, constraint = graph.nodes[-2:]
    assert (objective.title, objective.details) == ("π⁰ rejection", ("▲ maximize",))  # its title
    assert (constraint.title, constraint.details) == ("ecal_barrel.channels", ("≤ 5e+07",))
    assert set(graph.samples) == set(SAMPLES)  # read in other roles: not alike, not stacked
    assert graph.node("gamma_50gev/simulation").group == "stage:simulation"
    assert graph.node("pi0_rejection").group is None  # it measures the candidate
    with pytest.raises(VisualizationError, match="no node 'nothing'"):
        graph.node("nothing")


def test_only_the_reads_no_other_path_implies_are_edges() -> None:
    reads = edges(task_graph(parse_problem(example())), "reads")
    assert ("geometry", "gamma_50gev/simulation") in reads
    assert ("gamma_50gev/generation", "gamma_50gev/simulation") in reads
    assert ("gamma_50gev/simulation", "gamma_50gev/calo.shower_shape") in reads
    assert ("geometry", "readout") in reads  # it reads the geometry alone
    # implied through the simulation, and through the evaluations of the samples
    assert ("geometry", "gamma_50gev/calo.shower_shape") not in reads
    assert ("gamma_50gev/generation", "gamma_50gev/calo.shower_shape") not in reads
    assert ("geometry", "pi0_rejection") not in reads
    assert reads[("gamma_50gev/calo.shower_shape", "pi0_rejection")] == "signal"
    assert reads[("pi0_50gev/calo.shower_shape", "pi0_rejection")] == "background"
    measures = edges(task_graph(with_constraint()), "measures")
    assert set(measures) == {
        ("pi0_rejection", "objective:pi0_rejection/analysis/background_rejection"),
        ("readout", "constraint:readout/geometry/ecal_barrel.channels"),
    }


def test_a_parameter_reaches_what_it_sets_and_the_task_names_it() -> None:
    graph = task_graph(parse_problem(example("cld-ecal-bindings")), stack=False)
    samples = ("gamma_10gev", "gamma_50gev")
    assert edges(graph, "sets") == {
        ("parameter:ecal_cell_size", "geometry"): "ECal_cell_size",
        **{("parameter:range_cut", f"{s}/simulation"): "physics.rangecut" for s in samples},
        **{("parameter:cell_threshold", f"{s}/calo.photon"): "threshold" for s in samples},
    }
    # one parameter varies each: its edge says which, the task only what
    assert "varies ECal_cell_size" in graph.node("geometry").details
    assert "varies physics.rangecut" in graph.node("gamma_10gev/simulation").details
    assert "varies threshold" in graph.node("gamma_50gev/calo.photon").details
    assert graph.node("parameter:range_cut").details == ("[0.1, 1] mm",)


def test_a_task_several_parameters_vary_says_which_varies_what() -> None:
    document = example("cld-ecal-bindings")
    document["parameters"]["ecal_cell_size"]["title"] = "ECAL cell size"
    document["parameters"]["ecal_depth"] = {
        "type": "continuous",
        "bounds": [100.0, 200.0],
        "unit": "mm",
        "targets": {"geometry": "ECalBarrel_outer_radius"},
    }
    details = task_graph(parse_problem(document)).node("geometry").details
    assert "ECal_cell_size ← ECAL cell size" in details
    assert "ECalBarrel_outer_radius ← ecal_depth" in details
    document["parameters"]["ecal_depth"]["title"] = "ECAL cell size"  # two parameters, one title
    details = task_graph(parse_problem(document)).node("geometry").details
    assert "ECal_cell_size ← ECAL cell size (ecal_cell_size)" in details
    assert "ECalBarrel_outer_radius ← ECAL cell size (ecal_depth)" in details


def test_parameters_samples_and_measured_metrics_are_drawn_by_their_titles() -> None:
    document = example("cld-ecal-bindings")
    document["parameters"]["ecal_cell_size"]["title"] = "ECAL cell size"
    document["samples"]["gamma_10gev"]["title"] = "10 GeV γ"
    document["samples"]["gamma_50gev"]["title"] = "50 GeV γ"
    for objective in document["objectives"]:
        objective["title"] = "Angular residual"
    graph = task_graph(parse_problem(document))
    parameter = graph.node("parameter:ecal_cell_size")
    assert (
        parameter.title == "ECAL cell size" and ("parameter", "ecal_cell_size") in parameter.facts
    )
    (box,) = graph.samples.values()
    assert box.lines[0] == "10 GeV γ, 50 GeV γ"
    objective = graph.node(f"stack:gamma_10gev|objective:{RESIDUAL}")
    assert objective.title == "Angular residual"
    alone = task_graph(parse_problem(document), stack=False)
    assert [box.title for box in alone.samples.values()] == ["10 GeV γ", "50 GeV γ"]


def test_measurements_of_one_box_titled_alike_are_told_apart_by_what_they_measure() -> None:
    document = example("cld-ecal-photon-energies")
    for objective in document["objectives"]:
        objective["title"] = "Resolution"  # both of gamma_50gev, of two metrics
    graph = task_graph(parse_problem(document), stack=False)
    fired = "calo.photon/simulation/ecal_barrel.fired_cells"
    assert graph.node(f"objective:gamma_10gev/{RESIDUAL}").title == "Resolution"  # alone in it
    assert graph.node(f"objective:gamma_50gev/{RESIDUAL}").title == RESIDUAL
    assert graph.node(f"objective:gamma_50gev/{fired}").title == fired


def test_samples_whose_objectives_are_titled_otherwise_are_not_stacked() -> None:
    document = example("cld-ecal-bindings")
    document["objectives"][0]["title"] = "Residual at 10 GeV"
    document["objectives"][1]["title"] = "Residual at 50 GeV"
    graph = task_graph(parse_problem(document))
    assert [box.is_stack for box in graph.samples.values()] == [False, False]
    assert graph.node(f"objective:gamma_10gev/{RESIDUAL}").title == "Residual at 10 GeV"


def test_a_stage_shows_how_it_is_made_and_what_tells_the_samples_apart() -> None:
    graph = task_graph(parse_problem(example()))
    generation = graph.node("pi0_50gev/generation")
    assert generation.details == ("gun, run by ddsim", "particle: pi0")  # not the same energy
    assert ("option energy", "50 GeV") in generation.facts  # all of it in the facts
    assert graph.node("pi0_50gev/simulation").details[0] == "ddsim · 500 events"
    assert ("seed", "2") in graph.node("pi0_50gev/simulation").facts


def test_a_diagnostic_is_not_required() -> None:
    graph = task_graph(without_constraint())
    readout = graph.node("readout")
    assert not readout.required and "diagnostic" in readout.details
    assert graph.node("pi0_rejection").required


def test_a_study_holds_the_times_of_each_stage_and_the_trials_a_diagnostic_failed_in() -> None:
    failed = ({"task": "readout", "status": "failed", "message": "no channels"},)
    trials = [
        trial(0, stage_seconds={"gamma_50gev/simulation": 30.0}, diagnostics=failed),
        trial(1, stage_seconds={"gamma_50gev/simulation": 90.0}),
        trial(2, stage_seconds={"gamma_50gev/simulation": 600.0}, diagnostics=failed),
        TrialResult(3, "interrupted", {}),  # no record: says nothing of its diagnostics
    ]
    graph = task_graph(study(without_constraint(), trials))
    simulation = graph.node("gamma_50gev/simulation")
    assert dict(simulation.seconds or {}) == {0: 30.0, 1: 90.0, 2: 600.0}
    assert simulation.median_seconds == 90.0
    assert graph.node("pi0_50gev/simulation").median_seconds is None
    assert graph.node("readout").failed == (0, 2) and graph.recorded == 3
    dot = pipeline_graph(study(without_constraint(), trials))
    assert "median 1.5 min · 3 trials" in dot
    assert '<font point-size="10" color="#e11d48">⚠ did not succeed in 2 of 3 trials</font>' in dot
    assert "did not succeed" not in pipeline_graph(without_constraint())


def test_dot_shows_what_a_problem_names_as_it_is_but_a_backslash_in_a_tooltip() -> None:
    document = example()
    document["samples"]["gamma_50gev"]["properties"] = {"note": '<b>"x" & \\N</b>'}
    document["evaluators"][2]["settings"]["note"] = 'say "\\G" &amp;\nthen'
    dot = pipeline_graph(parse_problem(document))
    label = "note: &lt;b&gt;&quot;x&quot; &amp; \\N&lt;/b&gt;"  # an HTML-like label expands nothing
    tooltip = "setting note: say &quot;\N{REVERSE SOLIDUS OPERATOR}G&quot; &amp;amp;"
    assert label in dot and f"{tooltip}\\nthen" in dot
    if shutil.which("dot") is None:
        pytest.skip("Graphviz is not installed")
    shown = subprocess.run(["dot", "-Tsvg"], input=dot, capture_output=True, text=True, check=True)
    assert not complaints(shown.stderr)
    assert label in shown.stdout and f"{tooltip}&#10;then" in shown.stdout


def test_dot_takes_the_colors_stepped_for_a_dark_background() -> None:
    spec = parse_problem(example())
    light, dark = pipeline_graph(spec), pipeline_graph(spec, dark=True)
    assert 'fillcolor="#ffffff"' in light and 'fillcolor="#161b22"' in dark
    assert 'fillcolor="#ebeff5"' in light and 'fillcolor="#11161d"' in dark  # a sample's panel
    assert 'bgcolor="transparent"' in light and 'bgcolor="transparent"' in dark  # the page's


@GRAPHVIZ
@pytest.mark.parametrize("name", sorted(path.stem for path in EXAMPLES.glob("*.yaml")))
def test_graphviz_lays_out_every_example_and_the_figure_draws_each_node(name: str) -> None:
    spec = parse_problem(example(name))
    shown = subprocess.run(
        ["dot", "-Tsvg"], input=pipeline_graph(spec), capture_output=True, text=True, check=True
    )
    assert not complaints(shown.stderr) and "<svg" in shown.stdout
    graph = task_graph(spec)
    figure = pipeline_figure(spec)
    clickable = {row[0] for row in figure.data[-1].customdata}
    stacks = [key for key, box in graph.samples.items() if box.is_stack]
    assert clickable == {n.id for n in graph.nodes} | set(stacks)
    (image,) = figure.layout.images  # Graphviz's drawing: no box or label of the figure's own
    assert image.source.startswith("data:image/svg+xml;base64,")
    assert not figure.layout.shapes and not figure.layout.annotations


@GRAPHVIZ
def test_the_drawing_covers_the_layout_and_takes_the_dark_colors_where_preferred() -> None:
    figure = pipeline_figure(parse_problem(example()))
    (image,) = figure.layout.images
    svg = base64.b64decode(image.source.split(",", 1)[1]).decode()
    assert '[fill="#ffffff"] { fill: #161b22 }' in svg and "prefers-color-scheme: dark" in svg
    assert '[stroke="#dfe4ec"] { stroke: #2a313c }' in svg
    (x0, x1), (y0, y1) = figure.layout.xaxis.range, figure.layout.yaxis.range
    assert (image.x, image.x + image.sizex, image.y - image.sizey, image.y) == pytest.approx(
        (x0, x1, y0, y1)
    )
    clickable = figure.data[-1]  # a task is clicked anywhere in its box: many markers cover it
    assert sum(row[0] == "readout" for row in clickable.customdata) > 4


@GRAPHVIZ
def test_the_figure_marks_the_selected_node() -> None:
    spec = parse_problem(example())
    plain, chosen = pipeline_figure(spec), pipeline_figure(spec, selected="readout")
    (ring,) = chosen.layout.shapes
    assert not plain.layout.shapes and ring.line.width == 3 and ring.layer == "above"


def test_the_figure_says_when_graphviz_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(VisualizationError, match="needs Graphviz's dot program"):
        pipeline_figure(parse_problem(example()))


def test_samples_alike_but_for_their_options_are_stacked_until_expanded() -> None:
    spec = parse_problem(example("cld-ecal-bindings"))
    graph = task_graph(spec)
    (box,) = graph.samples.values()
    assert box.is_stack and box.members == ("gamma_10gev", "gamma_50gev")
    assert box.title == "2 samples" and "energy: 10 GeV, 50 GeV" in box.lines
    simulation = graph.node("stack:gamma_10gev|stage:simulation")
    assert simulation.samples == box.members and simulation.sample == "gamma_10gev"
    assert "varies physics.rangecut" in simulation.details  # shared by both
    assert not any(
        "energy" in detail for detail in graph.node("stack:gamma_10gev|stage:generation").details
    )
    objective = graph.node(
        "stack:gamma_10gev|objective:calo.photon/simulation/ecal_barrel.angular_residual"
    )
    assert "one for each of 2 samples" in objective.details
    reads = edges(graph, "reads")
    assert ("geometry", "stack:gamma_10gev|stage:simulation") in reads
    assert ("parameter:range_cut", "stack:gamma_10gev|stage:simulation") in edges(graph, "sets")
    for unstacked in (task_graph(spec, expanded={"gamma_10gev"}), task_graph(spec, stack=False)):
        assert set(unstacked.samples) == set(box.members)
        assert not any(node.samples for node in unstacked.nodes)


def test_a_stack_holds_each_trials_mean_time_over_its_samples() -> None:
    spec = parse_problem(example("cld-ecal-bindings"))
    times = {"gamma_10gev/simulation": 60.0, "gamma_50gev/simulation": 120.0}
    graph = task_graph(study(spec, [trial(0, stage_seconds=times)]))
    simulation = graph.node("stack:gamma_10gev|stage:simulation")
    assert dict(simulation.seconds or {}) == {0: 90.0}
    assert "median 1.5 min per sample · 1 trials" in pipeline_graph(
        study(spec, [trial(0, stage_seconds=times)])
    )


def test_a_stage_run_by_another_has_no_time_of_its_own() -> None:
    times = {"gamma_50gev/generation": 0.0003, "gamma_50gev/simulation": 60.0}
    graph = task_graph(study(parse_problem(example()), [trial(0, stage_seconds=times)]))
    assert graph.node("gamma_50gev/generation").seconds is None  # the gun, fired by ddsim
    assert graph.node("gamma_50gev/simulation").median_seconds == 60.0


def test_each_kind_has_its_own_color() -> None:
    dot = pipeline_graph(parse_problem(example()))
    line = lambda id: next(  # noqa: E731
        text
        for text in dot.splitlines()
        if f'tooltip="task: {id}' in text or f"<b>{id}</b>" in text
    )
    # white cards alike, each kind told by the dot before its title
    marker = '<font color="{}" point-size="10">●</font>'.format
    assert 'color="#dfe4ec", fillcolor="#ffffff"' in line("π⁰ rejection")
    assert marker("#3b82f6") in line("ECAL cell size")  # parameter: blue
    assert marker("#8b5cf6") in line("gamma_50gev/simulation")  # a sample's stage: violet
    assert marker("#14b8a6") in line("pi0_rejection")  # evaluation: teal
    assert marker("#f97316") in line("π⁰ rejection")  # objective: orange
    # a direction in green, up, or red, down: its text and its arrow alone
    assert '<font point-size="10" color="#16a34a">▲ maximize</font>' in line("π⁰ rejection")
    assert '<font point-size="10" color="#dc2626">▼ minimize</font>' in line("Readout channels")


def test_a_folded_sample_is_one_node_joined_as_its_tasks_were() -> None:
    spec = parse_problem(example())
    whole = task_graph(spec)
    graph = collapsed(whole, {"gamma_50gev"})
    folded = graph.node(f"{COLLAPSED}gamma_50gev")
    assert set(graph.samples) == {"pi0_50gev"} and folded.sample is None
    assert folded.details[0] == "generation → simulation → calo.shower_shape"
    assert not any(n.sample == "gamma_50gev" for n in graph.nodes)
    reads = edges(graph, "reads")
    assert ("geometry", folded.id) in reads and (folded.id, "pi0_rejection") in reads
    assert reads[(folded.id, "pi0_rejection")] == "signal"  # the role it was read in
    assert collapsed(whole, ()) is whole
    every = collapsed(whole, set(whole.samples))
    assert not every.samples and len(every.nodes) == len(whole.nodes) - 4


@GRAPHVIZ
def test_the_drawing_takes_the_colors_it_is_told() -> None:
    spec = parse_problem(example())

    def svg(figure: Any) -> str:
        (image,) = figure.layout.images
        return base64.b64decode(image.source.split(",", 1)[1]).decode()

    light, dark = svg(pipeline_figure(spec, dark=False)), svg(pipeline_figure(spec, dark=True))
    # drawn in the colors told, whatever the viewer prefers: no style that changes them
    assert "<style>" not in light and "<style>" not in dark
    assert 'stroke="#dfe4ec"' in light and 'stroke="#2a313c"' in dark  # a card's frame


NESTED = EXAMPLES.parent / "nested" / "cld-pi0-nested.yaml"


def test_a_nested_problem_is_drawn_as_loops_of_what_runs_for_each_trial_of_a_level() -> None:
    spec = parse_problem(yaml.safe_load(NESTED.read_text()))
    graph = task_graph(spec, stack=False)
    assert [(loop.level, loop.title) for loop in graph.loops] == [
        (0, "for each ECAL cell size"),
        (1, "for each Pandora ECAL hit threshold trial"),
    ]
    assert graph.loops[1].lines[1] == "best one evaluated again on new events"
    loops = {node.id: node.loop for node in graph.nodes}
    # what the cell size reaches runs once for it; what the threshold reaches, for each trial of it
    assert loops["parameter:ecal_cell_size"] == loops["geometry"] == 0
    assert loops["pi0_50gev/simulation"] == loops["pi0_50gev/generation"] == 0
    assert loops["parameter:ecal_mip_threshold"] == loops["pi0_50gev/reconstruction"] == 1
    assert loops["pi0_50gev/pfo.pi0"] == 1 and loops["readout"] == 0
    labels = {(e.source, e.kind): e.label for e in graph.edges}
    assert labels["pi0_50gev/pfo.pi0", "measures"] == "best, on new events"
    assert labels["readout", "measures"] is None  # measured for the cell size itself
    # folded, a sample is drawn in the outermost loop of its tasks
    (folded,) = [n for n in collapsed(graph, ["pi0_50gev"]).nodes if n.id.startswith("sample:")]
    assert folded.loop == 0


def test_the_loops_nest_as_the_levels_do_and_a_sample_box_holds_its_inner_loop() -> None:
    spec = parse_problem(yaml.safe_load(NESTED.read_text()))
    dot = pipeline_graph(spec, stack=False)
    outer, sample, inner = (
        dot.index("subgraph cluster_loop0_root"),
        dot.index("subgraph cluster_0 "),
        dot.index("subgraph cluster_loop1_cluster_0"),
    )
    assert outer < sample < inner
    assert "best, on new events" in dot and "rank=source" not in dot


def test_a_problem_without_levels_draws_no_loop() -> None:
    graph = task_graph(parse_problem(example()))
    assert graph.loops == () and all(node.loop is None for node in graph.nodes)
    assert "cluster_loop" not in pipeline_graph(parse_problem(example()))


@GRAPHVIZ
def test_graphviz_lays_out_the_loops_and_the_figure_draws_each_node() -> None:
    spec = parse_problem(yaml.safe_load(NESTED.read_text()))
    shown = subprocess.run(
        ["dot", "-Tsvg"], input=pipeline_graph(spec), capture_output=True, text=True, check=True
    )
    assert not complaints(shown.stderr) and "<svg" in shown.stdout
    figure = pipeline_figure(spec)
    drawn = {str(item[0]) for trace in figure.data for item in (trace.customdata or ())}
    assert {node.id for node in task_graph(spec).nodes} <= drawn


def nested(change: Any) -> ProblemSpec:
    document = yaml.safe_load(NESTED.read_text())
    change(document)
    return parse_problem(document)


def test_an_evaluation_is_in_the_loop_of_the_deepest_level_that_runs_it() -> None:
    def diagnosed(document: dict[str, Any]) -> None:
        # a diagnostic of the reconstruction the threshold reaches, which only the top runs
        document["evaluators"].append(
            document["evaluators"][0] | {"as": "pfo.diagnostic", "settings": {}}
        )

    loops = {n.id: n.loop for n in task_graph(nested(diagnosed), stack=False).nodes}
    assert loops["pi0_50gev/reconstruction"] == 1
    assert loops["pi0_50gev/pfo.diagnostic"] == 0  # after the tuning, on its best only

    def required(document: dict[str, Any]) -> None:
        document["evaluators"][1]["required"] = True  # the readout, in every inner trial too

    loops = {n.id: n.loop for n in task_graph(nested(required), stack=False).nodes}
    assert loops["readout"] == 1 and loops["geometry"] == 0  # no inner parameter reaches either


def test_a_stage_only_a_level_below_reaches_is_in_the_loop_of_the_level_that_runs_it() -> None:
    def tuned_without_it(document: dict[str, Any]) -> None:
        # the cell size reaches the readout alone, the threshold the readout and the
        # reconstruction; the inner level, tuned by the readout, does not reconstruct
        del document["evaluators"][1]["settings"]["layers"]
        document["parameters"]["ecal_cell_size"]["targets"] = {
            "evaluators": {"readout": "cell_size"}
        }
        document["parameters"]["ecal_mip_threshold"]["targets"]["evaluators"] = {
            "readout": "layers"
        }
        document["optimization"]["inner"]["objective"] = {
            "evaluator": "readout",
            "stage": "geometry",
            "metric": "ecal_barrel.channels",
            "direction": "minimize",
        }

    spec = nested(tuned_without_it)
    loops = {n.id: n.loop for n in task_graph(spec, stack=False).nodes}
    # reconstructed once for each cell size, with the threshold its study selected
    assert loops["pi0_50gev/reconstruction"] == 0
    assert level_costs(spec, 8)[stage_task("reconstruction", "pi0_50gev")] == 8


def test_the_loop_of_a_study_names_only_what_it_searches() -> None:
    def tuned_together(document: dict[str, Any]) -> None:
        document["parameters"]["range_cut"] = {
            "title": "Range cut",
            "type": "continuous",
            "bounds": [0.1, 1.0],
            "unit": "mm",
            "targets": {"simulation": "physics.rangecut"},
        }
        document["optimization"]["parameters"].append("range_cut")

    data = yaml.safe_load(NESTED.read_text())
    tuned_together(data)
    spec = parse_problem(data)
    assert task_graph(spec).loops[0].title == "for each ECAL cell size, Range cut"
    # in a study given one cell size (--fix), the loop is over the range cut alone
    fixed = _loop(spec, 0, 0, names(spec), None, {"ecal_cell_size": 5.1})
    assert fixed.title == "for each Range cut"
