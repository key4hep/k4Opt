# Visualization: figures of a study

`k4opt.visualization` draws a study from its results alone: a
[`StudyResults`](optimization.md#python-api), as `load_study` reads it from its directory or a
[publication](publication.md) gives it. It decides nothing about the study: which trials are
feasible and which are on the Pareto front are the study's own judgment, the same that
`k4opt study show` prints. It uses [Plotly](https://plotly.com/python/), never an optimizer's
plotting (no `optuna.visualization`), so a study of any optimizer draws alike.

```bash
pip install 'k4opt[plot]'   # Plotly; nothing else of k4opt imports it
```

## The trials of a study: `trials_figure`

```python
from k4opt.optimization import load_study
from k4opt.visualization import axis_choices, trials_figure

study = load_study("runs/cld-ecal-study")
axis_choices(study)
# ('objective:gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual',
#  'objective:gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells',
#  'parameter:ecal_cell_size')
figure = trials_figure(study)  # a plotly.graph_objects.Figure
figure.show()  # in a notebook; or figure.write_html("trials.html")
trials_figure(study, "parameter:ecal_cell_size", axis_choices(study)[0])  # x, then y
```

Each axis is an objective or a parameter, named as the columns of the
[CSV export](optimization.md#export-formats) name them. By default the figure shows the first two
objectives, or, for a study of one objective, the first parameter against it (`default_axes`). Each trial with a
value on both axes is one point:

| Shown as | The trial |
| --- | --- |
| filled blue circle | feasible and on the study's Pareto front |
| small gray circle | feasible, dominated |
| orange cross | valid, but a limit is violated; the hover says which, and by how much |
| open gray circle | without a result (invalid, failed, rejected, interrupted, an error, running): only where both axes are parameters, since it has no objective value |

Every other trial is counted below the plot by outcome ("Not shown, without a value on these
axes: 1 failed, 1 invalid"), never left out silently. The hover shows the trial's number and
outcome, each parameter with its unit, each objective with its uncertainty and unit, and the
limits it violates. Each axis title names the metric, its unit and its direction, with the
sample, evaluator and stage beneath, so that two objectives never share a title.

**The Pareto front** is that of all the study's objectives and limits: an infeasible trial is
never on it, however good its values. When the axes are exactly the two objectives of a
two-objective study, the front is joined by its step line, the boundary of what it dominates
(right for every pair of directions). Otherwise the points of the front are those of the whole
front, which on these two axes may look dominated, and the figure says that it shows a
projection.

What it refuses (`VisualizationError`, a `K4OptError`): an axis the study does not have, and an
objective whose values are recorded in more than one unit, which are never drawn on one scale.
A study without plottable trials gives a figure that says so.

`trials_figure(study, dark=True)` takes the colors stepped for a dark background. The colors are
the reference palette's, checked for color-vision deficiency, and the groups differ by symbol as
well. No Plotly template is forced: a notebook shows its default, and the
[dashboard](dashboard.md) its theme.

**Coloring by a value.** `trials_figure(study, color="parameter:ecal_cell_size")` shades the
feasible trials and those on the front by that value, on the palette's sequential blue with a
color bar; the front is then drawn as diamonds, so it stays told apart, and its step line in gray.
`color_choices(study)` lists what a figure can be colored by: `TRIAL` (`"trial"`, the order the
trials were proposed in), each objective, and each parameter that is not categorical. Infeasible
trials and those without a result keep their own marks.

**Highlighting trials.** `trials_figure(study, highlight=[37, 41], reference=12)` leaves every
other trial faint and rings trial 12: what a
[trade-off](optimization.md#trade-offs-against-a-reference-trial-trade_offs) found, and against
what.

## The history of an objective: `history_figure`

```python
from k4opt.visualization import history_figure

history_figure(study)  # the first objective, against the trial number
history_figure(study, "objective:gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells")
```

Each trial with a value of the objective is drawn at its number, as `trials_figure` draws it. For
a study of one objective, a step line follows the best feasible value so far, in the objective's
direction, and the best trials are marked as such; for several objectives, the trials of the
front are set apart instead, since no single trial is best. Trials without a value are counted
below the plot. It refuses a parameter, and values of the objective recorded in different units.

## How the parameters act on an objective: the analysis figures

```python
from k4opt.visualization import edf_figure, parallel_figure, rank_figure, slice_figure

parallel_figure(study)  # colored by the first objective; or parallel_figure(study, "objective:...")
slice_figure(study, "objective:gamma_10gev/calo.photon/simulation/ecal_barrel.fired_cells")
rank_figure(study)  # two parameters at least
edf_figure(study)
```

They are the figures of Optuna Dashboard's analytics, drawn from the study's own values:

| Figure | What it shows |
| --- | --- |
| `parallel_figure` | each feasible trial as a line through an axis per parameter (a categorical one with its choices, in their order), per objective and per bound of each constraint (its margin: zero or below when satisfied); colored by one objective, the better the further from the background, whatever its direction. Dragging along an axis keeps the trials within a range |
| `slice_figure` | the objective against each parameter, a panel each with the vertical axis shared, the feasible trials colored by their number, the infeasible ones marked apart |
| `rank_figure` | each pair of parameters, the feasible trials colored by how they rank in the objective (the share of the other feasible trials each is better than, from worst to best); infeasible ones in gray. It refuses a study of fewer than two parameters |
| `edf_figure` | for each objective, the share of the trials with a value of it whose value is at most each value |

Nothing is interpolated between trials: there is no contour plot, which would show values no
trial measured. Every trial a figure leaves out is counted below it, by outcome (in the
parallel coordinates, as their title, since Plotly draws no annotation over parallel axes).
Parallel coordinates are drawn with WebGL, which the browser must have.

## How a candidate is evaluated: `pipeline_figure`

```python
from k4opt.visualization import pipeline_figure, pipeline_graph, task_graph

pipeline_figure(study).show()  # laid out and drawn by Graphviz, shown with Plotly
pipeline_figure(study, stack=False)  # every sample in a box of its own
pipeline_figure(study, folded={"gamma_10gev"})  # that sample's box as one node
pipeline_graph(study)  # the same graph as Graphviz DOT text
task_graph(study).node("gamma_10gev/simulation").median_seconds
```

The figure draws the plan of the problem, the tasks every candidate is evaluated by, read left to
right. It works for a problem too (`pipeline_figure(load_problem("problem.yaml"))`), before any
study exists. Laying it out needs Graphviz's `dot` program (`apt install graphviz`,
`dnf install graphviz`, `brew install graphviz`); without it the figure raises
`VisualizationError` and says so, and `pipeline_graph` still writes the DOT text.

Graphviz both lays the graph out and draws it, as SVG; the figure shows that drawing as one image
in its own plane, under Plotly's hover and click. Its text is part of the drawing, so it zooms
with its boxes and never runs over them, and the figure fills the width it is given.

| Drawn as | What it is |
| --- | --- |
| white card, blue dot | a parameter, with its domain; a dashed edge to each task it sets something of, and the task names what (`ECal_cell_size ← ecal_cell_size`) |
| white card, slate dot | the geometry, made once per candidate from the compact file |
| gray panel, violet dots | a sample and its stages: how each is made (`ddsim · 1000 events`), what tells the samples apart (`energy: 10 GeV`) |
| white card, teal dot | an evaluation: of a sample inside its panel, of the candidate or combining samples outside; an item of a role reaches it by an edge labeled with the role |
| white card, orange dot | an objective (in green ▲ maximize, in red ▼ minimize) or a constraint, reached from the evaluation that measures it |
| dotted frame | a diagnostic: recorded, but no objective, constraint or `required: true` needs it |
| red frame, ⚠ line | a diagnostic that did not succeed in some trials, and in how many |

**Folding** keeps a large graph readable: `folded` names samples (or stacks, by key) whose box is
drawn as one node, listing its tasks in order (`generation → simulation → calo.photon`) with each
trial's time summed over its stages and the trials a diagnostic in it did not succeed in, joined
to whatever its tasks were joined to outside it; `collapsed(task_graph(study), keys)` gives that
graph. The dashboard folds every sample at first when a graph has more than 30 tasks.

An edge is drawn only where no other path implies it: every evaluation of a sample reads the
geometry, but through its simulation, so the geometry's edge goes to the simulation alone. Of a
study, each stage shows the median of its recorded times and in how many trials it recorded one;
a stage run by another (the particle gun, fired by ddsim) has no time of its own. Hovering a task
shows everything the problem says of it (options, settings, collections, what it reads, the
parameters that reach it, its directory); hovering an edge, what it carries.

**Stacks.** Samples alike but for the options of their stages (an energy scan: the same stages,
evaluators and objectives for each energy) are drawn as one box, *2 samples*, listing the samples
and what tells them apart (`energy: 10 GeV, 50 GeV`); each task in it stands for that task of
every sample, its time each trial's mean over them ("median 12 min per sample"). Samples read in
different roles (signal and background) are not alike, and neither are samples with other
objectives. `expanded={"gamma_10gev"}` draws the stack whose first sample that is sample by
sample; `stack=False` draws every one.

**Clicking.** The figure's last trace holds what can be clicked: a transparent marker over each
task, each `customdata[0]` the id of its node (`gamma_10gev/simulation`, `parameter:<name>`,
`objective:<label>`), and over each stack's title, the stack's key. `selected="<id>"` marks a
node. The dashboard expands a stack when one of its tasks is clicked, and lists what the problem
says of any other task and how it went in each trial.

The colors are the reference palette's categorical slots (blue, violet, aqua, orange) and its
status red, each kind also told apart by its place, its box and its label. The drawing takes the
colors stepped for a dark background wherever the viewer's system prefers a dark scheme (a style
in the SVG, since an image keeps its own colors whatever the page around it does); in the
dashboard, a theme chosen against the system's in Streamlit's menu leaves the drawing in the
system's. `pipeline_graph(study, dark=True)` writes the DOT text in the dark colors.

## Adding a figure

A new figure is a function of `StudyResults` (and its own options) returning a Plotly `Figure`,
in a module of its own in `k4opt/visualization/`, exported from `k4opt.visualization`, with its
tests in `tests/unit/visualization/`. It reads what the study holds (`study.problem`,
`study.trials`, each trial's `assessment`, `study.pareto`) and judges nothing again; anything a
figure needs that the study does not hold belongs in the study's records first. The dashboard
then shows it with `st.plotly_chart`; nothing about it is specific to Streamlit.

## Not done

Parameter importance, hypervolume and other optimizer diagnostics, figures of several studies,
and figures of what an evaluator recorded besides its objectives and limits.
