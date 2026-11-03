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

## How a candidate is evaluated: `pipeline_figure`

```python
from k4opt.visualization import pipeline_figure, pipeline_graph, task_graph

pipeline_figure(study).show()  # laid out by Graphviz, drawn with Plotly
pipeline_figure(study, stack=False)  # every sample in a box of its own
pipeline_graph(study)  # the same graph as Graphviz DOT text
task_graph(study).node("gamma_10gev/simulation").median_seconds
```

The figure draws the plan of the problem, the tasks every candidate is evaluated by, read left to
right. It works for a problem too (`pipeline_figure(load_problem("problem.yaml"))`), before any
study exists. Laying it out needs Graphviz's `dot` program (`apt install graphviz`,
`dnf install graphviz`, `brew install graphviz`); without it the figure raises
`VisualizationError` and says so, and `pipeline_graph` still writes the DOT text.

| Drawn as | What it is |
| --- | --- |
| blue box | a parameter, with its domain; a dashed blue edge to each task it sets something of, and the task names what (`ECal_cell_size ← ecal_cell_size`) |
| gray box | the geometry, made once per candidate from the compact file |
| violet box around violet tasks | a sample and its stages: how each is made (`ddsim · 1000 events`), what tells the samples apart (`energy: 10 GeV`) |
| aqua box | an evaluation: of a sample inside its box, of the candidate or combining samples outside; an item of a role reaches it by an edge labeled with the role |
| orange box | an objective (▲ maximize, ▼ minimize) or a constraint, reached from the evaluation that measures it |
| dotted border | a diagnostic: recorded, but no objective, constraint or `required: true` needs it |
| red border, ⚠ line | a diagnostic that did not succeed in some trials, and in how many |

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
status red, each kind also told apart by its place, its box and its label. The figure sets no
color of text and fills with translucent washes, so it reads on a light and a dark background
alike without being told which. `pipeline_graph(study, dark=True)` takes the colors stepped for a
dark background for the DOT text, whose labels carry their own colors.

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
