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
