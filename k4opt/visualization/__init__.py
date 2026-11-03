"""Visualization component: figures of a study, built from its results alone.

:func:`trials_figure` draws the trials of a study (:class:`~k4opt.optimization.StudyResults`, as
:func:`~k4opt.optimization.load_study` or a publication gives it) in the plane of two of its
objectives or parameters: each trial as feasible, infeasible or on the Pareto front, its values,
units and uncertainties on hover. Nothing here interprets the study again: feasibility and the
front are the study's own. It uses Plotly (``pip install 'k4opt[plot]'``), no optimizer's
plotting, and works the same in a notebook and in the dashboard.

How a candidate is evaluated is its task graph (:func:`task_graph`), read from the problem's plan:
what each parameter sets, each sample's stages, what each evaluation reads, and the evaluations the
objectives and constraints are measured by; of a study, also each stage's recorded times and the
trials each diagnostic did not succeed in; samples alike but for their options are stacked into
one box until expanded. :func:`pipeline_graph` writes it as Graphviz DOT text, and
:func:`pipeline_figure` has Graphviz lay it out and draws it with Plotly. See
``docs/visualization.md``.

Typical use::

    from k4opt.optimization import load_study
    from k4opt.visualization import axis_choices, trials_figure

    study = load_study("runs/cld-ecal-study")
    axis_choices(study)  # ('objective:gamma_10gev/calo.photon/...', 'parameter:ecal_cell_size')
    trials_figure(study).show()
    pipeline_figure(study).show()  # laid out by Graphviz: needs its dot program
"""

from k4opt.visualization.errors import VisualizationError
from k4opt.visualization.graph import GraphEdge, GraphNode, SampleBox, TaskGraph, task_graph
from k4opt.visualization.pipeline import pipeline_figure, pipeline_graph
from k4opt.visualization.trials import axis_choices, default_axes, trials_figure

__all__ = [
    "GraphEdge",
    "GraphNode",
    "SampleBox",
    "TaskGraph",
    "VisualizationError",
    "axis_choices",
    "default_axes",
    "pipeline_figure",
    "pipeline_graph",
    "task_graph",
    "trials_figure",
]
