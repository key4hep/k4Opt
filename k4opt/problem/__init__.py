"""Problem component: the declarative definition of a detector optimization problem.

Reads and strictly validates a ``k4opt.problem/v1alpha1`` YAML document into immutable,
typed models: the parameters (each a domain, and the destinations its value is written into: an
attribute of the geometry, an option of a stage, a setting of an evaluator), the named
``samples`` a candidate is measured on (each with its
pipeline stages, each stage run by an executor or given as a file), the ``evaluators`` (named
bindings of evaluator implementations, each measuring its samples or the candidate, or combining
the evaluations of several samples by role, :class:`Reference`) and the
objectives and constraints, each on a metric of one evaluation (:class:`MetricRef`).
``load_problem`` also resolves the geometry and every geometry destination in the active k4geo
tree (``$K4GEO``); ``parse_problem`` is the pure variant that needs no geometry. Evaluators and
metrics are not checked. See ``docs/problem.md``.

Typical use::

    from k4opt.problem import load_problem

    problem = load_problem("problem.yaml")
    print(problem.parameters[0].bounds, problem.parameters[0].targets, problem.evaluators[0].name)
    data = problem.to_dict()  # plain JSON-compatible data

Candidates of a problem are evaluated by :class:`k4opt.execution.EvaluationSession`, and what
their records say is read back by :mod:`k4opt.record`.
"""

from k4opt.problem.config import load_problem, parse_problem
from k4opt.problem.errors import CandidateError, ProblemConfigError
from k4opt.problem.models import (
    SCHEMA_VERSION,
    Constraint,
    Criteria,
    Destination,
    Environment,
    EvaluatorSpec,
    GeometryDestination,
    Key4hepStack,
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
from k4opt.problem.names import Names, names

__all__ = [
    "SCHEMA_VERSION",
    "CandidateError",
    "Constraint",
    "Criteria",
    "Destination",
    "Environment",
    "EvaluatorSpec",
    "GeometryDestination",
    "Key4hepStack",
    "Level",
    "MetricRef",
    "Names",
    "Objective",
    "Parameter",
    "ProblemConfigError",
    "ProblemSpec",
    "Reference",
    "Sample",
    "SettingDestination",
    "Stage",
    "StageDestination",
    "load_problem",
    "names",
    "parse_problem",
]
