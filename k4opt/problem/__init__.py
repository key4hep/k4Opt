"""Problem component: the declarative definition of a detector optimization problem.

Reads and strictly validates a ``k4opt.problem/v1alpha1`` YAML document into immutable,
typed models: the parameters, the pipeline ``stages`` (each run by an executor or given as a
file), the ``evaluators`` and the objectives and constraints. ``load_problem`` also resolves the
geometry entry and every parameter target in the active k4geo tree (``$K4GEO``);
``parse_problem`` is the pure variant that needs no geometry. Evaluators and metrics are not
checked. See ``docs/problem.md``.

Typical use::

    from k4opt.problem import load_problem

    problem = load_problem("problem.yaml")
    print(problem.parameters[0].bounds, problem.evaluators[0].name)
    data = problem.to_dict()  # plain JSON-compatible data
"""

from k4opt.problem.config import load_problem, parse_problem
from k4opt.problem.errors import CandidateError, ProblemConfigError
from k4opt.problem.models import (
    SCHEMA_VERSION,
    Constraint,
    Environment,
    EvaluatorSpec,
    GeometrySelector,
    GeometrySpec,
    Key4hepStack,
    Objective,
    Parameter,
    ProblemSpec,
    Stage,
)

__all__ = [
    "SCHEMA_VERSION",
    "CandidateError",
    "Constraint",
    "Environment",
    "EvaluatorSpec",
    "GeometrySelector",
    "GeometrySpec",
    "Key4hepStack",
    "Objective",
    "Parameter",
    "ProblemConfigError",
    "ProblemSpec",
    "Stage",
    "load_problem",
    "parse_problem",
]
