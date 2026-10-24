"""Problem component: the declarative definition of a detector optimization problem.

Reads and strictly validates a ``k4opt.problem/v1alpha1`` YAML document into immutable,
typed models. Parsing is a pure data operation: it needs neither the Key4hep stack nor an
evaluator, and it does not check that geometry targets, evaluators or metrics exist. See
``docs/problem.md``.

Typical use::

    from k4opt.problem import load_problem

    problem = load_problem("examples/problem/cld-ecal-granularity.yaml")
    print(problem.parameters[0].bounds, problem.evaluation.backend["ddsim"])
    data = problem.to_dict()  # plain JSON-compatible data
"""

from k4opt.problem.config import load_problem, parse_problem
from k4opt.problem.errors import ProblemConfigError
from k4opt.problem.models import (
    SCHEMA_VERSION,
    Constraint,
    Environment,
    Evaluation,
    GeometrySelector,
    GeometrySpec,
    Key4hepStack,
    Objective,
    Parameter,
    ProblemSpec,
)

__all__ = [
    "SCHEMA_VERSION",
    "Constraint",
    "Environment",
    "Evaluation",
    "GeometrySelector",
    "GeometrySpec",
    "Key4hepStack",
    "Objective",
    "Parameter",
    "ProblemConfigError",
    "ProblemSpec",
    "load_problem",
    "parse_problem",
]
