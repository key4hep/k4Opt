"""Problem component: the declarative definition of a detector optimization problem.

Reads and strictly validates a ``k4opt.problem/v1alpha1`` YAML document into immutable,
typed models: the parameters, the pipeline ``stages`` (each run by an executor or given as a
file), the ``evaluators`` (named bindings of evaluator implementations) and the objectives and
constraints, each on a metric of one of them (:class:`MetricRef`). ``load_problem`` also
resolves the geometry and every parameter target in the active k4geo tree (``$K4GEO``);
``parse_problem`` is the pure variant that needs no geometry. Evaluators and metrics are not
checked. See ``docs/problem.md``.

An :class:`EvaluationSession` evaluates candidates of a problem file with its evaluators, given
as classes of which every candidate gets new instances, each candidate into a run directory with
a ``summary.json``, and judges each against the objectives and constraints
(:class:`Assessment`). It knows no optimizer: whatever proposes candidates, a person or an
optimization library, calls it the same way. See ``docs/evaluation.md``.

Typical use::

    from k4opt.problem import EvaluationSession, load_problem

    problem = load_problem("problem.yaml")
    print(problem.parameters[0].bounds, problem.evaluators[0].name)
    data = problem.to_dict()  # plain JSON-compatible data

    session = EvaluationSession.open("problem.yaml", [MyEvaluator])  # the class
    outcome = session.evaluate({"constants.ECal_cell_size": 7.0}, "runs/7mm")
    outcome.assessment.feasible  # True, False or None (not assessed)
"""

from k4opt.problem._assess import Assessment, Limit, ObjectiveValue
from k4opt.problem._run import SUMMARY_NAME, EvaluationSession, RunOutcome
from k4opt.problem.config import load_problem, parse_problem
from k4opt.problem.errors import CandidateError, ProblemConfigError, RecordError, RunError
from k4opt.problem.models import (
    SCHEMA_VERSION,
    Constraint,
    Environment,
    EvaluatorSpec,
    GeometrySelector,
    Key4hepStack,
    MetricRef,
    Objective,
    Parameter,
    ProblemSpec,
    Stage,
)

__all__ = [
    "SCHEMA_VERSION",
    "SUMMARY_NAME",
    "Assessment",
    "CandidateError",
    "Constraint",
    "Environment",
    "EvaluationSession",
    "EvaluatorSpec",
    "GeometrySelector",
    "Key4hepStack",
    "Limit",
    "MetricRef",
    "Objective",
    "ObjectiveValue",
    "Parameter",
    "ProblemConfigError",
    "ProblemSpec",
    "RecordError",
    "RunError",
    "RunOutcome",
    "Stage",
    "load_problem",
    "parse_problem",
]
