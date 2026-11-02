"""Internal: one candidate evaluated, or only checked, from the command line's strings.

:func:`run_candidate` is ``k4opt evaluate``: the candidate given as ``NAME=VALUE`` strings and the
stages supplied as ``SAMPLE/STAGE=FILE`` strings, evaluated in an
:class:`~k4opt.execution.EvaluationSession` of its own. :func:`preview_candidate` is
``k4opt evaluate --dry-run``: the same candidate checked as an evaluation would before it creates
anything, and nothing more.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from k4opt.evaluation import EvaluatorProtocol, Scalar
from k4opt.execution._preflight import Rejected
from k4opt.execution._publish import RunOutcome
from k4opt.execution._session import EvaluationSession
from k4opt.execution.errors import RunError
from k4opt.geometry import Parameter as GeometryParameter
from k4opt.problem.candidate import Configuration, parse_candidate, validate_candidate
from k4opt.problem.models import ProblemSpec


def _parse_inputs(assignments: Sequence[str]) -> dict[str, str]:
    """``SAMPLE/STAGE=FILE`` strings as files by stage label; the session checks the samples,
    stages and files."""
    inputs: dict[str, str] = {}
    for assignment in assignments:
        label, separator, text = assignment.partition("=")
        if not separator or not label or not text:
            raise RunError(f"--input {assignment!r}: expected SAMPLE/STAGE=FILE")
        if label in inputs:
            raise RunError(f"--input {label} is given more than once")
        inputs[label] = text
    return inputs


@dataclass(frozen=True, slots=True)
class Preview:
    """A candidate checked as :func:`run_candidate` checks it, with nothing created or run: the
    problem (``spec``), the candidate's ``values``, what it configures (``configuration``) and the
    attribute each geometry destination sets, with its value now (``targets``, in the order of
    :attr:`~k4opt.problem.ProblemSpec.geometry_destinations`)."""

    spec: ProblemSpec
    values: Mapping[str, Scalar]
    configuration: Configuration
    targets: tuple[GeometryParameter, ...]


def preview_candidate(
    problem_path: str | os.PathLike[str],
    assignments: Sequence[str],
    *,
    implementations: Sequence[type[EvaluatorProtocol]],
    inputs: Sequence[str] = (),
    warn: Callable[[str], None] | None = None,
) -> Preview:
    """Check the candidate given as ``NAME=VALUE`` strings as :func:`run_candidate` does before it
    creates anything (:meth:`EvaluationSession.check`), and stop there: ``k4opt evaluate
    --dry-run``. ``warn`` is called with each of the session's warnings, as there. Raises
    :class:`~k4opt.evaluation.CandidateRejected` if an evaluator's ``check`` rejects it."""
    with EvaluationSession.open(
        problem_path, implementations, inputs=_parse_inputs(inputs)
    ) as session:
        values = parse_candidate(session.spec, assignments)
        checked, _ = validate_candidate(session.spec, values)
        if warn is not None:
            for warning in session.warnings:
                warn(warning)
        configuration = session.check(checked)
        return Preview(session.spec, checked, configuration, session.targets)


def run_candidate(
    problem_path: str | os.PathLike[str],
    assignments: Sequence[str],
    output: str | os.PathLike[str],
    *,
    implementations: Sequence[type[EvaluatorProtocol]],
    inputs: Sequence[str] = (),
    workers: int | Literal["auto", "inline"] = "inline",
    warn: Callable[[str], None] | None = None,
) -> RunOutcome | Rejected:
    """Evaluate the candidate given as ``NAME=VALUE`` strings; ``output`` must not exist. An
    evaluator's ``check`` may reject it instead (:class:`~k4opt.execution.Rejected`), with
    nothing created.

    :meth:`EvaluationSession.evaluate` in a session of its own: ``inputs`` are
    ``SAMPLE/STAGE=FILE`` strings, ``workers`` that of
    :meth:`EvaluationSession.open`. ``warn``
    is called with each of the session's warnings once the candidate is known to be valid, before
    anything is created, and with its :attr:`~EvaluationSession.notice` of diagnostics; the
    warnings are also kept in the summary.
    """
    with EvaluationSession.open(
        problem_path,
        implementations,
        inputs=_parse_inputs(inputs),
        workers=workers,
    ) as session:
        values = parse_candidate(session.spec, assignments)
        # a wrong candidate is reported before the warnings
        validate_candidate(session.spec, values)
        if warn is not None:
            for warning in session.warnings:
                warn(warning)
            if session.notice is not None:
                warn(session.notice)
        return session.evaluate(values, output)
