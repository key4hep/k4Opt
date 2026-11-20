"""Internal: what a study and the strategy that proposes its candidates say to each other.

A *strategy* proposes values for the parameters one study searches (:class:`Proposal`) when the
study asks for a trial (:meth:`Strategy.propose`); it may answer that it has nothing to propose
until a trial under way has ended (:data:`WAITING`), or nothing ever again (:data:`EXHAUSTED`).
It executes nothing and keeps nothing of the study: the study checks each proposal and keeps it in
its ledger (:mod:`._ledger`), as a trial to evaluate or, if it is no candidate, as a ``malformed``
one (:func:`checked`), and evaluates it. Then the strategy is told what the ledger kept: each
proposal (:meth:`Strategy.proposed`, malformed ones too, with the values as the ledger holds them)
and how each trial ended (:meth:`Strategy.tell`, with :class:`Feedback`), in the order the ledger
kept them.

A strategy is made for a study whenever the study opens, and told every proposal and ending its
ledger holds, in the order they were written (:func:`~k4opt.optimization._trials.replay`): the
very calls a strategy that ran all along received, so that what it learnt in an earlier session
it learns again, and holds nothing the ledger does not. A strategy decides in ``propose``, which is
never replayed, and remembers in ``proposed`` and ``tell``: what it decides that it could not
decide again from those calls (a random draw, a model's answer, a batch) belongs in the note of
its proposal, which the ledger keeps.

What it is told of a trial is what the study holds of it: its status, and for a valid one the
objective values and the normalized margin of each limit (``margin / scale``), computed from its
record, checked (:func:`~k4opt.optimization._trials.feedback`); never a value no evaluation
produced. A proposal the ledger cannot keep as given (no :class:`Proposal`, values or a note that
are no plain JSON object, more than :data:`LIMIT` bytes of them, or something that looks like a
credential) is no trial at all: :func:`recordable` refuses it, and the study stops.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from k4opt.credentials import suspected
from k4opt.evaluation import Scalar
from k4opt.frozen import FrozenMap, Json, freeze
from k4opt.optimization._scope import StudyScope
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import CandidateError, Constraint, Objective, Parameter
from k4opt.problem.candidate import validate_value

if TYPE_CHECKING:
    from k4opt.optimization._results import TrialResult

LIMIT = 64 * 1024
"""The most bytes of JSON a proposal's values and note may take together, as the ledger keeps
them: a note is for what a strategy must remember (a rationale, a model's id, a batch), not for
a transcript, which belongs in a file of its own that the note names by its SHA-256."""


class Answer(Enum):
    """What a strategy answers in place of a proposal."""

    WAITING = "waiting"
    EXHAUSTED = "exhausted"


WAITING = Answer.WAITING
"""Nothing now: ask again once a trial under way has ended."""
EXHAUSTED = Answer.EXHAUSTED
"""Nothing ever again: the study proposes no more trials (a grid all of whose points were
proposed)."""


@dataclass(frozen=True, slots=True)
class Proposal:
    """Values for the parameters the study searches, by name, and what the strategy notes of the
    proposal (plain JSON, recorded with the trial and given back to the strategy with it)."""

    values: Mapping[str, object]
    note: Mapping[str, object] = field(default_factory=lambda: FrozenMap({}))


@dataclass(frozen=True, slots=True)
class Feedback:
    """How a trial ended, as the study holds it.

    ``status`` is the trial's (``valid``, ``invalid``, ``failed``, ``rejected``, ``unscored``,
    ``interrupted``, ``error`` or ``malformed``), and ``result`` the trial as
    :func:`~k4opt.optimization.load_study` reads it back (without the study of the level below a
    trial may stand for): its candidate, the assessment of its record (objectives with value,
    unit and uncertainty; limits with their margins), its diagnostics and times. A valid trial has
    its objective ``values``, in the problem's order, as floats, the normalized margin of each
    limit by name (``constraints``: zero or below when satisfied) and whether every limit is
    satisfied (``feasible``); no other trial has any, whatever its record holds. ``record`` is the
    trial's ``summary.json``, with every measurement, for a trial that has one; ``message`` says
    why a trial without values ended as it did.
    """

    status: str
    result: TrialResult
    values: tuple[float, ...] | None = None
    constraints: Mapping[str, float] = field(default_factory=lambda: FrozenMap({}))
    feasible: bool | None = None
    record: Path | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class StudyContext:
    """What a strategy of one's own is made with of the study it proposes for: the study's
    ``study`` id (the same whenever it opens: to name what the strategy keeps elsewhere, or to
    derive a seed), the ``parameters`` it searches, in problem order, as the problem declares them
    (kind, bounds or choices, unit, title, what each sets), the values every trial has
    (``fixed``: ``k4opt optimize --fix``), the ``objectives`` its trials are judged by (label,
    direction, title), in the order of :attr:`Feedback.values`, its ``constraints`` and the name of
    each of their limits, as :attr:`Feedback.constraints` names them (``limits``). Read-only: the
    problem is the study's, not the strategy's."""

    study: str
    parameters: tuple[Parameter, ...]
    fixed: Mapping[str, Scalar]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]
    limits: tuple[str, ...]

    @classmethod
    def of(cls, scope: StudyScope, study: str) -> StudyContext:
        """The context of the study ``study`` of ``scope``."""
        declared = {parameter.name: parameter for parameter in scope.spec.parameters}
        criteria = scope.criteria
        return cls(
            study,
            tuple(declared[name] for name in scope.searched),
            scope.fixed,
            criteria.objectives,
            criteria.constraints,
            tuple(scope.limit_names),
        )


class Strategy(Protocol):
    """A strategy of one study, made for it whenever it opens, then told every proposal and
    ending its ledger holds, in order. It may say ``exhausted`` (a ``bool`` attribute or property,
    asked after each proposal kept and once it was told the ledger) when it will never propose
    again, so that the study knows at once; without it, it says so by answering
    :data:`EXHAUSTED`."""

    def propose(self, trial: int) -> Proposal | Answer:
        """A proposal for trial ``trial``, or :data:`WAITING` or :data:`EXHAUSTED`: decided, not
        remembered; never asked again for a trial the ledger holds."""
        ...

    def proposed(self, trial: int, proposal: Proposal) -> None:
        """The ledger kept ``proposal`` for trial ``trial``, as it holds it: a candidate, or values
        that are none (``tell`` says it is ``malformed`` right after)."""
        ...

    def tell(self, trial: int, feedback: Feedback) -> None:
        """Trial ``trial`` ended as ``feedback`` says."""
        ...


def exhausted(strategy: object) -> bool:
    """Whether ``strategy`` says it will never propose again."""
    return bool(getattr(strategy, "exhausted", False))


class MalformedError(Exception):
    """A proposal that is no candidate of the study; the message says why."""


def recordable(proposal: object) -> tuple[Mapping[str, Json], Mapping[str, Json]]:
    """The values and the note of ``proposal`` as the ledger keeps them, plain JSON; raises
    :class:`OptimizationError` for one the ledger cannot keep as given: not a :class:`Proposal`,
    values or a note that are no JSON object (NaN, an object, a key that is no text), more than
    :data:`LIMIT` bytes of them, or one that looks like it holds a credential, which a study may
    record nowhere."""
    if not isinstance(proposal, Proposal):
        raise OptimizationError(
            f"it proposed {_shown(proposal)}, neither a Proposal nor WAITING or EXHAUSTED"
        )
    kept = []
    for what, given in (("values", proposal.values), ("note", proposal.note)):
        try:
            plain = freeze(given, f"its {what}", OptimizationError)
        except OptimizationError as exc:
            raise OptimizationError(
                f"its {what} {_shown(given)} are no plain JSON: {exc}"
            ) from None
        if not isinstance(plain, Mapping):
            raise OptimizationError(f"its {what} {_shown(given)} are no JSON object")
        kept.append(plain)
    values, note = kept
    size = len(json.dumps({"values": values, "note": note}, default=dict, separators=(",", ":")))
    if size > LIMIT:
        raise OptimizationError(
            f"its values and note take {size} bytes of JSON, more than a study keeps of a "
            f"proposal ({LIMIT}); keep what is large in a file of its own and note its SHA-256"
        )
    for where, line, what, _ in suspected({"values": values, "note": note}):
        raise OptimizationError(
            f"its {where} (line {line}) looks like it holds {what}, which a study records nowhere"
        )
    return values, note


def checked(scope: StudyScope, values: Mapping[str, Json]) -> dict[str, Scalar]:
    """``values`` (as :func:`recordable` keeps them) as a candidate of the study of ``scope``: the
    values of the parameters it searches, in problem order, each of its parameter's type and in
    its domain; :class:`MalformedError` for values that are none: other parameters, missing ones, a
    value of another type or outside its domain."""
    searched = scope.searched
    unknown = [key for key in values if key not in searched]
    missing = [name for name in searched if name not in values]
    if unknown or missing:
        said = []
        if unknown:
            said.append(f"{', '.join(map(repr, unknown))}, which the study does not search")
        if missing:
            said.append(f"no value of {', '.join(missing)}")
        raise MalformedError(f"the strategy proposed {' and '.join(said)}")
    by_name = {parameter.name: parameter for parameter in scope.spec.parameters}
    for name in searched:
        try:
            validate_value(by_name[name], values[name])
        except CandidateError as exc:
            raise MalformedError(
                f"the strategy proposed a value outside the study's domain: {exc}"
            ) from None
    return {name: values[name] for name in searched}  # type: ignore[misc]  # checked


def _shown(value: object) -> str:
    text = str(value) if isinstance(value, Exception) else repr(value)
    return text if len(text) <= 200 else f"{text[:200]}..."
