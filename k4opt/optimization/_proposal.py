"""Internal: what a study and the strategy that proposes its candidates say to each other.

A *strategy* proposes values for the parameters one study searches (:class:`Proposal`), when the
study asks for a trial (:meth:`Strategy.propose`), and is told how each trial ended
(:meth:`Strategy.tell`, with :class:`Feedback`), in the order the trials end. It may answer that
it has nothing to propose until a trial under way has ended (:data:`WAITING`), or nothing ever
again (:data:`EXHAUSTED`). It executes nothing and keeps nothing of the study: the study checks
each proposal (:func:`checked`; one that is no candidate is ``malformed``), evaluates it, and
keeps every trial in its ledger (:mod:`._ledger`). A strategy is made for a study from all the
study's trials so far (:class:`Past`, by number), when the study opens: what it learnt in an
earlier session it learns again from them, so that it never holds what the ledger does not.

What it is told of a trial is what the study holds of it: its status, and for a valid one the
objective values and the normalized margin of each limit (``margin / scale``), computed from its
record, checked (:func:`~k4opt.optimization._trials.feedback`); never a value no evaluation
produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from k4opt.evaluation import Scalar
from k4opt.frozen import FrozenMap, Json, freeze
from k4opt.optimization._scope import StudyScope
from k4opt.problem import CandidateError
from k4opt.problem.candidate import validate_value


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
    ``interrupted``, ``error`` or ``malformed``). A valid trial has its objective ``values``, in
    the problem's order, the normalized margin of each limit by name (``constraints``: zero or
    below when satisfied) and whether every limit is satisfied (``feasible``); no other trial
    has any. ``record`` is the trial's ``summary.json``, with every measurement and diagnostic,
    for a trial that has one; ``message`` says why a trial without values ended as it did.
    """

    status: str
    values: tuple[float, ...] | None = None
    constraints: Mapping[str, float] = field(default_factory=lambda: FrozenMap({}))
    feasible: bool | None = None
    record: Path | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class Past:
    """A trial a strategy is made with: its number, the values proposed, what the strategy noted
    of the proposal, and how it ended (``None``: it is still under way, standing for a study of
    the level below)."""

    number: int
    values: Mapping[str, Scalar]
    note: Mapping[str, Json]
    feedback: Feedback | None


class Strategy(Protocol):
    """A strategy of one study, made for it from its trials so far."""

    def propose(self, number: int) -> Proposal | Answer:
        """A proposal for trial ``number``, or :data:`WAITING` or :data:`EXHAUSTED`."""
        ...

    def tell(self, number: int, feedback: Feedback) -> None:
        """Trial ``number`` (proposed by this strategy, or given with :class:`Past` as under way)
        ended as ``feedback`` says."""
        ...


class MalformedError(Exception):
    """A proposal that is no candidate of the study; the message says why."""


def checked(scope: StudyScope, proposal: object) -> tuple[dict[str, Scalar], Mapping[str, Json]]:
    """The values of ``proposal`` for the parameters ``scope`` searches, in problem order, each of
    its parameter's type and in its domain, and its note as plain JSON; :class:`MalformedError`
    for a proposal that is not one of those: other parameters, missing ones, a value of another
    type or outside its domain, a note that is no JSON object."""
    if not isinstance(proposal, Proposal):
        raise MalformedError(f"the strategy proposed {_shown(proposal)}, not a proposal")
    try:
        note = freeze(proposal.note, "the strategy's note", MalformedError)
    except MalformedError as exc:
        raise MalformedError(
            f"the strategy's note of its proposal is not plain JSON: {exc}"
        ) from None
    if not isinstance(note, Mapping):
        raise MalformedError(
            f"the strategy's note of its proposal is {_shown(note)}, not an object"
        )
    values = proposal.values
    if not isinstance(values, Mapping) or not all(isinstance(key, str) for key in values):
        raise MalformedError(f"the strategy proposed {_shown(values)}, not values by parameter")
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
    return {name: values[name] for name in searched}, note  # type: ignore[misc]  # checked


def _shown(value: object) -> str:
    text = str(value) if isinstance(value, Exception) else repr(value)
    return text if len(text) <= 200 else f"{text[:200]}..."
