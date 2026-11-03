"""Trade-offs against a reference trial: which feasible trials give what is asked of each
objective, compared with the reference's values.

Each :class:`Rule` asks one objective to *improve* by at least an amount, to *worsen* by at most
an amount, or to *keep* its value (worsen by none); an objective without a rule may do anything.
An amount is a share of the reference's value (``relative``: 0.05 is 5 %) or in the objective's
unit. Better and worse follow each objective's direction. With ``within_uncertainty``, a
difference within the uncertainty of the two values (each trial's, added in quadrature; none
counts as zero) counts as no difference: a trial must improve beyond its uncertainty to improve
at all, and worsening within it is no worsening. Nothing here is a judgment of the study: it is
a question asked of its results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from k4opt.frozen import FrozenMap
from k4opt.optimization._results import StudyResults, TrialResult
from k4opt.optimization.errors import OptimizationError
from k4opt.record import ObjectiveValue

type RuleKind = Literal["improve", "worsen", "keep"]
RULE_KINDS: tuple[RuleKind, ...] = ("improve", "worsen", "keep")


@dataclass(frozen=True, slots=True)
class Rule:
    """What a trade-off asks of the objective ``label``: ``improve`` by at least ``amount``,
    ``worsen`` by at most ``amount``, or ``keep`` its value (``amount`` is then ignored);
    ``amount`` a share of the reference's value if ``relative``, else in the objective's unit."""

    label: str
    kind: RuleKind
    amount: float = 0.0
    relative: bool = True

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise OptimizationError(f"a rule improves, worsens or keeps, not {self.kind!r}")
        if not math.isfinite(self.amount) or self.amount < 0:
            raise OptimizationError(
                f"the amount of the rule on {self.label} is {self.amount}; it must be zero or more"
            )


@dataclass(frozen=True, slots=True)
class TradeOffs:
    """The feasible trials (but the reference) that meet every rule, by number; each feasible
    trial's ``change`` of each objective against the reference (its value less the reference's,
    in the objective's unit), and how many of the other feasible trials each rule ``excludes``,
    by label, so that a trade-off no trial meets says which rule is the hardest."""

    reference: TrialResult
    rules: tuple[Rule, ...]
    matches: tuple[int, ...]
    change: Mapping[int, Mapping[str, float]]
    excludes: Mapping[str, int]
    compared: int  # the feasible trials compared with the reference


def trade_offs(
    results: StudyResults,
    reference: int,
    rules: Sequence[Rule],
    *,
    within_uncertainty: bool = True,
) -> TradeOffs:
    """The feasible trials of ``results`` that meet each of ``rules`` against the trial numbered
    ``reference``, which must have objective values. Raises :class:`OptimizationError` for a
    reference without them, a rule on an objective the study does not have or two on one, and a
    relative amount of an objective whose reference value is zero."""
    base = results.trial(reference)
    if base.assessment is None or base.assessment.objectives is None:
        raise OptimizationError(
            f"trial {reference} is {base.outcome} and has no objective values to compare with"
        )
    labels = [o.label for o in results.problem.objectives]
    seen: set[str] = set()
    for rule in rules:
        if rule.label not in labels:
            raise OptimizationError(
                f"the study has no objective {rule.label!r}; it has {', '.join(labels)}"
            )
        if rule.label in seen:
            raise OptimizationError(f"two rules on {rule.label}: one at most for each objective")
        seen.add(rule.label)
    reference_values = {o.label: o for o in base.assessment.objectives}
    limits = {}
    for rule in rules:
        value = reference_values[rule.label].value
        if rule.relative and rule.kind != "keep" and value == 0:
            raise OptimizationError(
                f"trial {reference} has {rule.label} = 0, of which no share can be asked; give "
                "the amount in the objective's unit"
            )
        limits[rule.label] = rule.amount * abs(value) if rule.relative else rule.amount
    compared = [t for t in results.trials if t.outcome == "feasible" and t.number != reference]
    matches = []
    change: dict[int, Mapping[str, float]] = {}
    excludes = dict.fromkeys((rule.label for rule in rules), 0)
    for trial in compared:
        assert trial.assessment is not None
        assert trial.assessment.objectives is not None
        values = {o.label: o for o in trial.assessment.objectives}
        change[trial.number] = FrozenMap(
            {label: values[label].value - reference_values[label].value for label in labels}
        )
        met = True
        for rule in rules:
            gain = _gain(values[rule.label], reference_values[rule.label], within_uncertainty)
            asked = limits[rule.label]
            if rule.kind == "improve":
                ok = gain >= asked
            elif rule.kind == "worsen":
                ok = -gain <= asked
            else:
                ok = gain >= 0
            if not ok:
                excludes[rule.label] += 1
                met = False
        if met:
            matches.append(trial.number)
    return TradeOffs(
        base, tuple(rules), tuple(matches), FrozenMap(change), FrozenMap(excludes), len(compared)
    )


def _gain(value: ObjectiveValue, reference: ObjectiveValue, within_uncertainty: bool) -> float:
    """How much better ``value`` is than ``reference``, in the objective's direction (negative:
    worse); with ``within_uncertainty``, less their combined uncertainty, and zero within it."""
    gain = value.value - reference.value
    if value.direction == "minimize":
        gain = -gain
    if not within_uncertainty:
        return gain
    noise = math.hypot(value.uncertainty or 0.0, reference.uncertainty or 0.0)
    return math.copysign(max(abs(gain) - noise, 0.0), gain)
