"""A strategy of your own: compass search on the numeric parameters a study searches.

It measures the middle of the domain, then all its neighbors one step away along each parameter
at once (a batch: it waits until every one of them ended), moves to the best of them if it
improves on the center, and halves the step otherwise; once the step is below ``min_step`` it has
nothing more to propose. Steps are fractions of each parameter's range. It learns only from
measured values: a trial without one (invalid, failed, rejected, malformed, interrupted) or one
that misses a limit (infeasible) is never compared, and a point measured once is not proposed
again.

It decides in ``propose`` and remembers in ``proposed`` and ``tell``, so that a study made again
from its ledger (a resume) tells it everything it was told before, in the same order, and it goes
on exactly where it was::

    k4opt optimize examples/problem/cld-ecal-granularity.yaml \\
      --evaluator examples/evaluators/calo_photon.py:CaloPhoton \\
      --strategy examples/strategies/compass.py:CompassSearch \\
      --strategy-settings examples/strategies/compass.yaml \\
      --trials 12 --workers 2 --output runs/cld-ecal-compass
"""

from __future__ import annotations

from k4opt.optimization import EXHAUSTED, WAITING, Answer, Feedback, Proposal, StudyContext


class CompassSearch:
    """Compass search: settings ``step`` (the first step, a fraction of each range), ``min_step``
    (exhausted below it) and ``objective`` (its label; the first objective by default)."""

    def __init__(
        self,
        study: StudyContext,
        *,
        step: float = 0.25,
        min_step: float = 0.02,
        objective: str | None = None,
    ) -> None:
        if any(parameter.kind == "categorical" for parameter in study.parameters):
            raise ValueError("it searches numeric parameters only; fix the others with --fix")
        labels = [o.label for o in study.objectives]
        if objective is not None and objective not in labels:
            raise ValueError(f"objective {objective!r} is none of {labels}")
        self.index = labels.index(objective) if objective is not None else 0
        self.sign = 1 if study.objectives[self.index].direction == "minimize" else -1
        self.parameters = study.parameters
        self.step, self.min_step = step, min_step
        self.center = self._point(
            {p.name: (p.bounds[0] + p.bounds[1]) / 2 for p in study.parameters}
        )
        self.best: float | None = None  # the score of the center, once measured
        self.round = [self.center]  # the points of this round
        self.queue = [self.center]  # those of them still to propose
        self.open: dict[int, dict[str, float | int]] = {}  # trial -> point, under way
        self.scores: dict[tuple[float | int, ...], float | None] = {}  # None: no value

    def propose(self, trial: int) -> Proposal | Answer:
        if self.queue:
            return Proposal(self.queue[0], {"step": self.step})
        return WAITING if self.open else EXHAUSTED

    def proposed(self, trial: int, proposal: Proposal) -> None:
        point = dict(proposal.values)
        if point in self.queue:
            self.queue.remove(point)
        self.open[trial] = point

    def tell(self, trial: int, feedback: Feedback) -> None:
        point = self.open.pop(trial)
        measured = feedback.status == "valid" and feedback.feasible and feedback.values
        self.scores[self._key(point)] = (
            self.sign * feedback.values[self.index] if measured and feedback.values else None
        )
        if not self.open and not self.queue:
            self._next_round()

    @property
    def exhausted(self) -> bool:
        return not (self.queue or self.open)

    def _next_round(self) -> None:
        """Move to the best point of the round that improves on the center, or halve the step;
        then the neighbors of the center not measured yet, or, if all are, decide again."""
        while self.step >= self.min_step:
            known = [
                (score, index, point)
                for index, point in enumerate(self.round)
                if (score := self.scores.get(self._key(point))) is not None
            ]
            if known and (self.best is None or min(known)[0] < self.best):
                self.best, _, self.center = min(known)
            elif self.round != [self.center]:
                self.step /= 2
            self.round = self._neighbors()
            self.queue = [p for p in self.round if self._key(p) not in self.scores]
            if self.queue:
                return
        self.round, self.queue = [], []

    def _neighbors(self) -> list[dict[str, float | int]]:
        found = []
        for parameter in self.parameters:
            low, high = parameter.bounds
            for sign in (-1, 1):
                moved = dict(self.center)
                moved[parameter.name] = min(
                    max(moved[parameter.name] + sign * self.step * (high - low), low), high
                )
                point = self._point(moved)
                if point != self.center and point not in found:
                    found.append(point)
        return found

    def _point(self, values: dict[str, float]) -> dict[str, float | int]:
        return {
            p.name: round(values[p.name]) if p.kind == "integer" else float(values[p.name])
            for p in self.parameters
        }

    def _key(self, point: dict[str, float | int]) -> tuple[float | int, ...]:
        return tuple(point[p.name] for p in self.parameters)
