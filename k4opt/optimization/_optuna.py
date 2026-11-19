"""Internal: Optuna's TPE sampler as a strategy, and the only module of k4opt that imports Optuna.

Optuna keeps nothing on disk: its study is made in memory when a study of k4opt opens
(:class:`TPE`), with every trial the study's ledger holds, in number order, as Optuna would have
held them (COMPLETE with their objective values and constraint values, FAIL, or RUNNING for one
still under way), so that its trials and the study's have the same numbers; then the sampler is
seeded for the session (:func:`session_seed`). TPE learns from the trials alone, so a session
proposes what one that kept Optuna's study on disk would have proposed, and a session stopped
anywhere loses nothing of it. ``study.json`` records the sampler, its seed and the distributions
(:func:`config`); a resume needs the same, and the same version of Optuna.

Each parameter becomes one distribution (:func:`search_space`): continuous uniformly between its
bounds, integer every integer between them, categorical one of its choices. A trial is told its
objective values and the normalized margin of each limit as its constraint value; one without
values (invalid, failed, rejected, ...) is FAIL, which TPE ignores.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Any, Literal, cast

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    distribution_to_json,
)
from optuna.trial import Trial, TrialState, create_trial

from k4opt.frozen import same_json
from k4opt.optimization._proposal import Answer, Feedback, Past, Proposal
from k4opt.optimization._scope import StudyScope
from k4opt.optimization.errors import OptimizationError
from k4opt.problem import ProblemSpec

NAME = "optuna"
STARTUP_TRIALS = 10  # random proposals until this many trials are COMPLETE
CONSTANT_LIAR = True  # trials still running count as bad ones, so trials at once spread out
SEEDING = "per session, from the seed and the number of trials"

_EXACT_INTEGERS = 2**53  # TPE samples integers through floats, exact up to here


def session_seed(study_seed: int, trials: int) -> int:
    """The sampler's seed for a session that starts when the study has ``trials`` trials.

    Derived, so that a resumed session does not replay the proposals the first session started
    with; the same history and seed give the same proposals again.
    """
    digest = hashlib.sha256(f"{study_seed}:{trials}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _float_bound(name: str, value: int | float) -> float:
    try:
        bound = float(value)
    except OverflowError:
        bound = None
    if bound is None or bound != value:
        text = str(value)
        shown = text if len(text) <= 24 else f"{text[:12]}... ({len(text)} digits)"
        raise OptimizationError(f"{name}: the bound {shown} is no float, which the sampler draws")
    return bound


def search_space(
    spec: ProblemSpec, searched: Collection[str] | None = None
) -> dict[str, BaseDistribution]:
    """The distribution of each parameter ``searched`` (every one, by default), by name, in
    problem order.

    Continuous parameters are sampled uniformly between their bounds, with no log scale and no
    step; their bounds must be floats exactly, and so must their range. Integer bounds must stay
    within 2**53, below which the sampler's floats hold every integer. A categorical parameter
    takes one of its declared choices, as declared.
    """
    space: dict[str, BaseDistribution] = {}
    for parameter in spec.parameters:
        name = parameter.name
        if searched is not None and name not in searched:
            continue
        if parameter.kind == "categorical":
            assert parameter.choices is not None
            space[name] = CategoricalDistribution(list(parameter.choices))
            continue
        assert parameter.bounds is not None
        low, high = parameter.bounds
        if parameter.kind == "integer":
            if max(abs(low), abs(high)) > _EXACT_INTEGERS:
                raise OptimizationError(
                    f"{name}: integer bounds beyond 2**53 in magnitude cannot be sampled exactly"
                )
            space[name] = IntDistribution(int(low), int(high))
            continue
        lower, upper = _float_bound(name, low), _float_bound(name, high)
        if upper - lower == float("inf"):
            raise OptimizationError(f"{name}: the range [{low}, {high}] is too wide to sample")
        space[name] = FloatDistribution(lower, upper)
    return space


def config(scope: StudyScope, seed: int) -> dict[str, Any]:
    """What ``study.json`` records of the sampler of the study of ``scope``, seeded with
    ``seed``."""
    return {
        "name": NAME,
        "version": optuna.__version__,
        "sampler": {
            "name": "TPESampler",
            "seed": seed,
            "n_startup_trials": STARTUP_TRIALS,
            "constant_liar": CONSTANT_LIAR,
            "seeding": SEEDING,
        },
        "distributions": {
            key: json.loads(distribution_to_json(distribution))
            for key, distribution in search_space(scope.spec, scope.searched).items()
        },
    }


def _seed(recorded: Mapping[str, Any]) -> int:
    seed = (recorded.get("sampler") or {}).get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise OptimizationError(f"the recorded sampler seed {seed!r} is no seed in [0, 2**32)")
    return seed


def differences(recorded: Mapping[str, Any], scope: StudyScope) -> list[str]:
    """Why the recorded sampler is not the one this k4opt would run the study of ``scope`` with,
    each said for a person; none if it is."""
    try:
        expected = config(scope, _seed(recorded))
    except OptimizationError as exc:
        return [str(exc)]
    found = [
        f"its recorded {key} is {recorded.get(key)!r}, not {value!r}"
        for key, value in expected.items()
        if key != "version" and not same_json(recorded.get(key), value)
    ]
    if recorded.get("version") != optuna.__version__:
        found.append(
            f"the study was made with Optuna {recorded.get('version')}, this is Optuna "
            f"{optuna.__version__}: install optuna=={recorded.get('version')} to resume it"
        )
    return found


def child(recorded: Mapping[str, Any], scope: StudyScope, number: int) -> dict[str, Any]:
    """The sampler of the study ``scope`` of the level below, started for trial ``number`` of
    the study whose sampler is ``recorded``: seeded from its seed and the trial's number."""
    return config(scope, session_seed(_seed(recorded), number))


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Optuna logs every trial at INFO; k4opt reports them itself."""
    level = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        yield
    finally:
        optuna.logging.set_verbosity(level)


class TPE:
    """Optuna's TPE sampler for one study, in an Optuna study in memory made from the study's
    trials so far (``history``), each as Optuna holds a trial that ended so, or as a RUNNING
    trial for one still under way."""

    def __init__(
        self, scope: StudyScope, recorded: Mapping[str, Any], history: Sequence[Past]
    ) -> None:
        self._space = search_space(scope.spec, scope.searched)
        self._running: dict[int, Trial] = {}
        with _quiet():
            # filled by a sampler that draws nothing, so that TPE's draws are those of a study
            # that held these trials all along
            self._study = optuna.create_study(
                directions=cast(list[Literal["minimize", "maximize"]], scope.directions),
                sampler=optuna.samplers.RandomSampler(seed=0),
            )
            for past in history:
                if past.feedback is None:
                    self._study.enqueue_trial(dict(past.values))
                    self._running[past.number] = self._take(past.number)
                    continue
                self._study.add_trial(self._frozen(past))
        self.seed = session_seed(_seed(recorded), len(history))
        self._study.sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=STARTUP_TRIALS, constant_liar=CONSTANT_LIAR
        )

    def _frozen(self, past: Past) -> optuna.trial.FrozenTrial:
        feedback = past.feedback
        assert feedback is not None
        params = dict(past.values)
        distributions = {key: self._space[key] for key in params}
        if feedback.status != "valid":
            return create_trial(state=TrialState.FAIL, params=params, distributions=distributions)
        assert feedback.values is not None
        return create_trial(
            params=params,
            distributions=distributions,
            values=list(feedback.values),
            constraints=dict(feedback.constraints),
        )

    def _take(self, number: int) -> Trial:
        """Ask Optuna for trial ``number``, which must be the next of its study too."""
        trial = self._study.ask(fixed_distributions=dict(self._space))
        if trial.number != number:
            raise OptimizationError(
                f"Optuna's trial {trial.number} is not trial {number} of the study"
            )
        return trial

    def propose(self, number: int) -> Proposal | Answer:
        with _quiet():
            trial = self._take(number)
        self._running[number] = trial
        return Proposal({key: trial.params[key] for key in self._space}, {"seed": self.seed})

    def tell(self, number: int, feedback: Feedback) -> None:
        trial = self._running.pop(number)
        with _quiet():
            if feedback.status != "valid":
                self._study.tell(trial, state=TrialState.FAIL)
                return
            assert feedback.values is not None
            for key, value in feedback.constraints.items():
                trial.set_constraint(key, value)
            self._study.tell(trial, list(feedback.values))
