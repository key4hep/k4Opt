"""Internal: the studies of one run, a tree of them: an ordinary study alone, or a study each of
whose trials stands for a study of the level below.

One session evaluates the candidates of every study of the tree, so that they share its workers
and its stages. :class:`Tree` is the source of those candidates
(:meth:`~k4opt.execution.EvaluationSession.evaluate_each`): each is asked for when a slot would
otherwise be idle, from the open studies below first, then a new trial of a study above; it gives
``WAIT`` while what it could do next waits for what runs. Each study's trials are asked for and
told as those of any study (:mod:`~k4opt.optimization._optuna.tell`).

A trial of a study with a level below (a *parent* trial) goes through these states, each decided
from what is stored, so that a run stopped anywhere goes on from where it was:

- asked, its ``CHILD`` mark set, then the study of the level below created, with the id and seed
  the trial decides, so that creating it again is making the same one;
- that study runs until ``trials`` of its trials ended (those an interruption stopped not
  counted);
- its best feasible trial is selected from the records (``SELECTED``), or, without one, the parent
  is told ``unscored``. Selected once: a selection that later differs stops the run;
- the selected candidate is evaluated once more, on new events (the scope's replicate), into the
  parent's trial directory; what an interruption left there is moved aside first;
- the parent is told that record, as any trial is told its own.

A parent trial is never told FAIL by an interruption: it stays RUNNING, and a resume goes on with
its study. The root's lock covers the tree, and a study of a level below is run through its root
alone.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import BaseDistribution
from optuna.storages import BaseStorage
from optuna.trial import Trial, TrialState

from k4opt.execution import WAIT, EvaluationSession, Rejected, RunOutcome, RunRequest, Wait
from k4opt.optimization._optuna import marks
from k4opt.optimization._optuna.optimizer import (
    CONSTANT_LIAR,
    STARTUP_TRIALS,
    check_distributions,
    check_optuna_study,
    check_resume,
    new_document,
    open_optuna,
)
from k4opt.optimization._optuna.read import results_of, trial_result
from k4opt.optimization._optuna.space import search_space
from k4opt.optimization._optuna.storage import close, open_storage
from k4opt.optimization._optuna.tell import (
    Progress,
    Warn,
    _close_interrupted,
    _recover,
    _settle,
    _Study,
    _unscore,
)
from k4opt.optimization._results import best_trial, no_best_trial
from k4opt.optimization._scope import StudyScope
from k4opt.optimization._study import (
    INNER_NAME,
    STUDY_NAME,
    inner_directory,
    read_study,
    session_seed,
    trial_directory,
    write_study,
)
from k4opt.optimization.errors import OptimizationError


@dataclass(slots=True)
class _Node:
    """One study of the tree: its ``budget`` of trials still to ask for, its trials being
    evaluated (``in_flight``: the trial and its directory, by number) and those that stand for
    an open study of the level below (``open``: the trial and that study, by number)."""

    run: _Study
    space: Mapping[str, BaseDistribution]
    seed: int
    budget: int
    in_flight: dict[int, tuple[Trial, Path]] = field(default_factory=dict)
    open: dict[int, tuple[Trial, _Node]] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.budget == 0 and not self.in_flight and not self.open


class Tree:
    """The studies of a run under ``root`` (its study, opened by the caller), evaluated in
    ``session``. ``identity`` is the session's; ``storage`` the URL they share, if any, and
    ``database`` the storage of it; ``progress`` is told each trial of the root as it ends."""

    def __init__(
        self,
        session: EvaluationSession,
        root: _Study,
        space: Mapping[str, BaseDistribution],
        trials: int,
        identity: Mapping[str, Any],
        storage: str | None,
        database: BaseStorage,
        progress: Progress,
        warn: Warn,
        resume: bool = False,
    ) -> None:
        self._session = session
        self._identity = identity
        self._storage = storage
        self._database = database
        self._progress = progress
        self._warn = warn
        self._nodes: dict[str, _Node] = {}
        self._journals: dict[str, BaseStorage] = {}  # of the open studies below, by study id
        self._root = self._node(root, space, trials, resume)

    def _node(
        self,
        run: _Study,
        space: Mapping[str, BaseDistribution],
        budget: int,
        resume: bool,
        total: bool = False,
    ) -> _Node:
        """The node of ``run``, whose sampler is seeded for this session, with ``budget`` trials
        to ask for: new ones, or with ``total`` as many as the study runs in all, those that ran
        or go on counted (those an interruption stopped not). A resumed study is read first, so
        that nothing new runs on a tree that does not hold up: every trial that ended and its
        study of the level below, checked to their records (one left running is checked when it
        is opened). Each trial it left standing for a study of the level below is opened again,
        one stopped while it was proposed told FAIL, as interrupted."""
        if resume and run.scope.nested:
            results_of(run.directory, run.document, run.scope, run.study, storage=self._storage)
        seed = session_seed(run.document["optimizer"]["sampler"]["seed"], len(run.study.trials))
        run.study.sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=STARTUP_TRIALS, constant_liar=CONSTANT_LIAR
        )
        node = _Node(run, space, seed, budget)
        self._nodes[run.document["id"]] = node
        for frozen in run.study.get_trials(deepcopy=False):
            if frozen.state != TrialState.RUNNING or not run.scope.nested:
                continue
            trial = Trial(run.study, frozen._trial_id)  # the one private access of Optuna
            if set(frozen.params) != set(space):  # stopped while it was asked for
                trial.set_user_attr(marks.RECOVERED, True)
                run.study.tell(trial, state=TrialState.FAIL)
                continue
            trial.set_user_attr(marks.RECOVERED, True)  # finished by this session, not its own
            self._open(node, trial)
        if total:
            for frozen in run.study.get_trials(deepcopy=False):
                if frozen.state != TrialState.RUNNING:
                    result, _ = trial_result(run.directory, run.document, run.scope, frozen, {})
                    node.budget -= result.status != "interrupted"
            node.budget = max(node.budget - len(node.open), 0)
        return node

    def _open(self, node: _Node, trial: Trial) -> None:
        """Open the study of the level below ``trial`` stands for: make it, or go on with it."""
        run, number = node.run, trial.number
        values = {key: trial.params[key] for key in node.space}
        spec, parent_id = run.scope.spec, run.document["id"]
        scope = StudyScope.of(
            spec,
            {**run.scope.fixed, **values},
            level=run.scope.level + 1,
            parent={"study": parent_id, "trial": number},
        )
        study_id = str(uuid.uuid5(uuid.UUID(parent_id), str(number)))
        trial.set_user_attr(marks.CHILD, study_id)
        directory = inner_directory(run.directory, number)
        space = search_space(spec, scope.searched)
        names = scope.limit_names
        resume = (directory / STUDY_NAME).is_file()
        if resume:
            document = read_study(directory)
            check_resume(document, self._identity, scope, space, names, self._storage, self._warn)
            check_distributions(document, scope)
        else:
            seed = session_seed(run.document["optimizer"]["sampler"]["seed"], number)
            document = new_document(
                scope, self._identity, space, names, seed, self._storage, study_id
            )
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OptimizationError(f"cannot create {directory}: {exc}") from exc
            write_study(directory, document)
        database = self._database
        if self._storage is None:
            database = open_storage(directory, None)
            self._journals[document["id"]] = database
        study = open_optuna(directory, document, database, resume)
        child = _Study(directory, document, scope, study, self._warn)
        if resume:
            check_optuna_study(study, document)
            _recover(child, lambda result: None)
            self._session.reuse([directory])  # every trial's record held up: their stages serve
        opened = self._node(child, space, run.scope.child_trials, resume, total=True)
        node.open[number] = (trial, opened)

    def requests(self) -> Iterator[RunRequest | Wait]:
        """The candidates of the tree, each when a slot would otherwise be idle."""
        while (request := self._next(self._root)) is not None:
            yield request

    def _next(self, node: _Node) -> RunRequest | Wait | None:
        """What ``node`` and the studies below it have to evaluate next: a candidate, ``WAIT``
        while all of that waits for what runs, or ``None`` once all of it ended."""
        waiting = False
        for number, (trial, child) in list(node.open.items()):
            if number in node.in_flight:
                waiting = True  # the selected candidate is evaluated on new events
                continue
            below = self._next(child)
            if below is WAIT:
                waiting = True
            elif below is not None:
                return below
            else:
                request = self._score(node, trial, child)
                if request is not None:
                    return request
        if node.budget > 0:
            node.budget -= 1
            return self._ask(node)
        return WAIT if waiting or node.in_flight else None

    def _ask(self, node: _Node) -> RunRequest | Wait | None:
        """A new trial of ``node``: the candidate to evaluate, or for a parent trial what its
        new study of the level below has to evaluate."""
        run = node.run
        trial = run.study.ask(fixed_distributions=dict(node.space))
        name = trial_directory(trial.number)
        output = run.directory / name
        if not run.scope.nested:
            node.in_flight[trial.number] = (trial, output)
        trial.set_user_attr(marks.DIR, name)
        trial.set_user_attr(marks.SAMPLER_SEED, node.seed)
        if run.scope.nested:
            self._open(node, trial)
            return self._next(node)
        # the complete candidate: what the optimizer proposed, with the values the study fixes
        candidate = run.scope.candidate({key: trial.params[key] for key in node.space})
        made_for = {"study": run.document["id"], "trial": trial.number}
        return RunRequest(candidate, output, made_for, run.scope.replicate, run.scope.level)

    def _score(self, node: _Node, trial: Trial, child: _Node) -> RunRequest | None:
        """The study ``child`` that ``trial`` stands for ended: its best trial, selected from
        its records, evaluated on new events (the request); ``None`` once ``trial`` was told
        ``unscored``, without a feasible trial."""
        run, below = node.run, child.run
        frozen = sorted(below.study.get_trials(deepcopy=False), key=lambda t: t.number)
        results = [
            trial_result(below.directory, below.document, below.scope, each, {})[0]
            for each in frozen
        ]
        best = best_trial(results, below.scope.criteria)
        stored = trial.user_attrs.get(marks.SELECTED)
        if best is None:
            if stored is not None:
                raise OptimizationError(
                    f"trial {trial.number} selected {stored!r}, but its inner study has no "
                    "feasible trial any more"
                )
            told = _unscore(run, trial, no_best_trial(results))
            self._release(node, trial.number)
            if node is self._root:
                self._progress(told)
            return None
        selected = {
            "study": below.document["id"],
            "trial": best.number,
            "values": {k: v for k, v in best.candidate.items() if k not in below.scope.fixed},
        }
        if stored is None:
            trial.set_user_attr(marks.SELECTED, selected)
        elif json.dumps(stored, sort_keys=True) != json.dumps(selected, sort_keys=True):
            raise OptimizationError(
                f"trial {trial.number} selected {stored!r}, but the best trial of its inner study "
                f"is now {selected!r}: the study changed"
            )
        name = trial_directory(trial.number)
        output = run.directory / name
        if output.exists():  # what an interrupted evaluation left: it had no record to tell
            aside = run.directory / INNER_NAME / f"interrupted-{name}-{uuid.uuid4().hex[:8]}"
            output.rename(aside)
        node.in_flight[trial.number] = (trial, output)
        made_for = {
            "study": run.document["id"],
            "trial": trial.number,
            "selected": {"study": selected["study"], "trial": selected["trial"]},
        }
        candidate = run.scope.candidate({**trial.params, **selected["values"]})
        return RunRequest(candidate, output, made_for, run.scope.replicate, run.scope.level)

    def ended(self, request: RunRequest, result: RunOutcome | Rejected | Exception) -> None:
        """Tell the study of ``request`` how its candidate ended; raise to stop the run."""
        assert isinstance(request.run, Mapping)
        node, number = self._nodes[str(request.run["study"])], request.run["trial"]
        assert isinstance(number, int)
        trial, output = node.in_flight.pop(number)  # settled by _settle, whatever it raises
        told = _settle(node.run, trial, output, result)
        self._release(node, number)
        if node is self._root:
            self._progress(told)

    def _release(self, node: _Node, number: int) -> None:
        """Trial ``number`` of ``node`` was told: the study it stood for ended, and what is kept
        of it is on disk, so that a run holds its open studies, however many it made."""
        opened = node.open.pop(number, None)
        if opened is None:
            return
        study = opened[1].run.document["id"]
        del self._nodes[study]
        journal = self._journals.pop(study, None)
        if journal is not None:
            close(journal)

    def close(self) -> None:
        """Nothing runs any more: close the trials still under way as interrupted, in number
        order (one that cannot be told is left RUNNING for a resume, as a parent trial is), and
        the storages of the studies below."""
        for node in self._nodes.values():
            for number in sorted(node.in_flight):
                with contextlib.suppress(Exception):  # one that cannot be told: --resume
                    _close_interrupted(node.run, *node.in_flight[number])
        for journal in self._journals.values():
            close(journal)
