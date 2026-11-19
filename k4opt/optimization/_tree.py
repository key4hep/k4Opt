"""Internal: the studies of one run, a tree of them: an ordinary study alone, or a study each of
whose trials stands for a study of the level below; the same whatever strategy proposes.

One session evaluates the candidates of every study of the tree, so that they share its workers
and its stages. :class:`Tree` is the source of those candidates
(:meth:`~k4opt.execution.EvaluationSession.evaluate_each`): each is asked for when a slot would
otherwise be idle, from the open studies below first, then a new trial of a study above; it gives
``WAIT`` while what it could do next waits for what runs. Each study's strategy proposes its
trials (:mod:`._proposal`), and each trial ends as any trial does (:mod:`._trials`): kept in the
study's ledger first, its strategy told after.

A strategy may answer that it waits for a trial under way to end, which holds no slot, or that it
is exhausted, or say so itself after its last proposal (a grid all of whose points were
proposed): its study keeps that in its ledger, asks for no more trials, and is done once those
under way ended. A proposal that is no candidate of the study is a
``malformed`` trial, and the study asks again.

A study below the top whose trials share a stage (:func:`_shares`) asks for one trial, then
holds back until something it started ended: the others would only wait for that stage, and their
proposals would know nothing yet. A stage that waits holds no slot, but the candidates under way
are as many as the slots at most (:meth:`~k4opt.execution.EvaluationSession.evaluate_each`), so
each place they would take is one another study could use. Meanwhile the places go to the other
studies, and to new trials above; one nothing else would use goes to a study held back all the
same. A resumed study is held back again until something it started ended: what its trials share
may never have been made, or be gone. The top study is never held back: nothing else would use
the places.

A trial of a study with a level below (a *parent* trial) goes through these states, each decided
from what its ledger and the study below hold, so that a run stopped anywhere goes on from where it
was:

- proposed, then the study of the level below created, with the id the trial decides
  (:func:`~k4opt.optimization._read.inner_id`), so that creating it again is making the same one;
- that study runs until ``trials`` of its trials ended (those an interruption stopped not
  counted), or its strategy is exhausted;
- its best feasible trial is selected from the records (``selected`` in the ledger), or, without
  one, the parent is ``unscored``. Selected once: a selection that later differs stops the run;
- the selected candidate is evaluated once more, on new events (the scope's replicate), into the
  parent's trial directory; what an interruption left there is moved aside first;
- the parent gets that record, as any trial gets its own.

A parent trial is never ended by an interruption: it stays running, and a resume goes on with its
study. The root's lock covers the tree, and a study of a level below is run through its root
alone.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from k4opt.execution import WAIT, EvaluationSession, Rejected, RunOutcome, RunRequest, Wait
from k4opt.frozen import FrozenMap, Json, freeze, same_json
from k4opt.optimization._ledger import LedgerWriter, NotSetUpError
from k4opt.optimization._proposal import EXHAUSTED, WAITING, MalformedError, Past, checked
from k4opt.optimization._read import inner_id, results_of, trial_result
from k4opt.optimization._results import TrialResult, best_trial, no_best_trial
from k4opt.optimization._scope import StudyScope
from k4opt.optimization._strategy import child, derived, open_strategy
from k4opt.optimization._study import (
    INNER_NAME,
    STUDY_NAME,
    inner_directory,
    new_document,
    now,
    read_study,
    strategy_digest,
    trial_directories,
    trial_directory,
    write_study,
)
from k4opt.optimization._trials import (
    OpenStudy,
    Progress,
    Warn,
    check_trials,
    close_interrupted,
    feedback,
    malform,
    recover,
    settle,
    unscore,
)
from k4opt.optimization.errors import OptimizationError
from k4opt.problem.plan import Kind, level_plan

type Check = Callable[[Mapping[str, Any], Mapping[str, Any], StudyScope], None]
"""What refuses to resume a study with what this session runs: ``study.json``, the session's
identity and the scope the study must have."""


@dataclass(slots=True)
class _Node:
    """One study of the tree: its ``budget`` of trials still to ask for, its trials being
    evaluated (``in_flight``: their directories, by number) and those that stand for an open
    study of the level below (``open``: that study, by number)."""

    study: OpenStudy
    budget: int
    in_flight: dict[int, Path] = field(default_factory=dict)
    open: dict[int, _Node] = field(default_factory=dict)
    # asks for no more while something of it is under way, until something it started ended
    held: bool = False
    # its strategy has nothing more to propose
    exhausted: bool = False
    parent: _Node | None = None


def _shares(scope: StudyScope) -> bool:
    """Whether the candidates a study of ``scope`` below the top has evaluated share a stage that
    makes files: one that no parameter it or a level below it searches reaches, which the first
    candidate makes and the others reuse (or would wait for). Its candidates are those of its own
    level (its trials, or for a study whose trials stand for studies below, their selected
    candidates on new events) and those of every level below it, each evaluated as its level
    plans it."""
    spec = scope.spec
    levels = range(scope.level, spec.depth + 1)
    searched = {n for level in levels for n in spec.parameters_of(level)} - set(scope.fixed)
    return any(
        task.id.kind is Kind.STAGE
        and task.given is None
        and task.directory is not None
        and not searched & set(task.varies)
        for level in levels
        for task in level_plan(spec, level).tasks
    )


def open_ledger(
    directory: Path, document: Mapping[str, Any], resume: bool
) -> tuple[LedgerWriter, bool]:
    """The ledger of the study ``document`` describes in ``directory`` to write, committed to its
    strategy: made for a new study; for a resumed one opened, a line a killed process cut short
    removed (then ``True``), or made if the process that made the study stopped before it made
    its ledger (nothing ran in it)."""
    study, strategy = document["id"], strategy_digest(document["strategy"])
    if not resume:
        return LedgerWriter.create(directory, study, strategy), False
    try:
        return LedgerWriter.open(directory, study, strategy)
    except NotSetUpError:
        if trial_directories(directory) or (directory / INNER_NAME).exists():
            raise OptimizationError(
                f"{directory} has trials, but no ledger that holds them; the study cannot go on "
                "without it"
            ) from None
    (directory / "trials.jsonl").unlink(missing_ok=True)  # not one line of it was written
    return LedgerWriter.create(directory, study, strategy), False


class Tree:
    """The studies of a run under ``root`` (its study, opened by the caller), evaluated in
    ``session``. ``identity`` is the session's, ``check`` refuses to resume a study of the level
    below with it; ``progress`` is told each trial of the root as it ends, those a resume
    recovers first."""

    def __init__(
        self,
        session: EvaluationSession,
        root: OpenStudy,
        trials: int,
        identity: Mapping[str, Any],
        check: Check,
        progress: Progress,
        warn: Warn,
        resume: bool = False,
    ) -> None:
        self._session = session
        self._identity = identity
        self._check = check
        self._progress = progress
        self._warn = warn
        self._nodes: dict[str, _Node] = {}
        self._root = self._node(root, trials, resume, progress)

    def _node(
        self, study: OpenStudy, budget: int, resume: bool, progress: Progress, total: bool = False
    ) -> _Node:
        """The node of ``study``, whose strategy is made from its trials, with ``budget`` trials
        to ask for: new ones, or with ``total`` as many as the study runs in all, those that ran
        or go on counted (those an interruption stopped not). A resumed study is read first, so
        that nothing changes on a tree that does not hold up: every trial and its study of the
        level below, checked to their records. Then the session begins in its ledger, the trials
        it left running are recovered (``progress``), and each that stands for a study of the
        level below is opened again."""
        results: list[TrialResult] = []
        if resume:
            if study.scope.nested:
                ledger = study.ledger.ledger
                results_of(study.directory, study.document, study.scope, ledger)
            results = check_trials(study)
        study.ledger.session(now())
        if resume:
            results = recover(study, results, progress)
        entries = study.ledger.ledger.entries
        history = [
            Past(
                entry.number,
                entry.values,
                entry.proposal,
                None
                if result.status == "running"
                else feedback(result, study.record(entry.number) if result.record else None),
            )
            for entry, result in zip(entries, results, strict=True)
        ]
        study.strategy = open_strategy(study.document["strategy"], study.scope, history)
        # resumed too: what its trials share may never have been made, or be gone since
        held = study.scope.level > 0 and _shares(study.scope)
        node = _Node(study, budget, held=held)
        # a strategy whose last proposal an interrupted session kept, but not that it was the last
        _exhausted(node)
        self._nodes[study.document["id"]] = node
        for result in results:
            if result.status == "running":  # it stands for a study of the level below
                self._open(node, result.number)
        if total:
            node.budget -= sum(
                result.status not in ("interrupted", "running") for result in results
            )
            node.budget = max(node.budget - len(node.open), 0)
        return node

    def _open(self, node: _Node, number: int) -> None:
        """Open the study of the level below trial ``number`` of ``node`` stands for: make it, or
        go on with it."""
        parent = node.study
        values = parent.ledger.entry(number).values
        scope = StudyScope.of(
            parent.scope.spec,
            {**parent.scope.fixed, **values},
            level=parent.scope.level + 1,
            parent={"study": parent.document["id"], "trial": number},
        )
        study_id = inner_id(parent.document["id"], number)
        directory = inner_directory(parent.directory, number)
        resume = (directory / STUDY_NAME).is_file()
        if resume:
            document = read_study(directory)
            if document["id"] != study_id:
                raise OptimizationError(
                    f"{directory} holds the study {document['id']!r}, not {study_id!r}, which "
                    f"trial {number} stands for"
                )
            self._check(document, self._identity, scope)
            problems = derived(parent.document["strategy"], document["strategy"], scope, number)
            if problems:
                raise OptimizationError(
                    f"{directory}: the study trial {number} stands for: {'; '.join(problems)}"
                )
        else:
            strategy = child(parent.document["strategy"], scope, number)
            document = new_document(scope, self._identity, strategy, study_id)
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OptimizationError(f"cannot create {directory}: {exc}") from exc
            write_study(directory, document)
        ledger, cut = open_ledger(directory, document, resume)
        try:
            if cut:
                self._warn(f"{ledger.path}: a line a killed process cut short was removed")
            study = OpenStudy(directory, document, scope, ledger, self._warn, keeper=parent.keeper)
            opened = self._node(study, parent.scope.child_trials, resume, _quiet, total=True)
        except BaseException:
            ledger.close()
            raise
        opened.parent = node
        node.open[number] = opened

    def requests(self) -> Iterator[RunRequest | Wait]:
        """The candidates of the tree, each when a slot would otherwise be idle."""
        while (request := self._next(self._root)) is not None:
            if request is WAIT:  # nothing else would use the slot: a study held back may
                request = self._next(self._root, eager=True) or WAIT
            yield request

    def _next(self, node: _Node, eager: bool = False) -> RunRequest | Wait | None:
        """What ``node`` and the studies below it have to evaluate next: a candidate, ``WAIT``
        while all of that waits for what runs, or ``None`` once all of it ended. A study held
        back asks for a new trial only if ``eager``."""
        waiting = False
        for number, below in list(node.open.items()):
            if number in node.in_flight:
                waiting = True  # the selected candidate is evaluated on new events
                continue
            request = self._next(below, eager)
            if request is WAIT:
                waiting = True
            elif request is not None:
                return request
            else:
                request = self._score(node, number, below)
                if request is not None:
                    return request
        while node.budget > 0 and not node.exhausted:
            if node.held and bool(node.in_flight or node.open) and not eager:
                break
            asked = self._ask(node)
            if asked is WAIT:
                waiting = True
                break
            if asked is not None:
                return asked
        return WAIT if waiting or node.in_flight else None

    def _ask(self, node: _Node) -> RunRequest | Wait | None:
        """A new trial of ``node``: the candidate to evaluate, or for a parent trial what its new
        study of the level below has to evaluate; ``WAIT`` while its strategy waits, ``None`` when
        it proposed nothing to evaluate (it is exhausted, or its proposal is malformed)."""
        study = node.study
        number = study.ledger.next_number
        assert study.strategy is not None
        try:
            answer = study.strategy.propose(number)
        except Exception as exc:
            raise OptimizationError(
                f"{study.document['name']}: the strategy could not propose trial {number}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if answer is EXHAUSTED:
            _exhausted(node, asked=True)
            return None
        if answer is WAITING:
            if not node.in_flight and not node.open:
                raise OptimizationError(
                    f"{study.document['name']}: the strategy waits for a trial to end, but none "
                    "of the study's trials is under way"
                )
            return WAIT
        node.budget -= 1
        try:
            values, note = checked(study.scope, answer)
        except MalformedError as exc:
            result = malform(study, number, _note(answer), str(exc))
            _exhausted(node)
            if node is self._root:
                self._progress(result)
            return None
        study.ledger.proposed(number, values, note)
        _exhausted(node)  # its last proposal: kept as such at once, whatever the budget
        if study.scope.nested:
            self._open(node, number)
            return self._next(node)
        output = study.directory / trial_directory(number)
        node.in_flight[number] = output
        # the complete candidate: what the strategy proposed, with the values the study fixes
        candidate = study.scope.candidate(values)
        made_for = {"study": study.document["id"], "trial": number}
        return RunRequest(candidate, output, made_for, study.scope.replicate, study.scope.level)

    def _score(self, node: _Node, number: int, below: _Node) -> RunRequest | None:
        """The study ``below`` that trial ``number`` of ``node`` stands for ended: its best trial,
        selected from its records, evaluated on new events (the request); ``None`` once the trial
        is ``unscored``, without a feasible trial."""
        parent, inner = node.study, below.study
        results = [
            trial_result(inner.directory, inner.document, inner.scope, entry, {})[0]
            for entry in inner.ledger.ledger.entries
        ]
        best = best_trial(results, inner.scope.criteria)
        entry = parent.ledger.entry(number)
        stored = entry.selected
        if best is None:
            if stored is not None:
                raise OptimizationError(
                    f"trial {number} selected {stored!r}, but its inner study has no feasible "
                    "trial any more"
                )
            told = unscore(parent, number, no_best_trial(results))
            self._release(node, number)
            if node is self._root:
                self._progress(told)
            return None
        selected = {
            "study": inner.document["id"],
            "trial": best.number,
            "values": {k: v for k, v in best.candidate.items() if k not in inner.scope.fixed},
        }
        if stored is None:
            parent.ledger.selected(number, selected)
        elif not same_json(stored, selected):
            raise OptimizationError(
                f"trial {number} selected {stored!r}, but the best trial of its inner study is "
                f"now {selected!r}: the study changed"
            )
        name = trial_directory(number)
        output = parent.directory / name
        if output.exists():  # what an interrupted evaluation left: it had no record to keep
            aside = parent.directory / INNER_NAME / f"interrupted-{name}-{uuid.uuid4().hex[:8]}"
            output.rename(aside)
        node.in_flight[number] = output
        made_for = {
            "study": parent.document["id"],
            "trial": number,
            "selected": {"study": selected["study"], "trial": selected["trial"]},
        }
        candidate = parent.scope.candidate({**entry.values, **selected["values"]})
        return RunRequest(candidate, output, made_for, parent.scope.replicate, parent.scope.level)

    def ended(self, request: RunRequest, result: RunOutcome | Rejected | Exception) -> None:
        """End the trial of ``request`` as its candidate's evaluation ended; raise to stop the
        run."""
        assert isinstance(request.run, Mapping)
        node, number = self._nodes[str(request.run["study"])], request.run["trial"]
        assert isinstance(number, int)
        output = node.in_flight.pop(number)  # settled by settle, whatever it raises
        above: _Node | None = node
        while above is not None:  # what it shares with the others of each study above it ended
            above.held, above = False, above.parent
        told = settle(node.study, number, output, result)
        self._release(node, number)
        if node is self._root:
            self._progress(told)

    def _release(self, node: _Node, number: int) -> None:
        """Trial ``number`` of ``node`` ended: the study it stood for ended, and what is kept of
        it is on disk, so that a run holds its open studies, however many it made."""
        opened = node.open.pop(number, None)
        if opened is None:
            return
        del self._nodes[opened.study.document["id"]]
        opened.study.ledger.close()

    def close(self) -> None:
        """Nothing runs any more: end the trials still under way as interrupted, in number order
        (one that cannot be is left running for a resume, as a parent trial is), and close the
        ledgers of the studies below."""
        for node in self._nodes.values():
            for number in sorted(node.in_flight):
                with contextlib.suppress(Exception):  # one that cannot be ended: --resume
                    close_interrupted(node.study, number, node.in_flight[number])
        for node in self._nodes.values():
            if node is not self._root:
                node.study.ledger.close()


def _exhausted(node: _Node, asked: bool = False) -> None:
    """Keep that the strategy of ``node`` has nothing more to propose, once it says so: it
    answered so when ``asked``, or tells it after its last proposal; once only."""
    study = node.study
    assert study.strategy is not None
    if not (asked or study.strategy.exhausted):
        node.exhausted = study.ledger.exhausted
        return
    if not study.ledger.exhausted:
        study.ledger.exhaust()
    node.exhausted = True


def _quiet(result: TrialResult) -> None:
    """The trials a resume recovers in a study below the top are not reported one by one."""


def _note(answer: object) -> Mapping[str, Json]:
    """What a malformed proposal noted, if it is plain JSON; nothing otherwise."""
    note = getattr(answer, "note", None)
    try:
        frozen = freeze(note, "note", OptimizationError)
    except OptimizationError:
        return FrozenMap({})
    return frozen if isinstance(frozen, Mapping) else FrozenMap({})
