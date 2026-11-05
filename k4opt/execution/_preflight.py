"""Internal: one candidate's evaluations prepared before anything of it is created or runs.

:func:`_preflight` prepares a candidate from its configuration
(:func:`~k4opt.problem.candidate.configure`) and what holds for every candidate
(:class:`~k4opt.execution._binding._Binding`): the stages of each sample with what the candidate
sets in them, checked complete; each evaluator's settings with what it sets in them, and the
request each evaluation gets (:class:`_Prepared`). Then each evaluator class's ``check`` is asked
about its request: a required evaluation may reject the candidate (:class:`Rejected`), which is
then neither created nor run. A diagnostic cannot: what the candidate's status does not depend on
cannot veto it either, so a diagnostic whose ``check`` rejects the candidate is only not run, and
recorded ``invalid``. No instance is created here: an evaluation's instance is created where it
runs.

``check`` runs in the session's own process and is not time-limited: it is the evaluator's
inexpensive check of its request, not part of the evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from k4opt.evaluation import (
    CandidateRejected,
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
    EvaluationRequest,
    check_settings,
    collection_names,
    with_defaults,
)
from k4opt.execution._binding import _Binding
from k4opt.execution._instances import _check_unchanged
from k4opt.execution._stages import DryRuns, StageContext, check_stages, try_candidate
from k4opt.frozen import FrozenMap, Json
from k4opt.problem.candidate import Configuration
from k4opt.problem.models import EvaluatorSpec
from k4opt.problem.plan import Kind, Plan, TaskId, evaluations


@dataclass(frozen=True, slots=True)
class Rejected:
    """A candidate rejected before anything of it was created, ``by`` a required task, and why
    (``message``): the evaluation whose ``check`` rejected it (``gamma/calo.photon``,
    ``readout``), or the reconstruction whose job does not take the arguments the candidate sets
    (``gamma/reconstruction``). It was not evaluated, so it has no record, no measurement and no
    status of an evaluation."""

    by: str
    message: str

    def __str__(self) -> str:
        return f"rejected by {self.by}: {self.message}"


@dataclass(frozen=True, slots=True)
class _Prepared:
    """What the preflight of one candidate established before anything runs: its
    ``configuration``, the stage conditions of each sample with the stages it configures
    (``contexts``, checked complete), and by evaluation, in plan order, the entry that binds it
    with what the candidate sets (``entries``) and the request its evaluator's ``check`` was
    asked about (``requests``); ``declined`` holds, with its message, each diagnostic whose
    ``check`` rejected the candidate, which is not run. ``level`` is the level of the
    optimization it is evaluated for, whose :attr:`plan` says which of its tasks are required.
    """

    binding: _Binding
    configuration: Configuration
    contexts: Mapping[str, StageContext]  # by sample
    entries: Mapping[TaskId, EvaluatorSpec]
    requests: Mapping[TaskId, EvaluationRequest]
    declined: Mapping[TaskId, str] = FrozenMap({})
    level: int = 0

    @property
    def plan(self) -> Plan:
        """The candidate's tasks, those required by the criteria of its level."""
        return self.binding.plans[self.level]


def _preflight(
    binding: _Binding,
    configuration: Configuration,
    dry_runs: DryRuns | None = None,
    level: int = 0,
) -> _Prepared | Rejected:
    """Prepare one candidate, configured as ``configuration`` says, before a file is created or a
    stage runs; :class:`Rejected` if the ``check`` of a required evaluation rejects it, or the job
    of a required reconstruction does not take the arguments the candidate's values set in it:
    required by the criteria of ``level``, the level of the optimization it is evaluated for.

    Checks that each evaluator class declares what it did when the session bound it, the stages
    of each sample with what the candidate sets in them, complete, the settings of each
    evaluation, with what the candidate sets in them, and its collections; then calls
    each evaluator class's ``check`` once with the request its evaluation would get for this
    candidate (its settings and collections resolved, the events, seed and properties of its
    sample; no roles, since nothing exists yet for them to read), in plan order, up to the first
    required evaluation whose ``check`` rejects it (:class:`~k4opt.evaluation.CandidateRejected`);
    a diagnostic's rejection only keeps that diagnostic from running (``declined``). Then, with
    ``dry_runs``, each required reconstruction whose arguments the candidate sets is dry-run
    (:func:`~k4opt.execution._stages.try_candidate`): the ends of a parameter's domain, which the
    session tried, do not decide every value between them. Raises
    :class:`EvaluationContractError` for settings that do not match what an evaluator declares
    and for a ``check`` that ends with ``EvaluationFailed`` or ``EvaluationInvalid`` (nothing ran
    that could fail or be invalid), and whatever else a ``check`` or a stage's check raises.
    """
    contexts = {
        sample: context.configured(configuration.samples[sample].stages)
        for sample, context in binding.contexts.items()
    }
    for sample, configured in contexts.items():
        check_stages(configured, *binding.targets[sample], provided=_supplied(binding, sample))
    entries = dict(evaluations(configuration.samples, configuration.evaluators))
    for task, bound in binding.evaluations:  # before any check(), which may use the candidate
        _check_unchanged(bound)  # the requests are made from what the class declares
        entry, cls = entries[task], bound.implementation
        check_settings(cls, with_defaults(cls, entry.settings))
        collection_names(cls, entry.collections)
    requests = {}
    for task, bound in binding.evaluations:
        context = None if task.sample is None else contexts[task.sample]
        properties = {} if task.sample is None else binding.properties[task.sample]
        requests[task] = _request(context, entries[task], bound.implementation, properties)
    declined: dict[TaskId, str] = {}
    for task, bound in binding.evaluations:
        check = getattr(bound.implementation, "check", None)
        if check is None:
            continue
        try:
            check(requests[task])
        except CandidateRejected as exc:
            if binding.plans[level][task].required:
                return Rejected(task.label, str(exc))
            declined[task] = str(exc)  # a diagnostic decides nothing of the candidate
        except (EvaluationFailed, EvaluationInvalid) as exc:
            raise EvaluationContractError(
                f"evaluator {entries[task].binding!r}: check() ended with "
                f"{type(exc).__name__} ({exc}), but nothing has run that could fail or be "
                "invalid; reject the candidate with cls.reject('...'), or raise "
                "EvaluationContractError for settings the evaluator cannot work with"
            ) from exc
    # a diagnostic reconstruction that refuses its arguments fails, and decides nothing
    for task in binding.rejecting[level] if dry_runs is not None else ():
        if task.kind is Kind.STAGE and task.name == "reconstruction" and dry_runs is not None:
            assert task.sample is not None
            pending = binding.contexts[task.sample].pending.get("reconstruction", ())
            reason = try_candidate(contexts[task.sample], pending, dry_runs)
            if reason is not None:
                return Rejected(task.label, reason)
    return _Prepared(
        binding,
        configuration,
        FrozenMap(contexts),
        FrozenMap({task: entries[task] for task, _ in binding.evaluations}),
        FrozenMap(requests),
        FrozenMap(declined),
        level,
    )


def _supplied(binding: _Binding, sample: str) -> set[str]:
    """The stages of ``sample`` the caller supplies, by name."""
    return {
        task.id.name
        for task in binding.plan.of(Kind.STAGE)
        if task.id.sample == sample and task.given == "supplied"
    }


def _request(
    context: StageContext | None,
    entry: EvaluatorSpec,
    evaluator: object = None,
    properties: Mapping[str, Json] | None = None,
) -> EvaluationRequest:
    """The request an evaluator receives, for ``check`` and ``evaluate`` alike.

    ``settings`` are those of its ``entry`` in the problem plus the defaults the evaluator class
    declares, each quantity in its declared unit; ``collections`` the collection of each role it
    declares; ``events`` and ``seed`` are those of the stage it reads in the sample of
    ``context``, ``None`` for an evaluation of the candidate (``context`` is ``None``), and
    ``properties`` those of that sample.
    """
    events, seed = (None, None) if context is None else context.conditions(entry.stage)
    return EvaluationRequest(
        settings=with_defaults(evaluator, entry.settings),
        collections=collection_names(evaluator, entry.collections),
        events=events,
        seed=seed,
        properties=properties or {},
    )
