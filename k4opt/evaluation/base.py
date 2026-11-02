"""The base class for evaluators and the shorthands it enables.

Subclassing :class:`Evaluator` is optional: any class with ``name``, ``version`` and
``evaluate`` is accepted (see :class:`~k4opt.evaluation.protocol.EvaluatorProtocol`). Whatever an
evaluator declares is plain data on its class, read without creating it
(:mod:`k4opt.evaluation.declaration`). The base class adds what most evaluators would write
anyway:

* ``collections = {"hits": Collection("SimCalorimeterHit")}`` declares the EDM4hep collections
  the evaluator reads from the file of its stage, by role: the problem names each one (or a
  ``default`` does), and ``self.events(request)`` yields every event with them as attributes
  (``event.hits``), all read from that one event;
* ``settings = {"threshold": Quantity("GeV"), "cut": 0.5}`` declares the evaluator's settings
  (:mod:`k4opt.evaluation.settings`): a type for a required key, a default value for an optional
  one, a :class:`~k4opt.evaluation.Quantity` for a number with a unit. ``check`` rejects missing
  and unknown keys, wrong types and blank required text before anything runs; the request carries
  the defaults, and each quantity as a float in its declared unit;
* ``metrics = {"hit_cells": "1"}`` maps each metric to its unit; its stage is ``stage``,
  or the metric is written ``(stage, metric)``;
* ``artifacts = {"histogram": "shower widths, JSON {edges, counts}"}`` names the files a
  ``valid`` result returns, each with what it holds: every one, written in ``request.workdir``,
  and no other;
* ``roles = {"signal": Role(metrics=[("analysis", "yield")])}`` makes it a combined evaluator,
  which reads no stage but the results of other evaluations, by role: the problem says which
  ones, and ``request.roles["signal"]`` holds them (:class:`~k4opt.evaluation.Role`);
* ``evaluate`` may return ``{"hit_cells": 3.2}`` or ``{"hit_cells": [3, 4, 2]}`` (a number, or
  per-event values, a list or a numpy array, that become their unweighted mean with its
  standard error) instead of an :class:`EvaluationResult`;
* ``cls.reject(...)`` in ``check`` rejects the candidate before anything of it runs;
  ``self.invalid(...)`` and ``self.fail(...)`` end an evaluation with that status.

``request.read(stage, collection, type)`` reads one collection of any stage's file; collections
read that way are not matched up event by event.

An evaluation's lifecycle: ``check`` (a classmethod) is called with the candidate's request in
the session's own process, before anything of the candidate is created; then, where the
evaluation runs (a worker process, or that same process), one new instance is created and its
``evaluate`` called once, and the instance is dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import NoReturn, cast

from k4opt.evaluation.declaration import (
    Collection,
    MetricKey,
    Role,
    _qualified,
    collection_names,
    declared_collections,
    metric_units,
)
from k4opt.evaluation.edm4hep import Event, read_events
from k4opt.evaluation.errors import (
    CandidateRejected,
    EvaluationContractError,
    EvaluationFailed,
    EvaluationInvalid,
)
from k4opt.evaluation.models import EvaluationRequest, EvaluationResult, Measurement, Status
from k4opt.evaluation.settings import check_settings

type Values = Mapping[MetricKey, int | float | Iterable[int | float]]
"""What ``evaluate`` may return instead of an :class:`EvaluationResult`."""


def result_from(evaluator: object, returned: EvaluationResult | Values) -> EvaluationResult:
    """The result that ``evaluate`` returned, or the ``valid`` one its metric values stand for."""
    if isinstance(returned, EvaluationResult):
        if returned.status is Status.BLOCKED:
            raise EvaluationContractError(
                "an evaluator never returns a blocked result: k4opt alone marks an evaluator "
                "blocked, without calling it, when something it reads did not succeed; return "
                "failed or invalid"
            )
        return returned
    if not isinstance(returned, Mapping):
        raise EvaluationContractError(
            "evaluator must return an EvaluationResult or a mapping of metric to value, "
            f"got {type(returned).__name__}"
        )
    units = metric_units(evaluator) or {}
    measurements = []
    for key, value in returned.items():
        stage, metric = _qualified(evaluator, key)
        unit = units.get((stage, metric))
        # per-event values: a list, a tuple or a numpy array, say; text is not a sequence of numbers
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
            try:
                values = list(value)
            except TypeError:  # iterable in type only, like a zero-dimensional numpy array
                raise EvaluationContractError(
                    f"metric {stage}/{metric}: {value!r} is neither a number nor per-event values"
                ) from None
            if not values:  # may be a bug, so judging validity is left to the evaluator
                raise EvaluationContractError(
                    f"metric {stage}/{metric}: no per-event values to average; if nothing could "
                    "be measured, end the evaluation with self.invalid('...')"
                )
            measurements.append(Measurement.from_values(stage, metric, values, unit))
        else:  # a number; Measurement rejects anything else
            measurements.append(Measurement(stage, metric, cast(float, value), unit))
    return EvaluationResult.valid(measurements)


class Evaluator:
    """Base class of evaluators; see the module documentation. ``name`` and ``version`` are
    required; ``stage`` (the pipeline stage it reads), ``collections``, ``settings``,
    ``metrics``, ``artifacts`` and ``roles`` (a combined evaluator's inputs) are optional, each
    plain data on the class."""

    name: str
    version: str
    stage: str | None = None
    collections: Mapping[str, Collection] | None = None
    settings: Mapping[str, object] | None = None
    metrics: Mapping[MetricKey, str] | None = None
    artifacts: Mapping[str, str] | None = None
    roles: Mapping[str, Role] | None = None

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult | Values:
        raise NotImplementedError

    @classmethod
    def check(cls, request: EvaluationRequest) -> None:
        """Check the request a candidate gives this evaluator, before anything of the candidate
        is created; ``cls.reject("...")`` rejects the candidate.

        The default rejects settings that do not match ``settings`` (:func:`check_settings`, an
        :class:`EvaluationContractError`: the problem's mistake, not the candidate's). Override
        it for rules beyond that, and call ``super().check(request)``. It is called with no
        instance, in the session's own process and without a time limit, so keep it to
        inexpensive checks of the request alone (its settings, collections, events, seed and
        sample properties): no files, geometry or physics, which belong in ``evaluate()``."""
        check_settings(cls, request.settings)

    @classmethod
    def reject(cls, message: str) -> NoReturn:
        """Reject the candidate (in ``check`` only): it cannot be evaluated, nothing of it runs,
        and an optimizer goes on with the next."""
        raise CandidateRejected(message)

    def events(self, request: EvaluationRequest) -> Iterator[Event]:
        """Each event of the file of ``stage``, with the declared ``collections`` as attributes
        (``event.hits``), all read from that one event; see
        :func:`~k4opt.evaluation.edm4hep.read_events`."""
        declared = declared_collections(self)
        if self.stage is None or not declared:
            raise EvaluationContractError(
                "events() needs stage = <stage> and collections = {role: Collection(type)}"
            )
        if self.stage not in request.inputs:
            raise EvaluationContractError(f"inputs[{self.stage!r}] is missing")
        names = collection_names(self, request.collections)
        wanted = {role: (names[role], declared[role].type) for role in declared}
        return read_events(request.inputs[self.stage], wanted)

    def invalid(self, message: str) -> NoReturn:
        """End the evaluation: it ran, but its result is not usable."""
        raise EvaluationInvalid(message)

    def fail(self, message: str) -> NoReturn:
        """End the evaluation: the execution failed."""
        raise EvaluationFailed(message)
