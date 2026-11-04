"""Internal: the ``samples`` of a problem document and the ``stages`` they share, parsed and
checked against each other and against the evaluators that measure them.

A stage is read in two phases. :func:`_samples` reads each one on its own, as the document
writes it; :func:`_check_options` checks its options as its executor would run them, once the
parameters are known: an option a parameter sets is not written in the stage but given by
every candidate, so it is pending, and the executor checks what the stage writes around it and
the value at each end of the parameter's domain written in (never stored)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from k4opt.ddsim import DdsimError
from k4opt.frozen import Json
from k4opt.problem._fields import (
    _choice,
    _fail,
    _integer,
    _mapping,
    _seconds,
    _show,
    _text,
    _title,
)
from k4opt.problem.candidate import edges, stage_value
from k4opt.problem.errors import ProblemConfigError
from k4opt.problem.models import (
    EvaluatorSpec,
    Parameter,
    Sample,
    Stage,
    StageDestination,
    freeze,
)
from k4opt.problem.plan import TaskId, stage_task, varied
from k4opt.problem.stages import STAGE_TYPES, executors, stage_reads

# the rest of a stage are executor options
_STAGE_KEYS = ("executor", "file", "events", "seed", "timeout")


# lowercase, so that no two names differ only in case (run directories are named after them)
_SAMPLE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


_ABSENT = object()  # a key the document leaves out, unlike one written as null


def _shared_stages(raw: object) -> dict[str, Mapping[str, Any]]:
    """The top-level ``stages``: by stage type, what that stage of every sample starts from.

    Each names how the stage is made (``executor`` or ``file``); whether it is complete is
    decided for each sample that takes it, once the sample's own keys are added."""
    m = _mapping(raw, "stages", STAGE_TYPES)
    if not m:
        raise _fail("stages", f"must name at least one stage: {', '.join(STAGE_TYPES)}")
    shared: dict[str, Mapping[str, Any]] = {}
    for name in STAGE_TYPES:
        if name not in m:
            continue
        path = f"stages.{name}"
        entry = _mapping(m[name], path, tuple(m[name]) if isinstance(m[name], Mapping) else ())
        if ("executor" in entry) == ("file" in entry):
            raise _fail(
                path,
                "needs exactly one of 'executor' and 'file': a shared stage says how the stage is "
                "made, and the samples add what differs",
            )
        others = [key for key in entry if key != "file"]
        if "file" in entry and others:
            raise _fail(path, f"a stage given as a file is not run, so it takes no {others}")
        shared[name] = entry
    return shared


@dataclass(frozen=True, slots=True)
class _Place:
    """Where the stage ``stage`` of the sample ``sample`` was written: in the sample (``own``),
    in the shared ``stages`` (the keys in ``shared``), or completed from both."""

    sample: str
    stage: str
    own: bool = True
    shared: frozenset[str] = frozenset()

    @property
    def path(self) -> str:
        """The stage, for messages: ``samples.zh.simulation (completing stages.simulation)``."""
        path = f"samples.{self.sample}.{self.stage}"
        if not self.shared:
            return path
        return f"{path} ({'completing' if self.own else 'from'} stages.{self.stage})"

    def key(self, key: str) -> str:
        """Where the key ``key`` of the stage was written."""
        if key in self.shared:
            return f"stages.{self.stage}.{key}"
        return f"samples.{self.sample}.{self.stage}.{key}"


def _samples(
    raw: object, shared: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Sample], dict[tuple[str, str], _Place]]:
    """The ``samples`` mapping in the document's order, each sample's stages in pipeline order
    whatever the order of the document, and where each stage was written.

    A sample takes each ``shared`` stage it does not define, and completes one it defines without
    ``executor`` or ``file``: its keys win, and a nested value replaces the shared one whole. A
    stage it defines with ``executor`` or ``file`` replaces the shared one."""
    m = _mapping(raw, "samples", tuple(raw) if isinstance(raw, Mapping) else ())
    if not m:
        raise _fail("samples", "must name at least one sample")
    samples: dict[str, Sample] = {}
    places: dict[tuple[str, str], _Place] = {}
    taken: set[str] = set()  # the shared stages some sample takes or completes
    for name, value in m.items():
        if not _SAMPLE_NAME.fullmatch(name):
            raise _fail(
                "samples",
                f"sample name {_show(name)} must start with a lowercase letter and hold only "
                "lowercase letters, digits and '_', at most 64 characters",
            )
        where = f"samples.{name}"
        stages = _mapping(value, where, ("title", *STAGE_TYPES, "properties"))
        made: dict[str, Stage] = {}
        for stage in STAGE_TYPES:
            if stage not in stages and stage not in shared:
                continue
            own = stages.get(stage, _ABSENT)  # null is written, not absent
            entry, place = _completed(name, stage, own, shared.get(stage))
            if place.shared:
                taken.add(stage)
            made[stage] = _stage(where, place, entry)
            places[name, stage] = place
        if not made:
            raise _fail(where, f"must define its stages: {', '.join(STAGE_TYPES)}")
        properties: Mapping[str, Json] = {}
        if "properties" in stages:
            at, raw = f"{where}.properties", stages["properties"]
            # any JSON, which evaluators read and k4opt does not: a mapping with string keys
            frozen = freeze(_mapping(raw, at, tuple(raw) if isinstance(raw, Mapping) else ()), at)
            assert isinstance(frozen, Mapping)
            properties = frozen
        samples[name] = Sample(name, made, properties, _title(stages, where))
    for stage in shared:
        if stage not in taken:
            raise _fail(
                f"stages.{stage}",
                "is not used: every sample replaces it, naming executor: or file: of its own; "
                "remove it, or leave those keys out of the samples that should complete it",
            )
    return samples, places


def _completed(
    sample: str, stage: str, own: object, shared: Mapping[str, Any] | None
) -> tuple[Mapping[str, Any], _Place]:
    """The stage ``stage`` of the sample ``sample``: its ``own`` entry (``_ABSENT`` if it defines
    none), the ``shared`` one, or the shared one completed by its own, and where it came from."""
    if own is _ABSENT:
        assert shared is not None
        return shared, _Place(sample, stage, own=False, shared=frozenset(shared))
    path = f"samples.{sample}.{stage}"
    m = _mapping(own, path, tuple(own) if isinstance(own, Mapping) else ())
    if "executor" in m or "file" in m or shared is None:
        return m, _Place(sample, stage)
    if "file" in shared:
        raise _fail(
            path,
            f"stages.{stage} gives a file, which takes no other keys; define the whole stage here, "
            "with executor: or file:",
        )
    taken = frozenset(key for key in shared if key not in m)
    return {**shared, **m}, _Place(sample, stage, shared=taken)


def _stage(where: str, place: _Place, m: Mapping[str, Any]) -> Stage:
    """The stage at ``place`` of the sample at ``where`` (``samples.gamma``), from its keys
    ``m``: the sample's own, the shared ones, or both."""
    name, path = place.stage, place.path
    if ("executor" in m) == ("file" in m):
        raise _fail(
            path,
            "needs exactly one of 'executor' (k4opt runs it) and 'file' (its output exists), or a "
            f"stages.{name} for it to complete",
        )
    if "file" in m:
        others = [key for key in m if key != "file"]
        if others:
            raise _fail(path, f"a stage given as a file is not run, so it takes no {others}")
        return Stage(name, file=_text(m["file"], place.key("file")))
    allowed = executors(name)
    executor = _choice(m["executor"], place.key("executor"), tuple(allowed))
    spec = allowed[executor]
    events = seed = None
    if spec.events:
        if "events" not in m:
            raise _fail(path, f"missing required key 'events': how many events {executor} runs")
        events = _integer(m["events"], place.key("events"), minimum=1)
        seed = _integer(m["seed"], place.key("seed")) if "seed" in m else None
    else:
        for key in ("events", "seed"):
            if key in m:
                inside, instead = f"{where}.{spec.inside}", f"{where}.{spec.inside}.{key}"
                if key in place.shared:
                    inside, instead = f"each sample's {spec.inside}", f"stages.{spec.inside}.{key}"
                raise _fail(
                    place.key(key),
                    f"{executor} runs inside {inside}, which sets the {key}; set {instead} instead",
                )
    timeout = _seconds(m["timeout"], place.key("timeout")) if "timeout" in m else None
    raw = {key: value for key, value in m.items() if key not in _STAGE_KEYS}
    try:
        options = freeze(raw, path)
    except ProblemConfigError as exc:  # its message starts with the stage: samples.g.simulation...
        raise ProblemConfigError(_relocated(str(exc), place, raw)) from exc
    assert isinstance(options, Mapping)
    return Stage(
        name, executor=executor, events=events, seed=seed, options=options, timeout=timeout
    )


def _relocated(message: str, place: _Place, options: Mapping[str, object]) -> str:
    """A ``message`` about an option of the stage at ``place`` that names the whole stage
    (``samples.g.simulation (from stages.simulation).physics.list: ...``), pointing at where
    the option was written."""
    for key in sorted(options, key=len, reverse=True):  # options may hold dots: physics.list
        prefix = f"{place.path}.{key}"
        if message.startswith(prefix) and message[len(prefix) : len(prefix) + 1] in (".", "[", ":"):
            return f"{place.key(key)}{message[len(prefix) :]}"
    return message


def _located(message: str, place: _Place, options: Mapping[str, Json]) -> str:
    """A ddsim ``message`` about the stage at ``place`` (``simulation.physics.list: ...``),
    pointing at where the option it names was written."""
    head, _, rest = message.partition(": ")
    if head == place.stage:
        return f"{place.path}: {rest}"
    for key in sorted(options, key=len, reverse=True):  # options may hold dots: physics.list
        prefix = f"{place.stage}.{key}"
        if head == prefix or head.startswith(f"{prefix}."):
            return f"{place.key(key)}{head[len(prefix) :]}: {rest}"
    return f"samples.{place.sample}.{message}"


def _check_options(
    samples: Mapping[str, Sample],
    places: Mapping[tuple[str, str], _Place],
    parameters: tuple[Parameter, ...],
) -> None:
    """The options of each stage that runs, as its executor checks them: as the problem writes
    them, with the options parameters set pending, then with each end of the domain of each such
    parameter written in. Every settable option's check decides a value by an interval or a set
    (:class:`~k4opt.ddsim.Settable`), so the ends decide every value between them."""
    for name, sample in samples.items():
        for stage_name, stage in sample.stages.items():
            if stage.executor is None:
                continue
            place, executor = places[name, stage_name], executors(stage_name)[stage.executor]
            set_by = {
                destination.option: parameter
                for parameter in parameters
                for destination in parameter.targets
                if isinstance(destination, StageDestination)
                and destination.stage == stage_name
                and name in destination.samples
            }
            pending = frozenset(set_by)
            try:
                executor.options(stage.options, pending)
            except DdsimError as exc:  # its message starts with the stage: generation.particle
                raise ProblemConfigError(_located(str(exc), place, stage.options)) from exc
            for option, parameter in set_by.items():
                settable = executor.settable(option)
                for edge in edges(parameter):
                    written = {**stage.options, option: stage_value(settable, parameter, edge)}
                    try:
                        executor.options(written, pending - {option})
                    except DdsimError as exc:
                        end = "bound" if parameter.bounds is not None else "choice"
                        raise ProblemConfigError(
                            f"parameters.{parameter.name}: at its {end} {edge!r}, "
                            f"{_located(str(exc), place, written)}"
                        ) from exc


def _check_pipeline(
    samples: Mapping[str, Sample],
    places: Mapping[tuple[str, str], _Place],
    evaluators: tuple[EvaluatorSpec, ...],
    parameters: tuple[Parameter, ...],
) -> None:
    """Each sample's stages fit together and fit the evaluators that measure it.

    Every stage an evaluator reads is defined in each sample it measures; a stage that runs has
    the stages it reads; no file stands in for a stage a parameter reaches (it was not made
    from the candidate, :func:`~k4opt.problem.plan.varied`); every sample is measured by an
    evaluator, and every stage of a sample is used, by an evaluator that measures it or by a
    stage that runs.
    """
    for index, evaluator in enumerate(evaluators):
        if evaluator.collections and evaluator.stage is None:
            raise _fail(
                f"evaluators[{index}].collections",
                "collections are read from the file of the stage the evaluator reads; add its "
                "stage",
            )
        for sample in evaluator.samples:
            if evaluator.stage not in samples[sample].stages:
                raise _fail(
                    f"evaluators[{index}].stage",
                    f"sample {sample} defines no {evaluator.stage} stage; add "
                    f"samples.{sample}.{evaluator.stage}",
                )
    read: dict[str, list[str]] = {name: [] for name in samples}  # the stages roles read, by sample
    for index, evaluator in enumerate(evaluators):
        for role, items in evaluator.roles.items():
            for item in items:
                if item.stage is None:
                    continue
                stages = samples[item.sample].stages
                where = f"evaluators[{index}].roles.{role}"
                if item.stage not in stages:
                    raise _fail(
                        where,
                        f"sample {item.sample} defines no {item.stage} stage; add "
                        f"samples.{item.sample}.{item.stage}",
                    )
                executor = stages[item.stage].executor
                inside = None if executor is None else executors(item.stage)[executor].inside
                if inside is not None:
                    raise _fail(
                        where,
                        f"{item.label} is fired inside samples.{item.sample}.{inside} and makes "
                        f"no file of its own; read samples.{item.sample}.{inside} instead",
                    )
                read[item.sample].append(item.stage)
    reach = varied(samples, evaluators, parameters)
    for spec in samples.values():
        measuring = [e for e in evaluators if spec.name in e.samples]
        places_of = {stage: places[spec.name, stage] for stage in spec.stages}
        _check_sample(spec, places_of, measuring, read[spec.name], reach)


def _check_sample(
    sample: Sample,
    places: Mapping[str, _Place],
    evaluators: list[EvaluatorSpec],
    read: list[str],
    reach: Mapping[TaskId, tuple[str, ...]],
) -> None:
    """The stages of one sample fit together and fit the ``evaluators`` that measure it and the
    stages of it that roles ``read``; ``places`` says where each stage was written, ``reach``
    which parameters reach each task."""
    where, stages = f"samples.{sample.name}", sample.stages
    for name, stage in stages.items():
        if stage.executor is None:
            continue
        for need in stage_reads(name):
            if need not in stages:
                choices = " or ".join(executors(need))
                raise _fail(
                    places[name].path,
                    f"{name} reads {need}: add {where}.{need} with executor: {choices}, or give "
                    "its output as file:",
                )
    for name, stage in stages.items():
        varies = reach.get(stage_task(name, sample.name), ())
        if stage.file is not None and varies:
            raise _fail(
                places[name].key("file"),
                f"{', '.join(f'parameters.{p}' for p in varies)} reach this stage, but this file "
                "was not made from a candidate's values, so every candidate would measure the same "
                f"events; let an executor run the stage (to try an evaluator on an existing file, "
                f"use k4opt evaluate --input {sample.name}/{name}=FILE)",
            )
    if not evaluators and not read:
        raise _fail(
            where,
            "is measured by no evaluator; list it in the samples of one, have a role read one of "
            "its stages, or remove it",
        )
    used: set[str] = set()

    def use(name: str) -> None:
        if name not in used:
            used.add(name)
            if stages[name].executor is not None:
                for need in stage_reads(name):
                    use(need)

    for evaluator in evaluators:
        assert evaluator.stage is not None  # an evaluator without a stage measures no sample
        use(evaluator.stage)
    for name in read:
        use(name)
    for name in stages:
        if name not in used:
            hint = ""
            if not places[name].own:
                hint = (
                    f"; every sample takes stages.{name}, so define {name} in the samples that "
                    "use it instead"
                )
            raise _fail(
                places[name].path,
                "is not used: no evaluator of the sample reads it, no role, and no stage that "
                f"runs{hint}",
            )
