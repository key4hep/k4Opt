"""Internal: how the pipeline stages of a sample run, each at most once per candidate.

What each stage type is (the stages it reads, its executors, its outputs) is
:mod:`k4opt.problem.stages`; how one runs is here, one runner by stage type (``_RUNNERS``). The
framework makes the stage an evaluator reads of each sample it measures, and the stages it
depends on, available and hands the evaluator the files of all of them as
``request.inputs[stage]``. However many evaluators read a stage of a sample, it runs once. An
evaluator therefore never builds a command, starts a tool or handles that tool's failures.

* ``generation``: the ``gun`` executor has no file of its own (``ddsim`` fires it); a
  generation given as a file (an event file such as HepMC) is checked, hashed and handed on as
  ``inputs["generation"]``.
* ``simulation``: ``ddsim`` on the prepared geometry. Its output, run or given, is checked to
  hold its events (the requested number, when it ran), so no evaluator has to.
* ``reconstruction``: a Gaudi job (``k4run``) on the prepared geometry and every event of the
  simulation, from a copy of the job's directory. Its resolved options are checked to show that
  it read them, and its output, run or given, to hold every event it read.

Each stage of a candidate is one task (:mod:`k4opt.problem.plan`), whose work
:func:`make_stage` does in the task's own directory: a stage that fails blocks only the tasks
that read it. Each stage has named outputs (``events``); ``request.inputs[stage]`` is the main
one. The stages of a sample take their conditions from its :class:`StageContext`, never from the
problem itself: the stages as one candidate configures them (an option a parameter sets is in
them), or, before any candidate, as the problem writes them, with what parameters set pending.

A file the caller supplies as ``inputs[stage]`` (``k4opt evaluate --input``) is authoritative:
that stage is not run, the stages that only it needs are not run either, and the file replaces
what the problem says. Unlike a ``file`` in the problem, it is recorded as ``supplied``: a
supplied stage that the parameters change was not made from the candidate.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from k4opt import k4run
from k4opt.ddsim import (
    FILE_OPTIONS,
    check_arguments,
    check_particle_gun,
    output_collections,
    simulate,
    threads,
)
from k4opt.evaluation import EvaluationContractError, EvaluationFailed, count_events
from k4opt.execution._files import file_identity, readable_identity
from k4opt.frozen import Json, thaw
from k4opt.geometry import Catalog
from k4opt.problem.candidate import edges, stage_value
from k4opt.problem.models import Parameter, ProblemSpec, Stage, StageDestination
from k4opt.problem.stages import (
    executors,
    graph,
    plan_stages,
    reads_geometry,
    stage_output,
    stage_reads,
    stage_type,
    writes_files,
)
from k4opt.program import locate

_VARIABLE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)


def _variable_names(value: str) -> list[str]:
    return [m["braced"] or m["bare"] for m in _VARIABLE.finditer(value)]


def _path_texts(stage: Stage) -> list[str]:
    """The paths ``stage`` names, as written: its ``file``, a file a ddsim option names
    (``physics.pdgfile``), a reconstruction's ``directory``, ``options`` and ``data``."""
    raw: list[object] = [stage.file]
    if stage.executor == "ddsim":
        raw += [stage.options.get(key) for key in FILE_OPTIONS]
    elif stage.executor == "k4run":
        data = stage.options.get(k4run.DATA)
        raw += [
            stage.options.get(k4run.DIRECTORY),
            stage.options.get(k4run.OPTIONS),
            *(data.values() if isinstance(data, Mapping) else ()),
        ]
    texts = [v for item in raw for v in (item if isinstance(item, (list, tuple)) else [item])]
    return [text for text in texts if isinstance(text, str)]


def _variables_named(stage: Stage) -> list[str]:
    """The environment variables the paths of ``stage`` name: those of a reconstruction, the
    only paths read with them (:meth:`StageContext.path`)."""
    if stage.executor != "k4run":
        return []
    return [name for text in _path_texts(stage) for name in _variable_names(text)]


@dataclass(frozen=True, slots=True)
class StageContext:
    """The conditions the pipeline stages of one sample run under: its ``stages``.

    ``pending`` holds, by stage, the options that parameters set, which ``stages`` then lack:
    those of a problem before any candidate. A candidate's context has its stages complete
    (:func:`~k4opt.problem.candidate.configure`) and nothing pending; only such a context runs.

    ``directory`` is where a relative stage ``file`` (or a file named in an option, such as
    ``physics.pdgfile``) is read from: the directory of the problem file, the current one if it
    is ``None``. ``timeout`` is how many seconds a stage that runs a program (ddsim) may take
    before the program is killed and the stage fails; ``None`` for no limit. It is the stage
    task's (:attr:`~k4opt.execution._tasks.TaskSpec.timeout`), never the problem's.
    ``sample`` names the sample in messages and labels. ``variables`` holds the value of each
    environment variable a path of a stage names (a reconstruction's ``$CLDCONFIG``), and the
    home directory each ``~`` or ``~user`` a path starts with stands for (by that prefix), taken
    when the context was made, the session's opening: every candidate reads the files the session
    identified, wherever the variable or ``HOME`` points later. ``programs`` holds, by name, the
    executable each program a stage of it that runs (``ddsim``, ``k4run``) was found at then
    (:meth:`running`), where its links lead (:func:`k4opt.program.locate`), or ``None`` if none
    was found: every candidate runs that one, whatever ``PATH`` finds later, and a stage whose
    program was not found then fails (:meth:`unfound`), since its record could not say which
    program ran. A context made without a session (no ``programs``) looks for a program on
    ``PATH`` when it runs.

    ``started`` and ``die_with_parent`` say how the process that runs a stage watches the
    program it starts (:func:`~k4opt.ddsim.simulate`): a worker process reports the program's
    process group, and has it die with the worker. They belong to that process alone and are
    never part of a description sent elsewhere.
    """

    stages: Mapping[str, Stage] = field(default_factory=dict)
    pending: Mapping[str, frozenset[str]] = field(default_factory=dict)
    directory: Path | None = None
    timeout: float | None = None
    sample: str | None = None
    variables: Mapping[str, str] = field(default_factory=dict)
    programs: Mapping[str, str | None] = field(default_factory=dict)
    started: Callable[[int], None] | None = field(default=None, compare=False, repr=False)
    die_with_parent: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def of(cls, spec: ProblemSpec, sample: str, directory: Path | None = None) -> StageContext:
        """The stages of the sample ``sample`` of ``spec``, as it writes them, whose file is in
        ``directory``, with the environment variables their paths name as they are now."""
        stages = spec.samples[sample].stages
        named = {name for stage in stages.values() for name in _variables_named(stage)}
        variables = {name: os.environ[name] for name in sorted(named) if name in os.environ}
        for text in (text for stage in stages.values() for text in _path_texts(stage)):
            prefix = text.split("/", 1)[0]
            if prefix.startswith("~") and os.path.expanduser(prefix) != prefix:
                variables[prefix] = os.path.expanduser(prefix)
        return cls(stages, spec.pending(sample), directory, sample=sample, variables=variables)

    def running(self, stages: Collection[str]) -> StageContext:
        """The same conditions, with the programs the ``stages`` that run (of those it has) found
        on ``PATH`` now (:attr:`programs`): only those, so that a program no stage runs plays no
        part in what a session identifies."""
        named = sorted(
            {program for name in stages if (program := _program(self, name)) is not None}
        )
        programs = {name: None if (found := locate(name)) is None else str(found) for name in named}
        return replace(self, programs=programs)

    def executable(self, program: str) -> Path | None:
        """The executable of ``program`` the session found (:attr:`programs`), ``None`` if it
        found none."""
        found = self.programs.get(program)
        return None if found is None else Path(found)

    def unfound(self, program: str | None) -> str | None:
        """Why a stage that runs ``program`` cannot run: the session looked for it when it
        opened and found none; ``None`` otherwise."""
        if program is None or program not in self.programs or self.programs[program] is not None:
            return None
        return (
            f"{program} was not found on PATH when the session opened (is the Key4hep "
            "environment set up?); a session runs only the programs it found then"
        )

    def configured(self, stages: Mapping[str, Stage]) -> StageContext:
        """The same conditions with the ``stages`` of one candidate, complete."""
        return replace(self, stages=stages, pending={})

    def where(self, stage: str) -> str:
        """Where the problem defines ``stage``: ``samples.<sample>.<stage>``."""
        return stage if self.sample is None else f"samples.{self.sample}.{stage}"

    def path(self, value: str, variables: bool = False) -> Path:
        """The absolute path ``value`` stands for, a relative one taken from ``directory``; with
        ``variables``, ``$NAME`` and ``${NAME}`` stand for the environment variables of
        :attr:`variables` (``$CLDCONFIG/share/CLDConfig``), one it does not hold left as written.
        A leading ``~`` or ``~user`` is the home directory :attr:`variables` holds for it, taken
        when the session opened, never the one ``HOME`` says now."""
        if variables:
            value = _VARIABLE.sub(
                lambda m: self.variables.get(m["braced"] or m["bare"], m[0]), value
            )
        prefix, slash, rest = value.partition("/")
        if prefix.startswith("~") and prefix in self.variables:
            value = self.variables[prefix] + slash + rest
        path = Path(value)
        if path.is_absolute():
            return path
        return Path(os.path.abspath((self.directory or Path.cwd()) / path))

    def files(self) -> dict[str, Path]:
        """The absolute path of every stage the problem gives as a file, by stage."""
        return {
            name: self.path(stage.file)
            for name, stage in self.stages.items()
            if stage.file is not None
        }

    def conditions(self, name: str | None) -> tuple[int | None, int | None]:
        """``(events, seed)`` of the stage ``name``, ``None`` where it has none: those of the
        stage whose every event it processes, for an executor that takes none of its own (a
        reconstruction, of the simulation)."""
        stage = None if name is None else self.stages.get(name)
        if stage is None or name is None:
            return None, None
        if stage.executor is not None:
            executor = executors(name)[stage.executor]
            if not executor.events and executor.inside is None:
                return self.conditions(stage_reads(name)[0])
        return stage.events, stage.seed


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What a stage left behind: its named outputs, its files and how it ran.

    ``outputs`` holds what later stages and evaluators read (``events``); ``artifacts`` the other
    files (logs, the command), never an output. ``error`` is ``None`` only if the stage did what
    it should; otherwise it says what failed. ``peak_memory_bytes`` is that of the program it ran
    (:attr:`~k4opt.ddsim.SimulationRun.peak_memory_bytes`), ``None`` if it ran none or it is not
    known.
    """

    outputs: Mapping[str, Path] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    provenance: Mapping[str, Json] = field(default_factory=dict)
    error: str | None = None
    peak_memory_bytes: int | None = None


def provided_provenance(path: Path) -> dict[str, Json]:
    """What is recorded for an input the caller supplied instead of running its stage."""
    provenance: dict[str, Json] = {"provided": str(path)}
    # an unusable file is the evaluator's to report; the record still names it
    with contextlib.suppress(OSError):
        provenance |= file_identity(path)
    return provenance


def _primaries(context: StageContext, given: Path | None) -> tuple[Json, Path | None]:
    """``(particle_gun, input_file)`` for ddsim: a generation file, given or supplied, wins."""
    if given is not None:
        return None, given
    generation = context.stages.get("generation")
    return (None if generation is None else generation.options), None


def _check_generation(context: StageContext, provided: Collection[str]) -> None:
    stage = context.stages.get("generation")
    if stage is None:
        raise EvaluationContractError(
            f"{context.where('generation')}: the sample has no generation stage to say where the "
            "particles come from"
        )
    if stage.executor is not None:
        check_particle_gun(stage.options, context.pending.get("generation", ()))


def _run_generation(
    context: StageContext, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    # only a gun runs: a generation given as a file is not run
    return StageOutcome(provenance={"particle_gun": context.stages["generation"].options})


def _option_files(context: StageContext, options: Mapping[str, Json]) -> dict[str, Path]:
    """The files that ddsim options read, by option, as absolute paths."""
    return {
        key: context.path(value)
        for key, value in options.items()
        if key in FILE_OPTIONS and isinstance(value, str)
    }


def _check_simulation(context: StageContext, provided: Collection[str]) -> None:
    stage = context.stages.get("simulation")
    if stage is None or stage.executor is None:
        raise EvaluationContractError(
            f"{context.where('simulation')}: the sample has no simulation to run"
        )
    if stage.events is None:
        raise EvaluationContractError(
            f"{context.where('simulation')}.events: ddsim needs the events to simulate"
        )
    given = "generation" in provided or "generation" in context.files()
    particle_gun, input_file = _primaries(context, Path("generation") if given else None)
    check_arguments(
        events=stage.events,
        seed=stage.seed,
        particle_gun=particle_gun,
        input_file=input_file,
        options=stage.options,
        pending_gun=context.pending.get("generation", ()),
        pending_options=context.pending.get("simulation", ()),
    )
    for key, path in _option_files(context, stage.options).items():
        if not path.is_file():
            raise EvaluationContractError(
                f"{context.where('simulation')}.{key}: {path} is not a file"
            )


def _run_simulation(
    context: StageContext, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    stage = context.stages["simulation"]
    assert stage.events is not None  # checked by _check_simulation
    assert workdir is not None  # checked by check_run_inputs
    particle_gun, input_file = _primaries(context, inputs.get("generation"))
    files = _option_files(context, stage.options)
    read = {key: readable_identity(path) for key, path in files.items()}  # before ddsim does
    run = simulate(
        compact=inputs["geometry"],
        workdir=workdir,
        events=stage.events,
        seed=stage.seed,
        particle_gun=particle_gun,
        input_file=input_file,
        options={key: str(files[key]) if key in files else v for key, v in stage.options.items()},
        timeout=context.timeout,
        started=context.started,
        die_with_parent=context.die_with_parent,
        executable=context.executable("ddsim"),
    )
    # it ran in workdir, the stage's own directory, which the record names relative to the run
    provenance: dict[str, Json] = {
        "argv": run.argv,
        "returncode": run.returncode,
        "timeout": context.timeout,
    }
    error = run.error
    if files:
        recorded: dict[str, Json] = {}
        for key, path in files.items():
            identity = read[key]
            recorded[key] = {"path": str(path), **(identity or {})}
            if identity is not None and readable_identity(path) != identity:
                changed = f"{key}: {path} changed while ddsim ran; the record cannot say which"
                error = changed if error is None else f"{error}\n{changed}"
        provenance["files"] = recorded
    logs = {role: path for role, path in run.files.items() if role != "edm4hep"}
    peak = run.peak_memory_bytes
    if error is not None:
        return StageOutcome(
            artifacts=logs, provenance=provenance, error=error, peak_memory_bytes=peak
        )
    return StageOutcome({"events": run.output}, logs, provenance, peak_memory_bytes=peak)


def _inspect_simulation(
    context: StageContext, path: Path, inputs: Mapping[str, Path]
) -> tuple[dict[str, Json], str | None]:
    """``(observations, error)`` for the EDM4hep file of the simulation stage."""
    try:
        count = count_events(path)
    except EvaluationFailed as exc:
        return {}, str(exc)
    observed: dict[str, Json] = {"events_read": count}
    expected, _ = context.conditions("simulation")
    if expected is not None and count != expected:
        return observed, f"incomplete simulation: {count} of {expected} events in {path.name}"
    if count == 0:
        return observed, f"{path.name} contains no events"
    return observed, None


def _variable_hint(context: StageContext, value: str) -> str:
    """Why the path ``value`` may not exist: an environment variable in it that was not set when
    the session opened."""
    unset = [name for name in _variable_names(value) if name not in context.variables]
    if not unset:
        return ""
    return f" (${', $'.join(unset)} not set: is the Key4hep environment set up?)"


def _job(context: StageContext, stage: Stage) -> tuple[list[Path], Path | None]:
    """``(files, directory)``: the absolute options files of the reconstruction ``stage`` and the
    directory the job runs from a copy of, ``None`` if it has none."""
    raw = stage.options.get(k4run.DIRECTORY)
    directory = None if raw is None else context.path(str(raw), variables=True)
    names = k4run.options_files(stage.options)
    if directory is not None:
        return [directory / name for name in names], directory
    return [context.path(name, variables=True) for name in names], None


def _data(context: StageContext, stage: Stage) -> dict[str, str]:
    """The absolute path of each argument the ``data`` of the reconstruction ``stage`` names, by
    argument, a trailing ``/`` kept: a job may append file names to it (ALLEGRO's
    ``--dataFolder``)."""
    return {
        name: str(context.path(value, variables=True)) + ("/" if value.endswith("/") else "")
        for name, value in k4run.data_arguments(stage.options).items()
    }


def _job_files(context: StageContext, stage: Stage) -> dict[str, Path]:
    """The files the reconstruction ``stage`` runs and reads, by where it names them: each file of
    its ``directory`` (``directory/PandoraSettingsCLD/PandoraSettingsDefault.xml``), which the job
    runs a copy of, or its ``options`` files (``options``, ``options[1]``), and each file its
    ``data`` names, read where it is: a file (``data.calibration``) or each file of a directory
    (``data.dataFolder/capacitances_ecalBarrelFCCee_theta.root``)."""
    files, directory = _job(context, stage)
    found: dict[str, Path]
    if directory is not None:
        found = {f"directory/{name}": path for name, path in k4run.job_files(directory).items()}
    elif len(files) == 1 and not isinstance(stage.options.get(k4run.OPTIONS), (list, tuple)):
        found = {"options": files[0]}
    else:
        found = {f"options[{index}]": path for index, path in enumerate(files)}
    for name, value in _data(context, stage).items():
        path = Path(value)
        if path.is_dir():
            found |= {f"data.{name}/{inner}": file for inner, file in k4run.job_files(path).items()}
        else:
            found[f"data.{name}"] = path
    return found


def _check_reconstruction(context: StageContext, provided: Collection[str]) -> None:
    stage = context.stages.get("reconstruction")
    where = context.where("reconstruction")
    if stage is None or stage.executor is None:
        raise EvaluationContractError(f"{where}: the sample has no reconstruction to run")
    k4run.check_options(stage.options, context.pending.get("reconstruction", ()))
    files, directory = _job(context, stage)
    if directory is not None and not directory.is_dir():
        raw = str(stage.options[k4run.DIRECTORY])
        raise EvaluationContractError(
            f"{where}.directory: {directory} is not a directory{_variable_hint(context, raw)}"
        )
    for name, path in zip(k4run.options_files(stage.options), files, strict=True):
        if not path.is_file():
            within = f" in {directory}" if directory is not None else _variable_hint(context, name)
            raise EvaluationContractError(f"{where}.options: {path} is not a file{within}")
    written = k4run.data_arguments(stage.options)
    for name, value in _data(context, stage).items():
        if not Path(value).exists():
            hint = _variable_hint(context, written[name])
            raise EvaluationContractError(
                f"{where}.data.{name}: {value.rstrip('/')} is no file or directory{hint}"
            )
    try:  # a link to a directory it lies in is refused here, before anything runs
        _job_files(context, stage)
    except k4run.K4runError as exc:
        raise EvaluationContractError(f"{where}: {exc}") from exc


def _run_reconstruction(
    context: StageContext, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    stage = context.stages["reconstruction"]
    assert workdir is not None  # checked by check_run_inputs
    files, directory = _job(context, stage)
    read = {label: readable_identity(path) for label, path in _job_files(context, stage).items()}
    run = k4run.reconstruct(
        files=files,
        directory=directory,
        compact=inputs["geometry"],
        input_file=inputs["simulation"],
        workdir=workdir,
        options=stage.options,
        data=_data(context, stage),
        timeout=context.timeout,
        started=context.started,
        die_with_parent=context.die_with_parent,
        executable=context.executable("k4run"),
    )
    # it ran in workdir/job, a copy of the directory; the record names workdir relative to the run
    provenance: dict[str, Json] = {
        "argv": run.argv,
        "returncode": run.returncode,
        "timeout": context.timeout,
    }
    error = run.error
    recorded: dict[str, Json] = {}
    changed = []
    for label, path in _job_files(context, stage).items():
        identity = read.get(label)
        recorded[label] = {"path": str(path), **(identity or {})}
        if identity is None:
            changed.append(f"{label}: {path} cannot be read")
        elif readable_identity(path) != identity:
            changed.append(f"{label}: {path}")
    changed.extend(f"{label}: gone" for label in sorted(read.keys() - recorded.keys()))
    changed.extend(f"{label}: appeared" for label in sorted(recorded.keys() - read.keys()))
    if changed:
        note = (
            "the job's files changed or could not be read while k4run ran, so the record cannot "
            "say what it ran: " + ", ".join(changed)
        )
        error = note if error is None else f"{error}\n{note}"
    provenance["files"] = recorded
    logs = {role: path for role, path in run.files.items() if role != "edm4hep"}
    peak = run.peak_memory_bytes
    if error is not None:
        return StageOutcome(
            artifacts=logs, provenance=provenance, error=error, peak_memory_bytes=peak
        )
    return StageOutcome({"events": run.output}, logs, provenance, peak_memory_bytes=peak)


def _inspect_reconstruction(
    context: StageContext, path: Path, inputs: Mapping[str, Path]
) -> tuple[dict[str, Json], str | None]:
    """``(observations, error)`` for the EDM4hep file of the reconstruction stage: it holds every
    event of the simulation it read, if it read one here (not given as a file)."""
    try:
        count = count_events(path)
        read = None if "simulation" not in inputs else count_events(inputs["simulation"])
    except EvaluationFailed as exc:
        return {}, str(exc)
    observed: dict[str, Json] = {"events_read": count}
    if read is not None and count != read:
        return observed, (
            f"incomplete reconstruction: {count} of the {read} simulated events in {path.name}"
        )
    if count == 0:
        return observed, f"{path.name} contains no events"
    return observed, None


def _variants(
    context: StageContext, stage: Stage, parameters: Sequence[Parameter]
) -> list[tuple[str, Mapping[str, Json]]]:
    """The options of the reconstruction ``stage`` that a dry run tries, each with what it
    writes in, for messages: with the options that parameters set at the first end of each
    one's domain, then each other end of each, one at a time; as the problem writes them if no
    parameter sets one."""
    executor = executors("reconstruction")[stage.executor or "k4run"]
    set_by = {
        destination.option: parameter
        for parameter in parameters
        for destination in parameter.targets
        if isinstance(destination, StageDestination)
        and destination.stage == "reconstruction"
        and context.sample in destination.samples
    }
    values = {
        option: [stage_value(executor.settable(option), parameter, e) for e in edges(parameter)]
        for option, parameter in set_by.items()
    }
    first = {option: ends[0] for option, ends in values.items()}
    variants: list[tuple[str, Mapping[str, Json]]] = [("", {**stage.options, **first})]
    variants.extend(
        (
            f"{option} = {value!r}, as parameters.{set_by[option].name} sets it",
            {**stage.options, **first, option: value},
        )
        for option, ends in values.items()
        for value in ends[1:]
    )
    if first:
        shown = ", ".join(f"{option} = {value!r}" for option, value in first.items())
        variants[0] = (f"{shown}, as the parameters set them", variants[0][1])
    return variants


class DryRuns:
    """The dry runs of a session's reconstructions (:func:`k4opt.k4run.dry_run`) on the geometry
    ``compact`` the problem starts from: what each distinct job said, so that none is asked
    twice, and the scratch directory they run in, made when the first runs and holding the
    placeholder input they share. :meth:`close` removes it; a later dry run makes a new one."""

    def __init__(self, compact: Path) -> None:
        self.compact = compact
        self._said: dict[str, str | None] = {}
        self._scratch: Path | None = None

    def said(self, context: StageContext, options: Mapping[str, Json]) -> str | None:
        """Why the job of the reconstruction of ``context``'s sample, with the stage's
        ``options``, does not take its arguments; ``None`` if it does."""
        stage = replace(context.stages["reconstruction"], options=options)
        files, directory = _job(context, stage)
        key = json.dumps(
            [[str(file) for file in files], str(directory), thaw(options)], sort_keys=True
        )
        if (unfound := context.unfound("k4run")) is not None:
            return unfound
        if key not in self._said:
            if self._scratch is None:
                self._scratch = Path(tempfile.mkdtemp(prefix="k4opt-dry-run-"))
            workdir = self._scratch / f"dry-run-{len(self._said)}"
            self._said[key] = k4run.dry_run(
                files=files,
                directory=directory,
                compact=self.compact,
                workdir=workdir,
                options=options,
                data=_data(context, stage),
                placeholder=self._scratch / "placeholder.edm4hep.root",
                executable=context.executable("k4run"),
            )
            shutil.rmtree(workdir, ignore_errors=True)  # a copy of the job: only its answer counts
        return self._said[key]

    def close(self) -> None:
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None


def try_reconstruction(
    context: StageContext, parameters: Sequence[Parameter], runs: DryRuns
) -> None:
    """Raise :class:`EvaluationContractError` if the job of the reconstruction of ``context``'s
    sample does not take its arguments: a dry run (:class:`DryRuns`) for each of its variants
    (each end of the domain of a parameter that sets one of its options). The ends do not decide
    every value between them (an argument of a few choices: CLD's ``cms``), so each candidate is
    tried as well (:func:`try_candidate`)."""
    stage = context.stages["reconstruction"]
    for written, options in _variants(context, stage, parameters):
        reason = runs.said(context, options)
        if reason is not None:
            raise EvaluationContractError(
                f"{context.where('reconstruction')}: {reason}"
                + (f" (with {written})" if written else "")
            )


def try_candidate(context: StageContext, pending: Collection[str], runs: DryRuns) -> str | None:
    """Why the job of the reconstruction of ``context``'s sample, as one candidate configures
    it, does not take its arguments, ``None`` if it does or if no parameter sets one of them
    (``pending``, as the problem writes the stage): then the session tried it when it opened."""
    if not pending:
        return None
    reason = runs.said(context, context.stages["reconstruction"].options)
    if reason is None:
        return None
    shown = ", ".join(
        f"{option} = {context.stages['reconstruction'].options[option]!r}"
        for option in sorted(pending)
    )
    return f"{reason} (with {shown}, as the candidate sets them)"


def _simulation_cpus(stage: Stage) -> int:
    return threads(stage.options) if stage.executor == "ddsim" else 1


def _simulated_collections(stage: Stage, catalog: Catalog) -> list[str] | None:
    return sorted(output_collections(catalog.readouts)) if stage.executor == "ddsim" else None


@dataclass(frozen=True, slots=True)
class _Runner:
    """How a stage type runs: what checks that it can run in a context (``check``, given the
    stages the caller supplies), what runs it (``run``, given the files it reads and its
    directory), what checks its main output, whether it ran or was given (``inspect``, given the
    files the stage read, none if it was given), how many processors running a stage keeps busy
    (``cpus``), which collections it is expected to write for a geometry, if anything predicts
    them (``collections``), and which files the problem names in a stage that runs, by where
    (``files``: ``physics.pdgfile``, ``directory/...``)."""

    check: Callable[[StageContext, Collection[str]], None]
    run: Callable[[StageContext, Mapping[str, Path], Path | None], StageOutcome]
    inspect: (
        Callable[[StageContext, Path, Mapping[str, Path]], tuple[dict[str, Json], str | None]]
        | None
    ) = None
    cpus: Callable[[Stage], int] = lambda stage: 1
    collections: Callable[[Stage, Catalog], list[str] | None] = lambda stage, catalog: None
    files: Callable[[StageContext, Stage], dict[str, Path]] = lambda context, stage: {}


_RUNNERS: Mapping[str, _Runner] = {
    "generation": _Runner(_check_generation, _run_generation),
    "simulation": _Runner(
        _check_simulation,
        _run_simulation,
        _inspect_simulation,
        cpus=_simulation_cpus,
        collections=_simulated_collections,
        files=lambda context, stage: _option_files(context, stage.options),
    ),
    "reconstruction": _Runner(
        _check_reconstruction,
        _run_reconstruction,
        _inspect_reconstruction,
        files=_job_files,
    ),
}


def stage_cpus(name: str, stage: Stage) -> int:
    """How many processors running ``stage``, of the type ``name``, keeps busy: the threads of a
    simulation ddsim runs (``numberOfThreads``), one for anything else."""
    return _RUNNERS[name].cpus(stage)


def expected_collections(name: str, stage: Stage, catalog: Catalog) -> list[str] | None:
    """The collections ``stage``, of the type ``name``, is expected to write for the geometry
    ``catalog``, sorted: for ddsim those of its readouts; ``None`` if nothing predicts them."""
    return _RUNNERS[name].collections(stage, catalog)


def stage_logs(stage: str, directory: Path) -> dict[str, Path]:
    """The files besides its outputs (logs, the command) that ``stage`` left in its
    ``directory``, by role, as far as they exist: what a stage whose process died left."""
    files = {role: directory / name for role, name in stage_type(stage).logs.items()}
    return {role: path for role, path in files.items() if path.is_file()}


# podio reads EDM4hep written with ROOT (TTree or RNTuple) or SIO; other event files (HepMC, ...)
# are inputs of ddsim, not files an evaluator can read collections from
_PODIO_SUFFIXES = (".root", ".sio")


def no_edm4hep_file(
    context: StageContext, stage: str, supplied: Mapping[str, Path] | None = None
) -> str | None:
    """Why ``stage`` makes no EDM4hep file that an evaluator could read collections from,
    ``None`` if it does. A file the caller supplies or the problem gives must be one podio reads
    (``.root``, ``.sio``), not an event file such as HepMC; an executor that runs inside another
    stage (the gun, fired by ddsim) writes no file of its own."""
    path = (supplied or {}).get(stage) or context.files().get(stage)
    if path is not None:
        if path.suffix in _PODIO_SUFFIXES:
            return None
        readable = " or ".join(_PODIO_SUFFIXES)
        return (
            f"its file {path.name} is no EDM4hep file that podio reads ({readable}); read the "
            f"collections from a stage that writes one, such as {context.where('simulation')}"
        )
    spec = context.stages.get(stage)
    if spec is None or spec.executor is None:
        return None  # not defined: reported elsewhere
    inside = executors(stage)[spec.executor].inside
    if inside is None:
        return None
    return (
        f"executor {spec.executor!r} runs inside {context.where(inside)} and writes no file of "
        f"its own; read the collections from {context.where(inside)}, or give "
        f"{context.where(stage)} as a file"
    )


def check_stages(context: StageContext, *targets: str, provided: Collection[str] = ()) -> list[str]:
    """Raise if a stage of ``targets`` or one they need is unknown or cannot run in
    ``context``; return the plan.

    ``provided`` names the stages the caller supplies. A stage the problem gives as a file must
    exist. An option in ``context.pending`` is not missing: before any candidate, the stages are
    checked as they will be once a candidate gives it. Needs no geometry and starts nothing, so
    it can run before any file is prepared.
    """
    files = {name: path for name, path in context.files().items() if name not in provided}
    given = {*provided, *files}
    plan = plan_stages(*targets, provided=given)
    for name in plan:
        _RUNNERS[name].check(context, given)
    for name, _ in graph(targets, given):
        if name in files and not files[name].is_file():
            raise EvaluationContractError(
                f"{context.where(name)}.file: {files[name]} is not a file"
            )
    return plan


def input_files(
    context: StageContext, *targets: str, provided: Collection[str] = ()
) -> dict[str, Path]:
    """Every file the problem names that the stages of ``targets`` read, by where it is named.

    ``samples.<sample>.<stage>.file`` for a stage given as a file,
    ``samples.<sample>.<stage>.<option>`` for a file an executor option names
    (``samples.gamma.simulation.physics.pdgfile``). The stages in ``provided``, and those only
    they need, read none: the caller supplies their files.
    """
    files = {name: path for name, path in context.files().items() if name not in provided}
    named: dict[str, Path] = {}
    for name, _ in graph(targets, {*provided, *files}):
        stage = context.stages.get(name)  # each one is defined once check_stages() passed
        if name in files:
            named[f"{context.where(name)}.file"] = files[name]
        elif name not in provided and stage is not None and stage.executor is not None:
            options = _RUNNERS[name].files(context, stage)
            named |= {f"{context.where(name)}.{key}": path for key, path in options.items()}
    return named


def check_run_inputs(run: Collection[str], geometry: bool, workdir: Path | None) -> None:
    """Raise if a stage that is to ``run`` needs a geometry and there is none, or a directory and
    there is no ``workdir`` to make it in; stages given as files need neither."""
    for name in run:
        if reads_geometry(name) and not geometry:
            raise EvaluationContractError(f"the {name!r} stage needs inputs['geometry']")
        if writes_files(name) and workdir is None:
            raise EvaluationContractError(f"the {name!r} stage needs a workdir")


def run_stage(
    context: StageContext, stage: str, *, inputs: Mapping[str, Path], workdir: Path | None
) -> StageOutcome:
    """Run one ``stage`` in its own new directory ``workdir``; ``inputs`` holds the geometry and
    the main output of each stage it reads, by name."""
    return _RUNNERS[stage].run(context, inputs, workdir)


def make_stage(
    context: StageContext,
    stage: str,
    *,
    inputs: Mapping[str, Path],
    workdir: Path | None,
    supplied: Path | None = None,
) -> tuple[StageOutcome, Path | None]:
    """Make ``stage`` available, the work of its task: ``(outcome, main output)``.

    A file the caller ``supplied`` stands in for the stage, then a ``file`` the problem gives
    (hashed: its size and SHA-256 are recorded); otherwise the stage runs (:func:`run_stage`).
    The main output, run or given, is then checked to hold what the stage promises (the requested
    events of a simulation). ``outcome.error`` says why the stage cannot be read; the main output
    is ``None`` then, and for a stage that gives no file (a particle gun).

    ``context`` must already have passed :func:`check_stages`; the preflight of a candidate does
    that before any file is prepared. It is not checked again here: a missing file is then a
    failed stage, not a contract error.
    """
    output, inspect = stage_output(stage), _RUNNERS[stage].inspect
    path = context.files().get(stage)
    if supplied is not None:
        outcome = StageOutcome({output: supplied}, provenance=provided_provenance(supplied))
    elif path is not None:
        try:
            outcome = StageOutcome(
                {output: path}, provenance={"file": str(path)} | file_identity(path)
            )
        except OSError as exc:
            outcome = StageOutcome(
                provenance={"file": str(path)}, error=f"cannot read {path}: {exc}"
            )
    elif (unfound := context.unfound(_program(context, stage))) is not None:
        outcome = StageOutcome(error=unfound)  # not looked for again: no record would name it
    else:
        outcome = run_stage(context, stage, inputs=inputs, workdir=workdir)
    main = outcome.outputs.get(output)
    if outcome.error is not None or main is None or inspect is None:
        return outcome, None if outcome.error is not None else main
    observed, error = inspect(context, main, inputs if path is None and supplied is None else {})
    provenance = {**outcome.provenance, **observed}
    outcome = replace(outcome, provenance=provenance, error=error)
    return outcome, None if error is not None else main


def _program(context: StageContext, stage: str) -> str | None:
    """The program of the stack the stage ``stage`` of ``context`` runs (``ddsim``), if any."""
    found = context.stages.get(stage)
    if found is None or found.executor is None:
        return None
    return executors(stage)[found.executor].program
