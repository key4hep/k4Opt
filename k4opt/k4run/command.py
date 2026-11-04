"""Translation of the options of a reconstruction stage into a ``k4run`` command line.

Verified against the ``k4run`` of the Key4hep nightly 2026-10-20 (k4FWCore's ``bin/k4run`` and
``k4FWCore/ApplicationMgr.py``) and the reconstructions of CLD (CLDConfig's
``CLDReconstruction.py``), ALLEGRO and IDEA (FCC-config's ``run_digi_reco.py``) and ILD
(ILDConfig's ``ILDReconstruction.py``) as k4Bench runs them:

* A job is one or more Gaudi options files: Python, which reads its own arguments
  (``--cms 91``, ``--truthTracking``) with ``parse_known_args``. ``k4run`` then adds an argument
  ``--<component>.<property>`` *and* ``--<property>.<component>`` for every property of every
  component the files configured, and parses the whole command line strictly: an argument that
  neither the job nor a property takes is an error. The properties given on the command line are
  set after the files ran, so they win over what the job set.
* So no table of k4opt says which arguments a job takes, or of what kind: an option of the stage
  is passed on as ``--<name>=<value>`` (a list as ``--<name> v1 v2``, ``true`` as the bare flag
  ``--<name>``), and the job's ``--dry-run`` is what decides that the job takes it
  (:func:`k4opt.k4run.dry_run`). ``false`` is refused: a flag of the job's own (CLD's
  ``--truthTracking``, an argparse ``store_true``) takes no value and cannot be switched off,
  and leaving it out means the job's default, which k4opt does not know; a Gaudi property, which
  does take one, is given the text ``'False'``.
* k4opt gives the job the candidate's geometry, the simulation's events and the file to write
  through k4FWCore's services (``GeoSvc.detectors``, ``IOSvc.Input``, ``IOSvc.Output``), or, for
  the geometry and the input, through the job's own argument the stage names (``geometry:
  compactFile`` for CLD and ILD, whose reconstruction derives more than the geometry service
  from it). The job's resolved options then must show that it read them
  (:func:`k4opt.k4run.verify`).
* A job may set its own number of events (``CLDReconstruction.py`` sets ``EvtMax = 3``), so
  k4opt always passes ``--num-events=-1``: every event of the input.

Everything here is pure: nothing is run or read.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path, PurePosixPath

from k4opt.frozen import Json
from k4opt.k4run.errors import K4runError
from k4opt.program import Settable

STAGE = "reconstruction"
"""The stage type k4run runs; the messages here name the options by it
(``reconstruction.cms: ...``)."""

OPTIONS, DIRECTORY, GEOMETRY, INPUT = "options", "directory", "geometry", "input"
EXECUTOR_KEYS = (OPTIONS, DIRECTORY, GEOMETRY, INPUT)
"""The options of a stage k4opt reads itself: the job's options files, the directory it runs from
a copy of, and the job's own arguments that take the geometry and the input. Every other option is
an argument of the job."""

_EVERY_EVENT = "k4opt reconstructs every event of the simulation it reads"
_BATCH = "k4opt runs the job in batch, to its end"

# the arguments of k4run itself that k4opt sets or that would not run the job to its end
_PROGRAM: Mapping[str, str] = {
    "num-events": _EVERY_EVENT,
    "n": _EVERY_EVENT,
    "evt-max": _EVERY_EVENT,
    "dry-run": _BATCH,
    "list": _BATCH,
    "l": _BATCH,
    "gdb": _BATCH,
    "interactive": _BATCH,
    "i": _BATCH,
    "interactive-root": _BATCH,
    "help": _BATCH,
    "h": _BATCH,
}

# the properties k4opt sets, by (component, property)
_MANAGED: Mapping[tuple[str, str], str] = {
    ("GeoSvc", "detectors"): (
        "the geometry is the candidate's; if the job takes the compact file as an argument of its "
        "own, name it with geometry:"
    ),
    ("IOSvc", "Input"): (
        "the input is the events of the simulation; if the job takes its input files as an "
        "argument of its own, name it with input:"
    ),
    ("IOSvc", "Output"): "k4opt chooses the output file",
    ("ApplicationMgr", "EvtMax"): _EVERY_EVENT,
}

# the components whose every property k4opt manages
_MANAGED_COMPONENTS: Mapping[str, str] = {
    "JobOptionsSvc": "k4opt reads the job's resolved options from it",
}

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_:\-]*(\.[A-Za-z_][A-Za-z0-9_:\-]*)*")


def _fail(key: str, message: str) -> K4runError:
    return K4runError(f"{STAGE}.{key}: {message}" if key else f"{STAGE}: {message}")


def _managed(name: str) -> str | None:
    """Why k4opt manages the argument ``--<name>``, ``None`` if it does not: k4run's own, or a
    property k4opt sets, in either of k4run's spellings (``GeoSvc.detectors``,
    ``detectors.GeoSvc``)."""
    if name in _PROGRAM:
        return _PROGRAM[name]
    if "." not in name:
        return None
    owner, _, prop = name.rpartition(".")  # --<component>.<property>; a tool's holds dots
    first, _, rest = name.partition(".")  # --<property>.<component>
    for component, key in ((owner, prop), (rest, first)):
        if (component, key) in _MANAGED:
            return _MANAGED[component, key]
        if component in _MANAGED_COMPONENTS:
            return _MANAGED_COMPONENTS[component]
    return None


def _check_name(key: object, path: str) -> str:
    """``key`` as the name of an argument of the job: without its dashes, as ``--help`` lists it
    (``cms``, ``GeoSvc.OutputLevel``), and not one k4opt manages."""
    if not isinstance(key, str) or not _NAME.fullmatch(key):
        hint = ""
        if isinstance(key, str) and key.startswith("-"):
            hint = f"; write it without the leading dashes: {key.lstrip('-')!r}"
        raise _fail(path, f"is no k4run argument name, as k4run --help lists them{hint}")
    reason = _managed(key)
    if reason is not None:
        raise _fail(path, f"k4opt sets this argument itself: {reason}")
    return key


def _scalar(value: Json, path: str) -> str:
    """One value of an argument as the job's command line reads it."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(path, f"must be a finite number, got {value!r}")
        return repr(value)
    if isinstance(value, str):
        return value
    raise _fail(path, "must be a number, a text, a boolean or a list of them")


def _argument(key: str, value: Json) -> tuple[str, ...]:
    """The arguments that pass ``value`` to ``--<key>``."""
    if value is True:
        return (f"--{key}",)
    if value is False:
        raise _fail(
            key,
            "false cannot be passed: a flag of the job's own (--truthTracking) takes no value and "
            "cannot be switched off, so leave the option out to keep the job's default; a Gaudi "
            "property takes the text 'False'",
        )
    if isinstance(value, (list, tuple)):
        if not value:
            raise _fail(key, "an empty list passes nothing; leave the option out")
        values = [_scalar(item, f"{key}[{index}]") for index, item in enumerate(value)]
        for index, text in enumerate(values):
            if text.startswith("-"):
                raise _fail(
                    f"{key}[{index}]",
                    "a value of a list must not start with '-': k4run would read it as an argument",
                )
        return (f"--{key}", *values)
    if isinstance(value, Mapping) or value is None:
        raise _fail(key, "must be a number, a text, a boolean or a list of them")
    return (f"--{key}={_scalar(value, key)}",)


def job_arguments(options: Mapping[str, Json]) -> tuple[str, ...]:
    """The arguments of the job that the options of the stage other than :data:`EXECUTOR_KEYS`
    pass, in their order; raises :class:`K4runError` for one k4opt manages or cannot pass."""
    arguments: list[str] = []
    for key, value in options.items():
        if key in EXECUTOR_KEYS:
            continue
        arguments += _argument(_check_name(key, key), value)
    return tuple(arguments)


def _text(value: Json, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, "must be a nonblank text")
    return value


def options_files(options: Mapping[str, Json]) -> tuple[str, ...]:
    """The job's options files as the stage names them, in the order k4run loads them: ``options``
    is one file or a list of them. Relative to ``directory`` if the stage has one (then they must
    lie inside it), otherwise to the problem file."""
    if OPTIONS not in options:
        raise _fail(
            "",
            "missing required key 'options': the Gaudi options file(s) of the job, such as "
            "CLDReconstruction.py",
        )
    raw = options[OPTIONS]
    files = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if not files:
        raise _fail(OPTIONS, "must name at least one options file")
    named = [
        _text(file, OPTIONS if not isinstance(raw, (list, tuple)) else f"{OPTIONS}[{index}]")
        for index, file in enumerate(files)
    ]
    if len(set(named)) != len(named):
        raise _fail(OPTIONS, "names a file twice")
    if DIRECTORY in options:
        for index, name in enumerate(named):
            parts = PurePosixPath(name).parts
            if Path(name).is_absolute() or ".." in parts or name.startswith(("~", "$")):
                where = OPTIONS if not isinstance(raw, (list, tuple)) else f"{OPTIONS}[{index}]"
                raise _fail(
                    where,
                    f"{name!r}: with directory:, an options file is named relative to it and lies "
                    "inside it, since the job runs from a copy of it",
                )
    return tuple(named)


def wiring(options: Mapping[str, Json]) -> tuple[str | None, str | None]:
    """``(geometry, input)``: the job's own arguments that take the compact file and the input
    files, ``None`` for those k4opt gives through ``GeoSvc.detectors`` and ``IOSvc.Input``."""
    named: list[str | None] = []
    for key in (GEOMETRY, INPUT):
        if key not in options:
            named.append(None)
            continue
        name = _check_name(_text(options[key], key), key)
        if name in options:
            raise _fail(name, f"is the argument {key}: names, which k4opt sets; leave it out")
        named.append(name)
    geometry, input_ = named
    if geometry is not None and geometry == input_:
        raise _fail(INPUT, f"names {input_!r}, the argument geometry: names too")
    return geometry, input_


def settable_option(key: str) -> Settable:
    """How a parameter sets the job argument ``key``: as its value itself, whose unit the job
    decides; :class:`K4runError` for one it cannot set: what k4opt reads itself (the options
    files, the directory, the arguments that take the geometry and the input) and what k4opt
    manages."""
    if key in EXECUTOR_KEYS:
        raise _fail(
            key,
            "a parameter cannot set it: it says which job runs, which k4opt identifies once per "
            "session, not per candidate",
        )
    _check_name(key, key)
    return Settable("value")


def check_options(options: Mapping[str, Json], pending: Collection[str] = ()) -> None:
    """Raise :class:`K4runError` if ``options`` (the stage's, :data:`EXECUTOR_KEYS` and the job's
    arguments) cannot be passed to k4run faithfully, or set what k4opt manages; the options in
    ``pending``, which parameters set, must be settable and not given. Whether the job takes its
    arguments is the job's to say (:func:`k4opt.k4run.dry_run`)."""
    options_files(options)
    if DIRECTORY in options:
        _text(options[DIRECTORY], DIRECTORY)
    geometry, input_ = wiring(options)
    for key in pending:
        settable_option(key)
        if key in options:
            raise _fail(key, "is given, but a parameter sets it")
        if key in (geometry, input_):
            raise _fail(key, "is the argument that takes the geometry or the input; k4opt sets it")
    job_arguments(options)


def k4run_arguments(
    *,
    files: Sequence[str],
    k4opt_options: Path,
    compact: Path,
    input_file: Path,
    output: Path,
    options: Mapping[str, Json],
    dry_run: bool = False,
) -> tuple[str, ...]:
    """The arguments of ``k4run``: the job's options ``files``, k4opt's own options file after
    them (so that it changes nothing the job set, and before every argument: a file after
    ``--inputFiles a.root`` would be read as one more input), then what k4opt sets (every event,
    the geometry, the input, the output) and the job's arguments, in order."""
    geometry, input_ = wiring(options)
    managed = [
        "--num-events=-1",
        f"--{geometry or 'GeoSvc.detectors'}={compact}",
        f"--{input_ or 'IOSvc.Input'}",
        str(input_file),
        f"--IOSvc.Output={output}",
    ]
    return (
        *files,
        str(k4opt_options),
        *(["--dry-run"] if dry_run else []),
        *managed,
        *job_arguments(options),
    )
