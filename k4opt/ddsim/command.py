"""Translation of the stage settings into a ``ddsim`` command line.

Verified against the ``ddsim`` of the Key4hep nightly 2026-10-20 (``ddsim --help`` and the
sources ``DDSim/DD4hepSimulation.py`` and ``DDSim/Helper/ConfigHelper.py``): ``--gun.*`` values
are DD4hep expressions, so a value carries its unit (``10*GeV``, ``60*deg``, ``0*mm``), and
``--gun.distribution cos(theta)`` is flat in cos(theta), whereas ``uniform`` is flat in theta.
Everything here is pure: nothing is run or read.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from k4opt.ddsim.errors import DdsimError
from k4opt.frozen import Json

# ddsim's own --gun.distribution values that are driven by theta and phi ranges
DIRECTION_DISTRIBUTIONS = ("cos(theta)", "uniform")

_ENERGY_UNITS = ("MeV", "GeV", "TeV")
_LENGTH_UNITS = ("mm", "cm", "m")
_ANGLE_UNITS = {"deg": 180.0, "rad": math.pi}  # the unit of a half turn

# ddsim options that k4opt sets itself; a user argument naming one is rejected. Entries ending
# in "." stand for a whole family. The primaries (gun or input file), the event count, the
# seed, the output and the geometry are what an evaluation defines, so every way of changing
# them is protected. Anything else is passed on and validated by ddsim.
_MANAGED_LONG = (
    "--compactFile",
    "--outputFile",
    "--outputConfig.forceLCIO",
    "--outputConfig.forceDD4HEP",
    "--numberOfEvents",
    "--runType",
    "--inputFiles",
    "--macroFile",
    "--enableGun",
    "--enableG4GPS",
    "--enableG4Gun",
    "--gun.",
    "--random.",
    "--vertexSigma",
    "--vertexOffset",
    "--crossingAngleBoost",
)
_MANAGED_SHORT = "NOGIM"  # -N, -O, -G, -I and -M, the short spellings of the above

# where a managed option comes from instead, for the error message; the rest are primaries
_MANAGED_BY = (
    ("--numberOfEvents", "set events instead"),
    ("--random.", "set seed instead"),
    ("--compactFile", "the geometry comes from the problem"),
    ("--outputFile", "k4opt chooses the output file"),
    ("--outputConfig.force", "k4opt reads the output as EDM4hep"),
    ("--runType", "k4opt runs ddsim in batch mode"),
)

# How ddsim's argparse reads an option (DD4hepSimulation.parseOptions and
# ConfigHelper.addAllHelper): an option of a group (``group.name``) whose name starts with
# "enable" or "force" is a flag without a value (store_true), and so are these top-level ones
_FLAGS = ("enableDetailedShowerMode", "disableSignalHandler")
# these options repeat (action="append") or take several values (nargs="+"); every other one
# takes a single value
_REPEATED = ("action.run", "action.event", "action.track", "action.step", "action.stack")
_SEVERAL = ("meta.eventParameters", "meta.runParameters")
# options k4opt refuses although it does not set them, with the reason
_REFUSED = {
    "steeringFile": "ddsim executes it as Python, which can change anything k4opt sets",
    "skipNEvents": "it changes which events are simulated, which the problem defines",
    "dumpParameter": "ddsim prints its configuration and exits without simulating",
    "dumpSteeringFile": "ddsim prints a steering file and exits without simulating",
    "gdb": "it runs ddsim under a debugger",
}
# options whose value is a file ddsim reads; the caller passes an absolute path and records it
FILE_OPTIONS = ("physics.pdgfile",)

_NESTED = (
    "write each option under its full ddsim name (physics.list: FTFP_BERT), not as a nested mapping"
)
_OPTION_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)?")


def output_collections(readouts: Iterable[str]) -> frozenset[str]:
    """The collections ddsim is expected to write for a geometry with these ``readouts``: one per
    readout, a ``<readout>Contributions`` next to each (written for calorimeters),
    ``MCParticles`` and ``EventHeader``.

    An expectation, checked on CLD_o2_v09, not a guarantee: the type of a sensitive detector is
    set in its C++ code, and DDG4 actions may define collections of their own.
    """
    names = set(readouts)
    return frozenset(
        names | {f"{name}Contributions" for name in names} | {"MCParticles", "EventHeader"}
    )


def _fail(path: str, message: str) -> DdsimError:
    return DdsimError(f"{path}: {message}")


def _mapping(
    value: Json, path: str, allowed: tuple[str, ...], required: tuple[str, ...]
) -> Mapping[str, Json]:
    if not isinstance(value, Mapping):
        raise _fail(path, f"must be a mapping, got {value!r}")
    unknown = [key for key in value if key not in allowed]
    if unknown:
        raise _fail(path, f"unknown key(s) {unknown}; supported: {list(allowed)}")
    missing = [key for key in required if key not in value]
    if missing:
        raise _fail(path, f"missing required key(s) {missing}")
    return value


def _number(value: Json, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(path, f"must be a number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise _fail(path, f"must be finite, got {value!r}")
    return value


def _unit(value: Json, path: str, units: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in units:
        raise _fail(path, f"unit must be one of {list(units)}, got {value!r}")
    return value


def _quantity(value: int | float, unit: str) -> str:
    return f"{value!r}*{unit}"


def _range(value: Json, path: str) -> tuple[int | float, int | float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _fail(path, f"must be a list [low, high], got {value!r}")
    low, high = _number(value[0], f"{path}[0]"), _number(value[1], f"{path}[1]")
    if not low < high:
        raise _fail(path, f"low must be smaller than high, got {value!r}")
    return low, high


def particle_gun_args(particle_gun: Json) -> tuple[str, ...]:
    """The ``ddsim`` particle-gun arguments for the generation stage of a problem.

    ``particle``, ``energy`` and ``direction`` are required, ``vertex`` defaults to the
    origin. ``energy`` is the total energy of the primary. ``direction.distribution`` is passed
    to ``--gun.distribution`` as ddsim names it: ``cos(theta)`` is flat in cos(theta), whereas
    ``uniform`` is flat in theta; ``theta`` and ``phi`` give the ranges. Unknown keys and
    other distributions are errors; nothing falls back to another distribution.
    """
    top = _mapping(
        particle_gun,
        "generation",
        ("particle", "energy", "vertex", "direction"),
        ("particle", "energy", "direction"),
    )
    particle = top["particle"]
    if not isinstance(particle, str) or not particle.strip() or particle.startswith("-"):
        raise _fail(
            "generation.particle",
            f"must be a particle name such as 'gamma', got {particle!r}",
        )

    energy = _mapping(top["energy"], "generation.energy", ("value", "unit"), ("value", "unit"))
    energy_value = _number(energy["value"], "generation.energy.value")
    if energy_value <= 0:
        raise _fail("generation.energy.value", f"must be positive, got {energy_value!r}")
    energy_unit = _unit(energy["unit"], "generation.energy.unit", _ENERGY_UNITS)

    vertex = _mapping(
        top.get("vertex", {"value": (0, 0, 0), "unit": "mm"}),
        "generation.vertex",
        ("value", "unit"),
        ("value", "unit"),
    )
    coordinates = vertex["value"]
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
        raise _fail("generation.vertex.value", f"must be a list [x, y, z], got {coordinates!r}")
    length_unit = _unit(vertex["unit"], "generation.vertex.unit", _LENGTH_UNITS)
    position = " ".join(
        _quantity(_number(c, f"generation.vertex.value[{i}]"), length_unit)
        for i, c in enumerate(coordinates)
    )

    direction = _mapping(
        top["direction"],
        "generation.direction",
        ("distribution", "theta", "phi", "unit"),
        ("distribution", "theta", "phi", "unit"),
    )
    distribution = direction["distribution"]
    if not isinstance(distribution, str) or distribution not in DIRECTION_DISTRIBUTIONS:
        raise _fail(
            "generation.direction.distribution",
            f"must be one of {list(DIRECTION_DISTRIBUTIONS)} (ddsim --gun.distribution), "
            f"got {distribution!r}",
        )
    angle_unit = _unit(direction["unit"], "generation.direction.unit", tuple(_ANGLE_UNITS))
    half_turn = _ANGLE_UNITS[angle_unit]
    theta_min, theta_max = _range(direction["theta"], "generation.direction.theta")
    if theta_min < 0 or theta_max > half_turn:
        raise _fail(
            "generation.direction.theta",
            f"must lie within [0, {half_turn:g}] {angle_unit}, got [{theta_min}, {theta_max}]",
        )
    phi_min, phi_max = _range(direction["phi"], "generation.direction.phi")
    if phi_max - phi_min > 2 * half_turn:
        raise _fail(
            "generation.direction.phi",
            f"must not span more than a full turn, got [{phi_min}, {phi_max}]",
        )

    return (
        "--enableGun",
        f"--gun.particle={particle}",
        "--gun.multiplicity=1",
        f"--gun.energy={_quantity(energy_value, energy_unit)}",
        f"--gun.position={position}",
        f"--gun.distribution={distribution}",
        f"--gun.thetaMin={_quantity(theta_min, angle_unit)}",
        f"--gun.thetaMax={_quantity(theta_max, angle_unit)}",
        f"--gun.phiMin={_quantity(phi_min, angle_unit)}",
        f"--gun.phiMax={_quantity(phi_max, angle_unit)}",
    )


def managed_conflict(arg: str) -> bool:
    """Whether ``arg`` names an option that k4opt sets itself.

    ``ddsim`` accepts ``--option=value``, ``-Nvalue`` and unambiguous prefixes of long
    options, so all of these spellings are recognized.
    """
    if not arg.startswith("-") or len(arg) < 2:
        return False
    if not arg.startswith("--"):
        return arg[1] in _MANAGED_SHORT
    name = arg.split("=", 1)[0]
    if len(name) <= 2:  # "--" ends the options
        return False
    return any(
        managed.startswith(name) or (managed.endswith(".") and name.startswith(managed))
        for managed in _MANAGED_LONG
    )


def _check_option_name(key: object, path: str) -> str:
    """``key`` if it may name a ``ddsim`` option the problem sets; raises otherwise."""
    if not isinstance(key, str) or not _OPTION_NAME.fullmatch(key):
        raise _fail(
            path,
            "must be a ddsim option as `ddsim --help` names it, without the leading --, "
            "such as physics.list",
        )
    option = f"--{key}"
    if managed_conflict(option):
        hint = next(
            (
                hint
                for managed, hint in _MANAGED_BY
                if option.startswith(managed) or managed.startswith(option)
            ),
            "the primaries come from the generation stage",
        )
        raise _fail(path, f"is an option that k4opt sets itself: {hint}")
    for refused, reason in _REFUSED.items():
        if refused.startswith(key):  # ddsim also accepts a prefix of an option's name
            raise _fail(path, f"--{refused} is not supported: {reason}")
    known = (*_FLAGS, *_REPEATED, *_SEVERAL, *FILE_OPTIONS)
    longer = [name for name in known if name.startswith(key) and name != key]
    if longer:
        raise _fail(path, f"write the full option name ({', '.join(longer)})")
    return key


def _is_flag(key: str) -> bool:
    _, dot, name = key.partition(".")
    return name.startswith(("enable", "force")) if dot else key in _FLAGS


def _option_text(value: Json, path: str) -> str:
    """One value as ddsim reads it: text unchanged, a number as written, a boolean as
    ``True``/``False`` (ddsim's ``ConfigHelper.makeBool``)."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(_number(value, path))
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Mapping):
        raise _fail(path, _NESTED)
    raise _fail(path, f"must be a text, a number or a boolean, got {value!r}")


def option_args(options: Mapping[str, Json]) -> tuple[str, ...]:
    """The ``ddsim`` arguments for ``options``, ddsim option names mapped to values, in order.

    A flag (``enableDetailedShowerMode``, a group option named ``enable*`` or ``force*``) is
    switched on with ``true``; ``false`` is an error, since ddsim cannot switch a flag off on
    its command line. The options that repeat (``action.run``, ...) or take several values
    (``meta.eventParameters``, ...) accept a list. Every other option takes one value, given as
    ``--name=value``. Options that k4opt sets itself (the geometry, output, event count, seed
    and primaries) and those it refuses (``steeringFile``, ...) raise :class:`DdsimError`, as
    does a value that cannot be written faithfully. Option names and values are otherwise
    validated by ddsim when it starts.
    """
    if not isinstance(options, Mapping):
        raise _fail("simulation", f"options must be a mapping, got {options!r}")
    args: list[str] = []
    for key, value in options.items():
        path = f"simulation.{key}"
        if isinstance(value, Mapping):
            raise _fail(path, _NESTED)
        key = _check_option_name(key, path)
        if _is_flag(key):
            if value is not True:
                raise _fail(
                    path,
                    "ddsim cannot switch a flag off on its command line; leave it out"
                    if value is False
                    else f"is a flag: write true to switch it on, got {value!r}",
                )
            args.append(f"--{key}")
        elif key in _REPEATED or key in _SEVERAL:
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            if not values:
                raise _fail(path, "must not be an empty list")
            texts = [_option_text(item, f"{path}[{i}]") for i, item in enumerate(values)]
            if key in _REPEATED:
                args.extend(f"--{key}={text}" for text in texts)
            else:
                # nargs='+' stops at an option token, even inside a list of metadata.
                # Passing it through would let a value override a managed option.
                if any(text.startswith("-") for text in texts):
                    raise _fail(path, "metadata values must not start with '-' (a ddsim option)")
                args.extend((f"--{key}", *texts))
        elif isinstance(value, (list, tuple)):
            raise _fail(
                path,
                "ddsim reads a single value for this option; write it as one text the way "
                "ddsim's command line expects it",
            )
        else:
            args.append(f"--{key}={_option_text(value, path)}")
    return tuple(args)


def ddsim_arguments(
    *,
    compact: Path,
    output: Path,
    events: int,
    seed: int | None,
    particle_gun: Json = None,
    sample: Path | None = None,
    options: Mapping[str, Json],
) -> tuple[str, ...]:
    """The arguments of ``ddsim`` (without the executable): managed ones, then ``options``.

    The primaries come from exactly one of ``particle_gun`` (see :func:`particle_gun_args`) and
    ``sample``, an event file that ``ddsim`` reads with ``--inputFiles`` (it picks the reader
    from the file extension and validates the content). ``options`` follow in order, translated
    by :func:`option_args`; one that sets an option managed by k4opt raises :class:`DdsimError`
    instead of silently winning or losing. ``events`` must be a positive integer and ``seed`` an
    integer or ``None``.
    """
    if isinstance(events, bool) or not isinstance(events, int) or events < 1:
        raise DdsimError(f"events must be a positive integer, got {events!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise DdsimError(f"seed must be an integer, got {seed!r}")
    if (particle_gun is None) == (sample is None):
        raise DdsimError(
            "generation: set exactly one of particle_gun and sample "
            f"(got {'both' if sample is not None else 'neither'})"
        )
    user = option_args(options)
    managed = [
        f"--compactFile={compact}",
        f"--outputFile={output}",
        f"--numberOfEvents={events}",
        *(particle_gun_args(particle_gun) if sample is None else (f"--inputFiles={sample}",)),
    ]
    if seed is not None:
        managed.append(f"--random.seed={seed}")
    return (*managed, *user)
