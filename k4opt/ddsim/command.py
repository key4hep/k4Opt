"""Translation of the evaluation settings into a ``ddsim`` command line.

Verified against the ``ddsim`` of the Key4hep nightly 2026-10-20 (``ddsim --help``):
``--gun.*`` values are DD4hep expressions, so a value carries its unit (``10*GeV``,
``60*deg``, ``0*mm``), and ``--gun.distribution cos(theta)`` is flat in cos(theta), whereas
``uniform`` is flat in theta. Everything here is pure: nothing is run or read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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


def _fail(path: str, message: str) -> DdsimError:
    return DdsimError(f"settings.{path}: {message}")


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


def generator_args(generator: Json) -> tuple[str, ...]:
    """The ``ddsim`` particle-gun arguments for ``settings.generator``.

    ``particle``, ``energy`` and ``direction`` are required, ``vertex`` defaults to the
    origin. ``energy`` is the total energy of the primary. ``direction.distribution`` is passed
    to ``--gun.distribution`` as ddsim names it: ``cos(theta)`` is flat in cos(theta), whereas
    ``uniform`` is flat in theta; ``theta`` and ``phi`` give the ranges. Unknown keys and
    other distributions are errors; nothing falls back to another distribution.
    """
    top = _mapping(
        generator,
        "generator",
        ("particle", "energy", "vertex", "direction"),
        ("particle", "energy", "direction"),
    )
    particle = top["particle"]
    if not isinstance(particle, str) or not particle.strip() or particle.startswith("-"):
        raise _fail(
            "generator.particle", f"must be a particle name such as 'gamma', got {particle!r}"
        )

    energy = _mapping(top["energy"], "generator.energy", ("value", "unit"), ("value", "unit"))
    energy_value = _number(energy["value"], "generator.energy.value")
    if energy_value <= 0:
        raise _fail("generator.energy.value", f"must be positive, got {energy_value!r}")
    energy_unit = _unit(energy["unit"], "generator.energy.unit", _ENERGY_UNITS)

    vertex = _mapping(
        top.get("vertex", {"value": (0, 0, 0), "unit": "mm"}),
        "generator.vertex",
        ("value", "unit"),
        ("value", "unit"),
    )
    coordinates = vertex["value"]
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
        raise _fail("generator.vertex.value", f"must be a list [x, y, z], got {coordinates!r}")
    length_unit = _unit(vertex["unit"], "generator.vertex.unit", _LENGTH_UNITS)
    position = " ".join(
        _quantity(_number(c, f"generator.vertex.value[{i}]"), length_unit)
        for i, c in enumerate(coordinates)
    )

    direction = _mapping(
        top["direction"],
        "generator.direction",
        ("distribution", "theta", "phi", "unit"),
        ("distribution", "theta", "phi", "unit"),
    )
    distribution = direction["distribution"]
    if not isinstance(distribution, str) or distribution not in DIRECTION_DISTRIBUTIONS:
        raise _fail(
            "generator.direction.distribution",
            f"must be one of {list(DIRECTION_DISTRIBUTIONS)} (ddsim --gun.distribution), "
            f"got {distribution!r}",
        )
    angle_unit = _unit(direction["unit"], "generator.direction.unit", tuple(_ANGLE_UNITS))
    half_turn = _ANGLE_UNITS[angle_unit]
    theta_min, theta_max = _range(direction["theta"], "generator.direction.theta")
    if theta_min < 0 or theta_max > half_turn:
        raise _fail(
            "generator.direction.theta",
            f"must lie within [0, {half_turn:g}] {angle_unit}, got [{theta_min}, {theta_max}]",
        )
    phi_min, phi_max = _range(direction["phi"], "generator.direction.phi")
    if phi_max - phi_min > 2 * half_turn:
        raise _fail(
            "generator.direction.phi",
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


def ddsim_arguments(
    *,
    compact: Path,
    output: Path,
    events: int,
    seed: int | None,
    generator: Json,
    user_args: Sequence[str],
) -> tuple[str, ...]:
    """The arguments of ``ddsim`` (without the executable): managed ones, then ``user_args``.

    ``user_args`` are appended unchanged and in order; one that sets an option managed by
    k4opt raises :class:`DdsimError` instead of silently winning or losing. ``events`` must be
    a positive integer and ``seed`` an integer or ``None``.
    """
    if isinstance(events, bool) or not isinstance(events, int) or events < 1:
        raise DdsimError(f"events must be a positive integer, got {events!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise DdsimError(f"seed must be an integer, got {seed!r}")
    for arg in user_args:
        if managed_conflict(arg):
            raise DdsimError(
                f"backend.ddsim.args: {arg!r} sets an option that k4opt manages (geometry, "
                "output, event count, seed and generator come from the problem); "
                "remove it from the arguments"
            )
    managed = [
        f"--compactFile={compact}",
        f"--outputFile={output}",
        f"--numberOfEvents={events}",
        *generator_args(generator),
    ]
    if seed is not None:
        managed.append(f"--random.seed={seed}")
    return (*managed, *user_args)
