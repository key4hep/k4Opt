"""The ddsim command line: generator translation, managed options, argument order."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import DdsimError
from k4opt.ddsim.command import ddsim_arguments, generator_args, managed_conflict
from k4opt.frozen import freeze

GENERATOR: dict[str, Any] = {
    "particle": "gamma",
    "energy": {"value": 10, "unit": "GeV"},
    "vertex": {"value": [0, 0, 0], "unit": "mm"},
    "direction": {
        "distribution": "uniform_solid_angle",
        "theta": [60, 120],
        "phi": [0, 360],
        "unit": "deg",
    },
}


def args(generator: dict[str, Any]) -> tuple[str, ...]:
    return generator_args(freeze(generator, "generator", DdsimError))


def test_generator_arguments_with_units() -> None:
    assert args(GENERATOR) == (
        "--enableGun",
        "--gun.particle=gamma",
        "--gun.multiplicity=1",
        "--gun.energy=10*GeV",
        "--gun.position=0*mm 0*mm 0*mm",
        "--gun.distribution=cos(theta)",
        "--gun.thetaMin=60*deg",
        "--gun.thetaMax=120*deg",
        "--gun.phiMin=0*deg",
        "--gun.phiMax=360*deg",
    )


def test_uniform_solid_angle_is_flat_in_cos_theta_never_flat_in_theta() -> None:
    # ddsim's 'uniform' is flat in theta; only 'cos(theta)' is flat in cos(theta)
    generated = args(GENERATOR)
    assert "--gun.distribution=cos(theta)" in generated
    assert not any("uniform" in a for a in generated)


def test_units_are_carried_into_the_expressions() -> None:
    generator = copy.deepcopy(GENERATOR)
    generator["energy"] = {"value": 500.5, "unit": "MeV"}
    generator["vertex"] = {"value": [1, -2.5, 3], "unit": "cm"}
    generator["direction"].update(theta=[0.5, 2.5], phi=[-1, 1], unit="rad")
    generated = args(generator)
    assert "--gun.energy=500.5*MeV" in generated
    assert "--gun.position=1*cm -2.5*cm 3*cm" in generated
    assert "--gun.thetaMin=0.5*rad" in generated
    assert "--gun.phiMin=-1*rad" in generated


def test_vertex_defaults_to_the_origin_explicitly() -> None:
    generator = copy.deepcopy(GENERATOR)
    del generator["vertex"]
    assert "--gun.position=0*mm 0*mm 0*mm" in args(generator)


def edit(path: str, value: object) -> dict[str, Any]:
    generator = copy.deepcopy(GENERATOR)
    node = generator
    *parents, last = path.split(".")
    for key in parents:
        node = node[key]
    if value is KeyError:
        del node[last]
    else:
        node[last] = value
    return generator


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("direction.distribution", "uniform", "uniform_solid_angle"),
        ("direction.distribution", "eta", "uniform_solid_angle"),
        ("direction.distribution", KeyError, "missing"),
        ("direction.unit", "grad", "unit"),
        ("direction.theta", [120, 60], "smaller"),
        ("direction.theta", [-10, 60], r"within \[0, 180\]"),
        ("direction.theta", [60, 181], r"within \[0, 180\]"),
        ("direction.phi", [0, 361], "full turn"),
        ("direction.extra", 1, "unknown"),
        ("energy.value", 0, "positive"),
        ("energy.value", "10", "number"),
        ("energy.unit", "eV", "unit"),
        ("vertex.value", [0, 0], "x, y, z"),
        ("vertex.unit", "km", "unit"),
        ("particle", "", "particle"),
        ("particle", "-x", "particle"),
        ("particle", 22, "particle"),
        ("particle", KeyError, "missing"),
        ("momentum", 1, "unknown"),
    ],
)
def test_unsupported_generator_settings_fail_instead_of_falling_back(
    path: str, value: object, message: str
) -> None:
    with pytest.raises(DdsimError, match=message):
        args(edit(path, value))


@pytest.mark.parametrize(
    "arg",
    [
        "--compactFile",
        "--compactFile=other.xml",
        "--outputFile=x.root",
        "-O",
        "-Ox.root",
        "--numberOfEvents",
        "-N",
        "-N5",
        "--random.seed",
        "--random.seed=1",
        "--random.enableEventSeed",
        "--enableGun",
        "-G",
        "--gun.energy=1*GeV",
        "--gun.distribution",
        "--gun",  # an abbreviation of managed options
        "--inputFiles",
        "-I",
        "--macroFile",
        "--runType",
        "--enableG4Gun",
        "--vertexSigma",
        "--crossingAngleBoost",
        "--compact",  # argparse accepts unique prefixes
        "--numberOf",
    ],
)
def test_managed_options_are_recognized_in_every_spelling(arg: str) -> None:
    assert managed_conflict(arg)


@pytest.mark.parametrize(
    "arg",
    [
        "--physics.list",
        "FTFP_BERT",
        "--part.keepAllParticles",
        "--steeringFile",
        "-S",
        "-j",
        "-j4",
        "-v",
        "--printLevel",
        "--action.calo",
        "--output.kernel",
        "--",
        "-",
        "-5",
        "",
    ],
)
def test_other_options_pass_through(arg: str) -> None:
    assert not managed_conflict(arg)


def build(user_args: tuple[str, ...] = (), seed: int | None = 42) -> tuple[str, ...]:
    return ddsim_arguments(
        compact=Path("/run/geometry/c.xml"),
        output=Path("/run/simulation/out.root"),
        events=7,
        seed=seed,
        generator=freeze(GENERATOR, "generator", DdsimError),
        user_args=user_args,
    )


def test_user_arguments_follow_the_managed_ones_in_order_with_repeats() -> None:
    user = ("--physics.list", "FTFP_BERT", "--part.keepAllParticles", "True", "-v", "3", "-v", "4")
    built = build(user)
    assert built[-len(user) :] == user
    assert built[0] == "--compactFile=/run/geometry/c.xml"
    assert "--outputFile=/run/simulation/out.root" in built
    assert "--numberOfEvents=7" in built
    assert built[len(built) - len(user) - 1] == "--random.seed=42"


def test_seed_is_optional() -> None:
    assert not any(a.startswith("--random") for a in build(seed=None))


@pytest.mark.parametrize("conflict", ["--numberOfEvents=3", "-N", "--gun.energy=1*GeV", "-O"])
def test_a_user_argument_competing_with_a_managed_option_is_rejected(conflict: str) -> None:
    with pytest.raises(DdsimError, match="manages"):
        build(("--physics.list", "FTFP_BERT", conflict))


def test_plain_python_lists_are_accepted() -> None:
    assert generator_args(GENERATOR) == args(GENERATOR)


@pytest.mark.parametrize("events", [0, -4, True, 2.0, "3"])
def test_invalid_event_counts_are_rejected(events: Any) -> None:
    with pytest.raises(DdsimError, match="events"):
        ddsim_arguments(
            compact=Path("c.xml"),
            output=Path("o.root"),
            events=events,
            seed=None,
            generator=GENERATOR,
            user_args=(),
        )


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_invalid_seeds_are_rejected(seed: Any) -> None:
    with pytest.raises(DdsimError, match="seed"):
        ddsim_arguments(
            compact=Path("c.xml"),
            output=Path("o.root"),
            events=1,
            seed=seed,
            generator=GENERATOR,
            user_args=(),
        )
