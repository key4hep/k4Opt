"""Where a parameter's value goes: its destinations, parsed in two phases, and a candidate
configured from them."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from k4opt.problem import (
    CandidateError,
    GeometryDestination,
    ProblemConfigError,
    SettingDestination,
    StageDestination,
    parse_problem,
)
from k4opt.problem.candidate import configure, validate_candidate
from k4opt.problem.plan import (
    GEOMETRY,
    evaluator_task,
    plan_tasks,
    represents_candidate,
    stage_task,
    varied,
)
from tests.support.problems import two_samples

EVALUATOR = "calo.single_particle_shower"


def reject(data: object, message: str) -> None:
    """Assert that ``data`` is refused with ``message`` (a literal fragment)."""
    with pytest.raises(ProblemConfigError, match=re.escape(message)):
        parse_problem(data)


def add(doc: dict[str, Any], name: str, sets: object, **domain: object) -> dict[str, Any]:
    """``doc`` with the parameter ``name``, continuous in [1, 2] unless ``domain`` says
    otherwise, that sets ``sets``."""
    numeric = domain.get("type", "continuous") != "categorical"
    parameter = ({"type": "continuous", "bounds": [1.0, 2.0]} if numeric else {}) | domain
    doc["parameters"][name] = parameter | {"sets": sets}
    return doc


def energy(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` whose gun energy is set by the parameter ``energy``, in GeV."""
    del doc["samples"]["gamma"]["generation"]["energy"]
    return add(doc, "energy", {"stage": "generation", "option": "energy"}, unit="GeV")


# --- destinations ----------------------------------------------------------------------------


def test_each_kind_of_destination(doc: dict[str, Any]) -> None:
    doc = energy(doc)
    del doc["evaluators"][0]["settings"]
    doc = add(doc, "range_cut", {"stage": "simulation", "option": "physics.rangecut"}, unit="mm")
    sets = {"evaluator": EVALUATOR, "setting": "collection"}
    doc = add(doc, "collection", sets, type="categorical", choices=["Barrel", "Endcap"])
    problem = parse_problem(doc)
    cell, gun, cut, collection = problem.parameters
    assert cell.sets == (GeometryDestination(constant="ECal_cell_size"),)
    assert gun.sets == (StageDestination("generation", "energy", ("gamma",)),)
    assert cut.sets == (StageDestination("simulation", "physics.rangecut", ("gamma",)),)
    assert collection.sets == (SettingDestination(EVALUATOR, "collection"),)
    assert [d.label for _, d in problem.destinations] == [
        "geometry constants.ECal_cell_size",
        "generation.energy of gamma",
        "simulation.physics.rangecut of gamma",
        f"{EVALUATOR}.collection",
    ]


def test_one_parameter_sets_several_destinations(doc: dict[str, Any]) -> None:
    del doc["evaluators"][0]["settings"]
    doc["parameters"]["ecal_cell_size"]["sets"] = [
        {"geometry": {"constant": "ECal_cell_size"}},
        {"evaluator": EVALUATOR, "setting": "collection"},
    ]
    (parameter,) = parse_problem(doc).parameters
    assert [type(d).__name__ for d in parameter.sets] == [
        "GeometryDestination",
        "SettingDestination",
    ]


def test_the_problem_keeps_what_it_declares(doc: dict[str, Any]) -> None:
    # no value of a parameter, no edge of its domain: the gun simply lacks its energy
    problem = parse_problem(energy(doc))
    assert "energy" not in problem.samples["gamma"].stages["generation"].options
    assert problem.pending("gamma") == {"generation": frozenset({"energy"})}
    assert parse_problem(problem.to_dict()) == problem


def test_the_normalized_form_lists_samples_and_destinations(doc: dict[str, Any]) -> None:
    data = parse_problem(energy(doc)).to_dict()
    assert data["parameters"]["energy"] == {  # type: ignore[index]
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "unit": "GeV",
        "sets": [{"stage": "generation", "option": "energy", "samples": ["gamma"]}],
    }


@pytest.mark.parametrize(
    ("sets", "message"),
    [
        ([], "parameters.x.sets: must name at least one destination"),
        ({}, "parameters.x.sets: needs exactly one of 'geometry'"),
        ({"geometry": {"constant": "X"}, "setting": "y"}, "a geometry destination takes no"),
        ({"stage": "generation"}, "needs 'option'"),
        ({"stage": "reconstruction", "option": "x"}, "parameters.x.sets.stage: must be one of"),
        ({"evaluator": EVALUATOR}, "needs 'setting'"),
        ({"evaluator": "other", "setting": "y"}, "the problem has no evaluator 'other'"),
        (
            {"stage": "simulation", "option": "physics.rangecut", "samples": ["zz"]},
            "parameters.x.sets.samples[0]: the problem has no sample 'zz'",
        ),
        (
            {"stage": "simulation", "option": "x", "samples": ["gamma", "gamma"]},
            "'gamma' is listed twice",
        ),
        ([{"geometry": {"constant": "X"}}, 5], "parameters.x.sets[1]: must be a mapping"),
    ],
)
def test_destination_errors(doc: dict[str, Any], sets: object, message: str) -> None:
    reject(add(doc, "x", sets), message)


def test_a_stage_destination_names_its_samples_when_there_are_several(
    doc: dict[str, Any],
) -> None:
    doc = add(two_samples(doc), "cut", {"stage": "simulation", "option": "physics.rangecut"})
    reject(doc, "the problem has the samples gamma, pion; add samples: with those whose")
    doc["parameters"]["cut"]["sets"]["samples"] = ["gamma", "pion"]
    (_, cut) = parse_problem(doc).parameters
    assert cut.sets == (StageDestination("simulation", "physics.rangecut", ("gamma", "pion")),)


def test_a_stage_given_as_a_file_has_no_options_to_set(doc: dict[str, Any]) -> None:
    doc["samples"]["gamma"]["generation"] = {"file": "events.hepmc"}
    reject(
        add(doc, "e", {"stage": "generation", "option": "energy"}, unit="GeV"),
        "samples.gamma.generation is given as a file, which is not run",
    )


# --- collisions ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda d: add(d, "again", {"geometry": {"target": "constants.ECal_cell_size"}}),
            "parameters.again.sets: the geometry's constants.ECal_cell_size is set by "
            "parameters.ecal_cell_size already",
        ),
        (
            lambda d: d["parameters"]["ecal_cell_size"].update(
                sets=[{"geometry": {"constant": "ECal_cell_size"}}] * 2
            ),
            "parameters.ecal_cell_size.sets[1]: the geometry's constants.ECal_cell_size is set "
            "by this parameter already",
        ),
        (
            lambda d: add(
                add(d, "a", {"stage": "simulation", "option": "physics.rangecut"}),
                "b",
                {"stage": "simulation", "option": "physics.rangecut"},
            ),
            "parameters.b.sets: samples.gamma.simulation.physics.rangecut is set by "
            "parameters.a already",
        ),
        (
            lambda d: add(
                add(d, "a", {"evaluator": EVALUATOR, "setting": "y"}),
                "b",
                {"evaluator": EVALUATOR, "setting": "y"},
            ),
            f"the setting y of evaluator {EVALUATOR!r} is set by parameters.a already",
        ),
        (
            lambda d: add(d, "pl", {"stage": "simulation", "option": "physics.list"}),
            "samples.gamma.simulation.physics.list: is set by parameters.pl; leave it out here",
        ),
        (
            lambda d: add(d, "c", {"evaluator": EVALUATOR, "setting": "collection"}),
            "evaluators[0].settings.collection: is set by parameters.c; leave it out here",
        ),
    ],
)
def test_a_destination_is_set_once(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    edit(doc)
    reject(doc, message)


def test_an_option_written_in_the_shared_stages_points_there(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    doc["stages"] = {"simulation": doc["samples"]["gamma"].pop("simulation")}
    del doc["samples"]["pion"]["simulation"]
    del doc["stages"]["simulation"]["seed"]
    sets = {"stage": "simulation", "option": "physics.list", "samples": ["gamma"]}
    reject(
        add(doc, "pl", sets, type="categorical", choices=["FTFP_BERT", "QGSP_BERT"]),
        "stages.simulation.physics.list: is set by parameters.pl; leave it out here",
    )


# --- what each destination takes -----------------------------------------------------------


@pytest.mark.parametrize(
    ("sets", "domain", "message"),
    [
        (
            {"stage": "generation", "option": "energy"},
            {"unit": "keV"},
            "sets a quantity in MeV, GeV, TeV: the parameter must be continuous or integer, with "
            "one of these units, got continuous in keV",
        ),
        ({"stage": "generation", "option": "energy"}, {}, "got continuous"),
        (
            {"stage": "generation", "option": "particle"},
            {},
            "sets a text: the parameter must be categorical, with string choices",
        ),
        (
            {"stage": "generation", "option": "direction"},
            {},
            "parameters.x.sets.option: gun: generation.direction: a parameter can set only the "
            "particle and energy of the particle gun",
        ),
        (
            {"stage": "simulation", "option": "numberOfThreads"},
            {"type": "integer", "bounds": [1, 4]},
            "it says how many processors the simulation keeps busy",
        ),
        ({"stage": "simulation", "option": "physics.pdgfile"}, {}, "it names a file"),
        ({"stage": "simulation", "option": "enableDetailedShowerMode"}, {}, "it is a flag"),
        ({"stage": "simulation", "option": "action.run"}, {}, "it takes a list of values"),
        ({"stage": "simulation", "option": "numberOfEvents"}, {}, "an option that k4opt sets"),
        ({"stage": "simulation", "option": "steeringFile"}, {}, "--steeringFile is not supported"),
        ({"stage": "simulation", "option": "physics.rangecut"}, {"unit": "s"}, "unit 's' cannot"),
        ({"geometry": {"constant": "X"}}, {"unit": "inch"}, "unit 'inch' cannot be written"),
    ],
)
def test_a_value_must_fit_its_destination(
    doc: dict[str, Any], sets: dict[str, str], domain: dict[str, object], message: str
) -> None:
    if sets.get("stage") == "generation":  # the gun gives it no more: the parameter does
        doc["samples"]["gamma"]["generation"].pop(sets["option"], None)
    reject(add(doc, "x", sets, **domain), message)


@pytest.mark.parametrize(
    ("domain", "message"),
    [
        (
            {"bounds": [0, 10]},
            "parameters.energy: at its bound 0, samples.gamma.generation.energy.value: must be "
            "positive, got 0",
        ),
        (
            {"type": "integer", "bounds": [-3, 3]},
            "parameters.energy: at its bound -3, samples.gamma.generation.energy.value",
        ),
    ],
)
def test_a_stage_option_is_checked_at_each_end_of_the_domain(
    doc: dict[str, Any], domain: dict[str, object], message: str
) -> None:
    doc = energy(doc)
    doc["parameters"]["energy"] |= domain
    reject(doc, message)


def test_each_choice_is_checked_where_it_goes(doc: dict[str, Any]) -> None:
    del doc["samples"]["gamma"]["generation"]["particle"]
    sets = {"stage": "generation", "option": "particle"}
    reject(
        add(doc, "p", sets, type="categorical", choices=["gamma", "-e"]),
        "parameters.p: at its choice '-e', samples.gamma.generation.particle: must be a particle",
    )


def test_a_pending_option_is_not_missing_but_the_others_are_checked(doc: dict[str, Any]) -> None:
    doc = energy(doc)
    parse_problem(doc)  # the energy is the candidate's
    direction = doc["samples"]["gamma"]["generation"]["direction"]
    direction["theta"] = [80, 300]
    reject(doc, "samples.gamma.generation.direction.theta: must lie within [0, 180] deg")
    direction["theta"] = [80, 100]
    del doc["parameters"]["energy"]
    reject(doc, "samples.gamma.generation: missing required key(s) ['energy']")


@pytest.mark.parametrize(
    ("domain", "message"),
    [
        ({"type": "integer", "bounds": [0, 2**53 + 2]}, "every integer beyond 2**53"),
        ({"bounds": [0, 10**400]}, "cannot hold 1000"),
        ({"type": "categorical", "choices": [1, 2**60 + 1]}, "cannot hold 1152921504606846977"),
    ],
)
def test_a_number_read_as_a_double_must_be_one(
    doc: dict[str, Any], domain: dict[str, object], message: str
) -> None:
    parameter = doc["parameters"]["ecal_cell_size"]
    parameter |= domain
    if parameter["type"] == "categorical":
        del parameter["bounds"], parameter["unit"]
    reject(doc, message)


# --- configure -------------------------------------------------------------------------------


def mixed(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` with a geometry, a gun, a ddsim and a setting parameter."""
    doc = energy(doc)
    doc = add(doc, "range_cut", {"stage": "simulation", "option": "physics.rangecut"}, unit="mm")
    sets = {"evaluator": EVALUATOR, "setting": "threshold"}
    return add(doc, "threshold", sets, bounds=[30.0, 150.0], unit="keV")


def test_configure_writes_each_value_where_it_goes(doc: dict[str, Any]) -> None:
    problem = parse_problem(mixed(doc))
    candidate = {"ecal_cell_size": 6, "energy": 1.5, "range_cut": 1.25, "threshold": 70}
    values, _ = validate_candidate(problem, candidate)
    configuration = configure(problem, values)
    assert configuration.geometry == ("6*mm",)
    stages = configuration.samples["gamma"].stages
    assert stages["generation"].options["energy"] == {"value": 1.5, "unit": "GeV"}
    assert stages["simulation"].options["physics.rangecut"] == "1.25*mm"
    assert list(stages["simulation"].options) == ["physics.list", "physics.rangecut"]
    (entry,) = configuration.evaluators
    assert entry.settings == {
        "collection": "ECalBarrelCollection",
        "threshold": {"value": 70, "unit": "keV"},
    }
    # nothing of the problem changes, and the same candidate gives the same configuration
    assert "energy" not in problem.samples["gamma"].stages["generation"].options
    assert problem.evaluators[0].settings == {"collection": "ECalBarrelCollection"}
    assert configure(problem, values) == configuration


def test_configure_sets_every_sample_and_entry_a_destination_names(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    doc["evaluators"].append(doc["evaluators"][0] | {"samples": ["pion"]})
    doc["evaluators"][0]["samples"] = ["gamma"]
    for item in (*doc["objectives"], *doc["constraints"]):
        item["evaluator"] = EVALUATOR
    sets = {"stage": "simulation", "option": "physics.rangecut", "samples": ["gamma", "pion"]}
    doc = add(doc, "cut", sets, unit="mm")
    doc = add(doc, "t", {"evaluator": EVALUATOR, "setting": "threshold"})
    problem = parse_problem(doc)
    values, _ = validate_candidate(problem, {"ecal_cell_size": 5.0, "cut": 1.0, "t": 2.0})
    configuration = configure(problem, values)
    for sample in ("gamma", "pion"):
        assert configuration.samples[sample].stages["simulation"].options["physics.rangecut"] == (
            "1.0*mm"
        )
    assert [e.settings["threshold"] for e in configuration.evaluators] == [2.0, 2.0]


def test_configure_refuses_a_value_no_double_holds(doc: dict[str, Any]) -> None:
    doc["parameters"]["ecal_cell_size"]["bounds"] = [0.0, 1e300]
    problem = parse_problem(doc)
    values, _ = validate_candidate(problem, {"ecal_cell_size": 2**60 + 1})
    with pytest.raises(CandidateError, match="cannot hold it exactly"):
        configure(problem, values)


# --- what each task depends on ---------------------------------------------------------------


def test_varied_follows_what_each_task_reads(doc: dict[str, Any]) -> None:
    problem = parse_problem(mixed(doc))
    reach = varied(problem.samples, problem.evaluators, problem.parameters)
    assert reach[GEOMETRY] == ("ecal_cell_size",)
    assert reach[stage_task("generation", "gamma")] == ("energy",)
    assert reach[stage_task("simulation", "gamma")] == ("ecal_cell_size", "energy", "range_cut")
    assert reach[evaluator_task(EVALUATOR, "gamma")] == (
        "ecal_cell_size",
        "energy",
        "range_cut",
        "threshold",
    )


def test_a_threshold_alone_leaves_a_simulation_file_usable(doc: dict[str, Any]) -> None:
    doc["samples"]["gamma"] = {"simulation": {"file": "events.edm4hep.root"}}
    doc["parameters"] = {}
    doc = add(doc, "threshold", {"evaluator": EVALUATOR, "setting": "threshold"}, unit="keV")
    problem = parse_problem(doc)  # nothing the file stands for depends on the candidate
    reach = varied(problem.samples, problem.evaluators, problem.parameters)
    assert reach[stage_task("simulation", "gamma")] == ()
    assert reach[evaluator_task(EVALUATOR, "gamma")] == ("threshold",)


def test_a_simulation_file_is_refused_when_its_generation_varies(doc: dict[str, Any]) -> None:
    doc = energy(doc)
    doc["parameters"] = {"energy": doc["parameters"]["energy"]}
    doc["samples"]["gamma"]["simulation"] = {"file": "events.edm4hep.root"}
    reject(doc, "samples.gamma.simulation.file: parameters.energy reach this stage")


def test_a_supplied_stage_represents_the_candidate_unless_a_parameter_reaches_it(
    doc: dict[str, Any],
) -> None:
    supplied = {stage_task("simulation", "gamma")}
    for parameters, represents in (
        ({"threshold": {"evaluator": EVALUATOR, "setting": "threshold"}}, True),
        ({"cell": {"geometry": {"constant": "ECal_cell_size"}}}, False),
    ):
        doc["parameters"] = {}
        for name, sets in parameters.items():
            add(doc, name, sets)
        problem = parse_problem(doc)
        plan = plan_tasks(
            problem.samples,
            problem.evaluators,
            supplied,
            references=problem.references,
            parameters=problem.parameters,
        )
        assert represents_candidate(plan) is represents


def test_the_example_of_every_destination_parses_and_fits_its_evaluator() -> None:
    from pathlib import Path

    import yaml

    from k4opt.evaluation import check_bound
    from tests.support.evaluators import calo_photon

    path = Path(__file__).parents[3] / "examples" / "problem" / "cld-ecal-bindings.yaml"
    problem = parse_problem(yaml.safe_load(path.read_text()), source=path)
    assert [(p.name, d.label) for p, d in problem.destinations] == [
        ("ecal_cell_size", "geometry constants.ECal_cell_size"),
        ("range_cut", "simulation.physics.rangecut of gamma_10gev"),
        ("cell_threshold", "calo.photon.threshold"),
    ]
    threshold = problem.parameters[2]
    check_bound(calo_photon()(), "threshold", float, threshold.unit)  # keV into its GeV
    values, _ = validate_candidate(
        problem, {"ecal_cell_size": 8.0, "range_cut": 0.5, "cell_threshold": 70.0}
    )
    configuration = configure(problem, values)
    simulation = configuration.samples["gamma_10gev"].stages["simulation"].options
    assert simulation["physics.rangecut"] == "0.5*mm"
    assert configuration.evaluators[0].settings["threshold"] == {"value": 70.0, "unit": "keV"}
