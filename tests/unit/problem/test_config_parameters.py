"""Where a parameter's value goes: its ``targets``, parsed in two phases into one destination
each, and a candidate configured from them."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.evaluation import check_bound
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
from tests.support.evaluators import calo_photon
from tests.support.problems import two_samples

EVALUATOR = "calo.single_particle_shower"
EXAMPLES = Path(__file__).parents[3] / "examples" / "problem"


def reject(data: object, message: str) -> None:
    """Assert that ``data`` is refused with ``message`` (a literal fragment)."""
    with pytest.raises(ProblemConfigError, match=re.escape(message)):
        parse_problem(data)


def add(doc: dict[str, Any], name: str, targets: object, **domain: object) -> dict[str, Any]:
    """``doc`` with the parameter ``name``, continuous in [1, 2] unless ``domain`` says
    otherwise, whose targets are ``targets``."""
    numeric = domain.get("type", "continuous") != "categorical"
    parameter = ({"type": "continuous", "bounds": [1.0, 2.0]} if numeric else {}) | domain
    doc["parameters"][name] = parameter | {"targets": targets}
    return doc


def energy(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` whose gun energy is the target of the parameter ``energy``, in GeV."""
    del doc["samples"]["gamma"]["generation"]["energy"]
    return add(doc, "energy", {"generation": "energy"}, unit="GeV")


def with_cost(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` with a second evaluator, ``cost``, of the candidate, and the shower evaluator's
    settings left to the parameters."""
    doc["evaluators"].append({"name": "cost", "version": "1"})
    del doc["evaluators"][0]["settings"]
    for item in (*doc["objectives"], *doc["constraints"]):
        item["evaluator"] = EVALUATOR
    return doc


def targets_of(doc: dict[str, Any], name: str) -> tuple[object, ...]:
    return next(p for p in parse_problem(doc).parameters if p.name == name).targets


# --- each kind of target -------------------------------------------------------------------


def test_each_kind_of_target_reads_into_one_destination(doc: dict[str, Any]) -> None:
    doc = energy(doc)
    del doc["evaluators"][0]["settings"]
    doc = add(doc, "range_cut", {"simulation": "physics.rangecut"}, unit="mm")
    add(doc, "collection", {"evaluators": {EVALUATOR: "collection"}}, type="categorical")
    doc["parameters"]["collection"]["choices"] = ["Barrel", "Endcap"]
    problem = parse_problem(doc)
    cell, gun, cut, collection = problem.parameters
    assert cell.targets == (GeometryDestination("ECal_cell_size"),)
    assert gun.targets == (StageDestination("generation", "energy", ("gamma",)),)
    assert cut.targets == (StageDestination("simulation", "physics.rangecut", ("gamma",)),)
    assert collection.targets == (SettingDestination(EVALUATOR, "collection"),)
    assert [d.label for _, d in problem.destinations] == [
        "geometry constants.ECal_cell_size",
        "generation energy of gamma",
        "simulation physics.rangecut of gamma",
        f"evaluator {EVALUATOR}: collection",
    ]


def test_geometry_targets_are_constants_by_name_or_canonical_references(
    doc: dict[str, Any],
) -> None:
    references = ["ECal_cell_size", "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"]
    doc["parameters"]["ecal_cell_size"]["targets"] = {"geometry": references}
    (cell, nphi) = targets_of(doc, "ecal_cell_size")
    assert isinstance(cell, GeometryDestination)
    assert isinstance(nphi, GeometryDestination)
    assert cell.canonical_target == "constants.ECal_cell_size"  # a name alone is a constant
    assert nphi.canonical_target == references[1]


def test_one_parameter_reaches_the_geometry_and_an_evaluator(doc: dict[str, Any]) -> None:
    del doc["evaluators"][0]["settings"]
    doc["parameters"]["ecal_cell_size"]["targets"] = {
        "geometry": "ECal_cell_size",
        "evaluators": {EVALUATOR: "collection"},
    }
    assert targets_of(doc, "ecal_cell_size") == (
        GeometryDestination("ECal_cell_size"),
        SettingDestination(EVALUATOR, "collection"),
    )


def test_one_parameter_sets_several_options_of_a_stage(doc: dict[str, Any]) -> None:
    doc = add(doc, "cut", {"simulation": ["physics.rangecut", "part.minimalKineticEnergy"]})
    assert targets_of(doc, "cut") == (
        StageDestination("simulation", "physics.rangecut", ("gamma",)),
        StageDestination("simulation", "part.minimalKineticEnergy", ("gamma",)),
    )


def test_stage_options_of_named_samples(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    doc = add(doc, "cut", {"simulation": "physics.rangecut"})
    reject(
        doc,
        "parameters.cut.targets.simulation: the problem has the samples gamma, pion; say which "
        "this parameter sets: {option: physics.rangecut, samples: [...]}",
    )
    doc["parameters"]["cut"]["targets"] = {
        "simulation": [
            {"option": "physics.rangecut", "samples": ["gamma", "pion"]},
            {"option": "part.minimalKineticEnergy", "samples": ["pion"]},
        ]
    }
    assert targets_of(doc, "cut") == (
        StageDestination("simulation", "physics.rangecut", ("gamma", "pion")),
        StageDestination("simulation", "part.minimalKineticEnergy", ("pion",)),
    )


def test_settings_of_several_evaluators_and_several_settings_of_one(doc: dict[str, Any]) -> None:
    doc = with_cost(doc)
    targets = {"evaluators": {EVALUATOR: ["collection", "threshold"], "cost": "cell"}}
    doc = add(doc, "x", targets)
    assert targets_of(doc, "x") == (
        SettingDestination(EVALUATOR, "collection"),
        SettingDestination(EVALUATOR, "threshold"),
        SettingDestination("cost", "cell"),
    )


def test_the_problem_keeps_what_it_declares(doc: dict[str, Any]) -> None:
    # no value of a parameter, no end of its domain: the gun simply lacks its energy
    problem = parse_problem(energy(doc))
    assert "energy" not in problem.samples["gamma"].stages["generation"].options
    assert problem.pending("gamma") == {"generation": frozenset({"energy"})}
    assert parse_problem(problem.to_dict()) == problem


def test_the_normalized_form_writes_one_form_of_each_target(doc: dict[str, Any]) -> None:
    doc = with_cost(doc)
    targets = {
        "geometry": ["A", "B.c"],
        "simulation": "physics.rangecut",
        "evaluators": {EVALUATOR: ["collection", "threshold"], "cost": "cell"},
    }
    problem = parse_problem(add(energy(doc), "x", targets))
    data = problem.to_dict()["parameters"]
    assert isinstance(data, dict)
    assert data["energy"]["targets"] == {"generation": {"option": "energy", "samples": ["gamma"]}}
    assert data["x"]["targets"] == {
        "geometry": ["A", "B.c"],
        "simulation": {"option": "physics.rangecut", "samples": ["gamma"]},
        "evaluators": {EVALUATOR: ["collection", "threshold"], "cost": "cell"},
    }
    assert parse_problem(problem.to_dict()) == problem


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ({}, "parameters.x.targets: must name at least one target: geometry, generation"),
        ({"stage": "simulation"}, "parameters.x.targets: unknown key(s) ['stage']"),
        ({"analysis": "x"}, "unknown key(s) ['analysis']"),
        ({"geometry": []}, "parameters.x.targets.geometry: must name at least one target"),
        ({"geometry": {"constant": "X"}}, "parameters.x.targets.geometry: must be a non-empty"),
        ({"simulation": {"samples": ["gamma"]}}, "targets.simulation: missing required key(s)"),
        ({"simulation": {"option": "a", "sample": "g"}}, "unknown key(s) ['sample']"),
        ({"evaluators": {}}, "targets.evaluators: must name an evaluator binding and the setting"),
        ({"evaluators": "calo.photon.threshold"}, "targets.evaluators: must be a mapping"),
        (
            {"evaluators": {"other": "y"}},
            "parameters.x.targets.evaluators.other: the problem has no evaluator 'other'",
        ),
        (
            {"simulation": {"option": "physics.rangecut", "samples": ["zz"]}},
            "parameters.x.targets.simulation.samples[0]: the problem has no sample 'zz'",
        ),
        (
            {"simulation": {"option": "x", "samples": ["gamma", "gamma"]}},
            "'gamma' is listed twice",
        ),
        ({"geometry": ["X", 5]}, "parameters.x.targets.geometry[1]: must be a non-empty string"),
    ],
)
def test_target_errors(doc: dict[str, Any], targets: object, message: str) -> None:
    reject(add(doc, "x", targets), message)


def test_a_stage_given_as_a_file_has_no_options_to_set(doc: dict[str, Any]) -> None:
    doc["samples"]["gamma"]["generation"] = {"file": "events.hepmc"}
    reject(
        add(doc, "e", {"generation": "energy"}, unit="GeV"),
        "parameters.e.targets.generation: samples.gamma.generation is given as a file, which is "
        "not run",
    )


# --- a destination is set once ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda d: add(d, "again", {"geometry": "constants.ECal_cell_size"}),
            "parameters.again.targets.geometry: the geometry's constants.ECal_cell_size is set by "
            "parameters.ecal_cell_size already",
        ),
        (
            lambda d: d["parameters"]["ecal_cell_size"].update(
                targets={"geometry": ["ECal_cell_size", "constants.ECal_cell_size"]}
            ),
            "parameters.ecal_cell_size.targets.geometry[1]: the geometry's "
            "constants.ECal_cell_size is set by this parameter already",
        ),
        (
            lambda d: add(
                add(d, "a", {"simulation": "physics.rangecut"}),
                "b",
                {"simulation": {"option": "physics.rangecut", "samples": ["gamma"]}},
            ),
            "parameters.b.targets.simulation: samples.gamma.simulation.physics.rangecut is set by "
            "parameters.a already",
        ),
        (
            lambda d: add(d, "a", {"simulation": ["physics.rangecut", "physics.rangecut"]}),
            "parameters.a.targets.simulation[1]: samples.gamma.simulation.physics.rangecut is set "
            "by this parameter already",
        ),
        (
            lambda d: add(
                add(d, "a", {"evaluators": {EVALUATOR: "y"}}), "b", {"evaluators": {EVALUATOR: "y"}}
            ),
            f"the setting y of evaluator {EVALUATOR!r} is set by parameters.a already",
        ),
        (
            lambda d: add(d, "pl", {"simulation": "physics.list"}),
            "samples.gamma.simulation.physics.list: is set by parameters.pl; leave it out here",
        ),
        (
            lambda d: add(d, "c", {"evaluators": {EVALUATOR: "collection"}}),
            "evaluators[0].settings.collection: is set by parameters.c; leave it out here",
        ),
    ],
)
def test_a_destination_is_set_once(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    edit(doc)
    reject(doc, message)


@pytest.mark.parametrize(
    ("stage", "key"),
    [("simulation", "seed"), ("simulation", "events"), ("simulation", "timeout")],
)
def test_a_key_of_the_stage_itself_is_no_option_a_parameter_sets(
    doc: dict[str, Any], stage: str, key: str
) -> None:
    # as an option, its definition would record the parameter's value, its run use the stage's
    reject(
        add(doc, "k", {stage: key}, type="integer", bounds=[1, 100]),
        f"parameters.k.targets.{stage}: {key!r} is a key of the stage itself, not an option of "
        "its executor: a parameter cannot set it",
    )


def test_an_option_written_in_the_shared_stages_points_there(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    doc["stages"] = {"simulation": doc["samples"]["gamma"].pop("simulation")}
    del doc["samples"]["pion"]["simulation"]
    targets = {"simulation": {"option": "physics.list", "samples": ["gamma"]}}
    reject(
        add(doc, "pl", targets, type="categorical", choices=["FTFP_BERT", "QGSP_BERT"]),
        "stages.simulation.physics.list: is set by parameters.pl; leave it out here",
    )


# --- what each target takes ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("targets", "domain", "message"),
    [
        (
            {"generation": "energy"},
            {"unit": "keV"},
            "parameters.x.targets.generation: option 'energy' takes a quantity in MeV, GeV, TeV: "
            "the parameter must be continuous or integer, with one of these units, not continuous "
            "in keV",
        ),
        ({"generation": "energy"}, {}, "not continuous"),
        (
            {"generation": "particle"},
            {},
            "option 'particle' takes a name: the parameter must be categorical, with string",
        ),
        (
            {"generation": "direction"},
            {},
            "parameters.x.targets.generation: gun option 'direction': a parameter can set only the "
            "particle and energy of the particle gun",
        ),
        (
            {"simulation": "numberOfThreads"},
            {"type": "integer", "bounds": [1, 4]},
            "it says how many processors the simulation keeps busy",
        ),
        ({"simulation": "physics.pdgfile"}, {}, "it names a file"),
        ({"simulation": "enableDetailedShowerMode"}, {}, "it is a flag"),
        ({"simulation": "action.run"}, {}, "it takes a list of values"),
        ({"simulation": "numberOfEvents"}, {}, "an option that k4opt sets"),
        ({"simulation": "steeringFile"}, {}, "--steeringFile is not supported"),
        ({"simulation": "physics.rangecut"}, {"unit": "s"}, "unit 's' cannot"),
        ({"geometry": "X"}, {"unit": "inch"}, "unit 'inch' cannot be written"),
    ],
)
def test_a_value_must_fit_its_target(
    doc: dict[str, Any], targets: dict[str, str], domain: dict[str, object], message: str
) -> None:
    if "generation" in targets:  # the gun gives it no more: the parameter does
        doc["samples"]["gamma"]["generation"].pop(targets["generation"], None)
    reject(add(doc, "x", targets, **domain), message)


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
    reject(
        add(doc, "p", {"generation": "particle"}, type="categorical", choices=["gamma", "-e"]),
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
    doc = add(doc, "range_cut", {"simulation": "physics.rangecut"}, unit="mm")
    targets = {"evaluators": {EVALUATOR: "threshold"}}
    return add(doc, "threshold", targets, bounds=[30.0, 150.0], unit="keV")


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


def test_configure_sets_every_sample_and_entry_a_target_names(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    doc["evaluators"].append(doc["evaluators"][0] | {"samples": ["pion"]})
    doc["evaluators"][0]["samples"] = ["gamma"]
    for item in (*doc["objectives"], *doc["constraints"]):
        item["evaluator"] = EVALUATOR
    rangecut = {"option": "physics.rangecut", "samples": ["gamma", "pion"]}
    doc = add(doc, "cut", {"simulation": rangecut}, unit="mm")
    doc = add(doc, "t", {"evaluators": {EVALUATOR: "threshold"}})
    problem = parse_problem(doc)
    values, _ = validate_candidate(problem, {"ecal_cell_size": 5.0, "cut": 1.0, "t": 2.0})
    configuration = configure(problem, values)
    for sample in ("gamma", "pion"):
        options = configuration.samples[sample].stages["simulation"].options
        assert options["physics.rangecut"] == "1.0*mm"
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
    doc = add(doc, "threshold", {"evaluators": {EVALUATOR: "threshold"}}, unit="keV")
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
    for targets, represents in (
        ({"evaluators": {EVALUATOR: "threshold"}}, True),
        ({"geometry": "ECal_cell_size"}, False),
    ):
        doc["parameters"] = {}
        problem = parse_problem(add(doc, "x", targets))
        plan = plan_tasks(
            problem.samples,
            problem.evaluators,
            supplied,
            references=problem.references,
            parameters=problem.parameters,
        )
        assert represents_candidate(plan) is represents


# --- the examples ----------------------------------------------------------------------------


def test_the_example_of_every_kind_of_target_parses_and_fits_its_evaluator() -> None:
    path = EXAMPLES / "cld-ecal-bindings.yaml"
    problem = parse_problem(yaml.safe_load(path.read_text()), source=path)
    assert [(p.name, d.label) for p, d in problem.destinations] == [
        ("ecal_cell_size", "geometry constants.ECal_cell_size"),
        ("range_cut", "simulation physics.rangecut of gamma_10gev, gamma_50gev"),
        ("cell_threshold", "evaluator calo.photon: threshold"),
    ]
    threshold = problem.parameters[2]
    check_bound(calo_photon()(), "threshold", float, threshold.unit)  # keV into its GeV
    candidate = {"ecal_cell_size": 8.0, "range_cut": 0.5, "cell_threshold": 70.0}
    values, _ = validate_candidate(problem, candidate)
    configuration = configure(problem, values)
    for sample in ("gamma_10gev", "gamma_50gev"):
        options = configuration.samples[sample].stages["simulation"].options
        assert options["physics.rangecut"] == "0.5*mm"
    assert configuration.evaluators[0].settings["threshold"] == {"value": 70.0, "unit": "keV"}


def test_the_cell_size_of_the_pi0_example_reaches_the_geometry_and_the_readout() -> None:
    path = EXAMPLES / "cld-ecal-pi0-rejection.yaml"
    problem = parse_problem(yaml.safe_load(path.read_text()), source=path)
    (cell,) = problem.parameters
    assert cell.targets == (
        GeometryDestination("ECal_cell_size"),
        SettingDestination("readout", "cell_size"),
    )
