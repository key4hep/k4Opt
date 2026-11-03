"""The samples of a problem document, their stages and the stages they share: parsed, and
checked against each other and the evaluators that measure them (no Key4hep)."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.problem import ProblemConfigError, parse_problem
from tests.support.problems import two_samples

Write = Callable[..., Path]
EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "problem" / "cld-ecal-granularity.yaml"


@pytest.fixture
def write(tmp_path: Path) -> Write:
    """Write text/bytes to a file under ``tmp_path`` and return its path."""

    def _write(content: bytes | str, name: str = "problem.yaml") -> Path:
        path = tmp_path / name
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    return _write


def reject(data: object, message: str) -> None:
    """Assert that ``data`` is refused with ``message`` (a literal fragment)."""
    with pytest.raises(ProblemConfigError, match=re.escape(message)):
        parse_problem(data)


# --- stages ----------------------------------------------------------------------------


def test_stages_are_kept_in_pipeline_order_and_round_trip(doc: dict[str, Any]) -> None:
    doc["samples"]["gamma"] = dict(reversed(list(doc["samples"]["gamma"].items())))
    problem = parse_problem(doc)
    stages = problem.samples["gamma"].stages
    assert list(stages) == ["generation", "simulation"]
    generation = stages["generation"]
    assert (generation.events, generation.seed, generation.file) == (None, None, None)
    assert problem.to_dict()["samples"]["gamma"]["simulation"] == {  # type: ignore[index]
        "executor": "ddsim",
        "events": 1000,
        "seed": 42,
        "physics.list": "FTFP_BERT",
    }
    assert parse_problem(problem.to_dict()) == problem


def test_ddsim_options_keep_their_order_and_values(doc: dict[str, Any]) -> None:
    options = {
        "physics.list": "QGSP_BERT",
        "part.keepAllParticles": True,
        "action.run": ["A", "B"],
        "printLevel": 3,
    }
    doc["samples"]["gamma"]["simulation"].update(options)
    simulation = parse_problem(doc).samples["gamma"].stages["simulation"]
    assert list(simulation.options) == list(options)
    assert simulation.options["action.run"] == ("A", "B")


def test_a_generation_file_replaces_the_gun(doc: dict[str, Any]) -> None:
    doc["samples"]["gamma"]["generation"] = {"file": "events/zh.hepmc"}
    problem = parse_problem(doc)
    generation = problem.samples["gamma"].stages["generation"]
    assert (generation.executor, generation.file, generation.options) == (
        None,
        "events/zh.hepmc",
        {},
    )
    generation_dict = problem.to_dict()["samples"]["gamma"]["generation"]  # type: ignore[index]
    assert generation_dict == {"file": "events/zh.hepmc"}
    assert parse_problem(problem.to_dict()) == problem


def test_an_evaluator_may_read_only_the_generation(doc: dict[str, Any]) -> None:
    del doc["samples"]["gamma"]["simulation"]
    doc["evaluators"][0]["stage"] = "generation"
    assert list(parse_problem(doc).samples["gamma"].stages) == ["generation"]


def test_parsing_never_runs_a_subprocess(
    doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess started")

    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("os.system", boom)
    doc["samples"]["gamma"]["simulation"].update(
        {"physics.list": "; rm -rf /", "printLevel": "$(whoami)"}
    )
    parse_problem(doc)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda s: s.update(reconstruction={"executor": "k4run"}),
            "samples.gamma: unknown key(s) ['reconstruction']; allowed: ['generation', "
            "'properties', 'simulation']",
        ),
        (
            lambda s: s.update(simulation="ddsim"),
            "samples.gamma.simulation: must be a mapping, got str",
        ),
        (
            lambda s: s["simulation"].pop("executor"),
            "samples.gamma.simulation: needs exactly one of 'executor' (k4opt runs it) and 'file'",
        ),
        (
            lambda s: s["generation"].update(file="zh.hepmc"),
            "samples.gamma.generation: needs exactly one",
        ),
        (
            lambda s: s.update(generation={"file": "zh.hepmc", "particle": "e-"}),
            "samples.gamma.generation: a stage given as a file is not run, so it takes no "
            "['particle']",
        ),
        (
            lambda s: s.update(generation={"file": ""}),
            "samples.gamma.generation.file: must be a non-empty",
        ),
        (
            lambda s: s["simulation"].update(executor="geant4"),
            "samples.gamma.simulation.executor: must be one of ['ddsim'], got 'geant4'",
        ),
        (
            lambda s: s["generation"].update(executor="ddsim"),
            "samples.gamma.generation.executor: must be one of ['gun'], got 'ddsim'",
        ),
        (
            lambda s: s["simulation"].pop("events"),
            "samples.gamma.simulation: missing required key 'events': how many events ddsim runs",
        ),
        (
            lambda s: s["simulation"].update(events=0),
            "samples.gamma.simulation.events: must be >= 1",
        ),
        (lambda s: s["simulation"].update(events=True), "events: must be an integer, got bool"),
        (lambda s: s["simulation"].update(events=1.5), "events: must be an integer, got float"),
        (
            lambda s: s["simulation"].update(seed=True),
            "simulation.seed: must be an integer, got bool",
        ),
        (lambda s: s["simulation"].update(seed="42"), "seed: must be an integer, got str"),
        (
            lambda s: s["generation"].update(events=10),
            "samples.gamma.generation.events: gun runs inside samples.gamma.simulation, which sets "
            "the events; "
            "set samples.gamma.simulation.events instead",
        ),
        (
            lambda s: s["generation"].update(seed=1),
            "samples.gamma.generation.seed: gun runs inside samples.gamma.simulation, which sets "
            "the seed",
        ),
        (
            lambda s: s["generation"].update(charge=-1),
            "samples.gamma.generation: unknown key(s) ['charge']",
        ),
        (
            lambda s: s["generation"].pop("particle"),
            "samples.gamma.generation: missing required key(s) ['particle']",
        ),
        (
            lambda s: s["generation"]["energy"].update(unit="keV"),
            "samples.gamma.generation.energy.unit: unit must be one of",
        ),
        (
            lambda s: s["simulation"].update({"random.seed": 1}),
            "samples.gamma.simulation.random.seed: is an option that k4opt sets itself: set seed "
            "instead",
        ),
        (
            lambda s: s["simulation"].update(steeringFile="steer.py"),
            "samples.gamma.simulation.steeringFile: --steeringFile is not supported",
        ),
        (
            lambda s: s["simulation"].update(physics={"list": "QGSP_BERT"}),
            "samples.gamma.simulation.physics: write each option under its full ddsim name",
        ),
        (
            lambda s: s["simulation"].update(enableDetailedShowerMode=False),
            "samples.gamma.simulation.enableDetailedShowerMode: ddsim cannot switch a flag off",
        ),
        (
            lambda s: s["simulation"].update({"physics.list": {1, 2}}),
            "samples.gamma.simulation.physics.list: unsupported value",
        ),
    ],
)
def test_stage_errors(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    edit(doc["samples"]["gamma"])
    reject(doc, message)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda d: d["evaluators"][0].update(stage="reconstruction"),
            "evaluators[0].stage: must be one of ['generation', 'simulation'], "
            "got 'reconstruction'",
        ),
        (
            lambda d: d["samples"]["gamma"].pop("simulation"),
            "evaluators[0].stage: sample gamma defines no simulation stage; add "
            "samples.gamma.simulation",
        ),
        (
            lambda d: d["samples"]["gamma"].pop("generation"),
            "samples.gamma.simulation: simulation reads generation: add samples.gamma.generation "
            "with executor: "
            "gun, or give its output as file:",
        ),
        (
            lambda d: d["samples"]["gamma"].update(simulation={"file": "events.edm4hep.root"}),
            "samples.gamma.simulation.file: parameters.ecal_cell_size reach this stage, but this "
            "file was not made from a candidate's values",
        ),
        (
            lambda d: d["samples"]["gamma"].update(simulation={"file": "events.edm4hep.root"}),
            "use k4opt evaluate --input gamma/simulation=FILE",
        ),
        (
            lambda d: d["evaluators"][0].update(stage="generation"),
            "samples.gamma.simulation: is not used: no evaluator of the sample reads it, no "
            "role, and no stage that runs",
        ),
        (
            lambda d: d["evaluators"][0].pop("stage"),
            "samples.gamma: is measured by no evaluator; list it in the samples of one, have a "
            "role read one of its stages, or remove it",
        ),
    ],
)
def test_the_stages_fit_together_and_fit_the_evaluators(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    edit(doc)
    reject(doc, message)


# --- samples ---------------------------------------------------------------------------


def test_an_evaluator_measures_each_sample_it_lists(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    doc["samples"] = dict(reversed(list(doc["samples"].items())))  # the document's order is kept
    doc["objectives"].append({"sample": "pion", "stage": "simulation", "metric": "a.rms"})
    doc["objectives"][-1]["direction"] = "maximize"
    problem = parse_problem(doc)
    assert list(problem.samples) == ["pion", "gamma"]
    assert problem.samples["pion"].stages["generation"].options["particle"] == "pi-"
    assert problem.evaluators[0].samples == ("gamma", "pion")
    # the same metric of one evaluator on two samples: two objectives, not a duplicate
    assert [o.label for o in problem.objectives] == [
        "gamma/calo.single_particle_shower/simulation/a.rms",
        "pion/calo.single_particle_shower/simulation/a.rms",
    ]
    assert parse_problem(problem.to_dict()) == problem


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("Gamma", "samples: sample name 'Gamma' must start with a lowercase letter"),
        ("1gamma", "sample name '1gamma' must start"),
        ("gamma-10", "hold only lowercase letters, digits and '_', at most 64 characters"),
        ("gamma.10", "sample name 'gamma.10' must start"),
        ("g/a", "sample name 'g/a' must start"),
        ("a" * 65, "at most 64 characters"),
    ],
)
def test_sample_names_are_lowercase_identifiers(
    doc: dict[str, Any], name: str, message: str
) -> None:
    doc["samples"] = {name: doc["samples"]["gamma"]}
    reject(doc, message)


def test_a_sample_name_may_have_64_characters(doc: dict[str, Any]) -> None:
    name = "g" + "0_" * 31 + "x"
    doc["samples"] = {name: doc["samples"]["gamma"]}
    assert list(parse_problem(doc).samples) == [name]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["samples"].update(gamma={}), "samples.gamma: must define its stages"),
        (lambda d: d["samples"].update(gamma="ddsim"), "samples.gamma: must be a mapping"),
        (
            lambda d: d["evaluators"][0].update(samples=["pion"]),
            "evaluators[0].samples[0]: the problem has no sample 'pion'; its samples are gamma",
        ),
        (
            lambda d: d["evaluators"][0].update(samples=["gamma", "gamma"]),
            "evaluators[0].samples[1]: 'gamma' is listed twice",
        ),
        (
            lambda d: d["evaluators"][0].update(samples=[]),
            "evaluators[0].samples: must list at least one sample",
        ),
        (
            lambda d: d["evaluators"][0].update(samples="gamma"),
            "evaluators[0].samples: must be a list, got str",
        ),
        (
            lambda d: d["evaluators"].append(
                {"name": "cost", "version": "1", "samples": ["gamma"]}
            ),
            "evaluators[1].samples: an evaluator measures samples at the stage it reads; add "
            "that stage, or remove samples to measure the candidate",
        ),
        (
            lambda d: d.pop("samples"),
            "evaluators[0]: it reads the simulation stage of a sample, but the problem has no "
            "samples",
        ),
    ],
)
def test_sample_errors(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    edit(doc)
    reject(doc, message)


def test_an_evaluator_of_several_samples_must_list_those_it_measures(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    del doc["evaluators"][0]["samples"]
    reject(
        doc,
        "evaluators[0]: the problem has the samples gamma, pion; add samples: with those this "
        "evaluator measures",
    )


def test_every_sample_is_measured_and_has_the_stage_its_evaluators_read(
    doc: dict[str, Any],
) -> None:
    doc = two_samples(doc)
    doc["evaluators"][0]["samples"] = ["gamma"]
    reject(doc, "samples.pion: is measured by no evaluator")
    doc["evaluators"][0]["samples"] = ["gamma", "pion"]
    del doc["samples"]["pion"]["simulation"]
    reject(doc, "evaluators[0].stage: sample pion defines no simulation stage")


def test_the_stages_of_each_sample_are_checked_on_their_own(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    del doc["samples"]["pion"]["generation"]
    reject(doc, "samples.pion.simulation: simulation reads generation: add samples.pion.generation")
    doc = two_samples(doc)
    doc["samples"]["pion"]["generation"] = {"executor": "gun", "particle": "pi-", "seed": 1}
    reject(doc, "samples.pion.generation.seed: gun runs inside samples.pion.simulation")


def test_one_binding_may_measure_samples_with_other_settings(doc: dict[str, Any]) -> None:
    doc = two_samples(doc)
    first = doc["evaluators"][0]
    first["samples"] = ["gamma"]
    doc["evaluators"].append(first | {"samples": ["pion"], "settings": {"collection": "Other"}})
    doc["objectives"].append(
        {"sample": "pion", "stage": "simulation", "metric": "a.rms", "direction": "maximize"}
    )
    problem = parse_problem(doc)
    assert [(e.binding, e.samples) for e in problem.evaluators] == [
        ("calo.single_particle_shower", ("gamma",)),
        ("calo.single_particle_shower", ("pion",)),
    ]
    assert [o.sample for o in problem.objectives] == ["gamma", "pion"]
    assert parse_problem(problem.to_dict()) == problem


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"samples": ["pion", "gamma"]},
            "evaluators[1]: 'calo.single_particle_shower' is already bound to gamma by "
            "evaluators[0]; name this one with as:",
        ),
        (
            {"version": "2"},
            "evaluators[1]: 'calo.single_particle_shower' binds calo.single_particle_shower "
            "version '1' at the simulation stage in evaluators[0], and one binding name binds "
            "one implementation, version and stage; name this one with as:",
        ),
        ({"stage": "generation"}, "one binding name binds one implementation, version and stage"),
        (
            {"name": "calo.other", "as": "calo.single_particle_shower"},
            "binds calo.single_particle_shower version '1'",
        ),
    ],
)
def test_a_binding_named_twice_binds_one_implementation_on_other_samples(
    doc: dict[str, Any], change: dict[str, Any], message: str
) -> None:
    doc = two_samples(doc)
    doc["evaluators"][0]["samples"] = ["gamma"]
    doc["evaluators"].append(doc["evaluators"][0] | {"samples": ["pion"]} | change)
    reject(doc, message)


@pytest.mark.parametrize(
    ("owner", "message"),
    [
        (
            {"sample": "pion"},
            "objectives[0].sample: evaluator 'calo.single_particle_shower' does not measure "
            "'pion'; it measures gamma",
        ),
        (
            {"evaluator": "cost", "sample": "gamma"},
            "objectives[0].sample: evaluator 'cost' measures the candidate, not a sample; remove "
            "sample:",
        ),
        ({"sample": ""}, "objectives[0].sample: must be a non-empty string"),
    ],
)
def test_an_objective_names_a_sample_its_evaluator_measures(
    doc: dict[str, Any], owner: dict[str, str], message: str
) -> None:
    doc = two_samples(doc)
    doc["evaluators"][0]["samples"] = ["gamma"]
    doc["evaluators"].append({"name": "cost", "version": "1"})
    doc["evaluators"].append(
        {"name": "calo.other", "version": "1", "stage": "simulation", "samples": ["pion"]}
    )
    doc["objectives"][0] = {"stage": "simulation", "metric": "a.rms", "direction": "minimize"}
    doc["objectives"][0] |= {"evaluator": "calo.single_particle_shower"} | owner
    reject(doc, message)


def test_an_objective_may_leave_out_what_the_problem_leaves_one_choice_for(
    doc: dict[str, Any],
) -> None:
    doc = two_samples(doc)
    doc["evaluators"][0]["samples"] = ["gamma"]
    doc["evaluators"].append(
        {"name": "calo.other", "version": "1", "stage": "simulation", "samples": ["pion"]}
    )
    doc["evaluators"].append({"name": "cost", "version": "1"})
    objective = {"stage": "simulation", "metric": "a.rms", "direction": "minimize"}
    doc["objectives"] = [
        objective | {"sample": "pion"},  # the sample's only evaluator
        objective | {"evaluator": "calo.single_particle_shower"},  # the evaluator's only sample
        objective | {"evaluator": "cost"},  # an evaluator of the candidate: no sample
    ]
    problem = parse_problem(doc)
    assert [(o.sample, o.evaluator) for o in problem.objectives] == [
        ("pion", "calo.other"),
        ("gamma", "calo.single_particle_shower"),
        (None, "cost"),
    ]
    assert [o.label for o in problem.objectives][2] == "cost/simulation/a.rms"
    assert parse_problem(problem.to_dict()) == problem


@pytest.mark.parametrize(
    ("owner", "message"),
    [
        (
            {"evaluator": "calo.single_particle_shower"},
            "objectives[0]: evaluator 'calo.single_particle_shower' measures the samples gamma, "
            "pion; add sample: with the one this metric is measured on",
        ),
        (
            {"sample": "gamma"},
            "objectives[0]: gamma is measured by the evaluators calo.single_particle_shower, "
            "tight; add evaluator: with one of them",
        ),
        (
            {},
            "objectives[0]: the problem has the evaluations gamma/calo.single_particle_shower, "
            "pion/calo.single_particle_shower, gamma/tight; add evaluator: (and sample:)",
        ),
        ({"sample": "neutron"}, "objectives[0].sample: no evaluator of the problem measures"),
    ],
)
def test_an_ambiguous_objective_is_refused_with_the_choices(
    doc: dict[str, Any], owner: dict[str, str], message: str
) -> None:
    doc = two_samples(doc)
    tight = doc["evaluators"][0] | {"as": "tight", "samples": ["gamma"]}
    doc["evaluators"].append(tight)
    doc["constraints"][0]["evaluator"] = "tight"
    doc["objectives"][0] = {"stage": "simulation", "metric": "a.rms", "direction": "minimize"}
    doc["objectives"][0] |= owner
    reject(doc, message)


# --- shared stages ---------------------------------------------------------------------


def shared(doc: dict[str, Any]) -> dict[str, Any]:
    """The two samples of :func:`two_samples`, written as shared ``stages`` that each sample
    completes with what differs: the particle and the seed."""
    doc = two_samples(doc)
    gamma = doc["samples"]["gamma"]
    doc["stages"] = {
        "generation": {k: v for k, v in gamma["generation"].items() if k != "particle"},
        "simulation": {k: v for k, v in gamma["simulation"].items() if k != "seed"},
    }
    doc["samples"] = {
        "gamma": {"generation": {"particle": "gamma"}, "simulation": {"seed": 42}},
        "pion": {"generation": {"particle": "pi-"}, "simulation": {"seed": 43}},
    }
    return doc


def test_samples_complete_the_shared_stages(doc: dict[str, Any]) -> None:
    explicit = parse_problem(two_samples(copy.deepcopy(doc)))
    problem = parse_problem(shared(doc))
    # parse-time only: the problem, its normalized form and so the study identity are those of
    # the samples written out in full
    assert problem == explicit
    assert problem.to_dict() == explicit.to_dict()
    assert "stages" not in problem.to_dict()
    pion = problem.samples["pion"].stages
    assert pion["generation"].options["particle"] == "pi-"
    assert pion["generation"].options["energy"] == {"value": 10, "unit": "GeV"}
    assert (pion["simulation"].events, pion["simulation"].seed) == (1000, 43)
    assert pion["simulation"].options == {"physics.list": "FTFP_BERT"}


def test_a_sample_takes_a_shared_stage_it_does_not_define(doc: dict[str, Any]) -> None:
    doc = shared(doc)
    doc["stages"]["simulation"]["seed"] = 7
    for sample in doc["samples"].values():
        del sample["simulation"]
    problem = parse_problem(doc)
    for sample in problem.samples.values():
        assert (sample.stages["simulation"].events, sample.stages["simulation"].seed) == (1000, 7)
    doc["samples"]["pion"] = {"properties": {"process": "pi-"}, **doc["samples"]["pion"]}
    assert parse_problem(doc).samples["pion"].properties == {"process": "pi-"}


def test_a_stage_naming_executor_or_file_replaces_the_shared_one(doc: dict[str, Any]) -> None:
    doc = shared(doc)
    doc["samples"]["pion"]["simulation"] = {"executor": "ddsim", "events": 10}
    doc["samples"]["pion"]["generation"] = {"file": "events/pion.hepmc"}
    problem = parse_problem(doc)
    pion = problem.samples["pion"].stages
    assert pion["simulation"].options == {}  # physics.list is not taken from stages.simulation
    assert (pion["simulation"].events, pion["simulation"].seed) == (10, None)
    assert pion["generation"].file == "events/pion.hepmc"
    assert problem.samples["gamma"].stages["simulation"].options == {"physics.list": "FTFP_BERT"}


def test_a_nested_value_replaces_the_shared_one_whole(doc: dict[str, Any]) -> None:
    doc = shared(doc)
    doc["samples"]["pion"]["generation"]["direction"] = {
        "distribution": "uniform",
        "theta": [10, 20],
        "phi": [0, 90],
        "unit": "deg",
    }
    direction = parse_problem(doc).samples["pion"].stages["generation"].options["direction"]
    assert direction == {
        "distribution": "uniform",
        "theta": (10, 20),
        "phi": (0, 90),
        "unit": "deg",
    }
    doc["samples"]["pion"]["generation"]["energy"] = {"value": 50}  # no unit: not merged
    reject(doc, "samples.pion.generation.energy: missing required key(s) ['unit']")


def takes_shared_simulation(doc: dict[str, Any]) -> None:
    """Every sample of ``doc`` takes ``stages.simulation`` as it is."""
    for sample in doc["samples"].values():
        del sample["simulation"]


def replaces_simulation(doc: dict[str, Any]) -> None:
    """Every sample of ``doc`` defines a simulation of its own."""
    for sample in doc["samples"].values():
        sample["simulation"] = {"executor": "ddsim", "events": 5}


def has_no_generation(doc: dict[str, Any]) -> None:
    """Neither ``stages`` nor any sample of ``doc`` defines a generation."""
    del doc["stages"]["generation"]
    for sample in doc["samples"].values():
        del sample["generation"]


def takes_shared_simulation_without_events(doc: dict[str, Any]) -> None:
    takes_shared_simulation(doc)
    del doc["stages"]["simulation"]["events"]


def takes_shared_simulation_file(doc: dict[str, Any]) -> None:
    takes_shared_simulation(doc)
    doc["stages"]["simulation"] = {"file": "events.edm4hep.root"}


@pytest.mark.parametrize(
    ("file", "particles", "energies", "seeds", "events"),
    [
        ("cld-ecal-photon-energies.yaml", ["gamma", "gamma"], [10, 50], [42, 43], 1000),
        ("cld-ecal-pi0-rejection.yaml", ["gamma", "pi0"], [50, 50], [1, 2], 500),
    ],
)
def test_the_checked_in_examples_share_their_stages(
    file: str, particles: list[str], energies: list[int], seeds: list[int], events: int
) -> None:
    example = EXAMPLE.with_name(file)
    assert "stages" in yaml.safe_load(example.read_text())
    problem = parse_problem(yaml.safe_load(example.read_text()), source=example)
    generation = [s.stages["generation"] for s in problem.samples.values()]
    simulation = [s.stages["simulation"] for s in problem.samples.values()]
    assert [g.options["particle"] for g in generation] == particles
    assert [g.options["energy"]["value"] for g in generation] == energies  # type: ignore[index]
    assert [s.seed for s in simulation] == seeds
    assert {(s.events, s.options["physics.list"]) for s in simulation} == {(events, "FTFP_BERT")}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(stages={}), "stages: must name at least one stage"),
        (lambda d: d.update(stages=[]), "stages: must be a mapping, got list"),
        (
            lambda d: d["stages"].update(reconstruction={"executor": "k4run"}),
            "stages: unknown key(s) ['reconstruction']",
        ),
        (
            lambda d: d["stages"]["simulation"].pop("executor"),
            "stages.simulation: needs exactly one of 'executor' and 'file'",
        ),
        (
            lambda d: d["stages"]["simulation"].update(executor="geant4"),
            "stages.simulation.executor: must be one of ['ddsim'], got 'geant4'",
        ),
        (
            lambda d: d["stages"].update(generation={"file": "a.hepmc", "particle": "gamma"}),
            "stages.generation: a stage given as a file is not run, so it takes no ['particle']",
        ),
        (
            lambda d: d.pop("samples"),
            "stages: there are no samples to take these stages; add samples:",
        ),
        (
            replaces_simulation,
            "stages.simulation: is not used: every sample replaces it",
        ),
        (
            # a partial stage with nothing to complete
            lambda d: d["stages"].pop("simulation"),
            "samples.gamma.simulation: needs exactly one of 'executor' (k4opt runs it) and 'file' "
            "(its output exists), or a stages.simulation for it to complete",
        ),
        (
            lambda d: d["stages"].update(generation={"file": "a.hepmc"}),
            "samples.gamma.generation: stages.generation gives a file, which takes no other keys; "
            "define the whole stage here, with executor: or file:",
        ),
        # a key still missing after completion names both places
        (
            lambda d: d["stages"]["simulation"].pop("events"),
            "samples.gamma.simulation (completing stages.simulation): missing required key "
            "'events'",
        ),
        (
            takes_shared_simulation_without_events,
            "samples.gamma.simulation (from stages.simulation): missing required key 'events'",
        ),
        # an invalid key is found where it was written
        (
            lambda d: d["stages"]["simulation"].update(events=0),
            "stages.simulation.events: must be >= 1, got 0",
        ),
        (
            lambda d: d["samples"]["pion"]["simulation"].update(seed="x"),
            "samples.pion.simulation.seed: must be an integer",
        ),
        (
            lambda d: d["stages"]["simulation"].update({"physics.list": ["a", "b"]}),
            "stages.simulation.physics.list: ddsim reads a single value for this option",
        ),
        (
            lambda d: d["samples"]["pion"]["simulation"].update({"physics.list": ["a", "b"]}),
            "samples.pion.simulation.physics.list: ddsim reads a single value for this option",
        ),
        (
            lambda d: d["stages"]["simulation"].update({"physics.list": float("nan")}),
            "stages.simulation.physics.list: non-finite number nan is not allowed",
        ),
        (
            lambda d: d["stages"]["generation"]["energy"].update(value=float("inf")),
            "stages.generation.energy.value: non-finite number inf is not allowed",
        ),
        (
            lambda d: d["samples"]["pion"]["simulation"].update({"physics.list": float("nan")}),
            "samples.pion.simulation.physics.list: non-finite number nan is not allowed",
        ),
        (
            lambda d: d["stages"]["generation"]["energy"].update(value=-1),
            "stages.generation.energy.value: must be positive, got -1",
        ),
        (
            lambda d: d["stages"]["generation"].pop("energy"),
            "samples.gamma.generation (completing stages.generation): missing required key(s) "
            "['energy']",
        ),
        (
            lambda d: d["stages"]["generation"].update(seed=1),
            "stages.generation.seed: gun runs inside each sample's simulation, which sets the "
            "seed; set stages.simulation.seed instead",
        ),
        (
            has_no_generation,
            "samples.gamma.simulation (completing stages.simulation): simulation reads generation: "
            "add samples.gamma.generation",
        ),
        (
            takes_shared_simulation_file,
            "stages.simulation.file: parameters.ecal_cell_size reach this stage",
        ),
    ],
)
def test_shared_stage_errors(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    doc = shared(doc)
    edit(doc)
    reject(doc, message)


@pytest.mark.parametrize("sharing", [True, False])
def test_a_stage_written_as_null_is_refused_not_taken_from_stages(
    doc: dict[str, Any], sharing: bool
) -> None:
    # null is a stage written wrong, not a stage left out: it neither takes stages.generation
    # nor crashes without one
    doc = shared(doc) if sharing else two_samples(doc)
    if sharing:
        doc["stages"]["generation"]["particle"] = "pi-"  # complete, so null would pass unseen
    doc["samples"]["pion"]["generation"] = None
    reject(doc, "samples.pion.generation: must be a mapping, got NoneType")


def test_a_shared_stage_a_sample_does_not_use_says_where_it_comes_from(
    doc: dict[str, Any],
) -> None:
    doc = shared(doc)
    # zh reads an event file; nothing simulates it, yet it takes stages.simulation
    doc["samples"]["zh"] = {"generation": {"file": "events/zh.hepmc"}}
    doc["evaluators"].append(
        {"name": "test.file_size", "version": "1", "stage": "generation", "samples": ["zh"]}
    )
    reject(
        doc,
        "samples.zh.simulation (from stages.simulation): is not used: no evaluator of the sample "
        "reads it, no role, and no stage that runs; every sample takes stages.simulation, so "
        "define simulation in the samples that use it instead",
    )
