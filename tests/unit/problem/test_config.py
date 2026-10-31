"""Loading and validating problem documents."""

from __future__ import annotations

import datetime
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt import ConfigError, K4OptError
from k4opt.problem import ProblemConfigError, load_problem, parse_problem

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


# --- happy paths -----------------------------------------------------------------------


def test_loads_the_checked_in_example() -> None:
    # the CLD geometry is not part of the repository, so only the parse is checked
    problem = parse_problem(yaml.safe_load(EXAMPLE.read_text()), source=EXAMPLE)
    assert problem.name == "cld-ecal-granularity"
    assert problem.environment.key4hep.channel == "nightly"
    assert problem.environment.key4hep.version == "2026-10-20"
    assert problem.geometry == "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml"
    (parameter,) = problem.parameters
    assert parameter.geometry.constant == "ECal_cell_size"
    assert parameter.geometry.canonical_target == "constants.ECal_cell_size"
    assert (parameter.kind, parameter.bounds, parameter.unit) == ("continuous", (2.5, 20.0), "mm")
    assert [(e.name, e.version, e.stage) for e in problem.evaluators] == [
        ("calo.photon", "1", "simulation")
    ]
    assert problem.evaluators[0].collections == {"hits": "ECalBarrelCollection"}
    assert problem.evaluators[0].settings == {"threshold": {"value": 70, "unit": "keV"}}
    generation, simulation = problem.stages.values()
    assert (generation.type, generation.executor) == ("generation", "gun")
    generator: Any = generation.options
    assert generator["particle"] == "gamma"
    assert generator["direction"]["theta"] == (80, 100)
    assert (simulation.executor, simulation.events, simulation.seed) == ("ddsim", 1000, 42)
    assert simulation.options == {"physics.list": "FTFP_BERT"}
    assert [(o.stage, o.metric, o.direction) for o in problem.objectives] == [
        ("simulation", "ecal_barrel.angular_residual", "minimize"),
        ("simulation", "ecal_barrel.fired_cells", "minimize"),
    ]
    assert problem.constraints == ()
    assert parse_problem(problem.to_dict()) == problem


def test_parses_in_memory_data_without_a_file(doc: dict[str, Any]) -> None:
    assert parse_problem(doc).name == "demo"


def test_name_defaults_to_the_file_stem(doc: dict[str, Any], tmp_path: Path) -> None:
    del doc["name"]
    assert parse_problem(doc).name is None
    assert parse_problem(doc, source=tmp_path / "my-study.yaml").name == "my-study"


@pytest.mark.parametrize("channel", ["release", "nightly"])
def test_accepts_both_channels(doc: dict[str, Any], channel: str) -> None:
    doc["environment"]["key4hep"] = {"channel": channel, "version": "2026-04-08"}
    assert parse_problem(doc).environment.key4hep.channel == channel


def test_minimal_document(doc: dict[str, Any]) -> None:
    # an evaluator that reads no stage needs no stages
    for key in ("name", "constraints", "stages"):
        del doc[key]
    del doc["evaluators"][0]["settings"], doc["evaluators"][0]["stage"]
    problem = parse_problem(doc)
    assert problem.stages == {}
    assert not problem.evaluators[0].settings and problem.evaluators[0].stage is None
    assert problem.constraints == ()
    assert "stages" not in problem.to_dict()


def test_integer_and_categorical_parameters(doc: dict[str, Any]) -> None:
    doc["parameters"] += [
        {"geometry": {"target": "T.layer[id=1].n"}, "type": "integer", "bounds": [60, 120]},
        {"geometry": {"constant": "Mode"}, "type": "categorical", "choices": ["a", "b"]},
        {"geometry": {"constant": "Flag"}, "type": "categorical", "choices": [True, False]},
        {"geometry": {"constant": "Num"}, "type": "categorical", "choices": [1, 2.5]},
    ]
    kinds = [(p.kind, p.bounds, p.choices) for p in parse_problem(doc).parameters[1:]]
    assert kinds == [
        ("integer", (60, 120), None),
        ("categorical", None, ("a", "b")),
        ("categorical", None, (True, False)),
        ("categorical", None, (1, 2.5)),
    ]


def test_constraints_with_min_max_or_both(doc: dict[str, Any]) -> None:
    doc["constraints"] = [
        {"stage": "s", "metric": "a", "min": 1},
        {"stage": "s", "metric": "b", "max": 2.5, "unit": "mm"},
        {"stage": "s", "metric": "c", "min": 1, "max": 1},
    ]
    assert [(c.min, c.max) for c in parse_problem(doc).constraints] == [
        (1, None),
        (None, 2.5),
        (1, 1),
    ]


def test_objective_and_constraint_may_share_a_metric(doc: dict[str, Any]) -> None:
    doc["constraints"] = [{"stage": "simulation", "metric": "a.rms", "max": 2}]
    assert parse_problem(doc).constraints[0].metric == "a.rms"


def test_several_objectives_keep_their_order(doc: dict[str, Any]) -> None:
    doc["objectives"].append({"stage": "analysis", "metric": "x", "direction": "maximize"})
    assert [o.metric for o in parse_problem(doc).objectives] == ["a.rms", "x"]


# --- stages ----------------------------------------------------------------------------


def test_stages_are_kept_in_pipeline_order_and_round_trip(doc: dict[str, Any]) -> None:
    doc["stages"] = dict(reversed(list(doc["stages"].items())))
    problem = parse_problem(doc)
    assert list(problem.stages) == ["generation", "simulation"]
    generation = problem.stages["generation"]
    assert (generation.events, generation.seed, generation.file) == (None, None, None)
    assert problem.to_dict()["stages"]["simulation"] == {
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
    doc["stages"]["simulation"].update(options)
    simulation = parse_problem(doc).stages["simulation"]
    assert list(simulation.options) == list(options)
    assert simulation.options["action.run"] == ("A", "B")


def test_a_generation_file_replaces_the_gun(doc: dict[str, Any]) -> None:
    doc["stages"]["generation"] = {"file": "events/zh.hepmc"}
    problem = parse_problem(doc)
    generation = problem.stages["generation"]
    assert (generation.executor, generation.file, generation.options) == (
        None,
        "events/zh.hepmc",
        {},
    )
    assert problem.to_dict()["stages"]["generation"] == {"file": "events/zh.hepmc"}
    assert parse_problem(problem.to_dict()) == problem


def test_an_evaluator_may_read_only_the_generation(doc: dict[str, Any]) -> None:
    del doc["stages"]["simulation"]
    doc["evaluators"][0]["stage"] = "generation"
    assert list(parse_problem(doc).stages) == ["generation"]


def test_parsing_never_runs_a_subprocess(
    doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess started")

    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("os.system", boom)
    doc["stages"]["simulation"].update({"physics.list": "; rm -rf /", "printLevel": "$(whoami)"})
    parse_problem(doc)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda s: s.update(reconstruction={"executor": "k4run"}),
            "stages: unknown key(s) ['reconstruction']; allowed: ['generation', 'simulation']",
        ),
        (lambda s: s.update(simulation="ddsim"), "stages.simulation: must be a mapping, got str"),
        (
            lambda s: s["simulation"].pop("executor"),
            "stages.simulation: needs exactly one of 'executor' (k4opt runs it) and 'file'",
        ),
        (lambda s: s["generation"].update(file="zh.hepmc"), "stages.generation: needs exactly one"),
        (
            lambda s: s.update(generation={"file": "zh.hepmc", "particle": "e-"}),
            "stages.generation: a stage given as a file is not run, so it takes no ['particle']",
        ),
        (
            lambda s: s.update(generation={"file": ""}),
            "stages.generation.file: must be a non-empty",
        ),
        (
            lambda s: s["simulation"].update(executor="geant4"),
            "stages.simulation.executor: must be one of ['ddsim'], got 'geant4'",
        ),
        (
            lambda s: s["generation"].update(executor="ddsim"),
            "stages.generation.executor: must be one of ['gun'], got 'ddsim'",
        ),
        (
            lambda s: s["simulation"].pop("events"),
            "stages.simulation: missing required key 'events': how many events ddsim runs",
        ),
        (lambda s: s["simulation"].update(events=0), "stages.simulation.events: must be >= 1"),
        (lambda s: s["simulation"].update(events=True), "events: must be an integer, got bool"),
        (lambda s: s["simulation"].update(events=1.5), "events: must be an integer, got float"),
        (
            lambda s: s["simulation"].update(seed=True),
            "simulation.seed: must be an integer, got bool",
        ),
        (lambda s: s["simulation"].update(seed="42"), "seed: must be an integer, got str"),
        (
            lambda s: s["generation"].update(events=10),
            "stages.generation.events: gun runs inside stages.simulation, which sets the events; "
            "set stages.simulation.events instead",
        ),
        (
            lambda s: s["generation"].update(seed=1),
            "stages.generation.seed: gun runs inside stages.simulation, which sets the seed",
        ),
        (
            lambda s: s["generation"].update(charge=-1),
            "stages.generation: unknown key(s) ['charge']",
        ),
        (
            lambda s: s["generation"].pop("particle"),
            "stages.generation: missing required key(s) ['particle']",
        ),
        (
            lambda s: s["generation"]["energy"].update(unit="keV"),
            "stages.generation.energy.unit: unit must be one of",
        ),
        (
            lambda s: s["simulation"].update({"random.seed": 1}),
            "stages.simulation.random.seed: is an option that k4opt sets itself: set seed instead",
        ),
        (
            lambda s: s["simulation"].update(steeringFile="steer.py"),
            "stages.simulation.steeringFile: --steeringFile is not supported",
        ),
        (
            lambda s: s["simulation"].update(physics={"list": "QGSP_BERT"}),
            "stages.simulation.physics: write each option under its full ddsim name",
        ),
        (
            lambda s: s["simulation"].update(enableDetailedShowerMode=False),
            "stages.simulation.enableDetailedShowerMode: ddsim cannot switch a flag off",
        ),
        (
            lambda s: s["simulation"].update({"physics.list": {1, 2}}),
            "stages.simulation.physics.list: unsupported value",
        ),
    ],
)
def test_stage_errors(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    edit(doc["stages"])
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
            lambda d: d["stages"].pop("simulation"),
            "evaluators[0].stage: the problem defines no simulation stage; add stages.simulation",
        ),
        (
            lambda d: d["stages"].pop("generation"),
            "stages.simulation: simulation reads generation: add stages.generation with executor: "
            "gun, or give its output as file:",
        ),
        (
            lambda d: d["stages"].update(simulation={"file": "events.edm4hep.root"}),
            "stages.simulation.file: the parameters vary the geometry, but this file was not made "
            "with a candidate's geometry",
        ),
        (
            lambda d: d["stages"].update(simulation={"file": "events.edm4hep.root"}),
            "use k4opt evaluate --input simulation=FILE",
        ),
        (
            lambda d: d["evaluators"][0].update(stage="generation"),
            "stages.simulation: is not used: no evaluator reads it, and no stage that runs does",
        ),
        (lambda d: d["evaluators"][0].pop("stage"), "stages.generation: is not used"),
    ],
)
def test_the_stages_fit_together_and_fit_the_evaluators(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    edit(doc)
    reject(doc, message)


# --- schema errors ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "message"),
    [
        ("k4opt.problem/v2", "unsupported version 'k4opt.problem/v2'"),
        ("v1alpha1", "is not a k4opt problem schema"),
        (1, "schema_version: must be a string"),
        (None, "schema_version: must be a string"),
    ],
)
def test_schema_version_is_enforced(doc: dict[str, Any], version: object, message: str) -> None:
    doc["schema_version"] = version
    reject(doc, message)


def test_missing_schema_version(doc: dict[str, Any]) -> None:
    del doc["schema_version"]
    reject(doc, "missing required key(s) ['schema_version']")


def test_wrong_version_is_reported_before_unknown_keys(doc: dict[str, Any]) -> None:
    doc["schema_version"] = "k4opt.problem/v9"
    doc["new_in_v9"] = {}
    reject(doc, "unsupported version")


@pytest.mark.parametrize(
    "key", ["environment", "geometry", "parameters", "evaluators", "objectives"]
)
def test_missing_top_level_sections(doc: dict[str, Any], key: str) -> None:
    del doc[key]
    reject(doc, f"missing required key(s) ['{key}']")


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ((), "top level: unknown key(s) ['extra']"),
        (("environment",), "environment: unknown key(s) ['extra']"),
        (("environment", "key4hep"), "environment.key4hep: unknown key(s) ['extra']"),
        (("parameters", 0), "parameters[0]: unknown key(s) ['extra']"),
        (("parameters", 0, "geometry"), "parameters[0].geometry: unknown key(s) ['extra']"),
        (("stages",), "stages: unknown key(s) ['extra']"),
        (("stages", "generation"), "stages.generation: unknown key(s) ['extra']"),
        (("evaluators", 0), "evaluators[0]: unknown key(s) ['extra']"),
        (("objectives", 0), "objectives[0]: unknown key(s) ['extra']"),
        (("constraints", 0), "constraints[0]: unknown key(s) ['extra']"),
    ],
)
def test_unknown_keys_are_rejected_with_a_location(
    doc: dict[str, Any], path: tuple[str | int, ...], message: str
) -> None:
    node: Any = doc
    for step in path:
        node = node[step]
    node["extra"] = 1
    reject(doc, message)


def test_the_collections_of_an_evaluator_are_kept_by_role(doc: dict[str, Any]) -> None:
    doc["evaluators"][0]["collections"] = {"hits": "ECalBarrelCollection", "particles": "Truth"}
    problem = parse_problem(doc)
    assert problem.evaluators[0].collections == {
        "hits": "ECalBarrelCollection",
        "particles": "Truth",
    }
    assert problem.to_dict()["evaluators"][0]["collections"] == {  # type: ignore[index]
        "hits": "ECalBarrelCollection",
        "particles": "Truth",
    }
    assert parse_problem(problem.to_dict()) == problem


@pytest.mark.parametrize(
    ("collections", "message"),
    [
        (["ECalBarrelCollection"], "evaluators[0].collections: must be a mapping, got list"),
        ({"hits": ""}, "evaluators[0].collections.hits: must be a non-empty string"),
        ({"hits": 3}, "evaluators[0].collections.hits: must be a non-empty string"),
        ({1: "C"}, "evaluators[0].collections: keys must be strings"),
    ],
)
def test_collection_errors(doc: dict[str, Any], collections: object, message: str) -> None:
    doc["evaluators"][0]["collections"] = collections
    reject(doc, message)


def test_collections_need_the_stage_they_are_read_from(doc: dict[str, Any]) -> None:
    del doc["stages"], doc["evaluators"][0]["stage"]
    doc["evaluators"][0]["collections"] = {"hits": "ECalBarrelCollection"}
    reject(doc, "evaluators[0].collections: collections are read from the file of the stage")


def test_unknown_keys_inside_settings_are_allowed(doc: dict[str, Any]) -> None:
    doc["evaluators"][0]["settings"] = {"anything": {"goes": [1, "x", None, True]}}
    assert parse_problem(doc).evaluators[0].to_dict()["settings"] == {
        "anything": {"goes": [1, "x", None, True]}
    }


def test_several_evaluators_keep_their_order_and_own_settings(doc: dict[str, Any]) -> None:
    doc["evaluators"].append({"name": "detector.cost", "version": "2"})
    problem = parse_problem(doc)
    first, second = problem.evaluators
    assert (first.name, first.version, second.name, second.version) == (
        "calo.single_particle_shower",
        "1",
        "detector.cost",
        "2",
    )
    assert first.settings == {"collection": "ECalBarrelCollection"} and second.settings == {}
    assert parse_problem(problem.to_dict()) == problem


@pytest.mark.parametrize(
    ("evaluators", "message"),
    [
        ([], "evaluators: must list at least one evaluator"),
        ({"name": "x", "version": "1"}, "evaluators: must be a list, got dict"),
        (["calo.photon"], "evaluators[0]: must be a mapping, got str"),
        ([{"name": "x"}], "evaluators[0]: missing required key(s) ['version']"),
        (
            [{"name": "x", "version": "1", "extra": 1}],
            "evaluators[0]: unknown key(s) ['extra']",
        ),
        ([{"name": "a/b", "version": "1"}], "evaluators[0].name: 'a/b' must start"),
        ([{"name": "..", "version": "1"}], "evaluators[0].name: '..' must start"),
        ([{"name": "-x", "version": "1"}], "must start with a letter or digit"),
        # records name the geometry and the stages by name, beside the evaluators
        ([{"name": "geometry", "version": "1"}], "evaluators[0].name: 'geometry' names a step"),
        ([{"name": "simulation", "version": "1"}], "reserved: geometry, generation, simulation"),
        (
            [{"name": "x", "version": "1"}, {"name": "x", "version": "2"}],
            "evaluators[1]: duplicates evaluators[0] (name 'x')",
        ),
    ],
)
def test_evaluator_list_errors(doc: dict[str, Any], evaluators: object, message: str) -> None:
    doc["evaluators"] = evaluators
    reject(doc, message)


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("environment", [], "environment: must be a mapping, got list"),
        ("environment", {}, "missing required key(s) ['key4hep']"),
        ("geometry", {"entry": "a.xml"}, "geometry: must be a non-empty string, got dict"),
        ("parameters", {}, "parameters: must be a list, got dict"),
        ("parameters", [], "parameters: must list at least one parameter"),
        ("parameters", ["x"], "parameters[0]: must be a mapping, got str"),
        ("stages", "ddsim", "stages: must be a mapping"),
        ("stages", ["generation"], "stages: must be a mapping"),
        ("objectives", "min", "objectives: must be a list"),
        ("objectives", [], "objectives: must list at least one objective"),
        ("constraints", {}, "constraints: must be a list"),
        ("constraints", None, "constraints: must be a list"),
    ],
)
def test_section_shapes(doc: dict[str, Any], section: str, value: object, message: str) -> None:
    doc[section] = value
    reject(doc, message)


@pytest.mark.parametrize("data", [None, [], "text", 3])
def test_document_must_be_a_mapping(data: object) -> None:
    reject(data, "top level: must be a mapping")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["environment"]["key4hep"].update(channel="stable"), "channel: must be one of"),
        (
            lambda d: d["environment"]["key4hep"].update(version=""),
            "key4hep.version: must be a non-empty",
        ),
        (lambda d: d["environment"]["key4hep"].update(version=2026), "got int 2026; quote it"),
        (
            lambda d: d["environment"]["key4hep"].update(version=datetime.date(2026, 10, 20)),
            "got date",
        ),
        (lambda d: d.update(geometry="  "), "geometry: must be a non-empty string"),
        (lambda d: d.update(name=""), "name: must be a non-empty string"),
        (
            lambda d: d["evaluators"][0].update(name=""),
            "evaluators[0].name: must be a non-empty",
        ),
        (
            lambda d: d["evaluators"][0].update(version=1),
            "evaluators[0].version: must be a non-empty",
        ),
        (
            lambda d: d["evaluators"][0].update(settings=[1]),
            "evaluators[0].settings: must be a mapping, got list",
        ),
        (
            lambda d: d["evaluators"][0].update(settings={"a": {1, 2}}),
            "evaluators[0].settings.a: unsupported value",
        ),
        (
            lambda d: d["evaluators"][0].update(settings={"a": float("nan")}),
            "non-finite",
        ),
        (
            lambda d: d["evaluators"][0].update(settings={"a": datetime.date(2026, 1, 1)}),
            "settings.a: unsupported",
        ),
        (
            lambda d: d["evaluators"][0].update(settings={1: "x"}),
            "settings: mapping keys must be strings",
        ),
        (lambda d: d["objectives"][0].update(stage=""), "objectives[0].stage: must be a non-empty"),
        (
            lambda d: d["objectives"][0].update(metric=" "),
            "objectives[0].metric: must be a non-empty",
        ),
        (lambda d: d["objectives"][0].update(direction="min"), "direction: must be one of"),
        (lambda d: d["objectives"][0].pop("direction"), "missing required key(s) ['direction']"),
    ],
)
def test_field_errors(
    doc: dict[str, Any], edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    edit(doc)
    reject(doc, message)


def test_identifiers_are_not_altered(doc: dict[str, Any]) -> None:
    doc["geometry"] = " padded/Entry.xml "
    doc["objectives"][0]["metric"] = " odd Metric "
    problem = parse_problem(doc)
    assert problem.geometry == " padded/Entry.xml "
    assert problem.objectives[0].metric == " odd Metric "


def test_non_string_keys_in_schema_objects(doc: dict[str, Any]) -> None:
    doc[1] = "x"
    reject(doc, "top level: keys must be strings, got 1")
    del doc[1]
    doc["parameters"][0][True] = "x"  # YAML 1.1 `yes:` / `on:`
    reject(doc, "parameters[0]: keys must be strings, got True")


# --- parameters ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param", "message"),
    [
        (
            {"geometry": {"constant": "X"}, "type": "continuous"},
            "continuous parameter needs 'bounds'",
        ),
        ({"geometry": {"constant": "X"}, "type": "integer"}, "integer parameter needs 'bounds'"),
        ({"geometry": {"constant": "X"}, "type": "categorical"}, "needs 'choices'"),
        ({"geometry": {"constant": "X"}, "type": "real", "bounds": [1, 2]}, "type: must be one of"),
        ({"geometry": {}, "type": "continuous", "bounds": [1, 2]}, "needs exactly one of"),
        (
            {
                "geometry": {"constant": "X", "target": "constants.X"},
                "type": "continuous",
                "bounds": [1, 2],
            },
            "needs exactly one of",
        ),
        (
            {"geometry": {"constant": ""}, "type": "continuous", "bounds": [1, 2]},
            "geometry.constant: must be a non-empty",
        ),
        (
            {"geometry": {"constant": 5}, "type": "continuous", "bounds": [1, 2]},
            "geometry.constant: must be a non-empty",
        ),
        (
            {"geometry": "ECal_cell_size", "type": "continuous", "bounds": [1, 2]},
            "parameters[1].geometry: must be a mapping",
        ),
        # kinds are mutually exclusive
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [1, 2], "choices": [1]},
            "'choices' is only allowed",
        ),
        (
            {
                "geometry": {"constant": "X"},
                "type": "categorical",
                "choices": ["a"],
                "bounds": [1, 2],
            },
            "'bounds' is not allowed",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": ["a"], "unit": "mm"},
            "'unit' is not allowed",
        ),
        # bounds
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [8, 4]},
            "low must be smaller than high, got [8, 4]",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [4, 4]},
            "low must be smaller than high",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [1]},
            "must be [low, high], got 1 item",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [1, 2, 3]},
            "must be [low, high], got 3 item",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": "1,2"},
            "bounds: must be a list",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [1, float("inf")]},
            "bounds[1]: must be finite",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [float("nan"), 1]},
            "bounds[0]: must be finite",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [True, 2]},
            "bounds[0]: must be a number, got bool",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": ["1", 2]},
            "bounds[0]: must be a number, got str",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "integer", "bounds": [1, 3.5]},
            "bounds[1]: must be an integer, got float",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "integer", "bounds": [1.0, 3]},
            "bounds[0]: must be an integer, got float",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "integer", "bounds": [False, 3]},
            "bounds[0]: must be an integer, got bool",
        ),
        # units
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [1, 2], "unit": ""},
            "unit: must be a non-empty",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "continuous", "bounds": [1, 2], "unit": 5},
            "unit: must be a non-empty",
        ),
        # choices
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": []},
            "must list at least one choice",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": "ab"},
            "choices: must be a list",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": ["a", "a"]},
            "choices[1]: duplicate choice 'a'",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": [1, 1.0]},
            "duplicate choice 1.0",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": ["a", 1]},
            "choices must all be of one kind",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": [1, True]},
            "choices must all be of one kind",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": [None]},
            "choices[0]: choices must be strings",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": [[1]]},
            "choices[0]: choices must be strings",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": [float("nan")]},
            "choices[0]: must be finite",
        ),
        (
            {"geometry": {"constant": "X"}, "type": "categorical", "choices": [""]},
            "choices[0]: must be a non-empty string",
        ),
    ],
)
def test_parameter_errors(doc: dict[str, Any], param: dict[str, Any], message: str) -> None:
    doc["parameters"].append(param)
    reject(doc, message)


def test_duplicate_parameters_are_rejected(doc: dict[str, Any]) -> None:
    doc["parameters"].append(
        {"geometry": {"constant": "ECal_cell_size"}, "type": "continuous", "bounds": [1, 2]}
    )
    reject(doc, "parameters[1]: duplicates parameters[0] (target 'constants.ECal_cell_size')")


def test_constant_and_target_spellings_are_duplicates(doc: dict[str, Any]) -> None:
    doc["parameters"].append(
        {
            "geometry": {"target": "constants.ECal_cell_size"},
            "type": "continuous",
            "bounds": [1, 2],
        }
    )
    reject(doc, "duplicates parameters[0]")


# --- objectives and constraints --------------------------------------------------------


def test_duplicate_objectives(doc: dict[str, Any]) -> None:
    doc["objectives"].append({"stage": "simulation", "metric": "a.rms", "direction": "maximize"})
    reject(doc, "objectives[1]: duplicates objectives[0]")


def test_same_metric_in_another_stage_is_not_a_duplicate(doc: dict[str, Any]) -> None:
    doc["objectives"].append({"stage": "analysis", "metric": "a.rms", "direction": "maximize"})
    assert len(parse_problem(doc).objectives) == 2


def test_duplicate_constraints(doc: dict[str, Any]) -> None:
    doc["constraints"].append({"stage": "simulation", "metric": "a.cells", "min": 1})
    reject(doc, "constraints[1]: duplicates constraints[0]")


@pytest.mark.parametrize(
    ("constraint", "message"),
    [
        ({"stage": "s", "metric": "m"}, "needs 'min' and/or 'max'"),
        (
            {"stage": "s", "metric": "m", "min": 3, "max": 2},
            "min must not exceed max, got min=3, max=2",
        ),
        ({"stage": "s", "metric": "m", "max": float("inf")}, "max: must be finite"),
        ({"stage": "s", "metric": "m", "min": float("nan")}, "min: must be finite"),
        ({"stage": "s", "metric": "m", "max": True}, "max: must be a number, got bool"),
        ({"stage": "s", "metric": "m", "max": "2"}, "max: must be a number, got str"),
        ({"stage": "s", "metric": "m", "max": None}, "max: must be a number"),
        ({"stage": "", "metric": "m", "max": 1}, "stage: must be a non-empty"),
        ({"stage": "s", "metric": 3, "max": 1}, "metric: must be a non-empty"),
        ({"stage": "s", "max": 1}, "missing required key(s) ['metric']"),
        ({"stage": "s", "metric": "m", "max": 1, "unit": ""}, "unit: must be a non-empty"),
    ],
)
def test_constraint_errors(doc: dict[str, Any], constraint: dict[str, Any], message: str) -> None:
    doc["constraints"] = [constraint]
    reject(doc, message)


# --- YAML and file handling ------------------------------------------------------------


def good_yaml() -> str:
    return EXAMPLE.read_text(encoding="utf-8")


def test_errors_belong_to_the_k4opt_hierarchy() -> None:
    assert issubclass(ProblemConfigError, ConfigError)
    assert issubclass(ProblemConfigError, K4OptError)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProblemConfigError, match="problem file not found"):
        load_problem(tmp_path / "nope.yaml")


def test_directory_instead_of_file(tmp_path: Path) -> None:
    with pytest.raises(ProblemConfigError, match="not found"):
        load_problem(tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_unreadable_file(write: Write) -> None:
    path = write(good_yaml())
    path.chmod(0)
    try:
        with pytest.raises(ProblemConfigError, match="cannot read"):
            load_problem(path)
    finally:
        path.chmod(0o644)


def test_invalid_utf8(write: Write) -> None:
    with pytest.raises(ProblemConfigError, match="not valid UTF-8"):
        load_problem(write(b"name: \xff\xfe\n"))


def test_error_names_the_file_and_location(write: Write) -> None:
    path = write(good_yaml().replace("bounds: [2.5, 20.0]", "bounds: [20.0, 2.5]"))
    with pytest.raises(ProblemConfigError) as info:
        load_problem(path)
    assert str(info.value).startswith(f"{path}: parameters[0].bounds: low must be smaller")


@pytest.mark.parametrize(
    "text",
    [
        "a: [1, 2\n",  # syntax error
        "a: 1\na: 2\n",  # duplicate key
        "a: 1\n---\nb: 2\n",  # several documents
        "a: !!python/object/apply:os.system ['true']\n",  # code-executing tag
        "a: !!python/name:os.system\n",
        "? [1, 2]\n: x\n",  # unhashable key
        "a: 2026-02-30\n",  # a date PyYAML recognizes but cannot construct
        "a: 2026-13-01 10:00:00\n",
        "a: !!timestamp nonsense\n",
        "a: !!int abc\n",
        "a: !!float abc\n",
    ],
)
def test_yaml_level_errors(write: Write, text: str) -> None:
    with pytest.raises(ProblemConfigError, match="invalid YAML"):
        load_problem(write(text))


def test_an_unconstructible_value_names_its_position(write: Write) -> None:
    with pytest.raises(ProblemConfigError, match=r"(?s)invalid timestamp '2026-02-30'.*line 2"):
        load_problem(write("name: x\nenvironment: {key4hep: 2026-02-30}\n"))


def test_duplicate_key_message_has_the_line(write: Write) -> None:
    with pytest.raises(ProblemConfigError, match=r"duplicate key 'geometry' \(line 3\)"):
        load_problem(write("name: x\ngeometry: a.xml\ngeometry: b.xml\n"))


def test_nested_duplicate_keys(write: Write) -> None:
    text = good_yaml().replace("    seed: 42\n", "    seed: 42\n    seed: 43\n")
    with pytest.raises(ProblemConfigError, match="duplicate key 'seed'"):
        load_problem(write(text))


@pytest.mark.parametrize("text", ["", "# only a comment\n", "[]\n", "42\n"])
def test_non_mapping_documents(write: Write, text: str) -> None:
    with pytest.raises(ProblemConfigError, match="top level: must be a mapping"):
        load_problem(write(text))


def test_non_string_key_in_a_file(write: Write) -> None:
    with pytest.raises(ProblemConfigError, match="keys must be strings"):
        load_problem(write(good_yaml().replace("name:", "yes:", 1)))


def test_unquoted_date_is_flagged(write: Write) -> None:
    text = good_yaml().replace('version: "2026-10-20"', "version: 2026-10-20")
    with pytest.raises(
        ProblemConfigError, match=r"key4hep\.version: must be a non-empty string, got date"
    ):
        load_problem(write(text))


def test_disallowed_scalars_in_settings_from_yaml(write: Write) -> None:
    text = good_yaml().replace("      threshold:", "      when: 2026-10-20\n      threshold:")
    with pytest.raises(ProblemConfigError, match=r"settings\.when: unsupported value of type date"):
        load_problem(write(text))
    text = good_yaml().replace("      threshold:", "      blob: !!binary aGk=\n      threshold:")
    with pytest.raises(
        ProblemConfigError, match=r"settings\.blob: unsupported value of type bytes"
    ):
        load_problem(write(text))
    text = good_yaml().replace("      threshold:", "      s: !!set {a, b}\n      threshold:")
    with pytest.raises(ProblemConfigError, match=r"settings\.s: unsupported value of type set"):
        load_problem(write(text))
    text = good_yaml().replace("      threshold:", "      x: .nan\n      threshold:")
    with pytest.raises(ProblemConfigError, match="non-finite"):
        load_problem(write(text))


def test_recursive_aliases_are_rejected(write: Write) -> None:
    text = good_yaml().replace("      threshold:", "      loop: &a [*a]\n      threshold:")
    with pytest.raises(ProblemConfigError, match="contains itself"):
        load_problem(write(text))
    text = good_yaml().replace("      threshold:", "      loop: &a {x: *a}\n      threshold:")
    with pytest.raises(ProblemConfigError, match="contains itself"):
        load_problem(write(text))


def test_recursive_alias_in_a_schema_object_is_a_clean_error(write: Write) -> None:
    text = good_yaml().replace("bounds: [2.5, 20.0]", "bounds: &a [*a, 1]")
    with pytest.raises(ProblemConfigError, match="bounds"):
        load_problem(write(text))


def test_very_deep_yaml_is_a_clean_error(write: Write) -> None:
    deep = "[" * 5000 + "]" * 5000
    with pytest.raises(ProblemConfigError):
        load_problem(write(f"name: x\ndeep: {deep}\n"))


def test_aliases_and_merge_keys_are_supported(write: Write) -> None:
    text = """
schema_version: k4opt.problem/v1alpha1
environment: {key4hep: {channel: release, version: "2026-04-08"}}
geometry: g.xml
parameters:
  - {geometry: {constant: X}, type: integer, bounds: [1, 3]}
evaluators: [{name: e, version: "1"}]
objectives:
  - &base {stage: simulation, metric: a, direction: minimize}
  - <<: *base
    metric: b
    stage: analysis
"""
    problem = parse_problem(yaml.safe_load(text))
    assert [(o.stage, o.metric) for o in problem.objectives] == [
        ("simulation", "a"),
        ("analysis", "b"),
    ]


@pytest.mark.parametrize("huge", [10**400, -(10**400)], ids=["positive", "negative"])
def test_huge_integers_are_ordinary_numbers(doc: dict[str, Any], huge: int) -> None:
    doc["parameters"][0]["bounds"] = sorted([huge, 0])
    doc["constraints"] = [{"stage": "s", "metric": "m", "max": 10**400}]
    doc["stages"]["simulation"]["seed"] = huge
    problem = parse_problem(doc)
    assert problem.constraints[0].max == 10**400
    assert parse_problem(problem.to_dict()) == problem


def test_huge_integer_next_to_a_float_bound(doc: dict[str, Any]) -> None:
    doc["parameters"][0]["bounds"] = [1.5, 10**400]
    assert parse_problem(doc).parameters[0].bounds == (1.5, 10**400)
    doc["parameters"][0]["bounds"] = [10**400, 1.5]
    reject(doc, "low must be smaller than high")
