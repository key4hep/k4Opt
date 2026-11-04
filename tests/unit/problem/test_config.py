"""Loading and validating problem documents."""

from __future__ import annotations

import copy
import datetime
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt import ConfigError, K4OptError
from k4opt.problem import (
    GeometryDestination,
    MetricRef,
    ProblemConfigError,
    load_problem,
    parse_problem,
)
from k4opt.problem.models import essential

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
    (destination,) = parameter.targets
    assert isinstance(destination, GeometryDestination)
    assert destination.reference == "ECal_cell_size"
    assert destination.canonical_target == "constants.ECal_cell_size"
    assert parameter.name == "ecal_cell_size"
    assert (parameter.kind, parameter.bounds, parameter.unit) == ("continuous", (2.5, 20.0), "mm")
    assert [(e.name, e.version, e.stage) for e in problem.evaluators] == [
        ("calo.photon", "1", "simulation")
    ]
    assert problem.evaluators[0].collections == {"hits": "ECalBarrelCollection"}
    assert problem.evaluators[0].settings == {"threshold": {"value": 70, "unit": "keV"}}
    assert problem.evaluators[0].samples == ("gamma_10gev",)  # the problem's only sample
    (sample,) = problem.samples.values()
    assert sample.name == "gamma_10gev"
    generation, simulation = sample.stages.values()
    assert (generation.type, generation.executor) == ("generation", "gun")
    generator: Any = generation.options
    assert generator["particle"] == "gamma"
    assert generator["direction"]["theta"] == (80, 100)
    assert (simulation.executor, simulation.events, simulation.seed) == ("ddsim", 1000, 42)
    assert simulation.options == {"physics.list": "FTFP_BERT"}
    assert [
        (o.sample, o.evaluator, o.stage, o.metric, o.direction) for o in problem.objectives
    ] == [
        ("gamma_10gev", "calo.photon", "simulation", "ecal_barrel.angular_residual", "minimize"),
        ("gamma_10gev", "calo.photon", "simulation", "ecal_barrel.fired_cells", "minimize"),
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
    # an evaluator that reads no stage measures the candidate, and needs no samples
    for key in ("name", "constraints", "samples"):
        del doc[key]
    del doc["evaluators"][0]["settings"], doc["evaluators"][0]["stage"]
    problem = parse_problem(doc)
    assert problem.samples == {}
    assert not problem.evaluators[0].settings and problem.evaluators[0].stage is None
    assert problem.evaluators[0].samples == () and problem.objectives[0].sample is None
    assert problem.constraints == ()
    assert "samples" not in problem.to_dict()
    assert "samples" not in problem.to_dict()["evaluators"][0]  # type: ignore[index]


def test_integer_and_categorical_parameters(doc: dict[str, Any]) -> None:
    doc["parameters"] |= {
        "n": {
            "targets": {"geometry": "T.layer[id=1].n"},
            "type": "integer",
            "bounds": [60, 120],
        },
        "mode": {
            "targets": {"geometry": "Mode"},
            "type": "categorical",
            "choices": ["a", "b"],
        },
        "flag": {
            "targets": {"geometry": "Flag"},
            "type": "categorical",
            "choices": [True, False],
        },
        "num": {
            "targets": {"geometry": "Num"},
            "type": "categorical",
            "choices": [1, 2.5],
        },
    }
    kinds = [(p.kind, p.bounds, p.choices) for p in parse_problem(doc).parameters[1:]]
    assert kinds == [
        ("integer", (60, 120), None),
        ("categorical", None, ("a", "b")),
        ("categorical", None, (True, False)),
        ("categorical", None, (1, 2.5)),
    ]


def test_constraints_with_min_max_or_both(doc: dict[str, Any]) -> None:
    doc["constraints"] = [
        {"stage": "s", "metric": "a", "min": 1, "scale": 1},
        {"stage": "s", "metric": "b", "max": 2.5, "unit": "mm", "scale": 0.5},
        {"stage": "s", "metric": "c", "min": 1, "max": 1, "scale": 3},
    ]
    assert [(c.min, c.max, c.scale) for c in parse_problem(doc).constraints] == [
        (1, None, 1),
        (None, 2.5, 0.5),
        (1, 1, 3),
    ]


def test_a_constraint_on_one_metric_needs_no_scale(doc: dict[str, Any]) -> None:
    doc["constraints"] = [{"stage": "s", "metric": "a", "min": 1, "max": 3}]
    (constraint,) = parse_problem(doc).constraints
    assert constraint.scale is None and "scale" not in constraint.to_dict()


def test_constraints_on_several_metrics_each_need_a_scale(doc: dict[str, Any]) -> None:
    # the optimizer adds how far a candidate is beyond each limit: only meaningful in units of
    # each metric's characteristic size, never in whichever unit happens to be largest
    doc["constraints"] = [
        {"stage": "s", "metric": "a", "min": 1, "scale": 1},
        {"stage": "s", "metric": "b", "max": 2.5, "unit": "mm"},
        {"stage": "s", "metric": "c", "max": 7},
    ]
    with pytest.raises(
        ProblemConfigError, match=r"constraints\[1\], constraints\[2\]: needs a scale"
    ):
        parse_problem(doc)


@pytest.mark.parametrize("scale", [0, -1.5, float("inf"), float("nan"), True, "1"])
def test_a_scale_is_a_positive_finite_number(doc: dict[str, Any], scale: object) -> None:
    doc["constraints"] = [{"stage": "s", "metric": "a", "max": 1, "scale": scale}]
    with pytest.raises(ProblemConfigError, match=r"constraints\[0\]\.scale: must be"):
        parse_problem(doc)


def test_objective_and_constraint_may_share_a_metric(doc: dict[str, Any]) -> None:
    doc["constraints"] = [{"stage": "simulation", "metric": "a.rms", "max": 2}]
    assert parse_problem(doc).constraints[0].metric == "a.rms"


def titled(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` with a title on its parameter, sample, objective and constraint."""
    doc["parameters"]["ecal_cell_size"]["title"] = "ECAL cell size"
    doc["samples"]["gamma"]["title"] = "10 GeV γ"
    doc["objectives"][0]["title"] = "σ(E)/E"
    doc["constraints"][0]["title"] = "Fired cells"
    return doc


def test_titles_are_kept_and_change_nothing_but_what_a_reader_is_shown(
    doc: dict[str, Any],
) -> None:
    plain = parse_problem(copy.deepcopy(doc))
    spec = parse_problem(titled(doc))
    assert spec.parameters[0].title == "ECAL cell size"
    assert spec.samples["gamma"].title == "10 GeV γ"
    assert (spec.objectives[0].title, spec.constraints[0].title) == ("σ(E)/E", "Fired cells")
    assert plain.parameters[0].title is None and plain.samples["gamma"].title is None
    assert parse_problem(spec.to_dict()) == spec
    assert spec != plain
    assert essential(spec.to_dict()) == essential(plain.to_dict())


@pytest.mark.parametrize(
    ("title", "message"),
    [
        ("two\nlines", "must be one line"),
        ("ends in a line break\n", "must be one line"),  # as a YAML | block does
        ("  ", "must be a non-empty string"),
        (3, "must be a non-empty string"),
    ],
)
def test_a_title_is_one_line_of_text(doc: dict[str, Any], title: object, message: str) -> None:
    doc["parameters"]["ecal_cell_size"]["title"] = title
    reject(doc, f"parameters.ecal_cell_size.title: {message}")
    del doc["parameters"]["ecal_cell_size"]["title"]
    doc["objectives"][0]["title"] = title
    reject(doc, f"objectives[0].title: {message}")


def test_several_objectives_keep_their_order(doc: dict[str, Any]) -> None:
    doc["objectives"].append({"stage": "analysis", "metric": "x", "direction": "maximize"})
    assert [o.metric for o in parse_problem(doc).objectives] == ["a.rms", "x"]


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
        (("parameters", "ecal_cell_size"), "parameters.ecal_cell_size: unknown key(s) ['extra']"),
        (
            ("parameters", "ecal_cell_size", "targets"),
            "parameters.ecal_cell_size.targets: unknown key(s) ['extra']",
        ),
        (("samples", "gamma"), "samples.gamma: unknown key(s) ['extra']"),
        (("samples", "gamma", "generation"), "samples.gamma.generation: unknown key(s) ['extra']"),
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
    del doc["samples"], doc["evaluators"][0]["stage"]
    doc["evaluators"][0]["collections"] = {"hits": "ECalBarrelCollection"}
    reject(doc, "evaluators[0].collections: collections are read from the file of the stage")


def test_unknown_keys_inside_settings_are_allowed(doc: dict[str, Any]) -> None:
    doc["evaluators"][0]["settings"] = {"anything": {"goes": [1, "x", None, True]}}
    assert parse_problem(doc).evaluators[0].to_dict()["settings"] == {
        "anything": {"goes": [1, "x", None, True]}
    }


def test_several_evaluators_keep_their_order_and_own_settings(doc: dict[str, Any]) -> None:
    doc["evaluators"].append({"name": "detector.cost", "version": "2"})
    for item in (*doc["objectives"], *doc["constraints"]):
        item["evaluator"] = "calo.single_particle_shower"
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
            "evaluators[1]: 'x' is already bound by evaluators[0]; name this one with as:",
        ),
        (
            [{"name": "x", "version": "1"}, {"name": "y", "as": "x", "version": "1"}],
            "evaluators[1]: 'x' is already bound by evaluators[0]",
        ),
        # a run directory per binding: a file system that folds case would hold one for both
        (
            [{"name": "x", "version": "1"}, {"name": "x", "as": "X", "version": "1"}],
            "evaluators[1]: 'X' differs only in case from 'x' of evaluators[0], which run "
            "directories may not tell apart; name this one with as:",
        ),
        ([{"name": "x", "as": "", "version": "1"}], "evaluators[0].as: must be a non-empty"),
        ([{"name": "x", "as": "a/b", "version": "1"}], "evaluators[0].as: 'a/b' must start"),
        ([{"name": "x", "as": 3, "version": "1"}], "evaluators[0].as: must be a non-empty"),
        ([{"name": "x", "as": "geometry", "version": "1"}], "evaluators[0].as: 'geometry' names"),
    ],
)
def test_evaluator_list_errors(doc: dict[str, Any], evaluators: object, message: str) -> None:
    doc["evaluators"] = evaluators
    reject(doc, message)


def test_one_implementation_can_be_bound_twice_under_two_names(doc: dict[str, Any]) -> None:
    first = doc["evaluators"][0]
    doc["evaluators"].append(first | {"as": "shower_tight", "settings": {"collection": "Other"}})
    doc["objectives"][0]["evaluator"] = "calo.single_particle_shower"
    doc["objectives"].append(doc["objectives"][0] | {"evaluator": "shower_tight"})
    doc["constraints"][0]["evaluator"] = "shower_tight"
    problem = parse_problem(doc)
    assert [(e.name, e.binding) for e in problem.evaluators] == [
        ("calo.single_particle_shower", "calo.single_particle_shower"),
        ("calo.single_particle_shower", "shower_tight"),
    ]
    # the same metric of two bindings: two objectives, not a duplicate
    assert [o.label for o in problem.objectives] == [
        "gamma/calo.single_particle_shower/simulation/a.rms",
        "gamma/shower_tight/simulation/a.rms",
    ]
    ref = MetricRef("shower_tight", "simulation", "a.cells", "gamma")
    assert problem.constraints[0].ref == ref
    data = problem.to_dict()
    assert "as" not in data["evaluators"][0]  # the binding is the name unless it is not
    assert data["evaluators"][1]["as"] == "shower_tight"
    assert parse_problem(data) == problem


def test_an_objective_of_the_one_evaluator_may_leave_it_out(doc: dict[str, Any]) -> None:
    problem = parse_problem(doc)
    (objective,) = problem.objectives
    ref = MetricRef("calo.single_particle_shower", "simulation", "a.rms", "gamma")
    assert objective.ref == ref
    assert objective.label == "gamma/calo.single_particle_shower/simulation/a.rms"
    assert (problem.constraints[0].sample, problem.constraints[0].evaluator) == (
        "gamma",
        "calo.single_particle_shower",
    )
    # the normalized problem, as records and studies hold it, names them always
    data: Any = problem.to_dict()
    assert data["evaluators"][0]["samples"] == ["gamma"]
    for section in ("objectives", "constraints"):
        assert data[section][0]["sample"] == "gamma"
        assert data[section][0]["evaluator"] == "calo.single_particle_shower"
    assert parse_problem(data) == problem
    named = copy.deepcopy(doc)
    named["objectives"][0]["evaluator"] = "calo.single_particle_shower"
    named["constraints"][0]["sample"] = "gamma"
    named["evaluators"][0]["samples"] = ["gamma"]
    assert parse_problem(named) == problem


@pytest.mark.parametrize("section", ["objectives", "constraints"])
def test_an_objective_or_constraint_of_several_evaluators_must_name_one(
    doc: dict[str, Any], section: str
) -> None:
    doc["evaluators"].append({"name": "detector.cost", "version": "1"})
    other = "constraints" if section == "objectives" else "objectives"
    doc[other][0]["evaluator"] = "detector.cost"
    reject(
        doc,
        f"{section}[0]: the problem has the evaluations gamma/calo.single_particle_shower, "
        "detector.cost; add evaluator: (and sample:) to say which one measures this metric",
    )


@pytest.mark.parametrize(
    ("evaluator", "message"),
    [
        ("detector.cost", "the problem has no evaluator 'detector.cost'; its evaluators are calo."),
        ("", "objectives[0].evaluator: must be a non-empty string"),
    ],
)
def test_an_objective_must_name_an_evaluator_of_the_problem(
    doc: dict[str, Any], evaluator: str, message: str
) -> None:
    doc["objectives"][0]["evaluator"] = evaluator
    reject(doc, message)


def test_an_evaluator_may_be_marked_required_and_keeps_it_through_the_normalized_form(
    doc: dict[str, Any],
) -> None:
    assert parse_problem(doc).evaluators[0].required is False
    assert "required" not in parse_problem(doc).to_dict()["evaluators"][0]
    doc["evaluators"][0]["required"] = True
    problem = parse_problem(doc)
    assert problem.evaluators[0].required is True
    assert problem.to_dict()["evaluators"][0]["required"] is True  # so in the study's identity
    assert parse_problem(problem.to_dict()) == problem


@pytest.mark.parametrize("value", [False, "yes", 1, None])
def test_required_takes_true_alone(doc: dict[str, Any], value: object) -> None:
    doc["evaluators"][0]["required"] = value
    reject(
        doc,
        f"evaluators[0].required: must be true, got {value!r}; an evaluator that no objective, "
        "constraint or required evaluator reads is a diagnostic without it",
    )


def test_an_objective_names_a_binding_not_an_implementation(doc: dict[str, Any]) -> None:
    doc["evaluators"][0]["as"] = "shower"
    doc["constraints"][0]["evaluator"] = "shower"
    doc["objectives"][0]["evaluator"] = "calo.single_particle_shower"
    reject(doc, "objectives[0].evaluator: the problem has no evaluator 'calo.single_particle")


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("environment", [], "environment: must be a mapping, got list"),
        ("environment", {}, "missing required key(s) ['key4hep']"),
        ("geometry", {"entry": "a.xml"}, "geometry: must be a non-empty string, got dict"),
        ("parameters", [], "parameters: must be a mapping, got list"),
        ("parameters", {}, "parameters: must name at least one parameter"),
        ("parameters", {"x": "y"}, "parameters.x: must be a mapping, got str"),
        ("parameters", {"X": {}}, "parameter name 'X' must start with a lowercase letter"),
        ("parameters", {"a.b": {}}, "parameter name 'a.b' must start with a lowercase letter"),
        ("samples", "ddsim", "samples: must be a mapping"),
        ("samples", ["gamma"], "samples: must be a mapping"),
        ("samples", {}, "samples: must name at least one sample"),
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
    doc["parameters"]["ecal_cell_size"][True] = "x"  # YAML 1.1 `yes:` / `on:`
    reject(doc, "parameters.ecal_cell_size: keys must be strings, got True")


# --- parameters ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param", "message"),
    [
        (
            {"targets": {"geometry": "X"}, "type": "continuous"},
            "continuous parameter needs 'bounds'",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "integer"},
            "integer parameter needs 'bounds'",
        ),
        ({"targets": {"geometry": "X"}, "type": "categorical"}, "needs 'choices'"),
        (
            {"targets": {"geometry": "X"}, "type": "real", "bounds": [1, 2]},
            "type: must be one of",
        ),
        (
            {"targets": {}, "type": "continuous", "bounds": [1, 2]},
            "parameters.added.targets: must name at least one target",
        ),
        (
            {"targets": {"geometry": "X", "target": "Y"}, "type": "continuous", "bounds": [1, 2]},
            "parameters.added.targets: unknown key(s) ['target']",
        ),
        (
            {"targets": {"geometry": ""}, "type": "continuous", "bounds": [1, 2]},
            "targets.geometry: must be a non-empty",
        ),
        (
            {"targets": {"geometry": 5}, "type": "continuous", "bounds": [1, 2]},
            "targets.geometry: must be a non-empty",
        ),
        (
            {"targets": "ECal_cell_size", "type": "continuous", "bounds": [1, 2]},
            "parameters.added.targets: must be a mapping",
        ),
        # kinds are mutually exclusive
        (
            {
                "targets": {"geometry": "X"},
                "type": "continuous",
                "bounds": [1, 2],
                "choices": [1],
            },
            "'choices' is only allowed",
        ),
        (
            {
                "targets": {"geometry": "X"},
                "type": "categorical",
                "choices": ["a"],
                "bounds": [1, 2],
            },
            "'bounds' is not allowed",
        ),
        (
            {
                "targets": {"geometry": "X"},
                "type": "categorical",
                "choices": ["a"],
                "unit": "mm",
            },
            "'unit' is not allowed",
        ),
        # bounds
        (
            {"targets": {"geometry": "X"}, "type": "continuous", "bounds": [8, 4]},
            "low must be smaller than high, got [8, 4]",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "continuous", "bounds": [4, 4]},
            "low must be smaller than high",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "continuous", "bounds": [1]},
            "must be [low, high], got 1 item",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "continuous", "bounds": [1, 2, 3]},
            "must be [low, high], got 3 item",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "continuous", "bounds": "1,2"},
            "bounds: must be a list",
        ),
        (
            {
                "targets": {"geometry": "X"},
                "type": "continuous",
                "bounds": [1, float("inf")],
            },
            "bounds[1]: must be finite",
        ),
        (
            {
                "targets": {"geometry": "X"},
                "type": "continuous",
                "bounds": [float("nan"), 1],
            },
            "bounds[0]: must be finite",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "continuous", "bounds": [True, 2]},
            "bounds[0]: must be a number, got bool",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "continuous", "bounds": ["1", 2]},
            "bounds[0]: must be a number, got str",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "integer", "bounds": [1, 3.5]},
            "bounds[1]: must be an integer, got float",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "integer", "bounds": [1.0, 3]},
            "bounds[0]: must be an integer, got float",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "integer", "bounds": [False, 3]},
            "bounds[0]: must be an integer, got bool",
        ),
        # units
        (
            {
                "targets": {"geometry": "X"},
                "type": "continuous",
                "bounds": [1, 2],
                "unit": "",
            },
            "unit: must be a non-empty",
        ),
        (
            {
                "targets": {"geometry": "X"},
                "type": "continuous",
                "bounds": [1, 2],
                "unit": 5,
            },
            "unit: must be a non-empty",
        ),
        # choices
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": []},
            "must list at least one choice",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": "ab"},
            "choices: must be a list",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": ["a", "a"]},
            "choices[1]: duplicate choice 'a'",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": [1, 1.0]},
            "duplicate choice 1.0",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": ["a", 1]},
            "choices must all be of one kind",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": [1, True]},
            "choices must all be of one kind",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": [None]},
            "choices[0]: choices must be strings",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": [[1]]},
            "choices[0]: choices must be strings",
        ),
        (
            {
                "targets": {"geometry": "X"},
                "type": "categorical",
                "choices": [float("nan")],
            },
            "choices[0]: must be finite",
        ),
        (
            {"targets": {"geometry": "X"}, "type": "categorical", "choices": [""]},
            "choices[0]: must be a non-empty string",
        ),
    ],
)
def test_parameter_errors(doc: dict[str, Any], param: dict[str, Any], message: str) -> None:
    doc["parameters"]["added"] = param
    reject(doc, message)


def test_parameters_that_set_one_target_are_rejected(doc: dict[str, Any]) -> None:
    doc["parameters"]["again"] = {
        "targets": {"geometry": "ECal_cell_size"},
        "type": "continuous",
        "bounds": [1, 2],
    }
    reject(
        doc,
        "parameters.again.targets.geometry: the geometry's constants.ECal_cell_size is set by "
        "parameters.ecal_cell_size already",
    )


def test_constant_and_target_spellings_are_one_target(doc: dict[str, Any]) -> None:
    doc["parameters"]["again"] = {
        "targets": {"geometry": "constants.ECal_cell_size"},
        "type": "continuous",
        "bounds": [1, 2],
    }
    reject(doc, "constants.ECal_cell_size is set by parameters.ecal_cell_size already")


# --- objectives and constraints --------------------------------------------------------


def test_duplicate_objectives(doc: dict[str, Any]) -> None:
    doc["objectives"].append({"stage": "simulation", "metric": "a.rms", "direction": "maximize"})
    reject(
        doc, "objectives[1]: duplicates objectives[0] (gamma/calo.single_particle_shower/simulation"
    )


def test_an_objective_written_with_and_without_its_evaluator_is_a_duplicate(
    doc: dict[str, Any],
) -> None:
    named = doc["objectives"][0] | {"evaluator": "calo.single_particle_shower"}
    doc["objectives"].append(named)
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
    assert str(info.value).startswith(
        f"{path}: parameters.ecal_cell_size.bounds: low must be smaller"
    )


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
    text = good_yaml().replace("      seed: 42\n", "      seed: 42\n      seed: 43\n")
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
  x: {targets: {geometry: X}, type: integer, bounds: [1, 3]}
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


# a destination that reads its value as a double refuses such bounds; an evaluator setting does not
SETTING = {"evaluators": {"calo.single_particle_shower": "size"}}


@pytest.mark.parametrize("huge", [10**400, -(10**400)], ids=["positive", "negative"])
def test_huge_integers_are_ordinary_numbers(doc: dict[str, Any], huge: int) -> None:
    doc["parameters"]["ecal_cell_size"]["targets"] = SETTING
    doc["parameters"]["ecal_cell_size"]["bounds"] = sorted([huge, 0])
    doc["constraints"] = [{"stage": "s", "metric": "m", "max": 10**400}]
    doc["samples"]["gamma"]["simulation"]["seed"] = huge
    problem = parse_problem(doc)
    assert problem.constraints[0].max == 10**400
    assert parse_problem(problem.to_dict()) == problem


def test_huge_integer_next_to_a_float_bound(doc: dict[str, Any]) -> None:
    doc["parameters"]["ecal_cell_size"]["targets"] = SETTING
    doc["parameters"]["ecal_cell_size"]["bounds"] = [1.5, 10**400]
    assert parse_problem(doc).parameters[0].bounds == (1.5, 10**400)
    doc["parameters"]["ecal_cell_size"]["bounds"] = [10**400, 1.5]
    reject(doc, "low must be smaller than high")


def test_time_limits_are_part_of_the_problem(doc: dict[str, Any]) -> None:
    doc["timeout"] = 3600
    doc["samples"]["gamma"]["simulation"]["timeout"] = 7200.5
    doc["evaluators"][0]["timeout"] = 60
    spec = parse_problem(doc)
    assert spec.timeout == 3600 and spec.evaluators[0].timeout == 60
    assert spec.samples["gamma"].stages["simulation"].timeout == 7200.5
    assert parse_problem(spec.to_dict()) == spec  # written back as given: the study's identity


@pytest.mark.parametrize("value", [0, -5.0, float("nan"), float("inf"), True, "60", 10**400])
@pytest.mark.parametrize(
    ("where", "path"),
    [((), "timeout"), (("samples", "gamma", "simulation"), r"samples\.gamma\.simulation\.timeout")],
)
def test_a_time_limit_is_a_positive_number_of_seconds(
    doc: dict[str, Any], value: object, where: tuple[str, ...], path: str
) -> None:
    target: Any = doc
    for key in where:
        target = target[key]
    target["timeout"] = value
    with pytest.raises(ProblemConfigError, match=path):
        parse_problem(doc)


def test_a_stage_given_as_a_file_takes_no_time_limit(doc: dict[str, Any]) -> None:
    doc["samples"]["gamma"]["generation"] = {"file": "events.hepmc", "timeout": 5}
    with pytest.raises(ProblemConfigError, match=r"is not run, so it takes no \['timeout'\]"):
        parse_problem(doc)
