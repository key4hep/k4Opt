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
EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "problem" / "cld-ecal-hit-count.yaml"


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
    assert problem.name == "cld-ecal-hit-count"
    assert problem.environment.key4hep.channel == "nightly"
    assert problem.environment.key4hep.version == "2026-10-20"
    assert problem.geometry.entry == "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml"
    (parameter,) = problem.parameters
    assert parameter.geometry.constant == "ECal_cell_size"
    assert parameter.geometry.canonical_target == "constants.ECal_cell_size"
    assert (parameter.kind, parameter.bounds, parameter.unit) == ("continuous", (4.0, 8.0), "mm")
    evaluation = problem.evaluation
    assert (evaluation.evaluator, evaluation.version) == ("calo.hit_multiplicity", "1")
    assert (evaluation.events, evaluation.seed) == (100, 42)
    assert evaluation.backend["ddsim"] == ("--physics.list", "FTFP_BERT")
    generator: Any = evaluation.generation
    assert generator["particle"] == "gamma"
    assert generator["direction"]["theta"] == (60, 120)
    assert [(o.stage, o.metric, o.direction) for o in problem.objectives] == [
        ("simulation", "ecal_barrel.mean_hit_cells", "minimize")
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
    for key in ("name", "constraints"):
        del doc[key]
    del doc["evaluation"]["events"], doc["evaluation"]["seed"]
    del doc["evaluation"]["settings"], doc["evaluation"]["backend"]
    del doc["evaluation"]["generation"]
    problem = parse_problem(doc)
    assert problem.evaluation.events is None
    assert not problem.evaluation.settings and not problem.evaluation.backend
    assert not problem.evaluation.generation
    assert problem.constraints == ()


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


# --- backend arguments -----------------------------------------------------------------


def test_backend_args_are_preserved_verbatim(doc: dict[str, Any]) -> None:
    args = ["--a", "1", "--a", "2", "--flag=x y", "pos", "", "  spaced  ", "$HOME", "*.root", "a;b"]
    doc["evaluation"]["backend"] = {"ddsim": {"args": args}, "k4run": {"args": ["--x=1"]}}
    evaluation = parse_problem(doc).evaluation
    assert evaluation.backend["ddsim"] == tuple(args)
    assert list(evaluation.backend) == ["ddsim", "k4run"]
    assert evaluation.to_dict()["backend"] == {
        "ddsim": {"args": args},
        "k4run": {"args": ["--x=1"]},
    }


def test_backend_args_may_be_empty(doc: dict[str, Any]) -> None:
    doc["evaluation"]["backend"] = {"ddsim": {"args": []}}
    assert parse_problem(doc).evaluation.backend["ddsim"] == ()


def test_parsing_never_runs_a_subprocess(
    doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess started")

    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("os.system", boom)
    doc["evaluation"]["backend"]["ddsim"]["args"] = ["; rm -rf /", "$(whoami)"]
    parse_problem(doc)


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        ({"ddsim": {"args": ["--n", 5]}}, "evaluation.backend.ddsim.args[1]: must be a string"),
        ({"ddsim": {"args": [None]}}, "args[0]: must be a string"),
        ({"ddsim": {"args": "--a --b"}}, "evaluation.backend.ddsim.args: must be a list"),
        ({"ddsim": {"args": {"--a": 1}}}, "must be a list"),
        ({"ddsim": {}}, "missing required key(s) ['args']"),
        ({"ddsim": {"args": [], "env": {}}}, "unknown key(s) ['env']"),
        ({"ddsim": ["--a"]}, "evaluation.backend.ddsim: must be a mapping"),
        ({"dsim": {"args": []}}, "unknown key(s) ['dsim']"),
        (["ddsim"], "evaluation.backend: must be a mapping"),
    ],
)
def test_backend_errors(doc: dict[str, Any], backend: object, message: str) -> None:
    doc["evaluation"]["backend"] = backend
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
    "key", ["environment", "geometry", "parameters", "evaluation", "objectives"]
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
        (("geometry",), "geometry: unknown key(s) ['extra']"),
        (("parameters", 0), "parameters[0]: unknown key(s) ['extra']"),
        (("parameters", 0, "geometry"), "parameters[0].geometry: unknown key(s) ['extra']"),
        (("evaluation",), "evaluation: unknown key(s) ['extra']"),
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


def test_generation_is_a_particle_gun_or_a_sample(doc: dict[str, Any]) -> None:
    doc["evaluation"]["generation"] = {"particle": "mu-", "energy": {"value": 5, "unit": "GeV"}}
    evaluation = parse_problem(doc).evaluation
    assert evaluation.generation["particle"] == "mu-"
    assert evaluation.to_dict()["generation"] == {
        "particle": "mu-",
        "energy": {"value": 5, "unit": "GeV"},
    }

    doc["evaluation"]["generation"] = {"sample": "samples/zh.hepmc"}
    evaluation = parse_problem(doc).evaluation
    assert evaluation.generation == {"sample": "samples/zh.hepmc"}
    assert parse_problem(parse_problem(doc).to_dict()) == parse_problem(doc)


@pytest.mark.parametrize(
    ("generation", "message"),
    [
        ({}, r"evaluation\.generation: set a particle gun .* or a sample"),
        (
            {"particle": "mu-", "sample": "zh.hepmc"},
            r"generation: sample cannot be combined with particle gun keys \['particle'\]",
        ),
        ({"sample": ""}, r"evaluation\.generation\.sample: must be a non-empty string"),
        ({"sample": 3}, r"evaluation\.generation\.sample: must be a non-empty string"),
        ([1], r"evaluation\.generation: must be a mapping, got list"),
        ("gamma", r"evaluation\.generation: must be a mapping, got str"),
    ],
)
def test_generation_rejects_wrong_shapes(
    doc: dict[str, Any], generation: Any, message: str
) -> None:
    doc["evaluation"]["generation"] = generation
    with pytest.raises(ProblemConfigError, match=message):
        parse_problem(doc)


def test_unknown_keys_inside_settings_are_allowed(doc: dict[str, Any]) -> None:
    doc["evaluation"]["settings"] = {"anything": {"goes": [1, "x", None, True, 2.5]}}
    assert parse_problem(doc).evaluation.to_dict()["settings"] == {
        "anything": {"goes": [1, "x", None, True, 2.5]}
    }


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("environment", [], "environment: must be a mapping, got list"),
        ("environment", {}, "missing required key(s) ['key4hep']"),
        ("geometry", "a.xml", "geometry: must be a mapping, got str"),
        ("parameters", {}, "parameters: must be a list, got dict"),
        ("parameters", [], "parameters: must list at least one parameter"),
        ("parameters", ["x"], "parameters[0]: must be a mapping, got str"),
        ("evaluation", "ddsim", "evaluation: must be a mapping"),
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
        (lambda d: d["geometry"].update(entry="  "), "geometry.entry: must be a non-empty string"),
        (lambda d: d["geometry"].clear(), "missing required key(s) ['entry']"),
        (lambda d: d.update(name=""), "name: must be a non-empty string"),
        (
            lambda d: d["evaluation"].update(evaluator=""),
            "evaluation.evaluator: must be a non-empty",
        ),
        (lambda d: d["evaluation"].update(version=1), "evaluation.version: must be a non-empty"),
        (lambda d: d["evaluation"].update(events=0), "events: must be >= 1"),
        (lambda d: d["evaluation"].update(events=True), "events: must be an integer, got bool"),
        (lambda d: d["evaluation"].update(events=1.5), "events: must be an integer, got float"),
        (lambda d: d["evaluation"].update(seed=True), "seed: must be an integer, got bool"),
        (lambda d: d["evaluation"].update(seed="42"), "seed: must be an integer, got str"),
        (lambda d: d["evaluation"].update(settings=[1]), "settings: must be a mapping, got list"),
        (lambda d: d["evaluation"].update(settings={"a": {1, 2}}), "settings.a: unsupported value"),
        (lambda d: d["evaluation"].update(settings={"a": float("nan")}), "non-finite"),
        (
            lambda d: d["evaluation"].update(settings={"a": datetime.date(2026, 1, 1)}),
            "settings.a: unsupported",
        ),
        (
            lambda d: d["evaluation"].update(settings={1: "x"}),
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
    doc["geometry"]["entry"] = " padded/Entry.xml "
    doc["objectives"][0]["metric"] = " odd Metric "
    problem = parse_problem(doc)
    assert problem.geometry.entry == " padded/Entry.xml "
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
    path = write(good_yaml().replace("bounds: [4.0, 8.0]", "bounds: [8.0, 4.0]"))
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
    ],
)
def test_yaml_level_errors(write: Write, text: str) -> None:
    with pytest.raises(ProblemConfigError, match="invalid YAML"):
        load_problem(write(text))


def test_duplicate_key_message_has_the_line(write: Write) -> None:
    with pytest.raises(ProblemConfigError, match=r"duplicate key 'geometry' \(line 3\)"):
        load_problem(write("name: x\ngeometry: {entry: a}\ngeometry: {entry: b}\n"))


def test_nested_duplicate_keys(write: Write) -> None:
    text = good_yaml().replace("  seed: 42\n", "  seed: 42\n  seed: 43\n")
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
    text = good_yaml().replace("    collection:", "    when: 2026-10-20\n    collection:")
    with pytest.raises(ProblemConfigError, match=r"settings\.when: unsupported value of type date"):
        load_problem(write(text))
    text = good_yaml().replace("    collection:", "    blob: !!binary aGk=\n    collection:")
    with pytest.raises(
        ProblemConfigError, match=r"settings\.blob: unsupported value of type bytes"
    ):
        load_problem(write(text))
    text = good_yaml().replace("    collection:", "    s: !!set {a, b}\n    collection:")
    with pytest.raises(ProblemConfigError, match=r"settings\.s: unsupported value of type set"):
        load_problem(write(text))
    text = good_yaml().replace("    collection:", "    x: .nan\n    collection:")
    with pytest.raises(ProblemConfigError, match="non-finite"):
        load_problem(write(text))


def test_recursive_aliases_are_rejected(write: Write) -> None:
    text = good_yaml().replace("    collection:", "    loop: &a [*a]\n    collection:")
    with pytest.raises(ProblemConfigError, match="contains itself"):
        load_problem(write(text))
    text = good_yaml().replace("    collection:", "    loop: &a {x: *a}\n    collection:")
    with pytest.raises(ProblemConfigError, match="contains itself"):
        load_problem(write(text))


def test_recursive_alias_in_a_schema_object_is_a_clean_error(write: Write) -> None:
    text = good_yaml().replace("bounds: [4.0, 8.0]", "bounds: &a [*a, 1]")
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
geometry: {entry: g.xml}
parameters:
  - {geometry: {constant: X}, type: integer, bounds: [1, 3]}
evaluation: {evaluator: e, version: "1"}
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
    doc["evaluation"]["seed"] = huge
    problem = parse_problem(doc)
    assert problem.constraints[0].max == 10**400
    assert parse_problem(problem.to_dict()) == problem


def test_huge_integer_next_to_a_float_bound(doc: dict[str, Any]) -> None:
    doc["parameters"][0]["bounds"] = [1.5, 10**400]
    assert parse_problem(doc).parameters[0].bounds == (1.5, 10**400)
    doc["parameters"][0]["bounds"] = [10**400, 1.5]
    reject(doc, "low must be smaller than high")
