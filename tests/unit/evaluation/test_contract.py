"""The evaluation contract: models, validation and stage-independent evaluators."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from k4opt.errors import K4OptError
from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    Measurement,
    Status,
)

M = Measurement("s", "m", 1)  # any measurement, for results that only need to be valid

# --- two evaluators behind the same protocol, run by hand, no Key4hep, no problem.yaml ---


class FakeSimulation:
    """Reads a prepared input and writes an artifact; it has no metric to report."""

    name = "fake.simulation"
    version = "1"

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if request.workdir is None or "geometry" not in request.inputs:
            raise ValueError("needs a workdir and a 'geometry' input")
        out = request.workdir / "events.txt"
        geometry = request.inputs["geometry"].read_text()
        out.write_text(f"{request.events} events, seed {request.seed}, {geometry}")
        return EvaluationResult.valid(artifacts={"events": out})


@dataclass(frozen=True)
class FakeAnalysis:
    """Reads an artifact and reports a measurement with unit and uncertainty."""

    name: str = "fake.analysis"
    version: str = "2"

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        text = request.inputs["events"].read_text()
        n = int(text.split()[0])
        return EvaluationResult.valid(
            [Measurement("analysis", "n_events", n, unit="1", uncertainty=math.sqrt(n))],
            provenance={"counted": n},
        )


def run(evaluator: Evaluator, request: EvaluationRequest) -> EvaluationResult:
    """Typed usage: mypy checks that the fakes satisfy the protocol."""
    return evaluator.evaluate(request)


def test_artifacts_pass_between_stages_by_hand(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.xml"
    geometry.write_text("<lccdd/>")
    work = tmp_path / "work"
    work.mkdir()

    produced = run(
        FakeSimulation(),
        EvaluationRequest(
            candidate={"constants.cell": 5.5},
            inputs={"geometry": geometry},
            workdir=work,
            events=25,
            seed=7,
        ),
    )
    assert produced.status is Status.VALID
    assert produced.measurements == ()
    assert produced.artifacts["events"] == work / "events.txt"

    measured = run(FakeAnalysis(), EvaluationRequest(inputs=dict(produced.artifacts)))
    (m,) = measured.measurements
    assert (m.stage, m.metric, m.value, m.unit, m.uncertainty) == (
        "analysis",
        "n_events",
        25,
        "1",
        5.0,
    )
    assert measured.provenance == {"counted": 25}
    assert not measured.artifacts


def test_evaluator_identity_is_plain_attributes() -> None:
    assert (FakeSimulation().name, FakeSimulation().version) == ("fake.simulation", "1")
    assert (FakeAnalysis().name, FakeAnalysis().version) == ("fake.analysis", "2")


# --- Measurement ---


def test_measurement_minimal_and_optional_fields() -> None:
    m = Measurement("simulation", "ecal.rms", 1.5)
    assert m.unit is None and m.uncertainty is None  # not provided, which is not zero
    assert Measurement("s", "m", 3, uncertainty=0).uncertainty == 0
    assert Measurement("s", "m", 10**400).value == 10**400  # huge but finite


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, False, "1", None, 1j])
def test_measurement_rejects_non_finite_and_non_numbers(value: Any) -> None:
    with pytest.raises(EvaluationContractError, match="value"):
        Measurement("s", "m", value)


@pytest.mark.parametrize("uncertainty", [-0.1, -1, math.nan, math.inf, True, "0.1"])
def test_measurement_rejects_bad_uncertainty(uncertainty: Any) -> None:
    with pytest.raises(EvaluationContractError, match="uncertainty"):
        Measurement("s", "m", 1.0, uncertainty=uncertainty)


@pytest.mark.parametrize("bad", ["", "  ", None, 3])
def test_measurement_rejects_blank_identifiers(bad: Any) -> None:
    with pytest.raises(EvaluationContractError, match="stage"):
        Measurement(bad, "m", 1)
    with pytest.raises(EvaluationContractError, match="metric"):
        Measurement("s", bad, 1)
    if bad is not None:
        with pytest.raises(EvaluationContractError, match="unit"):
            Measurement("s", "m", 1, unit=bad)


def test_identifiers_are_kept_exactly() -> None:
    m = Measurement(" odd stage ", "a.b", 1)
    assert m.stage == " odd stage "


def test_from_samples_gives_the_mean_and_its_standard_error() -> None:
    m = Measurement.from_samples("s", "m", [2, 4, 4, 4, 5, 5, 7, 9], unit="1")
    assert (m.stage, m.metric, m.unit, m.value) == ("s", "m", "1", 5)
    assert m.uncertainty == pytest.approx(math.sqrt(32 / 7) / math.sqrt(8))  # s with ddof=1


def test_from_samples_of_one_value_has_no_uncertainty_not_a_zero_one() -> None:
    m = Measurement.from_samples("s", "m", [12])
    assert (m.value, m.uncertainty) == (12, None)


def test_from_samples_of_identical_values_has_a_zero_uncertainty() -> None:
    assert Measurement.from_samples("s", "m", [5, 5, 5]).uncertainty == 0


def test_from_samples_needs_samples_and_finite_ones() -> None:
    with pytest.raises(EvaluationContractError, match="no samples"):
        Measurement.from_samples("s", "m", [])
    with pytest.raises(EvaluationContractError, match="finite"):
        Measurement.from_samples("s", "m", [1.0, math.inf])


def test_measurement_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        Measurement("s", "m", 1).value = 2  # type: ignore[misc]


# --- EvaluationResult ---


def test_result_defaults_and_statuses() -> None:
    valid = EvaluationResult.valid([Measurement("s", "m", 1)])
    assert valid.status is Status.VALID and valid.status == "valid"
    assert not valid.artifacts and not valid.provenance and valid.message is None
    for result, status in (
        (EvaluationResult.invalid("too leaky"), Status.INVALID),
        (EvaluationResult.failed("ddsim exited with 1"), Status.FAILED),
    ):
        assert result.status is status
        assert result.measurements == () and not result.artifacts and not result.provenance
    with pytest.raises(EvaluationContractError, match="status"):
        EvaluationResult("ok")  # type: ignore[arg-type]
    assert EvaluationResult("valid", (Measurement("s", "m", 1),)).status is Status.VALID


def test_a_valid_result_may_only_produce_artifacts() -> None:
    result = EvaluationResult.valid(artifacts={"events": Path("events.root")})
    assert result.status is Status.VALID and result.measurements == ()


@pytest.mark.parametrize("make", [EvaluationResult.invalid, EvaluationResult.failed])
def test_invalid_and_failed_results_need_a_message(make: Any) -> None:
    for message in ("", "  ", None):
        with pytest.raises(EvaluationContractError, match="needs a message"):
            make(message)


def test_failed_result_carries_no_invented_numbers() -> None:
    result = EvaluationResult.failed("ddsim exited with 1")
    assert result.measurements == ()
    assert result.message == "ddsim exited with 1"


@pytest.mark.parametrize("make", [EvaluationResult.invalid, EvaluationResult.failed])
def test_invalid_and_failed_results_may_keep_diagnostic_measurements(make: Any) -> None:
    result = make("too leaky", [Measurement("simulation", "containment", 0.4)])
    assert result.measurements == (Measurement("simulation", "containment", 0.4),)
    assert result.message == "too leaky"


def test_measurements_given_as_a_list_are_stored_as_a_tuple() -> None:
    given: list[Measurement] = [Measurement("s", "m", 1)]
    result = EvaluationResult.valid(given)
    given.append(Measurement("s", "other", 2))
    assert result.measurements == (Measurement("s", "m", 1),)


def test_one_result_may_mix_stages() -> None:
    result = EvaluationResult.valid(
        [
            Measurement("simulation", "ecal.rms", 1.0),
            Measurement("reconstruction", "ecal.rms", 2.0),  # same metric name, other stage
            Measurement("analysis", "mass", 91.2, unit="GeV", uncertainty=0.1),
        ]
    )
    assert [m.stage for m in result.measurements] == ["simulation", "reconstruction", "analysis"]


def test_duplicate_stage_metric_is_rejected() -> None:
    with pytest.raises(EvaluationContractError, match=r"duplicate measurement.*'a'.*'s'"):
        EvaluationResult.valid([Measurement("s", "a", 1), Measurement("s", "a", 2)])


def test_measurements_must_be_measurements() -> None:
    with pytest.raises(EvaluationContractError, match=r"measurements\[0\]"):
        EvaluationResult("valid", [("s", "m", 1)])  # type: ignore[arg-type]
    with pytest.raises(EvaluationContractError, match="list or tuple"):
        EvaluationResult("valid", Measurement("s", "m", 1))  # type: ignore[arg-type]


def test_artifact_paths_are_preserved_and_not_checked(tmp_path: Path) -> None:
    missing = tmp_path / "does" / "not" / "exist.root"
    result = EvaluationResult.valid([M], artifacts={"events": missing, "rel": Path("a/../b")})
    assert result.artifacts["events"] is missing
    assert result.artifacts["rel"] == Path("a/../b")
    assert not missing.exists()
    with pytest.raises(EvaluationContractError, match=r"pathlib\.Path"):
        EvaluationResult.valid([M], artifacts={"events": "a.root"})  # type: ignore[dict-item]
    with pytest.raises(EvaluationContractError, match="name"):
        EvaluationResult.valid([M], artifacts={" ": Path("a")})


def test_provenance_is_copied_and_read_only() -> None:
    nested: dict[str, Any] = {"versions": {"dd4hep": "1.30"}, "argv": ["ddsim", "-N", "1"]}
    result = EvaluationResult.valid([M], provenance=nested)
    nested["versions"]["dd4hep"] = "changed"
    nested["argv"].append("x")
    assert result.provenance == {"versions": {"dd4hep": "1.30"}, "argv": ("ddsim", "-N", "1")}
    with pytest.raises(TypeError):
        result.provenance["new"] = 1  # type: ignore[index]
    with pytest.raises(EvaluationContractError, match=r"provenance\.when"):
        EvaluationResult.valid([M], provenance={"when": cast(Any, object())})
    with pytest.raises(EvaluationContractError, match="non-finite"):
        EvaluationResult.valid([M], provenance={"x": math.nan})


def test_result_equality() -> None:
    a = EvaluationResult.valid([Measurement("s", "m", 1.0)], provenance={"k": (1,)})
    b = EvaluationResult.valid([Measurement("s", "m", 1.0)], provenance={"k": (1,)})
    assert a == b
    assert a != EvaluationResult.failed("boom")


# --- EvaluationRequest ---


def test_empty_request_needs_nothing() -> None:
    request = EvaluationRequest()
    assert not request.candidate and not request.settings and not request.inputs
    assert not request.backend_args
    assert request.workdir is None and request.events is None and request.seed is None


def test_request_does_not_touch_the_file_system(tmp_path: Path) -> None:
    workdir = tmp_path / "new"
    EvaluationRequest(workdir=workdir, inputs={"x": tmp_path / "absent.xml"})
    assert not workdir.exists()


def test_settings_are_copied_and_read_only() -> None:
    settings: dict[str, Any] = {"generator": {"particle": "gamma", "theta": [60, 120]}}
    request = EvaluationRequest(settings=settings)
    settings["generator"]["particle"] = "e-"
    settings["generator"]["theta"].append(1)
    assert request.settings == {"generator": {"particle": "gamma", "theta": (60, 120)}}
    with pytest.raises(TypeError):
        request.settings["x"] = 1  # type: ignore[index]
    with pytest.raises(EvaluationContractError, match=r"settings\.bad"):
        EvaluationRequest(settings={"bad": cast(Any, {1, 2})})


def test_candidate_values_are_scalars() -> None:
    candidate: dict[str, Any] = {"constants.cell": 5.5, "layers": 3, "mode": "fast", "flag": True}
    request = EvaluationRequest(candidate=candidate)
    candidate["layers"] = 99
    assert request.candidate["layers"] == 3
    assert request.candidate["flag"] is True
    for bad in (math.nan, math.inf, None, [1], (1, 2), {"a": 1}):
        with pytest.raises(EvaluationContractError, match="candidate"):
            EvaluationRequest(candidate={"p": cast(Any, bad)})
    with pytest.raises(EvaluationContractError, match="candidate parameter"):
        EvaluationRequest(candidate={"": 1})


def test_backend_args_are_preserved_exactly() -> None:
    args = ["--physics.list", "FTFP_BERT", "-v", "-v", "", " spaced ", "--physics.list", "QBBC"]
    source: dict[str, Any] = {"ddsim": args, "k4run": ()}
    request = EvaluationRequest(backend_args=source)
    args.append("--late")
    assert request.backend_args["ddsim"] == (
        "--physics.list",
        "FTFP_BERT",
        "-v",
        "-v",
        "",
        " spaced ",
        "--physics.list",
        "QBBC",
    )
    assert request.backend_args["k4run"] == ()
    assert list(request.backend_args) == ["ddsim", "k4run"]


def test_backend_args_reject_bare_strings_and_non_strings() -> None:
    with pytest.raises(EvaluationContractError, match="list or tuple of strings"):
        EvaluationRequest(backend_args={"ddsim": "--physics.list FTFP_BERT"})  # type: ignore[dict-item]
    with pytest.raises(EvaluationContractError, match=r"\[1\] must be a string"):
        EvaluationRequest(backend_args={"ddsim": ("-N", 5)})  # type: ignore[dict-item]
    with pytest.raises(EvaluationContractError, match="tool"):
        EvaluationRequest(backend_args={"": ("-N",)})


def test_path_inputs_are_preserved(tmp_path: Path) -> None:
    rel = Path("../data/x.root")
    request = EvaluationRequest(inputs={"edm4hep": rel}, workdir=Path("w"))
    assert request.inputs["edm4hep"] is rel
    assert request.workdir == Path("w")
    with pytest.raises(EvaluationContractError, match=r"pathlib\.Path"):
        EvaluationRequest(inputs={"edm4hep": "x.root"})  # type: ignore[dict-item]
    with pytest.raises(EvaluationContractError, match="workdir"):
        EvaluationRequest(workdir="w")  # type: ignore[arg-type]


@pytest.mark.parametrize("events", [0, -1, True, 1.5, "10"])
def test_events_must_be_a_positive_integer(events: Any) -> None:
    with pytest.raises(EvaluationContractError, match="events"):
        EvaluationRequest(events=events)


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_seed_must_be_an_integer(seed: Any) -> None:
    with pytest.raises(EvaluationContractError, match="seed"):
        EvaluationRequest(seed=seed)
    assert EvaluationRequest(seed=0).seed == 0
    assert EvaluationRequest(seed=-3).seed == -3


def test_errors_are_k4opt_errors() -> None:
    assert issubclass(EvaluationContractError, K4OptError)


def test_package_does_not_pull_in_problem_or_geometry() -> None:
    import subprocess
    import sys

    code = (
        "import sys, k4opt.evaluation;"
        "banned = ('k4opt.problem', 'k4opt.geometry', 'lxml');"
        "bad = [m for m in sys.modules if m.startswith(banned)];"
        "sys.exit(1 if bad else 0)"
    )
    assert (
        subprocess.run(
            [sys.executable, "-c", code], check=False, cwd=Path(__file__).parents[3]
        ).returncode
        == 0
    )


@pytest.mark.parametrize("field", ["candidate", "settings", "inputs", "backend_args"])
def test_request_mappings_must_be_mappings(field: str) -> None:
    with pytest.raises(EvaluationContractError, match=field):
        EvaluationRequest(**{field: [("a", 1)]})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["artifacts", "provenance"])
def test_result_mappings_must_be_mappings(field: str) -> None:
    with pytest.raises(EvaluationContractError, match=field):
        EvaluationResult.valid([M], **{field: [("a", 1)]})  # type: ignore[arg-type]


def test_message_must_be_text() -> None:
    with pytest.raises(EvaluationContractError, match="message"):
        EvaluationResult.failed(cast(Any, ValueError("boom")))


@pytest.mark.parametrize(
    "args", [{"--physics.list": "FTFP_BERT"}, {"-v", "-N"}, iter(["-v"]), "-v"]
)
def test_backend_args_must_be_ordered_sequences(args: Any) -> None:
    with pytest.raises(EvaluationContractError, match="list or tuple of strings"):
        EvaluationRequest(backend_args={"ddsim": args})


@pytest.mark.parametrize("measurements", [{Measurement("s", "m", 1)}, iter([]), {"a": 1}])
def test_measurements_must_be_ordered_sequences(measurements: Any) -> None:
    with pytest.raises(EvaluationContractError, match="list or tuple"):
        EvaluationResult("valid", measurements)


# --- the generation context of a request ---


def test_the_generation_context_is_copied_and_read_only() -> None:
    gun: dict[str, Any] = {"particle": "gamma", "energy": {"value": 10, "unit": "GeV"}}
    request = EvaluationRequest(generation=gun)
    gun["energy"]["value"] = 1
    assert request.generation == {"particle": "gamma", "energy": {"value": 10, "unit": "GeV"}}
    with pytest.raises(TypeError):
        request.generation["particle"] = "mu-"  # type: ignore[index]
    assert not EvaluationRequest().generation
    assert EvaluationRequest(generation={"sample": "zh.hepmc"}).generation == {"sample": "zh.hepmc"}


def test_the_generation_context_must_be_a_mapping() -> None:
    with pytest.raises(EvaluationContractError, match="generation"):
        EvaluationRequest(generation=[("sample", "zh.hepmc")])  # type: ignore[arg-type]
