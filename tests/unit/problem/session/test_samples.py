"""Several evaluators, several samples and combined evaluators of one candidate."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.ddsim import run as ddsim_run
from k4opt.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    Measurement,
    Quantity,
    Role,
    RoleInput,
)
from k4opt.frozen import thaw
from k4opt.problem import (
    SUMMARY_NAME,
    Assessment,
    EvaluationSession,
    RecordError,
    RunError,
    RunOutcome,
    _files,
)
from k4opt.problem._tasks import Kind, TaskId
from tests.support import simulation
from tests.support.common import scientific
from tests.support.evaluators import (
    CELL_COUNT,
    HIT_ENERGY,
    PARAMETER_ECHO,
    TOY_SIGNIFICANCE,
    YIELDS,
    cell_count,
    hit_energy,
    parameter_echo,
    toy_significance,
    yields,
)
from tests.support.geometry import MINI_COMPACT
from tests.support.simulation import COLLECTION, events_with
from tests.unit.problem.session.conftest import (
    CELLS,
    ECHO,
    ENERGY,
    RADIUS,
    UNEXPECTED,
    change,
    combined,
    declare,
    errors,
    evaluate,
    particle,
    run_candidate,
    two_samples,
)

# --- several evaluators of one candidate ---------------------------------------------------


def evaluate_with(references: list[str], *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    flags = [arg for reference in references for arg in ("--evaluator", reference)]
    code = main(["evaluate", *flags, *argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_the_command_measures_with_several_evaluators_on_one_simulation(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    declare(problem_file, ["test.cell_count", "test.hit_energy"], [CELLS, ENERGY])
    output = tmp_path / "a"
    # in any order on the command line; reported in the order of the problem
    code, out, err = evaluate_with(
        [HIT_ENERGY, CELL_COUNT],
        str(problem_file),
        "--param",
        f"{RADIUS}=450",
        "--output",
        str(output),
    )
    assert code == 0, err
    assert len(fake.calls) == 1  # one simulation for both
    assert sorted(p.name for p in output.iterdir()) == ["geometry", "samples", SUMMARY_NAME]
    assert sorted(p.name for p in (output / "samples" / "gamma").iterdir()) == [
        "evaluators",
        "simulation",
    ]
    assert sorted(p.name for p in (output / "samples" / "gamma" / "evaluators").iterdir()) == [
        "test.cell_count",
        "test.hit_energy",
    ]
    assert out.index("evaluator:   test.cell_count 1 on gamma: valid") < out.index(
        "evaluator:   test.hit_energy 1 on gamma: valid"
    )
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["status"] == "valid"
    evaluations = summary["samples"]["gamma"]["evaluations"]
    assert list(evaluations) == ["test.cell_count", "test.hit_energy"]
    assert [m["label"] for m in summary["metrics"]] == [
        "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
        "gamma/test.hit_energy/simulation/ecal_barrel.mean_hit_energy",
    ]
    assert summary["metrics"][1]["value"] == pytest.approx(10 / 3)  # 4, 2 and 4 hits of 1 GeV
    assert list(summary["samples"]["gamma"]["stages"]) == ["generation", "simulation"]


def test_the_command_reports_each_binding_of_an_implementation_given_once(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluators"].append(doc["evaluators"][0] | {"as": "cells_again"})
    doc["objectives"][0]["evaluator"] = "test.cell_count"
    doc["objectives"].append(doc["objectives"][0] | {"evaluator": "cells_again"})
    problem_file.write_text(yaml.safe_dump(doc))
    output = tmp_path / "a"
    code, out, err = evaluate_with(
        [CELL_COUNT], str(problem_file), "--param", f"{RADIUS}=450", "--output", str(output)
    )
    assert code == 0, err
    assert len(fake.calls) == 1
    assert "evaluator:   test.cell_count 1 on gamma: valid" in out
    assert "evaluator:   cells_again (test.cell_count 1) on gamma: valid" in out
    # two bindings that measure the same twice are worth a word, not a refusal
    assert "evaluators 'test.cell_count' and 'cells_again' bind test.cell_count" in err
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert list(summary["samples"]["gamma"]["evaluations"]) == ["test.cell_count", "cells_again"]
    assert sorted(p.name for p in (output / "samples" / "gamma" / "evaluators").iterdir()) == [
        "cells_again",
        "test.cell_count",
    ]


def test_a_failed_simulation_keeps_the_result_of_an_evaluator_that_needs_none(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=2, stderr="G4Exception: bad volume")
    declare(problem_file, ["test.cell_count", "test.parameter_echo"], [CELLS, ECHO])
    output = tmp_path / "a"
    code, out, err = evaluate_with(
        [CELL_COUNT, PARAMETER_ECHO],
        str(problem_file),
        "--param",
        f"{RADIUS}=450",
        "--output",
        str(output),
    )
    assert code == 1
    assert errors(err).startswith("k4opt: evaluation failed: gamma/simulation failed: ddsim exited")
    assert "evaluator:   test.parameter_echo 1: valid" in out
    summary = json.loads((output / SUMMARY_NAME).read_text())
    assert summary["status"] == "failed"
    (cells,) = summary["samples"]["gamma"]["evaluations"].values()
    (echo,) = summary["evaluations"].values()
    assert (cells["status"], cells["elapsed_seconds"], cells["workdir"]) == ("blocked", None, None)
    assert echo["status"] == "valid" and echo["measurements"][0]["value"] == 450.0
    # the measurement that was made is kept; the missing one is missing, not made up
    assert [m["value"] for m in summary["metrics"]] == [None, 450.0]
    assert summary["samples"]["gamma"]["stages"]["simulation"]["message"].startswith(
        "ddsim exited with status 2"
    )


@pytest.mark.parametrize(
    ("declared", "given", "message"),
    [
        (
            ["test.cell_count", "test.hit_energy"],
            [CELL_COUNT],
            "no evaluator given for 'test.hit_energy' version '1'",
        ),
        (["test.cell_count"], [CELL_COUNT, HIT_ENERGY], "declares no evaluator 'test.hit_energy'"),
        (["test.cell_count"], [CELL_COUNT, CELL_COUNT], "'test.cell_count' is given twice"),
    ],
)
def test_evaluators_that_do_not_fit_the_problem_are_refused_before_any_side_effect(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
    declared: list[str],
    given: list[str],
    message: str,
) -> None:
    declare(problem_file, declared, [CELLS])
    output = tmp_path / "a"
    code, out, err = evaluate_with(
        given, str(problem_file), "--param", f"{RADIUS}=450", "--output", str(output)
    )
    assert (code, out) == (1, "")
    assert err.startswith("k4opt: error: ") and message in err
    assert not output.exists() and fake.calls == [] and "materialize" not in guard


def test_the_summary_is_the_same_for_the_same_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    declare(
        problem_file,
        ["test.cell_count", "test.hit_energy", "test.parameter_echo"],
        [CELLS, ENERGY, ECHO],
    )
    first = run_candidate(
        problem_file,
        [f"{RADIUS}=450"],
        tmp_path / "a",
        evaluators=[cell_count(), hit_energy(), parameter_echo()],
    )
    second = run_candidate(
        problem_file,
        [f"{RADIUS}=450"],
        tmp_path / "b",
        evaluators=[parameter_echo(), hit_energy(), cell_count()],
    )

    def normalized(outcome: RunOutcome) -> str:
        """The summary as written, without its directory, its times, peak memory and own id."""
        return json.dumps(scientific(json.loads(outcome.summary_path.read_text())))  # key order

    assert normalized(first) == normalized(second)
    assert first.summary["id"] != second.summary["id"]  # two records, each its own identity


def test_a_misspelled_collection_warns_before_the_run_and_fails_after_it(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluators"][0]["collections"] = {"hits": "InnerTrackerBarelCollection"}
    problem_file.write_text(yaml.safe_dump(doc))
    code, _, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    # the geometry suggests the readout meant, but only the simulation's output decides
    assert err.startswith(
        "k4opt: warning: evaluator 'test.cell_count': ddsim is not expected to write the "
        "collection 'InnerTrackerBarelCollection' (hits) for "
        f"{MINI_COMPACT}; did you mean 'InnerTrackerBarrelCollection'?"
    )
    assert code == 1 and len(fake.calls) == 1
    summary = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    (evaluator,) = summary["samples"]["gamma"]["evaluations"].values()
    assert (evaluator["status"], evaluator["elapsed_seconds"]) == ("failed", None)
    assert evaluator["message"] == (
        "collection 'InnerTrackerBarelCollection' (hits) is not in events.edm4hep.root, which "
        f"holds: {COLLECTION}"
    )
    assert len(summary["warnings"]) == 1


def test_a_setting_with_a_unit_is_recorded_as_written_and_as_received(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    doc["evaluators"][0]["settings"] = {"threshold": {"value": 70, "unit": "keV"}}
    problem_file.write_text(yaml.safe_dump(doc))
    gated = type("Gated", (cell_count(),), {"settings": {"threshold": Quantity("GeV")}})
    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[gated])
    on_disk = json.loads(outcome.summary_path.read_text())
    (evaluator,) = on_disk["samples"]["gamma"]["evaluations"].values()
    assert evaluator["settings"] == {
        "declared": {"threshold": {"value": 70, "unit": "keV"}},
        "used": {"threshold": 70e-6},
    }


# --- several samples: each simulated once, on one geometry, measured on its own -------------


def test_each_sample_is_simulated_once_on_the_one_geometry_and_measured_on_its_own(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, guard: list[str]
) -> None:
    two_samples(problem_file)
    session = EvaluationSession.open(problem_file, [cell_count()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert guard == ["discover", "materialize"]  # one geometry for both
    assert [particle(argv) for argv, _ in fake.calls] == ["gamma", "pi-"]
    compact = str(tmp_path / "a" / "geometry" / MINI_COMPACT)
    assert all(f"--compactFile={compact}" in argv for argv, _ in fake.calls)
    # each sample's stages and evaluations in a directory of its own
    for sample in ("gamma", "pion"):
        made = tmp_path / "a" / "samples" / sample
        assert sorted(p.name for p in made.iterdir()) == ["evaluators", "simulation"]
        assert (made / "evaluators" / "test.cell_count").is_dir()
    record = outcome.candidate
    assert record.status == "valid"
    assert list(record.evaluations) == ["gamma/test.cell_count", "pion/test.cell_count"]
    gamma, pion = record.records
    assert gamma.request.seed == 5 and pion.request.seed == 6  # each its own sample's
    assert gamma.request.inputs["simulation"] != pion.request.inputs["simulation"]
    summary = outcome.summary
    assert list(summary["samples"]) == ["gamma", "pion"]  # type: ignore[call-overload]
    assert [m["label"] for m in summary["metrics"]] == [  # type: ignore[attr-defined]
        "gamma/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
        "pion/test.cell_count/simulation/ecal_barrel.mean_hit_cells",
    ]
    assert [o.label for o in outcome.assessment.objectives or ()] == [
        m["label"]
        for m in summary["metrics"]  # type: ignore[attr-defined]
    ]
    assert outcome.events == {"gamma": (3, 3), "pion": (3, 3)}
    written = json.loads(outcome.summary_path.read_text())
    assert Assessment.from_summary(session.spec, written) == outcome.assessment
    # one warning for the binding, however many samples ddsim simulates for it
    assert len([w for w in session.warnings if w.startswith(UNEXPECTED)]) == 1


def test_the_command_reports_each_evaluation_with_its_sample(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    two_samples(problem_file)
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 0, err
    assert out.index("evaluator:   test.cell_count 1 on gamma: valid") < out.index(
        "evaluator:   test.cell_count 1 on pion: valid"
    )
    assert "events:      gamma: 3 of 3" in out and "events:      pion: 3 of 3" in out
    stdout = tmp_path / "a" / "samples" / "pion" / "simulation" / "ddsim.stdout.log"
    assert f"stdout:      pion: {stdout}" in out


def test_a_sample_whose_simulation_fails_blocks_only_its_own_evaluations(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9]))

    def pions_crash(argv: list[str], **kwargs: Any) -> Any:
        code = fake(argv, **kwargs)
        return 2 if particle(argv) == "pi-" else code

    monkeypatch.setattr(ddsim_run, "_execute", pions_crash)
    doc = two_samples(problem_file)
    doc["evaluators"].append({"name": "test.parameter_echo", "version": "1"})
    doc["objectives"].append(
        {"evaluator": "test.parameter_echo", "stage": ECHO[0], "metric": ECHO[1]}
        | {"direction": "minimize"}
    )
    problem_file.write_text(yaml.safe_dump(doc))
    session = EvaluationSession.open(problem_file, [cell_count(), parameter_echo()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    record = outcome.candidate
    statuses = {label: r.result.status for label, r in record.evaluations.items()}
    assert statuses == {
        "gamma/test.cell_count": "valid",
        "pion/test.cell_count": "blocked",
        "test.parameter_echo": "valid",
    }
    assert record.status == "failed"
    message = record.message or ""
    assert message.startswith("pion/simulation failed: ddsim exited with status 2")
    assert message.endswith("; blocked: pion/test.cell_count")
    # what was measured is kept, and nothing stands in for what was not
    assert [m["value"] for m in outcome.summary["metrics"]] == [3.0, None, 450.0]  # type: ignore[attr-defined]
    assert outcome.assessment.objectives is None
    written = json.loads(outcome.summary_path.read_text())
    assert written["samples"]["pion"]["evaluations"]["test.cell_count"]["blocked_by"] == [
        "pion/simulation"
    ]
    assert written["samples"]["gamma"]["stages"]["simulation"]["status"] == "done"
    Assessment.from_summary(session.spec, written)  # and the record holds up


def test_a_file_read_by_two_samples_is_hashed_once_and_named_for_each(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    from k4opt.problem._files import _sha256

    hepmc = tmp_path / "zh.hepmc"  # next to the problem file
    hepmc.write_bytes(b"HepMC::Version 3\n")
    doc = yaml.safe_load(problem_file.read_text())
    doc["samples"]["gamma"]["generation"] = {"file": hepmc.name}
    pion = copy.deepcopy(doc["samples"]["gamma"]) | {"simulation": {"executor": "ddsim"}}
    pion["simulation"] |= {"events": 3, "seed": 6}
    problem_file.write_text(yaml.safe_dump(doc))
    two_samples(problem_file, **pion)
    _sha256.cache_clear()
    session = EvaluationSession.open(problem_file, [cell_count()])
    inputs = session.identity()["inputs"]
    assert list(inputs) == ["samples.gamma.generation.file", "samples.pion.generation.file"]
    assert inputs["samples.gamma.generation.file"] == inputs["samples.pion.generation.file"]
    assert _sha256.cache_info().misses == 1  # read once
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert [f"--inputFiles={hepmc}" in argv for argv, _ in fake.calls] == [True, True]
    assert outcome.candidate.status == "valid"


def test_a_supplied_stage_of_one_sample_leaves_the_other_simulated(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    two_samples(problem_file)
    events_file = tmp_path / "earlier.edm4hep.root"
    events_file.write_bytes(b"fake")
    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450"], tmp_path / "a", inputs=[f"pion/simulation={events_file}"]
    )
    assert [particle(argv) for argv, _ in fake.calls] == ["gamma"]
    summary = outcome.summary
    assert (summary["represents_candidate"], summary["supplied"]) == (False, ["pion/simulation"])
    assert list(summary["samples"]["pion"]["stages"]) == ["simulation"]  # type: ignore[index]
    gamma = summary["samples"]["gamma"]["stages"]  # type: ignore[index]
    assert list(gamma) == ["generation", "simulation"]
    assert outcome.assessment.feasible is None  # measured, but not judged


def test_a_file_that_changes_while_the_second_sample_runs_leaves_no_record(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, events_with([1], [2], [3]))
    hepmc = tmp_path / "zh.hepmc"
    hepmc.write_bytes(b"HepMC::Version 3\n")
    doc = two_samples(problem_file)
    doc["samples"]["gamma"]["generation"] = {"file": hepmc.name}
    problem_file.write_text(yaml.safe_dump(doc))

    def rewriting(argv: list[str], **kwargs: Any) -> Any:
        if len(fake.calls) == 1:  # the second simulation, of pion, after gamma read the file
            hepmc.write_bytes(b"HepMC::Version 3\nE 1\n")
        return fake(argv, **kwargs)

    monkeypatch.setattr(ddsim_run, "_execute", rewriting)
    session = EvaluationSession.open(problem_file, [cell_count()])
    with pytest.raises(RunError, match=r"samples\.gamma\.generation\.file: .* changed after"):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert len(fake.calls) == 2
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()
    for sample in ("gamma", "pion"):  # what ran is kept, to look at
        assert (tmp_path / "a" / "samples" / sample / "simulation").is_dir()


def test_samples_with_identical_stages_are_a_warning(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    two_samples(problem_file, **doc["samples"]["gamma"])
    session = EvaluationSession.open(problem_file, [cell_count()])
    assert (
        "samples gamma and pion have identical stages; each is simulated separately"
        in session.warnings
    )
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert len(fake.calls) == 2  # each all the same
    assert outcome.candidate.status == "valid"


# --- combined evaluators: the evaluations of several samples, read by role -------------------


def toy(signal: float, *backgrounds: float) -> float:
    """The toy significance of ``signal`` over the sum of ``backgrounds``."""
    b = sum(backgrounds)
    return math.sqrt(2 * ((signal + b) * math.log1p(signal / b) - signal))


def test_a_combined_evaluator_reads_the_evaluations_of_its_samples_by_role(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    combined(problem_file)
    session = EvaluationSession.open(problem_file, [toy_significance(), yields()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    record = outcome.candidate
    assert record.status == "valid", record.message
    assert list(record.evaluations) == [
        "sig/test.yields",
        "zz/test.yields",
        "ww/test.yields",
        "significance",
    ]
    # every event is selected, so each sample is expected as often as its cross section says
    luminosity = 1e3 * 10.8
    expected = toy(200 * luminosity, 1360 * luminosity, 16400 * luminosity)
    (objective,) = outcome.assessment.objectives or ()
    assert objective.label == "significance/analysis/expected_significance"
    assert objective.value == pytest.approx(expected)
    plan = record.plan[TaskId(Kind.EVALUATOR, "significance")]
    assert [need.label for need in plan.needs] == [  # in the order the roles name them
        "geometry",
        "zz/test.yields",
        "ww/test.yields",
        "sig/test.yields",
    ]
    assert plan.directory == "evaluators/significance"
    summary = json.loads(outcome.summary_path.read_text())
    evaluation = summary["evaluations"]["significance"]
    assert evaluation["roles"] == {
        "signal": ["sig/test.yields"],
        "backgrounds": ["zz/test.yields", "ww/test.yields"],
    }
    assert evaluation["reads"] == [need.label for need in plan.needs]
    assert summary["implementations"]["test.toy_significance"]["roles"] == {
        "signal": {
            "many": False,
            "stage": None,
            "metrics": [
                ["analysis", "selected.sum_weights"],
                ["analysis", "generated.sum_weights"],
            ],
            "artifacts": [],
        },
        "backgrounds": {
            "many": True,
            "stage": None,
            "metrics": [
                ["analysis", "selected.sum_weights"],
                ["analysis", "generated.sum_weights"],
            ],
            "artifacts": [],
        },
    }
    assert summary["implementations"]["test.yields"]["roles"] is None
    assert summary["problem"]["definition"]["samples"]["zz"]["properties"] == {
        "cross_section": {"value": 1360, "unit": "fb"}
    }
    assert Assessment.from_summary(session.spec, summary) == outcome.assessment


def test_each_item_of_a_role_holds_what_its_task_made_and_the_sample_it_is_of(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    combined(problem_file)
    seen: dict[str, EvaluationRequest] = {}

    class Watched(toy_significance()):  # type: ignore[misc]
        def check(self, request: EvaluationRequest) -> None:
            super().check(request)
            seen["check"] = request

        def evaluate(self, request: EvaluationRequest) -> Any:
            seen["evaluate"] = request
            return super().evaluate(request)

    session = EvaluationSession.open(problem_file, [Watched, yields()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert seen["check"].roles == {}  # nothing exists to read before the candidate runs
    request = seen["evaluate"]
    run_dir = tmp_path / "a"
    assert dict(request.inputs) == {"geometry": run_dir / "geometry" / MINI_COMPACT}
    assert request.properties == {}  # it measures no sample of its own
    (signal,) = request.roles["signal"]
    zz, ww = request.roles["backgrounds"]
    assert [item.label for item in (signal, zz, ww)] == [
        "sig/test.yields",
        "zz/test.yields",
        "ww/test.yields",
    ]
    upstream = outcome.candidate.record("zz/test.yields")
    assert zz.measurements == upstream.result.measurements
    assert (zz.events, zz.seed) == (3, 2)
    assert zz.properties == {"cross_section": {"value": 1360, "unit": "fb"}}
    assert zz.settings == {"min_energy": 0.0}  # as it received them
    histogram = zz.artifacts["histogram"]
    written = run_dir / "samples" / "zz" / "evaluators" / "test.yields" / "histogram.json"
    assert (histogram.ref, histogram.path) == ("zz/test.yields#histogram", written)
    summary = json.loads(outcome.summary_path.read_text())
    recorded = summary["samples"]["zz"]["evaluations"]["test.yields"]["artifacts"]["histogram"]
    assert histogram.sha256 == recorded["sha256"] and histogram.bytes == recorded["bytes"]


def test_a_combined_evaluator_is_blocked_by_an_input_that_did_not_succeed(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch, events_with([1, 1, 2, 3], [4, 5], [6, 7, 8, 9]))

    def ww_crashes(argv: list[str], **kwargs: Any) -> Any:
        code = fake(argv, **kwargs)
        return 2 if particle(argv) == "pi+" else code

    monkeypatch.setattr(ddsim_run, "_execute", ww_crashes)
    doc = combined(problem_file)
    doc["evaluators"].append({"name": "test.parameter_echo", "as": "readout", "version": "1"})
    doc["constraints"] = [
        {"evaluator": "readout", "stage": ECHO[0], "metric": ECHO[1], "max": 1000}
    ]
    problem_file.write_text(yaml.safe_dump(doc, sort_keys=False))
    session = EvaluationSession.open(problem_file, [toy_significance(), yields(), parameter_echo()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    record = outcome.candidate
    statuses = {label: r.result.status for label, r in record.evaluations.items()}
    assert statuses == {
        "sig/test.yields": "valid",
        "zz/test.yields": "valid",
        "ww/test.yields": "blocked",
        "significance": "blocked",  # never called with a background missing
        "readout": "valid",  # reads neither, so keeps its result
    }
    assert record.results[TaskId(Kind.EVALUATOR, "significance")].blocked_by == (
        TaskId(Kind.STAGE, "simulation", "ww"),
    )
    assert record.status == "failed"
    assert (record.message or "").endswith("; blocked: ww/test.yields, significance")
    assert outcome.assessment.objectives is None
    written = json.loads(outcome.summary_path.read_text())
    blocked_record = written["evaluations"]["significance"]
    assert blocked_record["message"] == "not run: ww/simulation failed"
    assert blocked_record["workdir"] is None
    assert not (tmp_path / "a" / "evaluators" / "significance").exists()
    Assessment.from_summary(session.spec, written)


def test_an_invalid_input_blocks_a_combined_evaluator_and_an_empty_background_is_invalid(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = combined(problem_file)
    plain = copy.deepcopy(doc)
    # one binding name, other settings on the backgrounds: nothing of them is selected
    selection = doc["evaluators"][0]
    selection["samples"] = ["sig"]
    tight = selection | {"samples": ["zz", "ww"], "settings": {"min_energy": {"value": 1e3}}}
    tight["settings"]["min_energy"]["unit"] = "GeV"
    doc["evaluators"].insert(1, tight)
    problem_file.write_text(yaml.safe_dump(doc, sort_keys=False))
    session = EvaluationSession.open(problem_file, [toy_significance(), yields()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    significance = outcome.candidate.record("significance").result
    assert significance.status == "invalid"
    assert significance.message == "no expected background: the toy significance is undefined"
    assert outcome.candidate.status == "invalid"

    class Invalid(yields()):  # type: ignore[misc]
        def evaluate(self, request: EvaluationRequest) -> Any:
            if request.seed == 3:  # ww
                self.invalid("too few events selected")
            return super().evaluate(request)

    problem_file.write_text(yaml.safe_dump(plain, sort_keys=False))
    session = EvaluationSession.open(problem_file, [toy_significance(), Invalid])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    held = outcome.candidate.record("significance").result
    assert (held.status, held.message) == ("blocked", "not run: ww/test.yields invalid")
    # a background that ran but is unusable: nothing failed, and the candidate is invalid
    assert outcome.candidate.status == "invalid"
    assert outcome.candidate.message == (
        "ww/test.yields invalid: too few events selected; blocked: significance"
    )
    Assessment.from_summary(session.spec, json.loads(outcome.summary_path.read_text()))


def test_what_a_role_reads_reaches_the_evaluator_unchanged_and_k4opt_computes_nothing(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    """Signed weights summing below zero, opaque properties and a covariance file are the
    evaluators' to interpret: delivered and recorded as they are."""
    covariance = [[4.0, -1.5], [-1.5, 9.0]]

    class Signed(yields()):  # type: ignore[misc]
        artifacts = {**yields().artifacts, "covariance": "a 2x2 covariance, JSON"}

        def evaluate(self, request: EvaluationRequest) -> Any:
            measured = super().evaluate(request)
            assert request.workdir is not None
            path = request.workdir / "covariance.json"
            path.write_text(json.dumps(covariance))
            values = [
                Measurement(*key, -7.25 if key[1] == "generated.sum_weights" else 1.5, "1", 0.5)
                for key in (
                    ("analysis", "selected.sum_weights"),
                    ("analysis", "generated.sum_weights"),
                )
            ]
            artifacts = dict(measured.artifacts) | {"covariance": path}
            return EvaluationResult.valid(values, artifacts=artifacts)

    seen: list[RoleInput] = []

    class Echo(Evaluator):
        name, version = "test.echo", "1"
        roles = {"inputs": Role(many=True, artifacts=["covariance"])}
        metrics = {("analysis", "seen"): "1"}

        def evaluate(self, request: EvaluationRequest) -> Any:
            seen.extend(request.roles["inputs"])
            return {("analysis", "seen"): len(seen)}

    doc = combined(problem_file)
    properties = {"process": "e+e- -> ZZ", "weights": [-1, 1], "k_factor": None}
    doc["samples"]["zz"]["properties"] = properties
    doc["evaluators"][1] = {
        "name": "test.echo",
        "version": "1",
        "roles": {"inputs": {"samples": ["zz", "ww"], "evaluator": "test.yields"}},
    }
    doc["objectives"] = [{"evaluator": "test.echo", "stage": "analysis", "metric": "seen"}]
    doc["objectives"][0]["direction"] = "maximize"
    problem_file.write_text(yaml.safe_dump(doc, sort_keys=False))
    session = EvaluationSession.open(problem_file, [Echo, Signed])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == "valid"
    zz, _ = seen
    assert zz.measurement("analysis", "generated.sum_weights").value == -7.25
    assert zz.measurement("analysis", "generated.sum_weights").uncertainty == 0.5
    assert thaw(zz.properties) == properties
    assert json.loads(zz.artifacts["covariance"].path.read_text()) == covariance
    summary = json.loads(outcome.summary_path.read_text())
    recorded = summary["samples"]["zz"]["evaluations"]["test.yields"]
    assert {m["metric"]: m["value"] for m in recorded["measurements"]} == {
        "selected.sum_weights": 1.5,
        "generated.sum_weights": -7.25,
    }
    assert recorded["artifacts"]["covariance"]["sha256"] == zz.artifacts["covariance"].sha256
    assert summary["problem"]["definition"]["samples"]["zz"]["properties"] == properties
    # the combined evaluation records what it read by reference, nothing of it twice
    assert summary["evaluations"]["test.echo"]["roles"] == {
        "inputs": ["zz/test.yields", "ww/test.yields"]
    }
    assert "measurements" not in json.dumps(summary["evaluations"]["test.echo"]["roles"])
    Assessment.from_summary(session.spec, summary)


def test_a_role_that_reads_a_stage_gets_its_file_unhashed(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashed: list[str] = []

    def sha256(path: str, *key: int) -> str:
        hashed.append(path)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    monkeypatch.setattr(_files, "_sha256", sha256)  # uncached: every hash is counted
    seen: list[RoleInput] = []

    class Files(Evaluator):
        name, version = "test.files", "1"
        roles = {"events": Role(many=True, stage="simulation")}
        metrics = {("analysis", "bytes"): "1"}

        def evaluate(self, request: EvaluationRequest) -> Any:
            seen.extend(request.roles["events"])
            return {("analysis", "bytes"): sum(i.artifacts["simulation"].bytes for i in seen)}

    doc = combined(problem_file)
    doc["evaluators"][0]["samples"] = ["sig"]  # zz and ww are read by the role alone
    doc["evaluators"][1] = {
        "name": "test.files",
        "version": "1",
        "roles": {"events": {"samples": ["zz", "ww"], "stage": "simulation"}},
    }
    doc["objectives"] = [{"evaluator": "test.files", "stage": "analysis", "metric": "bytes"}]
    doc["objectives"][0]["direction"] = "minimize"
    problem_file.write_text(yaml.safe_dump(doc, sort_keys=False))
    session = EvaluationSession.open(problem_file, [Files, yields()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == "valid"
    assert [particle(argv) for argv, _ in fake.calls] == ["gamma", "pi-", "pi+"]
    zz, _ww = seen
    assert (zz.label, zz.stage, zz.binding, (zz.events, zz.seed)) == (
        "zz/simulation",
        "simulation",
        None,
        (3, 2),
    )
    events = zz.artifacts["simulation"]
    assert events.ref == "zz/simulation#events"
    assert events.path == tmp_path / "a" / "samples" / "zz" / "simulation" / "events.edm4hep.root"
    assert events.sha256 is None and not any(path.endswith(".root") for path in hashed)
    summary = json.loads(outcome.summary_path.read_text())
    assert summary["evaluations"]["test.files"]["reads"] == [
        "geometry",
        "zz/simulation",
        "ww/simulation",
    ]
    assert list(summary["samples"]["zz"]["evaluations"]) == []
    Assessment.from_summary(session.spec, summary)


def test_a_record_whose_roles_disagree_with_its_problem_is_refused(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    combined(problem_file)
    session = EvaluationSession.open(problem_file, [toy_significance(), yields()])
    outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    written = json.loads(outcome.summary_path.read_text())
    roles = "implementations/test.toy_significance/roles"
    for path, value, message in [
        (
            "evaluations/significance/roles/backgrounds",
            ["zz/test.yields"],
            "evaluator 'significance': roles {'backgrounds': ['zz/test.yields'], 'signal': "
            "['sig/test.yields']}, but the problem's are",
        ),
        (
            f"{roles}/signal/many",
            "no",
            "implementation 'test.toy_significance': roles.signal: many 'no' or stage None is "
            "not a role's",
        ),
        (
            f"{roles}/backgrounds/many",
            False,
            "evaluator 'significance': role 'backgrounds' reads one item, given 2",
        ),
        (
            roles,
            None,
            "evaluator 'significance': the problem gives it roles, which it does not declare",
        ),
        (
            f"{roles}/signal/metrics",
            [["analysis", "selected.sum_weights"], ["analysis", "other"]],
            "evaluator 'significance': role 'signal' reads what sig/test.yields does not declare",
        ),
        (
            f"{roles}/signal/metrics",
            [["analysis", "selected.sum_weights"], ["analysis", "selected.sum_weights"]],
            "implementation 'test.toy_significance': roles.signal.metrics: "
            "analysis/selected.sum_weights is named twice",
        ),
        (
            f"{roles}/signal/metrics",
            [["analysis", " "]],
            "implementation 'test.toy_significance': roles.signal.metrics: ['analysis', ' '] is "
            "no [stage, metric]",
        ),
        (
            "samples/sig/evaluations/test.yields/measurements",
            [],
            "evaluator 'sig/test.yields' is valid without [('analysis', 'selected.sum_weights'), "
            "('analysis', 'generated.sum_weights')], which role 'signal' of 'significance' reads",
        ),
    ]:
        tampered = copy.deepcopy(written)
        change(tampered, path, value)
        with pytest.raises(RecordError, match=re.escape(message)):
            Assessment.from_summary(session.spec, tampered)


def test_the_command_reports_what_each_role_of_a_combined_evaluator_read(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    combined(problem_file)
    out, err = io.StringIO(), io.StringIO()
    references = ["--evaluator", TOY_SIGNIFICANCE, "--evaluator", YIELDS]
    argv = [str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")]
    code = main(["evaluate", *references, *argv], stdout=out, stderr=err)
    assert code == 0, err.getvalue()
    report = out.getvalue()
    assert "evaluator:   test.yields 1 on ww: valid\n" in report
    assert (
        "evaluator:   significance (test.toy_significance 1): valid\n"
        "roles:       backgrounds zz/test.yields, ww/test.yields; signal sig/test.yields\n"
        "metric:      analysis/expected_significance = "
    ) in report
