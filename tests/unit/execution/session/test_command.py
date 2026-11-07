"""``k4opt evaluate``, and the stage files it takes with ``--input``."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.evaluation import (
    EvaluationContractError,
    Quantity,
    result_from,
)
from k4opt.execution import RunError
from k4opt.execution._command import preview_candidate
from k4opt.execution.cli import _preview
from k4opt.geometry import GeometrySource
from k4opt.problem import load_problem
from k4opt.record import SUMMARY_NAME
from tests.support import simulation
from tests.support.evaluators import (
    cell_count,
)
from tests.support.problems import add_targets
from tests.support.simulation import events_with
from tests.unit.execution.session.conftest import (
    RADIUS,
    UNEXPECTED,
    Spy,
    evaluate,
    run_candidate,
    two_samples,
)

# --- the command ---------------------------------------------------------------------------


def test_command_prints_the_result_and_exits_zero(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 0, err
    assert out.startswith(f"candidate:   {RADIUS} = 450.0 mm\nstatus:      valid\n")
    assert "evaluator:   test.cell_count 1 on gamma: valid" in out
    assert "simulation/ecal_barrel.mean_hit_cells = 3 +- 0.57735 [1]" in out
    assert "feasible:    yes" in out and "limit:" not in out  # the problem has no constraints
    assert "events:      gamma: 3 of 3" in out
    assert "geometry:    " in out
    assert f"summary:     {tmp_path / 'a' / SUMMARY_NAME}" in out
    assert "ddsim.stderr.log" in out
    assert err.startswith(f"k4opt: warning: {UNEXPECTED}")  # printed before anything ran


def test_a_dry_run_shows_what_the_candidate_sets_and_runs_nothing(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    add_targets(doc["parameters"][RADIUS], simulation="physics.rangecut")
    problem_file.write_text(yaml.safe_dump(doc))
    code, out, err = evaluate(str(problem_file), "--param", f"{RADIUS}=450", "--dry-run")
    assert code == 0, err
    assert out.splitlines() == [
        "Candidate",
        "",
        f"{RADIUS} = 450.0 mm",
        "  geometry: constants.InnerTracker_Barrel_radius_1 (compact/det_v1/tracker.xml)",
        "    400*mm -> 450.0*mm",
        "  simulation: physics.rangecut",
        "    gamma: 450.0*mm",
        "",
        "Configuration validated: the candidate's values, the options of every stage it runs, and "
        "every evaluator's settings and check().",
        "No simulation executed; no evaluation output created.",
    ]
    assert fake.calls == [] and sorted(p.name for p in tmp_path.iterdir()) == ["problem.yaml"]
    # a candidate outside its domain is refused as an evaluation refuses it
    code, _, err = evaluate(str(problem_file), "--param", f"{RADIUS}=50", "--dry-run")
    assert code == 1 and "outside the bounds" in err


def test_a_dry_run_checks_a_problem_whose_evaluations_have_time_limits(
    tmp_path: Path, problem_file: Path
) -> None:
    # a dry run evaluates nothing, so it needs no worker to keep the limits
    doc = yaml.safe_load(problem_file.read_text())
    doc["timeout"] = 600
    doc["evaluators"][0]["timeout"] = 60
    problem_file.write_text(yaml.safe_dump(doc))
    code, out, err = evaluate(str(problem_file), "--param", f"{RADIUS}=450", "--dry-run")
    assert code == 0, err
    assert "Configuration validated" in out
    code, _, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 1 and "cannot be stopped when its time runs out" in err
    assert not (tmp_path / "a").exists()


def test_a_dry_run_refuses_the_options_of_a_run(problem_file: Path) -> None:
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--dry-run", "--workers", "2"
    )
    assert (code, out) == (2, "")
    assert "k4opt evaluate: --workers says how a candidate runs, and --dry-run runs none" in err


def test_a_dry_run_shows_a_setting_in_every_entry_of_its_binding(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    # one binding, two entries on disjoint samples: the threshold reaches both
    doc = two_samples(problem_file)
    entry = doc["evaluators"][0]
    doc["evaluators"] = [entry | {"samples": ["gamma"]}, entry | {"samples": ["pion"]}]
    for objective in doc["objectives"]:
        objective["evaluator"] = "test.cell_count"
    doc["parameters"]["threshold"] = {
        "type": "continuous",
        "bounds": [30.0, 150.0],
        "unit": "keV",
        "targets": {"evaluators": {"test.cell_count": "threshold"}},
    }
    problem_file.write_text(yaml.safe_dump(doc, sort_keys=False))

    class Thresholded(cell_count()):  # type: ignore[misc]
        settings = {"threshold": Quantity("GeV")}

    preview = preview_candidate(
        problem_file, [f"{RADIUS}=450", "threshold=70"], implementations=[Thresholded]
    )
    out = io.StringIO()
    _preview(preview, out)
    lines = out.getvalue().splitlines()
    at = lines.index("threshold = 70.0 keV")
    assert lines[at + 1 : at + 4] == [
        "  evaluator test.cell_count: threshold",
        "    gamma: 70.0 keV",
        "    pion: 70.0 keV",
    ]


@pytest.mark.parametrize("flags", [[], ["--dry-run", "--output", "x"]])
def test_evaluate_needs_an_output_or_a_dry_run(problem_file: Path, flags: list[str]) -> None:
    code, _, err = evaluate(str(problem_file), "--param", f"{RADIUS}=450", *flags)
    assert code == 2 and "give exactly one of --output DIR and --dry-run" in err


def test_a_setting_a_parameter_cannot_set_names_that_parameter(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text())
    add_targets(doc["parameters"][RADIUS], evaluators={"test.cell_count": "threshold"})
    problem_file.write_text(yaml.safe_dump(doc))

    class Thresholded(cell_count()):  # type: ignore[misc]
        settings = {"threshold": Quantity("GeV")}

    with pytest.raises(EvaluationContractError) as info:
        run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[Thresholded])
    assert str(info.value) == (
        f"parameters.{RADIUS}.targets.evaluators.test.cell_count: setting 'threshold' expects a "
        "quantity in GeV, a unit of energy (eV, keV, MeV, GeV, TeV), but the parameter's values "
        "are numbers in mm"
    )
    assert not (tmp_path / "a").exists()


def test_command_exits_nonzero_for_a_failed_run(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, returncode=1, stderr="boom")
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 1
    assert "status:      failed" in out
    assert "feasible:    not assessed" in out
    assert "events:" not in out  # ddsim wrote no file, so no events were counted
    assert "k4opt: evaluation failed: gamma/simulation failed: ddsim exited with status 1" in err
    assert "evaluator:   test.cell_count 1 on gamma: blocked" in out


def test_command_exits_nonzero_for_an_invalid_run(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, events_with([], [], []))
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--output", str(tmp_path / "a")
    )
    assert code == 1
    assert "status:      invalid" in out
    assert "metric:" not in out
    assert "no hits" in err


def test_command_reports_a_bad_candidate_without_creating_anything(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    code, out, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=9999", "--output", str(tmp_path / "a")
    )
    assert code == 1
    assert out == ""
    assert err.startswith("k4opt: error: ")
    assert "outside the bounds" in err
    assert not (tmp_path / "a").exists()


def test_a_generation_file_is_simulated_and_its_hash_recorded(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    hepmc = tmp_path / "zh.hepmc"
    hepmc.write_bytes(b"HepMC::Version 3\n")
    doc = yaml.safe_load(problem_file.read_text(encoding="utf-8"))
    doc["samples"]["gamma"]["generation"] = {"file": "zh.hepmc"}  # next to the problem file
    problem_file.write_text(yaml.safe_dump(doc), encoding="utf-8")

    outcome = run_candidate(problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "s")

    ((argv, _),) = fake.calls
    assert f"--inputFiles={hepmc}" in argv and not any(a.startswith("--gun") for a in argv)
    generation = outcome.summary["samples"]["gamma"]["stages"]["generation"]  # type: ignore[index]
    assert generation["file"] == str(hepmc)
    assert generation["sha256"] == hashlib.sha256(hepmc.read_bytes()).hexdigest()
    assert outcome.candidate.status == "valid" and outcome.candidate.represents_candidate


def test_a_missing_generation_file_is_a_clean_error_before_any_side_effect(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(problem_file.read_text(encoding="utf-8"))
    doc["samples"]["gamma"]["generation"] = {"file": "gone.hepmc"}
    problem_file.write_text(yaml.safe_dump(doc), encoding="utf-8")
    output = tmp_path / "runs" / "s"
    with pytest.raises(EvaluationContractError, match=r"samples\.gamma\.generation\.file"):
        run_candidate(problem_file, [f"{RADIUS}=450.0"], output)
    assert not output.exists() and fake.calls == []


# --- stage files supplied on the command line ----------------------------------------------


def test_a_supplied_simulation_is_evaluated_but_never_counts_as_the_candidate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    events_file = tmp_path / "earlier.edm4hep.root"
    events_file.write_bytes(b"fake")
    code, out, err = evaluate(
        str(problem_file),
        "--param",
        f"{RADIUS}=450",
        "--input",
        f"gamma/simulation={events_file}",
        "--output",
        str(tmp_path / "a"),
    )
    assert code == 0, err
    assert fake.calls == []  # nothing was simulated
    assert f"supplied:    gamma/simulation from {events_file}" in out
    assert "this is no evaluation of the candidate" in out
    assert "feasible:    not assessed" in out  # whatever it measured
    summary = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    supplied = (False, ["gamma/simulation"])
    assert (summary["represents_candidate"], summary["supplied"]) == supplied
    assert summary["samples"]["gamma"]["stages"] == {
        "simulation": {
            "status": "done",
            "message": None,
            # what it stands for, as the problem defines it for the candidate
            "definition": {
                "executor": "ddsim",
                "events": 3,
                "seed": 5,
                "physics.list": "FTFP_BERT",
            },
            "provided": str(events_file),
            "bytes": 4,
            "sha256": hashlib.sha256(b"fake").hexdigest(),
            "events_read": 3,
            # it ran nowhere and made nothing: the file is the caller's, by its path
            "directory": None,
            "outputs": {},
            "logs": {},
            "reads": [],  # it is given, so it reads nothing
            "blocked_by": [],
            "varies": [RADIUS],  # what the stage it stands for depends on
            "elapsed_seconds": summary["samples"]["gamma"]["stages"]["simulation"][
                "elapsed_seconds"
            ],
            # what k4opt held while it read the file; no program ran
            "peak_memory_bytes": {
                "process": summary["samples"]["gamma"]["stages"]["simulation"]["peak_memory_bytes"][
                    "process"
                ],
                "program": None,
            },
        }
    }
    assert summary["metrics"][0]["value"] == 3.0


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ("gamma/simulation", "expected SAMPLE/STAGE=FILE"),
        ("gamma/analysis={file}", "unknown stage; use generation, simulation, reconstruction"),
        ("gamma/simulation={missing}", "is not a file"),
        ("gamma/generation={file}", "no evaluator of the problem reads that stage of gamma"),
        # a stage of which sample: the problem says, even with only one
        (
            "simulation={file}",
            "input 'simulation': name the sample too, as SAMPLE/simulation; the problem's "
            "samples are gamma",
        ),
        ("pion/simulation={file}", "unknown sample 'pion'; the samples are gamma"),
    ],
)
def test_unusable_inputs_are_rejected_before_any_side_effect(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    guard: list[str],
    given: str,
    message: str,
) -> None:
    supplied = tmp_path / "mine.root"
    supplied.write_bytes(b"fake")
    argument = given.format(file=supplied, missing=tmp_path / "gone.root")
    if message.startswith("no evaluator"):
        # the evaluator reads the simulation, which the caller supplies too: generation is unused
        argument = [argument, f"gamma/simulation={supplied}"]
    output = tmp_path / "a"
    with pytest.raises(RunError, match=message):
        run_candidate(
            problem_file,
            [f"{RADIUS}=450"],
            output,
            inputs=argument if isinstance(argument, list) else [argument],
        )
    assert not output.exists() and fake.calls == [] and "materialize" not in guard


def test_an_input_given_twice_is_rejected(tmp_path: Path, problem_file: Path) -> None:
    supplied = tmp_path / "mine.root"
    supplied.write_bytes(b"fake")
    with pytest.raises(RunError, match="given more than once"):
        run_candidate(
            problem_file,
            [f"{RADIUS}=450"],
            tmp_path / "a",
            inputs=[f"gamma/simulation={supplied}", f"gamma/simulation={supplied}"],
        )


def test_the_command_needs_an_evaluator(problem_file: Path, tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    with pytest.raises(SystemExit):
        main(
            ["evaluate", str(problem_file), "--output", str(tmp_path / "o")], stdout=out, stderr=err
        )
    assert not (tmp_path / "o").exists()


def test_the_command_reports_an_unloadable_evaluator_without_creating_anything(
    problem_file: Path, tmp_path: Path
) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        [
            "evaluate",
            str(problem_file),
            "--evaluator",
            "gone.py:Nope",
            "--output",
            str(tmp_path / "o"),
        ],
        stdout=out,
        stderr=err,
    )
    assert code == 1 and "does not exist" in err.getvalue()
    assert not (tmp_path / "o").exists()


def test_the_evaluator_provenance_is_saved_in_the_summary(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    class Annotating(Spy):
        def evaluate(self, request: Any) -> Any:
            result = result_from(self, super().evaluate(request))
            return dataclasses.replace(result, provenance={"selection": "barrel", "algo": [2, 1]})

    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "a", evaluators=[Annotating()]
    )
    on_disk = json.loads(outcome.summary_path.read_text())
    (evaluator,) = on_disk["samples"]["gamma"]["evaluations"].values()
    assert evaluator["provenance"] == {"selection": "barrel", "algo": [2, 1]}
    assert (
        on_disk["samples"]["gamma"]["stages"]["simulation"]["returncode"] == 0
    )  # the stage's record stays apart


def test_the_preflight_request_has_the_conditions_the_evaluation_gets(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    spy = Spy()
    run_candidate(problem_file, [f"{RADIUS}=450.0"], tmp_path / "runs" / "a", evaluators=[spy])
    (evaluated,) = spy.evaluated
    # The validated request is reused after materialization; check() runs exactly once.
    assert len(spy.checked) == 1
    for checked in spy.checked:
        assert (checked.events, checked.seed) == (3, 5)  # those of the simulation it reads
        for field in ("settings", "events", "seed"):
            assert getattr(checked, field) == getattr(evaluated, field)


@pytest.mark.parametrize("invalid", ["settings", "collections"])
def test_a_plain_evaluator_is_validated_before_materialization(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, invalid: str
) -> None:
    class Plain:
        name = "test.cell_count"
        version = "1"
        stage = "simulation"
        collections = {"hits": cell_count().collections["hits"]}
        metrics = cell_count().metrics
        settings = {"threshold": Quantity("GeV", default=1)}

        def evaluate(self, request: Any) -> Any:
            pytest.fail("invalid requests must be rejected before execution")

    doc = yaml.safe_load(problem_file.read_text())
    if invalid == "settings":
        doc["evaluators"][0]["settings"] = {"threshold": {"value": 1, "unit": "bananas"}}
    else:
        doc["evaluators"][0]["collections"] = {}
    problem_file.write_text(yaml.safe_dump(doc))
    supplied = tmp_path / "events.root"
    supplied.write_bytes(b"fake")
    with pytest.raises(EvaluationContractError, match=invalid):
        run_candidate(
            problem_file,
            [f"{RADIUS}=450"],
            tmp_path / "a",
            evaluators=[Plain()],
            inputs=[f"gamma/simulation={supplied}"],
        )
    assert not (tmp_path / "a").exists() and fake.calls == []


def test_summary_preserves_the_definition_loaded_before_execution(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    definition = load_problem(problem_file).to_dict()

    class Editing(Spy):
        def evaluate(self, request: Any) -> Any:
            problem_file.write_text("replaced while evaluating")
            return super().evaluate(request)

    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[Editing()])
    assert outcome.summary["problem"]["definition"] == definition
    assert json.loads(outcome.summary_path.read_text())["problem"]["definition"] == definition


def test_summary_write_failure_is_a_controlled_error(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from k4opt.execution import _publish

    def denied(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("summary is not writable")

    monkeypatch.setattr(_publish, "write_json", denied)
    with pytest.raises(RunError, match="cannot write summary"):
        run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert len(fake.calls) == 1
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()


def test_the_candidate_time_counts_the_checks_and_every_task_the_geometry_too(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from k4opt.execution import _drive, _session, _work

    class Clock:
        now = 0.0

        def perf_counter(self) -> float:
            return self.now

    clock = Clock()
    for module in (_drive, _session, _work):
        monkeypatch.setattr(module, "time", clock)
    materialize = GeometrySource.materialize

    def slow_materialize(self: GeometrySource, *args: Any, **kwargs: Any) -> Any:
        clock.now += 100.0
        return materialize(self, *args, **kwargs)

    class Checking(Spy):
        @classmethod
        def check(cls, request: Any) -> None:
            clock.now += 0.5
            super().check(request)

    monkeypatch.setattr(GeometrySource, "materialize", slow_materialize)
    outcome = run_candidate(
        problem_file, [f"{RADIUS}=450"], tmp_path / "a", evaluators=[Checking()]
    )
    assert outcome.summary["elapsed_seconds"] == 100.5
    assert outcome.summary["geometry"]["elapsed_seconds"] == 100.0  # type: ignore[index]


def test_an_unwritable_evaluator_directory_is_a_recorded_failure(
    tmp_path: Path,
    problem_file: Path,
    fake: simulation.FakeDdsim,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mkdir = Path.mkdir

    def denied(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name == "test.cell_count":
            raise PermissionError("evaluator directory is not writable")
        mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    outcome = run_candidate(problem_file, [f"{RADIUS}=450"], tmp_path / "a")
    assert outcome.candidate.status == "failed"
    assert "cannot create evaluator directory" in outcome.candidate.message
    assert outcome.summary_path.is_file()
    (evaluation,) = outcome.summary["samples"]["gamma"]["evaluations"].values()  # type: ignore[index]
    assert evaluation["workdir"] is None  # it was not called


def test_the_command_observes_as_it_is_told(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch)
    arguments = [str(problem_file), "--param", f"{RADIUS}=450", "--observe", "none"]
    code, _, err = evaluate(*arguments, "--output", str(tmp_path / "a"))
    assert code == 0, err
    record = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    assert record["samples"]["gamma"]["stages"]["simulation"]["observed"]["level"] == "none"


def test_the_command_refuses_an_unknown_level(tmp_path: Path, problem_file: Path) -> None:
    with pytest.raises(SystemExit):
        evaluate(str(problem_file), "--observe", "everything", "--output", str(tmp_path / "a"))


def test_a_dry_run_observes_nothing_and_says_so(problem_file: Path) -> None:
    code, _, err = evaluate(
        str(problem_file), "--param", f"{RADIUS}=450", "--dry-run", "--observe", "none"
    )
    assert code == 2
    assert "--observe says how a candidate runs, and --dry-run runs none" in err
