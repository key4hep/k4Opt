"""A candidate whose evaluator reads a reconstruction: the fake ddsim, then the fake k4run of
``tests/support/fake_stack`` (a real process) from a copy of the job's directory."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.evaluation import EvaluationContractError, edm4hep
from k4opt.execution import EvaluationSession, ProvenanceError, Rejected, RunError
from k4opt.k4run import properties
from k4opt.record import SUMMARY_NAME
from tests.support import simulation, stack
from tests.support.evaluators import implementations, reco_cells
from tests.support.simulation import COLLECTION, events_with
from tests.support.stack import install_k4run
from tests.unit.execution.session.conftest import RADIUS, run_candidate


@pytest.fixture
def job(tmp_path: Path) -> Path:
    """A reconstruction's directory: its options file and a file it reads where it runs."""
    directory = tmp_path / "CLDConfig"
    (directory / "PandoraSettingsCLD").mkdir(parents=True)
    (directory / "CLDReconstruction.py").write_text("# the job\n")
    (directory / "PandoraSettingsCLD" / "Likelihood.xml").write_text("<data/>\n")
    return directory


@pytest.fixture
def reco_file(problem_file: Path, job: Path) -> Path:
    """The problem, whose evaluator reads a reconstruction of the simulation."""
    doc = yaml.safe_load(problem_file.read_text())
    doc["samples"]["gamma"]["reconstruction"] = {
        "executor": "k4run",
        "directory": str(job),
        "options": "CLDReconstruction.py",
        "geometry": "compactFile",
        "cms": 91,
    }
    doc["evaluators"] = [
        {
            "name": "test.reco_cells",
            "version": "1",
            "stage": "reconstruction",
            "collections": {"hits": COLLECTION},
        }
    ]
    doc["objectives"] = [
        {"stage": "reconstruction", "metric": "ecal_barrel.mean_hit_cells", "direction": "minimize"}
    ]
    problem_file.write_text(yaml.safe_dump(doc))
    return problem_file


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> simulation.FakeDdsim:
    fake = simulation.install(monkeypatch, events_with([1, 1, 2], [4, 5], [6, 7, 8, 9]))
    install_k4run(monkeypatch)
    return fake


def evaluate(problem: Path, output: Path, **changes: Any) -> Any:
    return run_candidate(problem, [f"{RADIUS}=450"], output, evaluators=[reco_cells()], **changes)


def test_the_reconstruction_runs_on_the_candidate_s_geometry_and_simulation(
    tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim, job: Path
) -> None:
    outcome = evaluate(reco_file, tmp_path / "run")
    assert outcome.candidate.status == "valid"
    assert len(fake.calls) == 1
    summary = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
    assert list(summary["samples"]["gamma"]["stages"]) == [
        "generation",
        "simulation",
        "reconstruction",
    ]
    record = summary["samples"]["gamma"]["stages"]["reconstruction"]
    assert record["status"] == "done"
    assert record["reads"] == ["geometry", "gamma/simulation"]
    assert record["varies"] == [RADIUS]
    assert record["events_read"] == 3
    here = tmp_path / "run" / "samples" / "gamma"
    argv = record["argv"]
    assert argv[1] == "CLDReconstruction.py"
    assert "--IOSvc.Input" in argv
    assert argv[argv.index("--IOSvc.Input") + 1] == str(here / "simulation" / "events.edm4hep.root")
    assert any(arg.startswith("--compactFile=") and "geometry" in arg for arg in argv)
    assert record["outputs"]["events"]["path"] == "samples/gamma/reconstruction/events.edm4hep.root"
    assert sorted(record["logs"]) == ["command", "joboptions", "options", "stderr", "stdout"]
    assert sorted(record["files"]) == [
        "directory/CLDReconstruction.py",
        "directory/PandoraSettingsCLD/Likelihood.xml",
    ]
    assert record["files"]["directory/CLDReconstruction.py"]["bytes"] == len("# the job\n")
    assert (here / "reconstruction" / "job" / "PandoraSettingsCLD" / "Likelihood.xml").is_file()
    programs = summary["environment"]["observed"]["programs"]
    assert programs["k4run"]["path"].endswith("fake_stack/bin/k4run")
    assert (
        programs["k4run"]["sha256"]
        == hashlib.sha256(Path(programs["k4run"]["path"]).read_bytes()).hexdigest()
    )
    assert argv[0] == programs["k4run"]["path"]  # the executable identified is the one run
    assert programs["ddsim"]["path"] == simulation.DDSIM
    (metric,) = summary["metrics"]
    assert metric["label"] == "gamma/test.reco_cells/reconstruction/ecal_barrel.mean_hit_cells"


def test_the_session_identifies_every_file_of_the_job_s_directory(
    reco_file: Path, fake: simulation.FakeDdsim, job: Path
) -> None:
    with EvaluationSession.open(reco_file, implementations(reco_cells())) as session:
        inputs = session.identity()["inputs"]
        label = "samples.gamma.reconstruction.directory/PandoraSettingsCLD/Likelihood.xml"
        assert inputs[label]["bytes"] == len("<data/>\n")
        assert "sha256" in inputs[label]
        (job / "PandoraSettingsCLD" / "Likelihood.xml").write_text("<other/>\n")
        with pytest.raises(ProvenanceError, match=r"Likelihood\.xml"):
            session.verify_inputs()


def test_a_job_that_does_not_take_its_arguments_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim
) -> None:
    install_k4run(monkeypatch, refuse="argument --cms: invalid choice: '91'")
    with pytest.raises(EvaluationContractError) as raised:
        evaluate(reco_file, tmp_path / "run")
    assert str(raised.value) == (
        "samples.gamma.reconstruction: the job does not take its arguments: k4run: error: "
        "argument --cms: invalid choice: '91'"
    )
    assert fake.calls == []
    assert not (tmp_path / "run").exists()


def test_a_missing_directory_names_the_variable_that_is_not_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim
) -> None:
    monkeypatch.delenv("K4OPT_NO_SUCH_CONFIG", raising=False)
    doc = yaml.safe_load(reco_file.read_text())
    doc["samples"]["gamma"]["reconstruction"]["directory"] = "$K4OPT_NO_SUCH_CONFIG/CLDConfig"
    reco_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(EvaluationContractError, match=r"\$K4OPT_NO_SUCH_CONFIG not set"):
        evaluate(reco_file, tmp_path / "run")


def test_a_job_that_read_another_geometry_fails_its_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim
) -> None:
    install_k4run(monkeypatch, geometry=["/k4geo/CLD_o2_v07.xml"])
    outcome = evaluate(reco_file, tmp_path / "run")
    assert outcome.candidate.status == "failed"
    record = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
    message = record["samples"]["gamma"]["stages"]["reconstruction"]["message"]
    assert "GeoSvc.detectors is ['/k4geo/CLD_o2_v07.xml']" in message
    assert "that k4opt passed as --compactFile" in message


def test_a_reconstruction_that_lost_events_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim
) -> None:
    read = edm4hep._frames
    monkeypatch.setattr(
        edm4hep,
        "_frames",
        lambda path: read(path)[: 2 if Path(path).parent.name == "reconstruction" else 3],
    )
    outcome = evaluate(reco_file, tmp_path / "run")
    assert outcome.candidate.status == "failed"
    record = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
    stage = record["samples"]["gamma"]["stages"]["reconstruction"]
    assert stage["message"] == (
        "incomplete reconstruction: 2 of the 3 simulated events in events.edm4hep.root"
    )


def test_a_supplied_simulation_is_reconstructed_without_ddsim(
    tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim
) -> None:
    given = tmp_path / "frozen.edm4hep.root"
    given.write_text("frozen")
    outcome = evaluate(reco_file, tmp_path / "run", inputs=[f"gamma/simulation={given}"])
    assert fake.calls == []
    assert outcome.candidate.status == "valid"
    assert not outcome.candidate.represents_candidate  # the radius reaches the simulation
    record = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
    argv = record["samples"]["gamma"]["stages"]["reconstruction"]["argv"]
    assert argv[argv.index("--IOSvc.Input") + 1] == str(given)
    # no stage runs ddsim: the session identifies the k4run it runs, and only that
    assert list(record["environment"]["observed"]["programs"]) == ["k4run"]


def test_a_reconstruction_runs_in_a_worker_and_loses_events_there_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reco_file: Path
) -> None:
    stack.install(monkeypatch)  # the fake ddsim and podio, here and in the workers
    install_k4run(monkeypatch)
    outcome = evaluate(reco_file, tmp_path / "run", workers=1)
    assert outcome.candidate.status == "valid"
    install_k4run(monkeypatch, drop=1)
    outcome = evaluate(reco_file, tmp_path / "lost", workers=1)
    assert outcome.candidate.status == "failed"
    record = json.loads((tmp_path / "lost" / SUMMARY_NAME).read_text())
    stage = record["samples"]["gamma"]["stages"]["reconstruction"]
    assert stage["message"].startswith("incomplete reconstruction: 2 of the 3 simulated events")


def test_the_variable_that_names_a_job_s_directory_stops_a_session_when_it_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reco_file: Path,
    fake: simulation.FakeDdsim,
    job: Path,
) -> None:
    other = tmp_path / "elsewhere" / "CLDConfig"
    other.mkdir(parents=True)
    (other / "CLDReconstruction.py").write_text("# another job\n")
    monkeypatch.setenv("K4OPT_TEST_CONFIG", str(job.parent))
    doc = yaml.safe_load(reco_file.read_text())
    doc["samples"]["gamma"]["reconstruction"]["directory"] = "$K4OPT_TEST_CONFIG/CLDConfig"
    reco_file.write_text(yaml.safe_dump(doc))
    with EvaluationSession.open(reco_file, implementations(reco_cells())) as session:
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "run")
        assert outcome.candidate.status == "valid"
        # the variable points elsewhere now: no candidate runs in another environment
        monkeypatch.setenv("K4OPT_TEST_CONFIG", str(other.parent))
        with pytest.raises(RunError, match="differs in K4OPT_TEST_CONFIG"):
            session.evaluate({RADIUS: 500.0}, tmp_path / "other")
    assert not (tmp_path / "other").exists()
    summary = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
    files = summary["samples"]["gamma"]["stages"]["reconstruction"]["files"]
    assert files["directory/CLDReconstruction.py"]["path"] == str(job / "CLDReconstruction.py")
    copied = tmp_path / "run" / "samples" / "gamma" / "reconstruction" / "job"
    assert (copied / "CLDReconstruction.py").read_text() == "# the job\n"


def test_a_candidate_whose_value_the_job_does_not_take_is_rejected_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim
) -> None:
    doc = yaml.safe_load(reco_file.read_text())
    del doc["samples"]["gamma"]["reconstruction"]["cms"]
    # the ends of the domain are taken, a value between them is not: they decide nothing
    doc["parameters"]["cms"] = {
        "type": "integer",
        "bounds": [91, 365],
        "targets": {"reconstruction": "cms"},
    }
    reco_file.write_text(yaml.safe_dump(doc))
    install_k4run(monkeypatch, refuse_args={"--cms=250": "argument --cms: invalid choice: '250'"})
    with EvaluationSession.open(reco_file, implementations(reco_cells())) as session:
        rejected = session.evaluate({RADIUS: 450.0, "cms": 250}, tmp_path / "a")
        assert rejected == Rejected(
            "gamma/reconstruction",
            "the job does not take its arguments: k4run: error: argument --cms: invalid choice: "
            "'250' (with cms = 250, as the candidate sets them)",
        )
        assert fake.calls == [] and not (tmp_path / "a").exists()
        outcome = session.evaluate({RADIUS: 450.0, "cms": 240}, tmp_path / "b")
    assert not isinstance(outcome, Rejected) and outcome.candidate.status == "valid"
    record = json.loads((tmp_path / "b" / SUMMARY_NAME).read_text())
    assert "--cms=240" in record["samples"]["gamma"]["stages"]["reconstruction"]["argv"]


def test_a_parameter_reaches_a_key_of_a_map_and_the_record_says_what_it_varied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim
) -> None:
    key = "Pandora.Parameters[Cut]"
    doc = yaml.safe_load(reco_file.read_text())
    doc["parameters"]["cut"] = {
        "type": "continuous",
        "bounds": [0.1, 5.0],
        "targets": {"reconstruction": key},
    }
    reco_file.write_text(yaml.safe_dump(doc))
    install_k4run(monkeypatch)
    with EvaluationSession.open(reco_file, implementations(reco_cells())) as session:
        for name, cut in (("a", 0.5), ("b", 2.0)):
            outcome = session.evaluate({RADIUS: 450.0, "cut": cut}, tmp_path / name)
            assert not isinstance(outcome, Rejected) and outcome.candidate.status == "valid"
    assert len(fake.calls) == 1  # the threshold does not reach the simulation, which is reused
    for name, cut in (("a", 0.5), ("b", 2.0)):
        record = json.loads((tmp_path / name / SUMMARY_NAME).read_text())
        stage = record["samples"]["gamma"]["stages"]["reconstruction"]
        assert stage["definition"][key] == cut
        assert stage["varies"] == ["cut", RADIUS]  # in problem order: the document's is sorted
        assert record["samples"]["gamma"]["stages"]["simulation"]["varies"] == [RADIUS]
        assert not any("Cut" in argument for argument in stage["argv"])
        written = (tmp_path / name / stage["logs"]["options"]).read_text()
        assert f"_set('Pandora', 'Parameters', 'Cut', [{str(cut)!r}])" in written
        assert properties((tmp_path / name / stage["logs"]["joboptions"]).read_text())[
            "Pandora.Parameters"
        ] == {"Cut": [str(cut)]}


def test_the_data_a_job_reads_is_passed_and_identified_and_a_change_is_caught(
    tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim
) -> None:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "noise.root").write_text("noise")
    doc = yaml.safe_load(reco_file.read_text())
    doc["samples"]["gamma"]["reconstruction"]["data"] = {"dataFolder": f"{calibration}/"}
    reco_file.write_text(yaml.safe_dump(doc))
    with EvaluationSession.open(reco_file, implementations(reco_cells())) as session:
        label = "samples.gamma.reconstruction.data.dataFolder/noise.root"
        assert session.identity()["inputs"][label]["bytes"] == len("noise")
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "run")
        assert outcome.candidate.status == "valid"
        record = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
        stage = record["samples"]["gamma"]["stages"]["reconstruction"]
        assert f"--dataFolder={calibration}/" in stage["argv"]
        assert stage["files"]["data.dataFolder/noise.root"]["bytes"] == len("noise")
        (calibration / "noise.root").write_text("other noise")
        with pytest.raises(ProvenanceError, match=r"noise\.root"):
            session.evaluate({RADIUS: 450.0}, tmp_path / "again")


def test_missing_data_and_a_job_s_link_loop_are_refused_when_the_session_opens(
    tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim, job: Path
) -> None:
    doc = yaml.safe_load(reco_file.read_text())
    doc["samples"]["gamma"]["reconstruction"]["data"] = {"dataFolder": str(tmp_path / "none")}
    reco_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(
        EvaluationContractError, match=r"reconstruction\.data\.dataFolder: .* is no"
    ):
        EvaluationSession.open(reco_file, implementations(reco_cells()))
    del doc["samples"]["gamma"]["reconstruction"]["data"]
    reco_file.write_text(yaml.safe_dump(doc))
    (job / "loop").symlink_to(".")
    with pytest.raises(EvaluationContractError, match="is a link to a directory it lies in"):
        EvaluationSession.open(reco_file, implementations(reco_cells()))


@pytest.mark.parametrize("where", ["data", "job"])
def test_a_file_added_to_a_directory_the_job_reads_stops_the_next_candidate(
    tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim, job: Path, where: str
) -> None:
    calibration = tmp_path / "calibration"
    calibration.mkdir()  # empty when the session opens
    doc = yaml.safe_load(reco_file.read_text())
    doc["samples"]["gamma"]["reconstruction"]["data"] = {"dataFolder": f"{calibration}/"}
    reco_file.write_text(yaml.safe_dump(doc))
    with EvaluationSession.open(reco_file, implementations(reco_cells())) as session:
        assert session.evaluate({RADIUS: 450.0}, tmp_path / "run").candidate.status == "valid"
        added = (calibration if where == "data" else job) / "maps.root"
        added.write_text("map")
        label = "data.dataFolder" if where == "data" else "directory"
        with pytest.raises(ProvenanceError, match=rf"reconstruction\.{label}/maps\.root appeared"):
            session.evaluate({RADIUS: 450.0}, tmp_path / "again")
        assert not (tmp_path / "again").exists()
        added.unlink()  # as it was: the session goes on
        assert session.evaluate({RADIUS: 450.0}, tmp_path / "third").candidate.status == "valid"


def test_a_link_in_the_job_s_directory_that_leads_nowhere_is_refused_when_the_session_opens(
    tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim, job: Path
) -> None:
    (job / "PandoraSettingsCLD" / "old.xml").symlink_to(tmp_path / "removed.xml")
    with pytest.raises(EvaluationContractError, match=r"old\.xml is a link to .*which does not"):
        EvaluationSession.open(reco_file, implementations(reco_cells()))


def test_the_same_job_in_another_directory_makes_the_same_reconstruction(
    tmp_path: Path, reco_file: Path, fake: simulation.FakeDdsim, job: Path
) -> None:
    # it runs from a copy: where its files were never reaches it, so it is signed by them
    def signature(output: Path) -> str:
        with EvaluationSession.open(reco_file, implementations(reco_cells())) as session:
            outcome = session.evaluate({RADIUS: 450.0}, output)
        assert outcome.candidate.status == "valid"
        reco = outcome.summary["samples"]["gamma"]["stages"]["reconstruction"]  # type: ignore[index]
        return str(reco["signature"]["sha256"])

    first = signature(tmp_path / "a")
    moved = shutil.copytree(job, tmp_path / "moved" / "CLDConfig")
    doc = yaml.safe_load(reco_file.read_text())
    doc["samples"]["gamma"]["reconstruction"]["directory"] = str(moved)
    reco_file.write_text(yaml.safe_dump(doc))
    assert signature(tmp_path / "b") == first
    (moved / "CLDReconstruction.py").write_text("# another job\n")
    assert signature(tmp_path / "c") != first
