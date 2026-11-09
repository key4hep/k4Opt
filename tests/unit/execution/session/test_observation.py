"""What each stage used: its footprint, what the record keeps of it, the undeclared inputs a session
watches, and when an output of an earlier session may stand in for a stage."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from k4opt.execution import EvaluationSession, ProvenanceError, RunError, _session
from k4opt.problem import load_problem
from k4opt.program import Observation, unobservable
from k4opt.record import SUMMARY_NAME, Assessment, unused_local
from k4opt.software import Software
from k4opt.software.observe import FOOTPRINT, load
from tests.support import stack
from tests.support.evaluators import cell_count
from tests.unit.execution.session.conftest import RADIUS

pytestmark = pytest.mark.skipif(unobservable("files") is not None, reason="strace cannot trace")


def simulated(run: Path) -> dict[str, Any]:
    stage: dict[str, Any] = json.loads((run / SUMMARY_NAME).read_text())["samples"]["gamma"][
        "stages"
    ]["simulation"]
    return stage


def evaluated(
    problem: Path,
    run: Path,
    *,
    observe: Observation = "files",
    reuse: tuple[Path, ...] = (),
    resumed: bool = False,
    radius: float = 450.0,
) -> dict[str, Any]:
    """The simulation's part of the record of one candidate, of a session of its own."""
    with EvaluationSession.open(problem, [cell_count()], observe=observe) as session:
        session.reuse(reuse, resumed=resumed)
        outcome = session.evaluate({RADIUS: radius}, run)
    assert outcome.candidate.status == "valid", outcome.candidate.message
    return simulated(run)


@pytest.fixture
def calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A file the fake ddsim reads that the problem does not name."""
    path = tmp_path / "private" / "calibration.json"
    path.parent.mkdir()
    path.write_text('{"scale": 1.05}\n')
    stack.install(monkeypatch, **{"*": {"read": [str(path)]}})
    return path


def test_the_record_says_what_a_stage_used(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    stage = evaluated(problem_file, tmp_path / "a")
    observed = stage["observed"]
    assert observed["level"] == "files" and observed["complete"] is True
    assert str(calibration) in observed["undeclared"]
    assert observed["network"] == []
    footprint = tmp_path / "a" / stage["logs"]["footprint"]
    assert (
        footprint.name == FOOTPRINT and footprint.stat().st_size == observed["footprint"]["bytes"]
    )
    held = load(footprint)
    assert held.complete and str(calibration) in held.read
    assert not (footprint.parent / "trace").exists()  # the raw traces are gone


def test_a_stage_in_a_worker_is_observed_the_same(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()], workers=1) as session:
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert outcome.candidate.status == "valid", outcome.candidate.message
    assert str(calibration) in simulated(tmp_path / "a")["observed"]["undeclared"]


@pytest.mark.parametrize("observe", ["libraries", "none"])
def test_a_record_says_how_little_was_observed(
    tmp_path: Path, problem_file: Path, calibration: Path, observe: Observation
) -> None:
    observed = evaluated(problem_file, tmp_path / "a", observe=observe)["observed"]
    assert observed["level"] == observe and observed["complete"] is False
    assert not observed.get("undeclared")


def test_a_record_holds_one_digest_of_the_variables_and_holds_up(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a")
    record = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    software = record["environment"]["observed"]["software"]
    assert software["variables"] == Software.now().variables_sha256
    assert software["id"] == Software.now().id
    Assessment.from_summary(load_problem(problem_file), record)


def test_an_undeclared_input_that_changes_stops_the_session(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        calibration.write_text('{"scale": 1.10}\n')
        with pytest.raises(ProvenanceError, match=str(calibration)):
            session.evaluate({RADIUS: 460.0}, tmp_path / "b")


def test_an_input_of_a_reused_output_is_watched_as_one_of_a_stage_that_ran(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.reuse([tmp_path / "a"])
        session.evaluate({RADIUS: 450.0}, tmp_path / "b")
        reused = simulated(tmp_path / "b")["reused"]
        assert str(calibration) in reused["undeclared"]
        calibration.write_text('{"scale": 1.10}\n')
        with pytest.raises(ProvenanceError, match=str(calibration)):
            session.evaluate({RADIUS: 450.0}, tmp_path / "c")


def test_an_input_that_changes_while_a_reusing_candidate_runs_leaves_it_unrecorded(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    class Changing(cell_count()):  # type: ignore[misc]
        def evaluate(self, request: Any) -> Any:
            calibration.write_text('{"scale": 1.10}\n')
            return super().evaluate(request)

    evaluated(problem_file, tmp_path / "a")
    with EvaluationSession.open(problem_file, [Changing]) as session:
        session.reuse([tmp_path / "a"])
        with pytest.raises(ProvenanceError, match=str(calibration)):
            session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert not (tmp_path / "b" / SUMMARY_NAME).exists()


def test_a_file_a_recorded_stage_looked_for_that_appears_stops_the_session(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # trial a takes the defaults without an optional override; b would read it
    override = tmp_path / "override.json"
    stack.install(monkeypatch, **{"*": {"read": [str(override)]}})
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        override.write_text("{}")
        with pytest.raises(ProvenanceError, match=f"{override}, which a stage of this session"):
            session.evaluate({RADIUS: 460.0}, tmp_path / "b")
    assert not (tmp_path / "b" / SUMMARY_NAME).exists()


def test_a_file_a_reused_stage_looked_for_that_appears_stops_the_session(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # study A simulates without the override; B reuses that, then the override appears
    override = tmp_path / "override.json"
    stack.install(monkeypatch, **{"*": {"read": [str(override)]}})
    evaluated(problem_file, tmp_path / "A" / "t0")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.reuse([tmp_path / "A"])
        session.evaluate({RADIUS: 450.0}, tmp_path / "B" / "t0")
        assert "reused" in simulated(tmp_path / "B" / "t0")
        override.write_text("{}")
        with pytest.raises(ProvenanceError, match=str(override)):
            session.evaluate({RADIUS: 450.0}, tmp_path / "B" / "t1")


def test_a_file_that_appears_while_its_candidate_runs_leaves_it_unrecorded(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "override.json"

    class Appearing(cell_count()):  # type: ignore[misc]
        def evaluate(self, request: Any) -> Any:
            override.write_text("{}")
            return super().evaluate(request)

    stack.install(monkeypatch, **{"*": {"read": [str(override)]}})
    with (
        EvaluationSession.open(problem_file, [Appearing]) as session,
        pytest.raises(ProvenanceError, match=str(override)),
    ):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert not (tmp_path / "a" / SUMMARY_NAME).exists()


def test_a_stage_whose_input_changed_while_it_ran_fails(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file, marks = tmp_path / "calibration", tmp_path / "marks"
    file.write_text("read")
    marks.mkdir()
    stack.install(monkeypatch, marks, **{"*": {"read": [str(file)], "sleep": 1}})

    def another_process_changes_it() -> None:  # any time while the stage runs: it cannot tell
        assert stack.eventually(lambda: (marks / "a.gamma.started").exists())
        file.write_text("changed while it ran")

    changing = threading.Thread(target=another_process_changes_it)
    changing.start()
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    changing.join()
    # what it read is not known: no record says it read anything, nor is it reused
    assert outcome.candidate.status == "failed"
    made = simulated(tmp_path / "a")
    assert made["status"] == "failed" and made["observed"]["undeclared"][str(file)] is None
    assert f"{file} changed while it ran, so the record cannot say what it read" in made["message"]


def test_an_unknown_level_is_refused(problem_file: Path) -> None:
    with pytest.raises(RunError, match="observe must be one of files, libraries, none"):
        EvaluationSession.open(problem_file, [cell_count()], observe="all")  # type: ignore[arg-type]


def test_where_strace_cannot_run_the_session_says_so_and_observes_what_it_can(
    tmp_path: Path, problem_file: Path, calibration: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _session, "unobservable", lambda level: "no ptrace here" if level == "files" else None
    )
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        (warning,) = [text for text in session.warnings if "observed" in text]
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
    assert "the libraries level, not files: no ptrace here" in warning
    assert simulated(tmp_path / "a")["observed"]["level"] == "libraries"


@pytest.mark.parametrize("observe", ["files", "libraries", "none"])
def test_a_session_shares_its_stages_however_they_are_observed(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch, observe: Observation
) -> None:
    marks = tmp_path / "marks"
    marks.mkdir()
    stack.install(monkeypatch, marks)
    with EvaluationSession.open(problem_file, [cell_count()], observe=observe) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        session.evaluate({RADIUS: 450.0}, tmp_path / "b")
    assert [path.name for path in marks.glob("*.started")] == ["a.gamma.started"]
    assert simulated(tmp_path / "b")["reused"]["by_footprint"] is False


def test_another_study_reuses_an_output_whose_footprint_holds(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a")
    stage = evaluated(problem_file, tmp_path / "b", reuse=(tmp_path / "a",))
    assert stage["reused"]["by_footprint"] is True
    assert "argv" not in stage


def test_an_input_that_changed_keeps_another_study_from_reusing_the_output(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a")
    calibration.write_text('{"scale": 1.10}\n')
    stage = evaluated(problem_file, tmp_path / "b", reuse=(tmp_path / "a",))
    assert "reused" not in stage and "argv" in stage  # it ran
    assert f"{calibration} changed" in stage["not_reused"]


def test_a_file_that_appeared_keeps_another_study_from_reusing_the_output(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rootmap = tmp_path / "private.rootmap"  # looked for, not found: a ~/.rootmap
    stack.install(monkeypatch, **{"*": {"read": [str(rootmap)]}})
    evaluated(problem_file, tmp_path / "a")
    rootmap.touch()
    stage = evaluated(problem_file, tmp_path / "b", reuse=(tmp_path / "a",))
    assert "reused" not in stage and f"{rootmap} appeared" in stage["not_reused"]


@pytest.mark.parametrize("observe", ["libraries", "none"])
def test_a_stage_not_observed_closely_is_not_reused_by_another_study(
    tmp_path: Path, problem_file: Path, calibration: Path, observe: Observation
) -> None:
    evaluated(problem_file, tmp_path / "a", observe=observe)
    stage = evaluated(problem_file, tmp_path / "b", reuse=(tmp_path / "a",))
    assert "reused" not in stage and "not observed at the files level" in stage["not_reused"]


def test_a_stage_that_used_the_network_is_not_reused_by_another_study(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch, **{"*": {"connect": 9}})
    made = evaluated(problem_file, tmp_path / "a")
    assert made["observed"]["network"] == ["127.0.0.1:9", "AF_INET socket"]
    stage = evaluated(problem_file, tmp_path / "b", reuse=(tmp_path / "a",))
    assert "reused" not in stage and "used the network" in stage["not_reused"]


@pytest.mark.parametrize("damage", ["removed", "altered"])
def test_a_footprint_that_is_gone_or_altered_vouches_for_nothing(
    tmp_path: Path, problem_file: Path, calibration: Path, damage: str
) -> None:
    made = evaluated(problem_file, tmp_path / "a")
    footprint = tmp_path / "a" / made["logs"]["footprint"]
    if damage == "removed":
        footprint.unlink()
    else:
        forged = json.loads(gzip.decompress(footprint.read_bytes()))
        forged["read"] = {}
        footprint.write_bytes(gzip.compress(json.dumps(forged).encode()))
    stage = evaluated(problem_file, tmp_path / "b", reuse=(tmp_path / "a",))
    assert "reused" not in stage and "its footprint" in stage["not_reused"]


def test_a_resumed_study_needs_no_footprint_of_its_own_stages(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a", observe="none")
    stage = evaluated(problem_file, tmp_path / "b", reuse=(tmp_path / "a",), resumed=True)
    assert stage["reused"]["by_footprint"] is False


def test_a_resumed_study_whose_input_changed_is_refused(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    # one study, one set of inputs: trials of another would mix in
    evaluated(problem_file, tmp_path / "a")
    calibration.write_text('{"scale": 1.10}\n')
    with (
        EvaluationSession.open(problem_file, [cell_count()]) as session,
        pytest.raises(ProvenanceError, match=f"{calibration}, read by the study's trials"),
    ):
        session.reuse([tmp_path / "a"], resumed=True)


def test_a_resumed_study_watches_what_its_trials_read(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.reuse([tmp_path / "a"], resumed=True)
        calibration.write_text('{"scale": 1.10}\n')  # before any trial of this session
        with pytest.raises(ProvenanceError, match=str(calibration)):
            session.evaluate({RADIUS: 470.0}, tmp_path / "b")


def test_two_trials_that_read_an_input_as_two_files_cannot_be_resumed(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a")
    record = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    stage = record["samples"]["gamma"]["stages"]["simulation"]
    stage["observed"]["undeclared"][str(calibration)] = "sha256:" + "0" * 64
    shutil.copytree(tmp_path / "a" / "samples", tmp_path / "b" / "samples")  # its footprint too
    (tmp_path / "b" / SUMMARY_NAME).write_text(json.dumps(record))
    with (
        EvaluationSession.open(problem_file, [cell_count()]) as session,
        pytest.raises(ProvenanceError, match="was not the same file for a/gamma/simulation"),
    ):
        session.reuse([tmp_path], resumed=True)


def test_an_output_copied_elsewhere_with_its_record_still_serves(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a")
    shutil.move(tmp_path / "a", tmp_path / "moved")
    stage = evaluated(problem_file, tmp_path / "b", reuse=(tmp_path / "moved",))
    assert stage["reused"]["by_footprint"] is True


def test_a_local_build_no_stage_read_is_named_and_one_that_ran_is_not(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    used, unused = tmp_path / "used", tmp_path / "unused"
    for directory in (used, unused):
        directory.mkdir()
        (directory / "helper.py").write_text("SCALE = 1\n")
    stack.install(monkeypatch, **{"*": {"read": [str(used / "helper.py")]}})
    monkeypatch.setenv("PYTHONPATH", f"{used}:{unused}:{os.environ.get('PYTHONPATH', '')}")
    evaluated(problem_file, tmp_path / "a")
    record = json.loads((tmp_path / "a" / SUMMARY_NAME).read_text())
    assert str(used) in simulated(tmp_path / "a")["observed"]["local"]
    assert str(unused) in unused_local(record) and str(used) not in unused_local(record)


def test_a_resumed_study_one_of_whose_trials_missed_a_file_that_is_there_now_is_refused(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # an optional override, looked for and not found: the trials took the defaults
    override = tmp_path / "override.json"
    stack.install(monkeypatch, **{"*": {"read": [str(override)]}})
    evaluated(problem_file, tmp_path / "a")
    override.write_text("{}")
    with (
        EvaluationSession.open(problem_file, [cell_count()]) as session,
        pytest.raises(ProvenanceError, match=f"{override}, which a/gamma/simulation looked for"),
    ):
        session.reuse([tmp_path], resumed=True)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_resumed_study_one_of_whose_trials_was_refused_a_file_it_can_read_now_is_refused(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "override.json"
    override.write_text("{}")
    override.chmod(0)
    stack.install(monkeypatch, **{"*": {"read": [str(override)]}})
    evaluated(problem_file, tmp_path / "a")
    override.chmod(0o644)
    with (
        EvaluationSession.open(problem_file, [cell_count()]) as session,
        pytest.raises(ProvenanceError, match=f"{override}, which a/gamma/simulation looked for"),
    ):
        session.reuse([tmp_path], resumed=True)


def test_a_resumed_study_watches_what_its_trials_looked_for(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "override.json"
    stack.install(monkeypatch, **{"*": {"read": [str(override)]}})
    evaluated(problem_file, tmp_path / "a")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.reuse([tmp_path], resumed=True)
        override.write_text("{}")
        with pytest.raises(ProvenanceError, match=str(override)):
            session.evaluate({RADIUS: 460.0}, tmp_path / "b")


def test_what_a_stage_of_the_study_made_keeps_its_resume(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a looks for a cache and does not find it; b, a later trial, writes it
    cache = tmp_path / "cache"
    told = {"a.gamma": {"read": [str(cache)]}, "b.gamma": {"write": {str(cache): "made"}}}
    stack.install(monkeypatch, **told)
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        session.evaluate({RADIUS: 460.0}, tmp_path / "b")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.reuse([tmp_path], resumed=True)  # not refused


@pytest.mark.parametrize("damage", ["removed", "replaced"])
def test_a_resumed_study_whose_footprint_is_gone_or_another_is_refused(
    tmp_path: Path, problem_file: Path, calibration: Path, damage: str
) -> None:
    made = evaluated(problem_file, tmp_path / "a")
    footprint = tmp_path / "a" / made["logs"]["footprint"]
    if damage == "removed":
        footprint.unlink()
    else:
        footprint.write_bytes(gzip.compress(b"{}"))
    with (
        EvaluationSession.open(problem_file, [cell_count()]) as session,
        pytest.raises(ProvenanceError, match="what it looked for cannot be checked"),
    ):
        session.reuse([tmp_path / "a"], resumed=True)


def test_a_stage_not_observed_has_no_footprint_to_check_on_resume(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "a", observe="none")
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.reuse([tmp_path / "a"], resumed=True)  # nothing to check, nothing refused


def test_a_study_that_reused_another_s_output_checks_what_that_stage_missed_on_resume(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # study A simulates without an optional file; study B reuses it; the file appears; B resumes
    override = tmp_path / "override.json"
    stack.install(monkeypatch, **{"*": {"read": [str(override)]}})
    evaluated(problem_file, tmp_path / "A" / "t0")
    stage = evaluated(problem_file, tmp_path / "B" / "t0", reuse=(tmp_path / "A",))
    assert stage["reused"]["footprint"]["path"].startswith(str(tmp_path / "A"))
    override.write_text("{}")
    with (
        EvaluationSession.open(problem_file, [cell_count()]) as session,
        pytest.raises(ProvenanceError, match=f"{override}, which t0/gamma/simulation looked for"),
    ):
        session.reuse([tmp_path / "B"], resumed=True)


def test_a_study_whose_reused_source_is_gone_cannot_be_resumed(
    tmp_path: Path, problem_file: Path, calibration: Path
) -> None:
    evaluated(problem_file, tmp_path / "A" / "t0")
    evaluated(problem_file, tmp_path / "B" / "t0", reuse=(tmp_path / "A",))
    shutil.rmtree(tmp_path / "A")  # its events went with it, and what its stage found
    with (
        EvaluationSession.open(problem_file, [cell_count()]) as session,
        pytest.raises(ProvenanceError, match="resumes while they are there"),
    ):
        session.reuse([tmp_path / "B"], resumed=True)
