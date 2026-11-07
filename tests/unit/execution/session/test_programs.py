"""The programs of the stack a session runs (ddsim, k4run): found once, when it opens, run as found
whatever ``PATH`` says later, identified in every record and checked before each candidate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from k4opt.execution import EvaluationSession, ProvenanceError
from k4opt.record import SUMMARY_NAME
from tests.support import stack
from tests.support.evaluators import cell_count
from tests.unit.execution.session.conftest import RADIUS


def build(directory: Path, label: str) -> Path:
    """A copy of the fake stack's ``ddsim`` in ``directory``, told apart by ``label``: another
    build of the same program."""
    directory.mkdir(parents=True)
    text = (stack.FAKE_STACK / "bin" / "ddsim").read_text()
    path = directory / "ddsim"
    path.write_text(text.replace("\n", f"\n# build {label}\n", 1))
    path.chmod(0o755)
    return path


@pytest.fixture
def builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Two builds of ddsim, the first one first on ``PATH``."""
    stack.install(monkeypatch)
    first, second = build(tmp_path / "a", "a"), build(tmp_path / "b", "b")
    monkeypatch.setenv("PATH", f"{first.parent}{os.pathsep}{os.environ['PATH']}")
    return first, second


def argv0(run: Path) -> str:
    record = json.loads((run / SUMMARY_NAME).read_text())
    return str(record["samples"]["gamma"]["stages"]["simulation"]["argv"][0])


def test_a_session_runs_and_records_the_ddsim_it_found_when_it_opened(
    tmp_path: Path, problem_file: Path, builds: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = builds
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        programs = session.identity()["programs"]
        ddsim = dict(programs["ddsim"])
        interpreter = ddsim.pop("interpreter")
        assert ddsim == {
            "path": str(first),
            "bytes": first.stat().st_size,
            "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        }
        # and the interpreter its #! line names, as PATH found it
        shebang = first.read_text().splitlines()[0].split()
        name = shebang[-1] if Path(shebang[0][2:]).name == "env" else shebang[0][2:]
        assert interpreter["path"] == (shutil.which(name) if "/" not in name else name)
        session.evaluate({RADIUS: 450.0}, tmp_path / "run")
        assert argv0(tmp_path / "run") == str(first)
        record = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
        assert record["environment"]["observed"]["programs"] == programs
        # another ddsim first on PATH later: no candidate runs in another environment
        monkeypatch.setenv("PATH", f"{second.parent}{os.pathsep}{os.environ['PATH']}")
        with pytest.raises(ProvenanceError, match="differs in PATH"):
            session.evaluate({RADIUS: 460.0}, tmp_path / "later")
    assert not (tmp_path / "later").exists()


def test_a_ddsim_that_changes_after_the_session_opened_stops_the_next_candidate(
    tmp_path: Path, problem_file: Path, builds: tuple[Path, Path]
) -> None:
    first, _ = builds
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        session.evaluate({RADIUS: 450.0}, tmp_path / "one")
        first.write_text(first.read_text() + "# rebuilt\n")
        with pytest.raises(ProvenanceError, match=r"ddsim: .* changed after the session opened"):
            session.evaluate({RADIUS: 460.0}, tmp_path / "two")
    assert not (tmp_path / "two").exists()


def test_a_program_not_found_when_the_session_opened_is_not_looked_for_later(
    tmp_path: Path, problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack.install(monkeypatch)
    later = build(tmp_path / "later", "later")
    found = os.environ["PATH"].split(os.pathsep)
    without = [entry for entry in found if not (Path(entry) / "ddsim").exists()]
    monkeypatch.setenv("PATH", os.pathsep.join(without))
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        assert session.identity()["programs"] == {"ddsim": None}
        session.evaluate({RADIUS: 450.0}, tmp_path / "run")
        # a ddsim on PATH now: the session did not identify it, so no record could name it
        monkeypatch.setenv("PATH", os.pathsep.join([str(later.parent), *without]))
        with pytest.raises(ProvenanceError, match="differs in PATH"):
            session.evaluate({RADIUS: 460.0}, tmp_path / "later")
    record = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
    assert record["status"] == "failed"
    simulation = record["samples"]["gamma"]["stages"]["simulation"]
    assert simulation["status"] == "failed" and "argv" not in simulation
    assert "ddsim was not found on PATH when the session opened" in simulation["message"]
    assert "signature" not in simulation or simulation["signature"]["document"]["program"] is None


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(lambda ddsim: ddsim.chmod(0o311), id="unreadable"),
        pytest.param(
            lambda ddsim: ddsim.write_text("#!/usr/bin/env k4opt-no-such-python\n"),
            id="interpreter-not-found",
        ),
    ],
)
def test_a_program_found_but_not_identified_is_refused_when_the_session_opens(
    problem_file: Path, builds: tuple[Path, Path], broken: Callable[[Path], None]
) -> None:
    first, _ = builds
    broken(first)
    if os.access(first, os.R_OK) and first.stat().st_mode & 0o444 == 0:
        pytest.skip("this user reads files without read permission (root)")
    with pytest.raises(ProvenanceError, match="no record could say which program ran"):
        EvaluationSession.open(problem_file, [cell_count()])
