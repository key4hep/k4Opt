"""The programs of the stack a session runs (ddsim, k4run): found once, when it opens, run as found
whatever ``PATH`` says later, identified in every record and checked before each candidate."""

from __future__ import annotations

import hashlib
import json
import os
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
        assert programs == {
            "ddsim": {
                "path": str(first),
                "bytes": first.stat().st_size,
                "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            }
        }
        # another ddsim first on PATH later: the session still runs the one it identified
        monkeypatch.setenv("PATH", f"{second.parent}{os.pathsep}{os.environ['PATH']}")
        session.evaluate({RADIUS: 450.0}, tmp_path / "run")
    assert argv0(tmp_path / "run") == str(first)
    record = json.loads((tmp_path / "run" / SUMMARY_NAME).read_text())
    assert record["environment"]["observed"]["programs"] == programs


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
