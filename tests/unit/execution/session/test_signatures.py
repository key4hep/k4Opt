"""Each stage that ran records its signature: what it was made from, and the SHA-256 of that
document. Equal for stages made from the same things, whatever the candidate or the run
directory; another geometry, seed or option makes another one; a record whose signature is not
what the record itself says is refused."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.execution import EvaluationSession
from k4opt.record import Assessment, RecordError
from k4opt.record.signature import digest
from tests.support import simulation
from tests.support.evaluators import cell_count
from tests.unit.execution.session.conftest import RADIUS

SIMULATED = ("samples", "gamma", "stages", "simulation")


def signature(record: dict[str, Any], stage: str = "simulation") -> dict[str, Any]:
    found: dict[str, Any] = record["samples"]["gamma"]["stages"][stage]["signature"]
    return found


def evaluate(session: EvaluationSession, radius: float, output: Path) -> dict[str, Any]:
    outcome = session.evaluate({RADIUS: radius}, output)
    data: dict[str, Any] = json.loads(outcome.summary_path.read_text())
    return data


def test_a_stage_made_from_the_same_things_has_the_same_signature(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        first = evaluate(session, 450.0, tmp_path / "a")
        again = evaluate(session, 450.0, tmp_path / "b")  # another run directory, same values
        other = evaluate(session, 460.0, tmp_path / "c")  # another geometry
    simulated = signature(first)
    assert simulated["sha256"] == digest(simulated["document"])
    assert signature(again) == simulated
    assert signature(other)["sha256"] != simulated["sha256"]
    document, changed = simulated["document"], signature(other)["document"]
    differ = [key for key in document if document[key] != changed[key]]
    assert differ == ["reads"]  # the geometry it read, and nothing else
    assert document["reads"]["geometry"]["fingerprint"] == first["geometry"]["fingerprint"]
    # the gun the simulation fires is a stage too: its signature is what the simulation reads
    assert document["reads"]["generation"] == signature(first, "generation")["sha256"]
    assert document["definition"] == {
        key: value
        for key, value in first["samples"]["gamma"]["stages"]["simulation"]["definition"].items()
        if key != "timeout"
    }
    assert document["program"] == first["environment"]["observed"]["programs"]["ddsim"]
    assert document["k4opt"] == first["environment"]["observed"]["k4opt_source"]["sha256"]


def test_another_seed_makes_another_signature(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        first = evaluate(session, 450.0, tmp_path / "a")
    document = yaml.safe_load(problem_file.read_text())
    document["samples"]["gamma"]["simulation"]["seed"] += 1
    problem_file.write_text(yaml.safe_dump(document))
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        reseeded = evaluate(session, 450.0, tmp_path / "b")
    assert signature(first)["sha256"] != signature(reseeded)["sha256"]
    assert signature(reseeded)["document"]["definition"]["seed"] == (
        signature(first)["document"]["definition"]["seed"] + 1
    )


def _resigned(data: dict[str, Any], change: Any) -> None:
    """Change the simulation's signature document with ``change`` and give it the matching
    SHA-256: a signature that holds together, but not with the record."""
    held = signature(data)
    change(held["document"])
    held["sha256"] = digest(held["document"])


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda data: signature(data).update(sha256="a" * 64),
            "its signature's SHA-256 is not that of its document",
        ),
        (
            lambda data: _resigned(data, lambda d: d["definition"].update(seed=7)),
            "its signature says it was made from other definition than the record does",
        ),
        (
            lambda data: _resigned(data, lambda d: d["reads"]["geometry"].update(fingerprint="x")),
            "its signature says it was made from other reads than the record does",
        ),
        (
            lambda data: _resigned(data, lambda d: d.update(k4opt="0" * 64)),
            "its signature says it was made from other k4opt than the record does",
        ),
        (
            lambda data: data["samples"]["gamma"]["stages"]["simulation"].pop("signature"),
            "stage 'gamma/simulation': signature must be a mapping",
        ),
        (
            lambda data: data["samples"]["gamma"]["stages"]["generation"].pop("signature"),
            "stage 'gamma/generation': signature must be a mapping",
        ),
    ],
)
def test_a_record_whose_signature_is_not_what_it_says_is_refused(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, tamper: Any, message: str
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        data = evaluate(session, 450.0, tmp_path / "a")
        Assessment.from_summary(session.spec, data)  # as written, it holds up
        tamper(data)
        with pytest.raises(RecordError, match=re.escape(message)):
            Assessment.from_summary(session.spec, data)
