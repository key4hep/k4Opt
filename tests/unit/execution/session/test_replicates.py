"""A replicate of a candidate is the same candidate on other events: every seed derived from the
problem's, recorded and checked when the record is read. A source of candidates may give ``WAIT``
when it has none now: the session then waits for a candidate to end before it asks again."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.execution import WAIT, EvaluationSession, RunError, RunRequest, Wait
from k4opt.problem import CandidateError, load_problem, parse_problem
from k4opt.problem.candidate import REPLICATES, check_replicate, configure, replicate_seed
from k4opt.record import SUMMARY_NAME, Assessment, RecordError
from k4opt.record.signature import digest
from tests.support import simulation
from tests.support.evaluators import cell_count
from tests.unit.execution.session.conftest import RADIUS

SEED = 5  # the problem file's simulation seed


def summary(run: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run / SUMMARY_NAME).read_text())
    return data


def simulated(record: dict[str, Any]) -> dict[str, Any]:
    stage: dict[str, Any] = record["samples"]["gamma"]["stages"]["simulation"]
    return stage


def seeds(fake: simulation.FakeDdsim) -> list[int]:
    """The seed of each ddsim call."""
    return [
        int(argument.partition("=")[2])
        for argv, _ in fake.calls
        for argument in argv
        if argument.startswith("--random.seed=")
    ]


# --- the seeds of a replicate -------------------------------------------------------------------


def test_no_two_replicates_of_a_seed_share_their_seed() -> None:
    # once the same with a hash cut to 31 bits; the stride makes each replicate's seed its own
    assert replicate_seed(SEED, 3367) != replicate_seed(SEED, 20785)
    many = range(200_000)
    assert len({replicate_seed(SEED, replicate) for replicate in many}) == len(many)
    last = REPLICATES - 1  # the seed of replicate 0 comes back only after every other one
    assert replicate_seed(SEED, last) != SEED and 0 <= replicate_seed(SEED, last) < 2**31


def test_replicate_0_keeps_the_seed_and_others_derive_one_alike_everywhere() -> None:
    assert replicate_seed(SEED, 0) == SEED
    derived = [replicate_seed(SEED, replicate) for replicate in (1, 2, 3)]
    assert len(set(derived)) == 3 and SEED not in derived
    assert all(0 <= seed < 2**31 for seed in derived)
    assert derived == [replicate_seed(SEED, replicate) for replicate in (1, 2, 3)]
    assert replicate_seed(SEED + 1, 1) != derived[0]  # another sample's seed, another series


def test_a_replicate_changes_the_seeds_and_nothing_else(problem_file: Path) -> None:
    spec = load_problem(problem_file)
    values = {RADIUS: 450.0}
    problem, other = configure(spec, values), configure(spec, values, replicate=2)
    stages, again = problem.samples["gamma"].stages, other.samples["gamma"].stages
    assert stages["simulation"].seed == SEED
    assert again["simulation"].seed == replicate_seed(SEED, 2)
    assert {name: s.to_dict() for name, s in stages.items() if name != "simulation"} == {
        name: s.to_dict() for name, s in again.items() if name != "simulation"
    }
    assert (other.geometry, other.evaluators) == (problem.geometry, problem.evaluators)


@pytest.mark.parametrize("replicate", [-1, True, 1.0, "1", 2**31])
def test_a_replicate_is_an_integer_of_31_bits(problem_file: Path, replicate: object) -> None:
    with pytest.raises(CandidateError, match=r"replicate must be an integer from 0 to 2147483647"):
        check_replicate(load_problem(problem_file), replicate)


def test_a_problem_that_makes_no_events_from_a_seed_has_no_replicate() -> None:
    # it measures the geometry alone: another replicate would be the candidate itself
    spec = parse_problem(
        {
            "schema_version": "k4opt.problem/v1alpha1",
            "environment": {"key4hep": {"channel": "nightly", "version": "2026-10-20"}},
            "geometry": "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml",
            "parameters": {
                "cell": {
                    "type": "continuous",
                    "bounds": [2.5, 20],
                    "unit": "mm",
                    "targets": {"geometry": "ECal_cell_size"},
                }
            },
            "evaluators": [{"name": "ecal.channels", "version": "1"}],
            "objectives": [
                {
                    "evaluator": "ecal.channels",
                    "stage": "geometry",
                    "metric": "ecal_barrel.channels",
                    "direction": "minimize",
                }
            ],
        }
    )
    assert check_replicate(spec, 0) == 0
    with pytest.raises(CandidateError, match="the problem makes no events from a seed"):
        check_replicate(spec, 1)


def test_a_sample_that_reads_its_events_from_a_file_has_no_replicate(
    tmp_path: Path, problem_file: Path
) -> None:
    # generated elsewhere (HepMC): another seed of ddsim's showers the same collisions again
    doc = yaml.safe_load(problem_file.read_text())
    events = tmp_path / "zh.hepmc"
    events.write_text("HepMC::Version 3\n")
    doc["samples"]["gamma"]["generation"] = {"file": str(events)}
    problem_file.write_text(yaml.safe_dump(doc))
    spec = load_problem(problem_file)
    assert check_replicate(spec, 0) == 0
    with pytest.raises(CandidateError, match=r"samples\.gamma reads its generation from a file"):
        check_replicate(spec, 1)


# --- a replicate evaluated and recorded ---------------------------------------------------------


def test_a_replicate_runs_records_and_reads_back_with_its_own_seeds(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        first = session.evaluate({RADIUS: 450.0}, tmp_path / "a")
        other = session.evaluate({RADIUS: 450.0}, tmp_path / "b", replicate=1)
        again = session.evaluate({RADIUS: 450.0}, tmp_path / "c", replicate=1)
        for outcome in (first, other, again):
            Assessment.from_summary(session.spec, summary(outcome.run_dir))  # each holds up
    assert seeds(fake) == [SEED, replicate_seed(SEED, 1)]  # c reused b's events, not a's
    assert [summary(tmp_path / name)["replicate"] for name in "abc"] == [0, 1, 1]
    made, replicated = simulated(summary(tmp_path / "a")), simulated(summary(tmp_path / "b"))
    assert replicated["definition"]["seed"] == replicate_seed(SEED, 1)
    assert replicated["signature"]["sha256"] != made["signature"]["sha256"]
    assert simulated(summary(tmp_path / "c"))["reused"]["record"] == other.summary["id"]


@pytest.mark.parametrize("replicate", [0, 2, None, "1"])
def test_a_record_whose_replicate_is_not_its_seeds_is_refused(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim, replicate: object
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a", replicate=1)
        data = summary(outcome.run_dir)
        data["replicate"] = replicate
        with pytest.raises(RecordError, match=r"definition|replicate must be an integer"):
            Assessment.from_summary(session.spec, data)


def test_a_record_whose_seed_is_resigned_to_another_replicate_is_refused(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    with EvaluationSession.open(problem_file, [cell_count()]) as session:
        outcome = session.evaluate({RADIUS: 450.0}, tmp_path / "a", replicate=1)
        data = summary(outcome.run_dir)
        stage = simulated(data)
        stage["definition"]["seed"] = replicate_seed(SEED, 2)  # the stage and its signature agree
        stage["signature"]["document"]["definition"]["seed"] = replicate_seed(SEED, 2)
        stage["signature"]["sha256"] = digest(stage["signature"]["document"])
        with pytest.raises(RecordError, match="definition"):  # but not with the record's replicate
            Assessment.from_summary(session.spec, data)


def test_a_replicate_with_a_supplied_stage_is_refused_before_anything_is_created(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    events = tmp_path / "earlier.edm4hep.root"
    events.write_bytes(b"fake")
    inputs = {"gamma/simulation": events}
    with (
        EvaluationSession.open(problem_file, [cell_count()], inputs=inputs) as session,
        pytest.raises(CandidateError, match="gamma/simulation are supplied"),
    ):
        session.evaluate({RADIUS: 450.0}, tmp_path / "a", replicate=1)
    assert not (tmp_path / "a").exists() and fake.calls == []


def test_k4opt_evaluate_replicate(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    from tests.support.evaluators import CELL_COUNT

    out, err = io.StringIO(), io.StringIO()
    code = main(
        [
            "evaluate",
            str(problem_file),
            "--evaluator",
            CELL_COUNT,
            "--param",
            f"{RADIUS}=450",
            "--replicate",
            "3",
            "--output",
            str(tmp_path / "run"),
        ],
        stdout=out,
        stderr=err,
    )
    assert code == 0, err.getvalue()
    assert summary(tmp_path / "run")["replicate"] == 3
    assert seeds(fake) == [replicate_seed(SEED, 3)]


# --- a source of candidates that waits (with workers: tests/unit/execution/test_parallel.py) -----


def test_a_source_that_waits_with_nothing_under_way_is_an_error(
    tmp_path: Path, problem_file: Path, fake: simulation.FakeDdsim
) -> None:
    def source() -> Iterator[RunRequest | Wait]:
        yield WAIT

    with (
        EvaluationSession.open(problem_file, [cell_count()]) as session,
        pytest.raises(RunError, match=re.escape("waits for a candidate to end, but none")),
        session.evaluate_each(source()) as evaluations,
    ):
        list(evaluations)
    assert fake.calls == []
