"""``k4opt study export``'s formats: one CSV row per trial and the JSON of the whole study, each
value as recorded and nothing made up, the same bytes every time, written whole to a new file
outside the study."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from k4opt import jsonio
from k4opt.frozen import FrozenMap
from k4opt.optimization import (
    OptimizationError,
    StudyResults,
    TrialResult,
    load_study,
    optimize,
)
from k4opt.optimization._export import columns, export, to_csv, to_json
from k4opt.optimization._show import render_results
from k4opt.problem import Assessment, ObjectiveValue, parse_problem
from tests.support import simulation
from tests.support.common import tree_digest
from tests.support.evaluators import cell_count, trade_off
from tests.unit.optimization.conftest import RADIUS, evaluator_from, problem_document, summary


def run(problem: Path, output: Path, trials: int, evaluators: list[Any] | None = None) -> None:
    optimize(problem, evaluators or [trade_off()], output, trials=trials, seed=1)


def test_the_csv_holds_one_row_per_trial_with_every_value_as_recorded(
    tmp_path: Path, mixed_problem: Path
) -> None:
    output = tmp_path / "s"
    run(mixed_problem, output, 8)
    study = load_study(output)
    rows = list(csv.DictReader(io.StringIO(to_csv(study))))
    assert list(rows[0]) == columns(study)
    assert columns(study) == [
        "study",
        "trial",
        "status",
        "feasible",
        "pareto",
        "recovered",
        f"parameter:{RADIUS}",
        "objective:test.trade_off/analysis/a",
        "objective:test.trade_off/analysis/a:unit",
        "objective:test.trade_off/analysis/a:uncertainty",
        "objective:test.trade_off/analysis/b",
        "objective:test.trade_off/analysis/b:unit",
        "objective:test.trade_off/analysis/b:uncertainty",
        "limit:test.trade_off/analysis/a:max:value",
        "limit:test.trade_off/analysis/a:max:margin",
        "limit:test.trade_off/analysis/a:max:satisfied",
        "elapsed_seconds",
        "record",
        "record_sha256",
        "record_path",
        "message",
    ]
    assert [row["trial"] for row in rows] == [str(n) for n in range(8)]
    assert {row["status"] for row in rows} == {"valid", "invalid", "failed"}  # to mean much
    for row, trial in zip(rows, study.trials, strict=True):
        record = summary(output, trial.number)
        assert (row["study"], row["status"], row["recovered"]) == (study.id, trial.status, "false")
        assert row["pareto"] == ("true" if trial.number in study.pareto else "false")
        assert row[f"parameter:{RADIUS}"] == str(record["candidate"][RADIUS])
        assert float(row["elapsed_seconds"]) == record["elapsed_seconds"]  # every digit
        assert (row["record"], row["record_path"]) == (
            record["id"],
            f"trial-{trial.number:06d}/summary.json",
        )
        assert row["record_sha256"] == trial.sha256
        assert (
            row["objective:test.trade_off/analysis/a:uncertainty"] == ""
        )  # none recorded, none made up
        assert row["message"] == (record["message"] or "")
        if trial.status == "valid":
            b = next(m for m in record["metrics"] if m["metric"] == "b")
            assert (
                float(row["objective:test.trade_off/analysis/b"]),
                row["objective:test.trade_off/analysis/b:unit"],
            ) == (
                b["value"],
                "1/mm",
            )
            (limit,) = record["assessment"]["limits"]
            assert float(row["limit:test.trade_off/analysis/a:max:margin"]) == limit["margin"]
            assert row["feasible"] == row["limit:test.trade_off/analysis/a:max:satisfied"]
            assert row["feasible"] == ("true" if record["assessment"]["feasible"] else "false")
        else:  # missing, never 0
            missing = [key for key in row if key.startswith(("objective:", "limit:"))]
            assert [row[key] for key in [*missing, "feasible"]] == [""] * (len(missing) + 1)


def test_two_bindings_of_one_metric_are_two_columns_and_two_lines_of_show(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    strict = {"name": "test.trade_off", "as": "strict", "version": "1"}
    problem = write_problem(
        evaluators=[{"name": "test.trade_off", "version": "1"}, strict],
        objectives=[
            ("test.trade_off", "analysis", "a", "minimize"),
            ("strict", "analysis", "a", "maximize"),
        ],
        constraints=[{"evaluator": "strict", "stage": "analysis", "metric": "b", "max": 1}],
    )
    output = tmp_path / "s"
    run(problem, output, 2)
    study = load_study(output)
    assert [c for c in columns(study) if c.startswith(("objective:", "limit:"))] == [
        "objective:test.trade_off/analysis/a",
        "objective:test.trade_off/analysis/a:unit",
        "objective:test.trade_off/analysis/a:uncertainty",
        "objective:strict/analysis/a",
        "objective:strict/analysis/a:unit",
        "objective:strict/analysis/a:uncertainty",
        "limit:strict/analysis/b:max:value",
        "limit:strict/analysis/b:max:margin",
        "limit:strict/analysis/b:max:satisfied",
    ]
    rows = list(csv.DictReader(io.StringIO(to_csv(study))))
    for trial, row in zip(study.trials, rows, strict=True):
        a = trial.candidate[RADIUS]
        assert row["objective:test.trade_off/analysis/a"] == row["objective:strict/analysis/a"]
        assert float(row["objective:strict/analysis/a"]) == a
        assert float(row["limit:strict/analysis/b:max:value"]) == 1 / a
    shown = io.StringIO()
    render_results(study, shown)
    lines = shown.getvalue().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("objectives:"))
    assert lines[start : start + 3] == [
        "objectives: test.trade_off/analysis/a  minimize  (mm)",
        "            strict/analysis/a  maximize  (mm)",
        "limits:     strict/analysis/b:max  <= 1",
    ]


def test_the_samples_name_the_objective_and_time_columns_and_the_lines_of_show(
    tmp_path: Path, sampled_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    output = tmp_path / "s"
    run(sampled_problem, output, 2, evaluators=[cell_count()])
    study = load_study(output)
    header = columns(study)
    cells = "test.cell_count/simulation/ecal_barrel.mean_hit_cells"
    assert [c for c in header if c.startswith("objective:") and ":" not in c[10:]] == [
        f"objective:gamma/{cells}",
        f"objective:pion/{cells}",
    ]
    assert [c for c in header if c.startswith("stage:")] == [
        f"stage:{label}:elapsed_seconds"
        for label in ("gamma/generation", "gamma/simulation", "pion/generation", "pion/simulation")
    ]
    rows = list(csv.DictReader(io.StringIO(to_csv(study))))
    for trial, row in zip(study.trials, rows, strict=True):
        record = summary(output, trial.number)
        seconds = record["samples"]["pion"]["stages"]["simulation"]["elapsed_seconds"]
        assert float(row["stage:pion/simulation:elapsed_seconds"]) == seconds
        assert float(row[f"objective:pion/{cells}"]) == 1.5  # 2 and 1 cells
    shown = io.StringIO()
    render_results(study, shown)
    lines = shown.getvalue().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("objectives:"))
    assert lines[start : start + 2] == [
        f"objectives: gamma/{cells}  minimize  (1)",
        f"            pion/{cells}  maximize  (1)",
    ]


def test_column_names_stay_distinct_whatever_the_stages_and_metrics_are_called() -> None:
    named = [("analysis", "count"), ("analysis", "count:unit"), ("analysis/x", "y")]
    named += [("analysis", "x/y"), ("analysis", "50%"), ("analysis", "cells\x1b[2J")]  # all valid
    spec = parse_problem(problem_document(objectives=[(s, m, "minimize") for s, m in named]))
    measured = tuple(
        ObjectiveValue("test.trade_off", s, m, "minimize", float(i), unit=f"u{i}")
        for i, (s, m) in enumerate(named)
    )
    trial = TrialResult(0, "valid", {RADIUS: 300.0}, Path("t"), Assessment(measured, (), True))
    study = StudyResults("s", "id", "now", Path("s"), spec, FrozenMap({}), (trial,), (0,))
    header = columns(study)
    assert len(set(header)) == len(header)
    escaped = ["analysis/count", "analysis/count%3Aunit", "analysis%2Fx/y", "analysis/x%2Fy"]
    escaped += ["analysis/50%25"]
    (row,) = csv.DictReader(io.StringIO(to_csv(study)))
    escaped += ["analysis/cells%1B[2J"]  # a terminal showing the header runs nothing
    escaped = [f"test.trade_off/{name}" for name in escaped]
    assert "\x1b" not in to_csv(study)
    assert [row[f"objective:{name}"] for name in escaped] == [f"{i}.0" for i in range(6)]
    assert [row[f"objective:{name}:unit"] for name in escaped] == [f"u{i}" for i in range(6)]
    # the evaluator, stage and metric read back: split at the two "/" left, then unescape each
    assert [tuple(map(urllib.parse.unquote, n.split("/"))) for n in escaped] == [
        ("test.trade_off", *name) for name in named
    ]


def test_the_json_export_holds_the_study_and_each_record_as_it_is(
    tmp_path: Path, mixed_problem: Path
) -> None:
    output = tmp_path / "s"
    run(mixed_problem, output, 6)
    study = load_study(output)
    data = json.loads(to_json(study))
    assert data["format"] == "k4opt.study-results/1"
    assert data["directory"] == str(output)
    assert data["study"] == json.loads((output / "study.json").read_text())
    assert (data["pareto"], data["outcomes"]) == (list(study.pareto), dict(study.outcomes))
    assert [entry["number"] for entry in data["trials"]] == list(range(6))
    for entry in data["trials"]:
        record = summary(output, entry["number"])
        data_bytes = (output / f"trial-{entry['number']:06d}" / "summary.json").read_bytes()
        assert entry["assessment"] == record["assessment"]
        assert entry["candidate"] == record["candidate"]
        assert entry["elapsed_seconds"] == record["elapsed_seconds"]
        assert (entry["status"], entry["message"]) == (record["status"], record["message"])
        assert entry["record"] == {
            "id": record["id"],
            "path": f"trial-{entry['number']:06d}/summary.json",
            "sha256": hashlib.sha256(data_bytes).hexdigest(),
        }
        assert entry["pareto"] == (entry["number"] in study.pareto)


def test_an_export_is_the_same_bytes_every_time(tmp_path: Path, mixed_problem: Path) -> None:
    run(mixed_problem, tmp_path / "s", 4)
    first, second = load_study(tmp_path / "s"), load_study(tmp_path / "s")
    assert to_csv(first) == to_csv(second)
    assert to_json(first) == to_json(second)


ODD = """
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            self.invalid('cells, "quoted"\\nnext line, \\u00e9, \\x1b[31mred\\x1b[0m')
"""


def test_text_from_a_record_reads_back_and_carries_no_terminal_escape(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    odd = evaluator_from(tmp_path / "evaluators", "odd", ODD)
    run(write_problem(), tmp_path / "s", 1, evaluators=[odd])
    study = load_study(tmp_path / "s")
    (trial,) = study.trials
    expected = 'test.trade_off invalid: cells, "quoted"\nnext line, é, \x1b[31mred\x1b[0m'
    assert trial.message == expected
    text = to_csv(study)
    assert "\x1b" not in text
    (row,) = csv.DictReader(io.StringIO(text))
    assert row["message"] == expected.replace("\x1b", "\\x1b")  # escaped, the rest as it is
    exported = to_json(study)
    assert "\x1b" not in exported
    assert json.loads(exported)["trials"][0]["message"] == expected  # exactly


def test_an_export_is_a_new_file_outside_the_study_written_whole(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "s"
    run(write_problem(), output, 2)
    study = load_study(output)
    before = tree_digest(output)
    printed = io.StringIO()
    export(study, "json", printed, None)
    assert printed.getvalue() == to_json(study)
    target = tmp_path / "results.csv"
    export(study, "csv", io.StringIO(), target)
    assert target.read_text() == to_csv(study)
    with pytest.raises(OptimizationError, match=r"results\.csv already exists; choose a new"):
        export(study, "json", io.StringIO(), target)
    assert target.read_text() == to_csv(study)  # left as it was
    (tmp_path / "link").symlink_to(output)
    for inside in (output / "results.csv", tmp_path / "link" / "results.csv"):
        with pytest.raises(OptimizationError, match="is in the study directory"):
            export(study, "csv", io.StringIO(), inside)
    assert tree_digest(output) == before

    def failing(descriptor: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(jsonio.os, "fsync", failing)
    with pytest.raises(OptimizationError, match=r"cannot write .*other\.csv: disk full"):
        export(study, "csv", io.StringIO(), tmp_path / "other.csv")
    assert not [p.name for p in tmp_path.iterdir() if "other.csv" in p.name]  # nothing partial
