"""``k4opt study export``'s formats: one CSV row per trial and the JSON of the whole study, each
value as recorded and nothing made up, the same bytes every time, written whole to a new file
outside the study."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from k4opt import jsonio
from k4opt.optimization import OptimizationError, load_study, optimize
from k4opt.optimization._export import columns, export, to_csv, to_json
from tests.support.common import tree_digest
from tests.support.evaluators import trade_off
from tests.unit.optimization.conftest import RADIUS, evaluator_from, summary


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
        "objective:analysis/a",
        "objective:analysis/a:unit",
        "objective:analysis/a:uncertainty",
        "objective:analysis/b",
        "objective:analysis/b:unit",
        "objective:analysis/b:uncertainty",
        "limit:analysis/a:max:value",
        "limit:analysis/a:max:margin",
        "limit:analysis/a:max:satisfied",
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
        assert row["objective:analysis/a:uncertainty"] == ""  # none recorded, none made up
        assert row["message"] == (record["message"] or "")
        if trial.status == "valid":
            b = next(m for m in record["metrics"] if m["metric"] == "b")
            assert (float(row["objective:analysis/b"]), row["objective:analysis/b:unit"]) == (
                b["value"],
                "1/mm",
            )
            (limit,) = record["assessment"]["limits"]
            assert float(row["limit:analysis/a:max:margin"]) == limit["margin"]
            assert row["feasible"] == row["limit:analysis/a:max:satisfied"]
            assert row["feasible"] == ("true" if record["assessment"]["feasible"] else "false")
        else:  # missing, never 0
            missing = [key for key in row if key.startswith(("objective:", "limit:"))]
            assert [row[key] for key in [*missing, "feasible"]] == [""] * (len(missing) + 1)


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
