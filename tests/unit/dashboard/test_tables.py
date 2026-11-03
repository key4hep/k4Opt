"""The dashboard's tables: every value as the study records it, an empty cell where it has none."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

from k4opt.optimization import load_study
from k4opt.publication import Catalog, CatalogEntry, Snapshot
from tests.support.studies import mixed_study


def test_a_trial_row_holds_its_values_and_nothing_made_up(tmp_path: Path) -> None:
    import tables

    results = load_study(mixed_study(tmp_path / "s"))
    rows = tables.trials(results)
    assert [row["Trial"] for row in rows] == [t.number for t in results.trials]
    a = "test.trade_off/analysis/a [mm]"
    for row, trial in zip(rows, results.trials, strict=True):
        assert (row["Outcome"], row["Pareto"]) == (trial.outcome, trial.number in results.pareto)
        assert row["radius"] == trial.candidate["radius"]
        if trial.objectives is None:  # invalid, failed: no value, no margin, never 0
            assert row[a] is None and row["margin test.trade_off/analysis/a:max [mm]"] is None
        else:
            assert row[a] == trial.objectives["test.trade_off/analysis/a"]
            margin = row["margin test.trade_off/analysis/a:max [mm]"]
            assert margin == trial.assessment.limits[0].margin
            assert (margin <= 0) == (trial.outcome == "feasible")
        assert (row["Record"], row["Message"]) == (trial.record, trial.message)
    assert [row["Trial"] for row in tables.trials(results, results.pareto)] == list(results.pareto)
    assert tables.objectives(results)[1] == {
        "Objective": "test.trade_off/analysis/b",
        "Direction": "minimize",
        "Unit": "1/mm",
    }
    assert tables.constraints(results) == [
        {
            "Constraint": "test.trade_off/analysis/a",
            "Min": None,
            "Max": 400,
            "Unit": "mm",
            "Scale": None,
        }
    ]
    made = {row[""]: row["Value"] for row in tables.provenance(results)}
    assert made["KEY4HEP_STACK"] == "/stack/2026-10-20" and made["Study id"] == results.id
    assert [row["Evaluator"] for row in tables.evaluators(results)] == ["test.trade_off"]


def test_the_latest_published_study_comes_first() -> None:
    import tables

    def entry(study: str, published: str) -> CatalogEntry:
        outcomes = dict.fromkeys(
            (
                "feasible",
                "infeasible",
                "invalid",
                "failed",
                "rejected",
                "interrupted",
                "error",
                "running",
            ),
            0,
        )
        return CatalogEntry(
            study,
            f"name-{study[0]}",
            None,
            "then",
            published,
            0,
            outcomes,
            1,
            1,
            0,
            Snapshot("p", "0" * 64, 1),
        )

    catalog = Catalog(
        (
            entry("a" * 8 + "-0000-4000-8000-" + "0" * 12, "2026-11-01"),
            entry("b" * 8 + "-0000-4000-8000-" + "0" * 12, "2026-11-02"),
        )
    )
    assert [row["Study"] for row in tables.studies(catalog)] == ["name-b", "name-a"]
