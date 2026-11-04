"""The dashboard's tables: every value as the study records it, an empty cell where it has none,
headed by the names a reader knows the values by."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

from k4opt.optimization import Rule, load_study, trade_offs
from tests.support.studies import mixed_study


def test_a_trial_row_holds_its_values_and_nothing_made_up(tmp_path: Path) -> None:
    import tables

    results = load_study(mixed_study(tmp_path / "s"))
    rows = tables.trials(results)
    assert [row["Trial"] for row in rows] == [t.number for t in results.trials]
    a, margin = "a [mm]", "margin a max [mm]"  # the metric's name; its label is the column's help
    assert tables.helps(results)[a] == "test.trade_off/analysis/a"
    assert "test.trade_off/analysis/a:max" in tables.helps(results)[margin]
    uncertain = any(
        o.uncertainty is not None
        for t in results.trials
        if t.assessment is not None
        for o in t.assessment.objectives or ()
    )
    for row, trial in zip(rows, results.trials, strict=True):
        assert (row["Outcome"], row["Pareto"]) == (trial.outcome, trial.number in results.pareto)
        assert row["radius"] == trial.candidate["radius"]
        assert ("a [mm] ±" in row) == uncertain  # a column of uncertainties only if there are any
        if trial.objectives is None:  # invalid, failed: no value, no margin, never 0
            assert row[a] is None and row[margin] is None
        else:
            assert row[a] == trial.objectives["test.trade_off/analysis/a"]
            assert row[margin] == trial.assessment.limits[0].margin
            assert (row[margin] <= 0) == (trial.outcome == "feasible")
        assert row["Message"] == trial.message and "Record" not in row  # the record: in its sheet
    assert [row["Trial"] for row in tables.trials(results, results.pareto)] == list(results.pareto)
    made = {row[""]: row["Value"] for row in tables.provenance(results)}
    assert made["KEY4HEP_STACK"] == "/stack/2026-10-20" and made["Study id"] == results.id
    assert [row["Evaluator"] for row in tables.evaluators(results)] == ["test.trade_off"]


def test_the_best_of_each_objective_is_the_first_feasible_trial_to_reach_it(
    tmp_path: Path,
) -> None:
    import tables

    results = load_study(mixed_study(tmp_path / "s"))
    feasible = [t for t in results.trials if t.outcome == "feasible"]
    bests = tables.bests(results)
    assert [best.name for best in bests] == ["a", "b"]
    for best in bests:  # both minimized
        values = [t.objectives[best.label] for t in feasible]
        assert best.value == min(values) and best.direction == "minimize"
        first = next(t for t in feasible if t.objectives[best.label] == best.value)
        assert best.trial == first.number
    assert bests[0].unit == "mm" and bests[1].unit == "1/mm"


def test_a_trial_is_shown_with_all_it_recorded(tmp_path: Path) -> None:
    import tables

    results = load_study(mixed_study(tmp_path / "s"))
    trial = next(t for t in results.trials if t.outcome == "infeasible")
    record = {row[""]: row["Value"] for row in tables.trial_record(trial)}
    assert record == {"Record": trial.record, "Record SHA-256": trial.sha256}
    (limit,) = tables.trial_limits(results, trial)
    assert (limit["Constraint"], limit["Bound"], limit["Limit"]) == ("a", "max", 400)
    assert limit["Satisfied"] is False and limit["Margin"] == trial.assessment.limits[0].margin > 0
    stages = tables.trial_stages(trial)
    assert sorted(row["Stage"] for row in stages) == sorted(trial.stage_seconds)
    assert [row["Seconds"] for row in stages] == sorted(trial.stage_seconds.values(), reverse=True)
    assert tables.trial_diagnostics(trial) == []
    failed = next(t for t in results.trials if t.outcome == "failed")
    assert [(row["Value"], row["Satisfied"]) for row in tables.trial_limits(results, failed)] == [
        (None, None)
    ]


def test_a_trade_off_lists_the_reference_and_each_match_with_its_changes(tmp_path: Path) -> None:
    import tables

    results = load_study(mixed_study(tmp_path / "s"))
    feasible = [t for t in results.trials if t.outcome == "feasible"]
    reference = feasible[0]
    found = trade_offs(
        results, reference.number, [Rule("test.trade_off/analysis/a", "worsen", 10.0)]
    )
    rows = tables.trade_offs(results, found)
    assert [row["Trial"] for row in rows] == [reference.number, *found.matches]
    assert rows[0]["Reference"] is True and rows[0]["Δ a [%]"] == 0.0
    for row in rows[1:]:
        trial = results.trial(row["Trial"])
        base = reference.objectives["test.trade_off/analysis/a"]
        change = trial.objectives["test.trade_off/analysis/a"] - base
        assert row["Δ a [%]"] == pytest.approx(100 * change / abs(base))


def test_parameters_titled_alike_each_keep_a_column_of_their_own() -> None:
    import yaml

    from k4opt.frozen import FrozenMap
    from k4opt.optimization import StudyResults, TrialResult
    from k4opt.problem import parse_problem

    import tables

    examples = Path(__file__).resolve().parents[3] / "examples" / "problem"
    document = yaml.safe_load((examples / "cld-ecal-bindings.yaml").read_text())
    document["parameters"]["ecal_cell_size"]["title"] = "Size"
    document["parameters"]["range_cut"]["title"] = "Size"  # in mm too: one heading but for names
    spec = parse_problem(document)
    candidate = {"ecal_cell_size": 5.0, "range_cut": 0.5, "cell_threshold": 70.0}
    trial = TrialResult(0, "valid", candidate, record="r")
    results = StudyResults("s", "id", "now", spec, FrozenMap({}), (trial,), ())
    (row,) = tables.trials(results)
    assert row["Size (ecal_cell_size) [mm]"] == 5.0 and row["Size (range_cut) [mm]"] == 0.5


def test_no_column_takes_the_place_of_another_whatever_the_titles(tmp_path: Path) -> None:
    from k4opt.optimization import StudyResults
    from k4opt.problem import parse_problem

    import tables

    results = load_study(mixed_study(tmp_path / "s"))
    document = results.problem.to_dict()
    document["parameters"]["radius"]["title"] = "Message"  # a fixed column's heading
    document["objectives"][0]["title"] = "b"  # the other objective's metric, in another unit
    document["objectives"][1]["title"] = "a"
    document["constraints"][0]["title"] = "b"
    spec = parse_problem(document)
    titled = StudyResults(
        results.name, results.id, results.created, spec, results.document, results.trials, ()
    )
    rows = tables.trials(titled)
    count = len(rows[0])
    assert count == len(tables.trials(results)[0])  # every column kept
    trial = next(t for t in results.trials if t.objectives is not None)
    (row,) = (r for r in rows if r["Trial"] == trial.number)
    assert row["Message"] == trial.message and row["Message (radius)"] == trial.candidate["radius"]
    assert row["b [mm]"] == trial.objectives["test.trade_off/analysis/a"]
    assert row["margin b max [mm]"] == trial.assessment.limits[0].margin
