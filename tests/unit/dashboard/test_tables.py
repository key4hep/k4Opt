"""The dashboard's tables: every value as the study records it, an empty cell where it has none,
headed by the names a reader knows the values by."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from k4opt.frozen import FrozenMap
from k4opt.optimization import Rule, load_study, trade_offs
from k4opt.publication import PublishedArtifact, PublishedRequirement, SoftwareSheet
from tests.support.software import LABEL
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
    assert made["Study id"] == results.id and "KEY4HEP_STACK" not in made
    assert made["Key4hep build"] == LABEL
    assert made["Software id"] == results.document["software"]["id"]
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


def test_a_heading_still_alike_after_its_whole_name_is_numbered() -> None:
    from k4opt.frozen import FrozenMap
    from k4opt.optimization import StudyResults, TrialResult
    from k4opt.problem import parse_problem
    from tests.unit.optimization.conftest import RADIUS, problem_document

    import tables

    # the parameter and an objective titled alike; each then takes its whole name, which are
    # alike too: the objective's metric is named as the parameter's title and name
    title = "test.trade_off/analysis/a"
    document = problem_document(objectives=[("analysis", f"a ({RADIUS})", "minimize")])
    document["parameters"][RADIUS]["title"] = title
    document["objectives"][0]["title"] = title
    spec = parse_problem(document)
    trial = TrialResult(0, "valid", {RADIUS: 300.0}, record="r")
    results = StudyResults("s", "id", "now", spec, FrozenMap({}), (trial,), ())
    (row,) = tables.trials(results)
    assert len(row) == 3 + 1 + 1 + 2  # the fixed columns, the parameter, the objective
    assert row[f"{title} ({RADIUS})"] == 300.0  # the parameter's, the first column of the two
    assert row[f"{title} ({RADIUS}) (2)"] is None  # the objective's, which the trial has not


def test_a_study_of_a_level_below_is_tabled_by_the_objective_it_is_judged_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("optuna")
    from tests.support.studies import nested_study

    import tables

    inner = load_study(nested_study(tmp_path / "s", monkeypatch)).trial(0).inner
    assert inner is not None
    (row, *_) = tables.trials(inner)
    assert "seen" in row and not any("cells" in heading for heading in row)
    (best,) = tables.bests(inner)
    assert best.label == "test.candidate_check/analysis/seen"


def test_the_tuned_value_stands_beside_the_objective_it_tuned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import nested_study

    import tables

    top = load_study(nested_study(tmp_path / "s", monkeypatch))
    (row, *_) = tables.trials(top)
    headings = list(row)
    # the inner objective is no objective of the top: its tuned value comes after the objectives
    assert headings.index("seen, tuned") > headings.index("ecal_barrel.mean_hit_cells")
    trial = top.trial(0)
    found = tables.tuned(trial)
    assert trial.inner is not None and trial.selected is not None and found is not None
    assert found == trial.inner.trial(int(str(trial.selected["trial"]))).assessment.objectives[0]
    assert row["seen, tuned"] == found.value


def test_the_tuned_value_is_headed_with_the_unit_the_tuning_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.optimization import StudyResults
    from tests.support.studies import nested_study

    import tables

    data = load_study(nested_study(tmp_path / "s", monkeypatch)).to_dict()
    for trial in data["trials"]:  # an inner-only metric measured in mm
        for inner in trial["inner"]["trials"]:
            for value in inner["assessment"]["objectives"]:
                value["unit"] = "mm"
    (row, *_) = tables.trials(StudyResults.from_dict(data))
    assert "seen, tuned [mm]" in row


def _requirement(**changes: object) -> PublishedRequirement:
    values: dict[str, Any] = {
        "id": "package:ddmarlinpandora",
        "kind": "package",
        "label": "ddmarlinpandora",
        "identified": "821384f + 1 uncommitted files (verified link)",
        "used_by": ("pi0/reconstruction",),
        "state": "available",
        "access": "public",
        "notes": (),
        "facts": {"base": "821384f", "link": "verified", "rebuildable": True, "changed": 1},
        "ran": FrozenMap(
            {
                "original": "fingerprint-only",
                "files": (FrozenMap({"file": "lib/libD.so", "sha256": "1f2e" * 16}),),
            }
        ),
        "unused": False,
    }
    return PublishedRequirement(**(values | changes))


def _base(checked: str = "2026-11-12T10:00:00+00:00") -> PublishedRequirement:
    return _requirement(
        id="base",
        kind="base",
        label="Key4hep LCG devkey-head, built Sat Nov 7",
        identified="x86_64-el9-gcc16-opt, 900 packages",
        used_by=("pi0/simulation", "pi0/reconstruction"),
        notes=("a nightly: a later build overwrites its slot, and this one is then gone",),
        facts={},
        ran=FrozenMap({"original": "on-cvmfs"}),
    )


def _sheet(*requirements: PublishedRequirement, **changes: object) -> SoftwareSheet:
    values: dict[str, Any] = {
        "study": "s",
        "snapshot": "0" * 64,
        "status": "Not rerun · rebuild needed",
        "checked": "2026-11-12T10:00:00+00:00",
        "first_checked": "2026-11-12T10:00:00+00:00",
        "first_status": "Not rerun · rebuild needed",
        "gaps": (),
        "altered": (),
        "not_kept": 0,
        "on_request": (),
        "requirements": requirements,
        "artifacts": (),
    }
    return SoftwareSheet(**(values | changes))


def test_a_local_build_says_rebuildable_and_never_that_its_binary_is_available() -> None:
    import tables

    (row,) = tables.local_changes(_sheet(_requirement()))
    assert row["Change"] == "Modified"
    assert row["Source"] == "821384f + 1 uncommitted files · Verified"
    assert row["Rebuild"] == "Rebuildable from source"
    assert "Original binary: fingerprint only" in str(row["What ran"])
    assert "libD.so 1f2e1f2e…" in str(row["What ran"])
    assert "fingerprint" in tables.REBUILD_NOTE


def test_a_build_that_did_not_run_is_said_not_to_have_run() -> None:
    import tables

    (row,) = tables.local_changes(_sheet(_requirement(unused=True, ran=None, used_by=())))
    assert row["Change"] == "Changed locally — not used"
    assert row["Rebuild"] is None and row["What ran"] == "not read by any stage"


def test_which_stages_used_the_changes_is_said_only_when_known() -> None:
    import tables

    sheet = _sheet(_base(), _requirement())
    assert tables.stages_used(sheet) == (
        "pi0/simulation used the nightly only; pi0/reconstruction used the changes above."
    )
    assert tables.stages_used(_sheet(_base(), _requirement(), gaps=("x not observed",))) is None


def test_a_nightly_checked_long_ago_may_have_been_overwritten() -> None:
    import tables

    sheet = _sheet(_base())
    assert not tables.software_warnings(sheet, datetime(2026, 11, 15, tzinfo=UTC))
    (line,) = tables.software_warnings(sheet, datetime(2026, 12, 1, tzinfo=UTC))
    assert "may have been overwritten" in line


def test_a_first_check_and_a_later_one_are_kept_apart() -> None:
    import tables

    assert "not checked since" in tables.checked(_sheet())
    later = tables.checked(_sheet(checked="2026-12-01T10:00:00+00:00"))
    assert "At publication, 12 Nov 2026" in later and "Last checked 1 Dec 2026" in later


def test_a_private_input_shows_its_commitment_and_a_published_one_does_not() -> None:
    import tables

    private = _requirement(
        id="input:1",
        kind="input",
        label="ecal_scale.json",
        identified="observed input",
        access="on request",
        facts={"commitment": "ab" * 32},
        ran=None,
    )
    (row,) = tables.inputs(_sheet(private))
    assert row["Access"] == "On request: ask the author"
    assert row["Commitment"] == f"sha256:{'ab' * 6}…"
    (row,) = tables.inputs(_sheet(replace(private, access="public")))
    assert row["Commitment"] is None and row["Known from"] == "observed"


def test_the_steps_of_a_rerun_say_where_each_thing_is() -> None:
    import tables

    private = _requirement(
        id="input:1", kind="input", label="e.json", access="on request", facts={}, ran=None
    )
    steps = tables.rerun_steps(_sheet(_base(), _requirement(), private))
    assert steps[0] == "Source the nightly: Key4hep LCG devkey-head, built Sat Nov 7."
    assert "Clone ddmarlinpandora at 821384f, apply its changes (published here)" in steps[1]
    assert steps[2] == "Obtain e.json (ask the author)."
    assert steps[-1].startswith("Evaluate the candidate")


def test_only_a_served_publication_links_to_its_files() -> None:
    import tables

    artifact = PublishedArtifact(
        "code", "o", "o.patch", "patch", "a" * 64, 1, "studies/x/artifacts/aaaa", (), (), "b", ()
    )
    assert tables.artifact_link("/eos/pub", artifact) is None
    assert (
        tables.artifact_link("https://k4opt.web.cern.ch/", artifact)
        == "https://k4opt.web.cern.ch/studies/x/artifacts/aaaa"
    )


def test_a_rerun_says_how_to_get_the_geometry_the_evaluators_and_k4opt() -> None:
    import tables

    geometry = _requirement(
        id="geometry", kind="geometry", label="C/d.xml", identified="4 files, kept", facts={}
    )
    evaluator = _requirement(
        id="evaluator:mine", kind="evaluator", label="mine v1", access="on request", facts={}
    )
    k4opt = _requirement(id="k4opt", kind="k4opt", label="k4opt 0.1", facts={"commit": "a" * 40})
    steps = "\n".join(tables.rerun_steps(_sheet(_base(), geometry, evaluator, k4opt)))
    assert "Restore the geometry C/d.xml (published here) and set K4GEO to it" in steps
    assert "Get the evaluator mine v1 (ask the author)" in steps
    assert f"Use k4opt at commit {'a' * 12}" in steps


def test_a_rerun_says_when_what_is_published_does_not_suffice() -> None:
    import tables

    private = _requirement(
        id="input:1", kind="input", label="e.json", access="on request", facts={}, ran=None
    )
    assert tables.rerun_missing(_sheet(_base(), _requirement())) == []
    assert tables.rerun_missing(_sheet(_base(), private)) == ["e.json"]


def test_a_build_says_whether_its_options_can_be_had_beside_its_source() -> None:
    import tables

    published = _requirement(
        facts={"link": "verified", "rebuildable": True, "configuration": "published"}
    )
    (row,) = tables.local_changes(_sheet(published))
    assert (row["Rebuild"], row["Build options"]) == ("Rebuildable from source", "Published")
    asked = _requirement(
        facts={"link": "verified", "rebuildable": True, "configuration": "on request"}
    )
    (row,) = tables.local_changes(_sheet(asked))
    assert row["Build options"] == "On request"
    step = next(s for s in tables.rerun_steps(_sheet(asked)) if s.startswith("Clone"))
    assert "its CMake options (ask the author)" in step


def test_a_build_published_in_part_says_there_is_more_to_ask_for() -> None:
    import tables

    partial = _requirement(
        facts={"link": "verified", "rebuildable": True, "configuration": "partial"}
    )
    (row,) = tables.local_changes(_sheet(partial))
    assert row["Build options"] == "Partially published: ask the author for the rest"
    step = next(s for s in tables.rerun_steps(_sheet(partial)) if s.startswith("Clone"))
    assert "its CMake options (in part here; ask the author for the rest)" in step


def test_a_python_environment_a_stage_used_is_said_to_be_missing_from_a_rerun() -> None:
    import tables

    used = _requirement(
        id="environment:1",
        kind="environment",
        label="a Python environment",
        identified="60 distributions",
        access="none",
        facts={},
        ran=None,
    )
    assert tables.rerun_missing(_sheet(_base(), used)) == ["a Python environment"]
    assert any("60 distributions" in step for step in tables.rerun_steps(_sheet(_base(), used)))
    idle = replace(used, used_by=())
    assert tables.rerun_missing(_sheet(_base(), idle)) == []
