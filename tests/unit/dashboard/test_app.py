"""The dashboard, run as Streamlit runs it (``AppTest``) on real publications: what it shows of a
study, and what it says when there is nothing to show, from a directory and over HTTP."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("optuna")

from streamlit.testing.v1 import AppTest

from k4opt.optimization import StudyResults, load_study
from k4opt.publication import CATALOG_NAME, Catalog, CatalogEntry, Snapshot, publish
from k4opt.visualization import task_graph
from tests.support.serve import served
from tests.support.studies import mixed_study
from tests.unit.dashboard.conftest import DASHBOARD
from tests.unit.publication.test_read import closed_port


def app(monkeypatch: pytest.MonkeyPatch, location: object | None) -> AppTest:
    if location is not None:
        monkeypatch.setenv("K4OPT_DASHBOARD_DATA", str(location))
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=60)


def texts(at: AppTest) -> str:
    shown = [*at.info, *at.warning, *at.error, *at.caption, *at.markdown, *at.title]
    return "\n".join(str(element.value) for element in shown)


def study_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, results: StudyResults, **params: str
) -> AppTest:
    """The Study page of ``results``, published in ``tmp_path / "pub"``, with the URL's
    ``params`` besides the study's id."""
    at = app(monkeypatch, tmp_path / "pub")
    at.switch_page("study.py")
    for key, value in {"study": results.id, **params}.items():
        at.query_params[key] = value
    return at.run()


def published(tmp_path: Path, name: str = "s", trials: int = 8) -> StudyResults:
    results = load_study(mixed_study(tmp_path / name, trials))
    publish(results, tmp_path / "pub")
    return results


def test_without_a_publication_it_starts_and_says_how_to_name_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = app(monkeypatch, None).run()
    assert not at.exception
    assert "No publication is configured. Set `K4OPT_DASHBOARD_DATA`" in texts(at)
    source = "\n".join(m.value for m in at.sidebar.markdown)
    assert "not configured" in source and "set K4OPT_DASHBOARD_DATA" in source
    assert at.sidebar.button[0].disabled
    at.switch_page("study.py").run()
    assert not at.exception and "No publication is configured" in texts(at)


@pytest.mark.parametrize(
    ("content", "level", "message"),
    [
        (None, "info", "Nothing is published here yet"),
        ('{"format": "k4opt.catalog/1", "studies": []}', "info", "names no study yet"),
        ("{", "error", "The publication does not hold up"),
    ],
)
def test_an_empty_or_broken_publication_is_said_to_be_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None, level: str, message: str
) -> None:
    if content is not None:
        (tmp_path / CATALOG_NAME).write_text(content)
    at = app(monkeypatch, tmp_path).run()
    assert not at.exception
    assert message in "\n".join(str(e.value) for e in getattr(at, level))


def test_a_publication_that_does_not_answer_is_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = app(monkeypatch, closed_port()).run()
    assert not at.exception
    assert "cannot be read just now" in at.warning[0].value


def test_the_studies_of_a_publication_are_listed_and_one_is_shown_by_its_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = app(monkeypatch, tmp_path / "pub").run()
    assert not at.exception and at.title[0].value == "Studies"
    (listed,) = at.dataframe[0].value.to_dict("records")
    assert (listed["Study"], listed["Problem"], listed["Trials"]) == (results.name, "trade", 8)
    assert listed["Detector"] == PurePosixPath(results.problem.geometry).stem
    assert listed["Key4hep"] == "nightly 2026-10-20"
    assert listed["Feasible"] == results.outcomes["feasible"]
    picker = {box.label: box for box in at.sidebar.selectbox}
    assert picker["Detector"].value == "*" and picker["Key4hep"].options == [
        "Every release",
        "nightly 2026-10-20",
    ]
    picker["Key4hep"].select("nightly 2026-10-20").run()  # narrows the table
    assert not at.exception and len(at.dataframe[0].value) == 1
    picker = {box.label: box for box in at.sidebar.selectbox}
    picker["Study"].select(results.id).run()  # chosen in the sidebar: it opens
    assert not at.exception and at.query_params["study"] == results.id
    assert f"**{results.name}** · problem trade" in texts(at)  # under the tabs, what is shown
    # the Overview: how far it got, what it optimizes, how a candidate is evaluated
    assert [chart.key for chart in at.get("plotly_chart")] == ["tasks-0"]
    metrics = {m.label: m for m in at.metric}
    assert metrics["Trials"].value == "8"
    assert metrics["Pareto front"].value == str(len(results.pareto))
    feasible = [t for t in results.trials if t.outcome == "feasible"]
    best = min(feasible, key=lambda t: t.objectives["test.trade_off/analysis/a"])
    assert metrics["Best a"].value == f"{best.objectives['test.trade_off/analysis/a']:.4g} mm"
    assert metrics["Best a"].proto.delta == f"trial {best.number}"
    # the Results: the trials in the plane of two axes, and the front
    at = study_page(monkeypatch, tmp_path, results, view="Results")
    assert [chart.key for chart in at.get("plotly_chart")] == ["trials-0"]
    (front,) = [frame.value for frame in at.dataframe]
    assert list(front["Trial"]) == list(results.pareto)
    assert at.query_params["x"].startswith("objective:")  # a link reopens the same view
    assert at.query_params["color"] == "outcome"


def test_the_study_page_without_a_study_shows_the_latest_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    studies = {s.id: s for s in (published(tmp_path, "one", 2), published(tmp_path, "two", 2))}
    at = app(monkeypatch, tmp_path / "pub")
    at.switch_page("study.py").run()
    assert not at.exception
    import data

    latest = data.newest_first(data.current().catalog)[0]  # the sidebar's first
    assert at.query_params["study"] == latest.id
    assert f"**{studies[latest.id].name}**" in texts(at)


def test_the_latest_published_study_comes_first() -> None:
    import data

    def entry(study: str, published: str) -> CatalogEntry:
        outcomes = dict.fromkeys(
            ("feasible", "infeasible", "invalid", "failed", "rejected", "interrupted", "error"),
            0,
        )
        return CatalogEntry(
            study,
            f"name-{study[0]}",
            None,
            "Ring/compact/Ring.xml",
            "nightly 2026-10-20",
            "then",
            published,
            0,
            outcomes | {"running": 0},
            1,
            1,
            0,
            Snapshot("p", "0" * 64, 1),
        )

    first, second = (f"{c * 8}-0000-4000-8000-{'0' * 12}" for c in "ab")
    catalog = Catalog((entry(first, "2026-11-01"), entry(second, "2026-11-02")))
    assert [e.name for e in data.newest_first(catalog)] == ["name-b", "name-a"]


def test_the_trials_tab_lists_every_trial_and_shows_the_one_a_link_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, view="Trials")
    assert not at.exception and not at.get("plotly_chart")  # the Results do not run
    (every,) = [frame.value for frame in at.dataframe]
    assert list(every["Outcome"]) == [t.outcome for t in results.trials]
    assert "select one to open all it recorded" in texts(at)
    trial = next(t for t in results.trials if t.outcome == "infeasible")
    for view in ("Trials", "Results"):  # a link opens its sheet over any tab
        at = study_page(monkeypatch, tmp_path, results, view=view, trial=str(trial.number))
        assert not at.exception and ":orange-badge[infeasible]" in texts(at)
        tiles = {m.label: m.value for m in at.metric}
        assert tiles["▼ a"] == f"{trial.objectives['test.trade_off/analysis/a']:.4g} mm"
        assert tiles["radius"] == f"{trial.candidate['radius']:.4g}"
        frames = [frame.value for frame in at.dataframe]
        (limits,) = [f for f in frames if "Margin" in f.columns]
        assert list(limits["Satisfied"]) == [False]
        assert any(trial.record in list(f.iloc[:, 1]) for f in frames)  # its record, by its id
    for missing in ("999", "x"):  # a trial the study lacks names none, and is dropped
        at = study_page(monkeypatch, tmp_path, results, view="Trials", trial=missing)
        assert not at.exception and "trial" not in at.query_params
        assert not any(m.label == "radius" for m in at.metric)


def test_the_problem_tab_says_what_the_study_optimizes_and_what_it_was_made_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, view="Overview")
    assert not at.exception
    said = texts(at)
    assert "**radius**" in said and "**b**" in said and ":red-badge[▼ minimize]" in said
    assert ":gray-badge[1/mm]" in said and "test.trade\\_off/analysis/b" in said
    assert "**a**" in said and ":red-badge[≤ 400 mm]" in said  # the constraint, by its bound
    frames = [frame.value for frame in at.dataframe]
    assert any("KEY4HEP_STACK" in list(f.iloc[:, 0]) for f in frames)  # its provenance


def test_the_trials_are_colored_by_what_a_link_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, view="Results", color="trial")
    assert not at.exception and at.query_params["color"] == "trial"
    at = study_page(
        monkeypatch, tmp_path, results, view="Results", color="parameter:radius"
    )  # categorical
    assert not at.exception and at.query_params["color"] == "outcome"


def test_the_tasks_of_a_candidate_are_drawn_and_one_sample_has_no_stack_to_expand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, view="Overview")
    assert not at.exception
    assert [chart.key for chart in at.get("plotly_chart")] == ["tasks-0"]
    assert not at.toggle and not task_graph(results).samples  # nothing to stack or fold
    (full,) = [button for button in at.button if button.label == "Full screen"]
    full.click().run()  # the model across the screen, the wheel zooming it
    assert not at.exception
    assert [chart.key for chart in at.get("plotly_chart")] == ["tasks-0", "tasks-full-chart"]


def test_a_link_to_a_study_the_catalog_lacks_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published(tmp_path)
    at = app(monkeypatch, tmp_path / "pub").run()
    at.switch_page("study.py")
    at.query_params["study"] = "00000000-0000-4000-8000-000000000000"
    at.run()
    assert not at.exception
    assert "has no study 00000000-0000-4000-8000-000000000000; choose" in at.warning[0].value
    assert not at.metric  # nothing shown as if it were that study


def test_the_same_studies_are_shown_over_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    with served(tmp_path / "pub") as server:
        at = app(monkeypatch, server.url).run()
        at.switch_page("study.py")
        at.query_params["study"] = results.id
        at.run()
        assert not at.exception and f"**{results.name}**" in texts(at)
        assert server.requests[0] == CATALOG_NAME
        assert server.requests[1].startswith(f"studies/{results.id}/snapshots/")


def test_refresh_reads_a_newly_published_study_and_a_failed_read_is_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published(tmp_path, "one", 2)
    with served(tmp_path / "pub") as server:
        at = app(monkeypatch, server.url).run()
        assert len(at.dataframe[0].value) == 1
        published(tmp_path, "two", 2)
        at.run()
        assert len(at.dataframe[0].value) == 1  # the catalog is kept a while
        at.sidebar.button[0].click().run()
        assert len(at.dataframe[0].value) == 2  # Refresh reads it again
        server.faults[CATALOG_NAME] = "error"
        at.sidebar.button[0].click().run()
        assert "cannot be read just now" in at.warning[0].value
        server.faults.clear()
        at.run()  # no Refresh: the failure was not kept
        assert not at.warning and len(at.dataframe[0].value) == 2


def test_text_from_a_link_or_a_catalog_is_shown_as_it_is_never_as_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published(tmp_path)
    at = app(monkeypatch, tmp_path / "pub").run()
    at.switch_page("study.py")
    at.query_params["study"] = "`[sign in](https://example.org)`"
    at.run()
    (warning,) = at.warning
    assert r"\`\[sign in\](https://example.org)\`" in warning.value  # escaped: no link


def test_a_catalog_entry_is_checked_against_its_snapshot_on_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    results = published(tmp_path)
    at = app(monkeypatch, tmp_path / "pub").run()
    at.switch_page("study.py")
    at.query_params["study"] = results.id
    at.run()
    assert f"**{results.name}**" in texts(at)  # read, and kept by its snapshot
    catalog = json.loads((tmp_path / "pub" / CATALOG_NAME).read_text())
    catalog["studies"][0]["name"] = "renamed"  # the same snapshot, another entry
    (tmp_path / "pub" / CATALOG_NAME).write_text(json.dumps(catalog))
    at.sidebar.button[0].click().run()
    assert "not the study its catalog entry describes: its name" in at.error[0].value


def test_the_analysis_tab_shows_how_the_parameters_act_on_the_chosen_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, view="Analysis")
    assert not at.exception
    shown = at.segmented_control(key="analysis")  # one view at a time; one parameter: no rank
    assert shown.options == ["History", "Parallel coordinates", "Slice", "Distribution"]
    assert [chart.key for chart in at.get("plotly_chart")] == ["history-0"]
    assert at.query_params["objective"] == "objective:test.trade_off/analysis/a"
    at = study_page(
        monkeypatch,
        tmp_path,
        results,
        view="Analysis",
        objective="objective:test.trade_off/analysis/b",
        analysis="Slice",
    )
    assert not at.exception and at.query_params["objective"].endswith("/b")
    assert [chart.key for chart in at.get("plotly_chart")] == ["slice-0"]


def test_a_trade_off_in_the_url_highlights_the_trials_that_meet_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    reference = next(t for t in results.trials if t.outcome == "feasible")
    at = study_page(
        monkeypatch,
        tmp_path,
        results,
        view="Trade-offs",
        ref=str(reference.number),
        rules="0:worsen:1000:%",
    )
    assert not at.exception
    assert [chart.key for chart in at.get("plotly_chart")] == ["trade-0"]
    others = results.outcomes["feasible"] - 1
    assert f"**{others} of {others} feasible trials** meet this" in texts(at)
    table = next(f.value for f in at.dataframe if "Reference" in f.value.columns)
    assert table["Trial"].iloc[0] == reference.number and table["Reference"].iloc[0]
    assert at.query_params["rules"] == "0:worsen:1000:%"
    at = study_page(
        monkeypatch,
        tmp_path,
        results,
        view="Trade-offs",
        ref=str(reference.number),
        rules="1:keep:0:%,x",
    )
    assert not at.exception and at.query_params["rules"] == "1:keep:0:%"  # the rest dropped
    at = study_page(
        monkeypatch, tmp_path, results, view="Trade-offs", ref="999", rules="0:keep:0:%"
    )
    assert not at.exception and "ref" not in at.query_params  # a trial the study lacks: none


def test_a_study_of_a_level_below_opens_by_the_url_and_a_bad_path_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import nested_study

    results = load_study(nested_study(tmp_path / "s", monkeypatch))
    publish(results, tmp_path / "pub")
    at = study_page(monkeypatch, tmp_path, results, inner="1")
    assert not at.exception
    assert "The study of level 1 for trial 1" in texts(at)
    for held in ("²", "1.x", "7", "-1"):  # no such trial, or no number at all
        at = study_page(monkeypatch, tmp_path, results, inner=held)
        assert not at.exception, held
        assert "The study of level 1" not in texts(at)
