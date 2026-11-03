"""The dashboard, run as Streamlit runs it (``AppTest``) on real publications: what it shows of a
study, and what it says when there is nothing to show, from a directory and over HTTP."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("optuna")

from streamlit.testing.v1 import AppTest

from k4opt.optimization import StudyResults, load_study
from k4opt.publication import CATALOG_NAME, publish
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
    assert at.sidebar.code[0].value == "not configured"
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
    assert listed["Feasible"] == results.outcomes["feasible"]
    assert listed["Feasible share"] == results.outcomes["feasible"] / 8
    assert "1 study; catalog read" in "\n".join(c.value for c in at.sidebar.caption)
    at.switch_page("study.py")
    at.query_params["study"] = results.id
    at.run()
    assert not at.exception
    assert at.title[0].value == results.name
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Trials"] == "8"
    assert metrics["Infeasible"] == str(results.outcomes["infeasible"])
    assert metrics["Pareto front"] == str(len(results.pareto))
    # the Overview: the trials in the plane of two axes, each objective's history, the front
    assert [chart.key for chart in at.get("plotly_chart")] == [
        "trials-0",
        "history-0-0",
        "history-1-0",
    ]
    (front,) = [frame.value for frame in at.dataframe]
    assert list(front["Trial"]) == list(results.pareto)
    assert at.query_params["x"].startswith("objective:")  # a link reopens the same view
    assert at.query_params["color"] == "outcome"


def test_the_trials_tab_lists_every_trial_and_shows_the_one_a_link_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, view="Trials")
    assert not at.exception and not at.get("plotly_chart")  # the Overview does not run
    (every,) = [frame.value for frame in at.dataframe]
    assert list(every["Outcome"]) == [t.outcome for t in results.trials]
    assert "Select a trial for all it recorded" in texts(at)
    trial = next(t for t in results.trials if t.outcome == "infeasible")
    at = study_page(monkeypatch, tmp_path, results, view="Trials", trial=str(trial.number))
    assert not at.exception
    assert f"#### Trial {trial.number}" in texts(at) and ":orange-badge[infeasible]" in texts(at)
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["a"] == f"{trial.objectives['test.trade_off/analysis/a']:.4g} mm"
    frames = [frame.value for frame in at.dataframe]
    (limits,) = [f for f in frames if "Margin" in f.columns]
    assert list(limits["Satisfied"]) == [False]
    candidate = next(f for f in frames if "Parameter" in f.columns)
    assert candidate.to_dict("records")[0]["Value"] == trial.candidate["radius"]
    for missing in ("999", "x"):  # a trial the study lacks names none
        at = study_page(monkeypatch, tmp_path, results, view="Trials", trial=missing)
        assert not at.exception and "Select a trial" in texts(at)


def test_the_problem_tab_says_what_the_study_optimizes_and_what_it_was_made_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, view="Problem")
    assert not at.exception
    frames = [frame.value for frame in at.dataframe]
    assert list(frames[0]["Parameter"]) == ["radius"]
    assert "test.trade_off/analysis/b" in list(frames[1]["Objective"])
    assert any("KEY4HEP_STACK" in list(f.iloc[:, 0]) for f in frames)


def test_the_trials_are_colored_by_what_a_link_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, color="trial")
    assert not at.exception and at.query_params["color"] == "trial"
    at = study_page(monkeypatch, tmp_path, results, color="parameter:radius")  # categorical
    assert not at.exception and at.query_params["color"] == "outcome"


def test_the_tasks_of_a_candidate_are_drawn_and_one_sample_has_no_stack_to_expand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = published(tmp_path)
    at = study_page(monkeypatch, tmp_path, results, view="Pipeline")
    assert not at.exception
    if shutil.which("dot") is None:  # the page says what it needs, and shows the rest
        assert "needs Graphviz's dot program" in texts(at)
    else:
        assert "Click a task for what it is" in texts(at)
    buttons = {button.label: button for button in at.button}
    assert buttons["Draw each sample"].disabled and buttons["Stack alike samples"].disabled


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
        assert not at.exception and at.title[0].value == results.name
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
    assert at.title[0].value == results.name  # read, and kept by its snapshot
    catalog = json.loads((tmp_path / "pub" / CATALOG_NAME).read_text())
    catalog["studies"][0]["name"] = "renamed"  # the same snapshot, another entry
    (tmp_path / "pub" / CATALOG_NAME).write_text(json.dumps(catalog))
    at.sidebar.button[0].click().run()
    assert "not the study its catalog entry describes: its name" in at.error[0].value
