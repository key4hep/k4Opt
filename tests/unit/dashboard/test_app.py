"""The dashboard, run as Streamlit runs it (``AppTest``) on real publications: what it shows of a
study, and what it says when there is nothing to show, from a directory and over HTTP."""

from __future__ import annotations

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
    assert len(at.get("plotly_chart")) == 1
    frames = [frame.value for frame in at.dataframe]
    every = next(f for f in frames if len(f) == 8 and "Outcome" in f.columns)
    assert list(every["Outcome"]) == [t.outcome for t in results.trials]
    front = next(f for f in frames if "Outcome" in f.columns and len(f) == len(results.pareto))
    assert list(front["Trial"]) == list(results.pareto)
    assert at.query_params["x"].startswith("objective:")  # a link reopens the same view


def test_a_link_to_a_study_the_catalog_lacks_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published(tmp_path)
    at = app(monkeypatch, tmp_path / "pub").run()
    at.switch_page("study.py")
    at.query_params["study"] = "00000000-0000-4000-8000-000000000000"
    at.run()
    assert not at.exception
    assert "has no study `00000000-0000-4000-8000-000000000000`" in at.warning[0].value
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
