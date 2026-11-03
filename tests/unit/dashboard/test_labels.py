"""How the dashboard words values: four significant digits, a count by its SI prefix, no unit of
``1``, times in UTC as dates read, the name a metric is known by."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("value", "unit", "uncertainty", "shown"),
    [
        (0.83412, "1", None, "0.8341"),
        (11.9136, "mm", None, "11.91 mm"),
        (0.864, None, 0.0123, "0.864 ± 0.012"),
        (16827574.8, "1", None, "16.83 M"),
        (16827574.8, None, 120000.0, "16.83 ± 0.12 M"),
        (15360309.9, "mm", None, "1.536e7 mm"),
        (2.5e-6, "m", None, "2.5e-6 m"),
        (42, "events", None, "42 events"),
        (12340, None, None, "12.34 k"),
        ("tungsten", None, None, "tungsten"),
        (True, None, None, "True"),
    ],
)
def test_a_number_is_worded_to_four_significant_digits(
    value: object, unit: str | None, uncertainty: float | None, shown: str
) -> None:
    import labels

    assert labels.number(value, unit, uncertainty) == shown


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [(0.00023, "0.23 ms"), (4.12, "4.1 s"), (1756.8, "29 min 17 s"), (7512, "2 h 05 min")],
)
def test_a_duration_reads_in_its_largest_units(seconds: float, shown: str) -> None:
    import labels

    assert labels.duration(seconds) == shown


def test_a_time_is_a_date_in_utc_and_recent_times_are_how_long_ago() -> None:
    import labels

    assert labels.when("2026-11-03T15:29:25+00:00") == "3 Nov 2026, 15:29 UTC"
    assert labels.when("2026-11-03T16:29:25+01:00") == "3 Nov 2026, 15:29 UTC"
    assert labels.when("then") == "then"  # no time: as it is
    now = datetime(2026, 11, 3, 18, 0, tzinfo=UTC)
    assert labels.ago("2026-11-03T17:59:30+00:00", now) == "just now"
    assert labels.ago("2026-11-03T17:00:00+00:00", now) == "1 hour ago"
    assert labels.ago("2026-11-01T12:00:00+00:00", now) == "2 days ago"
    assert labels.ago("2026-10-01T12:00:00+00:00", now) == "1 Oct 2026, 12:00 UTC"
    assert (
        labels.count(1, "study", "studies") == "1 study" and labels.count(2, "trial") == "2 trials"
    )


def test_text_never_formats_as_markdown() -> None:
    import labels

    assert labels.plain("`[x](y)` *a_b*") == r"\`\[x\](y)\` \*a\_b\*"


def test_a_metric_is_named_by_its_name_and_by_its_evaluation_only_where_two_are_alike(
    tmp_path: Path,
) -> None:
    pytest.importorskip("optuna")
    from k4opt.optimization import load_study
    from tests.support.studies import mixed_study

    import labels

    results = load_study(mixed_study(tmp_path / "s"))
    assert labels.names(results) == {
        "test.trade_off/analysis/a": "a",
        "test.trade_off/analysis/b": "b",
    }
    assert labels.axis("objective:test.trade_off/analysis/b", results) == "b · objective"
    assert labels.axis("parameter:radius", results) == "radius · parameter"
    assert (labels.axis("trial", results), labels.axis("outcome", results)) == (
        "trial number",
        "outcome",
    )
