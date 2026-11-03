"""The pieces every tab of the Study page is made of: a choice kept in the URL, a figure whose
trials open on a click, a table that formats its numbers and opens a row, the axes of a figure of
the trials."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import plotly.graph_objects as go
import streamlit as st

from k4opt.optimization import StudyResults
from k4opt.visualization import axis_choices, color_choices, default_axes

import labels
import tables

# the URL's parameters of the figure of the trials: its axes and what colors it
X, Y, COLOR = "x", "y", "color"
BY_OUTCOME = "outcome"  # the figure's colors unless chosen: each trial by its outcome
# a counter bumped whenever a trial opens, so that every figure and table that can open one comes
# back without a selection: closing the trial leaves nothing selected
PICKS = "trial-picks"


def choice(
    label: str,
    options: list[str],
    key: str,
    default: str | None,
    format_func: Callable[[str], str] = str,
    placeholder: str | None = None,
) -> str | None:
    """A choice kept in the URL's parameter ``key``, so that a link reopens it; a value the options
    do not hold gives ``default``."""
    held = st.query_params.get(key)
    current = held if held in options else default
    chosen = st.selectbox(
        label,
        options,
        index=None if current is None else options.index(current),
        format_func=format_func,
        placeholder=placeholder,
        key=f"choice-{key}",
    )
    if chosen is not None and chosen != held:
        st.query_params[key] = chosen
    return chosen


def keep_in_url(key: str, value: str | None) -> None:
    """``value`` as the URL's parameter ``key``, none if ``None``."""
    if value is None:
        if key in st.query_params:
            del st.query_params[key]
    elif st.query_params.get(key) != value:
        st.query_params[key] = value


def picks() -> int:
    """How many trials were opened in this session: the suffix of every selectable key."""
    return int(st.session_state.setdefault(PICKS, 0))


def chart(figure: go.Figure, name: str | None = None) -> int | None:
    """``figure``, drawn across the page; if ``name`` names it, each of its points is a trial (its
    ``customdata``), and the number of the one clicked is returned."""
    if name is None:
        st.plotly_chart(figure, theme="streamlit")
        return None
    event = st.plotly_chart(
        figure,
        theme="streamlit",
        key=f"{name}-{picks()}",
        on_select="rerun",
        selection_mode="points",
    )
    selection = getattr(event, "selection", None) or {}
    for point in selection.get("points", []):
        curve, index = point.get("curve_number"), point.get("point_index")
        numbers = None if curve is None else figure.data[curve].customdata
        if numbers is not None and index is not None:
            return int(numbers[index])
    return None


def _large(values: Sequence[object]) -> bool:
    """Whether ``values`` are numbers, not all integers, some of them 100 000 or more: shown in
    engineering notation (``15.36E6``)."""
    present = [value for value in values if value is not None]
    numbers = [v for v in present if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return (
        bool(present)
        and len(numbers) == len(present)
        and not all(isinstance(value, int) for value in numbers)
        and max(abs(value) for value in numbers) >= 1e5
    )


def _column(values: Sequence[object], heading: str, said: str | None) -> Any:
    """How a column of ``values`` is shown: a number to four significant digits (in engineering
    notation from 100 000 on), a flag as a tick."""
    present = [value for value in values if value is not None]
    if present and all(isinstance(value, bool) for value in present):
        return st.column_config.CheckboxColumn(heading, help=said, width="small")
    if present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        if all(isinstance(value, int) for value in present):
            return st.column_config.NumberColumn(heading, help=said)
        shown = "engineering" if _large(values) else "%.4g"
        return st.column_config.NumberColumn(heading, help=said, format=shown)
    return st.column_config.Column(heading, help=said)


def table(
    rows: list[tables.Row],
    name: str | None = None,
    *,
    helps: Mapping[str, str] | None = None,
    hidden: Sequence[str] = (),
    config: Mapping[str, Any] | None = None,
    height: int | Literal["auto"] = "auto",
) -> int | None:
    """``rows`` as a table, an empty cell as a dash, each column formatted by its values (or by
    ``config``), the ``hidden`` columns left out; if ``name`` names it, a row can be selected, and
    the index of the one selected is returned."""
    if not rows:
        return None
    columns = [column for column in rows[0] if column not in hidden]
    shown = {
        column: _column([row[column] for row in rows], column, (helps or {}).get(column))
        for column in columns
    } | dict(config or {})
    # engineering notation shows what it is given: four significant digits of each, as elsewhere
    large = [c for c in columns if c not in (config or {}) and _large([r[c] for r in rows])]
    drawn = [
        row | {c: None if row[c] is None else float(f"{row[c]:.4g}") for c in large} for row in rows
    ]
    shared: dict[str, Any] = {
        "hide_index": True,
        "column_order": columns,
        "column_config": shown,
        "placeholder": "—",
        "height": height,
    }
    if name is None:
        st.dataframe(drawn, **shared)
        return None
    event = st.dataframe(
        drawn, key=f"{name}-{picks()}", on_select="rerun", selection_mode="single-row", **shared
    )
    picked = event.selection.rows
    return picked[0] if picked else None


def axes(results: StudyResults, *, color: bool) -> tuple[str | None, str | None, str | None]:
    """The axes of a figure of the trials, and what colors it if ``color``, chosen in one row and
    kept in the URL (the same for every figure of the trials)."""
    choices = list(axis_choices(results))
    default_x, default_y = default_axes(results)
    named: Callable[[str], str] = lambda key: labels.axis(key, results)  # noqa: E731
    columns = st.columns(3 if color else 2)
    with columns[0]:
        x = choice("Horizontal axis", choices, X, default_x, format_func=named)
    with columns[1]:
        y = choice("Vertical axis", choices, Y, default_y, format_func=named)
    shade = None
    if color:
        with columns[2]:
            shades = [BY_OUTCOME, *color_choices(results)]
            shade = choice("Color", shades, COLOR, BY_OUTCOME, format_func=named)
    return x, y, shade
