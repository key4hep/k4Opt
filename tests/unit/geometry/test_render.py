"""Tests for the human-readable renderer (width handling, color, long targets)."""

from __future__ import annotations

import io
import re

from k4opt.geometry import Catalog
from k4opt.geometry.render import ELLIPSIS, display_value, fit, render_parameters

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _render(catalog: Catalog, parameters, **kwargs) -> str:
    out = io.StringIO()
    render_parameters(parameters, catalog, out, **kwargs)
    return out.getvalue()


def test_fit_and_display_value() -> None:
    assert fit("abcdef", None) == "abcdef"
    assert fit("abcdef", 6) == "abcdef"
    assert fit("abcdef", 4) == "abc" + ELLIPSIS
    assert display_value("400*mm") == "400*mm"
    assert display_value(" 0") == '" 0"'
    assert display_value("") == '""'


def test_narrow_width_shortens_values_but_never_targets(mini_catalog: Catalog) -> None:
    parameters = mini_catalog.filter("InnerTrackerBarrel")
    wide = _render(mini_catalog, parameters)
    narrow = _render(mini_catalog, parameters, width=60)
    for parameter in parameters:
        assert parameter.target in narrow, "targets must stay copy-pastable"
    assert ELLIPSIS in narrow and ELLIPSIS not in wide
    body = [line for line in narrow.splitlines() if line.startswith("  ")]
    assert body and all(len(line) <= 60 or ELLIPSIS not in line for line in body)


def test_long_target_puts_value_on_continuation_line(mini_catalog: Catalog) -> None:
    parameters = mini_catalog.filter("InnerTrackerBarrelModule_01.module_envelope")
    long_target = max(parameters, key=lambda p: len(p.target))
    text = _render(mini_catalog, parameters, width=40)
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if long_target.target in line)
    assert long_target.value not in lines[index]
    assert long_target.value in lines[index + 1]


def test_colour_is_optional_and_does_not_change_the_text(mini_catalog: Catalog) -> None:
    parameters = tuple(mini_catalog)
    plain = _render(mini_catalog, parameters, color=False)
    colored = _render(mini_catalog, parameters, color=True)
    assert "\x1b[" not in plain and "\x1b[" in colored
    assert ANSI.sub("", colored) == plain


def test_tip_only_for_long_unfiltered_listings(mini_catalog: Catalog) -> None:
    everything = _render(mini_catalog, tuple(mini_catalog))
    assert "Tip: narrow the list with --filter" in everything
    filtered = _render(mini_catalog, mini_catalog.filter("nphi"), filter_text="nphi")
    assert "Tip:" not in filtered


def test_legend_is_on_top_lists_only_present_flags_and_is_yellow(mini_catalog: Catalog) -> None:
    only_o = mini_catalog.filter("zplane")
    assert all(p.order_sensitive and not p.ambiguous for p in only_o)
    text = _render(mini_catalog, only_o)
    assert text.index("Flags in the first column:") < text.index("tracker.xml")
    assert "  O  order-sensitive" in text and "  A  ambiguous" not in text

    no_flags = mini_catalog.filter("constants.SolenoidField")
    assert all(not p.ambiguous and not p.order_sensitive for p in no_flags)
    assert "Flags in the first column" not in _render(mini_catalog, no_flags)

    colored = _render(mini_catalog, only_o, color=True)
    assert "\x1b[33mFlags in the first column:\x1b[0m" in colored
    assert "\x1b[33m  O  order-sensitive" in colored
