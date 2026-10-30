"""Human-readable rendering of catalogs and resolved changes.

Parameters are grouped under the file they live in; targets are never shortened
(they must stay copy-pastable), long values are shortened only when a terminal
``width`` is given. Color is applied only when ``color`` is true.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TextIO

from k4opt.geometry.discovery import Catalog, Parameter
from k4opt.geometry.overrides import ResolvedOverride

ELLIPSIS = "…"
INDENT = 2
GUTTER = 2  # width of the flag column ("A", "O" or both)
MIN_VALUE_WIDTH = 16
MAX_EXTERNALS_SHOWN = 5
TIP_THRESHOLD = 40

LEGEND = {
    "A": "ambiguous: also defined in another file; select one with --set FILE:TARGET=VALUE",
    "O": "order-sensitive: positional [index] in the target, breaks if siblings are reordered",
}


class Style:
    """Tiny ANSI helper; every method returns the text unchanged when disabled."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled and text else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


def fit(text: str, width: int | None) -> str:
    """Shorten ``text`` to ``width`` characters with an ellipsis; ``None`` keeps it whole."""
    if width is None or len(text) <= width:
        return text
    return text[: max(width - 1, 0)] + ELLIPSIS


def display_value(value: str) -> str:
    """Raw value for display; empty or space-padded values are quoted so they stay visible."""
    return f'"{value}"' if value == "" or value != value.strip() else value


def _flags(parameter: Parameter) -> str:
    return ("A" if parameter.ambiguous else "") + ("O" if parameter.order_sensitive else "")


def _group_by_file(
    parameters: Sequence[Parameter], file_order: Sequence[str]
) -> list[tuple[str, list[Parameter]]]:
    groups: dict[str, list[Parameter]] = {}
    for parameter in parameters:
        groups.setdefault(parameter.file, []).append(parameter)
    ordered = [f for f in file_order if f in groups] + [f for f in groups if f not in file_order]
    return [(f, groups[f]) for f in ordered]


def render_parameters(
    parameters: Sequence[Parameter],
    catalog: Catalog,
    out: TextIO,
    *,
    filter_text: str | None = None,
    width: int | None = None,
    color: bool = False,
) -> None:
    """Print ``parameters`` (a subset of ``catalog``) grouped by file with a header and legend."""
    style = Style(color)
    shown = f"{len(parameters)} of {len(catalog)} parameters in {len(catalog.files)} files"
    if filter_text:
        shown += f", matching {filter_text!r}"
    print(f"{style.dim('geometry')}  {style.bold(catalog.compact)}", file=out)
    print(f"{style.dim('root    ')}  {catalog.root}", file=out)
    print(f"{style.dim('showing ')}  {shown}", file=out)

    if not parameters:
        print(file=out)
        print("No parameters match. Try a shorter --filter text.", file=out)
        return

    present = {flag for p in parameters for flag in _flags(p)}
    legend = [(flag, text) for flag, text in LEGEND.items() if flag in present]
    if legend:
        print(file=out)
        print(style.yellow("Flags in the first column:"), file=out)
        for flag, text in legend:
            print(style.yellow(f"  {flag}  {text}"), file=out)

    longest = max(len(p.target) for p in parameters)
    target_width = longest if width is None else min(longest, max(24, int(width * 0.5)))
    value_start = INDENT + GUTTER + 1 + target_width + 2
    value_width = None if width is None else max(width - value_start, MIN_VALUE_WIDTH)

    for file, group in _group_by_file(parameters, catalog.files):
        print(file=out)
        print(style.bold(file) + "  " + style.dim(f"({len(group)})"), file=out)
        for p in group:
            # `X.name` rows only echo an element's own name: keep them, but quiet
            quiet = p.attribute == "name"
            flags = style.yellow(_flags(p).ljust(GUTTER))
            value = fit(display_value(p.value), value_width)
            value = style.dim(value) if quiet else style.cyan(value)
            head = " " * INDENT + flags + " "
            if len(p.target) <= target_width:
                target = p.target.ljust(target_width)
                print(f"{head}{style.dim(target) if quiet else target}  {value}", file=out)
            else:  # never truncate a target: put the value on a continuation line
                print(f"{head}{style.dim(p.target) if quiet else p.target}", file=out)
                print(" " * value_start + value, file=out)

    if catalog.externals:
        print(file=out)
        print(style.dim("Not followed (external, left untouched):"), file=out)
        for external in catalog.externals[:MAX_EXTERNALS_SHOWN]:
            print(style.dim(f"  {external.file}:{external.line}  {external.ref}"), file=out)
        hidden = len(catalog.externals) - MAX_EXTERNALS_SHOWN
        if hidden > 0:
            print(style.dim(f"  … and {hidden} more"), file=out)
    if not filter_text and len(parameters) > TIP_THRESHOLD:
        print(file=out)
        print(
            style.dim("Tip: narrow the list with --filter TEXT (matches target, value and file)."),
            file=out,
        )


def render_changes(
    changes: Sequence[ResolvedOverride], out: TextIO, *, color: bool = False
) -> None:
    """Print resolved overrides grouped by file as ``old -> new``."""
    style = Style(color)
    if not changes:
        print("No overrides given; nothing to change.", file=out)
        return
    seen: list[str] = []
    for change in changes:
        if change.file not in seen:
            seen.append(change.file)
    for index, file in enumerate(seen):
        if index:
            print(file=out)
        print(style.bold(file), file=out)
        for change in (c for c in changes if c.file == file):
            line = f"{style.red(display_value(change.old_value))}  ->  "
            line += style.green(display_value(change.new_value))
            if not change.changed:
                line += style.dim("  (unchanged)")
            print(f"  {change.target}", file=out)
            print(f"      {line}", file=out)
