"""Constants and helpers for the geometry tests."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "geometry"
MINI_GEO = FIXTURES / "mini_geo"
MINI_COMPACT = "compact/det_v1/det_v1.xml"
CLD_COMPACT = "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml"


def write_config(
    path: Path,
    geometry: str,
    root: Path | None = None,
    overrides: list[dict[str, str]] | None = None,
    extra: str = "",
) -> Path:
    """Write a k4opt YAML at ``path``; ``root`` (if given) is relative to the YAML directory."""
    lines = [f"geometry: {geometry}"]
    if root is not None:
        lines.append(f"root: {os.path.relpath(root, path.parent)}")
    if extra:
        lines.append(textwrap.dedent(extra).strip("\n"))
    if overrides is not None:
        lines.append("overrides:")
        for item in overrides:
            first = True
            for key, value in item.items():
                prefix = "  - " if first else "    "
                first = False
                lines.append(f"{prefix}{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
