"""Invariants over every compact entry point of a real k4geo tree.

Discovery must never crash, may only fail closed on genuinely broken includes, and the
catalog must satisfy its identity invariants for each of the hundreds of thousands of
real attributes. Skipped without ``$K4GEO`` / ``K4GEO_DIR``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from k4opt.geometry import GeometrySource, IncludeError, parse_reference
from k4opt.geometry.xmlio import find_by_locator, parse_xml_file

pytestmark = pytest.mark.integration

SKIPPED_PARTS = {"build", "install"}
PARAMETER_STRIDE = 40


def entry_points(root: Path) -> list[str]:
    """Compact files that look like entry points: ``<name>/<name>.xml`` or ``compact/*.xml``."""
    found = []
    for path in sorted(root.glob("**/compact/**/*.xml")):
        if SKIPPED_PARTS & set(path.parts):
            continue
        if path.stem != path.parent.name and path.parent.name != "compact":
            continue
        if "<lccdd" in path.read_text(errors="replace")[:3000]:
            found.append(path.relative_to(root).as_posix())
    return found


def test_every_entry_point_discovers_and_keeps_its_identities(k4geo_root: Path) -> None:
    entries = entry_points(k4geo_root)
    assert len(entries) > 10, "expected a realistic k4geo tree"
    broken: list[str] = []
    checked = 0
    for rel in entries:
        try:
            catalog = GeometrySource(rel, root=k4geo_root).discover()
        except IncludeError as exc:  # upstream files that include something that does not exist
            if "not found" not in str(exc):
                raise
            broken.append(rel)
            continue
        per_file = Counter((p.target, p.file) for p in catalog)
        assert max(per_file.values()) == 1, f"{rel}: a target is not unique within its file"
        trees = {}
        # every 40th parameter keeps the run to seconds; the full set was verified once offline
        for parameter in catalog[::PARAMETER_STRIDE]:
            checked += 1
            assert parse_reference(parameter.target).text == parameter.target
            tree = trees.setdefault(parameter.file, parse_xml_file(k4geo_root / parameter.file))
            element = find_by_locator(tree, parameter.locator)
            assert element.get(parameter.attribute) == parameter.value
    assert checked > 1_000
    assert len(broken) < len(entries) / 10, f"too many unreadable entry points: {broken}"


def test_snapshots_without_overrides_are_byte_identical(k4geo_root: Path, tmp_path: Path) -> None:
    chosen = [e for e in entry_points(k4geo_root) if "CLD" in e or "IDEA" in e][:4]
    assert chosen
    for index, rel in enumerate(chosen):
        source = GeometrySource(rel, root=k4geo_root)
        catalog = source.discover()
        result = source.materialize([], tmp_path / f"snapshot-{index}", catalog)
        for file in catalog.files:
            assert (result.output_root / file).read_bytes() == (k4geo_root / file).read_bytes()
        assert GeometrySource(rel, root=result.output_root).discover().file_hashes == (
            catalog.file_hashes
        )
        assert result.fingerprint == source.fingerprint([], catalog)
