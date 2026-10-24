"""Tests for safe, reproducible materialization of modified geometry snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from k4opt.geometry import (
    MANIFEST_NAME,
    GeometrySource,
    MaterializationError,
    Override,
    SourceError,
)
from k4opt.geometry import materialize as materialize_module
from k4opt.geometry.xmlio import parse_xml_file
from tests.support.common import tree_digest, write_tree
from tests.support.geometry import MINI_COMPACT, MINI_GEO

OVERRIDES = [
    Override("constants.InnerTracker_Barrel_radius_1", "405*mm"),
    Override("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi", "90"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_leftovers(parent: Path) -> bool:
    return not any(p.name.endswith(".k4opt-tmp") for p in parent.iterdir())


def test_materialize_success(mini_source: GeometrySource, tmp_path: Path) -> None:
    before = tree_digest(MINI_GEO)
    catalog = mini_source.discover()
    output = tmp_path / "generated" / "mini-test"
    result = mini_source.materialize(OVERRIDES, output, catalog)

    assert tree_digest(MINI_GEO) == before, "source tree was modified"
    assert result.output_root == output
    assert result.compact_path == output / MINI_COMPACT
    assert result.compact_path.is_file()
    assert result.manifest_path == output / MANIFEST_NAME
    assert result.source_root == mini_source.root
    assert result.output_path("compact/det_v1/tracker.xml") == output / "compact/det_v1/tracker.xml"
    assert [c.target for c in result.changes] == [o.target for o in OVERRIDES]
    assert _no_leftovers(output.parent)

    # exactly the reachable files and resources are copied, plus the manifest
    copied = set(tree_digest(output))
    assert copied == {*catalog.files, *catalog.resources, MANIFEST_NAME}
    assert "compact/det_v1/README.txt" not in copied
    assert "compact/det_v2/det_v2.xml" not in copied

    # exactly one file changed, with hashes recorded correctly
    assert [f.path for f in result.files] == ["compact/det_v1/tracker.xml"]
    changed = result.files[0]
    assert changed.source_sha256 == _sha256(MINI_GEO / changed.path)
    assert changed.output_sha256 == _sha256(output / changed.path)
    assert changed.source_sha256 != changed.output_sha256

    # every other copied file is byte-identical to the source
    for rel in copied - {changed.path, MANIFEST_NAME}:
        assert _sha256(output / rel) == before[rel], rel
    assert (output / "compact/det_v1/cad/support.stl").read_text().startswith("solid support")


def test_only_intended_attributes_change(mini_source: GeometrySource, tmp_path: Path) -> None:
    output = tmp_path / "out"
    mini_source.materialize(OVERRIDES, output)
    rel = "compact/det_v1/tracker.xml"
    original = parse_xml_file(MINI_GEO / rel)
    edited = parse_xml_file(output / rel)
    before = list(original.iter())
    after = list(edited.iter())
    assert len(before) == len(after)
    diffs = []
    for old, new in zip(before, after, strict=True):
        assert old.tag == new.tag
        assert old.text == new.text and old.tail == new.tail
        if isinstance(old.tag, str):
            diffs.extend(
                (original.getpath(old), key, old.get(key), new.get(key))
                for key in set(old.attrib) | set(new.attrib)
                if old.get(key) != new.get(key)
            )
    assert sorted(diffs) == [
        ("/lccdd/define/constant[2]", "value", "400*mm", "405*mm"),
        ("/lccdd/detectors[2]/detector/layer[2]/rphi_layout", "nphi", "43*2", "90"),
    ]
    text = (output / rel).read_text(encoding="utf-8")
    assert 'nphi="90"' in text and 'nphi="14*2"' in text
    assert "<!-- the composite shares its name with the detector below -->" in text
    assert "<id>${GlobalTrackerReadoutID}</id>" in text
    assert "<comment>Tracking detectors</comment>" in text


def test_round_trip_through_discovery(mini_source: GeometrySource, tmp_path: Path) -> None:
    result = mini_source.materialize(OVERRIDES, tmp_path / "out")
    snapshot = GeometrySource(MINI_COMPACT, root=result.output_root).discover()
    assert snapshot.lookup("constants.InnerTracker_Barrel_radius_1").value == "405*mm"
    assert snapshot.lookup("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi").value == "90"
    assert snapshot.lookup("InnerTrackerBarrel.layer[id=0].rphi_layout.nphi").value == "14*2"
    assert snapshot.lookup("constants.DetID_NOTUSED").value == " 0"
    assert snapshot.lookup("constants.SolenoidField").value == "2*tesla"
    assert len(snapshot) == len(mini_source.discover())
    assert snapshot.resources == ("compact/det_v1/cad/support.stl",)


def test_manifest_contents(mini_source: GeometrySource, tmp_path: Path) -> None:
    result = mini_source.materialize(OVERRIDES, tmp_path / "out")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest == result.manifest
    assert manifest["schema"] == "k4opt.manifest/1"
    assert manifest["source"]["root"] == str(mini_source.root)
    assert manifest["source"]["compact"] == MINI_COMPACT
    assert "git" in manifest["source"]
    assert manifest["output"]["compact"] == MINI_COMPACT
    assert manifest["reachable_files"] == list(mini_source.discover().files)
    assert manifest["resources"] == ["compact/det_v1/cad/support.stl"]
    assert manifest["external_includes"][0]["ref"].startswith("${DD4hepINSTALL}")
    assert [c["target"] for c in manifest["changes"]] == [o.target for o in OVERRIDES]
    assert manifest["changes"][0] == {
        "target": "constants.InnerTracker_Barrel_radius_1",
        "file": "compact/det_v1/tracker.xml",
        "attribute": "value",
        "locator": "/lccdd/define/constant[2]",
        "line": 4,
        "old_value": "400*mm",
        "new_value": "405*mm",
    }
    assert manifest["files"] == [
        {
            "path": "compact/det_v1/tracker.xml",
            "source_sha256": _sha256(MINI_GEO / "compact/det_v1/tracker.xml"),
            "output_sha256": _sha256(result.output_root / "compact/det_v1/tracker.xml"),
        }
    ]


def test_xml_declaration_preserved_or_absent(mini_source: GeometrySource, tmp_path: Path) -> None:
    result = mini_source.materialize(
        [
            Override("constants.SolenoidField", "3*tesla"),
            Override("constants.Tracker_cooling_gap", "6*mm", file="compact/det_v1/tracker.xml"),
        ],
        tmp_path / "out",
    )
    root_text = (result.output_root / "compact/det_v1/det_v1.xml").read_bytes()
    first_line = root_text.split(b"\n", 1)[0]
    assert first_line.replace(b"'", b'"') == b'<?xml version="1.0" encoding="UTF-8"?>'
    tracker_text = (result.output_root / "compact/det_v1/tracker.xml").read_bytes()
    assert not tracker_text.startswith(b"<?xml")
    assert b'value="3*tesla"' in root_text
    assert b"<!-- resolved by DD4hep at runtime" in root_text


def test_explicit_standalone_declaration_is_kept(tmp_path: Path) -> None:
    geo = tmp_path / "geo"
    geo.mkdir()
    (geo / "main.xml").write_bytes(
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b'<lccdd><define><constant name="A" value="1"/></define></lccdd>\n'
    )
    result = GeometrySource("main.xml", root=geo).materialize(
        [Override("constants.A", "2")], tmp_path / "out"
    )
    first_line = (result.output_root / "main.xml").read_bytes().split(b"\n", 1)[0]
    assert b"standalone='yes'" in first_line or b'standalone="yes"' in first_line


def test_zero_overrides_produces_plain_snapshot(
    mini_source: GeometrySource, tmp_path: Path
) -> None:
    result = mini_source.materialize([], tmp_path / "nominal")
    assert result.changes == () and result.files == ()
    assert result.compact_path.is_file()
    assert json.loads(result.manifest_path.read_text())["changes"] == []


def test_existing_output_rejected(mini_source: GeometrySource, tmp_path: Path) -> None:
    before = tree_digest(MINI_GEO)
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(MaterializationError, match="already exists"):
        mini_source.materialize(OVERRIDES, existing)
    as_file = tmp_path / "file.txt"
    as_file.write_text("x")
    with pytest.raises(MaterializationError, match="already exists"):
        mini_source.materialize(OVERRIDES, as_file)
    assert tree_digest(MINI_GEO) == before
    assert _no_leftovers(tmp_path)


def test_output_inside_source_rejected(mini_source: GeometrySource, tmp_path: Path) -> None:
    before = tree_digest(MINI_GEO)
    with pytest.raises(MaterializationError, match="inside the source tree"):
        mini_source.materialize(OVERRIDES, MINI_GEO / "generated")
    with pytest.raises(MaterializationError, match="inside the source tree"):
        mini_source.materialize(OVERRIDES, MINI_GEO / "compact" / "new")
    link = tmp_path / "link-into-source"
    os.symlink(MINI_GEO / "compact", link)
    with pytest.raises(MaterializationError, match="inside the source tree"):
        mini_source.materialize(OVERRIDES, link / "via-symlink")
    assert tree_digest(MINI_GEO) == before
    assert not (MINI_GEO / "generated").exists()


def test_failure_leaves_no_trace(
    mini_source: GeometrySource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = tree_digest(MINI_GEO)
    output = tmp_path / "out"

    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated failure during edit")

    monkeypatch.setattr(materialize_module, "_apply_edits", explode)
    with pytest.raises(RuntimeError, match="simulated failure"):
        mini_source.materialize(OVERRIDES, output)
    assert not output.exists()
    assert _no_leftovers(tmp_path)
    assert tree_digest(MINI_GEO) == before


def test_source_change_after_discovery_detected(tmp_path: Path) -> None:
    geo = tmp_path / "geo"
    shutil.copytree(MINI_GEO, geo)
    source = GeometrySource(MINI_COMPACT, root=geo)
    catalog = source.discover()
    tracker = geo / "compact/det_v1/tracker.xml"
    tracker.write_text(tracker.read_text().replace('value="400*mm"', 'value="401*mm"'))
    with pytest.raises(MaterializationError, match="changed since discovery"):
        source.materialize(OVERRIDES, tmp_path / "out", catalog)
    assert not (tmp_path / "out").exists()


def test_override_breaking_an_include_is_rejected(
    mini_source: GeometrySource, tmp_path: Path
) -> None:
    # overriding an include ref is technically addressable, but the snapshot must stay consistent
    before = tree_digest(MINI_GEO)
    output = tmp_path / "out"
    with pytest.raises(MaterializationError, match="not self-consistent"):
        mini_source.materialize([Override("lccdd.include[3].ref", "does_not_exist.xml")], output)
    assert not output.exists()
    assert _no_leftovers(tmp_path)
    assert tree_digest(MINI_GEO) == before


def test_catalog_from_other_source_rejected(mini_source: GeometrySource, tmp_path: Path) -> None:
    other = GeometrySource("compact/det_v2/det_v2.xml", root=MINI_GEO).discover()
    with pytest.raises(SourceError, match="different geometry source"):
        mini_source.materialize(OVERRIDES, tmp_path / "out", other)


def test_symlinked_files_are_copied_as_regular_files(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": '<lccdd><include ref="alias/b.xml"/></lccdd>',
            "geo/real/b.xml": '<lccdd><define><constant name="B" value="2"/></define></lccdd>',
        },
    )
    os.symlink("real", tmp_path / "geo" / "alias")
    result = GeometrySource("main.xml", root=tmp_path / "geo").materialize(
        [Override("constants.B", "3")], tmp_path / "out"
    )
    copied = result.output_root / "alias" / "b.xml"
    assert copied.is_file() and not copied.is_symlink()
    assert not (result.output_root / "alias").is_symlink()
    assert b'value="3"' in copied.read_bytes()


def test_doctype_and_entity_references_survive_an_edit(tmp_path: Path) -> None:
    geo = tmp_path / "geo"
    geo.mkdir()
    (geo / "main.xml").write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE lccdd [<!ENTITY note "kept as written">]>\n'
        b'<lccdd><define><constant name="A" value="1"/></define><comment>&note;</comment></lccdd>\n'
    )
    result = GeometrySource("main.xml", root=geo).materialize(
        [Override("constants.A", "2")], tmp_path / "out"
    )
    text = (result.output_root / "main.xml").read_text(encoding="utf-8")
    assert "<!DOCTYPE lccdd [" in text and '<!ENTITY note "kept as written">' in text
    assert "&note;" in text and 'value="2"' in text
