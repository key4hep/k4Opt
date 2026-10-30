"""Tests for safe, reproducible materialization of modified geometry snapshots."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from k4opt.geometry import (
    MANIFEST_NAME,
    GeometrySource,
    MaterializationError,
    Override,
    SourceError,
    includes,
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
    with pytest.raises(MaterializationError, match="not found"):
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


def _copied_geometry(tmp_path: Path) -> tuple[Path, GeometrySource]:
    geo = tmp_path / "geo"
    shutil.copytree(MINI_GEO, geo)
    return geo, GeometrySource(MINI_COMPACT, root=geo)


def test_change_to_untouched_file_or_resource_detected(tmp_path: Path) -> None:
    geo, source = _copied_geometry(tmp_path)
    catalog = source.discover()
    for rel in ("compact/det_v1/materials.xml", "compact/det_v1/cad/support.stl"):
        path = geo / rel
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(MaterializationError, match="changed since discovery"):
            source.materialize([], tmp_path / "out", catalog)
        assert not (tmp_path / "out").exists()
        assert _no_leftovers(tmp_path)
        path.write_bytes(original)


def test_noop_override_does_not_rewrite_xml(mini_source: GeometrySource, tmp_path: Path) -> None:
    catalog = mini_source.discover()
    same = Override("constants.InnerTracker_Barrel_radius_1", "400*mm")
    assert mini_source.resolve([same], catalog)[0].changed is False
    result = mini_source.materialize([same], tmp_path / "out", catalog)
    assert result.files == ()
    assert (
        tree_digest(result.output_root).items()
        >= {k: v for k, v in tree_digest(MINI_GEO).items() if k in catalog.files}.items()
    )
    assert result.fingerprint == mini_source.fingerprint([], catalog)


def test_redirected_resource_reference_is_copied_and_fingerprinted(tmp_path: Path) -> None:
    geo, source = _copied_geometry(tmp_path)
    (geo / "compact/det_v1/cad/other.stl").write_text("solid other\nendsolid other\n")
    catalog = source.discover()
    target = next(
        p.target for p in catalog if p.attribute == "ref" and p.value == "cad/support.stl"
    )
    override = Override(target, "cad/other.stl")
    result = source.materialize([override], tmp_path / "out", catalog)
    assert (
        (result.output_root / "compact/det_v1/cad/other.stl").read_text().startswith("solid other")
    )
    assert result.fingerprint == source.fingerprint([override], catalog)
    assert set(result.manifest["resources"]) == {
        "compact/det_v1/cad/support.stl",
        "compact/det_v1/cad/other.stl",
    }

    # the fingerprint covers the new resource's content
    (geo / "compact/det_v1/cad/other.stl").write_text("solid other2\nendsolid other2\n")
    assert source.fingerprint([override], catalog) != result.fingerprint


def test_output_appearing_before_publication_is_not_replaced(
    mini_source: GeometrySource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    real_verify = materialize_module._verify_snapshot

    def create_output_then_verify(*args, **kwargs):
        output.mkdir()  # a concurrent worker wins the race
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(materialize_module, "_verify_snapshot", create_output_then_verify)
    with pytest.raises(MaterializationError, match="already exists"):
        mini_source.materialize(OVERRIDES, output)
    assert list(output.iterdir()) == []
    assert _no_leftovers(tmp_path)


def test_redirect_to_missing_resource_rejected(tmp_path: Path) -> None:
    _, source = _copied_geometry(tmp_path)
    catalog = source.discover()
    target = next(
        p.target for p in catalog if p.attribute == "ref" and p.value == "cad/support.stl"
    )
    for value in ("cad/missing.stl", "../outside/missing.stl"):
        with pytest.raises(MaterializationError, match="does not exist"):
            source.materialize([Override(target, value)], tmp_path / "out", catalog)
        assert not (tmp_path / "out").exists()
    # external references and symbolic names are still accepted
    for value in ("${DD4hepINSTALL}/cad/support.stl", "SomeName"):
        source.materialize([Override(target, value)], tmp_path / "out", catalog)
        shutil.rmtree(tmp_path / "out")


def test_publish_never_replaces_an_existing_empty_directory(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "f").write_text("x")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(MaterializationError, match="already exists"):
        materialize_module._publish(snapshot, out)
    assert list(out.iterdir()) == [] and (snapshot / "f").is_file()
    out.rmdir()
    materialize_module._publish(snapshot, out)
    assert (out / "f").read_text() == "x"


def test_manifest_name_collision_rejected(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "d.xml": '<lccdd><shape ref="k4opt-manifest.json"/><constant name="X" value="1"/></lccdd>'  # noqa: E501
        },
    )
    (geo / "k4opt-manifest.json").write_text("{}")
    source = GeometrySource("d.xml", root=geo)
    with pytest.raises(MaterializationError, match="name of the generated manifest"):
        source.materialize([Override("constants.X", "2")], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_read_only_source_files_can_be_edited(tmp_path: Path) -> None:
    geo, source = _copied_geometry(tmp_path)
    for path in geo.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    result = source.materialize(OVERRIDES, tmp_path / "out")
    assert "405*mm" in (result.output_root / "compact/det_v1/tracker.xml").read_text()
    shutil.rmtree(tmp_path / "out")  # the snapshot is a normal, deletable tree


def test_redirect_to_resource_outside_root_rejected(tmp_path: Path) -> None:
    geo, source = _copied_geometry(tmp_path)
    (tmp_path / "outside.stl").write_text("solid outside\nendsolid outside\n")
    catalog = source.discover()
    target = next(
        p.target for p in catalog if p.attribute == "ref" and p.value == "cad/support.stl"
    )
    value = os.path.relpath(tmp_path / "outside.stl", geo / "compact/det_v1")
    with pytest.raises(MaterializationError, match="outside the geometry root"):
        source.materialize([Override(target, value)], tmp_path / "out", catalog)
    assert not (tmp_path / "out").exists()


def test_manifest_externals_describe_the_generated_geometry(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {"d.xml": '<lccdd><include ref="${LIB_A}/old.xml"/><constant name="X" value="1"/></lccdd>'},
    )
    source = GeometrySource("d.xml", root=geo)
    target = next(p.target for p in source.discover() if p.attribute == "ref")
    result = source.materialize([Override(target, "${LIB_B}/new.xml")], tmp_path / "out")
    assert [e["ref"] for e in result.manifest["external_includes"]] == ["${LIB_B}/new.xml"]


def test_publish_fallback_without_renameat2_still_refuses_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoRenameat2:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr(materialize_module.ctypes, "CDLL", NoRenameat2)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(MaterializationError, match="already exists"):
        materialize_module._publish(snapshot, out)
    out.rmdir()
    materialize_module._publish(snapshot, out)
    assert out.is_dir() and not snapshot.exists()


def test_external_resource_redirected_to_missing_local_file_rejected(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {"d.xml": '<lccdd><shape name="Support" ref="${CAD_DIR}/support.stl"/></lccdd>'},
    )
    source = GeometrySource("d.xml", root=geo)
    with pytest.raises(MaterializationError, match="does not exist"):
        source.materialize([Override("Support.ref", "cad/missing.stl")], tmp_path / "out")
    assert not (tmp_path / "out").exists()
    # a plain name is not assumed to be a file
    source.materialize([Override("Support.ref", "SomeName")], tmp_path / "out")


def test_manifest_keeps_unchanged_external_with_the_same_value(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "d.xml": """
                <lccdd>
                  <shape name="A" ref="${CAD}/outside.stl"/>
                  <shape name="B" ref="${CAD}/outside.stl"/>
                </lccdd>
            """
        },
    )
    (tmp_path / "outside.stl").write_text("solid x\nendsolid x\n")
    source = GeometrySource("d.xml", root=geo)
    result = source.materialize([Override("A.ref", "${X}/new.stl")], tmp_path / "out")
    externals = [(e["ref"], e["line"]) for e in result.manifest["external_includes"]]
    # the override's own external value is listed too; B keeps its unchanged outside file
    assert externals == [("${X}/new.stl", 2), ("${CAD}/outside.stl", 3)]


@pytest.mark.parametrize("redirect", [False, True])
def test_relocation_rejects_reachable_resources_outside_the_root(
    tmp_path: Path, redirect: bool
) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="a.xml"/></lccdd>',
            "a.xml": '<lccdd><shape ref="../outside.stl"/></lccdd>',
            "b.xml": '<lccdd><shape ref="../outside.stl"/></lccdd>',
        },
    )
    (tmp_path / "outside.stl").write_text("solid x\nendsolid x\n")
    source = GeometrySource("main.xml", root=geo)
    edits = [Override("lccdd.include.ref", "b.xml")] if redirect else []
    with pytest.raises(MaterializationError, match="cannot keep its meaning after relocation"):
        source.materialize(edits, tmp_path / "elsewhere" / "out")
    assert not (tmp_path / "elsewhere" / "out").exists()

    # A broader root captures the original reference without rewriting it.
    source = GeometrySource("geo/main.xml", root=tmp_path)
    result = source.materialize(edits, tmp_path.parent / f"{tmp_path.name}-snapshot")
    assert (result.output_root / "outside.stl").read_bytes() == (
        tmp_path / "outside.stl"
    ).read_bytes()


def test_a_replaced_outside_reference_in_a_redirected_include_is_accepted(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="a.xml"/></lccdd>',
            "a.xml": '<lccdd><detectors><include ref="k.xml"/></detectors></lccdd>',
            "b.xml": '<lccdd><detectors><include ref="k.xml"/></detectors></lccdd>',
            "k.xml": '<lccdd><shape name="S" ref="../outside.stl"/></lccdd>',
        },
    )
    (tmp_path / "outside.stl").write_text("solid x\nendsolid x\n")
    edits = [Override("lccdd.include.ref", "b.xml"), Override("S.ref", "${CAD}/x.stl")]
    result = GeometrySource("main.xml", root=geo).materialize(edits, tmp_path / "out")
    assert [e["ref"] for e in result.manifest["external_includes"]] == ["${CAD}/x.stl"]


def test_the_alias_left_reachable_decides_whether_a_resource_leaves_the_root(
    tmp_path: Path,
) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="sub/f.xml"/><include ref="g.xml"/></lccdd>',
            "sub/f.xml": '<lccdd><shape name="S" ref="../x.stl"/></lccdd>',
            "a.xml": "<lccdd/>",
        },
    )
    (geo / "x.stl").write_text("inside\n")
    (tmp_path / "x.stl").write_text("outside\n")
    (geo / "g.xml").symlink_to("sub/f.xml")
    source = GeometrySource("main.xml", root=geo)
    # only g.xml remains, and from the root ../x.stl is the file outside it
    with pytest.raises(MaterializationError, match=r"g\.xml: resource '\.\./x\.stl' lies outside"):
        source.materialize([Override("lccdd.include[1].ref", "a.xml")], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_symlinked_file_also_named_as_a_resource_stays_one_file(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="a.xml"/><include ref="b.xml"/>'
            '<shape name="S" ref="b.xml"/></lccdd>',
            "a.xml": '<lccdd><define><constant name="X" value="1"/></define></lccdd>',
        },
    )
    (geo / "b.xml").symlink_to("a.xml")
    out = tmp_path / "out"
    GeometrySource("main.xml", root=geo).materialize([Override("constants.X", "2")], out)
    assert (out / "b.xml").is_symlink()
    assert (out / "b.xml").read_text() == (out / "a.xml").read_text()
    assert 'value="2"' in (out / "a.xml").read_text()


def test_a_resource_edit_resolves_from_an_alias_a_redirect_introduces(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="sub/f.xml"/><include ref="a.xml"/></lccdd>',
            "sub/f.xml": '<lccdd><shape name="S" ref="old.stl"/></lccdd>',
            "sub/old.stl": "old\n",
            "sub/new.stl": "new below sub\n",
            "other/new.stl": "new below other\n",
            "a.xml": "<lccdd/>",
        },
    )
    (geo / "other" / "f.xml").symlink_to("../sub/f.xml")
    edits = [Override("lccdd.include[2].ref", "other/f.xml"), Override("S.ref", "new.stl")]
    out = tmp_path / "out"
    GeometrySource("main.xml", root=geo).materialize(edits, out)
    # other/f.xml is sub/f.xml, and from other/ its new reference names other/new.stl
    assert (out / "other" / "new.stl").read_text() == "new below other\n"
    assert (out / "sub" / "new.stl").read_text() == "new below sub\n"


def test_a_resource_the_edited_geometry_needs_must_be_in_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><shape name="S" ref="a.stl"/></lccdd>',
            "a.stl": "a\n",
            "b.stl": "b\n",
        },
    )
    additions = materialize_module.new_files

    def forgetting(*args: Any) -> Any:
        return dataclasses.replace(additions(*args), resources={})

    monkeypatch.setattr(materialize_module, "new_files", forgetting)
    with pytest.raises(MaterializationError, match=r"references b\.stl, which was not copied"):
        GeometrySource("main.xml", root=geo).materialize(
            [Override("S.ref", "b.stl")], tmp_path / "o"
        )
    assert not (tmp_path / "o").exists()


def test_an_edit_after_parsing_cannot_move_an_override_to_another_element(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {"main.xml": '<lccdd><box name="A" value="1"/><box name="B" value="1"/></lccdd>'},
    )
    read = includes.read_xml_file

    def read_then_reorder(path: Path, *, label: str | None = None) -> bytes:
        data = read(path, label=label)
        path.write_text('<lccdd><box name="B" value="1"/><box name="A" value="1"/></lccdd>')
        return data

    source = GeometrySource("main.xml", root=geo)
    monkeypatch.setattr(includes, "read_xml_file", read_then_reorder)
    catalog = source.discover()
    monkeypatch.undo()
    with pytest.raises(MaterializationError, match="changed since discovery"):
        source.materialize([Override("A.value", "9")], tmp_path / "out", catalog=catalog)
    assert not (tmp_path / "out").exists()


def test_unwritable_output_location_is_a_controlled_error(
    mini_source: GeometrySource, tmp_path: Path
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(MaterializationError, match="materialization failed"):
        mini_source.materialize(OVERRIDES, blocker / "out")


def test_manifest_externals_skip_files_made_unreachable(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="a.xml"/><include ref="b.xml"/></lccdd>',
            "a.xml": '<lccdd><shape name="A" ref="../outside.stl"/></lccdd>',
            "b.xml": '<lccdd><constant name="X" value="1"/></lccdd>',
        },
    )
    (tmp_path / "outside.stl").write_text("solid x\nendsolid x\n")
    source = GeometrySource("main.xml", root=geo)
    catalog = source.discover()
    assert [e.ref for e in catalog.externals] == ["../outside.stl"]
    first_include = next(p.target for p in catalog if p.attribute == "ref" and p.value == "a.xml")
    result = source.materialize([Override(first_include, "b.xml")], tmp_path / "out", catalog)
    assert result.manifest["external_includes"] == []


def test_fingerprint_uses_the_verified_resource_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geo, source = _copied_geometry(tmp_path)
    other = geo / "compact/det_v1/cad/other.stl"
    other.write_text("ORIGINAL\n")
    catalog = source.discover()
    target = next(
        p.target for p in catalog if p.attribute == "ref" and p.value == "cad/support.stl"
    )
    override = Override(target, "cad/other.stl")
    expected = source.fingerprint([override], catalog)

    real_verify = materialize_module._verify_snapshot

    def mutate_then_verify(*args, **kwargs):
        other.write_text("MUTATED\n")  # the source changes after the copy was verified
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(materialize_module, "_verify_snapshot", mutate_then_verify)
    result = source.materialize([override], tmp_path / "out", catalog)
    assert (result.output_root / "compact/det_v1/cad/other.stl").read_text() == "ORIGINAL\n"
    assert result.fingerprint == expected


def test_symlinked_include_alias_is_kept_in_snapshot(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="part.xml"/><include ref="part_link.xml"/></lccdd>',
            "part.xml": '<lccdd><constant name="X" value="1"/></lccdd>',
        },
    )
    (geo / "part_link.xml").symlink_to("part.xml")
    source = GeometrySource("main.xml", root=geo)
    catalog = source.discover()
    assert dict(catalog.file_aliases) == {"part_link.xml": "part.xml"}

    result = source.materialize([Override("constants.X", "2")], tmp_path / "out", catalog)
    link = result.output_root / "part_link.xml"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link))
    assert 'value="2"' in link.read_text()
    assert result.fingerprint == source.fingerprint([Override("constants.X", "2")], catalog)


def test_redirected_include_brings_new_xml_closure(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="part.xml"/></lccdd>',
            "part.xml": '<lccdd><constant name="X" value="1"/></lccdd>',
            "alt.xml": '<lccdd><include ref="sub/deep.xml"/><constant name="Y" value="1"/></lccdd>',
            "sub/deep.xml": '<lccdd><constant name="Z" value="1"/></lccdd>',
        },
    )
    source = GeometrySource("main.xml", root=geo)
    catalog = source.discover()
    target = next(p.target for p in catalog if p.attribute == "ref")
    override = Override(target, "alt.xml")
    result = source.materialize([override], tmp_path / "out", catalog)

    out = result.output_root
    assert (out / "alt.xml").is_file() and (out / "sub/deep.xml").is_file()
    assert result.fingerprint == source.fingerprint([override], catalog)
    assert source.fingerprint([override], catalog) != source.fingerprint([], catalog)

    (geo / "sub/deep.xml").write_text('<lccdd><constant name="Z" value="2"/></lccdd>')
    assert source.fingerprint([override], catalog) != result.fingerprint


def test_redirected_include_to_missing_file_rejected(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="part.xml"/></lccdd>',
            "part.xml": '<lccdd><constant name="X" value="1"/></lccdd>',
        },
    )
    source = GeometrySource("main.xml", root=geo)
    target = next(p.target for p in source.discover() if p.attribute == "ref")
    with pytest.raises(MaterializationError, match="not found"):
        source.materialize([Override(target, "nope.xml")], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_publish_falls_back_when_renameat2_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Unsupported:
        def __init__(self, *_args, **_kwargs) -> None:
            self.renameat2 = self

        def __call__(self, *_args) -> int:
            materialize_module.ctypes.set_errno(materialize_module.errno.EOPNOTSUPP)
            return -1

    monkeypatch.setattr(materialize_module.ctypes, "CDLL", Unsupported)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    materialize_module._publish(snapshot, tmp_path / "out")
    assert (tmp_path / "out").is_dir() and not snapshot.exists()


def test_redirected_include_to_symlink_of_reachable_file_is_linked(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="part.xml"/><include ref="other.xml"/></lccdd>',
            "part.xml": '<lccdd><constant name="X" value="1"/></lccdd>',
            "other.xml": '<lccdd><constant name="Y" value="1"/></lccdd>',
        },
    )
    (geo / "part_link.xml").symlink_to("part.xml")
    source = GeometrySource("main.xml", root=geo)
    catalog = source.discover()
    target = next(p.target for p in catalog if p.attribute == "ref" and p.value == "other.xml")
    result = source.materialize(
        [Override(target, "part_link.xml"), Override("constants.X", "2")], tmp_path / "out", catalog
    )
    link = result.output_root / "part_link.xml"
    assert link.is_symlink() and 'value="2"' in link.read_text()


def _alias_geometry(root: Path, target: str) -> GeometrySource:
    geo = write_tree(
        root,
        {
            "main.xml": (
                '<lccdd><include ref="a.xml"/><include ref="b.xml"/>'
                '<include ref="link.xml"/></lccdd>'
            ),
            "a.xml": '<lccdd><constant name="A" value="1"/></lccdd>',
            "b.xml": '<lccdd><constant name="B" value="2"/></lccdd>',
        },
    )
    (geo / "link.xml").symlink_to(target)
    return GeometrySource("main.xml", root=geo)


def test_fingerprint_depends_on_where_an_include_alias_points(tmp_path: Path) -> None:
    a = _alias_geometry(tmp_path / "ga", "a.xml")
    b = _alias_geometry(tmp_path / "gb", "b.xml")
    assert a.fingerprint([]) != b.fingerprint([])
    assert a.materialize([], tmp_path / "oa").fingerprint == a.fingerprint([])


def test_alias_introduced_by_override_is_in_the_fingerprint(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="part.xml"/></lccdd>',
            "part.xml": '<lccdd><constant name="X" value="1"/></lccdd>',
            "other.xml": '<lccdd><constant name="Y" value="1"/></lccdd>',
        },
    )
    (geo / "l1.xml").symlink_to("other.xml")
    (geo / "l2.xml").symlink_to("part.xml")
    source = GeometrySource("main.xml", root=geo)
    catalog = source.discover()
    target = next(p.target for p in catalog if p.attribute == "ref")
    assert source.fingerprint([Override(target, "l1.xml")], catalog) != source.fingerprint(
        [Override(target, "l2.xml")], catalog
    )


def test_external_reference_of_a_redirected_include_is_listed_once(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="a.xml"/><include ref="b.xml"/></lccdd>',
            "a.xml": '<lccdd><constant name="A" value="1"/></lccdd>',
            "b.xml": '<lccdd><shape name="s" ref="${CAD}/external.stl"/></lccdd>',
        },
    )
    source = GeometrySource("main.xml", root=geo)
    catalog = source.discover()
    target = next(p.target for p in catalog if p.attribute == "ref" and "a.xml" in p.value)
    result = source.materialize([Override(target, "b.xml")], tmp_path / "out", catalog)
    refs = [(e["file"], e["ref"]) for e in result.manifest["external_includes"]]
    assert len(refs) == len(set(refs))


def test_include_through_aliases_in_two_directories_is_followed(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": (
                '<lccdd><include ref="a/shared.xml"/><include ref="b/shared.xml"/></lccdd>'
            ),
            "a/shared.xml": '<lccdd><include ref="local.xml"/></lccdd>',
            "a/local.xml": '<lccdd><constant name="LA" value="1"/></lccdd>',
            "b/local.xml": '<lccdd><constant name="LB" value="2"/></lccdd>',
        },
    )
    (geo / "b" / "shared.xml").symlink_to("../a/shared.xml")
    source = GeometrySource("main.xml", root=geo)
    catalog = source.discover()
    assert "b/local.xml" in catalog.files
    result = source.materialize([Override("constants.LB", "3")], tmp_path / "out", catalog)
    assert 'value="3"' in (result.output_root / "b/local.xml").read_text()


def _shared_alias_geometry(tmp_path: Path, files: dict[str, str]) -> GeometrySource:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": (
                '<lccdd><include ref="a/shared.xml"/><include ref="b/shared.xml"/></lccdd>'
            ),
            **files,
        },
    )
    (geo / "b" / "shared.xml").symlink_to("../a/shared.xml")
    return GeometrySource("main.xml", root=geo)


def test_resource_override_reaches_every_alias_directory(tmp_path: Path) -> None:
    source = _shared_alias_geometry(
        tmp_path,
        {
            "a/shared.xml": '<lccdd><shape ref="local.stl"/></lccdd>',
            "a/local.stl": "a",
            "a/new.stl": "na",
            "b/local.stl": "b",
            "b/new.stl": "nb",
        },
    )
    catalog = source.discover()
    target = next(p.target for p in catalog if p.attribute == "ref" and p.value == "local.stl")
    result = source.materialize([Override(target, "new.stl")], tmp_path / "out", catalog)
    assert (result.output_root / "a/new.stl").read_text() == "na"
    assert (result.output_root / "b/new.stl").read_text() == "nb"
    assert {"a/new.stl", "b/new.stl"} <= set(result.manifest["resources"])


def test_include_override_reaches_every_alias_directory(tmp_path: Path) -> None:
    source = _shared_alias_geometry(
        tmp_path,
        {
            "a/shared.xml": '<lccdd><include ref="local.xml"/></lccdd>',
            "a/local.xml": "<lccdd/>",
            "a/new.xml": "<lccdd/>",
            "b/local.xml": "<lccdd/>",
            "b/new.xml": "<lccdd/>",
        },
    )
    catalog = source.discover()
    target = next(p.target for p in catalog if p.attribute == "ref" and "local" in p.value)
    result = source.materialize([Override(target, "new.xml")], tmp_path / "out", catalog)
    assert (result.output_root / "b/new.xml").is_file()


def test_changed_symlink_target_after_discovery_is_rejected(tmp_path: Path) -> None:
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="a.xml"/><include ref="link.xml"/></lccdd>',
            "a.xml": '<lccdd><constant name="X" value="1"/></lccdd>',
            "b.xml": '<lccdd><constant name="X" value="1"/></lccdd>',
        },
    )
    (geo / "link.xml").symlink_to("a.xml")
    source = GeometrySource("main.xml", root=geo)
    catalog = source.discover()
    (geo / "link.xml").unlink()
    (geo / "link.xml").symlink_to("b.xml")
    with pytest.raises(MaterializationError, match="symlink changed since discovery"):
        source.materialize([], tmp_path / "out", catalog)
    assert not (tmp_path / "out").exists()
