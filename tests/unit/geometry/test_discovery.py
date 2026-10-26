"""Tests for include walking and parameter discovery."""

from __future__ import annotations

import dataclasses
import os
from collections import Counter
from pathlib import Path

import pytest

from k4opt.geometry import (
    AmbiguousTargetError,
    Catalog,
    GeometrySource,
    IncludeError,
    UnknownTargetError,
    parse_reference,
)
from k4opt.geometry.xmlio import find_by_locator, parse_xml_file
from tests.support.common import write_tree
from tests.support.geometry import MINI_GEO

# ---------------------------------------------------------------------------
# include walking


def test_reachable_files_in_discovery_order(mini_catalog: Catalog) -> None:
    assert mini_catalog.files == (
        "compact/det_v1/det_v1.xml",
        "compact/det_v1/materials.xml",
        "compact/common/shared_constants.xml",
        "compact/common/common_extra.xml",
        "compact/det_v1/display.xml",
        "compact/det_v1/tracker.xml",
        "compact/det_v1/module_down.xml",
        "compact/det_v1/module_up.xml",
    )
    assert all(not f.startswith("compact/det_v2") for f in mini_catalog.files)
    assert "constants.OnlyInV2" not in mini_catalog.targets


def test_duplicate_include_parsed_once(mini_catalog: Catalog) -> None:
    assert Counter(mini_catalog.files)["compact/det_v1/display.xml"] == 1
    display_edges = [e for e in mini_catalog.edges if e.target == "compact/det_v1/display.xml"]
    assert len(display_edges) == 2  # referenced twice, parsed once


def test_external_include_reported_not_resolved(mini_catalog: Catalog) -> None:
    assert len(mini_catalog.externals) == 1
    external = mini_catalog.externals[0]
    assert external.ref == "${DD4hepINSTALL}/DDDetectors/compact/detector_types.xml"
    assert external.file == "compact/det_v1/det_v1.xml"
    assert "environment variable" in external.reason
    # the include attribute itself is still addressable, the referenced file is not
    assert mini_catalog.lookup("lccdd.include[1].ref").value == external.ref


def test_local_resources_are_recorded(mini_catalog: Catalog) -> None:
    assert mini_catalog.resources == ("compact/det_v1/cad/support.stl",)
    assert mini_catalog.lookup("InnerTrackerSupportCAD.shape.ref").value == "cad/support.stl"
    # material references such as ref="Si" are names, not files
    assert mini_catalog.lookup("Silicon.composite.ref").value == "Si"


def test_resource_outside_root_is_reported(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": '<lccdd><shape type="CAD_Assembly" ref="../model.stl"/></lccdd>',
            "model.stl": "solid x\nendsolid x\n",
        },
    )
    catalog = GeometrySource("main.xml", root=tmp_path / "geo").discover()
    assert catalog.resources == ()
    assert [e.reason for e in catalog.externals] == ["file outside the geometry root"]


def test_missing_local_include_fails(tmp_path: Path) -> None:
    write_tree(tmp_path, {"geo/main.xml": '<lccdd><include ref="nope.xml"/></lccdd>'})
    with pytest.raises(IncludeError, match=r"nope\.xml.*not found"):
        GeometrySource("main.xml", root=tmp_path / "geo").discover()


def test_include_cycle_terminates(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/a.xml": (
                '<lccdd><include ref="sub/b.xml"/>'
                '<define><constant name="A" value="1"/></define></lccdd>'
            ),
            "geo/sub/b.xml": (
                '<lccdd><include ref="../a.xml"/>'
                '<define><constant name="B" value="2"/></define></lccdd>'
            ),
        },
    )
    catalog = GeometrySource("a.xml", root=tmp_path / "geo").discover()
    assert catalog.files == ("a.xml", "sub/b.xml")
    assert catalog.lookup("constants.B").value == "2"


def test_include_escaping_root_rejected(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": '<lccdd><include ref="../outside.xml"/></lccdd>',
            "outside.xml": '<lccdd><define><constant name="Secret" value="1"/></define></lccdd>',
        },
    )
    with pytest.raises(IncludeError, match="escapes the geometry root"):
        GeometrySource("main.xml", root=tmp_path / "geo").discover()


def test_symlink_escaping_root_rejected(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": '<lccdd><include ref="link.xml"/></lccdd>',
            "secret.xml": '<lccdd><define><constant name="Secret" value="1"/></define></lccdd>',
        },
    )
    os.symlink(tmp_path / "secret.xml", tmp_path / "geo" / "link.xml")
    with pytest.raises(IncludeError, match="symlink"):
        GeometrySource("main.xml", root=tmp_path / "geo").discover()


def test_symlink_inside_root_is_followed(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": '<lccdd><include ref="alias/b.xml"/></lccdd>',
            "geo/real/b.xml": '<lccdd><define><constant name="B" value="2"/></define></lccdd>',
        },
    )
    os.symlink("real", tmp_path / "geo" / "alias")
    catalog = GeometrySource("main.xml", root=tmp_path / "geo").discover()
    assert catalog.lookup("constants.B").file == "alias/b.xml"


def test_deep_include_chain_is_an_include_error(tmp_path: Path) -> None:
    files = {"geo/main.xml": '<lccdd><include ref="0.xml"/></lccdd>'}
    for i in range(500):
        files[f"geo/{i}.xml"] = f'<lccdd><include ref="{i + 1}.xml"/></lccdd>'
    files["geo/500.xml"] = "<lccdd/>"
    write_tree(tmp_path, files)
    with pytest.raises(IncludeError, match="nested deeper"):
        GeometrySource("main.xml", root=tmp_path / "geo").discover()


def test_cyclic_directory_symlink_is_an_include_cycle(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": '<lccdd><include ref="dir/a.xml"/></lccdd>',
            "geo/dir/a.xml": (
                '<lccdd><include ref="loop/a.xml"/>'
                '<define><constant name="A" value="1"/></define></lccdd>'
            ),
        },
    )
    os.symlink(".", tmp_path / "geo" / "dir" / "loop")
    catalog = GeometrySource("main.xml", root=tmp_path / "geo").discover()
    assert catalog.lookup("constants.A").file == "dir/a.xml"


def test_include_of_directory_or_malformed_xml_rejected(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": '<lccdd><include ref="sub"/></lccdd>',
            "geo/sub/x.xml": "<lccdd/>",
            "geo/bad.xml": '<lccdd><include ref="broken.xml"/></lccdd>',
            "geo/broken.xml": "<lccdd><unclosed></lccdd>",
        },
    )
    with pytest.raises(IncludeError, match="not a regular file"):
        GeometrySource("main.xml", root=tmp_path / "geo").discover()
    with pytest.raises(IncludeError, match="not well-formed"):
        GeometrySource("bad.xml", root=tmp_path / "geo").discover()


def test_include_errors_report_the_include_chain(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": '<lccdd><include ref="sub/a.xml"/></lccdd>',
            "geo/sub/a.xml": '<lccdd>\n<include ref="b.xml"/></lccdd>',
            "geo/sub/b.xml": '<lccdd>\n\n<include ref="missing.xml"/></lccdd>',
        },
    )
    with pytest.raises(
        IncludeError, match=r"sub/b\.xml:3 \(included via main\.xml:1 -> sub/a\.xml:2\)"
    ):
        GeometrySource("main.xml", root=tmp_path / "geo").discover()


def test_absolute_and_url_refs_are_external(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": (
                '<lccdd><include ref="/etc/hostname"/><include ref="file:///etc/hostname"/>'
                '<include ref="http://127.0.0.1:9/x.xml"/><include ref=""/></lccdd>'
            ),
        },
    )
    catalog = GeometrySource("main.xml", root=tmp_path / "geo").discover()
    assert catalog.files == ("main.xml",)
    assert [e.reason for e in catalog.externals] == [
        "absolute path",
        "URL or scheme-qualified reference",
        "URL or scheme-qualified reference",
        "empty reference",
    ]


def test_absolute_resource_refs_are_reported_as_external(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "geo/main.xml": (
                '<lccdd><shape ref="/external/support.stl"/><shape ref="${HERE}/b.stl"/>'
                '<shape ref="Si"/><shape ref="/"/></lccdd>'
            ),
        },
    )
    catalog = GeometrySource("main.xml", root=tmp_path / "geo").discover()
    assert catalog.resources == ()
    assert [(e.ref, e.reason) for e in catalog.externals] == [
        ("/external/support.stl", "absolute path"),
        ("${HERE}/b.stl", "environment variable or placeholder reference"),
        ("/", "absolute path"),
    ]


def test_external_entities_are_not_expanded(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    write_tree(
        tmp_path,
        {
            "geo/main.xml": f"""
            <?xml version="1.0"?>
            <!DOCTYPE lccdd [
              <!ENTITY leak SYSTEM "file://{secret}">
              <!ENTITY net SYSTEM "http://127.0.0.1:9/never">
            ]>
            <lccdd><define><constant name="A" value="1"/></define>
            <comment>&leak;&net;</comment></lccdd>
            """,
        },
    )
    catalog = GeometrySource("main.xml", root=tmp_path / "geo").discover()
    assert catalog.lookup("constants.A").value == "1"
    tree = parse_xml_file(tmp_path / "geo" / "main.xml")
    assert "TOP-SECRET" not in "".join(tree.getroot().itertext())


# ---------------------------------------------------------------------------
# parameter discovery


def test_constants_including_from_included_files(mini_catalog: Catalog) -> None:
    assert mini_catalog.lookup("constants.SolenoidField").value == "2*tesla"
    radius = mini_catalog.lookup("constants.InnerTracker_Barrel_radius_1")
    assert radius.value == "400*mm"
    assert radius.file == "compact/det_v1/tracker.xml"
    assert radius.attribute == "value"
    assert radius.tag == "constant"
    assert (
        mini_catalog.lookup("constants.SharedThickness").file
        == "compact/common/shared_constants.xml"
    )
    assert mini_catalog.lookup("constants.ExtraConstant").file == "compact/common/common_extra.xml"


def test_direct_attributes_at_multiple_depths(mini_catalog: Catalog) -> None:
    assert mini_catalog.lookup("det_v1.title").value == "Mini detector v1"
    assert mini_catalog.lookup("GlobalSolenoid.inner_field").value == "SolenoidField"
    assert mini_catalog.lookup("InnerTrackerBarrel.type").value == "TrackerBarrel_o1_v06"
    assert (
        mini_catalog.lookup("InnerTrackerBarrelModule_01.module_envelope.width").value == "30.1*mm"
    )
    assert mini_catalog.lookup("InnerTrackerBarrel.layer[id=0].z_layout.nz").value == "32"
    assert mini_catalog.lookup("Air.D.value").value == "0.0012"
    assert mini_catalog.lookup("ITVis.r").file == "compact/det_v1/display.xml"


def test_friendly_nested_reference_resolves(mini_catalog: Catalog) -> None:
    parameter = mini_catalog.lookup("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi")
    assert parameter.value == "43*2"
    assert parameter.file == "compact/det_v1/tracker.xml"
    assert parameter.attribute == "nphi"
    assert parameter.locator == "/lccdd/detectors[2]/detector/layer[2]/rphi_layout"
    assert parameter.order_sensitive is False
    assert parameter.ambiguous is False
    assert mini_catalog.lookup("InnerTrackerBarrel.layer[id=0].rphi_layout.nphi").value == "14*2"


def test_repeated_siblings_use_index_fallback(mini_catalog: Catalog) -> None:
    first = mini_catalog.lookup("InnerTrackers.shape.zplane[1].z")
    second = mini_catalog.lookup("InnerTrackers.shape.zplane[2].z")
    assert (first.value, second.value) == ("-10*mm", "+10*mm")
    assert first.order_sensitive and second.order_sensitive
    assert "InnerTrackers.shape.zplane.z" not in mini_catalog.targets


def test_same_file_name_collision_uses_tag_qualified_anchor(mini_catalog: Catalog) -> None:
    # <composite name="InnerTrackerBarrel"/> and <detector name="InnerTrackerBarrel"> share a file
    assert "InnerTrackerBarrel.name" not in mini_catalog.targets
    composite = mini_catalog.lookup("composite[name=InnerTrackerBarrel].name")
    detector = mini_catalog.lookup("detector[name=InnerTrackerBarrel].name")
    assert composite.tag == "composite" and detector.tag == "detector"
    with pytest.raises(UnknownTargetError) as info:
        mini_catalog.lookup("InnerTrackerBarrel.name")
    assert "composite[name=InnerTrackerBarrel].name" in str(info.value)
    assert "detector[name=InnerTrackerBarrel].name" in str(info.value)


def test_repeated_anchor_name_falls_back_to_next_named_ancestor(mini_catalog: Catalog) -> None:
    # two <sensor name="sensor"> elements: neither "sensor" nor "sensor[name=sensor]" is unique,
    # so the enclosing uniquely named <module> becomes the anchor instead of a positional root path
    assert "sensor.thickness" not in mini_catalog.targets
    assert "sensor[name=sensor].thickness" not in mini_catalog.targets
    first = mini_catalog.lookup("InnerTrackerBarrelModule_01.sensor.thickness")
    second = mini_catalog.lookup("InnerTrackerBarrelModule_02.sensor.thickness")
    assert (first.value, second.value) == ("0.050*mm", "0.075*mm")
    assert not first.order_sensitive
    with pytest.raises(UnknownTargetError) as info:
        mini_catalog.lookup("sensor.thickness")
    assert "InnerTrackerBarrelModule_01.sensor.thickness" in str(info.value)
    assert "InnerTrackerBarrelModule_02.sensor.thickness" in str(info.value)


def test_cross_file_collision_requires_file(mini_catalog: Catalog) -> None:
    candidates = [p for p in mini_catalog if p.target == "constants.Tracker_cooling_gap"]
    assert len(candidates) == 2
    assert all(p.ambiguous for p in candidates)
    with pytest.raises(AmbiguousTargetError) as info:
        mini_catalog.lookup("constants.Tracker_cooling_gap")
    assert len(info.value.candidates) == 2
    assert "file:" in str(info.value)
    tracker = mini_catalog.lookup(
        "constants.Tracker_cooling_gap", file="compact/det_v1/tracker.xml"
    )
    shared = mini_catalog.lookup(
        "constants.Tracker_cooling_gap", file="compact/common/shared_constants.xml"
    )
    assert (tracker.value, shared.value) == ("5*mm", "7*mm")


def test_root_anchored_positional_paths_collide_across_files(mini_catalog: Catalog) -> None:
    down = mini_catalog.lookup(
        "lccdd.module_component[3].thickness", file="compact/det_v1/module_down.xml"
    )
    up = mini_catalog.lookup(
        "lccdd.module_component[3].thickness", file="compact/det_v1/module_up.xml"
    )
    assert (down.value, up.value) == ("0.100*mm", "0.150*mm")
    assert down.order_sensitive and down.ambiguous
    with pytest.raises(AmbiguousTargetError):
        mini_catalog.lookup("lccdd.module_component[3].thickness")


def test_unsafe_name_falls_back_to_positional_path(mini_catalog: Catalog) -> None:
    assert not any(t.startswith("constants.Pixel") for t in mini_catalog.targets)
    parameter = mini_catalog.lookup("lccdd.define.constant[4].value")
    assert parameter.value == "1"
    assert parameter.order_sensitive is True
    assert mini_catalog.lookup("lccdd.define.constant[4].name").value == "Pixel array"


def test_constant_non_value_attributes_use_element_alias(mini_catalog: Catalog) -> None:
    assert mini_catalog.lookup("GlobalTrackerReadoutID.type").value == "string"
    assert mini_catalog.lookup("constants.GlobalTrackerReadoutID").value.startswith("system:5")
    with pytest.raises(UnknownTargetError, match=r"constants\.GlobalTrackerReadoutID"):
        mini_catalog.lookup("GlobalTrackerReadoutID.value")


def test_raw_values_are_preserved_without_interpretation(mini_catalog: Catalog) -> None:
    assert mini_catalog.lookup("constants.DetID_NOTUSED").value == " 0"
    assert mini_catalog.lookup("GlobalSolenoid.outer_field").value == "-1.0*tesla"
    assert mini_catalog.lookup("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi").value == "43*2"
    assert mini_catalog.lookup("constants.world_side").value == "6100*mm"


def test_namespaced_attribute_is_reported_unaddressable(mini_catalog: Catalog) -> None:
    assert len(mini_catalog.unaddressable) == 1
    assert mini_catalog.unaddressable[0].attribute.endswith("noNamespaceSchemaLocation")
    assert mini_catalog.unaddressable[0].locator == "/lccdd"


def test_every_target_is_unique_per_file_and_parses(mini_catalog: Catalog) -> None:
    pairs = Counter((p.target, p.file) for p in mini_catalog)
    assert max(pairs.values()) == 1
    counts = Counter(p.target for p in mini_catalog)
    for parameter in mini_catalog:
        assert parse_reference(parameter.target).text == parameter.target
        assert parameter.ambiguous == (counts[parameter.target] > 1)
        assert parameter.order_sensitive == parse_reference(parameter.target).order_sensitive


def test_locators_identify_the_recorded_attribute(mini_catalog: Catalog) -> None:
    trees = {rel: parse_xml_file(MINI_GEO / rel) for rel in mini_catalog.files}
    for parameter in mini_catalog:
        element = find_by_locator(trees[parameter.file], parameter.locator)
        assert element.get(parameter.attribute) == parameter.value
        assert element.tag == parameter.tag
        assert element.sourceline == parameter.line


def test_catalog_is_read_only_and_reproducible(
    mini_source: GeometrySource, mini_catalog: Catalog
) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        mini_catalog[0].value = "changed"  # type: ignore[misc]
    again = mini_source.discover()
    assert list(again) == list(mini_catalog)
    assert again.file_hashes == mini_catalog.file_hashes
    with pytest.raises(TypeError):
        mini_catalog.file_hashes["x"] = "y"  # type: ignore[index]


def test_filter_is_case_insensitive_substring(mini_catalog: Catalog) -> None:
    by_target = mini_catalog.filter("innertracker_barrel_radius")
    assert {p.target for p in by_target} >= {
        "constants.InnerTracker_Barrel_radius_1",
        "InnerTrackerBarrel.layer[id=1].rphi_layout.rc",
    }
    by_value = mini_catalog.filter("43*2")
    assert [p.target for p in by_value] == ["InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"]
    by_file = mini_catalog.filter("COMMON_EXTRA")
    assert {p.file for p in by_file} >= {"compact/common/common_extra.xml"}
    assert all("common_extra" in p.file or "common_extra" in p.value for p in by_file)
    assert mini_catalog.filter("no-such-thing") == ()


def test_lookup_suggests_close_matches(mini_catalog: Catalog) -> None:
    with pytest.raises(UnknownTargetError, match=r"did you mean: constants\.SolenoidField"):
        mini_catalog.lookup("constants.SolenoidFeld")


def test_unreadable_resource_is_a_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geo = write_tree(
        tmp_path / "geo", {"main.xml": '<lccdd><shape name="S" ref="cad.stl"/></lccdd>'}
    )
    (geo / "cad.stl").write_text("solid x\n")
    real_open = Path.open

    def deny_stl(self: Path, *args, **kwargs):
        # chmod cannot make a file unreadable for root, which is how CI runs
        if self.name == "cad.stl":
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_stl)
    with pytest.raises(IncludeError, match=r"cannot read .*cad\.stl"):
        GeometrySource("main.xml", root=geo).discover()
