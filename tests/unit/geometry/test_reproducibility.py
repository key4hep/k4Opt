"""Pickling (executor-agnostic use) and reproducibility identity (fingerprint, stack)."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path

import pytest

from k4opt.geometry import (
    AmbiguousTargetError,
    Catalog,
    GeometrySource,
    Override,
    TargetError,
    UnknownTargetError,
)
from tests.support.common import write_tree
from tests.support.geometry import MINI_COMPACT, MINI_GEO

RADIUS = "constants.InnerTracker_Barrel_radius_1"


def _roundtrip(obj: object) -> object:
    return pickle.loads(pickle.dumps(obj))


def test_core_objects_survive_pickling(mini_source: GeometrySource, tmp_path: Path) -> None:
    catalog = mini_source.discover()
    clone = _roundtrip(catalog)
    assert isinstance(clone, Catalog)
    assert list(clone) == list(catalog)
    assert dict(clone.file_hashes) == dict(catalog.file_hashes)
    assert clone.lookup(RADIUS) == catalog.lookup(RADIUS)
    with pytest.raises(TypeError):
        clone.file_hashes["x"] = "y"  # type: ignore[index]

    assert _roundtrip(mini_source) == mini_source
    override = Override(RADIUS, "405*mm", file="compact/det_v1/tracker.xml")
    assert _roundtrip(override) == override

    result = mini_source.materialize([override], tmp_path / "out", catalog)
    clone_result = _roundtrip(result)
    assert clone_result.manifest == result.manifest  # type: ignore[attr-defined]
    assert clone_result.compact_path == result.compact_path  # type: ignore[attr-defined]


def test_errors_keep_their_details_across_processes(mini_catalog: Catalog) -> None:
    with pytest.raises(AmbiguousTargetError) as ambiguous:
        mini_catalog.lookup("constants.Tracker_cooling_gap")
    clone = _roundtrip(ambiguous.value)
    assert isinstance(clone, AmbiguousTargetError)
    assert len(clone.candidates) == 2

    combined = TargetError("2 failed", [UnknownTargetError("a"), UnknownTargetError("b")])
    clone = _roundtrip(combined)
    assert isinstance(clone, TargetError)
    assert [str(e) for e in clone.errors] == ["a", "b"]


def test_fingerprint_is_deterministic_and_matches_the_manifest(
    mini_source: GeometrySource, tmp_path: Path
) -> None:
    overrides = [Override(RADIUS, "405*mm")]
    first = mini_source.materialize(overrides, tmp_path / "a")
    second = mini_source.materialize(overrides, tmp_path / "elsewhere" / "b")
    predicted = mini_source.fingerprint(overrides)
    assert first.fingerprint == second.fingerprint == predicted
    assert len(predicted) == 64
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["fingerprint"] == predicted


def test_fingerprint_tracks_effective_edits_only(mini_source: GeometrySource) -> None:
    baseline = mini_source.fingerprint([])
    restating = mini_source.fingerprint([Override(RADIUS, "400*mm")])
    changed = mini_source.fingerprint([Override(RADIUS, "405*mm")])
    other_value = mini_source.fingerprint([Override(RADIUS, "406*mm")])
    other_target = mini_source.fingerprint([Override("constants.SolenoidField", "3*tesla")])
    assert baseline == restating, "restating the current value does not change the geometry"
    assert len({baseline, changed, other_value, other_target}) == 4
    reordered = [
        Override("constants.SolenoidField", "3*tesla"),
        Override(RADIUS, "405*mm"),
    ]
    assert mini_source.fingerprint(reordered) == mini_source.fingerprint(reordered[::-1])


def test_fingerprint_depends_on_source_content(tmp_path: Path) -> None:
    geo = tmp_path / "geo"
    shutil.copytree(MINI_GEO, geo)
    source = GeometrySource(MINI_COMPACT, root=geo)
    before = source.fingerprint([])
    stl = geo / "compact/det_v1/cad/support.stl"
    stl.write_text(stl.read_text() + "\n")
    assert source.fingerprint([]) != before, "a changed resource changes the identity"
    xml = geo / "compact/det_v1/tracker.xml"
    after_stl = source.fingerprint([])
    xml.write_text(xml.read_text().replace("Tracking detectors", "Tracking detectors v2"))
    assert source.fingerprint([]) != after_stl, "a changed XML file changes the identity"


def test_manifest_records_the_software_stack(
    mini_source: GeometrySource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    monkeypatch.setenv("KEY4HEP_STACK", "/cvmfs/example/key4hep-stack/setup.sh")
    result = mini_source.materialize([], tmp_path / "with-stack")
    assert result.manifest["environment"] == {
        "KEY4HEP_STACK": "/cvmfs/example/key4hep-stack/setup.sh",
        "K4GEO": str(MINI_GEO),
    }
    monkeypatch.delenv("K4GEO")
    monkeypatch.delenv("KEY4HEP_STACK")
    bare = mini_source.materialize([], tmp_path / "without-stack")
    assert bare.manifest["environment"] == {}
    assert bare.fingerprint == result.fingerprint, "the stack does not alter the geometry identity"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_files_say_what_the_fingerprint_leaves_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aux = write_tree(
        tmp_path / "aux",
        {
            "component.xml": '<lccdd><include ref="parts/a.xml"/><shape ref="model.stl"/></lccdd>',
            "parts/a.xml": '<lccdd><include ref="${CAD}/b.xml"/></lccdd>',
            "model.stl": "solid support",
            "broken.xml": '<lccdd><include ref="valid.xml"/><include ref="missing.xml"/></lccdd>',
            "valid.xml": "<lccdd/>",
            "unparsable.xml": "<lccdd>",
        },
    )
    cad = write_tree(tmp_path / "cad", {"b.xml": "<lccdd/>"})
    stl = aux / "model.stl"
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": f"""\
                <lccdd>
                  <include ref="${{AUX}}/component.xml"/>
                  <include ref="${{AUX}}/broken.xml"/>
                  <include ref="${{AUX}}/unparsable.xml"/>
                  <include ref="${{UNSET}}/x.xml"/>
                  <shape ref="{stl}"/>
                </lccdd>
            """,
        },
    )
    monkeypatch.setenv("AUX", str(aux))
    monkeypatch.setenv("CAD", str(cad))
    monkeypatch.delenv("UNSET", raising=False)
    source = GeometrySource("main.xml", root=geo)
    found = source.external_files()
    assert found == {
        "${AUX}/component.xml": {
            "path": str(aux / "component.xml"),
            "sha256": _sha256(aux / "component.xml"),
            "files": {"parts/a.xml": _sha256(aux / "parts/a.xml"), "model.stl": _sha256(stl)},
        },
        "${AUX}/broken.xml": {
            "path": str(aux / "broken.xml"),
            "sha256": _sha256(aux / "broken.xml"),
            # the include that cannot be followed is listed; the one that can is still read
            "files": {"valid.xml": _sha256(aux / "valid.xml")},
            "unresolved": {"broken.xml:1": "missing.xml"},
        },
        "${AUX}/unparsable.xml": {
            "path": str(aux / "unparsable.xml"),
            "sha256": _sha256(aux / "unparsable.xml"),
            "files": None,  # it cannot be parsed itself
        },
        "${UNSET}/x.xml": {"path": None, "sha256": None},
        str(stl): {"path": str(stl), "sha256": _sha256(stl)},
        # reached through an external include, an entry of its own
        "${CAD}/b.xml": {"path": str(cad / "b.xml"), "sha256": _sha256(cad / "b.xml"), "files": {}},
    }

    # the same files elsewhere read the same content, which an edit then changes
    monkeypatch.setenv("AUX", str(shutil.copytree(aux, tmp_path / "moved")))
    moved = source.external_files()["${AUX}/component.xml"]
    assert moved["path"] != found["${AUX}/component.xml"]["path"]
    assert moved["files"] == found["${AUX}/component.xml"]["files"]
    (tmp_path / "moved" / "parts" / "a.xml").write_text("<lccdd/>")
    assert source.external_files()["${AUX}/component.xml"]["files"] != moved["files"]
    # so does an edit of a file an include reaches beside one it cannot follow
    broken = source.external_files()["${AUX}/broken.xml"]
    (tmp_path / "moved" / "valid.xml").write_text("<lccdd><!-- other --></lccdd>")
    assert source.external_files()["${AUX}/broken.xml"]["files"] != broken["files"]


def test_a_reference_written_alike_has_an_entry_for_each_file_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aux = write_tree(
        tmp_path / "aux", {"c.xml": '<lccdd><include ref="d.xml"/></lccdd>', "d.xml": "<lccdd/>"}
    )
    geo = write_tree(
        tmp_path / "geo",
        {
            "main.xml": '<lccdd><include ref="a/x.xml"/><include ref="b/y.xml"/></lccdd>',
            # a relative ${REL} names the file beside each of them
            "a/x.xml": '<lccdd><shape ref="${REL}/m.stl"/><shape ref="${AUX}/c.xml"/></lccdd>',
            "b/y.xml": '<lccdd><shape ref="${REL}/m.stl"/><include ref="${AUX}/c.xml"/></lccdd>',
            "a/cad/m.stl": "solid a",
            "b/cad/m.stl": "solid b",
        },
    )
    monkeypatch.setenv("REL", "cad")
    monkeypatch.setenv("AUX", str(aux))
    found = GeometrySource("main.xml", root=geo).external_files()
    assert found == {
        "${REL}/m.stl": {"path": str(geo / "a/cad/m.stl"), "sha256": _sha256(geo / "a/cad/m.stl")},
        "${AUX}/c.xml": {
            "path": str(aux / "c.xml"),
            "sha256": _sha256(aux / "c.xml"),
            # named as a resource first, then included: what it includes is read as well
            "files": {"d.xml": _sha256(aux / "d.xml")},
        },
        "${REL}/m.stl (in b/y.xml)": {
            "path": str(geo / "b/cad/m.stl"),
            "sha256": _sha256(geo / "b/cad/m.stl"),
        },
    }
