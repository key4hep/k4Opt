"""Tests for the command-line interface and API/CLI consistency."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from k4opt.cli import main
from k4opt.geometry import GeometrySource
from tests.support.common import tree_digest
from tests.support.geometry import MINI_COMPACT, MINI_GEO

GEOMETRY = (MINI_COMPACT, "--root", str(MINI_GEO))
INSPECT = ("geometry", "inspect", *GEOMETRY)
APPLY = (
    "geometry",
    "apply",
    *GEOMETRY,
    "--set",
    "constants.InnerTracker_Barrel_radius_1=405*mm",
    "--set",
    "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi=90",
)


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


ROW = re.compile(r"^  (?P<flags>.{2}) (?P<target>\S+)\s{2,}(?P<value>.*)$")


def parse_rows(out: str) -> dict[str, list[tuple[str, str, str]]]:
    """Parse text output into ``{file: [(flags, target, value), ...]}``."""
    groups: dict[str, list[tuple[str, str, str]]] = {}
    current = None
    for line in out.splitlines():
        if line and not line.startswith(" ") and line.endswith(")") and " (" in line:
            current = line.rsplit("  (", 1)[0]
            groups[current] = []
        elif current is not None and (m := ROW.match(line)):
            groups[current].append((m["flags"].strip(), m["target"], m["value"]))
    return groups


def test_inspect_text_is_grouped_by_file() -> None:
    code, out, err = run(*INSPECT)
    assert code == 0 and err == ""
    lines = out.splitlines()
    root = GeometrySource(MINI_COMPACT, root=MINI_GEO).root
    assert lines[0].split() == ["geometry", MINI_COMPACT]
    assert lines[1].split() == ["root", str(root)]
    assert re.fullmatch(r"showing\s+\d+ of \d+ parameters in 8 files", lines[2])

    groups = parse_rows(out)
    tracker = {
        target: (flags, value) for flags, target, value in groups["compact/det_v1/tracker.xml"]
    }
    assert tracker["InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"] == ("", "43*2")
    assert tracker["InnerTrackers.shape.zplane[2].z"] == ("O", "+10*mm")
    assert tracker["constants.Tracker_cooling_gap"] == ("A", "5*mm")
    shared = {t: (f, v) for f, t, v in groups["compact/common/shared_constants.xml"]}
    assert shared["constants.Tracker_cooling_gap"] == ("A", "7*mm")
    listed = sum(len(rows) for rows in groups.values())
    assert listed == len(GeometrySource(MINI_COMPACT, root=MINI_GEO).discover())

    legend_at = out.index("Flags in the first column:")
    assert legend_at < out.index("compact/det_v1/tracker.xml  ("), "legend comes before the listing"
    assert "  A  ambiguous: also defined in another file" in out
    assert "  O  order-sensitive: positional [index]" in out
    assert "Not followed (external, left untouched):" in out
    assert "${DD4hepINSTALL}/DDDetectors/compact/detector_types.xml" in out
    assert "\x1b[" not in out, "no color when output is not a terminal"
    assert all(len(r) <= 3 for rows in groups.values() for r in rows)


def test_geometry_relative_to_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K4GEO", raising=False)
    code, _, err = run("geometry", "inspect", MINI_COMPACT)
    assert code == 1 and "K4GEO is not set" in err
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    code, out, _ = run("geometry", "inspect", MINI_COMPACT, "--filter", "SolenoidField")
    assert code == 0 and "constants.SolenoidField" in out


def test_root_is_relative_to_the_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(MINI_GEO.parent)
    code, out, _ = run(
        "geometry", "inspect", MINI_COMPACT, "--root", MINI_GEO.name, "--filter", "SolenoidField"
    )
    assert code == 0 and "constants.SolenoidField" in out


def test_source_errors_exit_1(tmp_path: Path) -> None:
    code, out, err = run("geometry", "inspect", MINI_COMPACT, "--root", str(tmp_path / "nowhere"))
    assert code == 1 and out == "" and "not an existing directory" in err
    code, _, err = run("geometry", "inspect", "compact/missing.xml", "--root", str(MINI_GEO))
    assert code == 1 and "not found" in err


def test_inspect_filter_is_case_insensitive() -> None:
    code, out, err = run(*INSPECT, "--filter", "innertracker_barrel_radius_1")
    assert code == 0 and err == ""
    rows = [row for group in parse_rows(out).values() for row in group]
    assert rows and all("InnerTracker_Barrel_radius_1" in f"{t} {v}" for _, t, v in rows)
    assert f"showing   {len(rows)} of " in out
    assert "matching 'innertracker_barrel_radius_1'" in out
    _, out, _ = run(*INSPECT, "--filter", "43*2")
    assert [t for g in parse_rows(out).values() for _, t, _ in g] == [
        "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"
    ]
    code, out, _ = run(*INSPECT, "--filter", "no-such-thing")
    assert code == 0 and "No parameters match" in out and parse_rows(out) == {}


def test_inspect_json_matches_api() -> None:
    code, out, _ = run(*INSPECT, "--format", "json")
    assert code == 0
    payload = json.loads(out)
    catalog = GeometrySource(MINI_COMPACT, root=MINI_GEO).discover()
    assert payload["geometry"] == {"root": str(catalog.root), "compact": MINI_COMPACT}
    assert payload["schema"] == "k4opt.geometry.inspect/1"
    assert payload["files"] == list(catalog.files)
    assert payload["resources"] == list(catalog.resources)
    assert payload["parameters"] == [p.to_dict() for p in catalog]
    assert payload["external_includes"][0]["reason"].startswith("environment")
    _, filtered, _ = run(*INSPECT, "--filter", "InnerTracker", "--format", "json")
    assert json.loads(filtered)["parameters"] == [
        p.to_dict() for p in catalog.filter("InnerTracker")
    ]


def test_apply_dry_run_writes_nothing(tmp_path: Path) -> None:
    before = tree_digest(MINI_GEO)
    output = tmp_path / "generated" / "mini"
    code, out, err = run(*APPLY, "--output", str(output), "--dry-run")
    assert code == 0 and err == ""
    assert "compact/det_v1/tracker.xml\n" in out
    assert "  constants.InnerTracker_Barrel_radius_1\n      400*mm  ->  405*mm\n" in out
    assert "  InnerTrackerBarrel.layer[id=1].rphi_layout.nphi\n      43*2  ->  90\n" in out
    assert f"would write:   {output}" in out
    assert "dry run: 2 change(s) resolved; nothing written" in out
    assert not output.exists() and not output.parent.exists()
    assert tree_digest(MINI_GEO) == before

    code, out, _ = run(*APPLY, "--output", str(output), "--dry-run", "--format", "json")
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert [c["new_value"] for c in payload["changes"]] == ["405*mm", "90"]
    assert not output.exists()


def test_apply_success_then_existing_output_rejected(tmp_path: Path) -> None:
    output = tmp_path / "generated" / "mini"
    code, out, err = run(*APPLY, "--output", str(output))
    assert code == 0
    assert err == "" and "400*mm  ->  405*mm" in out
    assert f"wrote snapshot: {output}" in out
    assert f"compact:        {output / MINI_COMPACT}" in out
    assert re.search(r"fingerprint:    [0-9a-f]{64}", out)
    assert (output / MINI_COMPACT).is_file()
    assert (output / "k4opt-manifest.json").is_file()

    code, _, err = run(*APPLY, "--output", str(output))
    assert code == 1
    assert "already exists" in err


def test_apply_json_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    code, out, _ = run(*APPLY, "--output", str(output), "--format", "json")
    assert code == 0
    payload = json.loads(out)
    assert payload["dry_run"] is False
    assert payload["schema"] == "k4opt.geometry.apply/1"
    manifest = json.loads((output / "k4opt-manifest.json").read_text())
    assert payload["fingerprint"] == manifest["fingerprint"]
    assert payload["output_root"] == str(output)
    assert payload["compact_path"] == str(output / MINI_COMPACT)
    assert payload["files"][0]["path"] == "compact/det_v1/tracker.xml"
    assert manifest["changes"] == payload["changes"]


def test_apply_without_set_copies_the_geometry_unchanged(tmp_path: Path) -> None:
    output = tmp_path / "out"
    code, out, _ = run("geometry", "apply", *GEOMETRY, "--output", str(output))
    assert code == 0 and "No overrides given; nothing to change." in out
    assert json.loads((output / "k4opt-manifest.json").read_text())["changes"] == []


def test_apply_resolution_errors_exit_1_and_write_nothing(tmp_path: Path) -> None:
    output = tmp_path / "out"
    sets = ("--set", "constants.Nope=1", "--set", "constants.Tracker_cooling_gap=1")
    for extra in ([], ["--dry-run"]):
        code, out, err = run("geometry", "apply", *GEOMETRY, *sets, "--output", str(output), *extra)
        assert code == 1
        assert out == ""
        assert "k4opt: error:" in err
        assert "unknown target 'constants.Nope'" in err
        assert "constants.Tracker_cooling_gap" in err and "--set FILE:TARGET=VALUE" in err
        assert not output.exists()


def test_set_file_prefix_selects_an_ambiguous_target(tmp_path: Path) -> None:
    code, out, err = run(
        "geometry",
        "apply",
        *GEOMETRY,
        "--set",
        "compact/det_v1/tracker.xml:constants.Tracker_cooling_gap=6*mm",
        "--output",
        str(tmp_path / "out"),
        "--dry-run",
    )
    assert code == 0 and err == ""
    assert "compact/det_v1/tracker.xml\n  constants.Tracker_cooling_gap\n" in out
    assert "5*mm  ->  6*mm" in out


def test_set_value_is_everything_after_the_target(tmp_path: Path) -> None:
    code, out, _ = run(
        "geometry",
        "apply",
        *GEOMETRY,
        "--set",
        "constants.SolenoidField=a=b:c",
        "--output",
        str(tmp_path / "out"),
        "--dry-run",
        "--format",
        "json",
    )
    assert code == 0
    change = json.loads(out)["changes"][0]
    assert (change["target"], change["new_value"], change["file"]) == (
        "constants.SolenoidField",
        "a=b:c",
        "compact/det_v1/det_v1.xml",
    )


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        ("constants.SolenoidField", "expected [FILE:]TARGET=VALUE"),
        ("layer[id=1]", "expected [FILE:]TARGET=VALUE"),
        ("=5", "--set '=5':"),
        ("constants.Solenoid Field=5", "--set 'constants.Solenoid Field=5':"),
    ],
)
def test_set_syntax_errors_exit_1(tmp_path: Path, assignment: str, message: str) -> None:
    output = tmp_path / "out"
    code, out, err = run(
        "geometry", "apply", *GEOMETRY, "--set", assignment, "--output", str(output)
    )
    assert code == 1 and out == "" and message in err
    assert not output.exists()


def test_geometry_usage_errors_exit_2() -> None:
    with pytest.raises(SystemExit) as info:
        run("geometry", "inspect")  # the geometry is required
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        run("geometry", "apply", MINI_COMPACT)  # --output is required
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        run("geometry")  # an action is required
    assert info.value.code == 2
