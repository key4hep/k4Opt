"""Tests for the command-line interface, the YAML configuration and API/CLI consistency."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from k4opt.cli import main
from k4opt.geometry import GeometrySource
from tests.support.common import tree_digest
from tests.support.geometry import MINI_COMPACT, MINI_GEO, write_config

OVERRIDES = [
    {"target": "constants.InnerTracker_Barrel_radius_1", "value": '"405*mm"'},
    {"target": "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi", "value": '"90"'},
]


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path / "geometry.yaml", MINI_COMPACT, root=MINI_GEO, overrides=OVERRIDES
    )


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


def test_inspect_text_is_grouped_by_file(config: Path) -> None:
    code, out, err = run("geometry", "inspect", str(config))
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


def test_inspect_without_overrides_section(tmp_path: Path) -> None:
    config = write_config(tmp_path / "geometry.yaml", MINI_COMPACT, root=MINI_GEO)
    code, out, _ = run("geometry", "inspect", str(config))
    assert code == 0 and "constants.SolenoidField" in out


def test_geometry_relative_to_k4geo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_config(tmp_path / "geometry.yaml", MINI_COMPACT)
    monkeypatch.delenv("K4GEO", raising=False)
    code, _, err = run("geometry", "inspect", str(config))
    assert code == 1 and "K4GEO is not set" in err
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    code, out, _ = run("geometry", "inspect", str(config), "--filter", "SolenoidField")
    assert code == 0 and "constants.SolenoidField" in out


def test_inspect_filter_is_case_insensitive(config: Path) -> None:
    code, out, err = run(
        "geometry", "inspect", str(config), "--filter", "innertracker_barrel_radius_1"
    )
    assert code == 0 and err == ""
    rows = [row for group in parse_rows(out).values() for row in group]
    assert rows and all("InnerTracker_Barrel_radius_1" in f"{t} {v}" for _, t, v in rows)
    assert f"showing   {len(rows)} of " in out
    assert "matching 'innertracker_barrel_radius_1'" in out
    _, out, _ = run("geometry", "inspect", str(config), "--filter", "43*2")
    assert [t for g in parse_rows(out).values() for _, t, _ in g] == [
        "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"
    ]
    code, out, _ = run("geometry", "inspect", str(config), "--filter", "no-such-thing")
    assert code == 0 and "No parameters match" in out and parse_rows(out) == {}


def test_inspect_json_matches_api(config: Path) -> None:
    code, out, _ = run("geometry", "inspect", str(config), "--format", "json")
    assert code == 0
    payload = json.loads(out)
    catalog = GeometrySource(MINI_COMPACT, root=MINI_GEO).discover()
    assert payload["geometry"] == {"root": str(catalog.root), "compact": MINI_COMPACT}
    assert payload["schema"] == "k4opt.geometry.inspect/1"
    assert payload["files"] == list(catalog.files)
    assert payload["resources"] == list(catalog.resources)
    assert payload["parameters"] == [p.to_dict() for p in catalog]
    assert payload["external_includes"][0]["reason"].startswith("environment")
    _, filtered, _ = run(
        "geometry", "inspect", str(config), "--filter", "InnerTracker", "--format", "json"
    )
    assert json.loads(filtered)["parameters"] == [
        p.to_dict() for p in catalog.filter("InnerTracker")
    ]


def test_apply_dry_run_writes_nothing(config: Path, tmp_path: Path) -> None:
    before = tree_digest(MINI_GEO)
    output = tmp_path / "generated" / "mini"
    code, out, err = run("geometry", "apply", str(config), "--output", str(output), "--dry-run")
    assert code == 0
    assert code == 0 and err == ""
    assert "compact/det_v1/tracker.xml\n" in out
    assert "  constants.InnerTracker_Barrel_radius_1\n      400*mm  ->  405*mm\n" in out
    assert "  InnerTrackerBarrel.layer[id=1].rphi_layout.nphi\n      43*2  ->  90\n" in out
    assert f"would write:   {output}" in out
    assert "dry run: 2 change(s) resolved; nothing written" in out
    assert not output.exists() and not output.parent.exists()
    assert tree_digest(MINI_GEO) == before

    code, out, _ = run(
        "geometry", "apply", str(config), "--output", str(output), "--dry-run", "--format", "json"
    )
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert [c["new_value"] for c in payload["changes"]] == ["405*mm", "90"]
    assert not output.exists()


def test_apply_success_then_existing_output_rejected(config: Path, tmp_path: Path) -> None:
    output = tmp_path / "generated" / "mini"
    code, out, err = run("geometry", "apply", str(config), "--output", str(output))
    assert code == 0
    assert err == "" and "400*mm  ->  405*mm" in out
    assert f"wrote snapshot: {output}" in out
    assert f"compact:        {output / MINI_COMPACT}" in out
    assert re.search(r"fingerprint:    [0-9a-f]{64}", out)
    assert (output / MINI_COMPACT).is_file()
    assert (output / "k4opt-manifest.json").is_file()

    code, _, err = run("geometry", "apply", str(config), "--output", str(output))
    assert code == 1
    assert "already exists" in err


def test_apply_json_output(config: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    code, out, _ = run(
        "geometry", "apply", str(config), "--output", str(output), "--format", "json"
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["dry_run"] is False
    assert payload["schema"] == "k4opt.geometry.apply/1"
    assert (
        payload["fingerprint"]
        == json.loads((output / "k4opt-manifest.json").read_text())["fingerprint"]
    )
    assert payload["output_root"] == str(output)
    assert payload["compact_path"] == str(output / MINI_COMPACT)
    assert payload["files"][0]["path"] == "compact/det_v1/tracker.xml"
    manifest = json.loads((output / "k4opt-manifest.json").read_text())
    assert manifest["changes"] == payload["changes"]


def test_apply_resolution_errors_exit_1_and_write_nothing(tmp_path: Path) -> None:
    bad = write_config(
        tmp_path / "bad.yaml",
        MINI_COMPACT,
        root=MINI_GEO,
        overrides=[
            {"target": "constants.Nope", "value": '"1"'},
            {"target": "constants.Tracker_cooling_gap", "value": '"1"'},
        ],
    )
    output = tmp_path / "out"
    for extra in ([], ["--dry-run"]):
        code, out, err = run("geometry", "apply", str(bad), "--output", str(output), *extra)
        assert code == 1
        assert out == ""
        assert "k4opt: error:" in err
        assert "unknown target 'constants.Nope'" in err
        assert "constants.Tracker_cooling_gap" in err and "file:" in err
        assert not output.exists()


def test_config_errors(tmp_path: Path) -> None:
    unquoted = write_config(
        tmp_path / "a.yaml",
        MINI_COMPACT,
        root=MINI_GEO,
        overrides=[{"target": "constants.SolenoidField", "value": "90"}],
    )
    code, _, err = run("geometry", "inspect", str(unquoted))
    assert code == 1 and "must be a quoted string" in err

    unknown_key = write_config(
        tmp_path / "b.yaml", MINI_COMPACT, root=MINI_GEO, extra="exclude: [build]"
    )
    code, _, err = run("geometry", "inspect", str(unknown_key))
    assert code == 1 and "unknown key" in err

    code, _, err = run("geometry", "inspect", str(tmp_path / "missing.yaml"))
    assert code == 1 and "not found" in err

    (tmp_path / "dup.yaml").write_text(
        f"geometry: {MINI_COMPACT}\noverrides:\n"
        '  - target: constants.SolenoidField\n    value: "1"\n    value: "2"\n'
    )
    code, _, err = run("geometry", "inspect", str(tmp_path / "dup.yaml"))
    assert code == 1 and "duplicate key 'value'" in err

    # merge keys keep working (explicit keys win); unhashable keys are a controlled error
    merged = write_config(tmp_path / "m.yaml", MINI_COMPACT, root=MINI_GEO)
    merged.write_text(merged.read_text() + "x-base: &b {a: 1}\nx-use:\n  <<: *b\n  a: 2\n")
    code, _, err = run("geometry", "inspect", str(merged))
    assert "duplicate key" not in err and "constructor" not in err
    (tmp_path / "k.yaml").write_text("? [a, b]\n: 1\n")
    code, _, err = run("geometry", "inspect", str(tmp_path / "k.yaml"))
    assert code == 1 and "unhashable" in err

    (tmp_path / "c.yaml").write_text("geometry: x.xml\nroot: ./nowhere\n")
    code, _, err = run("geometry", "inspect", str(tmp_path / "c.yaml"))
    assert code == 1 and "not an existing directory" in err

    (tmp_path / "d.yaml").write_text("geometry: [1, 2]\n")
    code, _, err = run("geometry", "inspect", str(tmp_path / "d.yaml"))
    assert code == 1 and "must be a non-empty string" in err

    (tmp_path / "e.yaml").write_text("overrides: []\n")
    code, _, err = run("geometry", "inspect", str(tmp_path / "e.yaml"))
    assert code == 1 and "missing required key 'geometry'" in err


def test_geometry_usage_errors_exit_2() -> None:
    with pytest.raises(SystemExit) as info:
        run("geometry", "apply", "x.yaml")  # --output is required
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        run("geometry")  # an action is required
    assert info.value.code == 2
