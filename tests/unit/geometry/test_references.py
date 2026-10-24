"""Tests for the target reference grammar."""

from __future__ import annotations

import pytest

from k4opt.geometry import InvalidTargetError, parse_reference
from k4opt.geometry.references import Step


@pytest.mark.parametrize(
    "text",
    [
        "constants.InnerTracker_Barrel_radius_1",
        "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi",
        "detector[name=InnerTrackerBarrel].layer[id=1].rphi_layout.nphi",
        "lccdd.detectors[2].detector[1].layer[2].rphi_layout.nphi",
        "GlobalSolenoid.inner_field",
        "Vertex_FCC-SEED_o1_v01.title",
        "a.b[name=x.y+z-1].c",
        "_x.y_1",
    ],
)
def test_valid_references_round_trip(text: str) -> None:
    reference = parse_reference(text)
    assert reference.text == text
    assert str(reference) == text


def test_constant_reference_structure() -> None:
    reference = parse_reference("constants.SolenoidField")
    assert reference.kind == "constant"
    assert reference.attribute == "SolenoidField"
    assert reference.segments == ()
    assert reference.order_sensitive is False


def test_path_reference_structure() -> None:
    reference = parse_reference("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi")
    assert reference.kind == "path"
    assert reference.attribute == "nphi"
    assert reference.segments == (
        Step("InnerTrackerBarrel"),
        Step("layer", "id", "1"),
        Step("rphi_layout"),
    )
    assert reference.order_sensitive is False


def test_index_step_is_order_sensitive() -> None:
    reference = parse_reference("lccdd.module_component[2].thickness")
    assert reference.segments[1] == Step("module_component", "index", "2")
    assert reference.order_sensitive is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        " constants.X",
        "constants.X ",
        "constants",
        "constants.a.b",
        "constants.not valid",
        "nphi",
        "a..b",
        "a.",
        ".a",
        "a[1].b",
        "a[id=1].b",
        "a[0].b",
        "a[x=1].b",
        "a[name=].b",
        "a[name=has space].b",
        "a[name=x].b c",
        "1a.b",
        "a.b[.c",
        "a.b].c",
        "a.b:c",
        "a/b.c",
    ],
)
def test_invalid_references(text: str) -> None:
    with pytest.raises(InvalidTargetError):
        parse_reference(text)


def test_non_string_rejected() -> None:
    with pytest.raises(InvalidTargetError):
        parse_reference(42)  # type: ignore[arg-type]
