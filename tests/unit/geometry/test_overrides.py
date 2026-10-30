"""Tests for override validation and resolution."""

from __future__ import annotations

import pytest

from k4opt.geometry import (
    AmbiguousTargetError,
    Catalog,
    DuplicateTargetError,
    GeometrySource,
    InvalidTargetError,
    Override,
    TargetError,
    UnknownTargetError,
)


def test_override_requires_string_value_and_valid_target() -> None:
    with pytest.raises(InvalidTargetError, match="must be a string"):
        Override(target="constants.X", value=90)  # type: ignore[arg-type]
    with pytest.raises(InvalidTargetError, match="invalid target"):
        Override(target="constants", value="1")
    with pytest.raises(InvalidTargetError):
        Override(target="constants.X", value="1", file=3)  # type: ignore[arg-type]
    override = Override(target="constants.X", value="90")
    assert override.file is None


@pytest.mark.parametrize("value", ["a\x00b", "\ud800", "\x08", "\ufffe"])
def test_override_value_must_be_valid_xml_text(value: str) -> None:
    with pytest.raises(InvalidTargetError, match="not valid in XML"):
        Override(target="constants.X", value=value)


def test_resolve_binds_old_and_new_values(
    mini_source: GeometrySource, mini_catalog: Catalog
) -> None:
    resolved = mini_source.resolve(
        [
            Override("constants.InnerTracker_Barrel_radius_1", "405*mm"),
            Override("InnerTrackerBarrel.layer[id=1].rphi_layout.nphi", "90"),
        ],
        mini_catalog,
    )
    assert [(r.old_value, r.new_value) for r in resolved] == [("400*mm", "405*mm"), ("43*2", "90")]
    assert resolved[0].file == "compact/det_v1/tracker.xml"
    assert resolved[1].attribute == "nphi"
    assert all(r.changed for r in resolved)
    assert resolved[0].to_dict()["locator"] == "/lccdd/define/constant[2]"


def test_unchanged_value_still_resolves(mini_source: GeometrySource) -> None:
    (resolved,) = mini_source.resolve([Override("constants.SolenoidField", "2*tesla")])
    assert resolved.changed is False


def test_unknown_target(mini_source: GeometrySource) -> None:
    with pytest.raises(UnknownTargetError, match="unknown target"):
        mini_source.resolve([Override("constants.DoesNotExist", "1")])


def test_ambiguous_target_and_file_discriminator(
    mini_source: GeometrySource, mini_catalog: Catalog
) -> None:
    with pytest.raises(AmbiguousTargetError, match="name its file"):
        mini_source.resolve([Override("constants.Tracker_cooling_gap", "6*mm")], mini_catalog)
    (resolved,) = mini_source.resolve(
        [
            Override(
                "constants.Tracker_cooling_gap",
                "6*mm",
                file="./compact/common/../common/shared_constants.xml",
            )
        ],
        mini_catalog,
    )
    assert resolved.file == "compact/common/shared_constants.xml"
    assert resolved.old_value == "7*mm"


def test_file_discriminator_errors(mini_source: GeometrySource, mini_catalog: Catalog) -> None:
    with pytest.raises(UnknownTargetError, match=r"does not exist in compact/det_v1/display\.xml"):
        mini_source.resolve(
            [Override("constants.Tracker_cooling_gap", "6*mm", file="compact/det_v1/display.xml")]
        )
    with pytest.raises(UnknownTargetError, match="not among the reachable geometry files"):
        mini_source.resolve(
            [Override("constants.SolenoidField", "1*tesla", file="compact/det_v2/det_v2.xml")]
        )
    with pytest.raises(UnknownTargetError, match="escapes"):
        mini_catalog.lookup("constants.SolenoidField", file="../x.xml")
    with pytest.raises(UnknownTargetError, match="relative path"):
        mini_catalog.lookup("constants.SolenoidField", file="/abs/x.xml")


def test_duplicate_selection_rejected(mini_source: GeometrySource) -> None:
    with pytest.raises(DuplicateTargetError, match="both select"):
        mini_source.resolve(
            [
                Override("constants.SolenoidField", "1*tesla"),
                Override("constants.SolenoidField", "3*tesla"),
            ]
        )
    with pytest.raises(DuplicateTargetError):
        mini_source.resolve(
            [
                Override(
                    "constants.Tracker_cooling_gap", "1*mm", file="compact/det_v1/tracker.xml"
                ),
                Override(
                    "constants.Tracker_cooling_gap", "2*mm", file="compact/det_v1/tracker.xml"
                ),
            ]
        )


def test_all_errors_reported_together(mini_source: GeometrySource) -> None:
    with pytest.raises(TargetError) as info:
        mini_source.resolve(
            [
                Override("constants.DoesNotExist", "1"),
                Override("constants.Tracker_cooling_gap", "6*mm"),
                Override("constants.SolenoidField", "3*tesla"),
            ]
        )
    assert len(info.value.errors) == 2
    assert isinstance(info.value.errors[0], UnknownTargetError)
    assert isinstance(info.value.errors[1], AmbiguousTargetError)
    assert "2 overrides could not be resolved" in str(info.value)


def test_non_override_objects_rejected(mini_source: GeometrySource) -> None:
    with pytest.raises(InvalidTargetError, match="expected an Override"):
        mini_source.resolve([{"target": "constants.SolenoidField", "value": "1"}])  # type: ignore[list-item]
