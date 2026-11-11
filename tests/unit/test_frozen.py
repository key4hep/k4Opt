"""JSON values compared as JSON: type and value, whatever the order of an object's keys."""

from __future__ import annotations

from k4opt.frozen import freeze, same_json


def test_the_order_of_an_objects_keys_makes_no_other_json() -> None:
    plain = {"study": "s", "values": {"x": 1.5, "r": 12.0}}
    reordered = {"values": {"r": 12.0, "x": 1.5}, "study": "s"}
    assert same_json(plain, reordered)
    assert same_json(freeze(plain, "plain", ValueError), reordered)  # frozen or not


def test_a_value_of_another_type_or_order_in_a_list_is_other_json() -> None:
    assert not same_json({"x": 300}, {"x": 300.0})
    assert not same_json({"x": 1}, {"x": True})
    assert not same_json(["x", "r"], ["r", "x"])
    assert not same_json({"x": 1}, {"x": 1, "r": 2})
