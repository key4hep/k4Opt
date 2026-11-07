"""What a record says its stages used, read back: by stage, and the local builds none of them
read."""

from __future__ import annotations

from typing import Any

from k4opt.record import observed_stages, unused_local


def record(stages: dict[str, Any], *local: str) -> dict[str, Any]:
    software = {"local": [{"variable": "PYTHONPATH", "path": path} for path in local]}
    return {
        "samples": {"gamma": {"stages": stages}},
        "environment": {"observed": {"software": software}},
    }


def ran(level: str = "files", *local: str) -> dict[str, Any]:
    return {"argv": ["ddsim"], "observed": {"level": level, "local": list(local)}}


def test_each_stage_that_ran_a_program_says_what_it_used() -> None:
    stages = {"generation": {"particle_gun": {}}, "simulation": ran("files", "/a")}
    assert observed_stages(record(stages)) == {
        "gamma/simulation": {"level": "files", "local": ["/a"]}
    }


def test_a_local_build_no_stage_read_is_named() -> None:
    stages = {"simulation": ran("files", "/a"), "reconstruction": ran("files")}
    assert unused_local(record(stages, "/a", "/b", "/b")) == ["/b"]


def test_every_local_build_read_leaves_none_unused() -> None:
    assert unused_local(record({"simulation": ran("files", "/a")}, "/a")) == []


def test_whether_a_local_build_ran_is_not_said_unless_every_stage_was_observed_closely() -> None:
    assert unused_local(record({"simulation": ran("libraries")}, "/a")) == []
    assert unused_local(record({"simulation": ran("none")}, "/a")) == []
    assert unused_local(record({"simulation": {"reused": {}}}, "/a")) == []
