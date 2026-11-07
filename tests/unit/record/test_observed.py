"""What a record says its stages used, read back: by stage, and the local builds none of them
read."""

from __future__ import annotations

from typing import Any

import pytest

from k4opt.record import observed_stages, unused_local


def record(stages: dict[str, Any], *local: str) -> dict[str, Any]:
    software = {"local": [{"variable": "PYTHONPATH", "path": path} for path in local]}
    return {
        "samples": {"gamma": {"stages": stages}},
        "environment": {"observed": {"software": software}},
    }


def ran(level: str = "files", *local: str, complete: bool = True) -> dict[str, Any]:
    observed = {"level": level, "complete": complete, "local": list(local)}
    return {"status": "done", "argv": ["ddsim"], "observed": observed}


def test_each_stage_that_ran_a_program_says_what_it_used() -> None:
    stages = {"generation": {"status": "done"}, "simulation": ran("files", "/a")}
    assert observed_stages(record(stages)) == {
        "gamma/simulation": {"level": "files", "complete": True, "local": ["/a"]}
    }


def test_a_local_build_no_stage_read_is_named() -> None:
    stages = {"simulation": ran("files", "/a"), "reconstruction": ran("files")}
    assert unused_local(record(stages, "/a", "/b", "/b")) == ["/b"]


def test_every_local_build_read_leaves_none_unused() -> None:
    assert unused_local(record({"simulation": ran("files", "/a")}, "/a")) == []


def test_a_stage_without_a_program_says_nothing_of_local_builds() -> None:
    stages = {"generation": {"status": "done"}, "simulation": ran("files")}
    assert unused_local(record(stages, "/a")) == ["/a"]


@pytest.mark.parametrize(
    "stage",
    [
        ran("libraries"),
        ran("none"),
        ran("files", complete=False),  # a trace cut short may have held it
        {"status": "done", "reused": {}},
        {"status": "blocked"},  # never ran: it might have read it
        {"status": "failed"},  # a worker died under it, with what it read
        {"status": "done", "argv": ["ddsim"]},  # no word of how it was observed
    ],
)
def test_whether_a_local_build_ran_is_not_said_unless_it_is_known(stage: dict[str, Any]) -> None:
    assert unused_local(record({"simulation": stage}, "/a")) == []
