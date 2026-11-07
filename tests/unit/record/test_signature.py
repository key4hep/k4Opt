"""Stage signatures (``k4opt.record.signature``): what a stage is made from, without where."""

from __future__ import annotations

from typing import Any

from k4opt.problem import Stage
from k4opt.record.signature import signed, stage_document

SIMULATION = Stage("simulation", executor="ddsim", events=10, seed=1)


def _signature(program: Any) -> dict[str, Any]:
    document = stage_document(
        "simulation",
        SIMULATION,
        reads={"generation": "0" * 64},
        files={},
        program=program,
        software="s" * 64,
        k4opt="1" * 64,
    )
    return signed(document)


def test_a_program_is_signed_by_its_content_not_where_it_is() -> None:
    here = _signature({"path": "/cvmfs/a/bin/ddsim", "bytes": 5, "sha256": "a" * 64})
    there = _signature({"path": "/opt/copy/bin/ddsim", "bytes": 5, "sha256": "a" * 64})
    other = _signature({"path": "/cvmfs/a/bin/ddsim", "bytes": 5, "sha256": "b" * 64})
    assert here == there != other
    assert here["document"]["program"] == {"bytes": 5, "sha256": "a" * 64, "interpreter": None}
    assert _signature(None)["document"]["program"] is None  # a stage that runs no program


def test_the_software_a_stage_runs_on_is_signed() -> None:
    def signature(software: str) -> dict[str, Any]:
        program = {"path": "/a/ddsim", "bytes": 5, "sha256": "a" * 64, "interpreter": None}
        return signed(
            stage_document(
                "simulation",
                SIMULATION,
                reads={"generation": "0" * 64},
                files={},
                program=program,
                software=software,
                k4opt="1" * 64,
            )
        )

    assert signature("s" * 64) == signature("s" * 64) != signature("t" * 64)
    assert signature("s" * 64)["document"]["software"] == "s" * 64


def test_the_interpreter_a_program_runs_in_is_signed() -> None:
    def signature(interpreter: object) -> dict[str, Any]:
        entry = {"path": "/a/ddsim", "bytes": 5, "sha256": "a" * 64, "interpreter": interpreter}
        return _signature(entry)

    venv = {"path": "/cvmfs/venv/bin/python3", "bytes": 9, "sha256": "c" * 64}
    assert signature(venv) != signature({**venv, "path": "/other/venv/bin/python3"})
    assert signature(venv) != signature(None)
