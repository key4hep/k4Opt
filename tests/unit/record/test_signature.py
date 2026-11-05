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
        environment={"KEY4HEP_STACK": "/cvmfs/stack", "search": {}},
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


def _with_search(search: dict[str, str]) -> dict[str, Any]:
    program = {"path": "/cvmfs/a/bin/ddsim", "bytes": 5, "sha256": "a" * 64, "interpreter": None}
    document = stage_document(
        "simulation",
        SIMULATION,
        reads={"generation": "0" * 64},
        files={},
        program=program,
        environment={"KEY4HEP_STACK": "/cvmfs/stack", "search": search},
        k4opt="1" * 64,
    )
    return signed(document)


def test_path_is_not_signed_but_what_else_decides_what_a_program_loads_is() -> None:
    # PATH differs between shells that run the same programs; its interpreter is signed instead
    here = _with_search({"PATH": "p" * 64, "PYTHONPATH": "y" * 64, "LD_LIBRARY_PATH": "l" * 64})
    there = _with_search({"PATH": "q" * 64, "PYTHONPATH": "y" * 64, "LD_LIBRARY_PATH": "l" * 64})
    other = _with_search({"PATH": "p" * 64, "PYTHONPATH": "z" * 64, "LD_LIBRARY_PATH": "l" * 64})
    assert here == there != other
    assert "PATH" not in here["document"]["environment"]["search"]


def test_the_interpreter_a_program_runs_in_is_signed() -> None:
    def signature(interpreter: object) -> dict[str, Any]:
        entry = {"path": "/a/ddsim", "bytes": 5, "sha256": "a" * 64, "interpreter": interpreter}
        return _signature(entry)

    venv = {"path": "/cvmfs/venv/bin/python3", "bytes": 9, "sha256": "c" * 64}
    assert signature(venv) != signature({**venv, "path": "/other/venv/bin/python3"})
    assert signature(venv) != signature(None)
