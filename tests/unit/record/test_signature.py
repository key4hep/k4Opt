"""Stage signatures (``k4opt.record.signature``): what a stage is made from, without where."""

from __future__ import annotations

import json
from typing import Any

import pytest

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


def test_an_option_whose_files_are_signed_is_not_signed_by_where_they_are() -> None:
    def signature(directory: str, content: str) -> dict[str, Any]:
        stage = Stage(
            "reconstruction",
            executor="k4run",
            options={"directory": directory, "options": "CLDReconstruction.py", "cms": 240},
        )
        files = {
            "directory/CLDReconstruction.py": {"bytes": 1, "sha256": content * 64},
            "directory/PandoraSettings/Default.xml": {"bytes": 2, "sha256": "e" * 64},
        }
        return signed(
            stage_document(
                "reconstruction",
                stage,
                reads={"simulation": "0" * 64},
                files=files,
                program=None,
                software="s" * 64,
                k4opt="1" * 64,
            )
        )

    here = signature("$CLDCONFIG/share/CLDConfig", "a")
    assert here["document"]["definition"]["directory"] == "<its files>"
    assert here["document"]["definition"]["options"] == "CLDReconstruction.py"  # a name in it
    assert here == signature("/home/me/CLDConfig", "a") != signature("/home/me/CLDConfig", "b")


@pytest.mark.parametrize(
    ("options", "files"),
    [
        ({"options": "/home/me/configA/job.py"}, {"options": {"bytes": 1, "sha256": "a" * 64}}),
        (
            {"options": "job.py", "data": {"dataFolder": "/scratch/calibration/"}},
            {"data.dataFolder/x.root": {"bytes": 1, "sha256": "a" * 64}},
        ),
    ],
    ids=["an options file run where it is", "data read where it is, a trailing / kept"],
)
def test_a_path_a_program_is_given_as_it_is_stays_signed(
    options: dict[str, Any], files: dict[str, Any]
) -> None:
    def signature(changed: dict[str, Any]) -> str:
        stage = Stage("reconstruction", executor="k4run", options=changed)
        document = stage_document(
            "reconstruction",
            stage,
            reads={"simulation": "0" * 64},
            files=files,
            program=None,
            software="s" * 64,
            k4opt="1" * 64,
        )
        return str(signed(document)["sha256"])

    moved = json.loads(json.dumps(options).replace("/home/me/configA", "/home/me/configB"))
    moved = json.loads(json.dumps(moved).replace("calibration/", "calibration"))
    assert signature(options) != signature(moved)
