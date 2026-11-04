"""The options of a reconstruction stage, as k4opt passes them to k4run (no stack needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from k4opt.k4run import (
    K4runError,
    check_options,
    k4run_arguments,
    options_files,
    settable_option,
    wiring,
)
from k4opt.k4run.command import job_arguments
from k4opt.program import ProgramError, Settable

JOB = {"options": "CLDReconstruction.py"}


def test_each_option_is_an_argument_of_the_job_in_the_order_written() -> None:
    options = {
        **JOB,
        "cms": 91,
        "truthTracking": True,
        "conformalTracking": False,
        "GeoSvc.OutputLevel": 3,
        "threshold": 5e-05,
        "inputFiles2": ["a.root", "b.root"],
        "outputBasename": "reco",
    }
    assert job_arguments(options) == (
        "--cms=91",
        "--truthTracking",
        "--conformalTracking=False",
        "--GeoSvc.OutputLevel=3",
        "--threshold=5e-05",
        "--inputFiles2",
        "a.root",
        "b.root",
        "--outputBasename=reco",
    )


def test_k4opt_sets_every_event_the_geometry_the_input_and_the_output_after_the_files() -> None:
    arguments = k4run_arguments(
        files=["CLDReconstruction.py", "more.py"],
        k4opt_options=Path("/run/k4opt.opts.py"),
        compact=Path("/run/geometry/CLD.xml"),
        input_file=Path("/run/sim/events.edm4hep.root"),
        output=Path("/run/reco/events.edm4hep.root"),
        options={**JOB, "cms": 91},
    )
    assert arguments == (
        "CLDReconstruction.py",
        "more.py",
        "/run/k4opt.opts.py",
        "--num-events=-1",
        "--GeoSvc.detectors=/run/geometry/CLD.xml",
        "--IOSvc.Input",
        "/run/sim/events.edm4hep.root",
        "--IOSvc.Output=/run/reco/events.edm4hep.root",
        "--cms=91",
    )


def test_the_job_s_own_arguments_take_the_geometry_and_the_input_when_the_stage_names_them() -> (
    None
):
    options = {**JOB, "geometry": "compactFile", "input": "inputFiles"}
    arguments = k4run_arguments(
        files=["CLDReconstruction.py"],
        k4opt_options=Path("/k.py"),
        compact=Path("/g.xml"),
        input_file=Path("/s.root"),
        output=Path("/r.root"),
        options=options,
        dry_run=True,
    )
    assert arguments == (
        "CLDReconstruction.py",
        "/k.py",
        "--dry-run",
        "--num-events=-1",
        "--compactFile=/g.xml",
        "--inputFiles",
        "/s.root",
        "--IOSvc.Output=/r.root",
    )
    assert wiring(options) == ("compactFile", "inputFiles")
    assert wiring(JOB) == (None, None)


@pytest.mark.parametrize(
    "key",
    [
        "num-events",
        "n",
        "evt-max",
        "dry-run",
        "interactive",
        "gdb",
        "GeoSvc.detectors",
        "detectors.GeoSvc",  # k4run's other spelling of every property
        "IOSvc.Input",
        "Input.IOSvc",
        "IOSvc.Output",
        "ApplicationMgr.EvtMax",
        "EvtMax.ApplicationMgr",
        "JobOptionsSvc.DUMPFILE",
        "DUMPFILE.JobOptionsSvc",
    ],
)
def test_what_k4opt_sets_is_refused_in_either_spelling(key: str) -> None:
    with pytest.raises(K4runError, match=rf"reconstruction\.{key}: k4opt sets this argument"):
        check_options({**JOB, key: "x"})


def test_another_property_of_a_managed_service_is_the_job_s() -> None:
    check_options({**JOB, "IOSvc.OutputLevel": 2, "GeoSvc.EnableGeant4Geo": False})


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "reconstruction: missing required key 'options'"),
        ({"options": []}, "reconstruction.options: must name at least one options file"),
        ({"options": ["a.py", "a.py"]}, "reconstruction.options: names a file twice"),
        ({"options": ["a.py", " "]}, "reconstruction.options[1]: must be a nonblank text"),
        (
            {"options": "/abs/a.py", "directory": "/cld"},
            "reconstruction.options: '/abs/a.py': with directory:, an options file is named "
            "relative to it",
        ),
        (
            {"options": "../a.py", "directory": "/cld"},
            "reconstruction.options: '../a.py': with directory:",
        ),
        ({**JOB, "directory": ""}, "reconstruction.directory: must be a nonblank text"),
        ({**JOB, "--cms": 91}, "write it without the leading dashes: 'cms'"),
        ({**JOB, "cms": None}, "reconstruction.cms: must be a number, a text, a boolean"),
        ({**JOB, "cms": {"value": 91}}, "reconstruction.cms: must be a number"),
        ({**JOB, "cms": []}, "reconstruction.cms: an empty list passes nothing"),
        ({**JOB, "files": ["a", "-b"]}, "reconstruction.files[1]: a value of a list must not"),
        ({**JOB, "x": float("nan")}, "reconstruction.x: must be a finite number"),
        ({**JOB, "geometry": "GeoSvc.detectors"}, "reconstruction.geometry: k4opt sets this"),
        (
            {**JOB, "geometry": "compactFile", "compactFile": "/my.xml"},
            "reconstruction.compactFile: is the argument geometry: names, which k4opt sets",
        ),
        (
            {**JOB, "geometry": "files", "input": "files"},
            "reconstruction.input: names 'files', the argument geometry: names too",
        ),
    ],
)
def test_options_k4opt_cannot_pass_are_refused(options: dict[str, Any], message: str) -> None:
    with pytest.raises(K4runError) as raised:
        check_options(options)
    assert message in str(raised.value)
    assert isinstance(raised.value, ProgramError)


def test_options_files_are_one_or_a_list() -> None:
    assert options_files(JOB) == ("CLDReconstruction.py",)
    assert options_files({"options": ["a.py", "sub/b.py"], "directory": "/d"}) == (
        "a.py",
        "sub/b.py",
    )


def test_a_parameter_sets_a_job_argument_as_its_value() -> None:
    assert settable_option("cms") == Settable("value")
    assert settable_option("DDCaloDigi.ECALThreshold") == Settable("value")
    for key in ("options", "directory", "geometry", "input"):
        with pytest.raises(K4runError, match="it says which job runs"):
            settable_option(key)
    with pytest.raises(K4runError, match="k4opt sets this argument itself"):
        settable_option("IOSvc.Input")


def test_an_option_a_parameter_sets_is_not_also_written_nor_the_geometry_argument() -> None:
    check_options(JOB, pending={"cms"})
    with pytest.raises(K4runError, match=r"reconstruction\.cms: is given, but a parameter sets it"):
        check_options({**JOB, "cms": 91}, pending={"cms"})
    with pytest.raises(K4runError, match="the argument that takes the geometry"):
        check_options({**JOB, "geometry": "compactFile"}, pending={"compactFile"})
