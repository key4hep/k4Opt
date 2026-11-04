"""k4run component: run a Key4hep reconstruction (a Gaudi job, ``k4run``) on a simulation.

A job is one or more Gaudi options files, such as CLDConfig's ``CLDReconstruction.py``, run from
a copy of the directory they belong to. k4opt gives the job the candidate's geometry, the events
of the simulation and the file to write, through k4FWCore's services (``GeoSvc.detectors``,
``IOSvc.Input``, ``IOSvc.Output``) or through the job's own arguments the stage names, has it
reconstruct every event, passes the stage's other options on as the job's arguments
(``cms: 91`` as ``--cms=91``) and checks, from the job's resolved options, that it read what it
was given. Whether the job takes its arguments is decided by the job itself (:func:`dry_run`), not
by a table of k4opt. It does not read the output; that is up to the caller. See
``docs/k4run.md``.

Typical use::

    from pathlib import Path
    from k4opt.k4run import reconstruct

    cld = Path("/cvmfs/.../CLDConfig")  # $CLDCONFIG/share/CLDConfig
    run = reconstruct(
        files=[cld / "CLDReconstruction.py"],
        directory=cld,
        compact=Path("geometry/CLD_o2_v09.xml"),
        input_file=Path("simulation/events.edm4hep.root"),
        workdir=Path("run/reconstruction"),  # must not exist yet
        options={"options": "CLDReconstruction.py", "directory": str(cld),
                 "geometry": "compactFile", "cms": 91, "truthTracking": True},
    )
    print(run.error or run.output)
"""

from k4opt.k4run.command import (
    DIRECTORY,
    EXECUTOR_KEYS,
    GEOMETRY,
    INPUT,
    OPTIONS,
    check_options,
    k4run_arguments,
    options_files,
    settable_option,
    wiring,
)
from k4opt.k4run.errors import K4runError
from k4opt.k4run.joboptions import properties, verify
from k4opt.k4run.run import LOG_FILES, ReconstructionRun, dry_run, job_files, reconstruct

__all__ = [
    "DIRECTORY",
    "EXECUTOR_KEYS",
    "GEOMETRY",
    "INPUT",
    "LOG_FILES",
    "OPTIONS",
    "K4runError",
    "ReconstructionRun",
    "check_options",
    "dry_run",
    "job_files",
    "k4run_arguments",
    "options_files",
    "properties",
    "reconstruct",
    "settable_option",
    "verify",
    "wiring",
]
