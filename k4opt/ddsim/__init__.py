"""ddsim component: run Key4hep's ``ddsim`` on a prepared geometry.

The primaries come from a particle gun or from an input sample file.

Translates the generation settings of a problem into ``ddsim`` options (units included, the
direction distribution checked against what ``ddsim`` offers), protects the options that k4opt
manages from conflicting user arguments, runs the executable of the active Key4hep
environment and keeps command, exit code and logs. It does not read the output; that is up to
the caller. See ``docs/ddsim.md``.

Typical use::

    from pathlib import Path
    from k4opt.ddsim import simulate

    run = simulate(
        compact=Path("geometry/CLD_o2_v09.xml"),
        workdir=Path("run/simulation"),  # must not exist yet
        events=100,
        seed=42,
        particle_gun={...},  # evaluation.generation of a problem file
        # or sample=Path("events.hepmc") instead
        args=("--physics.list", "FTFP_BERT"),
    )
    print(run.error or run.output)
"""

from k4opt.ddsim.command import DIRECTION_DISTRIBUTIONS
from k4opt.ddsim.errors import DdsimError
from k4opt.ddsim.run import SimulationRun, check_arguments, simulate

__all__ = [
    "DIRECTION_DISTRIBUTIONS",
    "DdsimError",
    "SimulationRun",
    "check_arguments",
    "simulate",
]
