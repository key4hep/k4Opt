"""Exceptions raised by the ddsim component."""

from __future__ import annotations

from k4opt.errors import K4OptError


class DdsimError(K4OptError):
    """The generator settings or arguments for ``ddsim`` are unsupported or conflict.

    Raised before anything is run. A ``ddsim`` that fails is reported by
    :attr:`k4opt.ddsim.SimulationRun.error`, not by an exception.
    """
