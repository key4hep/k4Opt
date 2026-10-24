"""k4opt: a detector optimization framework for Key4hep.

The package is organized by component; each lives in its own subpackage with its
own public API, errors and command group:

* :mod:`k4opt.geometry` - discover and override parameters in DD4hep XML detector
  descriptions and materialize modified geometries.

Shared pieces live at the top level: :mod:`k4opt.errors` and the ``k4opt`` command
(:mod:`k4opt.cli`).
"""

from importlib.metadata import PackageNotFoundError, version

from k4opt.errors import ConfigError, K4OptError

try:
    __version__ = version("k4opt")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["ConfigError", "K4OptError", "__version__"]
