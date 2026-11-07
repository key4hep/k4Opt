"""A small Key4hep stack for tests of what candidates run on: an LCG view with the manifests of its
release, as ``/cvmfs/sft-nightlies.cern.ch/lcg`` holds them (``tests/fixtures/software``)."""

from __future__ import annotations

import shutil
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "software"
PLATFORM = "x86_64-el9-gcc16-opt"
VIEW = FIXTURES / "lcg" / "views" / "devkey-head" / "Sat" / PLATFORM / "setup.sh"
"""The checked-in view, for tests that only read it (``KEY4HEP_STACK`` of a test study)."""
LABEL = "LCG devkey-head, built Sat Nov 7 04:17:32 2026"
"""What :attr:`k4opt.software.Build.label` says of :data:`VIEW`."""


def lcg_view(root: Path) -> Path:
    """The ``setup.sh`` of the Saturday slot of a copy of the LCG view in ``root``, what
    ``KEY4HEP_STACK`` names."""
    shutil.copytree(FIXTURES / "lcg", root / "lcg", symlinks=True)
    return root / "lcg" / "views" / "devkey-head" / "Sat" / PLATFORM / "setup.sh"


def manifest(view: Path, name: str = "externals") -> Path:
    """The manifest of the release of the LCG view at ``view``."""
    release = view.parents[4] / "nightlies" / "devkey-head" / "Sat"
    return release / f"LCG_{name}_{PLATFORM}.txt"


def republish(view: Path) -> None:
    """A later build into the slot of the LCG view at ``view``, as a nightly writes it: the same
    paths, another time and other packages."""
    view.write_text(view.read_text().replace("Sat Nov  7", "Sat Nov 14"))
    release = manifest(view)
    release.write_text(release.read_text().replace("1a2b3", "9f8e7"))
