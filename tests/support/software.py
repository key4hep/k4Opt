"""Small Key4hep stacks for tests of what candidates run on: an LCG view with the manifests of its
release, as ``/cvmfs/sft-nightlies.cern.ch/lcg`` holds them, and a Spack release
(``tests/fixtures/software``)."""

from __future__ import annotations

import shutil
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "software"
PLATFORM = "x86_64-el9-gcc16-opt"


def lcg_view(root: Path) -> Path:
    """The ``setup.sh`` of the Saturday slot of a copy of the LCG view in ``root``, what
    ``KEY4HEP_STACK`` names."""
    shutil.copytree(FIXTURES / "lcg", root / "lcg", symlinks=True)
    return root / "lcg" / "views" / "devkey-head" / "Sat" / PLATFORM / "setup.sh"


def spack_release(root: Path) -> Path:
    """The ``setup.sh`` of a copy of the Spack release in ``root``."""
    shutil.copytree(FIXTURES / "spack", root / "spack")
    release = root / "spack" / "releases" / "2026-10-20" / "x86_64-almalinux9-gcc14.2.0-opt"
    return release / "key4hep-stack" / "2026-10-20-2kehwx" / "setup.sh"


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
