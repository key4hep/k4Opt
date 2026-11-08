"""The official build in full: what an LCG view says of each package it holds.

:class:`~k4opt.software.Build` identifies a build by the digests of its manifests. To diff a
local change against what the nightly built, and to name the build after it is gone, a study
also keeps what those manifests say (:class:`Release`): each package's name, version, LCG build
hash and where it is installed, and for a package built from a repository's ``HEAD`` the commit
it was built from and the revision of lcgcmake that built it, as the package's
``.buildinfo_<name>.txt`` says. k4opt never keeps a file of the build itself.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from k4opt.frozen import FrozenMap
from k4opt.software.errors import SoftwareError
from k4opt.software.identity import Build

MANIFESTS = ("LCG_externals", "LCG_contrib")
"""The manifests of an LCG release, by the role :class:`~k4opt.software.Build` digests them as."""

_FIELD = re.compile(r"\b([A-Z]+): ('?)([^,']*)\2(?:,|$)")
_HEX = re.compile(r"[0-9a-f]{4,40}")


@dataclass(frozen=True, slots=True)
class Package:
    """A package of an LCG release: its ``name`` (as the release's directories spell it:
    ``DD4hep``, ``ddmarlinpandora``), ``version`` (``HEAD`` for one built from a repository's
    head), LCG's build ``hash`` of it, and ``prefix``, where it is installed. A ``HEAD`` package
    also says the ``revision`` it was built from (an abbreviated commit) and ``lcgcmake``, the
    revision of the recipes that built it; ``None`` where its build information says none."""

    name: str
    version: str
    hash: str
    prefix: str
    revision: str | None = None
    lcgcmake: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "hash": self.hash,
            "prefix": self.prefix,
            "revision": self.revision,
            "lcgcmake": self.lcgcmake,
        }

    @classmethod
    def from_dict(cls, data: object) -> Package:
        keys = {"name", "version", "hash", "prefix", "revision", "lcgcmake"}
        if not isinstance(data, Mapping) or set(data) != keys:
            raise SoftwareError(f"no package of a release: {data!r}")
        name, version, hash_, prefix = (data[k] for k in ("name", "version", "hash", "prefix"))
        revision, lcgcmake = data["revision"], data["lcgcmake"]
        if not all(isinstance(v, str) for v in (name, version, hash_, prefix)) or not all(
            v is None or isinstance(v, str) for v in (revision, lcgcmake)
        ):
            raise SoftwareError(f"a package of a release is not text: {data!r}")
        return cls(name, version, hash_, prefix, revision, lcgcmake)


@dataclass(frozen=True, slots=True)
class Release:
    """What an LCG view says of its build: the ``build`` it is, the ``setup`` file
    ``KEY4HEP_STACK`` led to, the directory of its release's manifests (``nightlies``), each
    manifest's text by its role (``manifests``, what :attr:`Build.digests` hashes) and its
    ``packages`` by name."""

    build: Build
    setup: str
    nightlies: str
    manifests: Mapping[str, bytes]
    packages: Mapping[str, Package]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifests", FrozenMap(self.manifests))
        object.__setattr__(self, "packages", FrozenMap(self.packages))

    @classmethod
    def of(cls, stack: str | None) -> Release | None:
        """The release of the LCG view ``stack`` (``KEY4HEP_STACK``) names, read now (about a
        second for a nightly not read before: one file per ``HEAD`` package); ``None`` for any
        other stack, and for one whose manifests cannot be read."""
        build = Build.of(stack)
        if build.kind != "lcg" or not stack or build.platform is None:
            return None
        setup = Path(os.path.realpath(stack))
        release, slot, platform = setup.parts[-4:-1]
        nightlies = setup.parents[4] / "nightlies" / release / slot
        manifests: dict[str, bytes] = {}
        for role in MANIFESTS:
            try:
                manifests[role] = (nightlies / f"{role}_{platform}.txt").read_bytes()
            except OSError:
                return None
        packages: dict[str, Package] = {}
        for text in manifests.values():
            packages |= _packages(text.decode(errors="replace"), nightlies, f"{release}/{slot}")
        return cls(build, str(setup), str(nightlies), manifests, packages)

    def package(self, name: str) -> Package | None:
        """The package called ``name``, however its case and hyphens are spelt (as
        ``k4_local_repo`` names a checkout: ``DDMarlinPandora`` is ``ddmarlinpandora``)."""
        wanted = _normal(name)
        for package in self.packages.values():
            if _normal(package.name) == wanted:
                return package
        return None

    def owner(self, path: str) -> Package | None:
        """The package whose installation ``path`` (links followed) lies in, ``None`` if none."""
        real = os.path.realpath(path)
        for package in self.packages.values():
            for prefix in {package.prefix, os.path.realpath(package.prefix)}:
                if real == prefix or real.startswith(f"{prefix}/"):
                    return package
        return None

    def to_dict(self) -> dict[str, object]:
        """As JSON data: how many packages it has, and each ``HEAD`` package, the one thing the
        manifests do not say (the commit it was built from). Every other package is in the
        manifests, which a study keeps apart, by digest: a thousand packages would repeat them
        uncompressed (300 kB against 34 kB)."""
        return {
            "build": self.build.to_dict(),
            "setup": self.setup,
            "nightlies": self.nightlies,
            "count": len(self.packages),
            "packages": {
                name: p.to_dict()
                for name, p in sorted(self.packages.items())
                if p.version == "HEAD"
            },
        }


def _normal(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def _packages(text: str, nightlies: Path, where: str) -> dict[str, Package]:
    """The packages a manifest lists (``name; hash; version; path; dependencies``), each found in
    ``nightlies`` (the manifest's own directory): the path a manifest gives is where the release
    was published, ``…/<release>/<slot>/<package>/<version>/<platform>``, which a copy of it
    moves."""
    found: dict[str, Package] = {}
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(";")]
        if len(fields) < 4 or not fields[0] or not _HEX.fullmatch(fields[1]):
            continue  # the header (PLATFORM:, VERSION:, COMPILER:)
        name, hash_, version, published = fields[:4]
        _, inside, relative = published.partition(f"/{where}/")
        prefix = str(nightlies / relative) if inside else published
        revision, lcgcmake = _buildinfo(Path(prefix), name) if version == "HEAD" else (None, None)
        found[name] = Package(name, version, hash_, prefix, revision, lcgcmake)
    return found


def _buildinfo(prefix: Path, name: str) -> tuple[str | None, str | None]:
    """The revision a ``HEAD`` package was built from and that of lcgcmake that built it, as its
    ``.buildinfo_<name>.txt`` says (``REVISION: 821384f``, ``GITHASH: '5cc9f1ea0'``)."""
    try:
        text = (prefix / f".buildinfo_{name}.txt").read_text(errors="replace")
    except OSError:
        return None, None
    fields = {key: value.strip() for key, _, value in _FIELD.findall(text)}
    # "4644ad2|7" for some (DD4hep, podio): the commit, then what lcgcmake adds to it
    revision = fields.get("REVISION", "").partition("|")[0]
    lcgcmake = fields.get("GITHASH")
    return (
        revision if revision and _HEX.fullmatch(revision) else None,
        lcgcmake if lcgcmake and _HEX.fullmatch(lcgcmake) else None,
    )


__all__ = ["MANIFESTS", "Package", "Release"]
