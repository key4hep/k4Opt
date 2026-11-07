"""The identity of what candidates run on: the build of the stack, the system and the variables.

A Key4hep nightly is a slot that a later build overwrites, so a path says nothing of what is in it:
:class:`Build` reads what the stack says of itself. An LCG view is identified by when it was
generated and the SHA-256 of its ``setup.sh`` and of both manifests of its release. Anything else
is *unrecognized*, and identified by the content of the file ``KEY4HEP_STACK`` names, or by that
text if there is no such file. No location enters a build, so ``latest`` and the weekday it points
to are one build.

Every environment variable a program would inherit may change what it does, and a value leaves no
trace in any file. So every variable is compared unless :data:`IGNORED` lists it as known to be
irrelevant, a user's own ``MY_SCALE`` as much as ``MARLIN_DLL``. Values are kept as digests: a
difference names a variable, and cannot show its old value.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from k4opt.frozen import FrozenMap
from k4opt.software.errors import SoftwareError

type Kind = Literal["lcg", "unrecognized"]

IGNORED = (
    # shell and terminal state
    "_",
    "PWD",
    "OLDPWD",
    "SHLVL",
    "SHELL",
    "TERM",
    "COLUMNS",
    "LINES",
    "DISPLAY",
    "MOTD_SHOWN",
    "BASH_FUNC_*",
    # session, login, credentials
    "USER",
    "HOSTNAME",
    "TMPDIR",
    "KRB5CCNAME",
    "SSH_*",
    "XDG_*",
    "DBUS_*",
    "SELINUX_*",
    # new on every run of a service
    "INVOCATION_ID",
    "JOURNAL_STREAM",
    "MANAGERPID",
    "SYSTEMD_EXEC_PID",
    # bookkeeping of other tools
    "MODULEPATH",
    "CONDA_SHLVL",
    "CONDA_BACKUP_*",
    "VSCODE_*",
    "GIT_*",
    "PYTEST_*",
    # set by ROOT's Python in whatever process imports it, from the build
    "CPPYY_API_PATH",
    "CPPYY_NO_ROOT_FILTER",
    # identified by what they lead to: the programs and their interpreter, the geometry, the build
    "PATH",
    "K4GEO",
    "KEY4HEP_STACK",
)
"""The environment variables known to be irrelevant to what a program does, each justified in
``docs/software.md``; a name that ends in ``*`` stands for every name it starts. Every other
variable is compared. A wrongly compared variable costs a refusal that names it, a wrongly ignored
one could cost correctness, so the list stays short."""

_KINDS = ("lcg", "unrecognized")
_NAMES = frozenset(name for name in IGNORED if not name.endswith("*"))
_PREFIXES = tuple(name[:-1] for name in IGNORED if name.endswith("*"))
_GENERATED = re.compile(rb"^#\s*Generated:[ \t]*(.+?)[ \t\r]*$", re.MULTILINE)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _compared(name: str) -> bool:
    return name not in _NAMES and not name.startswith(_PREFIXES)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(path: Path) -> str | None:
    """The SHA-256 of the file ``path``, ``None`` if it cannot be read."""
    try:
        return _sha256(path.read_bytes())
    except OSError:
        return None


def _keys(data: object, keys: frozenset[str], where: str) -> Mapping[str, object]:
    """``data`` as a mapping with exactly ``keys``; :class:`SoftwareError` naming ``where``."""
    if not isinstance(data, Mapping) or set(data) != keys:
        raise SoftwareError(f"{where} is not an object of {sorted(keys)}: {data!r}")
    return data


def _text(value: object, where: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise SoftwareError(f"{where} is {value!r}, not text")
    return value


@dataclass(frozen=True, slots=True)
class Build:
    """One build of the stack, whichever place holds it.

    ``kind`` says how it was recognized: ``lcg`` or ``unrecognized``. ``release`` and
    ``platform`` are what its place says (``devkey-head``, ``x86_64-el9-gcc16-opt``), ``built``
    the time an LCG view says it was generated, as it writes it, and ``digests`` the SHA-256 of
    each file that identifies it, by role, ``None`` for one that cannot be read.
    """

    kind: Kind
    release: str | None
    platform: str | None
    built: str | None
    digests: Mapping[str, str | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "digests", FrozenMap(self.digests))

    @classmethod
    def of(cls, stack: str | None) -> Build:
        """The build ``stack`` (the value of ``KEY4HEP_STACK``) names, read now."""
        if not stack:
            return cls("unrecognized", None, None, None, {})
        path = Path(os.path.realpath(stack))  # latest/<platform> leads to a weekday's
        try:
            setup = path.read_bytes()
        except OSError:
            return cls("unrecognized", None, None, None, {"text": _sha256(stack.encode())})
        unrecognized = cls("unrecognized", None, None, None, {"setup": _sha256(setup)})
        return _lcg(path, setup) or unrecognized

    @property
    def label(self) -> str:
        """The build for a person: ``LCG devkey-head, built Sat Nov 7 04:17:32 2026``."""
        if self.kind == "lcg":
            return f"LCG {self.release}, built {' '.join((self.built or '').split())}"
        return "unrecognized stack"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "release": self.release,
            "platform": self.platform,
            "built": self.built,
            "digests": dict(self.digests),
        }

    @classmethod
    def from_dict(cls, data: object) -> Build:
        """The build :meth:`to_dict` wrote, checked; raises :class:`SoftwareError`."""
        entry = _keys(data, _BUILD, "the build")
        kind = entry["kind"]
        if kind not in _KINDS:
            raise SoftwareError(f"the build's kind is {kind!r}, no kind of build")
        digests = entry["digests"]
        if not isinstance(digests, Mapping):
            raise SoftwareError(f"the build's digests are {digests!r}, not an object")
        return cls(
            cast(Kind, kind),
            _text(entry["release"], "the build's release"),
            _text(entry["platform"], "the build's platform"),
            _text(entry["built"], "the build's time"),
            {str(role): _text(value, f"its {role} digest") for role, value in digests.items()},
        )


_BUILD = frozenset(("kind", "release", "platform", "built", "digests"))


def _lcg(path: Path, setup: bytes) -> Build | None:
    """The build of the LCG view whose ``setup.sh`` is at ``path``, as
    ``…/views/<release>/<slot>/<platform>/setup.sh``, with the manifests of its release in
    ``…/nightlies/<release>/<slot>/``; ``None`` for anything else."""
    generated = _GENERATED.search(setup)
    if generated is None or len(path.parts) < 6 or path.parts[-5] != "views":
        return None
    release, slot, name = path.parts[-4:-1]
    manifests = path.parents[4] / "nightlies" / release / slot
    return Build(
        "lcg",
        release,
        name,
        generated.group(1).decode(errors="replace"),
        {
            "setup": _sha256(setup),
            "LCG_externals": _digest(manifests / f"LCG_externals_{name}.txt"),
            "LCG_contrib": _digest(manifests / f"LCG_contrib_{name}.txt"),
        },
    )


@dataclass(frozen=True, slots=True)
class System:
    """What of the machine a program depends on: the ``os`` with its major version, the ``glibc``
    version and the ``architecture``, each ``None`` where the system does not say."""

    os: str | None
    glibc: str | None
    architecture: str | None

    @classmethod
    def now(cls) -> System:
        """This machine's."""
        try:
            release = platform.freedesktop_os_release()
        except OSError:
            name = None
        else:
            major = release.get("VERSION_ID", "").split(".")[0]
            name = " ".join(filter(None, (release.get("ID"), major))) or None
        library, version = platform.libc_ver()
        glibc = version if library == "glibc" and version else None
        return cls(name, glibc, platform.machine() or None)

    @property
    def label(self) -> str:
        """The system for a person: ``rhel 9, glibc 2.34, x86_64``."""
        glibc = None if self.glibc is None else f"glibc {self.glibc}"
        return ", ".join(part for part in (self.os, glibc, self.architecture) if part) or "unknown"

    def to_dict(self) -> dict[str, object]:
        return {"os": self.os, "glibc": self.glibc, "architecture": self.architecture}

    @classmethod
    def from_dict(cls, data: object) -> System:
        """The system :meth:`to_dict` wrote, checked; raises :class:`SoftwareError`."""
        entry = _keys(data, _SYSTEM, "the system")
        return cls(
            _text(entry["os"], "the system's os"),
            _text(entry["glibc"], "the system's glibc"),
            _text(entry["architecture"], "the system's architecture"),
        )


_SYSTEM = frozenset(("os", "glibc", "architecture"))


@dataclass(frozen=True, slots=True)
class Software:
    """What candidates run on: the ``build``, the ``system`` and the SHA-256 of each compared
    variable by its name (``variables``). Equal :attr:`id`\\ s, the same software."""

    build: Build
    system: System
    variables: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", FrozenMap(self.variables))

    @classmethod
    def now(cls, environment: Mapping[str, str] | None = None) -> Software:
        """What this process would run on now: of ``environment``, ``os.environ`` if ``None``."""
        variables = os.environ if environment is None else environment
        return cls(
            Build.of(variables.get("KEY4HEP_STACK")),
            System.now(),
            {
                name: _sha256(os.fsencode(value))
                for name, value in sorted(variables.items())
                if _compared(name)
            },
        )

    @property
    def id(self) -> str:
        """The SHA-256 of all of it."""
        parts = {
            "build": self.build.to_dict(),
            "system": self.system.to_dict(),
            "variables": dict(self.variables),
        }
        return _sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode())

    def differences(self, now: Software) -> list[str]:
        """What differs in ``now``, each said for a person; empty for the same software."""
        found = []
        if self.build != now.build:
            found.append(_build_difference(self.build, now.build))
        if self.system != now.system:
            found.append(f"the system is {now.system.label}, not {self.system.label}")
        names = sorted({*self.variables, *now.variables})
        changed = [n for n in names if self.variables.get(n) != now.variables.get(n)]
        if changed:
            found.append(f"the environment differs in {', '.join(changed)}")
        return found

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "build": self.build.to_dict(),
            "system": self.system.to_dict(),
            "variables": dict(self.variables),
        }

    @classmethod
    def from_dict(cls, data: object) -> Software:
        """The software :meth:`to_dict` wrote, checked: its ``id`` is that of its parts. Raises
        :class:`SoftwareError`."""
        entry = _keys(data, _SOFTWARE, "the software")
        variables = entry["variables"]
        if not isinstance(variables, Mapping) or not all(
            isinstance(value, str) and _SHA256.fullmatch(value) for value in variables.values()
        ):
            raise SoftwareError("the software's variables are not digests by name")
        software = cls(
            Build.from_dict(entry["build"]), System.from_dict(entry["system"]), dict(variables)
        )
        if entry["id"] != software.id:
            raise SoftwareError(f"the software's id {entry['id']!r} is not that of its parts")
        return software


_SOFTWARE = frozenset(("id", "build", "system", "variables"))


def _build_difference(then: Build, now: Build) -> str:
    if then.label != now.label:
        return f"the Key4hep build is {now.label}, not {then.label}"
    roles = dict.fromkeys([*then.digests, *now.digests])
    changed = [role for role in roles if then.digests.get(role) != now.digests.get(role)]
    return f"the Key4hep build {now.label} is another one (other {', '.join(changed)})"
