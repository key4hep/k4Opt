"""Local builds as they are made for tests of where they come from: a git checkout with a remote, a
CMake build tree and the installation ``k4_local_repo`` puts in front of the stack, whose libraries
are tiny ELF files with a GNU build-id."""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

GIT = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.org",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.org",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


def git(checkout: Path, *args: str) -> str:
    """The output of ``git args`` in ``checkout``, as a test runs it (its own identity, no user
    configuration)."""
    done = subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **GIT},
    )
    return done.stdout.strip()


def elf(build_id: str, payload: bytes = b"") -> bytes:
    """A minimal 64-bit ELF file whose one note is the GNU build-id ``build_id`` (hex), followed by
    ``payload``: what the linker writes, and an installed copy whose bytes differ (its run path)
    though its build-id does not."""
    identity = bytes.fromhex(build_id)
    note = struct.pack("<III", 4, len(identity), 3) + b"GNU\0" + identity
    note += b"\0" * (-len(note) % 4)
    header = struct.pack(
        "<4sBBBB8xHHIQQQIHHHHHH",
        b"\x7fELF",
        2,
        1,
        1,
        0,
        3,
        0x3E,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        1,
        0,
        0,
        0,
    )
    program = struct.pack("<IIQQQQQQ", 4, 4, 120, 0, 0, len(note), len(note), 4)
    return header + program + note + payload


def checkout(root: Path, name: str = "DDMarlinPandora") -> Path:
    """A git checkout ``root/name`` with one commit, cloned from a bare ``origin`` so that a remote
    has it, ignoring ``build/``."""
    origin = root / f"{name}.git"
    seed = root / f"{name}-seed"
    seed.mkdir(parents=True)
    (seed / "src").mkdir()
    (seed / "src" / "DDCaloHitCreator.cc").write_text("energy = eCalToEMGeV * hit;\n")
    (seed / "README.md").write_text("DDMarlinPandora\n")
    (seed / ".gitignore").write_text("build/\n*.log\n")
    git(seed, "init", "-q", "-b", "master")
    git(seed, "add", ".")
    git(seed, "commit", "-q", "-m", "first")
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(root / name)], check=True)
    return root / name


def build(repo: Path, build_id: str = "9854a49d", installed: str | None = None) -> Path:
    """A CMake build tree ``repo/build`` installing into ``repo/install`` (``k4_local_repo``'s
    layout), with ``libDDMarlinPandora.so`` in both; the installed one has the build-id
    ``installed`` (``build_id`` if ``None``) and other bytes. The installation prefix."""
    tree, prefix = repo / "build", repo / "install"
    (tree / "lib").mkdir(parents=True)
    (tree / "CMakeCache.txt").write_text(
        f"CMAKE_BUILD_TYPE:STRING=RelWithDebInfo\n"
        f"CMAKE_INSTALL_PREFIX:PATH={prefix}\n"
        f"CMAKE_HOME_DIRECTORY:INTERNAL={repo}\n"
    )
    (prefix / "lib").mkdir(parents=True)
    (prefix / "include").mkdir()
    (prefix / "include" / "DDCaloHitCreator.h").write_text("class DDCaloHitCreator;\n")
    name = "libDDMarlinPandora.so.0.14.0"
    (tree / "lib" / name).write_bytes(elf(build_id, b"RPATH=build"))
    (prefix / "lib" / name).write_bytes(elf(installed or build_id, b"RPATH=install"))
    for directory in (tree / "lib", prefix / "lib"):
        (directory / "libDDMarlinPandora.so").symlink_to(name)
    return prefix


def variables(prefix: Path) -> dict[str, str]:
    """What ``k4_local_repo`` puts in front of the stack for the build at ``prefix``, with
    ``MARLIN_DLL`` set by hand to its library."""
    return {
        "LD_LIBRARY_PATH": f"{prefix / 'lib'}:/cvmfs/stack/lib",
        "ROOT_INCLUDE_PATH": f"{prefix / 'include'}",
        "MARLIN_DLL": f"{prefix / 'lib' / 'libDDMarlinPandora.so'}",
    }
