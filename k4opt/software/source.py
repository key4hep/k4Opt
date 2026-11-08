"""Where local changes come from: each local build linked to its source, and that source's state.

The software identifies a local build by what can be loaded through it (:mod:`.overlays`); a rerun
needs more: what it was built from. :func:`origins` groups the local entries of the software into
their *origins* (:class:`Origin`), each of one of three kinds:

- ``build``: an installation that CMake made (``<checkout>/install`` from ``k4_local_repo``): the
  entries below one prefix (``lib``, ``include``, ``python``, ``lib/python3.14/site-packages``, a
  ``MARLIN_DLL`` library), linked to the checkout it was built from (:data:`Link`);
- ``directory``: local code that no build made (a helper directory on ``PYTHONPATH``, an
  editable install), in a git checkout or not;
- ``environment``: a ``site-packages`` directory of installed distributions (a virtual
  environment, the user's), known by their names and versions.

Two facts of a build are kept apart, since a patch does not rebuild the same binary (the same
source built in another directory has another build-id and other bytes): its *source*, the
complete state of the checkout (:class:`GitState`: ``HEAD``, staged, unstaged and untracked files,
ignored ones left out) diffed against the revision the nightly built the package from, and its
*binaries*, each by its SHA-256 and GNU build-id (:func:`build_id`). Nothing is written into a
checkout: the state is a git tree made with a temporary index and a temporary object store
that reads the checkout's objects through git's alternates.

The link of a build to its checkout is evidence, never proof: ``verified`` when the CMake cache of
a build tree installs into the prefix and the binaries of the build tree have the build-ids of
those installed; ``likely`` when the cache installs there but the binaries cannot be compared or
differ (the tree was rebuilt since); ``adjacent`` when the prefix lies in a checkout and no build
tree says it installs there; ``none`` otherwise. Which package of the stack a build stands in for
is found by ``k4_local_repo``'s own rule (the checkout's name, lower case, without hyphens), or by
the libraries it installs that a ``HEAD`` package of the release also installs.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from k4opt.hashing import sha256_of, stat_of
from k4opt.software.overlays import _LISTED, ROLES, SITE, Local, _file
from k4opt.software.stack import Package, Release

type Kind = Literal["build", "directory", "environment"]
type Link = Literal["verified", "likely", "adjacent", "none"]
type Base = Literal["nightly", "public", "none"]

LARGE = 10 << 20
"""An untracked file larger than this is listed, not part of the kept state."""
MAX_PATCH = 100 << 20
"""A diff larger than this is not kept: the state is recorded, its content is not."""

_SUFFIXES = ("lib/gaudi-plugins", "lib64/gaudi-plugins", "lib", "lib64", "include", "python")
_SITE = re.compile(r"/lib(?:64)?/python\d+\.\d+/(?:site|dist)-packages$")
_LIBRARY = re.compile(r"\.so(\.[\w.]*)?$|\.pcm$")
_CACHE = "CMakeCache.txt"
_SKIPPED = frozenset(("CMakeFiles", ".git", "__pycache__"))
_WALKED = 50_000
"""How many entries a look for a build tree's binaries reads at most."""
_TIMEOUT = 120


@dataclass(frozen=True, slots=True)
class Binary:
    """A file of a build that the loader loads: its ``file`` relative to the build's prefix, its
    SHA-256 and its GNU build-id (``None`` if it has none)."""

    file: str
    sha256: str
    build_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {"file": self.file, "sha256": self.sha256, "build_id": self.build_id}


@dataclass(frozen=True, slots=True)
class GitState:
    """The complete state of a git ``checkout``: its ``head``, the commit the diff starts from
    (``base``: the revision the nightly built, ``nightly``; a commit a remote has, ``public``; or
    none, ``none``, when no remote has any of its history, so the diff holds every file), how many
    commits ``HEAD`` is ``ahead`` of it, the ``uncommitted`` changes (``files``, ``insertions``,
    ``deletions`` against ``HEAD``), how many ``untracked`` files it holds and which are too
    ``large`` to keep (``ignored``: those git ignores, left out of the state), the git ``tree`` of
    the whole state, the files that differ from the base
    (``changed``), and the ``patch`` that makes the state from the base (``git apply``; ``None``
    if too large). ``fetch`` says that the nightly's revision is not in the checkout, so the diff
    against it needs a fetch; ``remote`` is where a remote that holds the base is (``origin`` if it
    does; ``origin`` too if none is seen to), without credentials, ``None`` if none is known."""

    checkout: str
    remote: str | None
    head: str | None
    base: str | None
    base_is: Base
    fetch: bool
    ahead: int
    uncommitted: Mapping[str, int]
    untracked: int
    large: tuple[str, ...]
    tree: str
    changed: tuple[str, ...]
    patch: bytes | None = field(default=None, repr=False)
    ignored: frozenset[str] = field(default=frozenset(), repr=False)

    @property
    def clean(self) -> bool:
        """Whether nothing differs from the base: a rerun needs only the base commit."""
        return not self.changed

    def to_dict(self, patch: str | None = None) -> dict[str, object]:
        """As JSON data; the patch by ``patch``, the digest it is kept by."""
        return {
            "checkout": self.checkout,
            "remote": self.remote,
            "head": self.head,
            "base": self.base,
            "base_is": self.base_is,
            "fetch": self.fetch,
            "ahead": self.ahead,
            "uncommitted": dict(self.uncommitted),
            "untracked": self.untracked,
            "large": list(self.large),
            "tree": self.tree,
            "changed": list(self.changed),
            "patch": patch,
        }


@dataclass(frozen=True, slots=True)
class Origin:
    """Where some local entries of the software come from (see the module).

    ``prefix`` is the build's installation, or the directory itself; ``entries`` the paths of the
    local entries it holds (:class:`~k4opt.software.Local`). ``package`` is the package of the
    release it stands in for and ``revision`` the commit the nightly built that package from;
    ``link`` how sure its checkout is (``evidence`` says why), ``build`` the build tree,
    ``binaries``
    what it installed for the loader, ``stale`` whether files of its source changed after its
    newest binary was made (``None`` where that is not known). ``git`` is the state of its
    checkout, if it is in one; ``files`` the files of a directory in none, by their path relative
    to it (their path and SHA-256), which is all a rerun would have of it; ``distributions`` the
    ``name==version`` of each distribution an environment holds; for a directory of no build,
    ``binaries`` are those it holds, identified only (their source is not known). For a build,
    ``files`` are what else it installed that can be loaded (Python modules, headers, registries),
    by their path relative to its prefix: what ran, whatever its source would install now. For a
    directory in a checkout, ``files`` and ``binaries`` are what can be loaded that git ignores.
    ``problem`` says what could not be read; ``configuration`` is the CMake cache a build was
    configured with (its options, compiler and flags), which a rebuild needs as much as its
    source."""

    kind: Kind
    prefix: str
    entries: tuple[str, ...]
    package: str | None = None
    revision: str | None = None
    link: Link = "none"
    evidence: tuple[str, ...] = ()
    build: str | None = None
    binaries: tuple[Binary, ...] = ()
    stale: bool | None = None
    git: GitState | None = None
    files: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    distributions: tuple[str, ...] = ()
    problem: str | None = None
    configuration: str | None = None

    @property
    def rebuildable(self) -> bool:
        """Whether its source is known well enough to build it again: a build whose state is kept
        (its patch, or nothing to patch) against a base that a known remote can give, or whole if
        no remote has any of its history, linked ``verified`` or ``likely``,
        and not changed after its binaries were built (``stale``): then the state kept is not
        what they were built from, and a rebuild from it could do something else; nor with an
        untracked file too large to keep, which the build may need."""
        git = self.git
        return (
            self.kind == "build"
            and self.stale is not True
            and self.link in ("verified", "likely")
            and git is not None
            and (git.base_is == "none" or (git.base is not None and git.remote is not None))
            and not git.large  # a file of it not kept: what is kept is not all of it
            and (git.clean or git.patch is not None)
        )


def origins(local: Sequence[Local], release: Release | None) -> list[Origin]:
    """Where the ``local`` entries of the software come from, each entry in one origin, in the
    order of their first entry; ``release`` is the stack's, to say which package a build stands
    in for (``None`` for an unrecognized stack)."""
    groups: dict[str, list[Local]] = {}
    alone: list[Local] = []
    for entry in local:
        prefix = _prefix(entry)
        if prefix is None:
            alone.append(entry)
        else:
            groups.setdefault(prefix, []).append(entry)
    found: list[Origin] = []
    for prefix, entries in groups.items():
        built = _build(prefix, entries, release)
        if built is None:
            alone += entries
        else:
            found.append(built)
    seen: set[str] = set()
    for entry in alone:
        if entry.path not in seen:
            seen.add(entry.path)
            found.append(_directory(entry, [e for e in alone if e.path == entry.path], release))
    order = {entry.path: index for index, entry in enumerate(local)}
    return sorted(found, key=lambda origin: min(order[path] for path in origin.entries))


def _role(entry: Local) -> str | None:
    return "python" if entry.variable == SITE else ROLES.get(entry.variable)


def _prefix(entry: Local) -> str | None:
    """The installation prefix ``entry`` would be a part of, ``None`` if it names no part of one."""
    path = entry.path.rstrip("/")
    if os.path.isfile(path):
        path = os.path.dirname(path)  # MARLIN_DLL=<prefix>/lib/libDDMarlinPandora.so
    site = _SITE.search(path)
    if site is not None:
        return path[: site.start()] or None
    for suffix in _SUFFIXES:
        if path.endswith(f"/{suffix}"):
            return path[: -len(suffix) - 1] or None
    return None


def _build(prefix: str, entries: list[Local], release: Release | None) -> Origin | None:
    """The build installed in ``prefix``: one that a CMake cache installs there, or that lies in
    the root of a git checkout (``k4_local_repo``'s layout); ``None`` for anything else."""
    paths = tuple(dict.fromkeys(entry.path for entry in entries))
    if os.path.isfile(os.path.join(prefix, "pyvenv.cfg")):
        return None  # a virtual environment: its site-packages are an environment
    cache = _cache_for(prefix)
    checkout: str | None
    if cache is not None:
        build, settings = cache
        home = settings.get("CMAKE_HOME_DIRECTORY")
        checkout = _toplevel(home) if home else None
        evidence = [f"{build}/{_CACHE} installs into {prefix}"]
    else:
        parent = os.path.dirname(prefix)
        if _toplevel(parent) != os.path.realpath(parent):
            return None
        build, checkout = None, parent
        evidence = [f"{prefix} lies in the git checkout {parent}, and no build tree installs there"]
    binaries, built = _binaries(prefix, entries)
    files = {
        os.path.relpath(path, prefix): (path, sha256)
        for entry in entries
        for path, sha256 in _files(entry).values()
    }
    link: Link = "adjacent"
    if build is not None:
        compared = _compared(build, prefix, binaries)
        absent = {os.path.basename(b.file) for b in binaries if b.build_id} - compared.keys()
        link = "verified" if compared and all(compared.values()) and not absent else "likely"
        if absent:
            evidence.append(f"{', '.join(sorted(absent))} not found in the build tree")
        if compared and not all(compared.values()):
            differ = sorted(name for name, same in compared.items() if not same)
            evidence.append(
                f"{', '.join(differ)} in the build tree differ from those installed (rebuilt since)"
            )
        elif compared:
            evidence.append(
                f"the build tree's {', '.join(sorted(compared))} have the same build-id"
            )
        else:
            evidence.append("no binary with a build-id to compare with the build tree")
    package = _package(checkout, binaries, release)
    excluded = [prefix, *([build] if build else []), *(_caches(checkout) if checkout else [])]
    git, problem = _state(checkout, package.revision if package else None, excluded)
    stale = None
    if git is not None and built is not None:
        stale = _changed_after(git, built)
    return Origin(
        "build",
        prefix,
        paths,
        package.name if package else None,
        package.revision if package else None,
        link,
        tuple(evidence),
        build,
        binaries,
        stale,
        git,
        files=files,
        problem=problem,
        configuration=os.path.join(build, _CACHE) if build else None,
    )


def _directory(entry: Local, entries: list[Local], release: Release | None) -> Origin:
    """Local code no build made: in a git checkout (its state), or not (its files); or a
    ``site-packages`` directory (its distributions)."""
    path = entry.path
    paths = tuple(dict.fromkeys(e.path for e in entries))
    if _SITE.search(path.rstrip("/")) or os.path.basename(path.rstrip("/")) in (
        "site-packages",
        "dist-packages",
    ):
        return Origin("environment", path, paths, distributions=_distributions(path))
    checkout = _toplevel(path if os.path.isdir(path) else os.path.dirname(path))
    if checkout is not None and not _ignored(checkout, path):
        package = release.package(os.path.basename(checkout)) if release else None
        git, problem = _state(checkout, package.revision if package else None, [])
        kept, compiled = _left_out(git, entries)
        return Origin(
            "directory",
            path,
            paths,
            package.name if package else None,
            package.revision if package else None,
            binaries=compiled,
            git=git,
            files=kept,
            problem=problem,
        )
    files: dict[str, tuple[str, str]] = {}
    binaries: dict[str, Binary] = {}
    for each in entries:  # what each variable loads from it: Python's modules, the libraries
        files |= _files(each)
        binaries |= {binary.file: binary for binary in _loose(each)}
    ordered = tuple(sorted(binaries.values(), key=lambda b: b.file))
    return Origin("directory", path, paths, binaries=ordered, files=files)


def _left_out(
    git: GitState | None, entries: Iterable[Local]
) -> tuple[dict[str, tuple[str, str]], tuple[Binary, ...]]:
    """What can be loaded through ``entries`` that the state ``git`` of their checkout leaves out,
    being ignored: the files, and the binaries, which are identified only."""
    if git is None or not git.ignored:
        return {}, ()

    def ignored(path: str) -> bool:
        return os.path.relpath(os.path.realpath(path), git.checkout) in git.ignored

    files: dict[str, tuple[str, str]] = {}
    binaries: dict[str, Binary] = {}
    for each in entries:
        files |= {name: file for name, file in _files(each).items() if ignored(file[0])}
        base = each.path if os.path.isdir(each.path) else os.path.dirname(each.path)
        binaries |= {b.file: b for b in _loose(each) if ignored(os.path.join(base, b.file))}
    return files, tuple(sorted(binaries.values(), key=lambda b: b.file))


def checkout(path: str, release: Release | None) -> Origin | None:
    """The origin of the code in ``path`` (k4opt's own package) as a ``directory`` in the git
    checkout it lies in, ``None`` if it lies in none."""
    top = _toplevel(path if os.path.isdir(path) else os.path.dirname(path))
    if top is None or _ignored(top, path):
        return None
    package = release.package(os.path.basename(top)) if release else None
    git, problem = _state(top, package.revision if package else None, [])
    return Origin(
        "directory",
        top,
        (),
        package.name if package else None,
        package.revision if package else None,
        git=git,
        problem=problem,
    )


# --- builds ----------------------------------------------------------------------------------


def _settings(cache: str) -> dict[str, str]:
    """The entries of a CMake cache that say where it installs and what it builds."""
    wanted = ("CMAKE_INSTALL_PREFIX", "CMAKE_HOME_DIRECTORY", "CMAKE_BUILD_TYPE")
    found: dict[str, str] = {}
    try:
        with open(cache, encoding="utf-8", errors="replace") as lines:
            for line in lines:
                name, _, value = line.rstrip("\n").partition("=")
                key = name.partition(":")[0]
                if key in wanted:
                    found[key] = value
    except OSError:
        return {}
    return found


def _caches(root: str, depth: int = 2) -> list[str]:
    """The build trees in ``root`` and up to ``depth`` levels below it: directories with a CMake
    cache."""
    found: list[str] = []
    level = [root]
    for _ in range(depth + 1):
        below: list[str] = []
        for directory in level:
            if os.path.isfile(os.path.join(directory, _CACHE)):
                found.append(directory)
                continue
            try:
                with os.scandir(directory) as entries:
                    below += [
                        e.path
                        for e in entries
                        if e.name not in _SKIPPED and e.is_dir(follow_symlinks=False)
                    ]
            except OSError:
                continue
        level = below
    return found


def _cache_for(prefix: str) -> tuple[str, dict[str, str]] | None:
    """The build tree whose CMake cache installs into ``prefix``, with its settings: looked for in
    the checkout around the prefix and beside the prefix."""
    real = os.path.realpath(prefix)
    roots = dict.fromkeys(
        root for root in (_toplevel(os.path.dirname(prefix)), os.path.dirname(prefix)) if root
    )
    for root in roots:
        for build in _caches(root):
            settings = _settings(os.path.join(build, _CACHE))
            installs = settings.get("CMAKE_INSTALL_PREFIX")
            if installs and os.path.realpath(os.path.join(build, installs)) == real:
                return build, settings
    return None


def _binaries(prefix: str, entries: Iterable[Local]) -> tuple[tuple[Binary, ...], int | None]:
    """What the build installed for the loader through ``entries`` (the libraries of its library
    directories, a ``MARLIN_DLL`` file), and when the newest of them was made."""
    found: dict[str, Binary] = {}
    newest: int | None = None
    for entry in entries:
        role = _role(entry)
        if role == "files" and os.path.isfile(entry.path):
            files = [entry.path]
        elif role == "libraries" and os.path.isdir(entry.path):
            files = sorted(
                os.path.join(entry.path, name)
                for name in _names(entry.path)
                if _LIBRARY.search(name)
            )
        elif role == "python" and os.path.isdir(entry.path):  # a compiled extension module
            listed = _LISTED["python"](entry.path, "", frozenset(), [])
            files = sorted(path for _, path, status in listed if status and compiled(path))
        else:
            continue
        for path in files:
            real = os.path.realpath(path)  # libX.so -> libX.so.0.14.0: one binary, by its file
            status = stat_of(real)
            if status is None or real in found:
                continue
            inside = real.startswith(f"{os.path.realpath(prefix)}/")
            relative = os.path.relpath(real, os.path.realpath(prefix)) if inside else path
            try:
                digest = sha256_of(real, status)
            except OSError:
                continue
            found[real] = Binary(relative, digest, build_id(real))
            newest = status[1] if newest is None else max(newest, status[1])
    return tuple(sorted(found.values(), key=lambda b: b.file)), newest


def _compared(build: str, prefix: str, binaries: Sequence[Binary]) -> dict[str, bool]:
    """Each installed binary with a build-id that the build tree holds a file of the same name
    of, by its name: whether the two have the same build-id."""
    ids = {os.path.basename(b.file): b.build_id for b in binaries if b.build_id}
    real_prefix = os.path.realpath(prefix)
    compared: dict[str, bool] = {}
    walked = 0
    for directory, names, files in os.walk(build):
        names[:] = [n for n in names if n not in _SKIPPED]
        if os.path.realpath(directory).startswith(real_prefix):
            names[:] = []
            continue
        walked += len(files) + len(names)
        for name in files:
            if name in ids and not os.path.islink(os.path.join(directory, name)):
                same = build_id(os.path.join(directory, name)) == ids[name]
                compared[name] = compared.get(name, True) and same
        if walked > _WALKED:
            break
    return compared


def _package(
    checkout: str | None, binaries: Sequence[Binary], release: Release | None
) -> Package | None:
    """The package of ``release`` a build stands in for: by its checkout's name, else the one
    ``HEAD`` package that installs libraries of the same names."""
    if release is None:
        return None
    if checkout is not None:
        named = release.package(os.path.basename(checkout))
        if named is not None:
            return named
    names = {os.path.basename(b.file) for b in binaries}
    if not names:
        return None
    matched = []
    for package in release.packages.values():
        if package.version != "HEAD":
            continue
        provided = set(_names(os.path.join(package.prefix, "lib"))) | set(
            _names(os.path.join(package.prefix, "lib64"))
        )
        if names & provided:
            matched.append(package)
    return matched[0] if len(matched) == 1 else None


def _changed_after(git: GitState, built: int) -> bool:
    """Whether a file of the checkout that differs from the base was changed after ``built``. A
    file no longer there is judged by the directory it was in, whose time its removal set (or the
    nearest one left): when it went is not known otherwise, so a later change there counts."""
    for name in git.changed:
        path = os.path.join(git.checkout, name)
        status = stat_of(path)
        while status is None and path != git.checkout:
            path = os.path.dirname(path)
            status = stat_of(path)
        if status is not None and status[1] > built:
            return True
    return False


# --- directories -----------------------------------------------------------------------------


def _names(directory: str) -> list[str]:
    try:
        return os.listdir(directory)
    except OSError:
        return []


def _distributions(directory: str) -> tuple[str, ...]:
    """The ``name==version`` of each distribution installed in ``directory``, by its metadata
    directory's name (``numpy-2.5.4.dist-info``)."""
    found = []
    for name in _names(directory):
        for suffix in (".dist-info", ".egg-info"):
            if name.endswith(suffix):
                base, _, version = name[: -len(suffix)].rpartition("-")
                if base:
                    found.append(f"{base}=={version}")
    return tuple(sorted(found, key=str.lower))


def _elf(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def compiled(path: str) -> bool:
    """Whether ``path`` is compiled: fingerprinted, never kept (an ELF file, a ROOT module)."""
    return path.endswith(".pcm") or _elf(path)


def _files(entry: Local) -> dict[str, tuple[str, str]]:
    """What can be loaded through ``entry`` (as the software lists it) but binaries, by its path
    relative to it: each file's path and SHA-256."""
    role = _role(entry) or "files"
    if os.path.isfile(entry.path):
        listed = _file(entry.path)
    else:
        listed = _LISTED[role](entry.path, "", frozenset(), [])
    found: dict[str, tuple[str, str]] = {}
    for relative, path, status in listed:
        if status is None:
            continue
        if compiled(path):  # identified among the binaries, never kept
            continue
        try:
            found[relative or os.path.basename(path)] = (path, sha256_of(path, status))
        except OSError:
            continue
    return found


def _loose(entry: Local) -> list[Binary]:
    """The binaries that can be loaded through ``entry``, a local entry of no build: each by its
    SHA-256 and build-id, as a build's are."""
    role = _role(entry) or "files"
    listed = (
        _file(entry.path)
        if os.path.isfile(entry.path)
        else _LISTED[role](entry.path, "", frozenset(), [])
    )
    found = []
    for relative, path, status in listed:
        if status is None or not compiled(path):
            continue
        try:
            found.append(
                Binary(relative or os.path.basename(path), sha256_of(path, status), build_id(path))
            )
        except OSError:
            continue
    return found


# --- git -------------------------------------------------------------------------------------


def _environment(**extra: str) -> dict[str, str]:
    """The environment git runs in: none of the caller's ``GIT_*`` (which would point it at
    another repository or index), no prompts, no optional locks, C messages."""
    kept = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    return {**kept, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **extra}


def _git(
    cwd: str,
    *args: str,
    env: Mapping[str, str] | None = None,
    stdin: bytes | None = None,
) -> bytes | None:
    """The output of ``git args`` in ``cwd``, ``None`` if it failed. Never writes into the
    checkout: no maintenance, no file system monitor, no untracked cache."""
    git = shutil.which("git")
    if git is None:
        return None
    command = [git, "-c", "gc.auto=0", "-c", "core.fsmonitor=false", "-c", "maintenance.auto=0"]
    command += ["-c", "core.untrackedCache=false", "-C", cwd, *args]
    try:
        done = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            env=dict(env) if env is not None else _environment(),
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _text(output: bytes | None) -> str | None:
    return None if output is None else output.decode(errors="replace").strip() or None


def _toplevel(path: str) -> str | None:
    """The real path of the git checkout ``path`` lies in, ``None`` if none."""
    if not os.path.isdir(path):
        return None
    top = _text(_git(path, "rev-parse", "--show-toplevel"))
    return None if top is None else os.path.realpath(top)


def _ignored(checkout: str, path: str) -> bool:
    """Whether the checkout ignores ``path`` (a virtual environment inside it)."""
    relative = os.path.relpath(os.path.realpath(path), checkout)
    return _git(checkout, "check-ignore", "-q", "--no-index", "--", relative) is not None


def _commit(checkout: str, revision: str) -> str | None:
    return _text(_git(checkout, "rev-parse", "--verify", "-q", f"{revision}^{{commit}}"))


def _public(checkout: str, head: str) -> str | None:
    """The newest commit of ``HEAD``'s history that a remote has (``HEAD`` itself if a remote has
    it), ``None`` if no remote has any."""
    listed = _git(checkout, "rev-list", "--boundary", head, "--not", "--remotes")
    if listed is None:
        return None
    lines = listed.decode().split()
    if not lines:
        return head if _git(checkout, "for-each-ref", "--count=1", "refs/remotes") else None
    boundary = [line[1:] for line in lines if line.startswith("-")]
    return boundary[0] if boundary else None


def _remote(checkout: str, base: str | None) -> str | None:
    """Where a remote holding the commit ``base`` is, without credentials: ``origin`` if it holds
    it or none is seen to, else the first that does; ``None`` if that gives no remote."""
    names = (_text(_git(checkout, "remote")) or "").split()
    listed = base and _git(
        checkout, "for-each-ref", "--contains", base, "--format=%(refname)", "refs/remotes"
    )
    refs = listed.decode(errors="replace").split() if listed else []
    holding = [n for n in names if any(r.startswith(f"refs/remotes/{n}/") for r in refs)]
    chosen = "origin" if "origin" in holding or not holding else sorted(holding)[0]
    url = _text(_git(checkout, "remote", "get-url", chosen))
    return None if url is None else public_url(url)


def public_url(url: str) -> str:
    """``url`` without what authenticates: a password, and over HTTP(S) any user part (a token is
    often written there); an SSH user (``ssh://git@host``, ``git@host:org/repo``) is kept, since
    the server needs it and it is no secret."""
    parts = urllib.parse.urlsplit(url)
    if not parts.scheme or "@" not in parts.netloc:
        return url  # git@host:org/repo, a path
    user, _, host = parts.netloc.rpartition("@")
    name = user.partition(":")[0]
    keep = name and parts.scheme in ("ssh", "git+ssh", "ssh+git")
    netloc = f"{name}@{host}" if keep else host
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc))


def _numstat(checkout: str, env: Mapping[str, str], old: str, new: str) -> dict[str, int]:
    out = _git(checkout, "diff", "--numstat", "-z", "--no-renames", old, new, env=env)
    if out is None:
        raise _GitError("its changes cannot be counted")
    files = insertions = deletions = 0
    for record in out.split(b"\0"):
        parts = record.split(b"\t")
        if len(parts) >= 3:
            files += 1
            insertions += int(parts[0]) if parts[0].isdigit() else 0
            deletions += int(parts[1]) if parts[1].isdigit() else 0
    return {"files": files, "insertions": insertions, "deletions": deletions}


def _state(
    checkout: str | None, revision: str | None, excluded: Sequence[str]
) -> tuple[GitState | None, str | None]:
    """The state of ``checkout`` against the nightly's ``revision`` of it (or its newest public
    commit), the ``excluded`` directories (its installation, its build trees) left out; and what
    could not be read."""
    if checkout is None:
        return None, None
    try:
        return _snapshot(checkout, revision, excluded), None
    except _GitError as exc:
        return None, f"{checkout}: {exc}"


class _GitError(Exception):
    pass


def _snapshot(checkout: str, revision: str | None, excluded: Sequence[str]) -> GitState:
    objects = _text(_git(checkout, "rev-parse", "--path-format=absolute", "--git-path", "objects"))
    if objects is None:
        raise _GitError("git cannot read the checkout")
    head = _text(_git(checkout, "rev-parse", "--verify", "-q", "HEAD^{commit}"))
    real = os.path.realpath(checkout)
    skipped = [
        os.path.relpath(os.path.realpath(path), real)
        for path in excluded
        if os.path.realpath(path).startswith(f"{real}/")
    ]
    with tempfile.TemporaryDirectory(prefix="k4opt-state-") as scratch:
        env = _environment(
            GIT_INDEX_FILE=os.path.join(scratch, "index"),
            GIT_OBJECT_DIRECTORY=os.path.join(scratch, "objects"),
            GIT_ALTERNATE_OBJECT_DIRECTORIES=objects,
        )
        os.makedirs(os.path.join(scratch, "objects"))
        start = ["read-tree", head] if head else ["read-tree", "--empty"]
        if _git(real, *start, env=env) is None:
            raise _GitError("its HEAD cannot be read into an index")
        if head and _git(real, "add", "--update", "--", ".", env=env) is None:
            raise _GitError("its changes cannot be read")
        listed = _git(real, "ls-files", "-z", "--others", "--exclude-standard", env=env)
        if listed is None:
            raise _GitError("its untracked files cannot be listed")
        untracked = [name.decode(errors="surrogateescape") for name in listed.split(b"\0") if name]
        untracked = [
            name
            for name in untracked
            if not any(name == s or name.startswith(f"{s}/") for s in skipped)
        ]
        left = _git(real, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", env=env)
        if left is None:
            raise _GitError("its ignored files cannot be listed")
        large = [n for n in untracked if (stat_of(os.path.join(real, n)) or (0,))[0] > LARGE]
        kept = [n for n in untracked if n not in large]
        if kept:
            names = b"\0".join(os.fsencode(n) for n in kept)
            added = _git(
                real, "add", "--pathspec-from-file=-", "--pathspec-file-nul", env=env, stdin=names
            )
            if added is None:
                raise _GitError("its untracked files cannot be read")
        tree = _text(_git(real, "write-tree", env=env))
        if tree is None:
            raise _GitError("its state cannot be written as a tree")
        nightly = _commit(real, revision) if revision else None
        public = _public(real, head) if head else None
        base = nightly or public
        base_is: Base = "nightly" if nightly else "public" if public else "none"
        empty = _text(_git(real, "hash-object", "-t", "tree", "/dev/null")) or ""
        start_from = base or empty
        changed = _git(real, "diff", "--name-only", "-z", "--no-renames", start_from, tree, env=env)
        options = ("--binary", "--full-index", "--no-renames", "--no-color", "--no-ext-diff")
        patch = _git(real, "diff", *options, "--no-textconv", start_from, tree, env=env)
        if changed is None or patch is None:  # a failure to look is no evidence of nothing changed
            raise _GitError("its changes cannot be diffed against its base")
        ahead = 0
        if head and base and base != head:
            counted = _text(_git(real, "rev-list", "--count", f"{base}..{head}"))
            if counted is None or not counted.isdigit():
                raise _GitError("its commits cannot be counted")
            ahead = int(counted)
        uncommitted = _numstat(real, env, head or empty, tree)
    return GitState(
        real,
        _remote(real, base),
        head,
        base,
        base_is,
        bool(revision) and nightly is None,
        ahead,
        uncommitted,
        len(untracked),
        tuple(large),
        tree,
        tuple(n.decode(errors="surrogateescape") for n in changed.split(b"\0") if n),
        patch if len(patch) <= MAX_PATCH else None,
        frozenset(n.decode(errors="surrogateescape") for n in left.split(b"\0") if n),
    )


# --- ELF -------------------------------------------------------------------------------------


def build_id(path: str | os.PathLike[str]) -> str | None:
    """The GNU build-id of the ELF file ``path`` (hex), from its notes; ``None`` if it is no ELF
    file or has none. The linker writes it from the linked bytes, before installing rewrites the
    run path, so a build tree's library and its installed copy share it though their bytes
    differ."""
    try:
        with open(path, "rb") as elf:
            head = elf.read(64)
            if len(head) < 52 or head[:4] != b"\x7fELF" or head[4] not in (1, 2):
                return None
            wide, order = head[4] == 2, "<" if head[5] == 1 else ">"
            if wide:
                (phoff,) = struct.unpack_from(f"{order}Q", head, 0x20)
                size, count = struct.unpack_from(f"{order}HH", head, 0x36)
            else:
                (phoff,) = struct.unpack_from(f"{order}I", head, 0x1C)
                size, count = struct.unpack_from(f"{order}HH", head, 0x2A)
            for index in range(min(count, 256)):
                elf.seek(phoff + index * size)
                header = elf.read(size)
                if len(header) < (56 if wide else 32):
                    return None
                (kind,) = struct.unpack_from(f"{order}I", header, 0)
                if kind != 4:  # PT_NOTE
                    continue
                word = f"{order}Q" if wide else f"{order}I"
                # p_offset and p_filesz: after p_flags in ELF64, before p_vaddr in ELF32
                ((offset,), (filesz,)) = (
                    struct.unpack_from(word, header, 8 if wide else 4),
                    struct.unpack_from(word, header, 32 if wide else 16),
                )
                elf.seek(offset)
                found = _note(elf.read(min(filesz, 1 << 16)), order)
                if found is not None:
                    return found
    except (OSError, struct.error):
        return None
    return None


def _note(notes: bytes, order: str) -> str | None:
    at = 0
    while at + 12 <= len(notes):
        namesz, descsz, kind = struct.unpack_from(f"{order}III", notes, at)
        name_at = at + 12
        desc_at = name_at + (namesz + 3) // 4 * 4
        if kind == 3 and notes[name_at : name_at + namesz].rstrip(b"\0") == b"GNU":
            return notes[desc_at : desc_at + descsz].hex() or None
        at = desc_at + (descsz + 3) // 4 * 4
    return None


__all__ = [
    "LARGE",
    "MAX_PATCH",
    "Binary",
    "GitState",
    "Origin",
    "build_id",
    "checkout",
    "origins",
]
