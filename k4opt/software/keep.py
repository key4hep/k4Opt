"""What a study keeps privately of what it ran with, so that a rerun has what is not public.

A study's ``software/`` directory (:data:`DIRECTORY`) holds, for no one but its author:

- ``software-<sha16>.json`` (:data:`SCHEMA`), taken when a session opens (:func:`take`): the
  software it runs on, the stack's release (:mod:`.stack`: its manifests, kept apart, and the
  commit of each package built from a repository's head), where each
  local build comes from and the state of its source (:mod:`.source`), each reconstruction's job
  directory against the package of the stack it is a copy of (its *steering*), and the files of
  the evaluators. One file per state, named by its content: a session that opens on the same state
  adds nothing.
- ``inputs.json`` (:data:`INPUTS`): each file the stages read that neither the stack nor the
  software identifies, from the problem or from the stages' footprints, with its identity, a
  random ``salt`` and the ``commitment`` ``sha256(salt ‖ sha256(file))``, which a publication can
  show without revealing what can be guessed (:func:`commitment`), and its copy, unless too large.
- ``used.json`` (:data:`USED`): the files of local builds each observed stage read, by footprint:
  which binaries of a local build ran.
- ``blobs/sha256/<2>/<62>.gz``: the bytes kept (:class:`Store`), gzipped, each named by the
  SHA-256 of its content, written whole, and checked against it on every read.

Kept: the manifests of the stack's release, the patch of each local checkout against the revision
the nightly built (its untracked files included, those over :data:`~.source.LARGE` listed), the
files of local code in no checkout, the changed files of a job directory, the evaluators' files and
each input up to :data:`MAX_INPUT`, all up to :data:`MAX_KEPT` per study. Never kept: a file of the
stack, a binary (identified by its SHA-256 and build-id only). Keeping never stops a study: what
cannot be kept is recorded with why (``problems``).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import k4opt
from k4opt.jsonio import publish, replace
from k4opt.software.errors import SoftwareError
from k4opt.software.identity import Software, compared_variables
from k4opt.software.observe import STACK, load
from k4opt.software.source import Origin, checkout, origins
from k4opt.software.stack import MANIFESTS, Package, Release

DIRECTORY = "software"
SCHEMA = "k4opt.kept/1"
INPUTS = "inputs.json"
INPUTS_SCHEMA = "k4opt.inputs/1"
USED = "used.json"
USED_SCHEMA = "k4opt.used/1"

MAX_INPUT = 100 << 20
"""The largest input a study keeps a copy of; a larger one is identified and located only."""
MAX_KEPT = 100 << 20
"""How much a study keeps in all (on disk); past it, what comes is identified only."""

type State = Literal["available", "altered", "gone"]

_SHA256 = "sha256:"
_LISTED = "names:"


def commitment(salt: str, sha256: str) -> str:
    """``sha256(salt ‖ sha256(file))`` of the hex ``salt`` and the file's hex SHA-256: it binds a
    published record to the file, and without the salt no one can test a guess of it."""
    return hashlib.sha256(bytes.fromhex(salt) + bytes.fromhex(sha256)).hexdigest()


class Store:
    """Bytes by their SHA-256, gzipped, in ``directory/blobs``: written whole, never over another
    file, and checked against their name on every read (:meth:`get`)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._used: int | None = None

    def path(self, sha256: str) -> Path:
        return self.directory / "blobs" / "sha256" / sha256[:2] / f"{sha256[2:]}.gz"

    def used(self) -> int:
        """The bytes the blobs take on disk."""
        if self._used is None:
            total = 0
            for directory, _, files in os.walk(self.directory / "blobs"):
                for name in files:
                    try:
                        total += os.stat(os.path.join(directory, name)).st_size
                    except OSError:
                        continue
            self._used = total
        return self._used

    def put(self, data: bytes) -> str:
        """Keep ``data``; its SHA-256. Raises :class:`SoftwareError` if it does not fit in
        :data:`MAX_KEPT`, :class:`OSError` if it cannot be written."""
        sha256 = hashlib.sha256(data).hexdigest()
        path = self.path(sha256)
        if path.is_file():
            return sha256
        compressed = gzip.compress(data, mtime=0)
        if self.used() + len(compressed) > MAX_KEPT:
            raise SoftwareError(f"the study keeps {MAX_KEPT >> 20} MiB at most")
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            with staged.open("xb") as handle:
                handle.write(compressed)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(staged, path)
            except FileExistsError:
                return sha256  # kept meanwhile, by name the same bytes
        finally:
            staged.unlink(missing_ok=True)
        self._used = self.used() + len(compressed)
        return sha256

    def put_file(self, path: str | os.PathLike[str], sha256: str | None = None) -> str:
        """Keep the file ``path``, which must be :data:`MAX_INPUT` at most and, if ``sha256`` is
        given, have that SHA-256; its SHA-256. Raises :class:`SoftwareError` otherwise and
        :class:`OSError` if it cannot be read or written."""
        with open(path, "rb") as handle:
            data = handle.read(MAX_INPUT + 1)
        if len(data) > MAX_INPUT:
            raise SoftwareError(f"{os.fspath(path)} is larger than {MAX_INPUT >> 20} MiB")
        digest = hashlib.sha256(data).hexdigest()
        if sha256 is not None and digest != sha256:
            raise SoftwareError(f"{os.fspath(path)} is no longer the file the stage read")
        return self.put(data)

    def get(self, sha256: str) -> bytes:
        """The bytes kept as ``sha256``; raises :class:`SoftwareError` if they are gone or are not
        those bytes any more."""
        try:
            data = gzip.decompress(self.path(sha256).read_bytes())
        except FileNotFoundError:
            raise SoftwareError(f"nothing is kept as {sha256}") from None
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise SoftwareError(f"what is kept as {sha256} cannot be read: {exc}") from exc
        if hashlib.sha256(data).hexdigest() != sha256:
            raise SoftwareError(f"what is kept as {sha256} was altered")
        return data

    def state(self, sha256: str) -> State:
        """Whether the bytes kept as ``sha256`` are there as they were."""
        try:
            self.get(sha256)
        except SoftwareError as exc:
            return "gone" if "nothing is kept" in str(exc) else "altered"
        return "available"


@dataclass(frozen=True, slots=True)
class Taken:
    """What :func:`take` kept: the file it wrote (or found, for a state kept before), and what
    could not be kept."""

    path: Path
    problems: tuple[str, ...]


def take(
    directory: Path,
    software: Software,
    identity: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> Taken:
    """Keep in ``directory`` (a study's :data:`DIRECTORY`) what a session that runs on
    ``software``, with the session ``identity``
    (:meth:`k4opt.execution.EvaluationSession.identity`), needs beyond what is public: see the
    module. ``environment`` is where ``KEY4HEP_STACK`` is read (``os.environ`` if ``None``). Never
    raises for what cannot be read or kept: it is recorded in ``problems``. Raises
    :class:`OSError` if ``directory`` cannot be written."""
    variables = os.environ if environment is None else environment
    directory.mkdir(parents=True, exist_ok=True)
    store = Store(directory)
    problems: list[str] = []
    release = Release.of(variables.get("KEY4HEP_STACK"))
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "software": software.to_dict(),
        "release": None,
        "origins": [],
        "steering": [],
        "evaluators": {},
        "k4opt": None,
    }
    if release is not None:
        manifests = {}
        for role in MANIFESTS:
            manifests[role] = _kept(
                store, problems, f"the {role} manifest", release.manifests[role]
            )
        document["release"] = {**release.to_dict(), "manifests": manifests}
    found = origins(software.local, release)
    code = checkout(os.path.dirname(os.path.abspath(k4opt.__file__)), release)
    if code is not None and all(code.prefix != (o.git.checkout if o.git else None) for o in found):
        found.append(code)  # k4opt run from its checkout, not through a search path
    document["k4opt"] = None if code is None else code.prefix
    for origin in found:
        document["origins"].append(_origin(store, problems, origin))
    document["steering"] = _steering(store, problems, identity, release)
    document["evaluators"] = _evaluators(store, problems, identity)
    _declared(directory, store, identity, problems)
    document["problems"] = problems
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    name = f"software-{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"
    path = directory / name
    with contextlib.suppress(FileExistsError):  # the same state, kept before
        publish(path, text)
    return Taken(path, tuple(problems))


def _kept(store: Store, problems: list[str], what: str, data: bytes) -> str | None:
    try:
        return store.put(data)
    except (SoftwareError, OSError) as exc:
        problems.append(f"{what} not kept: {exc}")
        return None


def _origin(store: Store, problems: list[str], origin: Origin) -> dict[str, Any]:
    if origin.problem:
        problems.append(origin.problem)
    git = None
    if origin.git is not None:
        patch = None
        if origin.git.patch is not None and not origin.git.clean:
            patch = _kept(
                store, problems, f"the changes of {origin.git.checkout}", origin.git.patch
            )
        elif origin.git.patch is None:
            problems.append(f"the changes of {origin.git.checkout} are too large to keep")
        git = origin.git.to_dict(patch)
    files: dict[str, Any] = {}
    for relative, (path, sha256) in origin.files.items():
        try:
            files[relative] = {"sha256": sha256, "kept": store.put_file(path, sha256)}
        except (SoftwareError, OSError) as exc:
            files[relative] = {"sha256": sha256, "kept": None}
            problems.append(f"{path} not kept: {exc}")
    return {
        "kind": origin.kind,
        "prefix": origin.prefix,
        "entries": list(origin.entries),
        "package": origin.package,
        "revision": origin.revision,
        "link": origin.link,
        "evidence": list(origin.evidence),
        "build": origin.build,
        "binaries": [binary.to_dict() for binary in origin.binaries],
        "stale": origin.stale,
        "rebuildable": origin.rebuildable and (origin.git is None or git is not None),
        "git": git,
        "files": files,
        "distributions": list(origin.distributions),
    }


def _steering(
    store: Store, problems: list[str], identity: Mapping[str, Any], release: Release | None
) -> list[dict[str, Any]]:
    """Each reconstruction's job directory (``samples.<sample>.<stage>.directory/<file>`` among
    the session's inputs) against the package of the stack it is, or is a copy of: the files that
    differ from that package's, and those it adds, kept; those it lacks, named."""
    directories: dict[str, dict[str, Mapping[str, Any]]] = {}
    for label, entry in (identity.get("inputs") or {}).items():
        where, separator, relative = label.partition(".directory/")
        if separator and isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
            directories.setdefault(where, {})[relative] = entry
    found = []
    for where, files in directories.items():
        relative, entry = next(iter(files.items()))
        root = str(entry["path"])[: -len(relative) - 1]
        package = release.owner(root) if release else None
        same_as = root if package is not None else None
        if package is None and release is not None:
            package, same_as = _copied(release, root)
        result: dict[str, Any] = {
            "stage": where.removeprefix("samples.").replace(".", "/"),
            "directory": root,
            "package": None if package is None else package.name,
            "revision": None if package is None else package.revision,
            "original": same_as,
            "unchanged": 0,
            "changed": {},
            "added": {},
            "removed": [],
        }
        theirs = _listing(same_as) if same_as and same_as != root else {}
        for name, file in sorted(files.items()):
            sha256 = file.get("sha256")
            if same_as == root:
                result["unchanged"] += 1
                continue
            original = theirs.pop(name, None)
            if original is not None and original == sha256:
                result["unchanged"] += 1
                continue
            kept = None
            if isinstance(sha256, str):
                try:
                    kept = store.put_file(file["path"], sha256)
                except (SoftwareError, OSError) as exc:
                    problems.append(f"{file['path']} not kept: {exc}")
            group = "changed" if original is not None else "added"
            result[group][name] = {"sha256": sha256, "kept": kept}
        result["removed"] = sorted(theirs)
        found.append(result)
    return found


def _copied(release: Release, root: str) -> tuple[Package | None, str | None]:
    """The package of ``release`` whose ``share/<name>`` the directory ``root`` is a copy of, by
    its name (``CLDConfig``), ``HEAD`` packages first; and that directory."""
    name = os.path.basename(root.rstrip("/"))
    packages = sorted(release.packages.values(), key=lambda p: p.version != "HEAD")
    for package in packages:
        candidate = os.path.join(package.prefix, "share", name)
        if os.path.isdir(candidate):
            return package, candidate
    return None, None


def _listing(root: str) -> dict[str, str]:
    """The SHA-256 of each file below ``root``, by its path relative to it."""
    found = {}
    for directory, _, names in os.walk(root, followlinks=True):
        for name in names:
            path = os.path.join(directory, name)
            try:
                with open(path, "rb") as handle:
                    found[os.path.relpath(path, root)] = hashlib.file_digest(
                        handle, "sha256"
                    ).hexdigest()
            except OSError:
                continue
    return found


def _evaluators(store: Store, problems: list[str], identity: Mapping[str, Any]) -> dict[str, Any]:
    """The file of each evaluator and the local modules it imports, kept by their SHA-256."""
    found: dict[str, Any] = {}
    for name, entry in (identity.get("implementations") or {}).items():
        source = entry.get("source") if isinstance(entry, Mapping) else None
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            continue
        path, sha256 = source["path"], source.get("sha256")
        files: dict[str, Any] = {}
        try:
            files[os.path.basename(path)] = store.put_file(path, sha256)
        except (SoftwareError, OSError) as exc:
            problems.append(f"evaluator {name!r}: {path} not kept: {exc}")
        for helper, digest in (source.get("helpers") or {}).items():
            located = _helper(path, helper, digest)
            if located is None:
                problems.append(f"evaluator {name!r}: its module {helper} not found to keep")
                continue
            try:
                files[helper] = store.put_file(located, digest)
            except (SoftwareError, OSError) as exc:
                problems.append(f"evaluator {name!r}: {located} not kept: {exc}")
        found[name] = {"path": path, "sha256": sha256, "files": files}
    return found


def _helper(path: str, relative: str, sha256: object) -> str | None:
    """Where the module ``relative`` an evaluator file imports is: beside it, or relative to the
    directory its top-level package is in, whichever has the bytes ``sha256``."""
    directory = os.path.dirname(path)
    for _ in range(6):
        candidate = os.path.join(directory, relative)
        try:
            with open(candidate, "rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() == sha256:
                    return candidate
        except OSError:
            pass
        directory = os.path.dirname(directory)
    return None


# --- inputs ----------------------------------------------------------------------------------


def _inputs(directory: Path) -> dict[str, Any]:
    try:
        document = json.loads((directory / INPUTS).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise SoftwareError(f"{directory / INPUTS} cannot be read: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != INPUTS_SCHEMA:
        raise SoftwareError(f"{directory / INPUTS} is not of {INPUTS_SCHEMA}")
    inputs = document.get("inputs")
    return dict(inputs) if isinstance(inputs, dict) else {}


def _variable(path: str, variables: Mapping[str, str]) -> str | None:
    """A compared variable that names the file ``path``: as its value, or as an entry of a search
    path (``…PATH``) it lies in. Not ``HOME``, which every file of the home directory lies in."""
    for name, value in sorted(variables.items()):
        if value == path:
            return name
    for name, value in sorted(variables.items()):
        if not name.endswith("PATH"):
            continue
        for part in value.split(os.pathsep):
            part = part.rstrip("/")
            if part and os.path.isabs(part) and path.startswith(f"{part}/"):
                return name
    return None


def keep_inputs(
    directory: Path,
    inputs: Mapping[str, tuple[str, Iterable[str], str]],
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Keep each of ``inputs`` (path: its identity as read, the stages that read it, its origin:
    ``declared`` or ``observed``) not kept before, in ``directory``'s :data:`INPUTS` and blobs;
    what could not be kept, said for a person. A file on CVMFS is identified, never copied; a
    directory a stage listed is no input to keep (its names are watched where it was read)."""
    known = _inputs(directory)
    store = Store(directory)
    variables = compared_variables(environment)
    problems: list[str] = []
    changed = False
    for path, (identity, stages, origin) in sorted(inputs.items()):
        if identity.startswith(_LISTED):  # a directory listed: nothing to keep but its names
            continue
        entry = known.get(path)
        if entry is not None:
            more = sorted({*entry.get("stages", []), *stages})
            if more != entry.get("stages"):
                entry["stages"], changed = more, True
            continue
        salt = secrets.token_hex(16)
        entry = {
            "identity": identity,
            "origin": origin,
            "stages": sorted(set(stages)),
            "variable": _variable(path, variables),
            "salt": salt,
            "commitment": None,
            "bytes": None,
            "kept": None,
            "why": None,
        }
        if identity.startswith(_SHA256):
            sha256 = identity.removeprefix(_SHA256)
            entry["commitment"] = commitment(salt, sha256)
            try:
                entry["bytes"] = os.stat(path).st_size
                entry["kept"] = store.put_file(path, sha256)
            except (SoftwareError, OSError) as exc:
                entry["why"] = str(exc)
                problems.append(f"{path} not kept: {exc}")
        elif identity.startswith("cvmfs:"):
            entry["why"] = "on CVMFS: identified by its content key, not copied"
        else:
            entry["why"] = "a device, a pipe or a socket: nothing to copy"
        known[path], changed = entry, True
    if changed:
        text = json.dumps({"schema": INPUTS_SCHEMA, "inputs": known}, indent=2, sort_keys=True)
        directory.mkdir(parents=True, exist_ok=True)
        replace(directory / INPUTS, text + "\n")
    return problems


def _declared(
    directory: Path, store: Store, identity: Mapping[str, Any], problems: list[str]
) -> None:
    """Keep the files the problem names that the stack does not hold, but a reconstruction's job
    directory (its steering, kept against its package)."""
    found: dict[str, tuple[str, Iterable[str], str]] = {}
    for label, entry in (identity.get("inputs") or {}).items():
        if ".directory/" in label or not isinstance(entry, Mapping):
            continue
        path, sha256 = entry.get("path"), entry.get("sha256")
        if not isinstance(path, str) or not isinstance(sha256, str):
            continue
        if any(path == root or path.startswith(f"{root}/") for root in STACK):
            continue
        parts = label.split(".")
        stage = "/".join(parts[1:3]) if len(parts) > 2 and parts[0] == "samples" else label
        found[path] = (f"{_SHA256}{sha256}", [stage], "declared")
    problems += keep_inputs(directory, found)


# --- what local builds ran -------------------------------------------------------------------


def keep_used(
    directory: Path, run: Path, footprints: Mapping[str, tuple[str, object]], roots: Sequence[str]
) -> None:
    """Note in ``directory``'s :data:`USED` which files below the local entries ``roots`` each of
    the ``footprints`` (stage label: its file, relative to the run directory ``run`` or absolute,
    and its SHA-256, as :func:`k4opt.record.write.recorded_footprints` gives them) read, each
    footprint once. A footprint that cannot be read is left out."""
    path = directory / USED
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        document = {"schema": USED_SCHEMA, "footprints": {}}
    except (OSError, ValueError) as exc:
        raise SoftwareError(f"{path} cannot be read: {exc}") from exc
    seen = document.setdefault("footprints", {})
    real = [(root.rstrip("/"), os.path.realpath(root)) for root in roots]
    changed = False
    for label, (file, sha256) in footprints.items():
        if not isinstance(sha256, str) or sha256 in seen or not file:
            continue
        try:
            footprint = load(run / file if not os.path.isabs(file) else Path(file))
        except SoftwareError:
            continue
        files = {
            name: identity
            for name, identity in footprint.read.items()
            if any(name in pair or name.startswith((f"{pair[0]}/", f"{pair[1]}/")) for pair in real)
        }
        seen[sha256] = {"stage": label, "files": dict(sorted(files.items()))}
        changed = True
    if changed:
        directory.mkdir(parents=True, exist_ok=True)
        replace(path, json.dumps(document, indent=2, sort_keys=True) + "\n")


__all__ = [
    "DIRECTORY",
    "INPUTS",
    "MAX_INPUT",
    "MAX_KEPT",
    "SCHEMA",
    "USED",
    "Store",
    "Taken",
    "commitment",
    "keep_inputs",
    "keep_used",
    "take",
]
