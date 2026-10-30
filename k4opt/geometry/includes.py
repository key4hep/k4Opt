"""Deterministic walker over the local XML include graph of a compact file.

Starting from the selected compact file, every ``ref`` attribute on a supported
include element is classified as either a *local* include (a relative path that
stays inside the geometry root) or an *external* reference (environment
variables such as ``${DD4hepINSTALL}``, URLs, absolute paths). Local includes
are parsed recursively in document order; external references are reported and
left untouched, and :func:`external_files` says what they read. ``ref`` attributes on
other elements that point at an existing local file (e.g. a CAD ``.stl`` model) are
recorded as *resources* so a snapshot can carry them along. This is an XML include
walker, not an interpreter of DD4hep's preprocessing or of C++ detector constructors.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from lxml import etree

from k4opt.frozen import Json
from k4opt.geometry.errors import IncludeError
from k4opt.geometry.xmlio import (
    element_locator,
    iter_elements,
    line_of,
    parse_xml_bytes,
    read_xml_file,
    sha256_file,
)

#: Element tags whose ``ref`` attribute DD4hep treats as an XML file to include.
INCLUDE_TAGS: frozenset[str] = frozenset({"include", "gdmlFile", "file", "xml"})

#: Reason recorded for a ``ref`` that points at an existing file outside the geometry root.
OUTSIDE_ROOT = "file outside the geometry root"

#: Deepest include chain followed; deeper nesting is an IncludeError, not a RecursionError.
MAX_INCLUDE_DEPTH = 100

_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")


@dataclass(frozen=True, slots=True)
class ExternalInclude:
    """A reference that is not a local file inside the geometry root."""

    file: str
    """Root-relative path of the file containing the reference."""
    line: int | None
    ref: str
    reason: str
    locator: str = ""
    """Structural XPath of the referencing element within its file."""
    include: bool = False
    """Whether it is the ``ref`` of an include element (an XML file), not of a resource."""


@dataclass(frozen=True, slots=True)
class IncludeEdge:
    """A resolved local include: ``source`` includes ``target`` (both root-relative)."""

    source: str
    line: int | None
    ref: str
    target: str


@dataclass(frozen=True, slots=True)
class IncludeGraph:
    """Result of walking the include graph. Files are in deterministic discovery order."""

    files: tuple[str, ...]
    trees: dict[str, etree._ElementTree]
    edges: tuple[IncludeEdge, ...]
    externals: tuple[ExternalInclude, ...]
    resources: tuple[str, ...]
    """Local non-XML files referenced by ``ref`` attributes (root-relative, discovery order)."""
    file_aliases: dict[str, str] = field(default_factory=dict)
    """Other root-relative names (symlinks) under which a parsed file is referenced, by file."""
    file_hashes: dict[str, str] = field(default_factory=dict)
    """SHA-256 of the exact bytes each file was parsed from, so a later edit cannot pair the
    locators of one revision with the hash of another."""


def classify_ref(ref: str) -> str | None:
    """Return ``None`` if ``ref`` is a local relative path, else the reason it is external."""
    if not ref.strip():
        return "empty reference"
    if "$" in ref:
        return "environment variable or placeholder reference"
    if _URL_RE.match(ref) and not os.path.isabs(ref):
        return "URL or scheme-qualified reference"
    if os.path.isabs(ref):
        return "absolute path"
    return None


def to_root_relative(root: Path, path: Path) -> str:
    """Return ``path`` relative to ``root`` as a POSIX string."""
    return PurePosixPath(path.relative_to(root)).as_posix()


def _is_within(root: Path, path: Path) -> bool:
    try:
        return os.path.commonpath([str(root), str(path)]) == str(root)
    except ValueError:  # pragma: no cover - different drives on Windows
        return False


def resolve_local_path(
    root: Path, root_real: Path, base_dir: Path, ref: str, *, label: str
) -> Path:
    """Resolve a local ``ref`` relative to ``base_dir`` and verify it stays inside ``root``.

    Returns the lexical (non-symlink-resolved) absolute path of the target. Raises
    :class:`IncludeError` when the reference escapes the root lexically, through a
    symlink, or when the target does not exist.
    """
    lexical = Path(os.path.normpath(base_dir / ref))
    if not _is_within(root, lexical):
        raise IncludeError(
            f"{label}: include {ref!r} escapes the geometry root {root}; "
            "set 'root' to a directory containing every included file"
        )
    if not lexical.exists():
        raise IncludeError(f"{label}: local include {ref!r} not found (expected {lexical})")
    real = Path(os.path.realpath(lexical))
    if not _is_within(root_real, real):
        raise IncludeError(
            f"{label}: include {ref!r} resolves through a symlink to {real}, "
            "outside the geometry root"
        )
    if not lexical.is_file():
        raise IncludeError(f"{label}: local include {ref!r} is not a regular file ({lexical})")
    return lexical


def looks_like_file(ref: str) -> bool:
    """True when ``ref`` has a path separator or an extension, i.e. is not a plain name.

    A cheap filter, so material names such as ``ref="Si"`` never touch the filesystem.
    """
    return "/" in ref or "." in os.path.basename(ref)


def is_local_path(ref: str) -> bool:
    """True when ``ref`` is a relative file path, not an external reference or a plain name."""
    return classify_ref(ref) is None and looks_like_file(ref)


def find_resource(root: Path, root_real: Path, base_dir: Path, ref: str) -> tuple[str | None, bool]:
    """Locate the local file a non-include ``ref`` points at.

    Returns ``(root-relative path, False)`` for an existing file inside the root,
    ``(None, True)`` for a path outside it, and ``(None, False)`` when ``ref`` is not a local
    path to an existing file (e.g. a material name). A path that climbs out of the root is
    outside it even if no file is there: moving the geometry could make it name one.
    """
    if not is_local_path(ref):
        return None, False
    lexical = Path(os.path.normpath(base_dir / ref))
    if not _is_within(root, lexical):
        return None, True
    if not lexical.is_file():
        return None, False
    if not _is_within(root_real, Path(os.path.realpath(lexical))):  # through a symlink
        return None, True
    return to_root_relative(root, lexical), False


def walk_includes(root: Path, compact: str, *, resources_at: Path | None = None) -> IncludeGraph:
    """Parse ``compact`` and every reachable local include below ``root``.

    ``root`` must be an absolute directory and ``compact`` a root-relative POSIX
    path. Each physical file is parsed once, even when it is included several
    times or through an include cycle; a symlinked name in another directory is followed
    again, because its relative references resolve against that directory.

    Non-include references are looked up below ``resources_at`` (``root`` by default), at the
    same root-relative directory. A snapshot is walked with its source there: a reference that
    leaves the root names the file next to the source, not next to the copy.
    """
    root_real = Path(os.path.realpath(root))
    lookup = root if resources_at is None else resources_at
    lookup_real = Path(os.path.realpath(lookup))
    entry = resolve_local_path(root, root_real, root, compact, label="geometry")

    files: list[str] = []
    trees: dict[str, etree._ElementTree] = {}
    edges: list[IncludeEdge] = []
    externals: list[ExternalInclude] = []
    resources: dict[str, None] = {}
    file_aliases: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    visited_real: dict[str, str] = {}
    scanned: set[tuple[str, Path]] = set()
    active: set[tuple[str, str]] = set()

    def note_resource(rel: str, line: int | None, locator: str, base_dir: Path, ref: str) -> None:
        reason = classify_ref(ref)
        if reason is not None:
            # left untouched and outside the fingerprint, but visible in the manifest
            if ref.strip() and looks_like_file(ref):
                externals.append(ExternalInclude(rel, line, ref, reason, locator))
            return
        where = lookup / base_dir.relative_to(root)
        found, escapes = find_resource(lookup, lookup_real, where, ref)
        if escapes:
            externals.append(ExternalInclude(rel, line, ref, OUTSIDE_ROOT, locator))
        elif found is not None:
            resources.setdefault(found)

    def visit(lexical: Path, chain: tuple[str, ...]) -> str:
        real = os.path.realpath(lexical)
        rel = to_root_relative(root, lexical)
        if real in visited_real:
            canonical = visited_real[real]
            if rel != canonical:
                file_aliases.setdefault(rel, canonical)
            # relative references resolve against the lexical directory, so an alias in
            # another directory has its own includes even though the file is the same
            scan(canonical, lexical.parent, real, chain)
            return canonical
        visited_real[real] = rel
        label = rel if not chain else f"{rel} (included via {' -> '.join(chain)})"
        data = read_xml_file(lexical, label=label)
        tree = parse_xml_bytes(data, label=label)
        files.append(rel)
        trees[rel] = tree
        file_hashes[rel] = hashlib.sha256(data).hexdigest()
        scan(rel, lexical.parent, real, chain)
        return rel

    def scan(rel: str, base_dir: Path, real: str, chain: tuple[str, ...]) -> None:
        """Follow the references of the parsed file ``rel`` as seen from ``base_dir``."""
        if (real, base_dir) in scanned:
            return
        scanned.add((real, base_dir))
        # a directory symlink that loops back (``loop -> .``) yields ever longer lexical
        # directories for one physical (file, directory); stop when it is already being scanned
        physical = (real, os.path.realpath(base_dir))
        if physical in active:
            return
        if len(chain) >= MAX_INCLUDE_DEPTH:
            raise IncludeError(
                f"{rel}: includes nested deeper than {MAX_INCLUDE_DEPTH} levels "
                f"(included via {' -> '.join(chain[:3])} -> ...)"
            )
        active.add(physical)
        try:
            follow(rel, base_dir, chain)
        finally:
            active.discard(physical)

    def follow(rel: str, base_dir: Path, chain: tuple[str, ...]) -> None:
        tree = trees[rel]
        for element in iter_elements(tree):
            ref = element.get("ref")
            if ref is None:
                continue
            tag = etree.QName(element).localname if element.tag.startswith("{") else element.tag
            line = line_of(element)
            locator = element_locator(tree, element)
            if tag not in INCLUDE_TAGS:
                note_resource(rel, line, locator, base_dir, ref)
                continue
            reason = classify_ref(ref)
            if reason is not None:
                externals.append(
                    ExternalInclude(
                        file=rel, line=line, ref=ref, reason=reason, locator=locator, include=True
                    )
                )
                continue
            here = f"{rel}:{line}"
            label = here if not chain else f"{here} (included via {' -> '.join(chain)})"
            target = resolve_local_path(root, root_real, base_dir, ref, label=label)
            target_rel = visit(target, (*chain, here))
            edges.append(IncludeEdge(source=rel, line=line, ref=ref, target=target_rel))

    visit(entry, ())
    return IncludeGraph(
        files=tuple(files),
        trees=trees,
        edges=tuple(edges),
        externals=tuple(externals),
        resources=tuple(r for r in resources if r not in trees),
        file_aliases={a: f for a, f in file_aliases.items() if a not in trees},
        file_hashes=file_hashes,
    )


def resolve_external(root: Path, external: ExternalInclude) -> Path | None:
    """The file ``external`` names now: its environment variables expanded, a relative result
    taken from the directory of the file that holds it (below ``root``). ``None`` if it names no
    existing file: a variable that is not set, a URL, a file that is not there."""
    ref = os.path.expandvars(external.ref)
    if "$" in ref or not ref.strip() or (_URL_RE.match(ref) and not os.path.isabs(ref)):
        return None
    path = Path(os.path.normpath((root / external.file).parent / ref))
    return path if path.is_file() else None


def external_files(root: Path, externals: Iterable[ExternalInclude]) -> dict[str, dict[str, Json]]:
    """What the ``externals`` of the files below ``root`` read now, by reference as written.

    Each entry holds the ``path`` the reference resolves to (:func:`resolve_external`) and the
    ``sha256`` of that file, ``None`` where there is no file or it cannot be read. An include
    also holds the ``files`` it reaches in turn (its local includes and the resources they
    reference) by path relative to its directory, each with its SHA-256, or ``None`` if they
    cannot be followed; the external references of those files are entries of their own.
    """
    found: dict[str, dict[str, Json]] = {}
    pending = deque((root, external) for external in externals)
    while pending:
        base, external = pending.popleft()
        if external.ref in found or not external.ref.strip():
            continue
        path = resolve_external(base, external)
        try:
            digest = None if path is None else sha256_file(path)
        except OSError:
            digest = None
        entry: dict[str, Json] = {"path": None if path is None else str(path), "sha256": digest}
        found[external.ref] = entry
        if path is None or digest is None or not external.include:
            continue
        top = Path(path.anchor)  # an external file may include from anywhere
        try:
            graph = walk_includes(top, path.relative_to(top).as_posix())
            reached = {rel: graph.file_hashes[rel] for rel in graph.files[1:]}
            reached |= {rel: sha256_file(top / rel) for rel in graph.resources}
        except (IncludeError, OSError):
            entry["files"] = None
            continue
        entry["files"] = {
            os.path.relpath(top / rel, path.parent): sha for rel, sha in sorted(reached.items())
        }
        pending.extend((top, nested) for nested in graph.externals)
    return found
