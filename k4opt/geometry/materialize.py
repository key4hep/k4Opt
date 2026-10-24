"""Safe, reproducible materialization of a modified geometry snapshot.

The reachable geometry files (the compact file, every local XML it includes,
and local resource files such as CAD models they reference) are copied, with
their root-relative layout preserved, into a temporary sibling of the requested
output directory. The selected attributes are edited structurally in that
copy, each edited file is re-parsed and compared against its original, the
include graph of the copy is re-walked, a manifest is written, and only then is
the copy atomically renamed into place. On any failure the temporary copy is
removed and neither source nor destination is touched.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lxml import etree

from k4opt.errors import K4OptError
from k4opt.geometry.discovery import Catalog
from k4opt.geometry.errors import IncludeError, MaterializationError
from k4opt.geometry.includes import (
    INCLUDE_TAGS,
    OUTSIDE_ROOT,
    ExternalInclude,
    IncludeGraph,
    classify_ref,
    find_resource,
    is_local_path,
    looks_like_file,
    resolve_local_path,
    to_root_relative,
    walk_includes,
)
from k4opt.geometry.overrides import ResolvedOverride
from k4opt.geometry.source import GeometrySource
from k4opt.geometry.xmlio import (
    find_by_locator,
    parse_xml_bytes,
    serialize_tree,
    sha256_bytes,
    sha256_file,
    xml_declaration,
)

MANIFEST_NAME = "k4opt-manifest.json"
MANIFEST_SCHEMA = "k4opt.manifest/1"

#: Environment variables recorded in the manifest to identify the software stack.
STACK_ENV_VARS = ("KEY4HEP_STACK", "K4GEO")


def compute_fingerprint(
    catalog: Catalog,
    resolved: Sequence[ResolvedOverride],
    extra: Additions | None = None,
) -> str:
    """Deterministic identity of the locally captured geometry that ``resolved`` would produce.

    A SHA-256 over the compact path, the content hashes of every reachable file and
    resource, the symlinked include names and where they point, and the *effective* edits
    (overrides that change a value). It does not depend on timestamps, output locations or
    on overrides that restate the current value, so equal fingerprints mean equal captured
    geometry files and it can be used to deduplicate snapshots.

    It identifies the local inputs only: the content of external references (such as
    ``${K4GEO}`` includes) and the Key4hep software stack are not part of it, so it is
    not sufficient as an evaluation cache key by itself. ``extra`` are the files,
    resources and symlinked names introduced by the overrides (see :func:`new_files`); they
    are computed when omitted, and passed in by :func:`materialize` so the fingerprint uses
    exactly the bytes it verified.
    """
    if extra is None:
        extra = new_files(catalog, resolved)
    payload = {
        "compact": catalog.compact,
        "files": dict(sorted(catalog.file_hashes.items())),
        "resources": dict(sorted({**catalog.resource_hashes, **extra.hashes}.items())),
        "aliases": dict(sorted({**catalog.file_aliases, **extra.aliases}.items())),
        "edits": sorted(
            (r.file, r.locator, r.attribute, r.new_value) for r in resolved if r.changed
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(canonical.encode("ascii"))


@dataclass(frozen=True, slots=True)
class Additions:
    """Local files that only the edited ``ref`` attributes make reachable."""

    files: Mapping[str, str]
    """XML files reached through a redirected include, with their hashes."""
    resources: Mapping[str, str]
    """Non-XML resources, with their hashes."""
    aliases: Mapping[str, str]
    """Symlinked names of the added files, mapped to the file they point at."""
    externals: tuple[ExternalInclude, ...]
    """References from the added files to files outside the geometry root."""

    @property
    def hashes(self) -> dict[str, str]:
        return {**self.files, **self.resources}


def new_files(catalog: Catalog, resolved: Sequence[ResolvedOverride]) -> Additions:
    """Files that only the edited ``ref`` attributes point at, with their hashes.

    Discovery records the files reachable from the unmodified geometry; an override
    that redirects a ``ref`` to another existing local file makes that file (and, for an
    include, everything it includes) part of the geometry, so it must be copied and
    covered by the fingerprint too.
    """
    root_real = Path(os.path.realpath(catalog.root))
    known = {*catalog.file_hashes, *catalog.resource_hashes}
    files: dict[str, str] = {}
    resources: dict[str, str] = {}
    aliases: dict[str, str] = {}
    externals: list[ExternalInclude] = []

    real_files = {os.path.realpath(catalog.root / f): f for f in catalog.files}

    def add(into: dict[str, str], rel: str) -> None:
        if rel in known or rel in files or rel in resources:
            return
        real = os.path.realpath(catalog.root / rel)
        if into is files and real in real_files:  # another name of a file already reachable
            aliases[rel] = real_files[real]
            return
        if into is files:
            real_files[real] = rel
        try:
            into[rel] = sha256_file(catalog.root / rel)
        except OSError as exc:
            raise MaterializationError(f"cannot read {rel}: {exc}") from exc

    for item in resolved:
        if not item.changed or item.attribute != "ref":
            continue
        base_dir = (catalog.root / item.file).parent
        if item.parameter.tag in INCLUDE_TAGS:
            if classify_ref(item.new_value) is not None:
                continue  # external; reported by the walk of the snapshot
            try:
                target = resolve_local_path(
                    catalog.root, root_real, base_dir, item.new_value, label=item.file
                )
                graph = walk_includes(catalog.root, to_root_relative(catalog.root, target))
            except IncludeError as exc:
                raise MaterializationError(f"{item.target}: {exc}") from exc
            for rel in graph.files:
                add(files, rel)
            for rel in graph.resources:
                add(resources, rel)
            aliases.update({a: f for a, f in graph.file_aliases.items() if a not in known})
            externals.extend(e for e in graph.externals if e.reason == OUTSIDE_ROOT)
            continue
        found, escapes = find_resource(catalog.root, root_real, base_dir, item.new_value)
        if escapes:
            raise MaterializationError(
                f"{item.target}: resource {item.new_value!r} lies outside the geometry root; "
                "a relocatable snapshot cannot reference it"
            )
        if found is None and is_local_path(item.new_value):
            # only when the old value was evidently a file (found, or an external path such as
            # ${CAD}/x.stl); a plain name may be a symbolic reference rather than a file
            was_resource, _ = find_resource(catalog.root, root_real, base_dir, item.old_value)
            if was_resource is not None or looks_like_file(item.old_value):
                raise MaterializationError(
                    f"{item.target}: resource {item.new_value!r} does not exist "
                    f"(relative to {item.file}); the snapshot would reference a missing file"
                )
        if found is not None:
            add(resources, found)
    return Additions(files, resources, aliases, tuple(externals))


@dataclass(frozen=True, slots=True)
class ChangedFile:
    """A file that differs between source and snapshot."""

    path: str
    """Root-relative POSIX path."""
    source_path: Path
    output_path: Path
    source_sha256: str
    output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """What :func:`materialize` produced."""

    source_root: Path
    output_root: Path
    compact_path: Path
    """The compact entry file inside the generated snapshot."""
    manifest_path: Path
    changes: tuple[ResolvedOverride, ...]
    files: tuple[ChangedFile, ...]
    manifest: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        """Deterministic identity of the generated geometry (see :func:`compute_fingerprint`)."""
        return str(self.manifest["fingerprint"])

    def output_path(self, source_relative: str) -> Path:
        """Map a root-relative source path to its location in the snapshot."""
        return self.output_root / source_relative

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "compact_path": str(self.compact_path),
            "manifest_path": str(self.manifest_path),
            "fingerprint": self.fingerprint,
            "changes": [c.to_dict() for c in self.changes],
            "files": [f.to_dict() for f in self.files],
        }


def _is_within(root: Path, path: Path) -> bool:
    try:
        return os.path.commonpath([str(root), str(path)]) == str(root)
    except ValueError:  # pragma: no cover - different drives on Windows
        return False


def check_output_location(source_root: Path, output: str | os.PathLike[str]) -> Path:
    """Validate the requested output directory and return it as an absolute path."""
    out = Path(os.path.abspath(os.fspath(output)))
    if out.exists() or out.is_symlink():
        raise MaterializationError(f"output {out} already exists; choose a new directory") from None
    if _is_within(source_root, out) or out == source_root:
        raise MaterializationError(f"output {out} lies inside the source tree {source_root}")
    ancestor = out.parent
    while not ancestor.exists():
        if ancestor.parent == ancestor:  # pragma: no cover - filesystem root always exists
            break
        ancestor = ancestor.parent
    ancestor_real = Path(os.path.realpath(ancestor))
    if _is_within(source_root, ancestor_real):
        raise MaterializationError(
            f"output {out} resolves to a location inside the source tree {source_root}"
        )
    return out


def materialize(
    source: GeometrySource,
    catalog: Catalog,
    resolved: Sequence[ResolvedOverride],
    output: str | os.PathLike[str],
) -> MaterializationResult:
    """Copy the reachable files of ``source`` to ``output`` with ``resolved`` applied."""
    out = check_output_location(source.root, output)
    extra = new_files(catalog, resolved)
    if MANIFEST_NAME in {*catalog.files, *catalog.resources, *extra.hashes}:
        raise MaterializationError(
            f"{MANIFEST_NAME} is part of the geometry but is the name of the generated manifest"
        )

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".k4opt-tmp", dir=out.parent)
        )
    except OSError as exc:
        raise MaterializationError(f"materialization failed: {exc}") from exc
    try:
        snapshot = staging / "snapshot"
        _copy_closure(source.root, snapshot, [*catalog.files, *catalog.resources, *extra.hashes])
        _link_aliases(snapshot, {**catalog.file_aliases, **extra.aliases})
        _verify_copies(snapshot, {**catalog.file_hashes, **catalog.resource_hashes, **extra.hashes})
        changed = _apply_edits(source.root, snapshot, [r for r in resolved if r.changed])
        graph = _verify_snapshot(snapshot, source.compact)
        manifest = _build_manifest(source, catalog, out, resolved, changed, graph, extra)
        (snapshot / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish(snapshot, out)
    except K4OptError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except (OSError, shutil.Error) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise MaterializationError(f"materialization failed: {exc}") from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)

    return MaterializationResult(
        source_root=source.root,
        output_root=out,
        compact_path=out / source.compact,
        manifest_path=out / MANIFEST_NAME,
        changes=tuple(resolved),
        files=tuple(
            ChangedFile(c.path, c.source_path, out / c.path, c.source_sha256, c.output_sha256)
            for c in changed
        ),
        manifest=manifest,
    )


def _publish(snapshot: Path, out: Path) -> None:
    """Move ``snapshot`` to ``out`` atomically, never replacing anything already there.

    ``rename(2)`` onto an existing *empty* directory succeeds silently, so where the
    platform offers ``renameat2(RENAME_NOREPLACE)`` it is used; elsewhere a plain rename
    preceded by an existence check, is the best available (filesystems such as NFS lack
    ``RENAME_NOREPLACE``, so failing closed there would make the tool unusable).
    """
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is not None:
        at_fdcwd, noreplace = -100, 1
        if renameat2(at_fdcwd, os.fsencode(snapshot), at_fdcwd, os.fsencode(out), noreplace) == 0:
            return
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise MaterializationError(f"output {out} already exists; choose a new directory")
        if code not in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            raise OSError(code, os.strerror(code), str(out))
    if os.path.lexists(out):
        raise MaterializationError(f"output {out} already exists; choose a new directory")
    os.rename(snapshot, out)


def _verify_copies(snapshot: Path, expected: Mapping[str, str]) -> None:
    """Every staged file must still match the hash recorded at discovery."""
    for rel, digest in expected.items():
        if sha256_file(snapshot / rel) != digest:
            raise MaterializationError(
                f"{rel} changed since discovery (hash mismatch); "
                "rerun discovery before materializing"
            )


def _copy_closure(source_root: Path, snapshot: Path, files: Sequence[str]) -> None:
    """Copy root-relative ``files`` into ``snapshot`` keeping the layout; symlinks are followed."""
    for rel in files:
        destination = snapshot / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / rel, destination)
        # installed geometries are often read-only; the snapshot must stay editable
        destination.chmod(destination.stat().st_mode | stat.S_IWUSR)


def _link_aliases(snapshot: Path, aliases: Mapping[str, str]) -> None:
    """Recreate symlinked include names as relative links, so edits reach every name."""
    for alias, target in aliases.items():
        link = snapshot / alias
        if os.path.lexists(link):
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(snapshot / target, link.parent), link)


def _verify_snapshot(snapshot: Path, compact: str) -> IncludeGraph:
    """Re-walk the include graph of the edited snapshot; every local include must still resolve."""
    try:
        return walk_includes(snapshot, compact)
    except IncludeError as exc:
        raise MaterializationError(f"generated snapshot is not self-consistent: {exc}") from exc


def _apply_edits(
    source_root: Path, snapshot: Path, resolved: Sequence[ResolvedOverride]
) -> list[ChangedFile]:
    by_file: dict[str, list[ResolvedOverride]] = {}
    for item in resolved:
        by_file.setdefault(item.file, []).append(item)

    snapshot_real = Path(os.path.realpath(snapshot))
    changed: list[ChangedFile] = []
    for rel, items in by_file.items():
        target_path = snapshot / rel
        if not _is_within(snapshot_real, Path(os.path.realpath(target_path))):
            raise MaterializationError(f"{rel} resolves outside the snapshot; refusing to edit")
        original = target_path.read_bytes()
        tree = parse_xml_bytes(original, label=rel)
        expected: dict[str, dict[str, str]] = {}
        for item in items:
            element = find_by_locator(tree, item.locator)
            current = element.get(item.attribute)
            if current != item.old_value:
                raise MaterializationError(
                    f"{rel}: {item.target} has value {current!r} but discovery recorded "
                    f"{item.old_value!r}; the source changed since discovery"
                )
            element.set(item.attribute, item.new_value)
            expected.setdefault(item.locator, {})[item.attribute] = item.new_value
        declared, standalone = xml_declaration(original)
        edited = serialize_tree(tree, declaration=declared, standalone=standalone)
        _verify_edit(
            rel, parse_xml_bytes(original, label=rel), parse_xml_bytes(edited, label=rel), expected
        )
        target_path.write_bytes(edited)
        changed.append(
            ChangedFile(
                path=rel,
                source_path=source_root / rel,
                output_path=target_path,
                source_sha256=sha256_bytes(original),
                output_sha256=sha256_bytes(edited),
            )
        )
    return changed


def _verify_edit(
    rel: str,
    original: etree._ElementTree,
    edited: etree._ElementTree,
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    """Assert that ``edited`` differs from ``original`` only in the expected attributes."""
    before = list(original.iter())
    after = list(edited.iter())
    if len(before) != len(after):
        raise MaterializationError(
            f"{rel}: node count changed during edit ({len(before)} -> {len(after)})"
        )
    for old, new in zip(before, after, strict=True):
        if old.tag != new.tag or old.text != new.text or old.tail != new.tail:
            raise MaterializationError(
                f"{rel}: unexpected structural change at {original.getpath(old)}"
            )
        if not isinstance(old.tag, str):
            continue
        attributes = {str(k): v for k, v in old.attrib.items()}
        attributes.update(expected.get(original.getpath(old), {}))
        if attributes != {str(k): v for k, v in new.attrib.items()}:
            raise MaterializationError(
                f"{rel}: unexpected attribute change at {original.getpath(old)}"
            )


def _externals(
    catalog: Catalog,
    graph: IncludeGraph,
    resolved: Sequence[ResolvedOverride],
    extra: Additions,
) -> list[ExternalInclude]:
    """External references of the *generated* geometry.

    Include-style references come from the edited snapshot. References to files outside
    the root depend on where the geometry lives, so those are taken from the source
    catalog, minus the ones an override has replaced or made unreachable.
    """
    edited = {(r.file, r.locator) for r in resolved if r.changed and r.attribute == "ref"}
    reachable = set(graph.files)
    kept = [
        e
        for e in catalog.externals
        if e.reason == OUTSIDE_ROOT and e.file in reachable and (e.file, e.locator) not in edited
    ]
    combined = (
        [e for e in graph.externals if e.reason != OUTSIDE_ROOT] + kept + list(extra.externals)
    )
    return list(dict.fromkeys(combined))  # a file reached by several redirects is listed once


def _build_manifest(
    source: GeometrySource,
    catalog: Catalog,
    out: Path,
    resolved: Sequence[ResolvedOverride],
    changed: Sequence[ChangedFile],
    graph: IncludeGraph,
    extra: Additions,
) -> dict[str, Any]:
    from k4opt import __version__

    git = source.git_info()
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "k4opt_version": __version__,
        "fingerprint": compute_fingerprint(catalog, resolved, extra),
        "environment": {k: os.environ[k] for k in STACK_ENV_VARS if k in os.environ},
        "source": {
            "root": str(source.root),
            "compact": source.compact,
            "git": git.to_dict() if git else None,
        },
        "output": {"root": str(out), "compact": source.compact},
        "reachable_files": list(graph.files),
        "resources": [*catalog.resources, *extra.resources],
        "external_includes": [
            {"file": e.file, "line": e.line, "ref": e.ref, "reason": e.reason}
            for e in _externals(catalog, graph, resolved, extra)
        ],
        "changes": [r.to_dict() for r in resolved],
        "files": [c.to_dict() for c in changed],
    }
