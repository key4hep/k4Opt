"""Read-only discovery of addressable XML attributes and their canonical references.

For every attribute of every element in the reachable geometry files, discovery
generates exactly one canonical target reference that is unique within its file
by construction. Candidates are tried from most to least readable:

1. ``constants.NAME`` for the ``value`` attribute of ``<constant name="NAME">``.
2. ``NAME.step....attr`` anchored at the nearest ancestor-or-self element that
   carries a usable ``name``, then the same path with a tag-qualified anchor
   (``TAG[name=NAME].step....attr``); then the same two forms for the next named
   ancestor further up, and so on.
3. A positional path from the document root, ``ROOTTAG.step....attr``.

Whenever two attributes in the same file would share a reference, both move to
their next candidate until every reference in the file is unique. References
that repeat across *different* files are kept but flagged ``ambiguous``; an
override must then name the file. Steps between the anchor and the attribute
use ``[name=...]`` or ``[id=...]`` when unique among same-tag siblings, nothing
when the tag is unique among siblings, and a 1-based ``[index]`` otherwise
(``order_sensitive``).
"""

from __future__ import annotations

import difflib
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, overload

from lxml import etree

from k4opt.geometry.errors import AmbiguousTargetError, IncludeError, UnknownTargetError
from k4opt.geometry.includes import ExternalInclude, IncludeEdge, walk_includes
from k4opt.geometry.references import CONSTANTS_PREFIX, Step, is_name, is_value, parse_reference
from k4opt.geometry.xmlio import element_locator, iter_elements, line_of, sha256_file


@dataclass(frozen=True, slots=True)
class Parameter:
    """One addressable XML attribute discovered in the geometry."""

    target: str
    """Canonical reference, e.g. ``InnerTrackerBarrel.layer[id=1].rphi_layout.nphi``."""
    file: str
    """Source file path relative to the geometry root (POSIX separators)."""
    attribute: str
    value: str
    """Raw attribute text, e.g. ``"43*2"``; never evaluated."""
    locator: str
    """Structural XPath of the owning element (internal identity, not user-facing)."""
    line: int | None
    tag: str
    order_sensitive: bool
    """True when the reference relies on a positional ``[index]`` step."""
    ambiguous: bool
    """True when the same reference exists in another reachable file (``file`` required)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "file": self.file,
            "attribute": self.attribute,
            "value": self.value,
            "locator": self.locator,
            "line": self.line,
            "tag": self.tag,
            "order_sensitive": self.order_sensitive,
            "ambiguous": self.ambiguous,
        }


@dataclass(frozen=True, slots=True)
class Unaddressable:
    """An attribute that discovery could not give a reference (namespaced names etc.)."""

    file: str
    locator: str
    attribute: str
    reason: str


def normalize_file(file: str) -> str:
    """Normalize a user-supplied root-relative file path to the catalog spelling."""
    text = file.strip().replace("\\", "/")
    if not text or os.path.isabs(text):
        raise UnknownTargetError(f"file {file!r} must be a relative path below the geometry root")
    normalized = PurePosixPath(os.path.normpath(text)).as_posix()
    if normalized == ".." or normalized.startswith("../"):
        raise UnknownTargetError(f"file {file!r} escapes the geometry root")
    return normalized


class Catalog(Sequence[Parameter]):
    """Immutable, indexed collection of discovered parameters for one geometry source."""

    def __init__(
        self,
        *,
        root: Path,
        compact: str,
        files: Iterable[str],
        file_hashes: Mapping[str, str],
        parameters: Iterable[Parameter],
        externals: Iterable[ExternalInclude] = (),
        edges: Iterable[IncludeEdge] = (),
        resources: Iterable[str] = (),
        resource_hashes: Mapping[str, str] | None = None,
        file_aliases: Mapping[str, str] | None = None,
        aliases: Mapping[str, Iterable[Parameter]] | None = None,
        unaddressable: Iterable[Unaddressable] = (),
    ) -> None:
        self._root = root
        self._compact = compact
        self._files = tuple(files)
        self._file_hashes = dict(
            file_hashes
        )  # exposed read-only; a plain dict keeps Catalog picklable
        self._parameters = tuple(parameters)
        self._externals = tuple(externals)
        self._edges = tuple(edges)
        self._resources = tuple(resources)
        self._resource_hashes = dict(resource_hashes or {})
        self._file_aliases = dict(file_aliases or {})
        self._unaddressable = tuple(unaddressable)
        by_target: dict[str, list[Parameter]] = defaultdict(list)
        for parameter in self._parameters:
            by_target[parameter.target].append(parameter)
        self._by_target: dict[str, tuple[Parameter, ...]] = {
            k: tuple(v) for k, v in by_target.items()
        }
        self._aliases: dict[str, tuple[Parameter, ...]] = {
            k: tuple(v) for k, v in (aliases or {}).items()
        }

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._parameters)

    @overload
    def __getitem__(self, index: int) -> Parameter: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[Parameter, ...]: ...
    def __getitem__(self, index: int | slice) -> Parameter | tuple[Parameter, ...]:
        return self._parameters[index]

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters)

    def __repr__(self) -> str:
        return (
            f"Catalog({len(self)} parameters, {len(self._files)} files, compact={self._compact!r})"
        )

    # -- metadata -----------------------------------------------------------
    @property
    def root(self) -> Path:
        return self._root

    @property
    def compact(self) -> str:
        return self._compact

    @property
    def files(self) -> tuple[str, ...]:
        """Reachable local XML files in discovery order, relative to the root."""
        return self._files

    @property
    def file_hashes(self) -> Mapping[str, str]:
        """SHA-256 of every reachable file at discovery time."""
        return MappingProxyType(self._file_hashes)

    @property
    def externals(self) -> tuple[ExternalInclude, ...]:
        """Include references left unresolved (environment variables, URLs, absolute paths)."""
        return self._externals

    @property
    def edges(self) -> tuple[IncludeEdge, ...]:
        return self._edges

    @property
    def resource_hashes(self) -> Mapping[str, str]:
        """SHA-256 of every local resource file at discovery time."""
        return MappingProxyType(self._resource_hashes)

    @property
    def resources(self) -> tuple[str, ...]:
        """Local non-XML files referenced from the geometry (e.g. CAD models), root-relative."""
        return self._resources

    @property
    def file_aliases(self) -> Mapping[str, str]:
        """Symlinked names under which a reachable file is also included, mapped to that file."""
        return MappingProxyType(self._file_aliases)

    @property
    def unaddressable(self) -> tuple[Unaddressable, ...]:
        return self._unaddressable

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self._by_target)

    # -- queries ------------------------------------------------------------
    def filter(self, text: str) -> tuple[Parameter, ...]:
        """Case-insensitive substring filter over target, file and value."""
        needle = text.lower()
        return tuple(
            p
            for p in self._parameters
            if needle in p.target.lower() or needle in p.file.lower() or needle in p.value.lower()
        )

    def lookup(self, target: str, file: str | None = None) -> Parameter:
        """Resolve ``target`` (optionally restricted to ``file``) to exactly one parameter."""
        key = parse_reference(target).text
        matches = self._by_target.get(key, ())
        if file is not None:
            wanted = normalize_file(file)
            if wanted not in self._file_hashes:
                raise UnknownTargetError(
                    f"file {file!r} is not among the reachable geometry files for target {key!r}"
                )
            hits = [m for m in matches if m.file == wanted]
            if len(hits) == 1:
                return hits[0]
            if matches:
                where = ", ".join(sorted({m.file for m in matches}))
                raise UnknownTargetError(
                    f"target {key!r} does not exist in {wanted}; it exists in: {where}"
                )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            listing = "\n".join(f"  - {m.file}:{m.line} (value {m.value!r})" for m in matches)
            raise AmbiguousTargetError(
                f"target {key!r} matches {len(matches)} attributes in different files; "
                f"add 'file: <path>' to the override to choose one:\n{listing}",
                candidates=matches,
            )
        raise UnknownTargetError(self._unknown_message(key))

    def _unknown_message(self, key: str) -> str:
        aliased = self._aliases.get(key)
        if aliased:
            listing = "\n".join(f"  - {p.target}  ({p.file})" for p in aliased)
            return (
                f"target {key!r} is not a canonical reference; the attribute(s) it describes are "
                f"addressed as:\n{listing}"
            )
        close = difflib.get_close_matches(key, self._by_target, n=3, cutoff=0.75)
        hint = f"; did you mean: {', '.join(close)}" if close else ""
        return f"unknown target {key!r}{hint}"


# ---------------------------------------------------------------------------
# reference generation


@dataclass(slots=True)
class _Entry:
    element: etree._Element
    attribute: str
    value: str
    locator: str
    candidates: list[tuple[str, bool]]
    tier: int = 0

    @property
    def text(self) -> str:
        return self.candidates[self.tier][0]

    @property
    def order_sensitive(self) -> bool:
        return self.candidates[self.tier][1]


def _sibling_step(element: etree._Element) -> Step:
    parent = element.getparent()
    if parent is None:  # the document root has no siblings
        return Step(element.tag)
    siblings = [c for c in parent if isinstance(c.tag, str) and c.tag == element.tag]
    if len(siblings) == 1:
        return Step(element.tag)
    name = element.get("name")
    if is_value(name) and sum(1 for s in siblings if s.get("name") == name) == 1:
        return Step(element.tag, "name", name)
    identifier = element.get("id")
    if is_value(identifier) and sum(1 for s in siblings if s.get("id") == identifier) == 1:
        return Step(element.tag, "id", identifier)
    index = next(i for i, s in enumerate(siblings, start=1) if s is element)
    return Step(element.tag, "index", str(index))


def _path_text(head: str, steps: Sequence[Step], attribute: str) -> str:
    return ".".join([head, *(str(s) for s in steps), attribute])


def _lineage(element: etree._Element) -> list[etree._Element]:
    """Return the ancestors-or-self of ``element`` from the document root down."""
    chain = [element]
    parent = element.getparent()
    while parent is not None:
        chain.append(parent)
        parent = parent.getparent()
    chain.reverse()
    return chain


def _discover_file(rel: str, tree: etree._ElementTree) -> tuple[list[_Entry], list[Unaddressable]]:
    root_tag = tree.getroot().tag
    reserved = {CONSTANTS_PREFIX, root_tag}
    entries: list[_Entry] = []
    unaddressable: list[Unaddressable] = []

    for element in iter_elements(tree):
        locator = element_locator(tree, element)
        lineage = _lineage(element)
        if not all(is_name(e.tag) for e in lineage):
            unaddressable.extend(
                Unaddressable(
                    rel, locator, str(attribute), "element or ancestor tag is not a plain name"
                )
                for attribute in element.attrib
            )
            continue
        root_steps = [_sibling_step(e) for e in lineage[1:]]
        root_sensitive = any(s.order_sensitive for s in root_steps)

        # Named ancestors-or-self, nearest first: each is a candidate anchor.
        anchors: list[tuple[int, str]] = []
        for i in range(len(lineage) - 1, -1, -1):
            anchor_name = lineage[i].get("name")
            if anchor_name is not None and is_name(anchor_name) and anchor_name not in reserved:
                anchors.append((i, anchor_name))

        constant_name = element.get("name") if element.tag == "constant" else None
        for attribute, value in element.attrib.items():
            attribute = str(attribute)
            if not is_name(attribute):
                unaddressable.append(
                    Unaddressable(rel, locator, attribute, "attribute name is not a plain name")
                )
                continue
            candidates: list[tuple[str, bool]] = []
            if attribute == "value" and is_name(constant_name):
                candidates.append((f"{CONSTANTS_PREFIX}.{constant_name}", False))
            for index, name in anchors:
                steps = root_steps[index:]
                sensitive = any(s.order_sensitive for s in steps)
                candidates.append((_path_text(name, steps, attribute), sensitive))
                candidates.append(
                    (_path_text(f"{lineage[index].tag}[name={name}]", steps, attribute), sensitive)
                )
            candidates.append((_path_text(root_tag, root_steps, attribute), root_sensitive))
            text = value if isinstance(value, str) else value.decode("utf-8")
            entries.append(_Entry(element, attribute, text, locator, candidates))

    _refine(entries, rel)
    return entries, unaddressable


def _refine(entries: list[_Entry], rel: str) -> None:
    """Bump colliding entries to their next candidate until all references are unique."""
    while True:
        groups: dict[str, list[_Entry]] = defaultdict(list)
        for entry in entries:
            groups[entry.text].append(entry)
        colliding = [e for group in groups.values() if len(group) > 1 for e in group]
        if not colliding:
            return
        progressed = False
        for entry in colliding:
            if entry.tier < len(entry.candidates) - 1:
                entry.tier += 1
                progressed = True
        if not progressed:  # pragma: no cover - positional paths are unique by construction
            raise RuntimeError(f"internal error: unresolvable reference collision in {rel}")


def discover(root: Path, compact: str) -> Catalog:
    """Walk the include graph below ``root`` starting at ``compact`` and build the catalog."""
    graph = walk_includes(root, compact)
    file_entries: list[tuple[str, _Entry]] = []
    unaddressable: list[Unaddressable] = []
    for rel in graph.files:
        entries, bad = _discover_file(rel, graph.trees[rel])
        file_entries.extend((rel, e) for e in entries)
        unaddressable.extend(bad)

    counts: dict[str, int] = defaultdict(int)
    for _, entry in file_entries:
        counts[entry.text] += 1

    parameters: list[Parameter] = []
    aliases: dict[str, list[Parameter]] = defaultdict(list)
    for rel, entry in file_entries:
        parameter = Parameter(
            target=entry.text,
            file=rel,
            attribute=entry.attribute,
            value=entry.value,
            locator=entry.locator,
            line=line_of(entry.element),
            tag=entry.element.tag,
            order_sensitive=entry.order_sensitive,
            ambiguous=counts[entry.text] > 1,
        )
        parameters.append(parameter)
        for text, _ in entry.candidates:
            if text != entry.text:
                aliases[text].append(parameter)

    try:
        file_hashes = {rel: sha256_file(root / rel) for rel in graph.files}
        resource_hashes = {rel: sha256_file(root / rel) for rel in graph.resources}
    except OSError as exc:
        raise IncludeError(f"cannot read {exc.filename or root}: {exc.strerror or exc}") from exc
    return Catalog(
        root=root,
        compact=compact,
        files=graph.files,
        file_hashes=file_hashes,
        parameters=parameters,
        externals=graph.externals,
        edges=graph.edges,
        resources=graph.resources,
        resource_hashes=resource_hashes,
        file_aliases=graph.file_aliases,
        aliases=aliases,
        unaddressable=unaddressable,
    )


__all__ = ["Catalog", "Parameter", "Unaddressable", "discover", "normalize_file"]
