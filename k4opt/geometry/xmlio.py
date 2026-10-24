"""Secure XML parsing and serialization helpers built on lxml.

All parsing in k4opt goes through :func:`make_parser`: entities are never
expanded, DTDs are never loaded and no network access is attempted.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from lxml import etree

from k4opt.geometry.errors import IncludeError

XML_DECLARATION_PREFIX = b"<?xml"
UTF8_BOM = b"\xef\xbb\xbf"


def make_parser() -> etree.XMLParser:
    """Return a hardened parser that preserves comments, whitespace and CDATA."""
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        dtd_validation=False,
        huge_tree=False,
        remove_blank_text=False,
        remove_comments=False,
        remove_pis=False,
        strip_cdata=False,
        collect_ids=False,
    )


def parse_xml_bytes(data: bytes, *, label: str) -> etree._ElementTree:
    """Parse ``data`` into an element tree, raising :class:`IncludeError` on syntax errors."""
    try:
        root = etree.fromstring(data, parser=make_parser())
    except etree.XMLSyntaxError as exc:
        raise IncludeError(f"{label}: not well-formed XML: {exc}") from exc
    return root.getroottree()


def parse_xml_file(path: Path, *, label: str | None = None) -> etree._ElementTree:
    """Parse the XML file at ``path``."""
    label = label or str(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IncludeError(f"{label}: cannot read file: {exc}") from exc
    return parse_xml_bytes(data, label=label)


_DECLARATION_RE = re.compile(rb"\A\s*<\?xml\s(?P<body>.*?)\?>", re.DOTALL)
_STANDALONE_RE = re.compile(rb"standalone\s*=\s*['\"](yes|no)['\"]")


def xml_declaration(data: bytes) -> tuple[bool, bool | None]:
    """Sniff the prolog: (has an XML declaration, explicit ``standalone`` value or None)."""
    if data.startswith(UTF8_BOM):
        data = data[len(UTF8_BOM) :]
    match = _DECLARATION_RE.match(data)
    if match is None:
        return False, None
    standalone = _STANDALONE_RE.search(match.group("body"))
    return True, None if standalone is None else standalone.group(1) == b"yes"


def serialize_tree(
    tree: etree._ElementTree, *, declaration: bool, standalone: bool | None = None
) -> bytes:
    """Serialize ``tree`` keeping its original encoding and declaration.

    lxml re-serializes the whole document: comments, text, element order,
    attribute order and namespaces are preserved, but whitespace *inside* start
    tags and quote styles are normalized. This is syntactically equivalent XML,
    not a byte-identical copy.
    """
    encoding = tree.docinfo.encoding or "UTF-8"
    if declaration and standalone is not None:
        data = etree.tostring(tree, encoding=encoding, xml_declaration=True, standalone=standalone)
    else:
        data = etree.tostring(tree, encoding=encoding, xml_declaration=declaration)
    if not isinstance(data, bytes):  # pragma: no cover - a named encoding always yields bytes
        raise TypeError(f"expected bytes from lxml, got {type(data).__name__}")
    return data


def element_locator(tree: etree._ElementTree, element: etree._Element) -> str:
    """Return the structural XPath identifying ``element`` within ``tree``."""
    return tree.getpath(element)


def find_by_locator(tree: etree._ElementTree, locator: str) -> etree._Element:
    """Return the single element addressed by a locator from :func:`element_locator`."""
    root = tree.getroot()
    namespaces = {k: v for k, v in root.nsmap.items() if k}
    try:
        matches = tree.xpath(locator, namespaces=namespaces)
    except etree.XPathError as exc:  # pragma: no cover - locators are generated internally
        raise IncludeError(f"invalid internal locator {locator!r}: {exc}") from exc
    if (
        not isinstance(matches, list)
        or len(matches) != 1
        or not isinstance(matches[0], etree._Element)
    ):
        raise IncludeError(f"internal locator {locator!r} did not identify exactly one element")
    return matches[0]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_of(element: etree._Element) -> int | None:
    """Source line of ``element`` or ``None`` when lxml has no line information."""
    line = element.sourceline
    return line if isinstance(line, int) else None


def iter_elements(tree: etree._ElementTree) -> Iterator[etree._Element]:
    """Yield every real element (no comments or processing instructions) in document order."""
    for node in tree.iter():
        if isinstance(node.tag, str):
            yield node
