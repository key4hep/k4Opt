"""The :class:`GeometrySource` facade: one compact XML entry point and the tree it lives in.

A geometry is named the way Key4hep tools name it: a path such as
``FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml`` relative to ``$K4GEO`` (set by
the Key4hep stack), an absolute path, or a path using environment variables
(``$K4GEO/...``). The *root* is the directory that contains the geometry and all
of its local includes; it is resolved automatically and only needs to be given
explicitly for trees that are neither ``$K4GEO`` nor a git checkout.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from k4opt.geometry.discovery import Catalog, discover
from k4opt.geometry.errors import SourceError
from k4opt.geometry.overrides import Override, ResolvedOverride, resolve_overrides

if TYPE_CHECKING:  # pragma: no cover - typing only
    from k4opt.geometry.materialize import MaterializationResult

#: Environment variable naming the installed k4geo tree (set by the Key4hep stack).
K4GEO_ENV = "K4GEO"


class GitInfo:
    """Git revision of the checkout containing the geometry root."""

    __slots__ = ("commit", "dirty")

    def __init__(self, commit: str, dirty: bool) -> None:
        self.commit = commit
        self.dirty = dirty
        """True when tracked files have uncommitted modifications (untracked files are ignored)."""

    def to_dict(self) -> dict[str, Any]:
        return {"commit": self.commit, "dirty": self.dirty}

    def __repr__(self) -> str:
        return f"GitInfo(commit={self.commit!r}, dirty={self.dirty!r})"


def _git(args: list[str], cwd: Path, timeout: int = 30) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def detect_git(root: Path) -> GitInfo | None:
    """Return the git revision of ``root`` or ``None`` when unavailable."""
    head = _git(["rev-parse", "HEAD"], root)
    if head is None:
        return None
    status = _git(["status", "--porcelain", "--untracked-files=no"], root, timeout=60)
    if status is None:
        return None
    return GitInfo(commit=head.strip(), dirty=bool(status.strip()))


def git_toplevel(path: Path) -> Path | None:
    """Return the real path of the git work tree containing ``path``, if any."""
    top = _git(["rev-parse", "--show-toplevel"], path)
    if not top or not top.strip():
        return None
    return Path(os.path.realpath(top.strip()))


def _is_within(root: Path, path: Path) -> bool:
    try:
        return os.path.commonpath([str(root), str(path)]) == str(root)
    except ValueError:  # pragma: no cover - different drives on Windows
        return False


def _expand(text: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(text))
    if "$" in expanded:
        raise SourceError(
            f"geometry path {text!r} references an environment variable that is not set"
        )
    return expanded


def _detect_root(compact_real: Path) -> Path:
    """Root of an absolute path: $K4GEO if inside it, else its git checkout, else its directory."""
    k4geo = os.environ.get(K4GEO_ENV)
    if k4geo and Path(k4geo).is_dir():
        k4geo_real = Path(os.path.realpath(k4geo))
        if _is_within(k4geo_real, compact_real):
            return k4geo_real
    top = git_toplevel(compact_real.parent)
    if top is not None and _is_within(top, compact_real):
        return top
    return compact_real.parent


def resolve_geometry(
    compact: str | os.PathLike[str], root: str | os.PathLike[str] | None
) -> tuple[Path, str]:
    """Resolve a geometry specification to ``(real root directory, root-relative compact path)``.

    * relative ``compact`` without ``root``: relative to ``$K4GEO``;
    * absolute ``compact`` without ``root``: root is ``$K4GEO`` when the file lies inside it,
      otherwise the enclosing git checkout, otherwise the file's directory;
    * with ``root``: ``compact`` is relative to ``root`` (or absolute and inside it).
    Environment variables (``$K4GEO/...``) are expanded in both.
    """
    compact_text = _expand(os.fspath(compact)).replace("\\", "/")
    if not compact_text.strip():
        raise SourceError("geometry path must not be empty")
    absolute = os.path.isabs(compact_text)

    if root is not None:
        root_path = Path(_expand(os.fspath(root)))
        if not root_path.is_dir():
            raise SourceError(f"geometry root {os.fspath(root)!r} is not an existing directory")
        root_real = Path(os.path.realpath(root_path))
    elif not absolute:
        k4geo = os.environ.get(K4GEO_ENV)
        if not k4geo:
            raise SourceError(
                f"geometry {compact_text!r} is relative but ${K4GEO_ENV} is not set; "
                "source the Key4hep stack (setup.sh), give an absolute path, or give a root"
            )
        if not Path(k4geo).is_dir():
            raise SourceError(f"${K4GEO_ENV}={k4geo!r} is not an existing directory")
        root_real = Path(os.path.realpath(k4geo))
    else:
        compact_real = Path(os.path.realpath(compact_text))
        if not compact_real.is_file():
            raise SourceError(f"geometry {compact_text!r} not found")
        root_real = _detect_root(compact_real)

    if absolute:
        compact_real = Path(os.path.realpath(compact_text))
        try:
            relative = compact_real.relative_to(root_real)
        except ValueError as exc:
            raise SourceError(
                f"geometry {compact_text!r} does not lie inside the root {root_real}"
            ) from exc
    else:
        relative = Path(os.path.normpath(compact_text))
        if relative.parts and relative.parts[0] == "..":
            raise SourceError(f"geometry {compact_text!r} escapes the root {root_real}")
    compact_rel = PurePosixPath(relative).as_posix()
    if not (root_real / compact_rel).is_file():
        raise SourceError(f"geometry {compact_rel!r} not found below {root_real}")
    return root_real, compact_rel


class GeometrySource:
    """A detector description: a compact XML entry point inside a local tree.

    ``compact`` is stored relative to ``root`` with POSIX separators and ``root``
    is an absolute, symlink-resolved directory; see :func:`resolve_geometry` for
    how both are determined from what the caller passed.
    """

    __slots__ = ("_compact", "_root")

    def __init__(
        self, compact: str | os.PathLike[str], root: str | os.PathLike[str] | None = None
    ) -> None:
        self._root, self._compact = resolve_geometry(compact, root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def compact(self) -> str:
        return self._compact

    @property
    def compact_path(self) -> Path:
        """Absolute path of the compact entry file in the source tree."""
        return self._root / self._compact

    def __repr__(self) -> str:
        return f"GeometrySource(compact={self._compact!r}, root={str(self._root)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeometrySource):
            return NotImplemented
        return (self._root, self._compact) == (other._root, other._compact)

    def __hash__(self) -> int:
        return hash((self._root, self._compact))

    def discover(self) -> Catalog:
        """Parse the compact file and its reachable local includes into a fresh catalog."""
        return discover(self._root, self._compact)

    def git_info(self) -> GitInfo | None:
        return detect_git(self._root)

    def resolve(
        self, overrides: Iterable[Override], catalog: Catalog | None = None
    ) -> tuple[ResolvedOverride, ...]:
        """Resolve overrides to concrete attributes without writing anything (dry run)."""
        return resolve_overrides(self._catalog_for(catalog), overrides)

    def fingerprint(self, overrides: Iterable[Override], catalog: Catalog | None = None) -> str:
        """Deterministic identity of the geometry ``overrides`` would produce, without writing it.

        Equal to the ``fingerprint`` recorded in the manifest by :meth:`materialize`.
        """
        from k4opt.geometry.materialize import compute_fingerprint

        catalog = self._catalog_for(catalog)
        return compute_fingerprint(catalog, resolve_overrides(catalog, overrides))

    def materialize(
        self,
        overrides: Iterable[Override],
        output: str | os.PathLike[str],
        catalog: Catalog | None = None,
    ) -> MaterializationResult:
        """Create an isolated copy of the reachable geometry files with ``overrides`` applied."""
        from k4opt.geometry.materialize import materialize

        catalog = self._catalog_for(catalog)
        resolved = resolve_overrides(catalog, overrides)
        return materialize(self, catalog, resolved, output)

    def _catalog_for(self, catalog: Catalog | None) -> Catalog:
        if catalog is None:
            return self.discover()
        if catalog.root != self._root or catalog.compact != self._compact:
            raise SourceError(
                "the supplied catalog was discovered from a different geometry source"
            )
        return catalog
