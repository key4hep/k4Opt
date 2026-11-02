"""Internal: which local modules evaluator code imports, and importing them from their source.

Which modules an evaluator imports from beside it, or from its own package, is read from the
source, not found by running it (:func:`local_modules`); where a module's file is, Python's path
finder says without importing anything (:func:`locate`). The modules :func:`serve` is given are
imported from the bytes of their source, never from a cached ``.pyc``, which a same-sized edit
that keeps the timestamp leaves stale (:class:`SourceLoader`); :func:`executed` says which bytes
those were.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import sysconfig
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from types import CodeType, ModuleType

# the modules imported from their source, by name
_SERVED: dict[str, Path] = {}


def prefixes(name: str) -> list[str]:
    """``a``, ``a.b`` and ``a.b.c`` for ``a.b.c``."""
    parts = name.split(".")
    return [".".join(parts[:depth]) for depth in range(1, len(parts) + 1)]


def _imported(node: ast.AST, name: str | None, package: bool) -> list[str]:
    """The modules an import statement ``node`` imports, absolute; ``name`` is the module the
    statement is in (``None`` for an evaluator file, which has no package to be relative to),
    ``package`` whether that module is a package's ``__init__``. ``from a import b`` imports
    ``a`` and, if it is one, the submodule ``a.b``."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    base = node.module or ""
    if node.level:
        if name is None:
            return []
        parts = name.split(".") if package else name.split(".")[:-1]
        if node.level - 1 >= len(parts):
            return []
        parent = ".".join(parts[: len(parts) - (node.level - 1)])
        base = f"{parent}.{base}" if base else parent
    if not base:
        return []
    return [base, *(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")]


def module_file(root: Path, name: str) -> Path | None:
    """The file that defines the module ``name`` under ``root``, if there is one."""
    *parents, last = name.split(".")
    directory = root.joinpath(*parents)
    for candidate in (directory / f"{last}.py", directory / last / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def local_modules(
    path: Path | None,
    root: Path,
    packages: Collection[str] | None = None,
    name: str | None = None,
    executed: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """The modules under ``root`` that ``path`` imports, directly or through one another, by
    name; with ``packages``, only the modules of those top-level packages, and ``name`` is the
    module ``path`` defines (``None`` for an evaluator file). ``executed`` are other modules
    Python executes to reach it (the packages it is in): they count, and what they import too.

    Found in the source, not by running it, so whatever order evaluators are loaded in and
    whether a function that imports has run yet, the same modules are found; a file that
    cannot be read or parsed imports nothing here (importing it says why)."""
    found = {module: file for module, file in (executed or {}).items() if file != path}
    pending: list[tuple[Path, str | None]] = [(f, m) for m, f in found.items()]
    if path is not None:
        pending.append((path, name))
    while pending:
        file, module_name = pending.pop()
        try:
            tree = ast.parse(file.read_bytes(), str(file))
        except (OSError, SyntaxError, ValueError):
            continue
        is_package = file.name == "__init__.py"
        for node in ast.walk(tree):
            for imported in _imported(node, module_name, is_package):
                parts = imported.split(".")
                if packages is not None and parts[0] not in packages:
                    continue
                for module in prefixes(imported):  # importing a.b imports a first
                    if module in found:
                        continue
                    module_path = module_file(root, module)
                    if module_path is None or module_path == path:
                        continue
                    found[module] = module_path
                    pending.append((module_path, module))
    return found


def package_root(path: Path, name: str) -> Path:
    """The directory the top-level package of the module ``name``, defined in ``path``, is in."""
    levels = name.count(".") + (1 if path.name == "__init__.py" else 0)
    root = path.parent
    for _ in range(levels):
        root = root.parent
    return root


def installed(directory: Path) -> bool:
    """Whether ``directory`` holds installed distributions or the standard library, whose
    modules are not an evaluator's own."""
    if directory.name in ("site-packages", "dist-packages"):
        return True
    paths = sysconfig.get_paths()
    return any(
        directory == Path(paths[key]).resolve() for key in ("stdlib", "platstdlib") if key in paths
    )


def locate(name: str) -> dict[str, Path]:
    """The source file of the module ``name`` and of each package it is in, by module name, as
    Python's path finder finds them on ``sys.path``, without importing any; up to the first that
    is no source file there (an extension module, a namespace package, or none at all)."""
    found: dict[str, Path] = {}
    search: list[str] | None = None
    for prefix in prefixes(name):
        spec = importlib.machinery.PathFinder.find_spec(prefix, search)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            break
        found[prefix] = Path(spec.origin)
        if spec.submodule_search_locations is None:
            break
        search = list(spec.submodule_search_locations)
    return found


class SourceLoader(importlib.machinery.SourceFileLoader):
    """Python's loader of a source file, which compiles the bytes of the source every time,
    never reading or writing a cached ``.pyc``; ``sha256`` is the SHA-256 of the bytes it
    compiled last. All else, package resources (``importlib.resources``, ``pkgutil.get_data``)
    included, is as for any module imported from a source file."""

    sha256: str | None = None

    def get_code(self, fullname: str) -> CodeType:
        data = self.get_data(self.path)
        self.sha256 = hashlib.sha256(data).hexdigest()
        return compile(data, self.path, "exec", dont_inherit=True)


class _SourceFinder(importlib.abc.MetaPathFinder):
    """Finds the modules :func:`serve` was given, for :class:`SourceLoader`; no other module."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        file = _SERVED.get(fullname)
        if file is None:
            return None
        loader = SourceLoader(fullname, str(file))
        return importlib.util.spec_from_file_location(fullname, file, loader=loader)


_FINDER = _SourceFinder()


def served(name: str) -> Path | None:
    """The file the module ``name`` is imported from, if :func:`serve` was given it."""
    return _SERVED.get(name)


def serve(modules: Mapping[str, Path]) -> list[str]:
    """Import ``modules`` (by name) from their source from now on, wherever they are imported;
    the names that were not served before, which :func:`unserve` takes back."""
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    new = [name for name in modules if name not in _SERVED]
    _SERVED.update(modules)
    return new


def unserve(names: Collection[str]) -> None:
    """Let Python import ``names`` itself again (after a failed load)."""
    for name in names:
        _SERVED.pop(name, None)


def executed(module: ModuleType) -> str | None:
    """The SHA-256 of the source :class:`SourceLoader` executed ``module`` from; ``None`` for a
    module Python imported otherwise."""
    spec = getattr(module, "__spec__", None)
    if spec is None or not isinstance(spec.loader, SourceLoader):
        return None
    return spec.loader.sha256
