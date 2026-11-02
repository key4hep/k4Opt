"""Load an evaluator class from a file or an importable module."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from weakref import WeakKeyDictionary

from k4opt.evaluation.errors import EvaluatorLoadError
from k4opt.evaluation.protocol import EvaluatorProtocol
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.revision import tracked_revision

# by class, not module: importlib.reload() runs new code in the same module object, and the
# classes it creates must not inherit the identity of the code they replace
_SOURCES: WeakKeyDictionary[type, FrozenMap[str, Json]] = WeakKeyDictionary()
# the reference each class was last loaded by, the file as an absolute path
_REFERENCES: WeakKeyDictionary[type, str] = WeakKeyDictionary()
# the identity of each FILE.py this process executed, by the module name it was imported as
_EXECUTED: dict[str, FrozenMap[str, Json]] = {}


def evaluator_source(cls: type) -> dict[str, Json]:
    """Identify the source of an evaluator class without hashing a file that may have changed.

    The identity is taken when :func:`load_evaluator` imports the file that defines the class:
    for ``FILE.py:Class`` the exact bytes executed, for a module that call imported its file
    read right after the import (Python may have executed a cached ``.pyc`` instead).
    ``helpers`` holds the SHA-256 of each local module the file imports, directly or through
    one another, by its path relative to the file's directory (for a module, the directory of
    its top-level package, and only modules of that package): every ``import`` statement
    counts, also one inside a function, but not a module imported by name at run time
    (``importlib``). ``revision`` is the commit and dirty flag of the git checkout that tracks
    the file, taken at the same time, ``None`` if no checkout tracks it; it labels what the
    hashes do not cover, such as data files. For a class whose module was imported before, or
    reloaded since, only the path is known: the current file cannot establish which bytes
    Python executed. ``reference`` is what loads the class again (``/abs/path/file.py:Class``
    or ``package.module:Class``), ``None`` for a class :func:`load_evaluator` did not load.
    """
    identity = _SOURCES.get(cls)
    source: dict[str, Json]
    if identity is not None:
        copy = thaw(identity)  # a plain copy all through: the revision is a mapping too
        assert isinstance(copy, dict)
        source = copy
    else:
        try:
            source = {"path": inspect.getsourcefile(cls)}
        except TypeError:
            source = {"path": None}
    return source | {"reference": _REFERENCES.get(cls)}


def _identity(
    path: str, data: bytes, helpers: Mapping[str, str], *, executed: bool
) -> FrozenMap[str, Json]:
    """``executed``: ``data`` are the bytes Python compiled, not the file read after the import;
    ``helpers``: :func:`_hashed` local modules."""
    digest = hashlib.sha256(data).hexdigest()
    git = tracked_revision(Path(path))
    identity = freeze(
        {
            "path": path,
            "bytes": len(data),
            "sha256": digest,
            "helpers": dict(helpers),
            "executed": executed,
            "revision": None if git is None else git.to_dict(),
        },
        "evaluator source",
        EvaluatorLoadError,
    )
    assert isinstance(identity, FrozenMap)
    return identity


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


def _module_file(root: Path, name: str) -> Path | None:
    """The file that defines the module ``name`` under ``root``, if there is one."""
    *parents, last = name.split(".")
    directory = root.joinpath(*parents)
    for candidate in (directory / f"{last}.py", directory / last / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _local_modules(
    path: Path, root: Path, package: str | None = None, name: str | None = None
) -> dict[str, Path]:
    """The modules under ``root`` that ``path`` imports, directly or through one another, by
    name; with ``package``, only the modules of that top-level package, and ``name`` is the
    module ``path`` defines (``None`` for an evaluator file).

    Found in the source, not by running it, so whatever order evaluators are loaded in and
    whether a function that imports has run yet, the same modules are found; a file that
    cannot be read or parsed imports nothing here (importing it says why)."""
    found: dict[str, Path] = {}
    pending: list[tuple[Path, str | None]] = [(path, name)]
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
                if package is not None and parts[0] != package:
                    continue
                for depth in range(1, len(parts) + 1):  # importing a.b imports a first
                    module = ".".join(parts[:depth])
                    if module in found:
                        continue
                    module_path = _module_file(root, module)
                    if module_path is None or module_path == path:
                        continue
                    found[module] = module_path
                    pending.append((module_path, module))
    return found


def _hashed(root: Path, modules: Mapping[str, Path]) -> dict[str, str]:
    """The SHA-256 of each of ``modules`` by its path relative to ``root``, sorted; one that
    cannot be read is ``"unreadable"``, so that it differs from any file."""
    hashes = {}
    for path in modules.values():
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = "unreadable"
        hashes[path.relative_to(root).as_posix()] = digest
    return dict(sorted(hashes.items()))


def _package_root(path: Path, name: str) -> Path:
    """The directory the top-level package of the module ``name``, defined in ``path``, is in."""
    levels = name.count(".") + (1 if path.name == "__init__.py" else 0)
    root = path.parent
    for _ in range(levels):
        root = root.parent
    return root


def _identify_import(cls: type) -> None:
    """Record the source file of a class whose module was imported just now, if readable."""
    if cls in _SOURCES:  # defined in a FILE.py, identified by the bytes it executed
        return
    try:
        path = inspect.getsourcefile(cls)
    except TypeError:  # defined in a built-in module
        return
    if path is not None:
        file = Path(path)
        root = _package_root(file, cls.__module__)
        helpers = _local_modules(file, root, cls.__module__.partition(".")[0], cls.__module__)
        with contextlib.suppress(OSError):
            _SOURCES[cls] = _identity(
                path, file.read_bytes(), _hashed(root, helpers), executed=False
            )


def _check_local(path: Path, modules: Mapping[str, Path]) -> None:
    """Refuse a module that ``path`` imports from beside it if Python already imported another
    module by that name: the import would give ``path`` that one."""
    for name, local in modules.items():
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        where = getattr(loaded, "__file__", None)
        if where is None or Path(where).resolve() != local.resolve():
            raise EvaluatorLoadError(
                f"{path} imports {name!r}, which is {local} beside it, but this process imported "
                f"{name!r} from {where or 'Python itself'} before, which Python would give it; "
                "rename one of them, or put the evaluators in packages of their own "
                "(package.module:Class)"
            )


def _changed(path: Path, identity: Mapping[str, Json]) -> str | None:
    """What changed in ``path``, or the local modules it imports, since it was executed with
    ``identity``; ``None`` if nothing did."""
    root = path.parent
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"it cannot be read ({exc})"
    if digest != identity["sha256"]:
        return "its bytes changed"
    helpers = _hashed(root, _local_modules(path, root))
    then = identity["helpers"]
    assert isinstance(then, Mapping)
    names = dict.fromkeys([*then, *helpers])
    changed = [name for name in names if then.get(name) != helpers.get(name)]
    return f"the modules it imports changed: {', '.join(changed)}" if changed else None


def _load_file(path: Path) -> ModuleType:
    """Import ``path`` (absolute) once under a name derived from its location, so repeats share a
    module."""
    if not path.is_file():
        raise EvaluatorLoadError(f"evaluator file {path} does not exist")
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    name = f"k4opt_evaluator_{path.stem}_{digest}"
    if name in sys.modules:
        change = _changed(path, _EXECUTED[name])
        if change is not None:
            raise EvaluatorLoadError(
                f"{path} changed since this process imported it ({change}); Python cannot "
                "import it again, so load it in a new process"
            )
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluatorLoadError(f"{path} cannot be imported as a Python module")
    local = _local_modules(path, path.parent)
    _check_local(path, local)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before executing: dataclasses and pickling look it up
    # like ``python file.py``, modules next to the file are importable, also later (an import
    # inside a method); a failed import leaves the path as it was
    directory = str(path.parent)
    added = directory not in sys.path
    if added:
        sys.path.insert(0, directory)
    try:
        # Read, hash and compile the same bytes. exec_module() may use a stale .pyc
        # when a same-sized edit preserves the source's timestamp.
        data = path.read_bytes()
        code = compile(data, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        _check_local(path, local)  # also a directory already later on the path than another
        identity = _identity(str(path), data, _hashed(path.parent, local), executed=True)
        for value in vars(module).values():  # the classes these bytes defined
            if isinstance(value, type) and value.__module__ == name:
                _SOURCES[value] = identity
    except BaseException as exc:
        del sys.modules[name]
        if added:
            sys.path.remove(directory)
        if isinstance(exc, EvaluatorLoadError):
            raise
        if not isinstance(exc, Exception):
            raise
        raise EvaluatorLoadError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    _EXECUTED[name] = identity
    return module


def load_evaluator(reference: str) -> type[EvaluatorProtocol]:
    """The evaluator class that ``reference`` names, ``FILE.py:Class`` or ``module:Class``.

    ``FILE.py`` is a path (a leading ``~`` is expanded) and its directory is added to
    ``sys.path``, so the file can import modules next to it; anything else is an importable
    module name. Modules are cached by name for the process, so a file that imports a module
    beside it by a name the process imported from elsewhere is refused (helpers of evaluators in
    different directories need different names, or packages of their own). A file is executed
    once per process: loading it again returns the same class, or raises if the file or a module
    it imports changed since.

    The class must have an ``evaluate`` method; it is not created here. Whoever evaluates with
    it creates an instance (without arguments) whenever it needs one, which must have ``name``
    and ``version`` too. The file is executed as ordinary code: only name evaluators you trust.
    Raises :class:`EvaluatorLoadError`.
    """
    where, separator, attribute = reference.rpartition(":")
    if not separator or not where or not attribute.isidentifier():
        raise EvaluatorLoadError(
            f"evaluator {reference!r}: expected FILE.py:Class or package.module:Class"
        )
    imported_before = set(sys.modules)
    if where.endswith(".py") or "/" in where:
        path = Path(where).expanduser().resolve()
        module = _load_file(path)
        canonical = f"{path}:{attribute}"
    else:
        canonical = reference
        try:
            module = importlib.import_module(where)
        except Exception as exc:
            raise EvaluatorLoadError(
                f"importing {where!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
    cls = getattr(module, attribute, None)
    if cls is None:
        raise EvaluatorLoadError(f"{where} has no {attribute!r}")
    if not isinstance(cls, type):
        raise EvaluatorLoadError(f"{attribute} is not a class: name the class of an evaluator")
    if not callable(getattr(cls, "evaluate", None)):
        raise EvaluatorLoadError(f"{attribute} is not an evaluator: it has no evaluate() method")
    if cls.__module__ not in imported_before:
        _identify_import(cls)
    _REFERENCES[cls] = canonical
    return cls
