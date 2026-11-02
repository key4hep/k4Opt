"""Load an evaluator class from a file or an importable module."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import sys
import sysconfig
from collections.abc import Collection, Mapping, Sequence
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
# the local modules of the evaluator files loaded, by module name: imported from their source
_LOCAL: dict[str, Path] = {}
# the files of the local modules each class's identity names, by the name it gives them
_HELPER_FILES: WeakKeyDictionary[type, FrozenMap[str, Path]] = WeakKeyDictionary()


def evaluator_source(cls: type) -> dict[str, Json]:
    """Identify the source of an evaluator class without hashing a file that may have changed.

    The identity is taken when :func:`load_evaluator` imports the file that defines the class:
    for ``FILE.py:Class`` the exact bytes executed, for a module that call imported its file
    read right after the import (Python may have executed a cached ``.pyc`` instead).
    ``helpers`` holds the SHA-256 of each local module the file imports, directly or through
    one another, by its path relative to the file's directory: every ``import`` statement
    counts, also one inside a function, but not a module imported by name at run time
    (``importlib``). For a module of a package, the local modules are those of its top-level
    package (and of the reference's), relative to the directory that package is in, and include
    the packages Python executed to reach the class and the module the reference names; a
    top-level module's are those beside it, unless it is installed (in ``site-packages`` or the
    standard library), where they are other distributions. The local modules of ``FILE.py`` are
    imported from the bytes of their source, never a cached ``.pyc``, and identified by those
    bytes; for a module, Python imports them itself and they are read right after the import, as
    its own file is. One imported later (inside a function) is identified by its file when the
    class is loaded (:func:`changed_helpers` sees later edits). ``revision`` is the commit and
    dirty flag of the git checkout that tracks the file, taken at the same time, ``None`` if no
    checkout tracks it; it labels what the hashes do not cover, such as data files. For a class
    whose module was imported before, or reloaded since, only the path is known: the current
    file cannot establish which bytes Python executed. ``reference`` is what loads the class
    again (``/abs/path/file.py:Class`` or ``package.module:Class``), ``None`` for a class
    :func:`load_evaluator` did not load.
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
    path: Path,
    root: Path,
    packages: Collection[str] | None = None,
    name: str | None = None,
    executed: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """The modules under ``root`` that ``path`` imports, directly or through one another, by
    name; with ``packages``, only the modules of those top-level packages, and ``name`` is the
    module ``path`` defines (``None`` for an evaluator file). ``executed`` are other modules
    Python executed to reach it (the packages it is in): they count, and what they import too.

    Found in the source, not by running it, so whatever order evaluators are loaded in and
    whether a function that imports has run yet, the same modules are found; a file that
    cannot be read or parsed imports nothing here (importing it says why)."""
    found = {module: file for module, file in (executed or {}).items() if file != path}
    pending: list[tuple[Path, str | None]] = [(path, name), *((f, m) for m, f in found.items())]
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


def _sha256(path: Path) -> str:
    """The SHA-256 of the file ``path``; ``"unreadable"``, which no file has, if it cannot be
    read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def _hashed(root: Path, modules: Mapping[str, Path]) -> dict[str, str]:
    """The SHA-256 of each of ``modules`` as its file is now, by its path relative to ``root``,
    sorted."""
    return dict(
        sorted((path.relative_to(root).as_posix(), _sha256(path)) for path in modules.values())
    )


class _SourceLoader(importlib.abc.Loader):
    """Executes a local module from the bytes of its source file, as :func:`_load_file` does an
    evaluator file: Python's own import may run a cached ``.pyc`` that a same-sized edit keeping
    the timestamp left stale. The SHA-256 of the bytes compiled is the spec's
    ``loader_state``."""

    def exec_module(self, module: ModuleType) -> None:
        spec = module.__spec__
        assert spec is not None
        assert spec.origin is not None
        data = Path(spec.origin).read_bytes()
        spec.loader_state = hashlib.sha256(data).hexdigest()
        exec(compile(data, spec.origin, "exec", dont_inherit=True), module.__dict__)


class _SourceFinder(importlib.abc.MetaPathFinder):
    """Finds the local modules of the evaluator files loaded (``_LOCAL``), for
    :class:`_SourceLoader`; no other module."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        file = _LOCAL.get(fullname)
        if file is None:
            return None
        package = [str(file.parent)] if file.name == "__init__.py" else None
        return importlib.util.spec_from_file_location(
            fullname, file, loader=_LOADER, submodule_search_locations=package
        )


_LOADER = _SourceLoader()
_FINDER = _SourceFinder()


def _executed(module: ModuleType) -> str | None:
    """The SHA-256 of the source :class:`_SourceLoader` executed ``module`` from; ``None`` for a
    module Python imported otherwise."""
    spec = getattr(module, "__spec__", None)
    if spec is None or spec.loader is not _LOADER:
        return None
    state = spec.loader_state
    assert isinstance(state, str)
    return state


def _read(root: Path, modules: Mapping[str, Path], before: Collection[str]) -> dict[str, str]:
    """The identity of ``modules`` (``helpers``), by path relative to ``root``, sorted.

    A module imported from its source (:class:`_SourceLoader`) runs those bytes: refused if its
    file changed since. One Python imported itself during this load, as it does the modules of
    a ``package.module:Class``, is identified by its file read now, right after the import; one
    it imported before this load (``before``) is refused: which of its code runs is not known.
    One not imported yet is identified by its file now, which :func:`changed_helpers` watches."""
    hashes = {}
    for module, path in modules.items():
        now = _sha256(path)
        loaded = sys.modules.get(module)
        executed = None if loaded is None else _executed(loaded)
        if executed is not None and executed != now:
            raise EvaluatorLoadError(
                f"{path} changed since this process imported it, and Python keeps running the "
                f"module {module!r} it imported then; load the evaluator in a new process"
            )
        if loaded is not None and executed is None and module in before:
            raise EvaluatorLoadError(
                f"{module!r} ({path}) was imported before load_evaluator read it, so which of its "
                "code runs is not known; load the evaluator in a new process, or before importing "
                "the modules it imports"
            )
        hashes[path.relative_to(root).as_posix()] = now
    return dict(sorted(hashes.items()))


def _identified(
    cls: type, identity: FrozenMap[str, Json], root: Path, modules: Mapping[str, Path]
) -> None:
    """Remember ``identity`` for ``cls``, and the files of the ``modules`` it names."""
    _SOURCES[cls] = identity
    _HELPER_FILES[cls] = FrozenMap(
        {path.relative_to(root).as_posix(): path for path in modules.values()}
    )


def changed_helpers(cls: type) -> list[str]:
    """The local modules the identity of ``cls`` names (``helpers`` of
    :func:`evaluator_source`) whose files no longer hold what they did when it was loaded; none
    for a class :func:`load_evaluator` did not identify. A module Python imports later, inside a
    function, would then run other code than the identity says."""
    identity, files = _SOURCES.get(cls), _HELPER_FILES.get(cls)
    if identity is None or files is None:
        return []
    then = identity["helpers"]
    assert isinstance(then, Mapping)
    return [name for name, path in files.items() if _sha256(path) != then.get(name)]


def _package_root(path: Path, name: str) -> Path:
    """The directory the top-level package of the module ``name``, defined in ``path``, is in."""
    levels = name.count(".") + (1 if path.name == "__init__.py" else 0)
    root = path.parent
    for _ in range(levels):
        root = root.parent
    return root


def _installed(directory: Path) -> bool:
    """Whether ``directory`` holds installed distributions or the standard library, whose
    modules are not an evaluator's own."""
    if directory.name in ("site-packages", "dist-packages"):
        return True
    paths = sysconfig.get_paths()
    return any(
        directory == Path(paths[key]).resolve() for key in ("stdlib", "platstdlib") if key in paths
    )


def _prefixes(name: str) -> list[str]:
    """``a``, ``a.b`` and ``a.b.c`` for ``a.b.c``."""
    parts = name.split(".")
    return [".".join(parts[:depth]) for depth in range(1, len(parts) + 1)]


def _identify_import(cls: type, where: str, before: Collection[str]) -> None:
    """Record the source file of a class whose module was imported just now, by the reference
    module ``where``, if readable; ``before`` are the modules imported before."""
    if cls in _SOURCES:  # defined in a FILE.py, identified by the bytes it executed
        return
    try:
        path = inspect.getsourcefile(cls)
    except TypeError:  # defined in a built-in module
        return
    if path is None:
        return
    file, module = Path(path).resolve(), cls.__module__
    if "." in module or file.name == "__init__.py":  # of a package, and so is what it imports
        root = _package_root(file, module)
        packages = {module.partition(".")[0], where.partition(".")[0]}
        names = [n for n in _prefixes(module) + _prefixes(where) if n.partition(".")[0] in packages]
        executed = {n: f for n in names if (f := _module_file(root, n)) is not None}
        local = _local_modules(file, root, packages, module, executed)
    else:  # a top-level module, which imports modules beside it like a file
        root = file.parent
        local = {} if _installed(root) else _local_modules(file, root)
        _check_local(file, local)
    helpers = _read(root, local, before)
    with contextlib.suppress(OSError):
        _identified(cls, _identity(path, file.read_bytes(), helpers, executed=False), root, local)


def _check_local(path: Path, modules: Mapping[str, Path]) -> None:
    """Refuse a module that ``path`` imports from beside it if Python already imported another
    module by that name, or an evaluator file loaded before imports another one by that name
    (perhaps only when it evaluates): the import would give ``path`` that one."""
    for name, local in modules.items():
        loaded = sys.modules.get(name)
        if loaded is not None:
            where = getattr(loaded, "__file__", None)
            other = f"this process imported {name!r} from {where or 'Python itself'} before"
        elif name in _LOCAL:
            where = str(_LOCAL[name])
            other = f"an evaluator file loaded before imports {name!r} from {where}"
        else:
            continue
        if where is None or Path(where).resolve() != local.resolve():
            raise EvaluatorLoadError(
                f"{path} imports {name!r}, which is {local} beside it, but {other}, which Python "
                "would give it; rename one of them, or put the evaluators in packages of their "
                "own (package.module:Class)"
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


def _load_file(path: Path, before: Collection[str]) -> ModuleType:
    """Import ``path`` (absolute) once under a name derived from its location, so repeats share a
    module; ``before`` are the modules imported before."""
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
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    registered = [name for name in local if name not in _LOCAL]
    _LOCAL.update(local)
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
        identity = _identity(str(path), data, _read(path.parent, local, before), executed=True)
        for value in vars(module).values():  # the classes these bytes defined
            if isinstance(value, type) and value.__module__ == name:
                _identified(value, identity, path.parent, local)
    except BaseException as exc:
        del sys.modules[name]
        for local_name in registered:
            del _LOCAL[local_name]
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
    different directories need different names, or packages of their own), and so is one Python
    imported before whose file changed since, or that this function never read. A file is
    executed once per process: loading it again returns the same class, or raises if the file or
    a module it imports changed since.

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
        module = _load_file(path, imported_before)
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
        _identify_import(cls, where, imported_before)
    _REFERENCES[cls] = canonical
    return cls
