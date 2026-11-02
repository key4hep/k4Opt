"""Load an evaluator class from a file or an importable module, and identify its source."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Collection, Mapping
from pathlib import Path
from types import ModuleType
from weakref import WeakKeyDictionary

from k4opt.evaluation._imports import (
    executed,
    installed,
    local_modules,
    locate,
    package_root,
    prefixes,
    resolve,
    serve,
    served,
    unserve,
)
from k4opt.evaluation.errors import EvaluatorLoadError
from k4opt.evaluation.protocol import EvaluatorProtocol
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.revision import tracked_revision

# by class, not module: importlib.reload() runs new code in the same module object, and the
# classes it creates must not inherit the identity of the code they replace
_SOURCES: WeakKeyDictionary[type, FrozenMap[str, Json]] = WeakKeyDictionary()
# the reference each class was last loaded by, the file as an absolute path
_REFERENCES: WeakKeyDictionary[type, str] = WeakKeyDictionary()
# the identity of each FILE.py this process executed, and the local modules it names, by the
# module name it was imported as
_EXECUTED: dict[str, tuple[FrozenMap[str, Json], Mapping[str, Path]]] = {}
# the local modules each class's identity names: by the name it gives them, the module and its file
_HELPER_FILES: WeakKeyDictionary[type, FrozenMap[str, tuple[str, Path]]] = WeakKeyDictionary()
# the module each class's reference names, whose globals its methods read, and the spec it had
# then: importlib.reload() executes the module again in place, under a new spec
_ENTRIES: WeakKeyDictionary[type, tuple[ModuleType, object]] = WeakKeyDictionary()


def evaluator_source(cls: type) -> dict[str, Json]:
    """Identify the source of an evaluator class without hashing a file that may have changed.

    The identity is taken when :func:`load_evaluator` imports the file its reference names
    (``FILE.py``, or the file of the module), also when that file imports the class from elsewhere
    and may change it: the exact bytes executed (``executed``), which it imports from the source,
    never from a cached ``.pyc``; for a module that only Python's own import finds (not a source
    file on ``sys.path``), the file read right after the import (Python may have executed a cached
    ``.pyc`` instead). ``helpers`` holds the SHA-256 of each local module the file imports, directly
    or through one another, the module defining the class among them, by its path relative to the
    file's directory: every ``import`` statement counts, also one inside a function, but not a
    module imported by name at run time (``importlib``). For a module of a package, the local
    modules are those of its top-level package (and of the class's), relative to the directory that
    package is in, and include the packages Python executes to reach it; a top-level module's are
    those beside it, unless it is installed (in ``site-packages`` or the standard library), where
    they are other distributions. Local modules are imported from their source too, and identified
    by the bytes executed. One imported later (inside a function) is identified by its file when the
    class is loaded (:func:`changed_code` sees later edits). ``revision`` is the commit and dirty
    flag of the git checkout that tracks the file, taken at the same time, ``None`` if no checkout
    tracks it; it labels what the hashes do not cover, such as data files. For a class whose module
    was imported before, or reloaded since, only the path is known: the current file cannot
    establish which bytes Python executed. ``reference`` is what loads the class again
    (``/abs/path/file.py:Class`` or ``package.module:Class``), ``None`` for a class
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


def _read(
    root: Path, modules: Mapping[str, Path], before: Collection[str]
) -> tuple[dict[str, str], bool]:
    """The identity of ``modules`` (``helpers``), by path relative to ``root``, sorted, and
    whether each imported one is known to run the bytes it names.

    A module imported from its source runs those bytes: refused if its file changed since. One
    imported otherwise during this load (by Python itself, or another import hook that comes
    first) is identified by its file read now, right after the import, which it may not have run
    (a cached ``.pyc``); one imported so before this load (``before``) is refused: which of its
    code runs is not known, and so is one imported from another file, or none. One not imported
    yet is identified by its file now, which :func:`changed_code` watches."""
    hashes, verified = {}, True
    for module, path in modules.items():
        now = _sha256(path)
        loaded = sys.modules.get(module)
        ran = None if loaded is None else executed(loaded)
        imported = None if loaded is None else getattr(loaded, "__file__", None)
        if loaded is not None and (imported is None or Path(imported).resolve() != path.resolve()):
            raise EvaluatorLoadError(
                f"{module!r} is {path}, but Python imported it from {imported or 'no file'} (an "
                "import hook before Python's path finder, or a directory earlier on sys.path); "
                "rename one of them, or put the evaluator in a package of its own"
            )
        if ran is not None and ran != now:
            raise EvaluatorLoadError(
                f"{path} changed since this process imported it, and Python keeps running the "
                f"module {module!r} it imported then; load the evaluator in a new process"
            )
        if loaded is not None and ran is None and module in before:
            raise EvaluatorLoadError(
                f"{module!r} ({path}) was imported before load_evaluator read it, so which of its "
                "code runs is not known; load the evaluator in a new process, or before importing "
                "the modules it imports"
            )
        verified = verified and (loaded is None or ran is not None)
        hashes[path.relative_to(root).as_posix()] = now
    return dict(sorted(hashes.items())), verified


def _identified(
    cls: type,
    identity: FrozenMap[str, Json],
    root: Path,
    modules: Mapping[str, Path],
    entry: ModuleType,
) -> None:
    """Remember ``identity`` for ``cls``, the ``modules`` it names and the module its reference
    names (``entry``)."""
    _SOURCES[cls] = identity
    _ENTRIES[cls] = entry, entry.__spec__
    _HELPER_FILES[cls] = FrozenMap(
        {path.relative_to(root).as_posix(): (name, path) for name, path in modules.items()}
    )


def changed_code(cls: type) -> list[str]:
    """What runs for ``cls`` and is no longer what its identity says: the module its reference
    names if it was reloaded since (``importlib.reload``, which executes it again in place, so
    that the class's methods read new globals), and the local modules the identity names
    (``helpers`` of :func:`evaluator_source`) whose files no longer hold what they did when it
    was loaded, that
    Python imported from another file since (one earlier on ``sys.path`` by then), or from none,
    or, for an identity of the code that runs (``executed``), that something else than k4opt's
    source loader imported, or from other bytes; none for a class :func:`load_evaluator` did not
    identify. A module Python imports later, inside a function, would then run other code than
    the identity says."""
    identity, files = _SOURCES.get(cls), _HELPER_FILES.get(cls)
    if identity is None or files is None:
        return []
    then = identity["helpers"]
    assert isinstance(then, Mapping)
    verified = identity.get("executed") is True
    changed = []
    entry, spec = _ENTRIES.get(cls, (None, None))
    if entry is not None and entry.__spec__ is not spec:
        changed.append(f"{Path(str(identity['path'])).name} (reloaded)")
    for key, (module, path) in files.items():
        loaded = sys.modules.get(module)
        imported = getattr(loaded, "__file__", None)
        ran = None if loaded is None else executed(loaded)
        if loaded is not None and (imported is None or Path(imported).resolve() != path.resolve()):
            changed.append(f"{key} (imported from {imported or 'no file'})")
        elif loaded is not None and verified and ran is None:
            changed.append(f"{key} (imported by another import hook, perhaps from a cached .pyc)")
        elif (loaded is not None and verified and ran != then.get(key)) or (
            _sha256(path) != then.get(key)
        ):
            changed.append(key)
    return changed


def _check_local(importer: object, modules: Mapping[str, Path]) -> None:
    """Refuse a module that ``importer`` imports from beside it, or from its package, if Python
    already imported another module by that name, or evaluator code loaded before imports
    another one by that name (perhaps only when it evaluates): the import would give
    ``importer`` that one."""
    for name, local in modules.items():
        loaded = sys.modules.get(name)
        if loaded is not None:
            where = getattr(loaded, "__file__", None)
            other = f"this process imported {name!r} from {where or 'no file'} before"
        elif (file := served(name)) is not None:
            where = str(file)
            other = f"evaluator code loaded before imports {name!r} from {where}"
        else:
            continue
        if where is None or Path(where).resolve() != local.resolve():
            raise EvaluatorLoadError(
                f"{importer} imports {name!r}, which is {local}, but {other}, which Python would "
                "give it; rename one of them, or put the evaluators in packages of their own "
                "(package.module:Class)"
            )


def _planned(where: str) -> dict[str, Path]:
    """The modules to import from their source for the module ``where``: it and the packages it
    is in that have a source file (a namespace package has none), as Python's path finder finds
    them, and what they import of their top-level package or, for a top-level module that is not
    installed, from beside it."""
    located = locate(where)
    file = located.get(where)
    if file is None:  # no source file Python's path finder imports: Python imports it itself
        return {}
    if "." in where or file.name == "__init__.py":
        top = where.partition(".")[0]
        return local_modules(None, package_root(file, where), {top}, executed=located)
    if installed(file.parent):
        return located
    return local_modules(None, file.parent, executed=located)


def _import(where: str) -> ModuleType:
    """Import the module ``where`` (a name), its local modules from their source where Python's
    path finder finds them as source files."""
    planned = {} if where in sys.modules else _planned(where)
    _check_local(where, planned)
    added = serve(planned)
    try:
        return importlib.import_module(where)
    except Exception as exc:
        unserve(added)
        raise EvaluatorLoadError(
            f"importing {where!r} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _identify_import(cls: type, where: str, before: Collection[str]) -> None:
    """Record the source of a class the module ``where``, imported just now, gives: that module's
    file, also when it imports the class from elsewhere and may change it, and the local modules
    it imports (the class's own among them); ``before`` are the modules imported before."""
    path = getattr(sys.modules.get(where), "__file__", None)
    if path is None or not path.endswith(".py"):  # built in, or no source file to identify
        return
    file = Path(path).resolve()
    if "." in where or file.name == "__init__.py":  # of a package, and so is what it imports
        root = package_root(file, where)
        packages = {where.partition(".")[0], cls.__module__.partition(".")[0]}
        names = [
            n for n in prefixes(where) + prefixes(cls.__module__) if n.split(".")[0] in packages
        ]
        reached = {n: f for n in names if (f := resolve(root, n)) is not None}
        local = local_modules(file, root, packages, where, reached)
    else:  # a top-level module, which imports modules beside it like a file
        root = file.parent
        local = {} if installed(root) else local_modules(file, root)
    helpers, verified = _read(root, local, before)
    ran = executed(sys.modules[where])
    try:
        data = file.read_bytes()
    except OSError:
        return
    if ran is not None and hashlib.sha256(data).hexdigest() != ran:
        raise EvaluatorLoadError(
            f"{file} changed while it was imported; load the evaluator in a new process"
        )
    identity = _identity(path, data, helpers, executed=verified and ran is not None)
    _identified(cls, identity, root, local, sys.modules[where])


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
    helpers = _hashed(root, local_modules(path, root))
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
        change = _changed(path, _EXECUTED[name][0])
        if change is not None:
            raise EvaluatorLoadError(
                f"{path} changed since this process imported it ({change}); Python cannot "
                "import it again, so load it in a new process"
            )
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluatorLoadError(f"{path} cannot be imported as a Python module")
    module = importlib.util.module_from_spec(spec)
    # like ``python file.py``, modules next to the file are importable, also later (an import
    # inside a method), and found on the path as it is then; a failed load leaves it as it was
    directory = str(path.parent)
    added = directory not in sys.path
    if added:
        sys.path.insert(0, directory)
    registered: list[str] = []
    try:
        local = local_modules(path, path.parent)
        _check_local(path, local)
        registered = serve(local)
        sys.modules[name] = module  # before executing: dataclasses and pickling look it up
        # Read, hash and compile the same bytes. exec_module() may use a stale .pyc
        # when a same-sized edit preserves the source's timestamp.
        data = path.read_bytes()
        code = compile(data, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        helpers, verified = _read(path.parent, local, before)
        identity = _identity(str(path), data, helpers, executed=verified)
        for value in vars(module).values():  # the classes these bytes defined
            if isinstance(value, type) and value.__module__ == name:
                _identified(value, identity, path.parent, local, module)
    except BaseException as exc:
        sys.modules.pop(name, None)
        unserve(registered)
        if added:
            sys.path.remove(directory)
        if isinstance(exc, EvaluatorLoadError):
            raise
        if not isinstance(exc, Exception):
            raise
        raise EvaluatorLoadError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    _EXECUTED[name] = identity, local
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
    importlib.invalidate_caches()  # files written since are found, as by a new process
    where, separator, attribute = reference.rpartition(":")
    if not separator or not where or not attribute.isidentifier():
        raise EvaluatorLoadError(
            f"evaluator {reference!r}: expected FILE.py:Class or package.module:Class"
        )
    imported_before = set(sys.modules)
    path = Path(where).expanduser().resolve() if where.endswith(".py") or "/" in where else None
    if path is not None:
        module = _load_file(path, imported_before)
        canonical = f"{path}:{attribute}"
    else:
        canonical = reference
        module = _import(where)
    cls = getattr(module, attribute, None)
    if cls is None:
        raise EvaluatorLoadError(f"{where} has no {attribute!r}")
    if not isinstance(cls, type):
        raise EvaluatorLoadError(f"{attribute} is not a class: name the class of an evaluator")
    evaluate = getattr(cls, "evaluate", None)
    if not callable(evaluate) or getattr(evaluate, "__isabstractmethod__", False):
        raise EvaluatorLoadError(f"{attribute} is not an evaluator: it has no evaluate() method")
    if path is not None:  # what the file executed, also for a class it imports and may change
        identity, local = _EXECUTED[module.__name__]
        _identified(cls, identity, path.parent, local, module)
    elif where not in imported_before:  # else, which of its code runs is not known
        _identify_import(cls, where, imported_before)
    _REFERENCES[cls] = canonical
    return cls
