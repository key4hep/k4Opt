"""Load an evaluator class from a file or an importable module."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import sys
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


def evaluator_source(cls: type) -> dict[str, Json]:
    """Identify the source of an evaluator class without hashing a file that may have changed.

    The identity is taken when :func:`load_evaluator` imports the file that defines the class:
    for ``FILE.py:Class`` the exact bytes executed, for a module that call imported its file
    read right after the import (Python may have executed a cached ``.pyc`` instead).
    ``revision`` is the commit and dirty flag of the git checkout that tracks the file, taken at
    the same time, ``None`` if no checkout tracks it; it labels the helper modules of the same
    checkout, which the hash does not cover. For a class whose module was imported before, or
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


def _identity(path: str, data: bytes, *, executed: bool) -> FrozenMap[str, Json]:
    """``executed``: ``data`` are the bytes Python compiled, not the file read after the import."""
    digest = hashlib.sha256(data).hexdigest()
    git = tracked_revision(Path(path))
    identity = freeze(
        {
            "path": path,
            "bytes": len(data),
            "sha256": digest,
            "executed": executed,
            "revision": None if git is None else git.to_dict(),
        },
        "evaluator source",
        EvaluatorLoadError,
    )
    assert isinstance(identity, FrozenMap)
    return identity


def _identify_import(cls: type) -> None:
    """Record the source file of a class whose module was imported just now, if readable."""
    if cls in _SOURCES:  # defined in a FILE.py, identified by the bytes it executed
        return
    try:
        path = inspect.getsourcefile(cls)
    except TypeError:  # defined in a built-in module
        return
    if path is not None:
        with contextlib.suppress(OSError):
            _SOURCES[cls] = _identity(path, Path(path).read_bytes(), executed=False)


def _load_file(path: Path) -> ModuleType:
    """Import ``path`` (absolute) once under a name derived from its location, so repeats share a
    module."""
    if not path.is_file():
        raise EvaluatorLoadError(f"evaluator file {path} does not exist")
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    name = f"k4opt_evaluator_{path.stem}_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluatorLoadError(f"{path} cannot be imported as a Python module")
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
        identity = _identity(str(path), data, executed=True)
        for value in vars(module).values():  # the classes these bytes defined
            if isinstance(value, type) and value.__module__ == name:
                _SOURCES[value] = identity
    except BaseException as exc:
        del sys.modules[name]
        if added:
            sys.path.remove(directory)
        if not isinstance(exc, Exception):
            raise
        raise EvaluatorLoadError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    return module


def load_evaluator(reference: str) -> type[EvaluatorProtocol]:
    """The evaluator class that ``reference`` names, ``FILE.py:Class`` or ``module:Class``.

    ``FILE.py`` is a path (a leading ``~`` is expanded) and its directory is added to
    ``sys.path``, so the file can import modules next to it; anything else is an importable
    module name. Modules are cached by name for the process, so helpers of different evaluator
    files need unique names (or one installed package, ``package.module:Class``).

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
