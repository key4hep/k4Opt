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
from k4opt.frozen import FrozenMap, Json

_SOURCES: WeakKeyDictionary[ModuleType, FrozenMap[str, Json]] = WeakKeyDictionary()


def evaluator_source(evaluator: EvaluatorProtocol) -> dict[str, Json]:
    """Identify the source of an evaluator without hashing a file that may have changed.

    The identity is taken when :func:`load_evaluator` imports the file that defines the class
    and is shared by cached instances: for ``FILE.py:Class`` the exact bytes executed, for a
    module that call imported its file read right after the import (Python may have executed
    a cached ``.pyc`` instead). For a class whose module was imported before, only the path is
    known: the current file cannot establish which bytes Python previously executed.
    """
    module = sys.modules.get(type(evaluator).__module__)
    if module is not None and module in _SOURCES:
        return dict(_SOURCES[module])
    try:
        path = inspect.getsourcefile(type(evaluator))
    except TypeError:
        path = None
    return {"path": path}


def _identity(path: str, data: bytes, *, executed: bool) -> FrozenMap[str, Json]:
    """``executed``: ``data`` are the bytes Python compiled, not the file read after the import."""
    digest = hashlib.sha256(data).hexdigest()
    return FrozenMap({"path": path, "bytes": len(data), "sha256": digest, "executed": executed})


def _identify_import(module: ModuleType) -> None:
    """Record the source file of a module imported just now, if it has one that can be read."""
    if module in _SOURCES:  # a FILE.py, identified by the bytes it executed
        return
    try:
        path = inspect.getsourcefile(module)
    except TypeError:  # a built-in module
        return
    if path is not None:
        with contextlib.suppress(OSError):
            _SOURCES[module] = _identity(path, Path(path).read_bytes(), executed=False)


def _load_file(path: Path) -> ModuleType:
    """Import ``path`` once under a name derived from its location, so repeats share a module."""
    path = path.expanduser().resolve()
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
        _SOURCES[module] = _identity(str(path), data, executed=True)
    except BaseException as exc:
        del sys.modules[name]
        if added:
            sys.path.remove(directory)
        if not isinstance(exc, Exception):
            raise
        raise EvaluatorLoadError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    return module


def load_evaluator(reference: str) -> EvaluatorProtocol:
    """Create the evaluator that ``reference`` names, ``FILE.py:Class`` or ``module:Class``.

    ``FILE.py`` is a path (a leading ``~`` is expanded) and its directory is added to
    ``sys.path``, so the file can import modules next to it; anything else is an importable
    module name. Modules are cached by name for the process, so helpers of different evaluator
    files need unique names (or one installed package, ``package.module:Class``).

    The class is created without arguments and must provide ``name``, ``version`` and
    ``evaluate``. The file is executed as ordinary code: only name evaluators you trust. Raises
    :class:`EvaluatorLoadError`.
    """
    where, separator, attribute = reference.rpartition(":")
    if not separator or not where or not attribute.isidentifier():
        raise EvaluatorLoadError(
            f"evaluator {reference!r}: expected FILE.py:Class or package.module:Class"
        )
    imported_before = set(sys.modules)
    if where.endswith(".py") or "/" in where:
        module = _load_file(Path(where))
    else:
        try:
            module = importlib.import_module(where)
        except Exception as exc:
            raise EvaluatorLoadError(
                f"importing {where!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
    cls = getattr(module, attribute, None)
    if cls is None:
        raise EvaluatorLoadError(f"{where} has no {attribute!r}")
    try:
        evaluator = cls()
    except Exception as exc:
        raise EvaluatorLoadError(
            f"creating {attribute}() failed: {type(exc).__name__}: {exc}"
        ) from exc
    missing = [
        member for member in ("name", "version", "evaluate") if not hasattr(evaluator, member)
    ]
    if missing or not callable(evaluator.evaluate):
        raise EvaluatorLoadError(
            f"{attribute} is not an evaluator: needs name, version and an evaluate() method"
            + (f" (missing {', '.join(missing)})" if missing else "")
        )
    defining = sys.modules.get(type(evaluator).__module__)
    if defining is not None and defining.__name__ not in imported_before:
        _identify_import(defining)
    result: EvaluatorProtocol = evaluator
    return result
