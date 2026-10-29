"""Load an evaluator class from a file or an importable module."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from k4opt.evaluation.errors import EvaluatorLoadError
from k4opt.evaluation.protocol import Evaluator


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
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[name]
        if added:
            sys.path.remove(directory)
        raise EvaluatorLoadError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    return module


def load_evaluator(reference: str) -> Evaluator:
    """Create the evaluator that ``reference`` names, ``FILE.py:Class`` or ``module:Class``.

    ``FILE.py`` is a path (a leading ``~`` is expanded) and its directory is added to
    ``sys.path``, so the file can import modules next to it; anything else is an importable
    module name. The class is created without arguments and must provide ``name``, ``version`` and
    ``evaluate``. The file is executed as ordinary code: only name evaluators you trust. Raises
    :class:`EvaluatorLoadError`.
    """
    where, separator, attribute = reference.rpartition(":")
    if not separator or not where or not attribute.isidentifier():
        raise EvaluatorLoadError(
            f"evaluator {reference!r}: expected FILE.py:Class or package.module:Class"
        )
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
    result: Evaluator = evaluator
    return result
