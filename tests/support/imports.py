"""An import hook for the tests of how k4opt imports evaluator code."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path


class FromFile(importlib.abc.MetaPathFinder):
    """Imports the module ``name`` from ``file`` as Python's own loader does, which may run a
    cached ``.pyc``: put before Python's path finder, k4opt cannot tell which bytes ran."""

    def __init__(self, name: str, file: Path) -> None:
        self.name, self.file = name, file

    def find_spec(
        self, fullname: str, path: object, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.name:
            return None
        return importlib.util.spec_from_file_location(fullname, self.file)
