"""The components of k4opt depend on each other one way only, and never on another's private
modules: what defines a problem knows nothing of its records, nor do records of what executes
candidates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "k4opt"
# each component may use only those before it
ORDER = ("geometry", "ddsim", "evaluation", "problem", "record", "execution", "optimization")


def imports(path: Path) -> list[str]:
    """Every module ``path`` imports from k4opt, by its dotted name."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("k4opt"):
            found.append(node.module)
            found += [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.startswith("k4opt")]
    return found


@pytest.mark.parametrize("component", ORDER)
def test_a_component_uses_only_those_before_it_and_their_non_private_modules(
    component: str,
) -> None:
    later, wrong = ORDER[ORDER.index(component) + 1 :], []
    for path in sorted((PACKAGE / component).rglob("*.py")):
        for name in imports(path):
            parts = name.split(".")
            if len(parts) < 2 or parts[1] == component or parts[1] not in ORDER:
                continue
            if parts[1] in later:
                wrong.append(f"{path.relative_to(PACKAGE)}: {name} (a later component)")
            elif len(parts) > 2 and parts[2].startswith("_"):
                wrong.append(f"{path.relative_to(PACKAGE)}: {name} (a private module)")
    assert wrong == []
