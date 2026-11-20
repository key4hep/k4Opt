"""Classes of the user's own code, whatever their kind: one file that defines an evaluator and a
strategy is one module, imported and identified once; a class is checked for the methods its kind
needs. Loading in all its detail is tested through ``load_evaluator``
(``tests/unit/evaluation/test_loader.py``)."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest

from k4opt.errors import CodeLoadError
from k4opt.evaluation import load_evaluator
from k4opt.usercode import load_class, source_of

BOTH = """
    from k4opt.evaluation import Evaluator

    class Measure(Evaluator):
        name, version = "demo.measure", "1"

        def evaluate(self, request):
            return {("analysis", "x"): 1.0}

    class Search:
        def propose(self, trial): ...

        def proposed(self, trial, proposal): ...

        def tell(self, trial, feedback): ...
"""


@pytest.fixture
def both(tmp_path: Path) -> Path:
    path = tmp_path / "study_code.py"
    path.write_text(textwrap.dedent(BOTH), encoding="utf-8")
    return path


def test_a_file_of_an_evaluator_and_a_strategy_is_one_module_identified_once(both: Path) -> None:
    evaluator = load_evaluator(f"{both}:Measure")
    strategy = load_class(f"{both}:Search", "strategy", ("propose", "proposed", "tell"))
    assert evaluator.__module__ == strategy.__module__
    digest = hashlib.sha256(both.read_bytes()).hexdigest()
    assert source_of(evaluator)["sha256"] == source_of(strategy)["sha256"] == digest
    assert source_of(strategy)["reference"] == f"{both}:Search"
    assert source_of(strategy)["executed"] is True


def test_a_class_without_the_methods_of_its_kind_is_refused_naming_them(both: Path) -> None:
    with pytest.raises(CodeLoadError, match=r"Measure is not a strategy: it has no propose\(\)"):
        load_class(f"{both}:Measure", "strategy", ("propose", "proposed", "tell"))
    with pytest.raises(CodeLoadError, match=r"strategy 'Search': expected FILE\.py:Class"):
        load_class("Search", "strategy")
    with pytest.raises(CodeLoadError, match=r"evaluator .*missing\.py does not exist"):
        load_class(f"{both.parent / 'missing.py'}:Search", "evaluator")
