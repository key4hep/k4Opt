"""``load_evaluator``: FILE.py:Class and module:Class, and every way the reference can be wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from k4opt.errors import K4OptError
from k4opt.evaluation import EvaluatorLoadError, load_evaluator

GOOD = """
from k4opt.evaluation import EvaluationResult, Measurement


class Good:
    name = "demo.good"
    version = "3"

    def evaluate(self, request):
        return EvaluationResult.valid([Measurement("analysis", "answer", 42)])


class NoEvaluate:
    name = "demo.bad"
    version = "1"


class NeedsArguments:
    name = "demo.args"
    version = "1"

    def __init__(self, required):
        self.required = required

    def evaluate(self, request):
        raise NotImplementedError
"""


@pytest.fixture
def evaluator_file(tmp_path: Path) -> Path:
    path = tmp_path / "my_evaluator.py"
    path.write_text(GOOD, encoding="utf-8")
    return path


def test_a_file_and_class_give_a_working_evaluator(evaluator_file: Path) -> None:
    evaluator = load_evaluator(f"{evaluator_file}:Good")
    assert (evaluator.name, evaluator.version) == ("demo.good", "3")
    result = evaluator.evaluate(None)  # type: ignore[arg-type]
    assert result.measurements[0].value == 42


def test_loading_a_file_twice_shares_the_module(evaluator_file: Path) -> None:
    first = load_evaluator(f"{evaluator_file}:Good")
    second = load_evaluator(f"{evaluator_file}:Good")
    assert type(first) is type(second)
    assert type(first).__module__ in sys.modules


def test_an_importable_module_name_works(
    tmp_path: Path, evaluator_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert load_evaluator("my_evaluator:Good").name == "demo.good"
    finally:
        sys.modules.pop("my_evaluator", None)


def test_a_home_relative_path_is_expanded(
    evaluator_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(evaluator_file.parent))
    assert load_evaluator("~/my_evaluator.py:Good").version == "3"


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("nocolon", "expected FILE.py:Class"),
        (":Good", "expected FILE.py:Class"),
        ("file.py:", "expected FILE.py:Class"),
        ("file.py:not an identifier", "expected FILE.py:Class"),
        ("{file}:Missing", "has no 'Missing'"),
        ("{file}:NoEvaluate", r"not an evaluator.*evaluate\(\)"),
        ("{file}:NeedsArguments", r"creating NeedsArguments\(\) failed"),
        ("{dir}/gone.py:Good", "does not exist"),
        ("no_such_module_anywhere:Good", "importing 'no_such_module_anywhere' failed"),
    ],
)
def test_bad_references_are_load_errors(
    tmp_path: Path, evaluator_file: Path, reference: str, message: str
) -> None:
    reference = reference.format(file=evaluator_file, dir=tmp_path)
    with pytest.raises(EvaluatorLoadError, match=message) as info:
        load_evaluator(reference)
    assert isinstance(info.value, K4OptError)


def test_a_file_that_fails_to_import_is_a_load_error_and_can_be_fixed(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(EvaluatorLoadError, match="RuntimeError: boom"):
        load_evaluator(f"{broken}:Good")
    broken.write_text(GOOD, encoding="utf-8")  # a failed import is not remembered
    assert load_evaluator(f"{broken}:Good").name == "demo.good"
