"""``load_evaluator``: FILE.py:Class and module:Class, and every way the reference can be wrong."""

from __future__ import annotations

import hashlib
import importlib
import os
import py_compile
import sys
from pathlib import Path

import pytest

from k4opt.errors import K4OptError
from k4opt.evaluation import EvaluatorLoadError, evaluator_source, load_evaluator
from tests.support.common import git_checkout, requires_git

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


def test_cached_evaluator_keeps_the_identity_of_the_executed_bytes(evaluator_file: Path) -> None:
    data = evaluator_file.read_bytes()
    first = load_evaluator(f"{evaluator_file}:Good")
    evaluator_file.write_text(GOOD.replace('"answer", 42', '"answer", 99'))
    second = load_evaluator(f"{evaluator_file}:Good")
    assert second.evaluate(None).measurements[0].value == 42  # type: ignore[arg-type]
    assert (
        evaluator_source(first)
        == evaluator_source(second)
        == {
            "path": str(evaluator_file),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "executed": True,
            "revision": None,  # no git checkout tracks it
        }
    )
    evaluator_file.unlink()
    assert evaluator_source(first)["sha256"] == hashlib.sha256(data).hexdigest()


def test_a_module_imported_by_the_loader_is_identified_right_after_the_import(
    evaluator_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = evaluator_file.read_bytes()
    monkeypatch.syspath_prepend(str(evaluator_file.parent))
    try:
        evaluator = load_evaluator("my_evaluator:Good")
        evaluator_file.write_text("# edited after the import\n")
        # not the bytes Python compiled, which may have come from a cached .pyc
        assert evaluator_source(evaluator) == {
            "path": str(evaluator_file),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "executed": False,
            "revision": None,
        }
    finally:
        sys.modules.pop("my_evaluator", None)


def test_a_reloaded_module_does_not_keep_the_identity_of_the_code_it_replaced(
    evaluator_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(evaluator_file.parent))
    try:
        first = load_evaluator("my_evaluator:Good")
        evaluator_file.write_text(GOOD.replace('"answer", 42', '"answer", 12345'))
        importlib.reload(sys.modules["my_evaluator"])
        second = load_evaluator("my_evaluator:Good")
        assert second.evaluate(None).measurements[0].value == 12345  # type: ignore[arg-type]
        assert evaluator_source(second) == {"path": str(evaluator_file)}
        # the instance created before the reload still runs, and names, the original code
        assert evaluator_source(first)["sha256"] == hashlib.sha256(GOOD.encode()).hexdigest()
    finally:
        sys.modules.pop("my_evaluator", None)


def test_a_class_re_exported_by_a_package_is_identified_by_its_own_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "study"
    package.mkdir()
    (package / "__init__.py").write_text("from study.impl import Good\n")
    (package / "impl.py").write_text(GOOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        source = evaluator_source(load_evaluator("study:Good"))
        assert source["path"] == str(package / "impl.py")
        assert source["sha256"] == hashlib.sha256(GOOD.encode()).hexdigest()
    finally:
        for name in ("study", "study.impl"):
            sys.modules.pop(name, None)


def test_a_module_imported_before_does_not_claim_an_unverified_source_hash(
    evaluator_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(evaluator_file.parent))
    try:
        importlib.import_module("my_evaluator")
        evaluator = load_evaluator("my_evaluator:Good")
        assert evaluator_source(evaluator) == {"path": str(evaluator_file)}
    finally:
        sys.modules.pop("my_evaluator", None)


def test_a_stale_bytecode_cache_cannot_disagree_with_the_source_hash(evaluator_file: Path) -> None:
    py_compile.compile(str(evaluator_file), doraise=True)
    original = evaluator_file.stat()
    data = GOOD.replace('"answer", 42', '"answer", 99').encode()
    evaluator_file.write_bytes(data)
    os.utime(evaluator_file, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert evaluator_file.stat().st_size == original.st_size
    evaluator = load_evaluator(f"{evaluator_file}:Good")
    assert evaluator.evaluate(None).measurements[0].value == 99  # type: ignore[arg-type]
    assert evaluator_source(evaluator)["sha256"] == hashlib.sha256(data).hexdigest()


@requires_git
def test_the_revision_of_the_checkout_is_taken_when_the_file_is_loaded(tmp_path: Path) -> None:
    files = {"tracked_good.py": GOOD, "later_good.py": GOOD, "helper.py": ""}
    commit = git_checkout(tmp_path / "study", files)
    clean = load_evaluator(f"{tmp_path / 'study' / 'tracked_good.py'}:Good")
    assert evaluator_source(clean)["revision"] == {"commit": commit, "dirty": False}
    # a helper module of the checkout changes, which the evaluator's own hash cannot see
    (tmp_path / "study" / "helper.py").write_text("changed = True\n")
    later = load_evaluator(f"{tmp_path / 'study' / 'later_good.py'}:Good")
    assert evaluator_source(later)["revision"] == {"commit": commit, "dirty": True}
    assert evaluator_source(clean)["revision"] == {"commit": commit, "dirty": False}  # as loaded
    untracked = tmp_path / "study" / "untracked_good.py"
    untracked.write_text(GOOD)  # in the checkout's directory, but in none of its commits
    assert evaluator_source(load_evaluator(f"{untracked}:Good"))["revision"] is None


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


SIBLING_EVALUATOR = """
from {helper} import resolution
from k4opt.evaluation import EvaluationResult, Measurement


class Study:
    name = "demo.study"
    version = "1"

    def evaluate(self, request):
        from {lazy} import offset  # imported only when the evaluator runs
        return EvaluationResult.valid([Measurement("analysis", "res", resolution() + offset())])
"""


def test_a_file_can_import_modules_next_to_it(tmp_path: Path) -> None:
    study = tmp_path / "my_study"
    study.mkdir()
    (study / "study_metrics_a.py").write_text("def resolution():\n    return 3.5\n")
    (study / "study_lazy_a.py").write_text("def offset():\n    return 0.5\n")
    (study / "evaluator.py").write_text(
        SIBLING_EVALUATOR.format(helper="study_metrics_a", lazy="study_lazy_a")
    )
    before = list(sys.path)
    try:
        evaluator = load_evaluator(f"{study / 'evaluator.py'}:Study")  # a top-level sibling import
        result = evaluator.evaluate(None)  # type: ignore[arg-type]  # and one inside the method
        assert result.measurements[0].value == 4.0
        assert sys.path[0] == str(study)
    finally:
        sys.path[:] = before
        for name in ("study_metrics_a", "study_lazy_a"):
            sys.modules.pop(name, None)


def test_a_failed_import_leaves_the_search_path_alone(tmp_path: Path) -> None:
    (tmp_path / "evaluator.py").write_text("import study_missing_sibling\n")
    before = list(sys.path)
    with pytest.raises(EvaluatorLoadError, match="study_missing_sibling"):
        load_evaluator(f"{tmp_path / 'evaluator.py'}:Study")
    assert sys.path == before
