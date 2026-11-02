"""``load_evaluator``: FILE.py:Class and module:Class, and every way the reference can be wrong."""

from __future__ import annotations

import hashlib
import importlib
import os
import py_compile
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from k4opt.errors import K4OptError
from k4opt.evaluation import EvaluatorLoadError, _imports, evaluator_source, load_evaluator
from k4opt.evaluation.loader import changed_helpers
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


def make_good():
    return Good()
"""


@pytest.fixture(autouse=True)
def new_process() -> Iterator[None]:
    """Each test loads as a new process would: the modules one served from their source are not
    the next one's, whose files of the same names are elsewhere."""
    served = dict(_imports._SERVED)
    yield
    _imports._SERVED.clear()
    _imports._SERVED.update(served)


@pytest.fixture
def evaluator_file(tmp_path: Path) -> Path:
    path = tmp_path / "my_evaluator.py"
    path.write_text(GOOD, encoding="utf-8")
    return path


def test_a_file_and_class_give_the_evaluator_class(evaluator_file: Path) -> None:
    cls = load_evaluator(f"{evaluator_file}:Good")
    assert isinstance(cls, type)
    evaluator = cls()
    assert (evaluator.name, evaluator.version) == ("demo.good", "3")
    result = evaluator.evaluate(None)  # type: ignore[arg-type]
    assert result.measurements[0].value == 42


def test_a_class_is_loaded_without_being_created(evaluator_file: Path) -> None:
    # whoever evaluates with it creates the instances, and reports a constructor that fails
    cls = load_evaluator(f"{evaluator_file}:NeedsArguments")
    assert cls.__name__ == "NeedsArguments"


def test_loading_a_file_twice_shares_the_module(evaluator_file: Path) -> None:
    first = load_evaluator(f"{evaluator_file}:Good")
    second = load_evaluator(f"{evaluator_file}:Good")
    assert first is second
    assert first.__module__ in sys.modules


def test_a_loaded_class_keeps_the_identity_of_the_executed_bytes(evaluator_file: Path) -> None:
    data = evaluator_file.read_bytes()
    first = load_evaluator(f"{evaluator_file}:Good")
    assert (
        evaluator_source(first)
        == evaluator_source(load_evaluator(f"{evaluator_file}:Good"))
        == {
            "path": str(evaluator_file),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "helpers": {},  # it imports no module beside it
            "executed": True,
            "revision": None,  # no git checkout tracks it
            "reference": f"{evaluator_file}:Good",  # what loads it again
        }
    )
    evaluator_file.unlink()
    assert evaluator_source(first)["sha256"] == hashlib.sha256(data).hexdigest()


def test_a_file_edited_since_it_was_executed_is_not_loaded_again(evaluator_file: Path) -> None:
    load_evaluator(f"{evaluator_file}:Good")
    evaluator_file.write_text(GOOD.replace('"answer", 42', '"answer", 99'))
    with pytest.raises(EvaluatorLoadError, match=r"its bytes changed.*load it in a new process"):
        load_evaluator(f"{evaluator_file}:Good")


IMPORTS = """
import {helper}
from {package} import tools

from k4opt.evaluation import EvaluationResult, Measurement


class Good:
    name = "demo.good"
    version = "3"

    def evaluate(self, request):
        import {late}  # run only when it evaluates, hashed all the same

        value = {helper}.value + tools.value + {late}.value
        return EvaluationResult.valid([Measurement("analysis", "answer", value)])
"""


@pytest.fixture
def with_helpers(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
    """An evaluator file importing a module, a package whose module imports relatively, and a
    module inside a method, all beside it; named for the test, and forgotten after it."""
    tag = hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    helper, package, late = f"helper_{tag}", f"pkg_{tag}", f"late_{tag}"
    (tmp_path / f"{helper}.py").write_text("value = 1\n")
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / package / "tools.py").write_text("from .constants import value\n")
    (tmp_path / package / "constants.py").write_text("value = 10\n")
    (tmp_path / f"{late}.py").write_text("value = 100\n")
    path = tmp_path / "with_helpers.py"
    path.write_text(IMPORTS.format(helper=helper, package=package, late=late))
    yield path
    for name in list(sys.modules):
        if name.split(".")[0] in (helper, package, late):
            del sys.modules[name]


def test_the_modules_a_file_imports_beside_it_are_hashed(with_helpers: Path) -> None:
    cls = load_evaluator(f"{with_helpers}:Good")
    assert cls().evaluate(None).measurements[0].value == 111  # type: ignore[arg-type]
    root = with_helpers.parent
    expected = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if path != with_helpers
    }
    assert len(expected) == 5  # the helper, the package's three files, the late import
    assert evaluator_source(cls)["helpers"] == dict(sorted(expected.items()))


def test_an_edited_helper_is_not_loaded_again(with_helpers: Path) -> None:
    load_evaluator(f"{with_helpers}:Good")
    constants = next(with_helpers.parent.glob("pkg_*/constants.py"))
    constants.write_text("value = 20\n")
    with pytest.raises(EvaluatorLoadError, match=r"the modules it imports changed: pkg_\w+/consta"):
        load_evaluator(f"{with_helpers}:Good")


def test_a_helper_of_the_same_name_from_another_directory_is_refused(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    name = "helper_" + hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    files = []
    for directory, value in (("a", 1), ("b", 2)):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / f"{name}.py").write_text(f"value = {value}\n")
        files.append(tmp_path / directory / "evaluator.py")
        files[-1].write_text(f"import {name}\n" + GOOD)
    try:
        load_evaluator(f"{files[0]}:Good")
        # Python would give the second file the first one's module
        with pytest.raises(
            EvaluatorLoadError, match=rf"imports '{name}', which is .*/b/{name}\.py, but"
        ):
            load_evaluator(f"{files[1]}:Good")
        # evaluators in one directory share their helpers
        (tmp_path / "a" / "other.py").write_text(f"import {name}\n" + GOOD)
        assert load_evaluator(f"{tmp_path / 'a' / 'other.py'}:Good").name == "demo.good"
    finally:
        sys.modules.pop(name, None)


def test_a_module_imported_by_the_loader_is_identified_by_the_bytes_it_executed(
    evaluator_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = evaluator_file.read_bytes()
    monkeypatch.syspath_prepend(str(evaluator_file.parent))
    try:
        cls = load_evaluator("my_evaluator:Good")
        evaluator_file.write_text("# edited after the import\n")
        assert evaluator_source(cls) == {
            "path": str(evaluator_file),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "helpers": {},
            "executed": True,  # imported from its source, not from a cached .pyc
            "revision": None,
            "reference": "my_evaluator:Good",
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
        assert second().evaluate(None).measurements[0].value == 12345  # type: ignore[arg-type]
        assert evaluator_source(second) == {
            "path": str(evaluator_file),
            "reference": "my_evaluator:Good",
        }
        # the class loaded before the reload still runs, and names, the original code
        assert first().evaluate(None).measurements[0].value == 42  # type: ignore[arg-type]
        assert evaluator_source(first)["sha256"] == hashlib.sha256(GOOD.encode()).hexdigest()
    finally:
        sys.modules.pop("my_evaluator", None)


def test_a_class_re_exported_by_a_package_is_identified_by_its_own_file_and_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "study"
    package.mkdir()
    # Python runs it to reach the class, and it may change the class
    init = "from study.impl import Good\nGood.version = '4'\n"
    (package / "__init__.py").write_text(init)
    (package / "impl.py").write_text(GOOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        source = evaluator_source(load_evaluator("study:Good"))
        assert source["path"] == str(package / "impl.py")
        assert source["sha256"] == hashlib.sha256(GOOD.encode()).hexdigest()
        assert source["helpers"] == {"study/__init__.py": hashlib.sha256(init.encode()).hexdigest()}
        assert source["reference"] == "study:Good"  # as it was loaded
    finally:
        for name in ("study", "study.impl"):
            sys.modules.pop(name, None)


def test_a_module_is_identified_with_the_modules_of_its_package_it_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "campaign"
    (package / "physics").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "physics" / "__init__.py").write_text("")
    (package / "physics" / "corrections.py").write_text("scale = 1.0\n")
    (package / "impl.py").write_text("import json\nfrom .physics import corrections\n" + GOOD)
    (tmp_path / "json.py").write_text("")  # beside the package, not of it: not its module
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        source = evaluator_source(load_evaluator("campaign.impl:Good"))
        assert sorted(source["helpers"]) == [  # type: ignore[arg-type]
            "campaign/__init__.py",
            "campaign/physics/__init__.py",
            "campaign/physics/corrections.py",
        ]
    finally:
        for name in [n for n in sys.modules if n.partition(".")[0] == "campaign"]:
            del sys.modules[name]


def test_a_top_level_module_is_identified_with_the_modules_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    tag = hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    module, helper = f"top_{tag}", f"corrections_{tag}"
    (tmp_path / f"{helper}.py").write_text("scale = 1.0\n")
    (tmp_path / f"{module}.py").write_text(f"import {helper}\n" + GOOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        source = evaluator_source(load_evaluator(f"{module}:Good"))
        assert source["helpers"] == {f"{helper}.py": hashlib.sha256(b"scale = 1.0\n").hexdigest()}
    finally:
        for name in (module, helper):
            sys.modules.pop(name, None)


def test_an_installed_top_level_module_does_not_count_other_distributions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    tag = hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    module, other = f"installed_{tag}", f"distribution_{tag}"
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / f"{other}.py").write_text("")
    (site / f"{module}.py").write_text(f"import {other}\n" + GOOD)
    monkeypatch.syspath_prepend(str(site))
    try:
        assert evaluator_source(load_evaluator(f"{module}:Good"))["helpers"] == {}
    finally:
        for name in (module, other):
            sys.modules.pop(name, None)


def test_a_helper_edited_after_python_imported_it_is_refused_for_another_evaluator(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # a second evaluator would run the module Python imported for the first, not the file
    helper = "shared_" + hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    request.addfinalizer(lambda: sys.modules.pop(helper, None))
    (tmp_path / f"{helper}.py").write_text("value = 1\n")
    for name in ("first", "second"):
        (tmp_path / f"{name}.py").write_text(f"import {helper}\n" + GOOD)
    load_evaluator(f"{tmp_path / 'first.py'}:Good")
    (tmp_path / f"{helper}.py").write_text("value = 2\n")
    with pytest.raises(
        EvaluatorLoadError, match=rf"{helper}\.py changed since this process imported it"
    ):
        load_evaluator(f"{tmp_path / 'second.py'}:Good")
    (tmp_path / f"{helper}.py").write_text("value = 1\n")  # what runs: then it is that code
    second = load_evaluator(f"{tmp_path / 'second.py'}:Good")
    assert evaluator_source(second)["helpers"] == {
        f"{helper}.py": hashlib.sha256(b"value = 1\n").hexdigest()
    }


def test_a_helper_imported_before_the_loader_read_it_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    helper = "early_" + hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    request.addfinalizer(lambda: sys.modules.pop(helper, None))
    (tmp_path / f"{helper}.py").write_text("value = 1\n")
    (tmp_path / "late.py").write_text(f"import {helper}\n" + GOOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.import_module(helper)  # which bytes ran is not known to k4opt
    with pytest.raises(EvaluatorLoadError, match="was imported before load_evaluator read it"):
        load_evaluator(f"{tmp_path / 'late.py'}:Good")


def test_changed_helpers_names_the_modules_edited_since_the_class_was_loaded(
    with_helpers: Path,
) -> None:
    cls = load_evaluator(f"{with_helpers}:Good")
    assert changed_helpers(cls) == []
    late = next(with_helpers.parent.glob("late_*.py"))  # imported only when it evaluates
    late.write_text("value = 200\n")
    assert changed_helpers(cls) == [late.name]


def test_a_module_imported_before_does_not_claim_an_unverified_source_hash(
    evaluator_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(evaluator_file.parent))
    try:
        importlib.import_module("my_evaluator")
        cls = load_evaluator("my_evaluator:Good")
        assert evaluator_source(cls) == {
            "path": str(evaluator_file),
            "reference": "my_evaluator:Good",
        }
    finally:
        sys.modules.pop("my_evaluator", None)


def test_a_stale_bytecode_cache_cannot_disagree_with_the_source_hash(evaluator_file: Path) -> None:
    py_compile.compile(str(evaluator_file), doraise=True)
    original = evaluator_file.stat()
    data = GOOD.replace('"answer", 42', '"answer", 99').encode()
    evaluator_file.write_bytes(data)
    os.utime(evaluator_file, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert evaluator_file.stat().st_size == original.st_size
    cls = load_evaluator(f"{evaluator_file}:Good")
    assert cls().evaluate(None).measurements[0].value == 99  # type: ignore[arg-type]
    assert evaluator_source(cls)["sha256"] == hashlib.sha256(data).hexdigest()


def test_a_stale_bytecode_cache_of_a_helper_cannot_disagree_with_its_hash(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    helper = "cached_" + hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    request.addfinalizer(lambda: sys.modules.pop(helper, None))
    path = tmp_path / f"{helper}.py"
    path.write_text("value = 11\n")
    py_compile.compile(str(path), doraise=True)
    original = path.stat()
    path.write_text("value = 22\n")  # Python's own import would run the .pyc of value = 11
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert path.stat().st_size == original.st_size
    evaluator = tmp_path / "uses_cached.py"
    evaluator.write_text(
        f"import {helper}\n" + GOOD.replace('"answer", 42', f'"answer", {helper}.value')
    )
    cls = load_evaluator(f"{evaluator}:Good")
    assert cls().evaluate(None).measurements[0].value == 22  # type: ignore[arg-type]
    assert evaluator_source(cls)["helpers"] == {
        f"{helper}.py": hashlib.sha256(b"value = 22\n").hexdigest()
    }


def test_a_helper_another_file_imports_only_when_it_evaluates_is_not_shared(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    helper = "lazy_" + hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    request.addfinalizer(lambda: sys.modules.pop(helper, None))
    for directory in ("a", "b"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / f"{helper}.py").write_text(f"where = {directory!r}\n")
    lazy = GOOD.replace(
        "def evaluate(self, request):", f"def evaluate(self, request):\n        import {helper}"
    )
    (tmp_path / "a" / "evaluator.py").write_text(lazy)
    (tmp_path / "b" / "evaluator.py").write_text(f"import {helper}\n" + GOOD)
    load_evaluator(f"{tmp_path / 'a' / 'evaluator.py'}:Good")  # imports nothing yet
    with pytest.raises(EvaluatorLoadError, match=rf"imports '{helper}', which is .*/b/{helper}"):
        load_evaluator(f"{tmp_path / 'b' / 'evaluator.py'}:Good")


def test_a_stale_bytecode_cache_of_a_module_cannot_disagree_with_its_hash(
    evaluator_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    py_compile.compile(str(evaluator_file), doraise=True)
    original = evaluator_file.stat()
    data = GOOD.replace('"answer", 42', '"answer", 99').encode()
    evaluator_file.write_bytes(data)  # Python's own import would run the .pyc of 42
    os.utime(evaluator_file, ns=(original.st_atime_ns, original.st_mtime_ns))
    monkeypatch.syspath_prepend(str(evaluator_file.parent))
    try:
        cls = load_evaluator("my_evaluator:Good")
        assert cls().evaluate(None).measurements[0].value == 99  # type: ignore[arg-type]
        source = evaluator_source(cls)
        assert (source["sha256"], source["executed"]) == (hashlib.sha256(data).hexdigest(), True)
    finally:
        sys.modules.pop("my_evaluator", None)


def test_a_helper_package_reads_its_resources_as_from_any_import(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    package = "data_" + hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    request.addfinalizer(lambda: sys.modules.pop(package, None))
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / package / "calibration.json").write_text("7")
    evaluator = tmp_path / "calibrated.py"
    evaluator.write_text(
        f"import pkgutil\nfrom importlib.resources import files\n\nimport {package}\n\n"
        f"FILES = files({package!r}).joinpath('calibration.json').read_text()\n"
        f"DATA = pkgutil.get_data({package!r}, 'calibration.json')\n" + GOOD
    )
    module = sys.modules[load_evaluator(f"{evaluator}:Good").__module__]
    assert (module.FILES, module.DATA) == ("7", b"7")
    assert _imports.executed(sys.modules[package]) is not None  # imported from its source


def test_a_file_that_re_exports_a_class_identifies_it_by_what_it_executed(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    helper = "defines_" + hashlib.sha256(request.node.name.encode()).hexdigest()[:8]
    request.addfinalizer(lambda: sys.modules.pop(helper, None))
    (tmp_path / f"{helper}.py").write_text(GOOD)
    wrapper = tmp_path / "wrapper.py"
    text = f"from {helper} import Good\n\nGood.scale = 2  # changes what it measures\n"
    wrapper.write_text(text)
    source = evaluator_source(load_evaluator(f"{wrapper}:Good"))
    assert (source["path"], source["sha256"]) == (
        str(wrapper),
        hashlib.sha256(text.encode()).hexdigest(),
    )
    assert source["helpers"] == {f"{helper}.py": hashlib.sha256(GOOD.encode()).hexdigest()}


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
    cls = load_evaluator("~/my_evaluator.py:Good")
    assert cls.version == "3"
    assert evaluator_source(cls)["reference"] == f"{evaluator_file}:Good"  # loadable anywhere


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("nocolon", "expected FILE.py:Class"),
        (":Good", "expected FILE.py:Class"),
        ("file.py:", "expected FILE.py:Class"),
        ("file.py:not an identifier", "expected FILE.py:Class"),
        ("{file}:Missing", "has no 'Missing'"),
        ("{file}:NoEvaluate", r"not an evaluator.*evaluate\(\)"),
        ("{file}:make_good", "make_good is not a class"),
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
        evaluator = load_evaluator(f"{study / 'evaluator.py'}:Study")()  # a top-level import
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
