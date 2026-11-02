"""A new study kept in a database (``--storage URL``): the database is reached only once the
study's directory is known to be usable, and before anything is made in it, so neither a
directory in use nor a wrong URL leaves anything behind; a SQLite database of a new study lies
outside its directory. The cases need SQLAlchemy and alembic."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("optuna")
if not os.environ.get("CI"):  # CI installs them (requirements.txt): these tests run there
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("alembic")

from k4opt.optimization import OptimizationError, load_study, optimize
from k4opt.optimization._optuna import storage
from tests.support.evaluators import trade_off


def test_a_directory_in_use_is_refused_before_the_database_is_reached(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "notes.txt").write_text("mine")
    url = f"sqlite:///{tmp_path / 'studies.db'}"
    with pytest.raises(OptimizationError, match="is in use; resume the study in it"):
        optimize(write_problem(), [trade_off()], tmp_path / "used", trials=1, storage=url)
    assert not (tmp_path / "studies.db").exists()
    assert [p.name for p in (tmp_path / "used").iterdir()] == ["notes.txt"]


def test_a_dangling_symlink_is_refused_before_the_database_is_reached(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    (tmp_path / "s").symlink_to(tmp_path / "gone")
    url = f"sqlite:///{tmp_path / 'studies.db'}"
    with pytest.raises(OptimizationError, match="is in use; resume the study in it"):
        optimize(write_problem(), [trade_off()], tmp_path / "s", trials=1, storage=url)
    assert not (tmp_path / "studies.db").exists()
    assert (tmp_path / "s").is_symlink() and not (tmp_path / "gone").exists()


def test_a_symlink_to_an_empty_directory_holds_a_new_study(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "s").symlink_to(tmp_path / "real")
    url = f"sqlite:///{tmp_path / 'studies.db'}"
    report = optimize(write_problem(), [trade_off()], tmp_path / "s", trials=1, seed=1, storage=url)
    assert [t.status for t in report.trials] == ["valid"]
    assert (tmp_path / "real" / "study.json").is_file() and (tmp_path / "studies.db").is_file()


def url_of(path: Path, form: str) -> str:
    """The URL of the SQLite database ``path``: as a path, or as a URI filename (SQLite's
    ``file:`` form, which SQLAlchemy passes on with ``uri=true``)."""
    return {
        "path": f"sqlite:///{path}",
        "uri": f"sqlite:///file:{path}?uri=true",
        "uri with authority": f"sqlite:///file://{path}?uri=true",
        "path, with a query it ignores": f"sqlite:///{path}?mode=memory",
    }[form]


@pytest.mark.parametrize(
    "form", ["path", "uri", "uri with authority", "path, with a query it ignores"]
)
@pytest.mark.parametrize(
    ("made", "database"),
    [
        (True, "s/studies.db"),  # an empty directory: new studies start there
        (False, "s/studies.db"),
        (False, "s/deeper/studies.db"),
        (False, "s"),  # the directory itself
        (True, "s"),
    ],
)
def test_a_new_studys_sqlite_database_inside_its_directory_is_refused_and_nothing_is_made(
    tmp_path: Path, write_problem: Callable[..., Path], made: bool, database: str, form: str
) -> None:
    problem = write_problem()
    if made:
        (tmp_path / "s").mkdir()
    url = url_of(tmp_path / database, form)
    with pytest.raises(OptimizationError, match=r"would be in the study directory .*/s; keep it"):
        optimize(problem, [trade_off()], tmp_path / "s", trials=1, storage=url)
    if made:
        assert (tmp_path / "s").is_dir() and not any((tmp_path / "s").iterdir())
    else:
        assert not (tmp_path / "s").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["problem.yaml", *(["s"] if made else [])]


def test_a_database_inside_the_directory_a_symlink_names_is_refused(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    url = f"sqlite:///{tmp_path / 'real' / 'studies.db'}"
    with pytest.raises(OptimizationError, match="would be in the study directory"):
        optimize(write_problem(), [trade_off()], tmp_path / "link", trials=1, storage=url)
    assert not any((tmp_path / "real").iterdir())


def test_a_wrong_url_makes_no_study_directory(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    with pytest.raises(OptimizationError, match="--storage nosuchdatabase://"):
        optimize(
            write_problem(),
            [trade_off()],
            tmp_path / "s",
            trials=1,
            storage="nosuchdatabase://host/studies",
        )
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize("form", ["path", "uri"])
def test_a_database_outside_the_directory_keeps_the_trials_and_no_journal_is_made(
    tmp_path: Path, write_problem: Callable[..., Path], form: str
) -> None:
    url = url_of(tmp_path / "studies.db", form)
    assert storage.sqlite_file(url) == tmp_path / "studies.db"  # where it is then made
    report = optimize(write_problem(), [trade_off()], tmp_path / "s", trials=2, seed=1, storage=url)
    assert [t.status for t in report.trials] == ["valid", "valid"]
    assert (tmp_path / "studies.db").is_file()
    assert sorted(p.name for p in (tmp_path / "s").iterdir()) == [
        "study.json",
        "trial-000000",
        "trial-000001",
    ]
    # read back with the URL it was made with, and nothing written
    before = (tmp_path / "studies.db").read_bytes()
    assert [t.status for t in load_study(tmp_path / "s", storage=url).trials] == ["valid"] * 2
    assert (tmp_path / "studies.db").read_bytes() == before


@pytest.mark.parametrize(
    "url",
    [
        "sqlite://",
        "sqlite:///:memory:",
        "sqlite:///file::memory:?uri=true",
        "sqlite:///file:studies.db?mode=memory&uri=true",
        "postgresql://host/studies",
    ],
)
def test_a_database_that_is_no_sqlite_file_is_none(url: str) -> None:
    assert storage.sqlite_file(url) is None


@pytest.mark.parametrize(
    "database",
    [
        "studies.db",
        "file:studies.db",  # without uri=true, a name like any other
        "file::memory:",
        "studies.db?mode=memory",  # without uri=true, the query is ignored
        "file:studies.db?mode=memory",
        "file:studies.db?uri=false",
        "file:studies.db?uri=true",
        "file:studies.db?uri=1",
        "file:studies.db?uri=True&mode=rwc",
        "file:my%20studies.db?uri=true",
        "file:my%2520studies.db?uri=true",
        "file:{here}/deeper/studies.db?uri=true",
        "file://{here}/deeper/studies.db?uri=true",
        "file://localhost{here}/deeper/studies.db?uri=true",
        "file:studies.db?mode=memory&uri=true",
        "file::memory:?uri=true",
        ":memory:",
    ],
)
def test_the_file_of_a_sqlite_url_is_the_one_sqlite_makes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    # however the installed SQLAlchemy and SQLite read the URL: a relative name is relative to
    # the working directory, and a database in memory makes no file
    import sqlalchemy

    monkeypatch.chdir(tmp_path)
    (tmp_path / "deeper").mkdir()
    url = f"sqlite:///{database.format(here=tmp_path)}"
    with warnings.catch_warnings():  # of query parameters it ignores, and of its pool
        warnings.simplefilter("ignore")
        engine = sqlalchemy.create_engine(url)
        engine.connect().close()
        engine.dispose()
    made = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert storage.sqlite_file(url) == (made[0] if made else None)
    assert len(made) <= 1
