"""A new study kept in a database (``--storage URL``): the database is reached only once the
study's directory is known to be usable, and before anything is made in it, so neither a
directory in use nor a wrong URL leaves anything behind; a SQLite database of a new study lies
outside its directory. The cases need SQLAlchemy and alembic."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("optuna")
pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")

from k4opt.optimization import OptimizationError, optimize
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


def url_of(path: Path, form: str) -> str:
    """The URL of the SQLite database ``path``: as a path, or as a URI filename (SQLite's
    ``file:`` form, which SQLAlchemy passes on with ``uri=true``)."""
    return {
        "path": f"sqlite:///{path}",
        "uri": f"sqlite:///file:{path}?uri=true",
        "uri with authority": f"sqlite:///file://{path}?uri=true",
    }[form]


@pytest.mark.parametrize("form", ["path", "uri", "uri with authority"])
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


def test_a_uri_filename_names_the_file_of_its_path_decoded_and_relative_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert storage.sqlite_file("sqlite:///file:studies.db?uri=true") == tmp_path / "studies.db"
    assert storage.sqlite_file("sqlite:///file:my%2520studies.db?uri=true") == (
        tmp_path / "my studies.db"  # %25 is decoded by the URL, %20 by SQLite
    )
