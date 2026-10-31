"""Reading a study kept in a database: the database itself refuses writes (SQLite opened read
only, PostgreSQL in read-only transactions), any other database is refused, and so is one that
Optuna would set up or repair as it opens it. The SQLite cases need SQLAlchemy and alembic."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("optuna")
pytest.importorskip("sqlalchemy")

from k4opt.optimization import OptimizationError, load_study, optimize
from k4opt.optimization._optuna import storage
from tests.support.common import tree_digest
from tests.support.evaluators import trade_off


def test_a_postgresql_database_is_read_in_read_only_transactions() -> None:
    url = "postgresql://k4opt:s3cret@db.cern.ch/studies?options=-c%20search_path%3Dk4opt"
    assert storage._read_only(url) == {
        "connect_args": {"options": "-c search_path=k4opt -c default_transaction_read_only=on"}
    }
    assert storage._read_only("postgresql+psycopg://k4opt:s3cret@db.cern.ch/studies") == {
        "connect_args": {"options": "-c default_transaction_read_only=on"}
    }


@pytest.mark.parametrize(
    "url",
    [
        "mysql://k4opt:s3cret@db.cern.ch/studies",
        "postgresql+pg8000://k4opt:s3cret@db.cern.ch/studies",
        "sqlite://",  # in memory: no study can be there
    ],
)
def test_a_database_that_cannot_be_opened_read_only_is_refused(url: str) -> None:
    with pytest.raises(OptimizationError) as info:
        storage._read_only(url)
    assert "s3cret" not in str(info.value)


@pytest.fixture
def database_study(tmp_path: Path, write_problem: Callable[..., Path]) -> tuple[Path, str]:
    """A study of 3 trials kept in the SQLite database ``tmp_path/studies.db``: its directory and
    the database's URL."""
    pytest.importorskip("alembic")
    url = f"sqlite:///{tmp_path / 'studies.db'}"
    optimize(write_problem(), [trade_off()], tmp_path / "s", trials=3, seed=1, storage=url)
    return tmp_path / "s", url


def test_a_study_in_a_database_reads_back_without_a_write(
    tmp_path: Path, database_study: tuple[Path, str]
) -> None:
    output, url = database_study
    database = tmp_path / "studies.db"
    assert database.read_bytes()[18:20] == b"\x01\x01"  # Optuna's: a rollback journal, not WAL
    before = (tree_digest(output), database.read_bytes())
    study = load_study(output, storage=url)
    assert [t.status for t in study.trials] == ["valid"] * 3
    assert study.pareto  # read from the database as from a journal
    assert (tree_digest(output), database.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["problem.yaml", "s", "studies.db"]


def test_a_database_in_wal_mode_is_refused_and_no_file_is_made_beside_it(
    tmp_path: Path, database_study: tuple[Path, str]
) -> None:
    output, url = database_study
    database = tmp_path / "studies.db"
    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    connection.close()  # the last connection: its -wal and -shm files go
    held = ["problem.yaml", "s", "studies.db"]
    assert sorted(p.name for p in tmp_path.iterdir()) == held
    before = database.read_bytes()
    with pytest.raises(OptimizationError, match=r"studies\.db is in SQLite's WAL mode"):
        load_study(output, storage=url)
    assert sorted(p.name for p in tmp_path.iterdir()) == held  # no -wal, no -shm
    assert database.read_bytes() == before


def test_without_its_url_a_database_study_is_refused_and_no_journal_is_made(
    tmp_path: Path, database_study: tuple[Path, str]
) -> None:
    output, _ = database_study
    with pytest.raises(
        OptimizationError,
        match=r"keeps its trials in the database sqlite:///.*studies\.db; give its URL with",
    ):
        load_study(output)
    assert not (output / "study.journal").exists()
    other = f"sqlite:///{tmp_path / 'other.db'}"
    with pytest.raises(OptimizationError, match=r"not in sqlite:///.*other\.db"):
        load_study(output, storage=other)
    assert not (tmp_path / "other.db").exists()


def test_a_database_that_is_gone_is_refused_and_not_made(
    tmp_path: Path, database_study: tuple[Path, str]
) -> None:
    output, url = database_study
    (tmp_path / "studies.db").rename(tmp_path / "moved.db")
    with pytest.raises(
        OptimizationError, match=r"studies\.db does not exist, so it holds no study"
    ):
        load_study(output, storage=url)
    assert not (tmp_path / "studies.db").exists()


@pytest.mark.parametrize("table", ["version_info", "alembic_version"])
def test_a_database_optuna_would_repair_is_refused_and_left_as_it_is(
    tmp_path: Path, database_study: tuple[Path, str], table: str
) -> None:
    output, url = database_study
    database = tmp_path / "studies.db"
    connection = sqlite3.connect(database)
    with connection:
        connection.execute(f"DELETE FROM {table}")  # what Optuna would write back on opening
    connection.close()
    before = database.read_bytes()
    with pytest.raises(OptimizationError, match=r"lacks its version rows.*never sets one up"):
        load_study(output, storage=url)
    assert database.read_bytes() == before
    connection = sqlite3.connect(database)
    assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
    connection.close()


def test_the_database_itself_refuses_a_write_on_the_readers_connection(
    database_study: tuple[Path, str],
) -> None:
    import sqlalchemy

    _, url = database_study
    engine = sqlalchemy.create_engine(url, **storage._read_only(url))
    try:
        with (
            engine.connect() as connection,
            pytest.raises(sqlalchemy.exc.OperationalError, match="readonly database"),
        ):
            connection.execute(sqlalchemy.text("DELETE FROM trials"))
    finally:
        engine.dispose()
