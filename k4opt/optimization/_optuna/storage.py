"""Internal: where the trials of a study are kept, and how failures of that storage leave.

The journal ``study.journal`` in the study directory, or a database (``--storage URL``). For a
run it is opened as Optuna opens it (:func:`open_storage`); for reading, an existing one only and
so that it refuses every write (:func:`existing_storage`). A database URL is recorded and shown
without its secrets (:func:`redacted`), and a failure of Optuna or the storage leaves as an
:class:`~k4opt.optimization.OptimizationError` with them scrubbed (:func:`leaving`).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import urllib.parse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import BaseStorage, JournalStorage, RDBStorage
from optuna.storages.journal import BaseJournalBackend, JournalFileBackend, JournalFileOpenLock

from k4opt.errors import K4OptError
from k4opt.optimization._study import JOURNAL_NAME, STUDY_NAME
from k4opt.optimization.errors import OptimizationError


def redacted(url: str) -> str:
    """``url`` as the study records it and messages show it: its password and the value of
    every query parameter (``?password=...``, ``sslkey=...``) are ``***``, and it has no
    fragment."""
    parts = urllib.parse.urlsplit(url)
    text = url.partition("#")[0]  # edited in place: rebuilding it would drop sqlite:///'s //
    if parts.password is not None:
        userinfo, _, host = parts.netloc.rpartition("@")
        text = text.replace(parts.netloc, f"{userinfo.partition(':')[0]}:***@{host}", 1)
    if parts.query:
        keys = [key for key, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)]
        text = text.partition("?")[0] + "?" + "&".join(f"{key}=***" for key in keys)
    return text


def _scrubbed(text: str, url: str) -> str:
    """``text`` (a database error, say) with ``url`` and every secret in it redacted."""
    parts = urllib.parse.urlsplit(url)
    secrets_ = [parts.password, parts.fragment]
    secrets_ += [value for _, value in urllib.parse.parse_qsl(parts.query)]
    text = text.replace(url, redacted(url))
    spellings = {
        spelling
        for secret in secrets_
        if secret
        for spelling in (secret, urllib.parse.unquote(secret), urllib.parse.quote(secret, safe=""))
    }
    for secret in sorted(spellings, key=len, reverse=True):
        text = text.replace(secret, "***")
    return text


def open_storage(directory: Path, url: str | None) -> BaseStorage:
    if url is None:
        path = str(directory / JOURNAL_NAME)
        return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    try:
        return RDBStorage(url)
    except ImportError as exc:
        raise OptimizationError(
            f"--storage {redacted(url)} needs SQLAlchemy ({exc}); install it, or leave out "
            "--storage to keep the study's storage in its directory"
        ) from None
    except Exception as exc:
        raise OptimizationError(f"--storage {redacted(url)}: {_scrubbed(str(exc), url)}") from None


def storage_record(url: str | None) -> dict[str, str]:
    return {"journal": JOURNAL_NAME} if url is None else {"url": redacted(url)}


def close(storage: BaseStorage | None) -> None:
    """Close the connections a database storage holds open; a journal holds none."""
    if isinstance(storage, optuna.storages.RDBStorage):  # the class, which a test may replace here
        storage.engine.dispose()


class _ReadOnlyJournal(BaseJournalBackend):
    """The journal of a study, read as it is: anything that would append to it raises."""

    def __init__(self, path: Path) -> None:
        # JournalFileBackend creates a file that does not exist: the caller checked this one does
        self._file = JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path)))

    def read_logs(self, log_number_from: int) -> Iterable[dict[str, Any]]:
        return self._file.read_logs(log_number_from)

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        raise OptimizationError("reading a study never writes to its journal")


_READ_ONLY_POSTGRESQL = "-c default_transaction_read_only=on"


_OPTUNA_TABLES = ("studies", "trials", "version_info", "alembic_version")


def _read_only(url: str) -> dict[str, Any]:
    """The engine options that open the database ``url`` names so that it refuses every write.

    SQLite is opened in its read-only mode, which also refuses a file that does not exist instead
    of creating it; a database in WAL mode is refused, since a reader writes files beside it
    there. PostgreSQL over libpq (``psycopg2``, ``psycopg``) with
    ``default_transaction_read_only``, added to the URL's own ``options``. Any other database is
    refused: nothing would keep it from being written.
    """
    try:
        from sqlalchemy.engine import make_url
        from sqlalchemy.exc import ArgumentError
    except ImportError as exc:
        raise OptimizationError(
            f"--storage {redacted(url)} needs SQLAlchemy ({exc}); install it"
        ) from None
    try:
        parsed = make_url(url)
    except (ArgumentError, ValueError):
        raise OptimizationError(f"--storage {redacted(url)} is not a database URL") from None
    backend, driver = parsed.get_backend_name(), parsed.get_driver_name()
    if backend == "sqlite":
        database = parsed.database or ""
        path = None if database.startswith("file:") else _database_file(database)
        if path is None:
            raise OptimizationError(f"--storage {redacted(url)} names no database file to read")
        if not path.is_file():
            raise OptimizationError(
                f"--storage {redacted(url)}: {path} does not exist, so it holds no study"
            )
        if _write_ahead(path):
            raise OptimizationError(
                f"--storage {redacted(url)}: {path} is in SQLite's WAL mode, in which even a "
                "read-only connection writes -wal and -shm files beside it; k4opt optimize makes "
                "no such database, and reading opens none"
            )
        uri = f"file:{urllib.parse.quote(str(path))}?mode=ro"  # codespell:ignore ro
        return {"creator": lambda: sqlite3.connect(uri, uri=True, check_same_thread=False)}
    if backend == "postgresql" and driver in ("psycopg2", "psycopg"):
        given = parsed.query.get("options", ())
        options = [given] if isinstance(given, str) else list(given)
        return {"connect_args": {"options": " ".join([*options, _READ_ONLY_POSTGRESQL])}}
    raise OptimizationError(
        f"--storage {redacted(url)}: a study is read only from a SQLite or a PostgreSQL "
        f"(psycopg2, psycopg) database, which can be opened so that it refuses every write; "
        f"this is {backend}+{driver}"
    )


def sqlite_file(url: str) -> Path | None:
    """The file of the SQLite database ``url`` names, as an absolute path (a relative one is
    relative to the working directory), also when it is named by a URI filename
    (``sqlite:///file:/path/studies.db?uri=true``); ``None`` for any other database, one in
    memory, and a URL that cannot be parsed (opening it says why)."""
    try:
        from sqlalchemy.engine import make_url
        from sqlalchemy.exc import ArgumentError
    except ImportError:
        return None
    try:
        parsed = make_url(url)
    except (ArgumentError, ValueError):
        return None
    if parsed.get_backend_name() != "sqlite" or parsed.query.get("mode") == "memory":
        return None
    return _database_file(parsed.database)


def _database_file(database: str | None) -> Path | None:
    """The file the database of a SQLite URL names, absolute; ``None`` for none or one in
    memory. A URI filename (``file:``) names the file of its path, which SQLite decodes."""
    if database and database.startswith("file:"):
        database = urllib.parse.unquote(urllib.parse.urlsplit(database).path)
    if not database or database == ":memory:":
        return None
    return Path(os.path.abspath(database))


def _write_ahead(path: Path) -> bool:
    """Whether the SQLite database ``path`` is in WAL mode: the file format versions in its
    header (bytes 18 and 19) are 2 rather than 1."""
    with path.open("rb") as handle:
        header = handle.read(20)
    return 2 in header[18:20]


def _check_set_up(url: str, options: Mapping[str, Any]) -> None:
    """Refuse a database that Optuna would set up or repair as it opens it (create its tables,
    insert its version, stamp its schema revision): reading sets nothing up. The database is
    read through an engine of its own, opened as ``options`` say."""
    import sqlalchemy

    engine = sqlalchemy.create_engine(url, **options)
    try:
        with engine.connect() as connection:
            tables = set(sqlalchemy.inspect(connection).get_table_names())
            rows = {
                table: connection.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {table}")).scalar()
                for table in ("version_info", "alembic_version")
                if table in tables
            }
    finally:
        engine.dispose()
    missing = [table for table in _OPTUNA_TABLES if table not in tables]
    if missing or rows != {"version_info": 1, "alembic_version": 1}:
        lacks = f"the tables {', '.join(missing)}" if missing else "its version rows"
        raise OptimizationError(
            f"--storage {redacted(url)} is not an Optuna database set up completely (it lacks "
            f"{lacks}); reading a study never sets one up or repairs it"
        )


def existing_storage(directory: Path, document: Mapping[str, Any], url: str | None) -> BaseStorage:
    """The storage that holds the study's trials, opened so that nothing is written to it: the
    journal in its directory, or the database ``url``, which ``study.json`` names redacted."""
    named = document["optimizer"].get("storage")
    database = named.get("url") if isinstance(named, Mapping) else None
    if url is None:
        if database is not None:
            raise OptimizationError(
                f"the study keeps its trials in the database {database}; give its URL with "
                f"--storage ({STUDY_NAME} does not hold its password)"
            )
        if named != storage_record(None):
            raise OptimizationError(f"{STUDY_NAME} names a storage k4opt does not read: {named!r}")
        journal = directory / JOURNAL_NAME
        if not journal.is_file():
            raise OptimizationError(
                f"{journal} does not exist: the study's trials were never stored, or the file was "
                "moved away"
            )
        try:
            return JournalStorage(_ReadOnlyJournal(journal))
        except (OSError, ValueError) as exc:
            raise OptimizationError(f"cannot read {journal}: {exc}") from exc
    if named != storage_record(url):
        kept = f"the database {database}" if database is not None else JOURNAL_NAME
        advice = "" if database is not None else "; leave out --storage"
        raise OptimizationError(
            f"the study keeps its trials in {kept}, not in {redacted(url)}{advice}"
        )
    options = _read_only(url)
    try:
        _check_set_up(url, options)
        return RDBStorage(url, engine_kwargs=options, skip_table_creation=True)
    except K4OptError:
        raise
    except Exception as exc:
        message = _scrubbed(f"{type(exc).__name__}: {exc}", url)
        raise OptimizationError(f"--storage {redacted(url)}: {message}") from None


def _chain(exc: BaseException) -> list[BaseException]:
    """``exc`` and the exceptions it was raised from or during, as a traceback shows them."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return chain


def _storage_errors() -> tuple[type[BaseException], ...]:
    """What Optuna and the storage raise when they fail, as opposed to a bug of k4opt."""
    errors: tuple[type[BaseException], ...] = (optuna.exceptions.OptunaError, OSError)
    database = sys.modules.get("sqlalchemy.exc")  # imported only if a database was used
    if database is not None:
        errors += (database.SQLAlchemyError,)
    return errors


def leaving(
    exc: Exception, url: str | None, what: str = "the study cannot go on"
) -> tuple[Exception | None, bool]:
    """What leaves :func:`optimize` or :func:`load` in place of ``exc`` (``None``: ``exc``
    itself), and whether it must leave without ``exc``, which then names a secret of the storage
    URL ``url``.

    A k4opt error leaves as it is, and so does a bug of k4opt; a failure of Optuna or the
    storage becomes an :class:`OptimizationError` that starts with ``what``. Each is redacted
    when anything in its chain names a secret of ``url``.
    """
    text = "\n".join(str(e) for e in _chain(exc))
    secret = url is not None and _scrubbed(text, url) != text
    if isinstance(exc, K4OptError) or not isinstance(exc, _storage_errors()):
        message = str(exc) if isinstance(exc, K4OptError) else f"{type(exc).__name__}: {exc}"
        if not secret:
            return None, False
        assert url is not None
        return OptimizationError(_scrubbed(message, url)), True
    message = f"{what}: {type(exc).__name__}: {exc}"
    return OptimizationError(message if url is None else _scrubbed(message, url)), secret
