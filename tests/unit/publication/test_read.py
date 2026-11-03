"""Reading a publication: the same studies from a directory and over HTTP, and nothing read as a
study that is not exactly what was published; a server that does not answer is told apart from
one that answers wrongly."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from k4opt.publication import (
    CATALOG_NAME,
    Catalog,
    CatalogEntry,
    NothingPublishedError,
    PublicationError,
    PublicationUnavailableError,
    open_publication,
)
from k4opt.publication import read as read_module
from tests.support.serve import Server, served

REPOSITORY = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A publication of one real study whose trials ended every way a record can."""
    pytest.importorskip("optuna")
    from k4opt.optimization import load_study
    from k4opt.publication import publish
    from tests.support.studies import mixed_study

    root = tmp_path_factory.mktemp("published")
    with pytest.MonkeyPatch.context() as monkeypatch:
        from tests.support.studies import mini_environment

        mini_environment(monkeypatch)
        results = load_study(mixed_study(root / "s"))
    publish(results, root / "pub")
    (root / "s.json").write_text(results.to_json())
    return root


@pytest.fixture
def server(published: Path) -> Iterator[Server]:
    with served(published / "pub") as running:
        yield running


def entry_of(root: Path) -> CatalogEntry:
    (entry,) = Catalog.from_json((root / CATALOG_NAME).read_bytes()).studies
    return entry


def copy_of(published: Path, tmp_path: Path) -> Path:
    """A copy of the publication, to break."""
    copy = tmp_path / "pub"
    for path in (published / "pub").rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(published / "pub")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return copy


def test_a_directory_and_its_url_give_the_same_studies(published: Path, server: Server) -> None:
    expected = (published / "s.json").read_text()
    for location in (published / "pub", server.url, f"{server.url}/"):
        publication = open_publication(location)
        catalog = publication.catalog()
        (entry,) = catalog.studies
        study = publication.study(entry)
        assert study.to_json() == expected  # the study as it was published
        assert (study.directory, study.id) == (None, entry.id)
    assert server.requests == [CATALOG_NAME, entry.snapshot.path] * 2  # never a listing


def test_nothing_published_is_an_answer_of_its_own(tmp_path: Path, server: Server) -> None:
    server.faults[CATALOG_NAME] = "missing"
    for location in (tmp_path, tmp_path / "none", server.url):
        with pytest.raises(NothingPublishedError, match="nothing is published at"):
            open_publication(location).catalog()
    (tmp_path / CATALOG_NAME).write_text(Catalog().to_json())
    assert open_publication(tmp_path).catalog().studies == ()


def test_a_server_that_does_not_answer_is_told_apart(
    published: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = entry_of(published / "pub")
    publication = open_publication(server.url)
    monkeypatch.setattr(read_module, "TIMEOUT", 0.5)
    for fault in ("error", "stall"):
        server.faults[CATALOG_NAME] = fault
        with pytest.raises(PublicationUnavailableError, match=r"cannot be read just now|answers"):
            publication.catalog()
    server.faults.clear()
    for fault in ("cut", "short", "error"):
        server.faults[entry.snapshot.path] = fault
        with pytest.raises(PublicationUnavailableError):
            publication.study(entry)
    server.faults.clear()
    assert publication.study(entry).id == entry.id  # nothing of a failed read was kept
    with pytest.raises(PublicationUnavailableError, match="just now"):
        open_publication(closed_port()).catalog()


def closed_port() -> str:
    """The URL of a port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


def test_a_redirect_is_refused_not_followed(published: Path, server: Server) -> None:
    server.faults[CATALOG_NAME] = "redirect"
    with pytest.raises(PublicationError, match=r"redirects to https://auth\.example\.org/login"):
        open_publication(server.url).catalog()
    assert server.requests == [CATALOG_NAME]


Change = Callable[[dict[str, Any]], object]
BROKEN_ENTRIES: dict[str, tuple[Change, str]] = {
    "a path out of the publication": (
        lambda e: e["snapshot"].update(path="../../etc/passwd"),
        "not 'studies/",
    ),
    "a URL": (lambda e: e["snapshot"].update(path="https://example.org/x.json"), "not 'studies/"),
    "an id that is a path": (lambda e: e.update(id="../x"), "not the id of a study"),
    "another SHA-256": (lambda e: e["snapshot"].update(sha256="0" * 64), "not 'studies/"),
    "a count that is no count": (lambda e: e.update(trials=-1), "not a count"),
    "counts that do not add up": (lambda e: e["outcomes"].update(running=1), "count 9 trials"),
    "an unknown outcome": (lambda e: e["outcomes"].update(done=0), "has the keys"),
    "a missing key": (lambda e: e.pop("published"), "has the keys"),
}


@pytest.mark.parametrize("case", list(BROKEN_ENTRIES))
def test_a_catalog_with_a_broken_entry_is_refused_whole(
    published: Path, tmp_path: Path, case: str
) -> None:
    copy = copy_of(published, tmp_path)
    data = json.loads((copy / CATALOG_NAME).read_text())
    change, message = BROKEN_ENTRIES[case]
    change(data["studies"][0])
    (copy / CATALOG_NAME).write_text(json.dumps(data))
    with pytest.raises(PublicationError, match=message):
        open_publication(copy).catalog()


@pytest.mark.parametrize(
    "text",
    [
        "{",
        '{"format": "k4opt.catalog/0", "studies": []}',
        '{"format": "k4opt.catalog/1", "studies": [], "more": 1}',
        '{"format": "k4opt.catalog/1", "studies": NaN}',
    ],
)
def test_a_malformed_catalog_is_refused(tmp_path: Path, text: str) -> None:
    (tmp_path / CATALOG_NAME).write_text(text)
    with pytest.raises(PublicationError, match=CATALOG_NAME):
        open_publication(tmp_path).catalog()


def test_a_study_named_twice_is_refused(published: Path, tmp_path: Path) -> None:
    copy = copy_of(published, tmp_path)
    data = json.loads((copy / CATALOG_NAME).read_text())
    data["studies"] *= 2
    (copy / CATALOG_NAME).write_text(json.dumps(data))
    with pytest.raises(PublicationError, match="name a study more than once"):
        open_publication(copy).catalog()


def test_a_snapshot_is_read_only_as_the_bytes_its_entry_names(
    published: Path, tmp_path: Path
) -> None:
    copy = copy_of(published, tmp_path)
    entry = entry_of(copy)
    path = copy / entry.snapshot.path
    data = path.read_bytes()
    publication = open_publication(copy)
    path.write_bytes(data.replace(b'"valid"', b'"VALID"', 1))  # as long, other bytes
    with pytest.raises(PublicationError, match="not the snapshot its catalog names"):
        publication.study(entry)
    path.write_bytes(data + b" ")
    with pytest.raises(PublicationError, match="larger than"):
        publication.study(entry)
    path.write_bytes(data[:100])  # copied in part, so far
    with pytest.raises(PublicationUnavailableError, match="arrived with 100 of"):
        publication.study(entry)
    path.unlink()
    with pytest.raises(PublicationError, match="not there, though the catalog names it"):
        publication.study(entry)


def test_a_snapshot_must_be_the_study_its_entry_describes(published: Path, tmp_path: Path) -> None:
    copy = copy_of(published, tmp_path)
    data = json.loads((copy / CATALOG_NAME).read_text())
    data["studies"][0]["name"] = "another"
    (copy / CATALOG_NAME).write_text(json.dumps(data))
    publication = open_publication(copy)
    (entry,) = publication.catalog().studies
    with pytest.raises(
        PublicationError, match="not the study its catalog entry describes: its name"
    ):
        publication.study(entry)


def test_a_symlink_out_of_the_publication_is_not_followed(published: Path, tmp_path: Path) -> None:
    copy = copy_of(published, tmp_path)
    (tmp_path / "elsewhere.json").write_text((copy / CATALOG_NAME).read_text())
    (copy / CATALOG_NAME).unlink()
    os.symlink(tmp_path / "elsewhere.json", copy / CATALOG_NAME)
    with pytest.raises(PublicationError, match="leads out of the publication"):
        open_publication(copy).catalog()


def test_only_a_directory_or_an_http_url_is_a_location() -> None:
    with pytest.raises(PublicationError, match="not ftp://"):
        open_publication("ftp://example.org/pub")
    with pytest.raises(PublicationError, match="names a host and no query"):
        open_publication("https://example.org/pub?x=1")


def test_a_publication_is_read_without_optuna(published: Path) -> None:
    code = f"""
import sys
sys.modules["optuna"] = None  # importing it fails
from k4opt.publication import open_publication
publication = open_publication({str(published / "pub")!r})
(entry,) = publication.catalog().studies
study = publication.study(entry)
assert len(study.trials) == 8 and study.pareto
print(study.to_json(), end="")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY,
        env=os.environ
        | {"PYTHONPATH": os.pathsep.join([str(REPOSITORY), os.environ.get("PYTHONPATH", "")])},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == (published / "s.json").read_text()
