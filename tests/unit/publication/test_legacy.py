"""What was published before k4opt recorded the software stays readable: a problem then declared
its stack, and a study held ``KEY4HEP_STACK`` and the search variables.

``tests/fixtures/publication/legacy`` was written by ``k4opt publish`` at 3157c81, the last commit
of that form, and must never be written again."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

from k4opt.optimization import StudyResults
from k4opt.publication import open_publication

LEGACY = Path(__file__).resolve().parents[2] / "fixtures" / "publication" / "legacy"


def studies() -> dict[str, StudyResults]:
    publication = open_publication(LEGACY)
    return {
        "nested" if read.scope.nested else "plain": read
        for read in map(publication.study, publication.catalog().studies)
    }


def test_a_published_snapshot_reads_back_and_is_what_its_catalog_says() -> None:
    read = studies()
    assert sorted(read) == ["nested", "plain"]
    assert [study.key4hep for study in read.values()] == ["nightly 2026-10-20"] * 2
    assert "software" not in read["plain"].document
    assert "environment" not in read["plain"].problem.to_dict()


def test_the_inner_studies_of_a_published_nested_study_read_back() -> None:
    inner = studies()["nested"].trial(0).inner
    assert inner is not None and inner.key4hep == "nightly 2026-10-20"
    assert len(inner.trials) == 3


def test_a_snapshot_says_again_what_it_said() -> None:
    plain = studies()["plain"]
    assert StudyResults.from_dict(plain.to_dict()).to_json() == plain.to_json()
