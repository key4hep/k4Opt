"""``k4opt publish``: a study's snapshot, whole and never changed, then its catalog entry, which
only a later state of the study replaces; the study itself is only read."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("optuna")

from k4opt.cli import main
from k4opt.optimization import load_study, optimize
from k4opt.publication import (
    CATALOG_NAME,
    Catalog,
    PublicationError,
    local_paths,
    publish,
)
from tests.support.common import tree_digest
from tests.support.evaluators import trade_off
from tests.support.software import LABEL
from tests.support.studies import TRADE, mixed_study, study
from tests.unit.optimization.conftest import evaluator_from


def files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_a_study_is_published_as_its_json_form_and_named_in_its_catalog(tmp_path: Path) -> None:
    results = load_study(mixed_study(tmp_path / "s"))
    before = tree_digest(tmp_path / "s")
    entry = publish(results, tmp_path / "pub")
    assert tree_digest(tmp_path / "s") == before  # the study is only read
    snapshot = f"studies/{results.id}/snapshots/{entry.snapshot.sha256[:16]}.json"
    assert entry.software is not None
    assert sorted(files(tmp_path / "pub")) == sorted(
        [CATALOG_NAME, snapshot, entry.software.path]
    )  # and no lock left
    data = (tmp_path / "pub" / snapshot).read_bytes()
    assert data == results.to_json().encode()  # what k4opt study export --format json writes
    assert (entry.snapshot.path, entry.snapshot.bytes) == (snapshot, len(data))
    assert (entry.id, entry.name, entry.problem, entry.created) == (
        results.id,
        results.name,
        "trade",
        results.created,
    )
    # what a list of studies sorts by: the detector it starts from, the Key4hep stack
    assert (entry.geometry, entry.key4hep) == (results.problem.geometry, LABEL)
    assert (entry.trials, entry.parameters, entry.objectives, entry.constraints) == (8, 1, 2, 1)
    assert entry.outcomes == results.outcomes and entry.finished == 8
    assert Catalog.from_json((tmp_path / "pub" / CATALOG_NAME).read_bytes()) == Catalog((entry,))


def test_publishing_the_same_state_again_at_the_same_check_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software import readiness

    monkeypatch.setattr(readiness, "_now", lambda: "2026-11-01T10:00:00+00:00")
    results = load_study(mixed_study(tmp_path / "s"))
    first = publish(results, tmp_path / "pub")
    held = files(tmp_path / "pub")
    assert publish(load_study(tmp_path / "s"), tmp_path / "pub") == first
    assert files(tmp_path / "pub") == held  # nor the catalog's time


def test_a_later_state_gets_its_own_snapshot_and_an_earlier_one_is_refused(
    tmp_path: Path,
) -> None:
    output = mixed_study(tmp_path / "s", trials=4)
    earlier = load_study(output)
    first = publish(earlier, tmp_path / "pub")
    old = (tmp_path / "pub" / first.snapshot.path).read_bytes()
    optimize(tmp_path / "s.yaml", [trade_off()], output, trials=2, resume=True)
    later = publish(load_study(output), tmp_path / "pub")
    assert later.snapshot != first.snapshot and later.trials == 6
    assert (tmp_path / "pub" / first.snapshot.path).read_bytes() == old  # never changed
    catalog = (tmp_path / "pub" / CATALOG_NAME).read_bytes()
    with pytest.raises(PublicationError, match=r"holds a later state of study .*: 6 trials"):
        publish(earlier, tmp_path / "pub")  # a publisher that read the study before
    assert (tmp_path / "pub" / CATALOG_NAME).read_bytes() == catalog
    assert Catalog.from_json(catalog).studies == (later,)


def test_studies_share_a_catalog_each_by_its_id(tmp_path: Path) -> None:
    one = publish(load_study(mixed_study(tmp_path / "one", trials=2)), tmp_path / "pub")
    two = publish(load_study(mixed_study(tmp_path / "two", trials=3)), tmp_path / "pub")
    catalog = Catalog.from_json((tmp_path / "pub" / CATALOG_NAME).read_bytes())
    assert catalog.studies == tuple(sorted((one, two), key=lambda entry: entry.id))


def test_a_publication_inside_the_study_is_refused(tmp_path: Path) -> None:
    results = load_study(mixed_study(tmp_path / "s", trials=2))
    os.symlink(tmp_path / "s", tmp_path / "link")
    for inside in (tmp_path / "s" / "pub", tmp_path / "link" / "pub"):
        with pytest.raises(PublicationError, match="in the study directory"):
            publish(results, inside)
    assert not (tmp_path / "s" / "pub").exists()


def test_a_malformed_catalog_is_refused_and_left_as_it_is(tmp_path: Path) -> None:
    results = load_study(mixed_study(tmp_path / "s", trials=2))
    (tmp_path / "pub").mkdir()
    (tmp_path / "pub" / CATALOG_NAME).write_text('{"format": "k4opt.catalog/1", "studies": [1]}')
    with pytest.raises(PublicationError, match=r"is left as it is: .*studies\[0\] is not an obj"):
        publish(results, tmp_path / "pub")
    held = (tmp_path / "pub" / CATALOG_NAME).read_text()
    assert held == '{"format": "k4opt.catalog/1", "studies": [1]}'
    assert not (tmp_path / "pub" / ".catalog.lock").exists()


def test_a_held_lock_refuses_and_stays(tmp_path: Path) -> None:
    results = load_study(mixed_study(tmp_path / "s", trials=2))
    (tmp_path / "pub").mkdir()
    (tmp_path / "pub" / ".catalog.lock").touch()
    with pytest.raises(PublicationError, match=r"\.catalog\.lock exists: another publication"):
        publish(results, tmp_path / "pub")
    assert (tmp_path / "pub" / ".catalog.lock").exists()  # another's: never taken over
    assert not (tmp_path / "pub" / CATALOG_NAME).exists()


def test_a_catalog_that_cannot_be_written_stays_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = publish(load_study(mixed_study(tmp_path / "one", trials=2)), tmp_path / "pub")
    catalog = (tmp_path / "pub" / CATALOG_NAME).read_bytes()

    def failing(path: Path, text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("k4opt.jsonio.replace", failing)
    with pytest.raises(PublicationError, match=r"cannot write .*catalog\.json: disk full"):
        publish(load_study(mixed_study(tmp_path / "two", trials=2)), tmp_path / "pub")
    assert (tmp_path / "pub" / CATALOG_NAME).read_bytes() == catalog
    assert Catalog.from_json(catalog).studies == (first,)
    assert not (tmp_path / "pub" / ".catalog.lock").exists()


def test_a_file_in_place_of_a_snapshot_is_never_replaced(tmp_path: Path) -> None:
    results = load_study(mixed_study(tmp_path / "s", trials=2))
    entry = publish(results, tmp_path / "elsewhere")
    path = tmp_path / "pub" / entry.snapshot.path
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(PublicationError, match="exists with other bytes"):
        publish(results, tmp_path / "pub")
    assert path.read_text() == "{}" and not (tmp_path / "pub" / CATALOG_NAME).exists()


PROBE = """
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        settings = {"x": float, "calibration": str}
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            x = request.settings["x"]
            if x < 200:
                self.invalid(f"x = {x}: see /home/me/logs/ddsim.log, set up by /cvmfs/sw/setup.sh")
            return {("analysis", "a"): x, ("analysis", "b"): 1 / x}
"""


def test_the_local_paths_a_snapshot_carries_are_told_before_it_is_written(tmp_path: Path) -> None:
    probe = evaluator_from(tmp_path / "evaluators", "probe", PROBE)
    output = study(
        tmp_path / "s",
        6,
        implementations=[probe],
        parameter={
            "targets": {"geometry": "InnerTracker_Barrel_radius_1"},
            "type": "categorical",
            "choices": [300, 170],
        },
        evaluators=[TRADE | {"settings": {"calibration": "/eos/user/m/me/calibration.json"}}],
    )
    results = load_study(output)
    invalid = [t.number for t in results.trials if t.status == "invalid"]
    told: list[str] = []
    publish(results, tmp_path / "pub", warn=told.append)
    expected = [
        "the problem's definition.evaluators[0].settings.calibration holds the local path "
        "/eos/user/m/me/calibration.json; it is published",
        *(
            f"trial {number}'s message holds the local path /home/me/logs/ddsim.log; it is "
            "published"
            for number in invalid
        ),
    ]
    assert invalid and told == expected  # not /cvmfs, which is public; nor "1/mm"
    assert [where for where, _ in local_paths(results)] == [
        "the problem's definition.evaluators[0].settings.calibration",
        *(f"trial {number}'s message" for number in invalid),
    ]


def test_k4opt_publish_says_what_it_published_and_reads_the_study_as_k4opt_study_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software import keep

    # k4opt's own checkout may hold edits at any moment: they are not this test's to publish
    monkeypatch.setattr(keep, "checkout", lambda path, release: None)
    output = mixed_study(tmp_path / "s")
    out, err = io.StringIO(), io.StringIO()
    command = ["publish", str(output), "--output", str(tmp_path / "pub"), "--yes"]
    assert main(command, stdout=out, stderr=err) == 0  # the mini geometry is local code
    (entry,) = Catalog.from_json((tmp_path / "pub" / CATALOG_NAME).read_bytes()).studies
    assert f"{entry.name} (snapshot {entry.snapshot.sha256[:8]}): code of " in out.getvalue()
    assert "8 trials (" in out.getvalue()
    assert out.getvalue().rstrip().endswith(f"as {tmp_path / 'pub'}/{entry.snapshot.path}")
    assert err.getvalue() == ""
    (tmp_path / "s" / "study.json").write_text(json.dumps({"schema": "other"}))
    err = io.StringIO()
    assert main(command, stdout=io.StringIO(), stderr=err) == 1
    assert err.getvalue().startswith("k4opt: error: ")


def test_a_study_whose_id_is_no_uuid_is_never_written_anywhere(tmp_path: Path) -> None:
    import dataclasses

    results = load_study(mixed_study(tmp_path / "s", trials=2))
    crafted = dataclasses.replace(results, id="../../outside")  # a study.json made to escape
    with pytest.raises(PublicationError, match="is not the id of a study"):
        publish(crafted, tmp_path / "pub")
    assert not (tmp_path / "outside").exists() and not (tmp_path / "pub").exists()


def test_a_snapshot_larger_than_a_reader_reads_is_never_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = load_study(mixed_study(tmp_path / "s", trials=2))
    monkeypatch.setattr("k4opt.publication.write.MAX_SNAPSHOT_BYTES", 1000)
    with pytest.raises(PublicationError, match=r"would be \d+ bytes, more than the 1000"):
        publish(results, tmp_path / "pub")
    assert not (tmp_path / "pub").exists()
