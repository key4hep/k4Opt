"""What of a study may leave the machine: only what is known to have been built or used, only what
its author approved, nothing a secret scan stopped, and in public nothing but the public form."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from k4opt.cli import main
from k4opt.optimization import StudyResults, load_study
from k4opt.publication import (
    BUDGET,
    CATALOG_NAME,
    NEEDS_ACCESS,
    Catalog,
    Offer,
    Piece,
    PublicationError,
    offer,
    open_publication,
    publish,
    scan,
    sections,
)
from tests.support.published import SECRET, STAGE, add_input, local_study, read_by_stage


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, StudyResults]:
    """One study with local work, for the tests that only read it."""
    monkeypatch = pytest.MonkeyPatch()
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    root = tmp_path_factory.mktemp("local")
    try:
        study = local_study(root, monkeypatch)
        yield root, load_study(study)  # type: ignore[misc]
    finally:
        monkeypatch.undo()


OURS = ("ddmarlinpandora", "helpers", "")  # the local work of these tests; the rest is the
# mini geometry and the test evaluators, local too, and offered as such


def names(found: Any) -> list[str]:
    return sorted(piece.label for piece in found if piece.owner in OURS)


PATCH = (
    b"diff --git a/src/a.cc b/src/a.cc\n"
    b"index 000..111 100644\n--- a/src/a.cc\n+++ b/src/a.cc\n"
    b"@@ -1,2 +1,2 @@\n context\n-old\n+new\n"
    b'diff --git "a/r\\303\\251sum b.txt" "b/r\\303\\251sum b.txt"\n'
    b"new file mode 100644\nindex 000..222\n--- /dev/null\n"
    b'+++ "b/r\\303\\251sum b.txt"\n@@ -0,0 +1 @@\n+diff --git not a header\n'
    b"diff --git a/d.bin b/d.bin\nindex 000..333\nGIT binary patch\nliteral 3\nKc${NkWB>pF\n\n"
)


def test_a_patch_splits_into_its_files_whatever_they_are_called() -> None:
    found = sections(PATCH)
    assert [(s.name, s.binary, s.added, s.removed) for s in found] == [
        ("src/a.cc", False, 1, 1),
        ("résum b.txt", False, 1, 0),
        ("d.bin", True, 0, 0),
    ]
    assert b"".join(s.data for s in found) == PATCH  # every byte is in one part


def test_what_is_used_is_offered_and_what_lies_beside_it_is_held(
    made: tuple[Path, StudyResults],
) -> None:
    found = offer(made[1])
    assert names(found.code) == [
        "ddmarlinpandora/New.h",  # untracked: installed by the build
        "ddmarlinpandora/src/DDCaloHitCreator.cc",  # a tracked file edited
        "helpers/ecal_tweak.py",  # a stage read it
    ]
    assert names(found.inputs) == ["ecal_scale.json"]
    held = {h.label: h.why for h in found.held}
    assert held["ddmarlinpandora/notes.txt"] == "untracked, not used"
    assert held["ddmarlinpandora/blob.dat"] == "untracked, not used"
    assert held["helpers/unrelated.py"] == "not read by any stage"
    assert not found.findings


def test_a_held_file_is_offered_once_named_and_a_name_of_nothing_is_refused(
    made: tuple[Path, StudyResults],
) -> None:
    found = offer(made[1], include=["notes.txt"])
    assert "ddmarlinpandora/notes.txt" in names(found.code)
    with pytest.raises(PublicationError, match=r"named nosuch\.txt"):
        offer(made[1], exclude=["nosuch.txt"])  # a slip must not publish what was meant to stay


def test_a_file_left_out_by_name_is_not_offered(made: tuple[Path, StudyResults]) -> None:
    found = offer(made[1], exclude=["ecal_tweak.py", "ecal_scale.json"])
    assert "helpers/ecal_tweak.py" not in names(found.code)
    assert not found.inputs
    assert sorted(names(found.excluded)) == ["ecal_scale.json", "helpers/ecal_tweak.py"]


def piece(name: str, size: int, group: str = "inputs") -> Piece:
    return Piece(group, "", name, b"x" * size, ("input:1",), (STAGE,))  # type: ignore[arg-type]


def test_a_secret_blocks_with_its_file_and_line_and_cannot_be_released() -> None:
    found = scan(
        [piece("a.txt", 0), Piece("code", "o", "b.txt", b"x\ny\n" + SECRET.encode(), (), ())]
    )
    assert [(f.label, f.line, f.what) for f in found] == [("o/b.txt", 3, "a GitHub token")]
    blocked = Offer(None, findings=tuple(found))
    with pytest.raises(PublicationError, match="possible secrets"):
        blocked.release(code=True)


def test_a_secret_in_a_patch_is_found_at_its_line_in_the_changed_file() -> None:
    data = f"diff --git a/x b/x\n@@ -9,2 +40,2 @@\n a\n-b\n+{SECRET}\n".encode()
    assert [f.line for f in scan([Piece("code", "o", "x", data, (), (), base="abc")])] == [41]


@pytest.mark.parametrize(
    "text",
    [
        "-----BEGIN RSA " + "PRIVATE KEY-----",
        "AKIAABCDEFGHIJKLMNOP",
        'password = "correct-horse"',
        "https://me:hunter22@host.example/x",
        "glpat-abcdefghijklmnopqrstu",
    ],
)
def test_the_scan_knows_the_usual_secrets(text: str) -> None:
    assert scan([Piece("code", "o", "x", text.encode(), (), ())])


@pytest.mark.parametrize(
    "text",
    ['password = os.environ["X"]', "https://github.com/key4hep/k4Opt", "url=${USER}:${PW}@host"],
)
def test_the_scan_leaves_ordinary_text_alone(text: str) -> None:
    assert not scan([Piece("code", "o", "x", text.encode(), (), ())])


def test_a_file_named_as_credentials_blocks() -> None:
    assert [f.what for f in scan([Piece("code", "o", ".netrc", b"", (), ())])] == [
        "a file of credentials, by its name"
    ]


def test_what_passes_the_budget_is_published_only_if_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    big = tmp_path / "overlay.root"
    big.write_bytes(b"y" * (BUDGET + 1))
    add_input(study, big)
    results = load_study(study)
    found = offer(results)
    assert names(found.large) == ["overlay.root"]
    assert "overlay.root" not in names(found.inputs)
    assert found.release(inputs=True).pieces == found.inputs
    assert not found.release().pieces  # nothing that was not approved
    with pytest.raises(PublicationError, match="--include-large"):
        found.release(inputs=["overlay.root"])
    asked = offer(results, include_large=True)
    assert "overlay.root" in names(asked.release(inputs=["overlay.root"]).pieces)


def published_files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_without_approval_only_the_public_form_is_published_and_it_needs_access(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    _, results = made
    entry = publish(results, tmp_path / "pub")
    publication = open_publication(tmp_path / "pub")
    sheet = publication.software(entry)
    assert sheet is not None and sheet.artifacts == ()
    assert sheet.status.endswith("needs access") and sheet.status != NEEDS_ACCESS
    assert sheet.status.startswith("Not rerun · rebuild needed")
    assert not any("artifacts" in name for name in published_files(tmp_path / "pub"))
    access = {r.label: r.access for r in sheet.requirements}
    assert access["ddmarlinpandora"] == "on request"
    assert access["ecal_scale.json"] == "on request"


def test_nothing_private_is_in_the_public_form(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    root, results = made
    entry = publish(results, tmp_path / "pub")
    assert entry.software is not None
    text = (tmp_path / "pub" / entry.software.path).read_text()
    assert str(root) not in text
    assert "ecal_tweak.py" not in text  # a file not approved is not named by a list of files
    assert "notes.txt" not in text and "blob.dat" not in text
    assert "MARLIN_DLL" not in text
    sheet = json.loads(text)
    calibration = next(r for r in sheet["requirements"] if r["label"] == "ecal_scale.json")
    assert calibration["identified"] == "observed input"  # no size, no variable
    assert set(calibration["facts"]) == {"commitment"}


def test_what_was_approved_is_published_exactly_and_reads_back(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    _, results = made
    found = offer(results)
    release = found.release(code=True, inputs=["ecal_scale.json"])
    entry = publish(results, tmp_path / "pub", release=release)
    publication = open_publication(tmp_path / "pub")
    sheet = publication.software(entry)
    assert sheet is not None
    by = {artifact.name: artifact for artifact in sheet.artifacts}
    patch = by["ddmarlinpandora.patch"]
    assert patch.kind == "patch"
    assert {file.name for file in patch.files} == {"New.h", "src/DDCaloHitCreator.cc"}
    data = publication.artifact(patch)
    assert b"1.05 * eCalToEMGeV" in data
    assert b"not for anyone" not in data and b"binary" not in data  # what was held back stays
    assert publication.artifact(by["ecal_scale.json"]) == b'{"scale": 1.05}\n'
    assert all(a.path.startswith(f"studies/{results.id}/artifacts/") for a in sheet.artifacts)
    assert {r.label: r.access for r in sheet.requirements}["ddmarlinpandora"] == "public"
    assert "rebuild needed" in sheet.status and "needs access" not in sheet.status
    rebuilt = next(r for r in sheet.requirements if r.label == "ddmarlinpandora")
    assert rebuilt.facts["rebuildable"] is True and rebuilt.ran is not None
    assert rebuilt.ran["original"] == "fingerprint-only"  # never the binary


def test_a_published_file_that_changed_is_refused(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    _, results = made
    release = offer(results).release(inputs=True)
    entry = publish(results, tmp_path / "pub", release=release)
    publication = open_publication(tmp_path / "pub")
    sheet = publication.software(entry)
    assert sheet is not None
    artifact = next(a for a in sheet.artifacts if a.name == "ecal_scale.json")
    (tmp_path / "pub" / artifact.path).write_bytes(b'{"scale": 9.99}\n')
    with pytest.raises(PublicationError, match="not the file"):
        publication.artifact(artifact)


def test_publishing_again_adds_a_check_and_keeps_the_first_and_no_new_snapshot(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software import readiness

    _, results = made
    monkeypatch.setattr(readiness, "_now", lambda: "2026-11-01T10:00:00+00:00")
    first = publish(results, tmp_path / "pub")
    monkeypatch.setattr(readiness, "_now", lambda: "2026-11-20T10:00:00+00:00")
    later = publish(results, tmp_path / "pub")
    assert later.snapshot == first.snapshot and later.published == first.published
    assert later.software != first.software
    sheet = open_publication(tmp_path / "pub").software(later)
    assert sheet is not None
    assert sheet.first_checked.startswith("2026-11-01") and sheet.checked.startswith("2026-11-20")


def test_a_sheet_of_another_study_or_state_is_refused(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    from k4opt.publication import SoftwareSheet

    _, results = made
    entry = publish(results, tmp_path / "pub")
    assert entry.software is not None
    text = (tmp_path / "pub" / entry.software.path).read_bytes()
    with pytest.raises(PublicationError, match="not the sheet"):
        SoftwareSheet.from_json(text, entry.id, "0" * 64, "here")


def test_an_artifact_path_that_leads_elsewhere_is_refused(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    from k4opt.publication import SoftwareSheet

    _, results = made
    release = offer(results).release(inputs=True)
    entry = publish(results, tmp_path / "pub", release=release)
    assert entry.software is not None
    sheet = json.loads((tmp_path / "pub" / entry.software.path).read_text())
    sheet["artifacts"][0]["path"] = "../../../etc/passwd"
    with pytest.raises(PublicationError, match="not the path"):
        SoftwareSheet.from_json(json.dumps(sheet), entry.id, entry.snapshot.sha256, "here")


# --- the command --------------------------------------------------------------------------------


def run(
    study: Path, pub: Path, answers: str | None, *more: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, str, str]:
    stream = Terminal(answers) if answers is not None else io.StringIO("")
    monkeypatch.setattr(sys, "stdin", stream)
    out, err = io.StringIO(), io.StringIO()
    code = main(["publish", str(study), "--output", str(pub), *more], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def artifacts_of(pub: Path) -> list[str]:
    catalog = Catalog.from_json((pub / CATALOG_NAME).read_bytes())
    sheet = open_publication(pub).software(catalog.studies[0])
    assert sheet is not None
    return sorted(a.name for a in sheet.artifacts if a.owner in OURS or a.group == "inputs")


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ("y\ny\n", ["ddmarlinpandora.patch", "ecal_scale.json", "ecal_tweak.py"]),
        ("y\nn\n", ["ddmarlinpandora.patch", "ecal_tweak.py"]),
        ("n\ny\n", ["ecal_scale.json"]),
        ("\n\n", []),  # the default is no
        ("", []),  # no answer at all: no
    ],
)
def test_the_two_questions_decide_separately(
    made: tuple[Path, StudyResults],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answers: str,
    expected: list[str],
) -> None:
    root, _ = made
    code, out, _ = run(root / "study", tmp_path / "pub", answers, monkeypatch=monkeypatch)
    assert code == 0
    assert artifacts_of(tmp_path / "pub") == expected
    assert "Publish the code above? [y/N]" in out
    assert f"Anyone who can read\n{tmp_path / 'pub'} can read them." in out


def test_the_listing_says_what_goes_and_what_does_not(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = made
    _, out, _ = run(root / "study", tmp_path / "pub", "n\nn\n", monkeypatch=monkeypatch)
    assert "patch against" in out and "src/DDCaloHitCreator.cc" in out and "+1" in out
    assert "not published" in out and "ddmarlinpandora/notes.txt" in out and "(untracked" in out
    assert f"read by {STAGE}" in out
    assert "scan" in out and "nothing found" in out
    assert out.count("published ") >= 1 and "(snapshot " in out
    assert str(root) not in out.split("to  ")[0]


def test_without_a_terminal_it_refuses_rather_than_decide(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = made
    code, _, err = run(root / "study", tmp_path / "pub", None, monkeypatch=monkeypatch)
    assert code == 1 and "--yes" in err
    assert not (tmp_path / "pub" / CATALOG_NAME).exists()


def test_yes_answers_only_for_the_code_and_a_script_names_its_inputs(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = made
    code, _, _ = run(root / "study", tmp_path / "a", None, "--yes", monkeypatch=monkeypatch)
    assert code == 0 and artifacts_of(tmp_path / "a") == ["ddmarlinpandora.patch", "ecal_tweak.py"]
    code, _, _ = run(
        root / "study",
        tmp_path / "b",
        None,
        "--yes",
        "--include-inputs",
        "ecal_scale.json",
        monkeypatch=monkeypatch,
    )
    assert code == 0 and "ecal_scale.json" in artifacts_of(tmp_path / "b")


def test_a_secret_stops_the_command_until_the_file_is_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    leaked = tmp_path / "token.txt"
    leaked.write_text(f"\n{SECRET}\n")
    add_input(study, leaked)
    pub = tmp_path / "pub"
    code, out, _ = run(study, pub, "y\ny\n", monkeypatch=monkeypatch)
    assert code == 1 and "blocked:" in out and "token.txt:2" in out
    assert not (pub / CATALOG_NAME).exists()  # not a question: nothing is published
    code, out, _ = run(study, pub, "n\nn\n", "--exclude", "token.txt", monkeypatch=monkeypatch)
    assert code == 0 and "blocked:" not in out
    catalog = Catalog.from_json((pub / CATALOG_NAME).read_bytes())
    sheet = open_publication(pub).software(catalog.studies[0])
    assert sheet is not None and sheet.on_request == ("token.txt",)


def test_a_build_nothing_read_is_not_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.checkouts import build, checkout, variables
    from tests.support.studies import mini_environment, mixed_study

    mini_environment(monkeypatch)
    repo = checkout(tmp_path / "src")
    prefix = build(repo)
    (repo / "src" / "DDCaloHitCreator.cc").write_text("energy = 2 * hit;\n")
    for name, value in variables(prefix).items():
        monkeypatch.setenv(name, value)
    results = load_study(mixed_study(tmp_path / "study", 1))
    found = offer(results)
    assert not [p for p in found.code if p.owner == "ddmarlinpandora"]  # it did not run


def test_a_binary_a_stage_read_is_held_and_named_as_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    read_by_stage(study, tmp_path / "src" / "DDMarlinPandora" / "blob.dat")
    results = load_study(study)
    found = offer(results)
    (held,) = [h for h in found.held if h.name == "blob.dat"]
    assert (held.why, held.needed) == ("binary", True)
    entry = publish(results, tmp_path / "pub", release=found.release(code=True, inputs=True))
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None  # the rest of its changes are published; a reader still has to ask
    assert {r.label: r.access for r in sheet.requirements}["ddmarlinpandora"] == "on request"


def test_a_local_geometry_and_the_evaluators_are_code_never_inputs(
    made: tuple[Path, StudyResults],
) -> None:
    found = offer(made[1])
    assert any(piece.owner == "geometry" for piece in found.code)
    assert any(piece.owner == "test.trade_off" for piece in found.code)
    assert all(piece.owner == "" for piece in found.inputs)


def test_read_helper_counts_only_when_a_stage_read_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    read_by_stage(study, tmp_path / "helpers" / "unrelated.py")
    assert "helpers/unrelated.py" in names(offer(load_study(study)).code)
