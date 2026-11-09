"""What of a study may leave the machine: only what is known to have been built or used, only what
its author approved, nothing a secret scan stopped, and in public nothing but the public form."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
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
    """One study with local work, for the tests that only read it; what it was run with is undone
    at once, so that no later test starts in front of this study's local build."""
    from tests.support.studies import mini_environment

    monkeypatch = pytest.MonkeyPatch()
    mini_environment(monkeypatch)
    root = tmp_path_factory.mktemp("local")
    try:
        study = local_study(root, monkeypatch)
        return root, load_study(study)
    finally:
        monkeypatch.undo()


def idle_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory of code on ``PYTHONPATH`` that no stage reads, for a study made after it."""
    import os

    idle = tmp_path / "idle"
    idle.mkdir()
    (idle / "notes.py").write_text("IDEA = 1\n")
    monkeypatch.setenv("PYTHONPATH", f"{idle}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"))
    return idle


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


def test_a_build_and_a_directory_a_stage_read_are_offered_whole(
    made: tuple[Path, StudyResults],
) -> None:
    found = offer(made[1])
    assert names(found.code) == [
        "ddmarlinpandora/cmake-options.txt",  # how it was configured, without its paths
        "ddmarlinpandora/ddmarlinpandora.patch",  # its whole state, untracked files and all
        "helpers/ecal_tweak.py",  # a stage read from this directory: all of it
        "helpers/unrelated.py",
    ]
    (patch,) = [p for p in found.code if p.is_patch]
    counts = {name: (added, removed) for name, added, removed in patch.files}
    assert counts["src/DDCaloHitCreator.cc"] == (1, 1)
    assert set(counts) == {"New.h", "notes.txt", "blob.dat", "src/DDCaloHitCreator.cc"}
    assert counts["blob.dat"] == (None, None)  # binary, and part of the state like the rest
    assert names(found.inputs) == ["ecal_scale.json"]
    assert not [h for h in found.held if h.owner in OURS] and not found.findings


def test_a_directory_no_stage_read_is_held_whole_and_offered_once_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idle = idle_directory(tmp_path, monkeypatch)
    results = load_study(fresh(tmp_path, monkeypatch))
    (held,) = [h for h in offer(results).held if h.owner == idle.name]
    assert (held.why, held.needed) == ("not read by any stage", False)
    assert "idle/notes.py" in [p.label for p in offer(results, include=["notes.py"]).code]
    with pytest.raises(PublicationError, match=r"named nosuch\.txt"):
        offer(results, exclude=["nosuch.txt"])  # a slip must not publish what was meant to stay


def test_a_file_left_out_by_name_leaves_out_all_of_its_unit(
    made: tuple[Path, StudyResults],
) -> None:
    found = offer(made[1], exclude=["unrelated.py", "ecal_scale.json", "notes.txt"])
    assert not [p for p in found.code if p.owner in ("helpers", "ddmarlinpandora")]
    assert not found.inputs
    assert {p.owner or p.name for p in found.excluded} == {
        "helpers",
        "ddmarlinpandora",
        "ecal_scale.json",
    }


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


@pytest.mark.parametrize(
    ("name", "found"),
    [
        (".env", True),
        (".env.local", True),
        (".env.production", True),
        ("deploy/.env.prod", True),
        (".envrc", True),  # direnv's
        (".env.example", False),  # a template, meant to be shared: its content is scanned
        ("deploy/.env.sample", False),
        (".env.template", False),
        ("env.py", False),
    ],
)
def test_an_environment_file_holds_credentials_by_its_name_and_a_template_does_not(
    name: str, found: bool
) -> None:
    named = "a file of credentials, by its name"
    plain = scan([Piece("code", "o", name, b"A=1\n", (), ())])
    patch = f"diff --git a/{name} b/{name}\n@@ -0,0 +1 @@\n+A=1\n".encode()
    patched = scan([Piece("code", "o", "checkout.patch", patch, (), (), base="abc")])
    for findings in (plain, patched):
        assert (named in [finding.what for finding in findings]) is found
    if found:  # a file a patch changes is named as such
        assert [finding.label for finding in patched] == [f"o/{name}"]


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
    assert patch.kind == "patch" and patch.requirements[0].startswith("package:ddmarlinpandora:")
    assert {file.name for file in patch.files} == {
        "New.h",
        "notes.txt",
        "blob.dat",
        "src/DDCaloHitCreator.cc",
    }
    assert b"1.05 * eCalToEMGeV" in publication.artifact(patch)
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


CODE = ["ddmarlinpandora.patch"]
HELPERS = ["ecal_tweak.py", "unrelated.py"]


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ("y\ny\n", ["cmake-options.txt", *CODE, "ecal_scale.json", *HELPERS]),
        ("y\nn\n", ["cmake-options.txt", *CODE, *HELPERS]),
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
    listing = out.split("Publish the code above?")[0]
    assert "patch against" in listing and "4 files" in listing
    for name in ("src/DDCaloHitCreator.cc", "notes.txt", "blob.dat", "New.h"):
        assert name in listing  # every file a unit holds is shown before the question
    assert f"read by {STAGE}" in listing
    assert "scan" in listing and "nothing found" in listing
    assert "(snapshot " in out
    assert str(root) not in listing.split("to  ")[0]


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
    assert code == 0
    assert artifacts_of(tmp_path / "a") == ["cmake-options.txt", *CODE, *HELPERS]
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


@pytest.mark.parametrize("cached", ["unrelated.cpython-314.pyc", "unrelated.cpython-314.opt-1.pyc"])
def test_a_module_python_read_from_its_cache_was_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: str
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    cache = tmp_path / "helpers" / "__pycache__"
    cache.mkdir()
    (cache / cached).write_bytes(b"compiled")  # with a valid cache, Python never opens the .py
    read_by_stage(study, cache / cached)
    found = offer(load_study(study))
    assert "helpers/unrelated.py" in names(found.code)
    assert not any(piece.name.endswith(".pyc") for piece in found.code)


def test_what_no_stage_was_seen_to_read_is_only_unused_if_every_stage_was_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.publication import artifacts

    idle = idle_directory(tmp_path, monkeypatch)
    results = load_study(fresh(tmp_path, monkeypatch))
    assert idle.name in [h.owner for h in offer(results).held]
    real = artifacts.requirements

    def unobserved(*args: Any, **kwargs: Any) -> Any:
        gap = ("pi0/reconstruction was not observed completely",)
        return replace(real(*args, **kwargs), gaps=gap, public_gaps=gap)

    monkeypatch.setattr(artifacts, "requirements", unobserved)
    found = offer(results)
    assert idle.name not in [h.owner for h in found.held]  # not known: offered, not judged unused
    assert f"{idle.name}/notes.py" in [p.label for p in found.code]


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


# --- what the first review of the publication found ----------------------------------------------


def two_of_a_name(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StudyResults, Path, Path]:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(root, monkeypatch)
    public, private = root / "public" / "config.json", root / "private" / "config.json"
    for path, text in ((public, "public\n"), (private, "private\n")):
        path.parent.mkdir()
        path.write_text(text)
        add_input(study, path)
    return load_study(study), public, private


def test_a_name_two_inputs_share_is_refused_and_each_can_be_named_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, public, private = two_of_a_name(tmp_path, monkeypatch)
    found = offer(results)
    with pytest.raises(PublicationError, match=r"config\.json names 2 files") as refused:
        found.release(inputs=["config.json"])
    assert str(public) in str(refused.value) and str(private) in str(refused.value)
    (one,) = found.release(inputs=[str(public)]).pieces
    assert one.data == b"public\n"
    (other,) = [p for p in found.inputs if p.place == str(private)]
    (by_id,) = found.release(inputs=[other.requirements[0]]).pieces
    assert by_id.data == b"private\n"


def test_the_command_refuses_a_shared_name_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, _, _ = two_of_a_name(tmp_path, monkeypatch)
    assert results.directory is not None
    code, _, err = run(
        results.directory,
        tmp_path / "pub",
        None,
        "--yes",
        "--include-inputs",
        "config.json",
        monkeypatch=monkeypatch,
    )
    assert code == 1 and "names 2 files" in err and not (tmp_path / "pub").exists()


def test_an_input_held_back_that_a_stage_read_is_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    (home / ".config" / "calibration.json").write_text("{}\n")
    (home / ".rootrc").write_text("Unix.*.Root.UseTTFonts: false\n")
    for path in (home / ".config" / "calibration.json", home / ".rootrc"):
        add_input(study, path)
    monkeypatch.setenv("HOME", str(home))
    results = load_study(study)
    found = offer(results)
    assert {"calibration.json", ".rootrc"} <= {p.name for p in found.inputs}
    entry = publish(results, tmp_path / "pub", release=found.release(code=True))
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None
    by = {r.label: r for r in sheet.requirements}
    assert by["calibration.json"].access == "on request" and "needs access" in sheet.status
    assert by[".rootrc"].access == "on request" and not by[".rootrc"].unused


def steering(tmp_path: Path, **changes: Any) -> tuple[Any, Any]:
    """The pieces of a job directory with a changed file, a removed one and ``changes``."""
    from dataclasses import replace

    from k4opt.publication import artifacts
    from k4opt.software.keep import Store
    from k4opt.software.readiness import Requirement

    store = Store(tmp_path)
    kept = store.put(b"a = 1\n")
    file = {"kept": kept, "sha256": kept, "executable": True, "link": None}
    entry = {
        "package": "cldconfig",
        "revision": "8879e4e",
        "original": None,
        "changed": {"a.py": file},
        "added": {"b.py": {**file, "executable": False, "link": "a.py"}},
        "removed": ["old/gone.py"],
    } | changes
    requirement = Requirement(
        "steering:pi0/reco", "steering", "pi0/reco", "a copy", ("pi0/reco",), entry=entry
    )
    pieces: list[Piece] = []
    artifacts._from_steering(replace(requirement), store, pieces, [])
    return pieces, entry


def test_what_a_copy_of_job_files_removed_links_and_marks_executable_is_published(
    tmp_path: Path,
) -> None:
    pieces, _ = steering(tmp_path)
    manifest = json.loads(next(p for p in pieces if p.name == "job-files.json").data)
    assert manifest["removed"] == ["old/gone.py"]
    assert manifest["executable"] == {"a.py": True, "b.py": False}
    assert manifest["links"] == {"b.py": "a.py"}


def test_a_link_out_to_a_local_path_is_not_published(tmp_path: Path) -> None:
    file = {"kept": None, "sha256": None, "executable": False, "link": "/home/me/secret.py"}
    pieces, _ = steering(tmp_path, added={"c.py": file})
    manifest = json.loads(next(p for p in pieces if p.name == "job-files.json").data)
    assert "c.py" not in manifest["links"] and b"/home/me" not in manifest.__repr__().encode()


def test_equal_bytes_elsewhere_do_not_make_a_directory_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idle = idle_directory(tmp_path, monkeypatch)
    study = fresh(tmp_path, monkeypatch)
    twin = tmp_path / "elsewhere.py"
    twin.write_bytes((idle / "notes.py").read_bytes())
    read_by_stage(study, twin)  # the same bytes as idle/notes.py, read from another place
    assert idle.name in [h.owner for h in offer(load_study(study)).held]


def test_a_builds_options_are_published_without_its_paths(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    root, results = made
    found = offer(results)
    options = next(p for p in found.code if p.label == "ddmarlinpandora/cmake-options.txt")
    assert b"CMAKE_BUILD_TYPE=RelWithDebInfo" in options.data
    assert str(root).encode() not in options.data
    entry = publish(results, tmp_path / "a", release=found.release(code=True))
    sheet = open_publication(tmp_path / "a").software(entry)
    assert sheet is not None
    by = {r.label: r for r in sheet.requirements}
    assert by["ddmarlinpandora"].facts["configuration"] == "published"
    entry = publish(results, tmp_path / "b")
    sheet = open_publication(tmp_path / "b").software(entry)
    assert sheet is not None
    by = {r.label: r for r in sheet.requirements}
    assert by["ddmarlinpandora"].facts["configuration"] == "on request"


def test_a_release_of_another_study_or_state_is_refused_before_anything_is_written(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from tests.support.studies import mini_environment, mixed_study

    _, results = made
    mini_environment(monkeypatch)
    other = load_study(mixed_study(tmp_path / "other", 1))
    with pytest.raises(PublicationError, match="made for another study"):
        publish(other, tmp_path / "pub", release=offer(results).release(code=True))
    assert not (tmp_path / "pub").exists()
    moved = replace(results, pareto=())  # the same study, in another state than was reviewed
    with pytest.raises(PublicationError, match="another state"):
        publish(moved, tmp_path / "pub", release=offer(results).release(code=True))
    assert not (tmp_path / "pub").exists()


def test_external_geometry_files_of_one_name_stay_apart(tmp_path: Path) -> None:
    from k4opt.publication import artifacts
    from k4opt.software.keep import Store
    from k4opt.software.readiness import Requirement

    store = Store(tmp_path)
    first, second = store.put(b"one\n"), store.put(b"two\n")
    files = {
        "/a/constants.xml": {"sha256": first, "kept": first},
        "/b/constants.xml": {"sha256": second, "kept": second},
        "compact/det.xml": {"sha256": first, "kept": first},
    }
    pieces: list[Piece] = []
    requirement = Requirement("geometry", "geometry", "c", "3 files, kept")
    artifacts._from_files(requirement, store, "geometry", files, "/geo", pieces, [])
    assert sorted(p.name for p in pieces) == sorted(
        [
            "compact/det.xml",
            f"externals/{first[:16]}/constants.xml",
            f"externals/{second[:16]}/constants.xml",
        ]
    )


def test_two_builds_of_one_name_are_two_patches() -> None:
    from k4opt.publication.artifacts import bundled

    pieces = [
        Piece("code", "pkg", "a.cc", f"diff {n}\n".encode(), (f"package:{n}",), (), base="abc")
        for n in (1, 2)
    ]
    study = "12345678-1234-4234-8234-123456789012"
    assert sorted(data for _, data in bundled(study, pieces)) == [b"diff 1\n", b"diff 2\n"]


# --- what the second review of the publication found ---------------------------------------------


def job_directory(
    tmp_path: Path, links: Sequence[tuple[str, str]] = ()
) -> tuple[list[Piece], list[Any], Path]:
    """The pieces of a real job directory, as a session keeps it: a changed file, a link within it
    and a link to a private file outside it."""
    import hashlib

    from k4opt.publication import artifacts
    from k4opt.software.keep import Store, _steering
    from k4opt.software.readiness import Requirement

    job, private = tmp_path / "job", tmp_path / "private" / "calibration.json"
    private.parent.mkdir()
    job.mkdir()
    private.write_text('{"scale": 1.10}\n')
    (job / "Pandora.py").write_text("x = 1\n")
    (job / "inner.py").symlink_to("Pandora.py")
    (job / "calibration.json").symlink_to(private)
    for name, target in links:
        (job / name).symlink_to(target)

    def digest(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:  # a broken link or a cycle
            return "0" * 64

    inputs = {
        f"samples.pi0.reco.directory/{path.name}": {"path": str(path), "sha256": digest(path)}
        for path in job.iterdir()
    }
    store = Store(tmp_path / "kept")
    (entry,) = _steering(store, [], {"inputs": inputs}, None)
    requirement = Requirement("steering:pi0/reco", "steering", "pi0/reco", "x", entry=entry)
    pieces: list[Piece] = []
    held: list[Any] = []
    artifacts._from_steering(requirement, store, pieces, held)
    _AGAIN[str(tmp_path)] = (requirement, store)
    return pieces, held, private


_AGAIN: dict[str, Any] = {}


def job_directory_again(tmp_path: Path) -> tuple[list[Piece], list[Any], None]:
    """The pieces of the job directory :func:`job_directory` made, gathered again from what is
    kept now."""
    from k4opt.publication import artifacts

    requirement, store = _AGAIN[str(tmp_path)]
    pieces: list[Piece] = []
    lost: list[Any] = []
    artifacts._from_steering(requirement, store, pieces, lost)
    return pieces, lost, None


def test_a_link_out_of_a_job_directory_is_data_and_not_code(tmp_path: Path) -> None:
    pieces, _, private = job_directory(tmp_path)
    assert not any(b"1.10" in p.data for p in pieces if p.group == "code")
    (out,) = [p for p in pieces if p.group == "inputs"]
    assert (out.name, out.place, out.unit) == (
        "calibration.json",
        str(private),
        "link:pi0/reco/calibration.json",
    )
    assert "steering:pi0/reco" in out.requirements  # withheld, its job directory is on request
    assert {"Pandora.py", "inner.py"} <= {p.name for p in pieces}


def test_a_credential_in_a_cmake_option_is_never_written(tmp_path: Path) -> None:
    from k4opt.publication.artifacts import _options

    text = _options(
        b"CMAKE_BUILD_TYPE:STRING=Release\n"
        b"MY_API_KEY:STRING=secret-value-123456\n"
        b"DB_PASSWORD:STRING=hunter2hunter2\n"
        b"GITLAB_TOKEN:STRING=abc\n"
        b"ENABLE_X:BOOL=ON\n"
    )
    assert b"Release" in text and b"ENABLE_X=ON" in text
    assert b"123456" not in text and b"hunter2" not in text and b"abc" not in text
    assert b"# 3 more options" in text


@pytest.mark.parametrize(
    "text",
    [
        "MY_API_KEY=secret-value-123456",
        "export DB_PASSWORD=hunter2hunter2",
        "auth_token: abc12345678",
    ],
)
def test_the_scan_sees_unquoted_and_prefixed_secrets(text: str) -> None:
    assert scan([Piece("code", "o", "x", text.encode(), (), ())])


@pytest.mark.parametrize(
    "text",
    [
        "password = getpass()",
        "secret = load_the_secret_now",
        "api_key: ${API_KEY}",
    ],
)
def test_the_scan_leaves_a_secret_that_is_only_referred_to_alone(text: str) -> None:
    assert not scan([Piece("code", "o", "x", text.encode(), (), ())])


def test_include_inputs_opts_in_no_code_and_include_no_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    data = tmp_path / "data" / "notes.txt"
    data.write_text("an input\n")
    add_input(study, data)
    results = load_study(study)
    found = offer(results, include_inputs=["notes.txt"])
    assert "ddmarlinpandora/notes.txt" not in names(found.code)  # untracked, not used: still held
    with pytest.raises(PublicationError, match="nothing of this study is named"):
        offer(results, include=["ecal_scale.json"])  # an input is approved as an input


def test_equal_inputs_at_two_places_stay_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    first, second = tmp_path / "one" / "config.json", tmp_path / "two" / "config.json"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text('{"scale": 1}\n')
        add_input(study, path)
    found = offer(load_study(study))
    same = [p for p in found.inputs if p.name == "config.json"]
    assert sorted(str(p.place) for p in same) == sorted([str(first), str(second)])
    (one,) = found.release(inputs=[str(first)]).pieces
    assert one.place == str(first) and len(one.requirements) == 1


def test_a_directory_of_code_in_a_checkout_is_held_with_its_whole_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software import keep, source
    from tests.support.checkouts import checkout
    from tests.support.studies import mini_environment, mixed_study

    mini_environment(monkeypatch)
    clean = checkout(tmp_path / "k4opt", "k4Opt")
    monkeypatch.setattr(
        keep,
        "checkout",
        lambda path, release, excluded=(): source.checkout(str(clean), release, excluded),
    )
    repo = checkout(tmp_path / "src", "shadows")
    (repo / "README.md").unlink()  # a deletion: part of its state like an edit
    (repo / "src" / "DDCaloHitCreator.cc").write_text("edited, and read by nothing\n")
    monkeypatch.setenv("PYTHONPATH", str(repo))
    results = load_study(mixed_study(tmp_path / "study", 1))
    (held,) = [h for h in offer(results).held if h.owner == "shadows"]
    (patch,) = held.pieces
    assert {name for name, _, _ in patch.files} == {"README.md", "src/DDCaloHitCreator.cc"}
    assert [p.name for p in offer(results, include=["shadows"]).code if p.owner == "shadows"] == [
        "shadows.patch"
    ]


def test_a_home_configuration_is_needed_and_only_a_venvs_marker_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".rootrc").write_text("Unix.*.Root.UseTTFonts: false\n")
    (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
    add_input(study, home / ".rootrc")
    add_input(study, tmp_path / "pyvenv.cfg")
    monkeypatch.setenv("HOME", str(home))
    found = offer(load_study(study))
    held = {h.name: h for h in found.held if h.group == "inputs"}
    assert ".rootrc" in [p.name for p in found.inputs] and not held["pyvenv.cfg"].needed


@pytest.mark.parametrize("field", ["checked", "first_checked"])
def test_a_time_that_is_no_time_refuses_the_sheet(
    made: tuple[Path, StudyResults], tmp_path: Path, field: str
) -> None:
    from k4opt.publication import SoftwareSheet

    entry = publish(made[1], tmp_path / "pub")
    assert entry.software is not None
    sheet = json.loads((tmp_path / "pub" / entry.software.path).read_text())
    sheet[field] = "yesterday"
    with pytest.raises(PublicationError, match="no time"):
        SoftwareSheet.from_json(json.dumps(sheet), entry.id, entry.snapshot.sha256, "here")


def test_two_installations_of_one_package_leave_no_path_in_the_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from tests.support.checkouts import build, checkout
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    other = build(checkout(tmp_path / "other"), build_id="aabbccdd")
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", f"{other / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    study = local_study(tmp_path, monkeypatch)
    read_by_stage(study, other / "lib" / "libDDMarlinPandora.so.0.14.0")
    results = load_study(study)
    for root, release in (
        (tmp_path / "public", None),
        (tmp_path / "approved", offer(results).release(code=True, inputs=True)),
    ):
        entry = publish(results, root, release=release)
        assert entry.software is not None
        text = (root / entry.software.path).read_text()
        assert str(tmp_path) not in text
        sheet = open_publication(root).software(entry)
        assert sheet is not None
        ids = sorted(r.id for r in sheet.requirements if r.label == "ddmarlinpandora")
        assert len(set(ids)) == 2 and all(i.startswith("package:ddmarlinpandora:") for i in ids)


# --- what the third review of the publication found ----------------------------------------------


@pytest.mark.parametrize("name", ["", "   "])
def test_an_empty_name_selects_nothing_and_is_refused(
    made: tuple[Path, StudyResults], name: str
) -> None:
    for chosen in ({"include": [name]}, {"include_inputs": [name]}, {"exclude": [name]}):
        with pytest.raises(PublicationError, match="empty name"):
            offer(made[1], **chosen)  # type: ignore[arg-type]
    with pytest.raises(PublicationError, match="empty name"):
        offer(made[1]).release(inputs=[name])
    assert not Piece("inputs", "", "a.json", b"x", ("input:1",), ()).matches(name)


def test_a_script_with_an_unset_variable_publishes_no_input(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = made
    arguments = ("--yes", "--include-inputs", "")
    code, _, err = run(root / "study", tmp_path / "pub", None, *arguments, monkeypatch=monkeypatch)
    assert code == 1 and "empty name" in err and not (tmp_path / "pub").exists()


def test_a_chain_of_links_out_of_a_job_directory_is_data(tmp_path: Path) -> None:
    pieces, _, private = job_directory(tmp_path, [("chain.json", "calibration.json")])
    assert not any(b"1.10" in p.data for p in pieces if p.group == "code")
    out = {p.unit: p.place for p in pieces if p.group == "inputs"}
    assert out == {
        "link:pi0/reco/calibration.json": str(private),
        "link:pi0/reco/chain.json": str(private),
    }
    manifest = next(p for p in pieces if p.name == "job-files.json")
    assert b"chain.json" not in manifest.data and b"calibration.json" not in manifest.data


def test_a_link_to_a_directory_outside_leaves_it_too(tmp_path: Path) -> None:
    from k4opt.software.keep import _outside

    (tmp_path / "job").mkdir()
    (tmp_path / "away").mkdir()
    (tmp_path / "away" / "x.py").write_text("x\n")
    (tmp_path / "job" / "sub").symlink_to(tmp_path / "away")
    root = str(tmp_path / "job")
    assert _outside(root, f"{root}/sub/x.py") == str(tmp_path / "away" / "x.py")
    (tmp_path / "job" / "own.py").write_text("x\n")
    assert _outside(root, f"{root}/own.py") is None


def test_broken_links_and_cycles_publish_nothing_and_stop_nothing(tmp_path: Path) -> None:
    links = [("broken", "nowhere"), ("loop-a", "loop-b"), ("loop-b", "loop-a")]
    pieces, lost, _ = job_directory(tmp_path, links)
    assert not {p.name for p in pieces} & {"broken", "loop-a", "loop-b"}
    assert {gone.key[2] for gone in lost} >= {"broken", "loop-a", "loop-b"}  # said, never shown


def test_what_was_published_before_stays_public_in_a_later_sheet(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    _, results = made
    found = offer(results)
    first = publish(results, tmp_path / "pub", release=found.release(code=True, inputs=True))
    said: list[str] = []
    later = publish(results, tmp_path / "pub", warn=said.append)  # approving nothing this time
    assert later.snapshot == first.snapshot
    publication = open_publication(tmp_path / "pub")
    sheet, earlier = publication.software(later), publication.software(first)
    assert sheet is not None and earlier is not None
    assert {a.path for a in sheet.artifacts} == {a.path for a in earlier.artifacts}
    by = {r.label: r.access for r in sheet.requirements}
    assert by["ddmarlinpandora"] == "public" and by["ecal_scale.json"] == "public"
    assert "needs access" not in sheet.status
    assert any("ecal_scale.json" in line and "stays" in line for line in said)
    package = next(r for r in sheet.requirements if r.label == "ddmarlinpandora")
    assert package.facts["configuration"] in ("published", "partial")


def test_an_excluded_file_that_was_published_before_is_not_said_to_be_private(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    _, results = made
    publish(results, tmp_path / "pub", release=offer(results).release(inputs=True))
    found = offer(results, exclude=["ecal_scale.json"])
    entry = publish(results, tmp_path / "pub", release=found.release())
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None and sheet.on_request == ()
    assert {r.label: r.access for r in sheet.requirements}["ecal_scale.json"] == "public"


def test_a_later_publication_trusts_only_an_earlier_sheet_that_is_intact(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software import readiness

    _, results = made
    monkeypatch.setattr(readiness, "_now", lambda: "2026-11-01T10:00:00+00:00")
    first = publish(results, tmp_path / "pub")
    assert first.software is not None
    path = tmp_path / "pub" / first.software.path
    path.write_text(path.read_text().replace("2026-11-01T10", "2020-01-01T10"))  # well formed
    monkeypatch.setattr(readiness, "_now", lambda: "2026-11-20T10:00:00+00:00")
    later = publish(results, tmp_path / "pub")
    sheet = open_publication(tmp_path / "pub").software(later)
    assert sheet is not None and sheet.first_checked.startswith("2026-11-20")


def test_options_left_out_make_a_builds_configuration_partial() -> None:
    from k4opt.publication.artifacts import _options

    whole = _options(b"CMAKE_BUILD_TYPE:STRING=Release\n")
    assert not whole.startswith(b"#")
    part = _options(b"CMAKE_BUILD_TYPE:STRING=Release\nDATA:STRING=/opt/private/geometry\n")
    assert part.startswith(b"# 1 more option")


def test_a_build_with_options_left_out_says_so_in_its_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    study = local_study(tmp_path, monkeypatch)
    results = load_study(study)
    release = offer(results).release(code=True)
    configured = [p for p in release.pieces if p.name == "cmake-options.txt"]
    partial = replace(configured[0], data=b"# 1 more options hold local paths\nA=1\n")
    pieces = tuple(partial if p is configured[0] else p for p in release.pieces)
    entry = publish(results, tmp_path / "pub", release=replace(release, pieces=pieces))
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None
    package = next(r for r in sheet.requirements if r.label == "ddmarlinpandora")
    assert package.facts["configuration"] == "partial"


# --- what the fourth review of the publication found ---------------------------------------------


def fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **changes: Any) -> Path:
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    return local_study(tmp_path, monkeypatch, **changes)


def access_in(entry: Any, root: Path) -> dict[str, str]:
    sheet = open_publication(root).software(entry)
    assert sheet is not None
    return {r.id: r.access for r in sheet.requirements}


def test_inputs_of_equal_bytes_approved_one_after_the_other_both_stay_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = fresh(tmp_path, monkeypatch)
    first, second = tmp_path / "one" / "config.json", tmp_path / "two" / "config.json"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text('{"scale": 1}\n')
        add_input(study, path)
    results = load_study(study)
    found = offer(results)
    ids = {p.place: p.requirements[0] for p in found.inputs if p.name == "config.json"}
    publish(results, tmp_path / "pub", release=found.release(inputs=[str(first)]))
    entry = publish(results, tmp_path / "pub", release=found.release(inputs=[str(second)]))
    access = access_in(entry, tmp_path / "pub")
    assert access[ids[str(first)]] == access[ids[str(second)]] == "public"


def test_a_large_file_that_would_be_published_is_listed_before_the_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = fresh(tmp_path, monkeypatch)
    big = tmp_path / "overlay.root"
    big.write_bytes(b"y" * (BUDGET + 1))
    add_input(study, big)
    code, out, _ = run(
        study, tmp_path / "pub", "n\ny\n", "--include-large", monkeypatch=monkeypatch
    )
    assert code == 0
    listing = out.split("Publish the code above?")[0]
    assert "overlay.root" in listing and "over 10 MB" in listing
    assert "overlay.root" in artifacts_of(tmp_path / "pub")


def test_two_installations_each_say_how_much_of_their_options_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from tests.support.checkouts import build, checkout
    from tests.support.studies import mini_environment

    mini_environment(monkeypatch)
    other = build(checkout(tmp_path / "other"), build_id="aabbccdd")
    cache = other.parent / "build" / "CMakeCache.txt"
    cache.write_text(cache.read_text() + "DATA:STRING=/opt/private/geometry\n")
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", f"{other / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    study = local_study(tmp_path, monkeypatch)
    read_by_stage(study, other / "lib" / "libDDMarlinPandora.so.0.14.0")
    results = load_study(study)
    entry = publish(results, tmp_path / "pub", release=offer(results).release(code=True))
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None
    states = sorted(
        str(r.facts["configuration"]) for r in sheet.requirements if r.label == "ddmarlinpandora"
    )
    assert states == ["partial", "published"]


def test_a_requirement_whose_kept_bytes_are_gone_is_not_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software.keep import DIRECTORY, Store

    study = fresh(tmp_path, monkeypatch)
    directory = study / DIRECTORY
    (state,) = directory.glob("software-*.json")
    files = json.loads(state.read_text())["geometry"]["files"]
    kept = next(file["kept"] for file in files.values())
    Store(directory).path(kept).unlink()
    results = load_study(study)
    entry = publish(results, tmp_path / "pub", release=offer(results).release(code=True))
    assert access_in(entry, tmp_path / "pub")["geometry"] == "none"


def test_what_a_build_installed_and_a_stage_read_is_offered_apart_from_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = fresh(tmp_path, monkeypatch)
    install = tmp_path / "src" / "DDMarlinPandora" / "install" / "include"
    read_by_stage(study, install / "New.h")
    found = offer(load_study(study))
    code = names(found.code)
    assert "ddmarlinpandora/installed/include/New.h" in code
    assert not any("DDCaloHitCreator.h" in name for name in code)  # installed, and read by no one
    assert not any("DDCaloHitCreator.h" in h.label for h in found.held)


@pytest.mark.parametrize(
    ("line", "partial"),
    [
        ("WITH_TRACKING:BOOL=ON", False),
        ("CUSTOM_GEOMETRY:FILEPATH=/home/user/geometry.xml", True),
        ("EXTERNAL_DATA:PATH=/opt/mydata", True),
        ("NOTHING:PATH=", False),
        ("CMAKE_CXX_COMPILER:FILEPATH=/cvmfs/gcc/bin/g++", False),
        ("Gaudi_DIR:PATH=/cvmfs/stack/Gaudi/lib/cmake/Gaudi", False),
        ("GIVEN:UNINITIALIZED=/opt/x", True),
    ],
)
def test_every_option_left_out_that_can_matter_makes_the_options_partial(
    line: str, partial: bool
) -> None:
    from k4opt.publication.artifacts import _options

    assert (
        _options(f"CMAKE_BUILD_TYPE:STRING=Release\n{line}\n".encode()).startswith(b"#") is partial
    )


def test_a_refused_publication_leaves_nothing_in_the_public_directory(
    tmp_path: Path,
) -> None:
    from k4opt.optimization import optimize
    from tests.support.evaluators import trade_off
    from tests.support.studies import mixed_study

    output = mixed_study(tmp_path / "s", trials=4)
    earlier = load_study(output)
    optimize(tmp_path / "s.yaml", [trade_off()], output, trials=2, resume=True)
    publish(load_study(output), tmp_path / "pub")
    before = published_files(tmp_path / "pub")
    release = offer(earlier).release(code=True)
    with pytest.raises(PublicationError, match="later state"):
        publish(earlier, tmp_path / "pub", release=release)
    assert published_files(tmp_path / "pub") == before


def test_the_name_of_a_private_record_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software.keep import DIRECTORY

    study = fresh(tmp_path, monkeypatch)
    (state,) = (study / DIRECTORY).glob("software-*.json")
    state.write_text(state.read_text() + " ")
    results = load_study(study)
    entry = publish(results, tmp_path / "pub")
    assert entry.software is not None
    text = (tmp_path / "pub" / entry.software.path).read_text()
    assert state.name not in text
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None and sheet.altered == ("a kept record of what the study ran on",)


def test_publishing_without_a_release_is_blocked_by_a_secret_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = fresh(tmp_path, monkeypatch)
    leaked = tmp_path / "token.txt"
    leaked.write_text(f"{SECRET}\n")
    add_input(study, leaked)
    with pytest.raises(PublicationError, match="possible secrets"):
        publish(load_study(study), tmp_path / "pub")
    assert not (tmp_path / "pub").exists()


def test_a_study_with_no_kept_software_left_is_published_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software.keep import DIRECTORY

    study = fresh(tmp_path, monkeypatch)
    for path in (study / DIRECTORY).glob("software-*.json"):
        path.unlink()
    results = load_study(study)
    entry = publish(results, tmp_path / "pub")
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None and sheet.requirements == ()
    assert "record incomplete" in sheet.status or "blocked" in sheet.status


def test_a_geometrys_aliases_are_published_and_no_path_with_them(tmp_path: Path) -> None:
    from k4opt.publication.artifacts import _aliases
    from k4opt.software.readiness import Requirement

    aliases = {"part_link.xml": "part.xml", "/home/me/out.xml": "part.xml", "a.xml": "/home/me/b"}
    entry = {"aliases": aliases}
    requirement = Requirement("geometry", "geometry", "c", "1 files, kept", entry=entry)
    piece = _aliases(requirement)
    assert piece is not None and piece.name == "aliases.json"
    assert json.loads(piece.data) == {"part_link.xml": "part.xml"}
    entry["aliases"] = {}
    assert _aliases(requirement) is None


def test_an_empty_cmake_setting_is_kept_since_it_differs_from_none(tmp_path: Path) -> None:
    from k4opt.publication.artifacts import _options

    text = _options(b"ENABLE_CORRECTION:BOOL=\nNOTE:STRING=\n")
    assert b"ENABLE_CORRECTION=\n" in text and b"NOTE=\n" in text and not text.startswith(b"#")


def test_inputs_of_equal_bytes_both_declined_later_each_keep_their_own_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = fresh(tmp_path, monkeypatch)
    for name in ("one", "two"):
        path = tmp_path / name / "config.json"
        path.parent.mkdir()
        path.write_text('{"scale": 1}\n')
        add_input(study, path)
    results = load_study(study)
    publish(results, tmp_path / "pub", release=offer(results).release(inputs=True))
    entry = publish(results, tmp_path / "pub")
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None
    same = [a for a in sheet.artifacts if a.name == "config.json"]
    assert len({a.path for a in same}) == 1  # one file, for two inputs, each its own entry
    assert len({a.requirements[0] for a in same}) == 2


# --- what the sixth review of the publication found ----------------------------------------------


def test_a_publication_that_fails_on_its_sheet_leaves_no_file_behind(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.publication import write

    _, results = made
    release = offer(results).release(code=True, inputs=True)
    monkeypatch.setattr(write, "MAX_SHEET_BYTES", 10)
    with pytest.raises(PublicationError, match="software sheet"):
        publish(results, tmp_path / "pub", release=release)
    assert not [p for p in (tmp_path / "pub").rglob("*") if p.is_file()]


def test_a_publication_that_fails_on_a_file_removes_what_it_wrote(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt import jsonio

    _, results = made
    publish(results, tmp_path / "pub")  # what was there before stays
    before = published_files(tmp_path / "pub")
    release = offer(results).release(code=True, inputs=True)
    real, calls = jsonio.publish, []

    def failing(path: Path, text: str | bytes, private: bool = False) -> None:
        calls.append(path)
        if len(calls) == 3:
            raise OSError("disk full")
        real(path, text, private)

    monkeypatch.setattr(jsonio, "publish", failing)
    with pytest.raises(PublicationError, match="disk full"):
        publish(results, tmp_path / "pub", release=release)
    assert published_files(tmp_path / "pub") == before


@pytest.mark.parametrize(
    ("line", "partial"),
    [
        ("CMAKE_TOOLCHAIN_FILE:FILEPATH=/opt/custom/toolchain.cmake", True),
        ("CMAKE_PREFIX_PATH:PATH=/home/user/custom-stack", True),
        ("Gaudi_DIR:PATH=/home/user/gaudi/lib/cmake/Gaudi", True),
        ("CMAKE_INSTALL_PREFIX:PATH=/home/user/pkg/install", False),
        ("CMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/gmake", False),
        ("CMAKE_CXX_COMPILER:FILEPATH=/cvmfs/sft.cern.ch/gcc/bin/g++", False),
    ],
)
def test_a_users_path_setting_left_out_makes_the_options_partial(line: str, partial: bool) -> None:
    from k4opt.publication.artifacts import _options

    text = _options(f"CMAKE_BUILD_TYPE:STRING=Release\n{line}\n".encode())
    assert text.startswith(b"#") is partial
    assert b"/home/" not in text and b"/opt/" not in text


def test_a_setting_on_the_stack_is_published_as_it_is() -> None:
    from k4opt.publication.artifacts import _options

    text = _options(b"CMAKE_CXX_COMPILER:FILEPATH=/cvmfs/sft.cern.ch/gcc/bin/g++\n")
    assert b"CMAKE_CXX_COMPILER=/cvmfs/sft.cern.ch/gcc/bin/g++" in text


def test_an_earlier_approval_of_one_of_two_equal_inputs_covers_only_that_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = fresh(tmp_path, monkeypatch)
    first, second = tmp_path / "one" / "config.json", tmp_path / "two" / "config.json"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text('{"scale": 1}\n')
        add_input(study, path)
    results = load_study(study)
    found = offer(results)
    ids = {p.place: p.requirements[0] for p in found.inputs if p.name == "config.json"}
    publish(results, tmp_path / "pub", release=found.release(inputs=[str(first)]))
    entry = publish(results, tmp_path / "pub")
    access = access_in(entry, tmp_path / "pub")
    assert access[ids[str(first)]] == "public" and access[ids[str(second)]] == "on request"


def test_a_build_published_with_options_in_part_is_not_all_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dashboard"))
    import tables

    results = load_study(fresh(tmp_path, monkeypatch))
    release = offer(results).release(code=True)
    pieces = tuple(
        replace(p, data=b"# 1 more option holds a local path\nA=1\n")
        if p.name == "cmake-options.txt"
        else p
        for p in release.pieces
    )
    entry = publish(results, tmp_path / "pub", release=replace(release, pieces=pieces))
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None
    assert "ddmarlinpandora build options" in tables.rerun_missing(sheet)


def test_republishing_a_studys_json_form_keeps_its_sheet(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    _, results = made
    first = publish(results, tmp_path / "pub")
    again = publish(StudyResults.from_dict(results.to_dict()), tmp_path / "pub")
    assert again.software == first.software is not None


def test_naming_a_package_of_two_installations_is_refused(tmp_path: Path) -> None:
    from k4opt.publication.artifacts import _named

    pieces = [Piece("code", "pkg", "a.h", b"1", (f"package:pkg:{n}",), ()) for n in (1, 2)]
    with pytest.raises(PublicationError, match="names 2 files"):
        _named(pieces, "pkg")
    assert len(_named(pieces, "package:pkg:2")) == 1
    one = [Piece("code", "pkg", n, b"1", ("package:pkg",), ()) for n in ("a.h", "b.h")]
    assert len(_named(one, "pkg")) == 2


def test_an_incomplete_record_still_says_it_needs_access(
    made: tuple[Path, StudyResults], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.publication import artifacts

    _, results = made
    real = artifacts.requirements

    def unobserved(*args: Any, **kwargs: Any) -> Any:
        gap = ("pi0/reconstruction was not observed completely",)
        return replace(real(*args, **kwargs), gaps=gap, public_gaps=gap)

    monkeypatch.setattr(artifacts, "requirements", unobserved)
    entry = publish(results, tmp_path / "pub")
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None and sheet.status == "Not rerun · record incomplete · needs access"


# --- what the seventh review of the publication found --------------------------------------------


def test_an_evaluator_inside_k4opt_is_offered_as_its_own_unit(tmp_path: Path) -> None:
    import k4opt
    from k4opt.publication import artifacts
    from k4opt.software.keep import Store
    from k4opt.software.readiness import Readiness, Requirement

    store = Store(tmp_path)
    kept = store.put(b"class Mine: ...\n")
    inside = Path(k4opt.__file__).parent / "mine.py"  # an untracked evaluator in k4opt's checkout
    entry = {"path": str(inside), "files": {"mine.py": {"sha256": kept, "kept": kept}}}
    evaluator = Requirement("evaluator:mine", "evaluator", "mine v1", "1 files, kept", entry=entry)
    readiness = Readiness((evaluator,), (), "2026-11-08T00:00:00+00:00", store=store)
    pieces, _, _ = artifacts._gathered(readiness)
    assert [(p.unit, p.name) for p in pieces] == [("evaluator:mine", "mine.py")]
    release = artifacts.Release(artifacts.Offer(readiness, code=tuple(pieces)))
    access = artifacts._access(readiness, release, {})
    assert access["evaluator:mine"] == "on request"  # never "public" for what was not published


def test_a_builds_untracked_files_are_in_its_patch_whatever_they_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names_ = ("weights.json", "config.h.in", "gen.py")  # configure_file, templates, generators
    results = load_study(fresh(tmp_path, monkeypatch, untracked=names_))
    (patch,) = [p for p in offer(results).code if p.is_patch]
    assert set(names_) <= {name for name, _, _ in patch.files}


def test_two_links_of_one_name_in_a_job_are_two_inputs_approved_apart(tmp_path: Path) -> None:
    from k4opt.publication import artifacts
    from k4opt.publication.artifacts import bundled

    first = Piece("inputs", "", "a/c.json", b"1", ("link:s/a/c.json", "steering:s"), ())
    second = Piece("inputs", "", "b/c.json", b"1", ("link:s/b/c.json", "steering:s"), ())
    found = artifacts.Offer(None, inputs=(first, second))
    study = "12345678-1234-4234-8234-123456789012"
    published = bundled(study, found.release(inputs=["a/c.json"]).pieces)
    assert len({a.requirements[0] for a, _ in bundled(study, (first, second))}) == 2
    _, earlier = artifacts.still_public(found.release(), published)
    assert set(earlier) == {first.key}  # the other, of equal bytes, was never approved


def test_republishing_with_a_unit_left_out_keeps_one_patch_that_applies(
    made: tuple[Path, StudyResults], tmp_path: Path
) -> None:
    _, results = made
    publish(results, tmp_path / "pub", release=offer(results).release(code=True))
    found = offer(results, exclude=["src/DDCaloHitCreator.cc"])  # one file of the patch
    entry = publish(results, tmp_path / "pub", release=found.release(code=True))
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None
    patches = [a for a in sheet.artifacts if a.kind == "patch" and a.owner == "ddmarlinpandora"]
    assert len(patches) == 1  # the earlier one, whole: nothing overlaps it
    assert {r.label: r.access for r in sheet.requirements}["ddmarlinpandora"] == "public"


def test_a_published_copy_stands_in_for_a_kept_file_that_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.software.keep import DIRECTORY, Store

    study = fresh(tmp_path, monkeypatch)
    results = load_study(study)
    first = publish(results, tmp_path / "pub", release=offer(results).release(code=True))
    geometry = {
        a.name
        for a in open_publication(tmp_path / "pub").software(first).artifacts  # type: ignore[union-attr]
        if a.owner == "geometry"
    }
    (state,) = (study / DIRECTORY).glob("software-*.json")
    files = json.loads(state.read_text())["geometry"]["files"]
    Store(study / DIRECTORY).path(next(f["kept"] for f in files.values())).unlink()
    entry = publish(load_study(study), tmp_path / "pub", release=offer(results).release(code=True))
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None
    assert {a.name for a in sheet.artifacts if a.owner == "geometry"} == geometry
    assert {r.id: r.access for r in sheet.requirements}["geometry"] == "public"


def test_a_build_is_public_only_with_its_source_and_all_of_its_options(tmp_path: Path) -> None:
    from k4opt.publication import artifacts
    from k4opt.software.readiness import Readiness, Requirement

    def access(rebuildable: bool, options: bytes) -> str:
        package = Requirement(
            "package:p", "package", "p", "x", ("s",), source={"rebuildable": rebuildable}, entry={}
        )
        readiness = Readiness((package,), (), "2026-11-08T00:00:00+00:00")
        piece = Piece(
            "code", "p", artifacts.OPTIONS, options, ("package:p",), ("s",), role="generated"
        )
        release = artifacts.Release(artifacts.Offer(readiness, code=(piece,)), (piece,))
        published = {piece.key: piece.data}
        found = artifacts._access(readiness, release, published)
        return f"{found['package:p']} {artifacts._status(readiness, found)}"

    assert access(False, b"A=1\n").startswith("none")  # options alone rebuild nothing
    assert access(True, b"# 1 more option holds a local path\nA=1\n").startswith("on request")
    assert access(True, b"# 1 more option holds a local path\nA=1\n").endswith("needs access")
    assert access(True, b"A=1\n").startswith("public")


# --- plain studies: their environment and their home configuration -------------------------------


def test_a_python_environment_a_stage_used_is_offered_as_its_list_of_distributions(
    tmp_path: Path,
) -> None:
    from k4opt.publication import artifacts
    from k4opt.software.keep import Store
    from k4opt.software.readiness import Check, Readiness, Requirement

    used = Requirement(
        "environment:site-packages",
        "environment",
        "/home/me/venv/lib/python3.14/site-packages",
        "2 distributions",
        ("pi0/simulation",),
        source={"distributions": ("pyyaml==6.0.2", "lxml==6.1.3")},
        checks=(Check("2026-11-08", "not-checked", "not checked against an index"),),
    )
    idle = replace(used, id="environment:site-packages:2", used_by=())
    readiness = Readiness((used, idle), (), "2026-11-08T00:00:00+00:00", store=Store(tmp_path))
    pieces, held, _ = artifacts._gathered(readiness)
    (listed,) = pieces
    assert (listed.owner, listed.name, listed.data) == (
        "python-environment",
        "requirements.txt",
        b"lxml==6.1.3\npyyaml==6.0.2\n",
    )
    assert [h.requirement for h in held] == ["environment:site-packages:2"]
    found = artifacts.Offer(readiness, code=tuple(pieces), held=tuple(held))
    published = {listed.key: listed.data}
    access = artifacts._access(readiness, found.release(code=True), published)
    assert access["environment:site-packages"] == "public"
    assert artifacts._access(readiness, found.release(), {})["environment:site-packages"] == (
        "on request"
    )
    assert b"/home/me" not in listed.data


def test_home_configuration_is_an_input_and_credentials_stay_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = fresh(tmp_path, monkeypatch)
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".rootrc").write_text("Unix.*.Root.UseTTFonts: false\n")
    (home / ".ssh" / "config").write_text("Host lxplus\n")
    for path in (home / ".rootrc", home / ".ssh" / "config"):
        add_input(study, path)
    monkeypatch.setenv("HOME", str(home))
    found = offer(load_study(study))
    assert ".rootrc" in [p.name for p in found.inputs]
    held = {h.name: h for h in found.held}
    assert (held["config"].why, held["config"].needed) == ("credentials of the account", True)


# --- the last review: what each file is, wherever it appears -------------------------------------


def test_external_geometry_files_of_equal_bytes_are_one_file(tmp_path: Path) -> None:
    from k4opt.publication import artifacts
    from k4opt.software.keep import Store
    from k4opt.software.readiness import Readiness, Requirement

    store = Store(tmp_path)
    kept = store.put(b"<constants/>\n")
    files = {
        "/a/constants.xml": {"sha256": kept, "kept": kept},
        "/b/constants.xml": {"sha256": kept, "kept": kept},
    }
    entry = {"root": "/geo", "files": files, "aliases": {}}
    geometry = Requirement("geometry", "geometry", "c.xml", "2 files, kept", entry=entry)
    readiness = Readiness((geometry,), (), "2026-11-08T00:00:00+00:00", store=store)
    pieces, _, _ = artifacts._gathered(readiness)
    assert [(p.name, p.role) for p in pieces] == [
        (f"externals/{kept[:16]}/constants.xml", "external")
    ]
    study = "12345678-1234-4234-8234-123456789012"
    keys = [(a.requirements[0], a.name) for a, _ in artifacts.bundled(study, pieces)]
    assert len(keys) == len(set(keys))


def test_a_job_file_made_not_executable_says_so(tmp_path: Path) -> None:
    file = {"executable": False, "link": None}
    pieces, _ = steering(tmp_path, changed={"a.py": file}, added={}, removed=[])
    manifest = json.loads(next(p for p in pieces if p.name == "job-files.json").data)
    assert manifest["executable"] == {"a.py": False}  # a permission change alone is a change


def test_an_earlier_file_its_unit_no_longer_holds_is_not_carried(tmp_path: Path) -> None:
    from k4opt.publication import artifacts
    from k4opt.publication.artifacts import bundled

    study = "12345678-1234-4234-8234-123456789012"
    patch = Piece("code", "p", "p.patch", b"diff\n", ("package:p:1",), (), base="abc")
    options = Piece(
        "code", "p", artifacts.OPTIONS, b"A=1\n", ("package:p:1",), (), role="generated"
    )
    published = bundled(study, (patch, options))
    clean = artifacts.Offer(None, code=(options,))  # its checkout is clean now: no patch
    _, earlier = artifacts.still_public(clean.release(), published)
    assert set(earlier) == {options.key}
    gone = artifacts.Lost(patch.key, patch.requirements, base="abc")
    lost = artifacts.Offer(None, code=(options,), lost=(gone,))
    _, earlier = artifacts.still_public(lost.release(), published)
    assert set(earlier) == {options.key, patch.key}  # still needed, its kept copy gone


def test_an_input_in_a_checkout_is_published_only_as_an_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = fresh(tmp_path, monkeypatch, untracked=("calibration.json",))
    repo = tmp_path / "src" / "DDMarlinPandora"
    (repo / "calibration.json").write_text('{"secret_scale": 1.05}\n')
    add_input(study, repo / "calibration.json")
    results = load_study(study)
    found = offer(results)
    (patch,) = [p for p in found.code if p.is_patch]
    assert "calibration.json" not in {name for name, _, _ in patch.files}
    assert b"secret_scale" not in patch.data
    (data,) = [p for p in found.inputs if p.name == "calibration.json"]
    assert patch.unit in data.requirements  # the build answers for it too
    entry = publish(results, tmp_path / "pub", release=found.release(code=True))
    for name in published_files(tmp_path / "pub"):
        assert b"secret_scale" not in (tmp_path / "pub" / name).read_bytes()
    assert access_in(entry, tmp_path / "pub")[patch.unit] == "on request"


def test_leaving_out_a_file_leaves_out_every_unit_that_holds_it() -> None:
    from k4opt.publication.artifacts import _left_out

    evaluator = Piece(
        "code", "mine", "mine.py", b"x", ("evaluator:mine",), (), paths=("/k/mine.py",)
    )
    checkout = Piece(
        "code",
        "k4Opt",
        "k4Opt.patch",
        b"y",
        ("code:k4Opt:1",),
        (),
        base="abc",
        files=(("mine.py", 1, 0),),
        paths=("/k/mine.py", "/k/other.py"),
    )
    alone = Piece("code", "h", "h.py", b"z", ("code:h:1",), (), paths=("/h/h.py",))
    assert _left_out((evaluator, checkout, alone), ["evaluator:mine"]) == {
        "evaluator:mine",
        "code:k4Opt:1",
    }


def test_a_published_job_link_whose_kept_copy_is_gone_stays_published(tmp_path: Path) -> None:
    from k4opt.publication import artifacts
    from k4opt.publication.artifacts import bundled
    from k4opt.software.keep import Store

    pieces, _, _ = job_directory(tmp_path)
    (link,) = [p for p in pieces if p.group == "inputs"]
    study = "12345678-1234-4234-8234-123456789012"
    published = bundled(study, [link])
    for blob in (tmp_path / "kept").rglob("*.gz"):
        if Store(tmp_path / "kept").get(blob.parent.name + blob.name[:-3]) == link.data:
            blob.unlink()
    again, lost, _ = job_directory_again(tmp_path)
    assert not [p for p in again if p.group == "inputs"]
    assert [gone.key for gone in lost] == [link.key]  # recorded under the identity it had
    offer_ = artifacts.Offer(None, code=tuple(again), lost=tuple(lost))
    _, earlier = artifacts.still_public(offer_.release(), published)
    assert set(earlier) == {link.key}


def test_a_real_file_called_as_a_generated_one_is_another_file(tmp_path: Path) -> None:
    from k4opt.publication.artifacts import bundled

    _, entry = steering(tmp_path)  # a kept file, to call as the generated manifest is
    real = entry["changed"]["a.py"]
    pieces, _ = steering(tmp_path, changed={"job-files.json": real}, added={}, removed=["x.py"])
    assert [p.name for p in pieces].count("job-files.json") == 2  # the real one and the manifest
    study = "12345678-1234-4234-8234-123456789012"
    assert len({p.key for p in pieces}) == len(pieces)
    keys = [(a.requirements[0], a.kind, a.name) for a, _ in bundled(study, pieces)]
    assert len(keys) == len(set(keys))


def test_two_pieces_of_one_identity_and_other_bytes_are_refused(tmp_path: Path) -> None:
    from k4opt.publication.artifacts import _unique

    one = Piece("code", "o", "a.py", b"1", ("code:o:1",), ())
    assert _unique([one, replace(one, stages=("s",))]) == [replace(one, stages=("s",))]
    with pytest.raises(PublicationError, match=r"a\.py"):
        _unique([one, replace(one, data=b"2")])


def test_a_patch_on_another_base_is_another_patch() -> None:
    from k4opt.publication import artifacts
    from k4opt.publication.artifacts import Lost, bundled

    study = "12345678-1234-4234-8234-123456789012"
    old = Piece("code", "p", "p.patch", b"diff\n", ("package:p:1",), (), base="aaaaaaa")
    published = bundled(study, [old])
    now = artifacts.Offer(None, code=(replace(old, base="bbbbbbb"),))
    assert not artifacts.still_public(now.release(), published)[1]
    gone = Lost(old.key, old.requirements, base="bbbbbbb")
    assert not artifacts.still_public(artifacts.Offer(None, lost=(gone,)).release(), published)[1]


def test_an_environment_with_distributions_not_from_an_index_is_on_request(
    tmp_path: Path,
) -> None:
    from k4opt.publication import artifacts
    from k4opt.software.keep import Store
    from k4opt.software.readiness import Readiness, Requirement

    used = Requirement(
        "environment:site-packages:1",
        "environment",
        "/v/lib/python3.14/site-packages",
        "1 distributions, 1 not from an index",
        ("pi0/simulation",),
        source={"distributions": ("lxml==6.1.3",), "unlisted": ("mytool",)},
    )
    readiness = Readiness((used,), (), "2026-11-08T00:00:00+00:00", store=Store(tmp_path))
    pieces, _, _ = artifacts._gathered(readiness)
    found = artifacts.Offer(readiness, code=tuple(pieces))
    published = {p.key: p.data for p in pieces}
    assert artifacts._access(readiness, found.release(code=True), published)[used.id] == (
        "on request"
    )


@pytest.mark.parametrize(
    ("text", "found"),
    [  # put together here, so that no line of this file holds one
        ('{"pass' + 'word": "correct-horse"}', True),  # a quoted name, as JSON writes it
        ("{'api_" + "key': 'abc12345678'}", True),  # and a dict
        ('{"api_key": "${API_KEY}"}', False),  # a variable it is read from
        ('{"password": "~/.secrets/db"}', False),  # a file it is read from
        ("SECRET" + '_KEY = "thisIsAnActual' + 'Secret012345"', True),  # Django's
        ("JWT_SECRET" + "_KEY=thisIsAnActual" + "Secret012345", True),
        ('public_key = "thisIsAnActual' + 'Secret012345"', False),  # whose words name no secret
        ('secret_key = os.environ["SECRET_KEY"]', False),
    ],
)
def test_the_scan_reads_a_name_that_is_quoted_as_one_that_is_not(text: str, found: bool) -> None:
    assert bool(scan([Piece("code", "o", "x", text.encode(), (), ())])) is found
