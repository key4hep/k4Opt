"""Running a k4run job from a copy of its directory, and its dry run (the fake k4run of
``tests/support/fake_stack``, a real process)."""

from __future__ import annotations

import json
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from k4opt.k4run import K4runError, dry_run, job_files, reconstruct
from tests.support.stack import install_k4run


@pytest.fixture
def job(tmp_path: Path) -> Iterator[Path]:
    """A job's directory, read-only as on CVMFS: its options file and a settings file it reads
    relative to where it runs."""
    directory = tmp_path / "CLDConfig"
    (directory / "PandoraSettingsCLD").mkdir(parents=True)
    (directory / "CLDReconstruction.py").write_text("# the job\n")
    (directory / "PandoraSettingsCLD" / "Likelihood.xml").write_text("<data/>\n")
    for path in (directory / "PandoraSettingsCLD", directory):
        for file in path.iterdir():
            if file.is_file():
                file.chmod(stat.S_IRUSR)
        path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    yield directory
    for path in (directory, directory / "PandoraSettingsCLD"):
        path.chmod(stat.S_IRWXU)


@pytest.fixture
def sim(tmp_path: Path) -> Path:
    path = tmp_path / "sim.edm4hep.root"
    path.write_text(json.dumps({"events": [[1], [2], [3]]}))
    return path


def run(job: Path, sim: Path, workdir: Path, **options: Any) -> Any:
    return reconstruct(
        files=[job / "CLDReconstruction.py"],
        directory=job,
        compact=Path("/geo/CLD.xml"),
        input_file=sim,
        workdir=workdir,
        options={"options": "CLDReconstruction.py", "directory": str(job), **options},
    )


def test_the_job_runs_from_a_writable_copy_of_its_directory_and_keeps_its_files(
    monkeypatch: pytest.MonkeyPatch, job: Path, sim: Path, tmp_path: Path
) -> None:
    install_k4run(monkeypatch, write={"histograms.root": "h"})
    workdir = tmp_path / "reconstruction"
    done = run(job, sim, workdir, cms=91)
    assert done.error is None
    assert done.cwd == workdir / "job"
    assert (workdir / "job" / "PandoraSettingsCLD" / "Likelihood.xml").read_text() == "<data/>\n"
    assert (workdir / "job" / "histograms.root").is_file()  # what the job writes stays there
    assert sorted(done.files) == [
        "command",
        "edm4hep",
        "joboptions",
        "options",
        "stderr",
        "stdout",
    ]
    assert done.output.read_text() == sim.read_text()
    assert done.argv[1:3] == ("CLDReconstruction.py", str(workdir / "k4opt.opts.py"))
    assert "--cms=91" in done.argv
    command = json.loads((workdir / "k4run.json").read_text())
    assert command["cwd"] == str(workdir / "job") and command["returncode"] == 0
    assert "DUMPFILE" in (workdir / "k4opt.opts.py").read_text()


def test_a_job_without_a_directory_runs_its_files_by_their_absolute_path(
    monkeypatch: pytest.MonkeyPatch, sim: Path, tmp_path: Path
) -> None:
    install_k4run(monkeypatch)
    options = tmp_path / "job.py"
    options.write_text("# the job\n")
    done = reconstruct(
        files=[options],
        directory=None,
        compact=Path("/geo/CLD.xml"),
        input_file=sim,
        workdir=tmp_path / "reco",
        options={"options": "job.py"},
    )
    assert done.error is None
    assert done.argv[1] == str(options)
    assert list((tmp_path / "reco" / "job").iterdir()) == []


def test_a_job_that_read_another_geometry_fails_saying_what_it_read(
    monkeypatch: pytest.MonkeyPatch, job: Path, sim: Path, tmp_path: Path
) -> None:
    install_k4run(monkeypatch)
    # the job's own --compactFile is not given: its default is what it read
    done = run(job, sim, tmp_path / "reco", geometry="outputBasename")
    assert done.error is not None
    assert "GeoSvc.detectors is ['/default/CLD_o2_v07.xml']" in done.error
    assert "that k4opt passed as --outputBasename" in done.error
    assert run(job, sim, tmp_path / "reco2", geometry="compactFile").error is None


@pytest.mark.parametrize(
    ("told", "message"),
    [
        ({"exit": 3, "stderr": "Pandora: cannot open settings\n"}, "k4run exited with status 3"),
        ({"no_output": True}, "k4run finished but wrote no events.edm4hep.root"),
        ({"no_dump": True}, "k4run finished but its resolved options cannot be read"),
        ({"input": ["/elsewhere.root"]}, "IOSvc.Input is ['/elsewhere.root']"),
    ],
)
def test_a_job_that_fails_says_why(
    monkeypatch: pytest.MonkeyPatch,
    job: Path,
    sim: Path,
    tmp_path: Path,
    told: dict[str, Any],
    message: str,
) -> None:
    install_k4run(monkeypatch, **told)
    done = run(job, sim, tmp_path / "reco")
    assert done.error is not None and message in done.error
    if "stderr" in told:
        assert "Pandora: cannot open settings" in done.error
    assert "stdout" in done.files and "command" in done.files


def test_a_job_that_runs_too_long_is_killed(
    monkeypatch: pytest.MonkeyPatch, job: Path, sim: Path, tmp_path: Path
) -> None:
    install_k4run(monkeypatch, sleep=30)
    done = reconstruct(
        files=[job / "CLDReconstruction.py"],
        directory=job,
        compact=Path("/geo/CLD.xml"),
        input_file=sim,
        workdir=tmp_path / "reco",
        options={"options": "CLDReconstruction.py", "directory": str(job)},
        timeout=0.5,
    )
    assert done.returncode is None
    assert done.error is not None and "k4run ran longer than 0.5 s and was killed" in done.error


def test_an_existing_directory_or_a_bad_timeout_is_refused(
    monkeypatch: pytest.MonkeyPatch, job: Path, sim: Path, tmp_path: Path
) -> None:
    install_k4run(monkeypatch)
    (tmp_path / "reco").mkdir()
    with pytest.raises(K4runError, match="already exists; use a new directory"):
        run(job, sim, tmp_path / "reco")
    with pytest.raises(K4runError, match="timeout must be a positive number"):
        reconstruct(
            files=[job / "CLDReconstruction.py"],
            directory=job,
            compact=Path("/g.xml"),
            input_file=sim,
            workdir=tmp_path / "other",
            options={"options": "CLDReconstruction.py", "directory": str(job)},
            timeout=0,
        )


def test_a_dry_run_says_which_argument_the_job_does_not_take(
    monkeypatch: pytest.MonkeyPatch, job: Path, tmp_path: Path
) -> None:
    options = {"options": "CLDReconstruction.py", "directory": str(job), "cms": 92}
    install_k4run(monkeypatch)
    files = [job / "CLDReconstruction.py"]
    common = {"files": files, "directory": job, "compact": Path("/g.xml"), "options": options}
    assert dry_run(workdir=tmp_path / "a", **common) is None
    install_k4run(monkeypatch, refuse="argument --cms: invalid choice: '92'")
    assert dry_run(workdir=tmp_path / "b", **common) == (
        "the job does not take its arguments: k4run: error: argument --cms: invalid choice: '92'"
    )
    install_k4run(monkeypatch, refuse="unrecognized arguments: --IOSvc.Output=/x")
    said = dry_run(workdir=tmp_path / "c", **common)
    assert said is not None and "the job does not use k4FWCore's IOSvc" in said


def test_dry_runs_that_share_a_placeholder_write_it_once(tmp_path: Path, job: Path) -> None:
    from k4opt.k4run.run import _placeholder_options

    placeholder = tmp_path / "placeholder.edm4hep.root"
    assert "root_io.Writer" in _placeholder_options(placeholder)
    placeholder.write_text("written")
    assert "root_io" not in _placeholder_options(placeholder)


def test_the_files_of_a_job_s_directory_are_listed_by_their_relative_path(job: Path) -> None:
    assert list(job_files(job)) == ["CLDReconstruction.py", "PandoraSettingsCLD/Likelihood.xml"]
