"""Running ddsim: paths, logs and failure reporting, with a fake process."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from k4opt.ddsim import DdsimError, simulate
from k4opt.ddsim import run as ddsim_run
from k4opt.frozen import freeze
from tests.support import simulation
from tests.unit.ddsim.test_command import GENERATOR


def generator() -> object:
    return freeze(GENERATOR, "generator", DdsimError)


def test_relative_paths_are_made_absolute_before_ddsim_runs_in_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    monkeypatch.chdir(tmp_path)
    Path("compact.xml").write_text("<lccdd/>")

    run = simulate(
        compact=Path("compact.xml"),
        workdir=Path("run"),
        events=1,
        seed=1,
        generator=generator(),  # type: ignore[arg-type]
        args=(),
    )

    ((argv, kwargs),) = fake.calls
    assert f"--compactFile={tmp_path / 'compact.xml'}" in argv
    assert f"--outputFile={tmp_path / 'run' / 'events.edm4hep.root'}" in argv
    assert not any(a.startswith(("--compactFile=compact", "--outputFile=run")) for a in argv)
    assert kwargs["cwd"] == tmp_path / "run"
    assert run.error is None
    assert run.output.is_file()
    assert os.path.isabs(run.output)


def test_a_relative_executable_found_on_path_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = simulation.install(monkeypatch)
    monkeypatch.setattr(ddsim_run.shutil, "which", lambda name: "bin/ddsim")  # PATH=bin:...
    monkeypatch.chdir(tmp_path)
    simulate(
        compact=tmp_path / "c.xml",
        workdir=Path("run"),
        events=1,
        seed=None,
        generator=generator(),  # type: ignore[arg-type]
        args=(),
    )
    ((argv, _),) = fake.calls
    assert argv[0] == str(tmp_path / "bin" / "ddsim")


def test_an_existing_workdir_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = simulation.install(monkeypatch)
    with pytest.raises(DdsimError, match="already exists"):
        simulate(
            compact=tmp_path / "c.xml",
            workdir=tmp_path,
            events=1,
            seed=None,
            generator=generator(),  # type: ignore[arg-type]
            args=(),
        )
    assert fake.calls == []
