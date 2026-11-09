"""CLD's reconstruction (CLDConfig's ``CLDReconstruction.py``) run by k4opt's k4run component
(needs the Key4hep stack).

Skipped without ``$K4GEO``, ``$CLDCONFIG``, ``ddsim``, ``k4run`` and ``podio``. Simulates three
photons once (about a minute, mostly ddsim start-up), then reconstructs them in a few seconds,
and dry-runs the job's arguments in about a second each.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from k4opt.ddsim import simulate
from k4opt.evaluation import count_events
from k4opt.k4run import dry_run, properties, reconstruct
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

EVENTS = 3
GUN = {
    "particle": "gamma",
    "energy": {"value": 10, "unit": "GeV"},
    "direction": {"distribution": "uniform", "theta": [80, 100], "phi": [0, 360], "unit": "deg"},
}


PANDORA = "MyDDMarlinPandora_10ns.Parameters"


@pytest.fixture(scope="module")
def cld() -> tuple[Path, Path]:
    """``(compact, CLDConfig)`` of the stack."""
    k4geo, config = os.environ.get("K4GEO"), os.environ.get("CLDCONFIG")
    if not k4geo or not (Path(k4geo) / CLD_COMPACT).is_file() or not config:
        pytest.skip("$K4GEO with the CLD geometry and $CLDCONFIG are required")
    if not all(shutil.which(p) for p in ("ddsim", "k4run")) or not importlib.util.find_spec(
        "podio"
    ):
        pytest.skip("ddsim, k4run and podio from the Key4hep stack are required")
    return Path(k4geo) / CLD_COMPACT, Path(config) / "share" / "CLDConfig"


@pytest.fixture(scope="module")
def simulated(cld: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    compact, _ = cld
    run = simulate(
        compact=compact,
        workdir=tmp_path_factory.mktemp("sim") / "simulation",
        events=EVENTS,
        seed=42,
        particle_gun=GUN,
        options={},
    )
    assert run.error is None, run.error
    return run.output


def job(cld: tuple[Path, Path], **options: Any) -> dict[str, Any]:
    _, config = cld
    return {
        "files": [config / "CLDReconstruction.py"],
        "directory": config,
        "options": {"options": "CLDReconstruction.py", "directory": str(config), **options},
    }


def test_the_job_takes_its_arguments_and_refuses_one_it_does_not_know(
    cld: tuple[Path, Path], tmp_path: Path
) -> None:
    compact, _ = cld
    placeholder = tmp_path / "placeholder.edm4hep.root"
    common = {"compact": compact, "placeholder": placeholder}
    good = job(cld, geometry="compactFile", cms=91, truthTracking=True)
    assert dry_run(workdir=tmp_path / "a", **good, **common) is None
    assert placeholder.is_file()  # written once, by the first dry run
    said = dry_run(workdir=tmp_path / "b", **job(cld, cms=92), **common)
    assert said is not None and "argument --cms: invalid choice: '92'" in said
    said = dry_run(workdir=tmp_path / "c", **job(cld, truthTraking=True), **common)
    assert said is not None and "unrecognized arguments: --truthTraking" in said


def test_a_key_the_job_does_not_set_is_refused_with_those_it_resembles(
    cld: tuple[Path, Path], tmp_path: Path
) -> None:
    compact, _ = cld
    common = {"compact": compact, "placeholder": tmp_path / "placeholder.edm4hep.root"}
    wiring = {"geometry": "compactFile", "cms": 91, "truthTracking": True}
    good = job(cld, **wiring, **{f"{PANDORA}[ECalMipThreshold]": 2.0})
    assert dry_run(workdir=tmp_path / "a", **good, **common) is None
    bad = job(cld, **wiring, **{f"{PANDORA}[ECalMipTreshold]": 2.0})
    said = dry_run(workdir=tmp_path / "b", **bad, **common)
    assert said is not None and "has no key ECalMipTreshold" in said
    assert "ECalMipThreshold" in said
    bad = job(cld, **wiring, **{"NoSuchAlg.Parameters[x]": 2.0})
    said = dry_run(workdir=tmp_path / "c", **bad, **common)
    assert said is not None and "the job has no component NoSuchAlg" in said
    bad = job(cld, **wiring, **{"MyDDMarlinPandora_10ns.ParameterMap[ECalMipThreshold]": 2.0})
    said = dry_run(workdir=tmp_path / "d", **bad, **common)
    assert said is not None and "has no map property ParameterMap" in said and "Parameters" in said
    bad = job(cld, **wiring, **{"MyDDMarlinPandora_10ns.OutputLevel[x]": 2.0})
    said = dry_run(workdir=tmp_path / "e", **bad, **common)
    assert said is not None and "has no map property OutputLevel" in said


def test_a_key_of_a_map_reaches_the_job_and_leaves_the_others_as_it_set_them(
    cld: tuple[Path, Path], simulated: Path, tmp_path: Path
) -> None:
    compact, _ = cld
    wiring = {"geometry": "compactFile", "cms": 91, "truthTracking": True}
    run = reconstruct(
        compact=compact,
        input_file=simulated,
        workdir=tmp_path / "reconstruction",
        **job(cld, **wiring, **{f"{PANDORA}[ECalMipThreshold]": 2.0}),
    )
    assert run.error is None, run.error
    held = properties(run.files["joboptions"].read_text())[PANDORA]
    assert isinstance(held, dict)
    assert held["ECalMipThreshold"] == ["2.0"]
    assert held["ClusterCollectionName"] == ["PandoraClusters"]


def test_every_simulated_event_is_reconstructed_on_the_geometry_k4opt_gave(
    cld: tuple[Path, Path], simulated: Path, tmp_path: Path
) -> None:
    compact, _ = cld
    run = reconstruct(
        compact=compact,
        input_file=simulated,
        workdir=tmp_path / "reconstruction",
        **job(cld, geometry="compactFile", cms=91, truthTracking=True),
    )
    assert run.error is None, run.error
    assert count_events(run.output) == EVENTS  # though CLDReconstruction.py sets EvtMax = 3 too
    resolved = properties(run.files["joboptions"].read_text())
    assert resolved["GeoSvc.detectors"] == [str(compact)]
    assert resolved["ApplicationMgr.EvtMax"] == -1
    assert (run.cwd / "PandoraSettingsCLD").is_dir()


def test_the_stage_s_seed_reaches_the_random_numbers_of_the_job(
    cld: tuple[Path, Path], simulated: Path, tmp_path: Path
) -> None:
    compact, _ = cld
    options = job(cld, geometry="compactFile", cms=91, truthTracking=True)
    assert dry_run(compact=compact, workdir=tmp_path / "dry", seed=1234, **options) is None
    run = reconstruct(
        compact=compact,
        input_file=simulated,
        workdir=tmp_path / "reconstruction",
        seed=1234,
        **options,
    )
    assert run.error is None, run.error
    resolved = properties(run.files["joboptions"].read_text())
    # whence Marlin's global seed; CLD's job draws nothing through UniqueIDGenSvc
    assert resolved["RndmGenSvc.Engine.Seeds"] == [1234]
    assert resolved["UniqueIDGenSvc.Seed"] == 1234


def test_a_job_that_keeps_its_own_default_geometry_is_caught(
    cld: tuple[Path, Path], simulated: Path, tmp_path: Path
) -> None:
    compact, _ = cld
    # outputBasename is no geometry argument: the job reads its default, CLD_o2_v07
    run = reconstruct(
        compact=compact,
        input_file=simulated,
        workdir=tmp_path / "reconstruction",
        **job(cld, geometry="outputBasename", cms=91, truthTracking=True),
    )
    assert run.error is not None
    assert "CLD_o2_v07.xml" in run.error and "not the candidate's geometry" in run.error
