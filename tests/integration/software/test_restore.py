"""The acceptance test of what a study keeps: a candidate restored from nothing but the study's
``software/`` and ``study.json``, rerun elsewhere in a clean environment, measures what it measured
(needs the Key4hep stack: ddsim, k4run, podio, ``$K4GEO``, ``$CLDCONFIG``, and Optuna).

The run changes the reconstruction three ways a physicist does: an edited copy of CLDConfig, whose
Pandora settings import a helper module on ``PYTHONPATH``, which reads a private calibration file
that a variable names. After the study, the originals are removed; the rerun gets the job directory
rebuilt from the stack's ``cldconfig`` and the kept changed file, the helper and the calibration
from the kept bytes, and must measure the same energy to the last digit.

With ``PYTEST_K4OPT_REBUILD=1`` (network, a compiler, about ten minutes), a C++ package too:
DDMarlinPandora cloned at the revision the nightly built, its ECAL energy scaled by 1.05, built and
put in front of the stack (``MARLIN_DLL`` set by hand), then rebuilt elsewhere from the kept patch.
The rebuilt library has other bytes and another build-id; the rerun measures the same, and the
nightly's library measures something else.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.optimization import load_study
from k4opt.publication import offer, open_publication, publish
from k4opt.software.readiness import ALL_AVAILABLE, REBUILD, UNVERIFIED, requirements
from k4opt.software.source import build_id
from k4opt.software.stack import Release
from tests.support.evaluators import PFO_ENERGY
from tests.support.geometry import CLD_COMPACT
from tests.support.restore import restore

pytestmark = pytest.mark.integration

EVENTS = 3
REPOSITORY = "https://github.com/iLCSoft/DDMarlinPandora.git"
LIBRARY = "libDDMarlinPandora.so"


def _stack() -> None:
    k4geo = os.environ.get("K4GEO")
    if not k4geo or not (Path(k4geo) / CLD_COMPACT).is_file() or not os.environ.get("CLDCONFIG"):
        pytest.skip("$K4GEO with the CLD geometry and $CLDCONFIG are required")
    if not all(shutil.which(p) for p in ("ddsim", "k4run", "git")):
        pytest.skip("ddsim, k4run and git are required")
    if not importlib.util.find_spec("podio"):
        pytest.skip("podio from the Key4hep stack is required")
    pytest.importorskip("optuna")


def changed_steering(root: Path) -> tuple[Path, Path, Path]:
    """A copy of CLDConfig whose HCal calibration a helper scales by a private file's number: the
    job directory, the helper's directory and the file."""
    job = shutil.copytree(Path(os.environ["CLDCONFIG"]) / "share" / "CLDConfig", root / "CLDConfig")
    for directory, names, files in os.walk(job):
        for name in names + files:
            os.chmod(os.path.join(directory, name), 0o755 if name in names else 0o644)
    pandora = job / "ParticleFlow" / "Pandora.py"
    text = pandora.read_text()
    text = text.replace(
        "from Configurables import DDPandoraPFANewAlgorithm",
        "from Configurables import DDPandoraPFANewAlgorithm\nfrom ecal_tweak import scale",
        1,
    )
    text = text.replace(
        '"HCalToEMGeVCalibration": 1.01776966108,',
        '"HCalToEMGeVCalibration": 1.01776966108 * scale(),',
        1,
    )
    assert "scale()" in text, "CLDConfig's Pandora settings changed: update the test"
    pandora.write_text(text)
    helpers = root / "myhelpers"
    helpers.mkdir()
    (helpers / "ecal_tweak.py").write_text(
        "import json, os\n\n\ndef scale():\n"
        "    with open(os.environ['MY_ECAL_SCALE_FILE']) as handle:\n"
        "        return json.load(handle)['scale']\n"
    )
    private = root / "private" / "ecal_scale.json"
    private.parent.mkdir()
    private.write_text('{"scale": 1.10}')
    return job, helpers, private


def problem(path: Path, job: Path) -> Path:
    document = {
        "schema_version": "k4opt.problem/v1alpha1",
        "name": "rerun",
        "geometry": str(CLD_COMPACT),
        "parameters": {
            "ecal_cell_size": {
                "type": "continuous",
                "bounds": [5.0, 5.1],
                "unit": "mm",
                "targets": {"geometry": "ECal_cell_size"},
            }
        },
        "samples": {
            "pi0": {
                "generation": {
                    "executor": "gun",
                    "particle": "pi0",
                    "energy": {"value": 50, "unit": "GeV"},
                    "direction": {
                        "distribution": "uniform",
                        "theta": [80, 100],
                        "phi": [0, 360],
                        "unit": "deg",
                    },
                },
                "simulation": {"executor": "ddsim", "events": EVENTS, "seed": 42},
                "reconstruction": {
                    "executor": "k4run",
                    "directory": str(job),
                    "options": "CLDReconstruction.py",
                    "geometry": "compactFile",
                    "cms": 240,
                    "truthTracking": True,
                },
            }
        },
        "evaluators": [{"name": "pfo.energy", "version": "1", "stage": "reconstruction"}],
        "objectives": [
            {
                "evaluator": "pfo.energy",
                "stage": "reconstruction",
                "metric": "pfo.energy",
                "direction": "minimize",
            }
        ],
    }
    path.write_text(yaml.safe_dump(document))
    return path


def measured(summary: dict[str, Any]) -> float:
    (measurement,) = summary["samples"]["pi0"]["evaluations"]["pfo.energy"]["measurements"]
    value: float = measurement["value"]
    return value


def study(problem_file: Path, output: Path) -> dict[str, Any]:
    out, err = io.StringIO(), io.StringIO()
    command = ["optimize", str(problem_file), "--evaluator", PFO_ENERGY, "--trials", "1"]
    command += ["--seed", "1", "--workers", "1", "--output", str(output)]
    assert main(command, stdout=out, stderr=err) == 0, out.getvalue() + err.getvalue()
    summary: dict[str, Any] = json.loads((output / "trial-000000" / "summary.json").read_text())
    return summary


def rerun(
    restored_problem: Path,
    evaluator: Path,
    candidate: dict[str, Any],
    output: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """``k4opt evaluate`` of ``candidate`` in a process of its own, in ``environment`` only."""
    command = [sys.executable, "-m", "k4opt", "evaluate", str(restored_problem)]
    command += ["--evaluator", f"{evaluator}:PfoEnergy", "--output", str(output)]
    command += [
        arg for name, value in candidate.items() for arg in ("--param", f"{name}={value!r}")
    ]
    done = subprocess.run(
        command, env=environment, cwd=output.parent, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    summary: dict[str, Any] = json.loads((output / "summary.json").read_text())
    return summary


def clean(home: Path, **variables: str) -> dict[str, str]:
    """This environment without what pointed at the originals: an empty home, the given variables
    in front."""
    home.mkdir(exist_ok=True)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("MY_ECAL_SCALE_FILE", "PYTHONUSERBASE") and not name.startswith("PYTEST_")
    }
    environment["HOME"] = str(home)
    k4opt = str(Path(__file__).resolve().parents[3])
    path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in [variables.pop("PYTHONPATH", ""), k4opt, path] if p
    )
    return {**environment, **variables}


def test_a_restored_candidate_measures_what_it_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stack()
    originals = tmp_path / "originals"
    originals.mkdir()
    job, helpers, private = changed_steering(originals)
    monkeypatch.setenv("PYTHONPATH", f"{helpers}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.setenv("MY_ECAL_SCALE_FILE", str(private))
    output = tmp_path / "study"
    summary = study(problem(originals / "problem.yaml", job), output)
    readiness = requirements(output, json.loads((output / "study.json").read_text()))
    # nothing gone, nothing unknown, no local build: at most what no one can check (the
    # distributions of an environment)
    assert readiness.status.startswith((ALL_AVAILABLE, UNVERIFIED)), readiness
    kinds = {r.id: r for r in readiness.requirements}
    steering = kinds["steering:pi0/reconstruction"]
    assert "a copy of cldconfig" in steering.identified and "1 changed" in steering.identified
    scale = next(r for r in readiness.requirements if r.label == "ecal_scale.json")
    assert "MY_ECAL_SCALE_FILE" in scale.identified and scale.available
    results = load_study(output)
    found = offer(results)  # from what the stages were really seen to read
    assert not found.findings
    release = found.release(code=True, inputs=["ecal_scale.json"])
    entry = publish(results, tmp_path / "pub", release=release)
    sheet = open_publication(tmp_path / "pub").software(entry)
    assert sheet is not None and entry.software is not None
    publication = open_publication(tmp_path / "pub")
    files = {artifact.name: artifact for artifact in sheet.artifacts}
    assert {"ecal_tweak.py", "ParticleFlow/Pandora.py", "ecal_scale.json"} <= files.keys()
    assert str(originals) not in (tmp_path / "pub" / entry.software.path).read_text()

    monkeypatch.undo()  # the rerun's environment names nothing of the originals
    restored = restore(output, tmp_path / "restored")
    shutil.rmtree(originals)  # the rerun can only read what was kept
    helper = restored.code[str(helpers)]
    # what a reader of the publication has, in place of what the study kept privately
    job = restored.steering["pi0/reconstruction"]
    for name, target in (
        ("ecal_tweak.py", helper / "ecal_tweak.py"),
        ("ParticleFlow/Pandora.py", job / "ParticleFlow" / "Pandora.py"),
        ("ecal_scale.json", restored.inputs[str(private)]),
    ):
        target.write_bytes(publication.artifact(files[name]))
    environment = clean(
        tmp_path / "home",
        PYTHONPATH=str(helper),
        MY_ECAL_SCALE_FILE=str(restored.inputs[str(private)]),
    )
    again = rerun(
        restored.problem,
        restored.evaluators["pfo.energy"],
        summary["candidate"],
        tmp_path / "rerun",
        environment,
    )
    assert measured(again) == measured(summary)


@pytest.mark.skipif(
    os.environ.get("PYTEST_K4OPT_REBUILD") != "1",
    reason="clones and builds DDMarlinPandora: set PYTEST_K4OPT_REBUILD=1",
)
def test_a_local_cpp_build_rebuilt_from_its_kept_patch_measures_what_it_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stack()
    if not shutil.which("cmake"):
        pytest.skip("cmake is required")
    release = Release.of(os.environ.get("KEY4HEP_STACK"))
    package = release.package("DDMarlinPandora") if release else None
    if package is None or package.revision is None:
        pytest.skip("the stack is no LCG nightly with DDMarlinPandora built from its head")
    originals = tmp_path / "originals"
    originals.mkdir()
    checkout = originals / "DDMarlinPandora"
    subprocess.run(["git", "clone", "-q", REPOSITORY, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "-q", package.revision], check=True)
    source = checkout / "src" / "DDCaloHitCreator.cc"
    text, count = re.subn(
        r"m_electromagneticEnergy = eCalToEMGeV \*",
        "m_electromagneticEnergy = 1.05 * eCalToEMGeV *",
        source.read_text(),
        count=1,
    )
    assert count == 1, "DDCaloHitCreator.cc changed: update the test"
    source.write_text(text)
    prefix = _build(checkout)

    output = tmp_path / "study"
    local = _in_front(prefix)
    for name, value in local.items():
        monkeypatch.setenv(name, value)
    job = Path(os.environ["CLDCONFIG"]) / "share" / "CLDConfig"
    summary = study(problem(originals / "problem.yaml", job), output)
    readiness = requirements(output, json.loads((output / "study.json").read_text()))
    assert readiness.status.startswith(REBUILD), readiness
    built = next(r for r in readiness.requirements if r.kind == "package")
    assert built.label == "ddmarlinpandora" and built.used_by == ("pi0/reconstruction",)
    assert built.source is not None and built.source["link"] == "verified"
    assert built.ran is not None and built.ran["original"] == "fingerprint-only"

    monkeypatch.undo()
    restored = restore(output, tmp_path / "restored", remotes={str(prefix): REPOSITORY})
    clone = restored.checkouts[str(prefix)]
    rebuilt = _build(clone)
    original_library = os.path.realpath(prefix / "lib" / LIBRARY)
    rebuilt_library = os.path.realpath(rebuilt / "lib" / LIBRARY)
    assert build_id(original_library) != build_id(rebuilt_library)  # another binary
    shutil.rmtree(originals)
    environment = clean(tmp_path / "home", **_in_front(rebuilt))
    again = rerun(
        restored.problem,
        restored.evaluators["pfo.energy"],
        summary["candidate"],
        tmp_path / "rerun",
        environment,
    )
    assert measured(again) == measured(summary)
    nightly = clean(tmp_path / "home")
    other = rerun(
        restored.problem,
        restored.evaluators["pfo.energy"],
        summary["candidate"],
        tmp_path / "nightly",
        nightly,
    )
    assert measured(other) != measured(summary)  # the change is what the measurement saw


def _build(checkout: Path) -> Path:
    """Build and install ``checkout`` into ``checkout/install``, as a physicist does."""
    tree = checkout / "build"
    tree.mkdir()
    for command in (
        [
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={checkout / 'install'}",
            "-DCMAKE_CXX_STANDARD=20",
            "..",
        ],
        ["cmake", "--build", ".", "-j", str(min(os.cpu_count() or 1, 16))],
        ["cmake", "--install", "."],
    ):
        subprocess.run(command, cwd=tree, check=True, capture_output=True)
    return checkout / "install"


def _in_front(prefix: Path) -> dict[str, str]:
    """What ``k4_local_repo`` puts in front of the stack for the build at ``prefix``, with
    ``MARLIN_DLL`` set to its library by hand (``k4_local_repo`` leaves it on the nightly's)."""
    library = str(prefix / "lib" / LIBRARY)
    marlin = [
        library if entry.endswith(f"/{LIBRARY}") else entry
        for entry in os.environ["MARLIN_DLL"].split(":")
    ]
    return {
        "LD_LIBRARY_PATH": f"{prefix / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "ROOT_INCLUDE_PATH": f"{prefix / 'include'}:{os.environ.get('ROOT_INCLUDE_PATH', '')}",
        "MARLIN_DLL": ":".join(marlin),
    }
