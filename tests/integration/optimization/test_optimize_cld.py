"""``k4opt optimize`` on the CLD example with the real ddsim (needs the Key4hep stack and Optuna).

Skipped without ``$K4GEO``, ``ddsim``, ``podio`` or Optuna. Runs a study of two candidates of a
few events each, then resumes it for one more; then one of four candidates two at a time, resumed
with one job; about six minutes, mostly ddsim start-up.
Verify by hand with::

    source setup.sh && pytest tests/integration/optimization -m integration -s
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from k4opt.cli import main
from k4opt.problem import Assessment, load_problem
from tests.support.evaluators import CALO_PHOTON
from tests.support.geometry import CLD_COMPACT

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "problem" / "cld-ecal-granularity.yaml"


@pytest.fixture(scope="module")
def problem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    k4geo = os.environ.get("K4GEO")
    if not k4geo or not (Path(k4geo) / CLD_COMPACT).is_file():
        pytest.skip("$K4GEO with the CLD geometry is required")
    if shutil.which("ddsim") is None or importlib.util.find_spec("podio") is None:
        pytest.skip("ddsim and podio from the Key4hep stack are required")
    pytest.importorskip("optuna")
    document = yaml.safe_load(EXAMPLE.read_text())
    document["samples"]["gamma_10gev"]["simulation"]["events"] = 4
    path = tmp_path_factory.mktemp("problem") / "cld-ecal-granularity.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def optimize(problem: Path, output: Path, *argv: str) -> str:
    out, err = io.StringIO(), io.StringIO()
    command = ["optimize", str(problem), "--evaluator", CALO_PHOTON, "--output", str(output)]
    code = main([*command, *argv], stdout=out, stderr=err)
    assert code == 0, out.getvalue() + err.getvalue()
    return out.getvalue()


def test_a_study_of_the_example_runs_resumes_and_tells_the_optimizer_its_records(
    problem: Path, tmp_path: Path
) -> None:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    output = tmp_path / "study"
    first = optimize(problem, output, "--trials", "2", "--seed", "1")
    assert "trials:   2: 2 valid" in first
    resumed = optimize(problem, output, "--trials", "1", "--resume")
    assert resumed.startswith("trial 2:") and "trials:   3: 3 valid" in resumed

    document = json.loads((output / "study.json").read_text())
    journal = str(output / "study.journal")
    storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
    study = optuna.load_study(study_name=document["optimizer"]["study"], storage=storage)
    spec = load_problem(problem)
    sizes = []
    for trial in study.trials:
        summary = json.loads((output / f"trial-{trial.number:06d}" / "summary.json").read_text())
        simulated = summary["samples"]["gamma_10gev"]["stages"]["simulation"]
        assert summary["status"] == "valid" and simulated["events_read"] == 4
        assessment = Assessment.from_summary(spec, summary)
        assert trial.values == [float(m.value) for m in assessment.objectives or ()]
        (size,) = trial.params.values()
        assert 2.5 <= size <= 20.0
        assert summary["geometry"]["changes"][0]["new_value"] == f"{size!r}*mm"
        sizes.append(size)
    assert len(set(sizes)) == 3  # the resumed session proposed a new candidate

    # read back as it is, with the units, uncertainties and times its records hold
    shown, exported = io.StringIO(), io.StringIO()
    assert main(["study", "show", str(output)], stdout=shown, stderr=io.StringIO()) == 0
    assert "trials:     3: 3 valid" in shown.getvalue()
    assert (
        "gamma_10gev/calo.photon/simulation/ecal_barrel.angular_residual  minimize  (mrad)"
        in shown.getvalue()
    )
    command = ["study", "export", str(output), "--format", "json"]
    assert main(command, stdout=exported, stderr=io.StringIO()) == 0
    for entry in json.loads(exported.getvalue())["trials"]:
        summary = json.loads((output / entry["record"]["path"]).read_text())
        assert entry["assessment"] == summary["assessment"]
        assert (
            entry["stage_seconds"]["gamma_10gev/simulation"]
            == summary["samples"]["gamma_10gev"]["stages"]["simulation"]["elapsed_seconds"]
        )
    assert [m["uncertainty"] is not None for m in entry["assessment"]["objectives"]] == [True, True]


def test_a_study_runs_trials_at_once_and_resumes_with_another_number_of_jobs(
    problem: Path, tmp_path: Path
) -> None:
    output = tmp_path / "study"
    first = optimize(problem, output, "--trials", "4", "--seed", "2", "--jobs", "2")
    assert "trials:   4: 4 valid" in first
    resumed = optimize(problem, output, "--trials", "1", "--resume", "--jobs", "1")
    assert resumed.startswith("trial 4:") and "trials:   5: 5 valid" in resumed
    spec = load_problem(problem)
    for number, jobs in ((0, 2), (1, 2), (2, 2), (3, 2), (4, 1)):
        summary = json.loads((output / f"trial-{number:06d}" / "summary.json").read_text())
        assert summary["environment"]["observed"]["execution"] == {"backend": "local", "jobs": jobs}
        assert summary["samples"]["gamma_10gev"]["stages"]["simulation"]["events_read"] == 4
        Assessment.from_summary(spec, summary)
    shown = io.StringIO()
    assert main(["study", "show", str(output)], stdout=shown, stderr=io.StringIO()) == 0
    assert "trials:     5: 5 valid" in shown.getvalue()
