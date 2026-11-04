"""The reconstruction stage of a problem: a k4run job on the simulation, parsed and checked
(no Key4hep)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.problem import ProblemConfigError, parse_problem
from k4opt.problem.candidate import configure, validate_candidate
from k4opt.problem.models import StageDestination
from k4opt.problem.plan import Kind, plan_tasks
from tests.support.problems import two_samples

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "problem"
CLD = {
    "executor": "k4run",
    "directory": "$CLDCONFIG/share/CLDConfig",
    "options": "CLDReconstruction.py",
    "geometry": "compactFile",
    "truthTracking": True,
}


@pytest.fixture
def reco(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` whose evaluator reads a reconstruction of the simulation."""
    doc["samples"]["gamma"]["reconstruction"] = {**CLD, "cms": 91}
    doc["evaluators"][0]["stage"] = "reconstruction"
    for item in (*doc["objectives"], *doc["constraints"]):
        item["stage"] = "reconstruction"
    return doc


def reject(data: object, message: str) -> None:
    with pytest.raises(ProblemConfigError, match=re.escape(message)):
        parse_problem(data)


def test_a_reconstruction_reads_the_simulation_and_keeps_its_options(reco: dict[str, Any]) -> None:
    problem = parse_problem(reco)
    stage = problem.samples["gamma"].stages["reconstruction"]
    assert (stage.executor, stage.events, stage.seed) == ("k4run", None, None)
    assert dict(stage.options) == {**{k: v for k, v in CLD.items() if k != "executor"}, "cms": 91}
    plan = plan_tasks(problem.samples, problem.evaluators, references=problem.references)
    assert [task.id.label for task in plan.of(Kind.STAGE)] == [
        "gamma/generation",
        "gamma/simulation",
        "gamma/reconstruction",
    ]
    assert parse_problem(problem.to_dict()) == problem


def test_the_samples_complete_a_shared_reconstruction(reco: dict[str, Any]) -> None:
    doc = two_samples(reco)
    del doc["samples"]["gamma"]["reconstruction"]
    del doc["samples"]["pion"]["reconstruction"]
    doc["stages"] = {"reconstruction": CLD}
    doc["samples"]["gamma"]["reconstruction"] = {"cms": 91}
    doc["samples"]["pion"]["reconstruction"] = {"cms": 240}
    problem = parse_problem(doc)
    assert problem.samples["gamma"].stages["reconstruction"].options["cms"] == 91
    assert problem.samples["pion"].stages["reconstruction"].options["cms"] == 240


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda s: s["reconstruction"].update(events=10),
            "samples.gamma.reconstruction.events: k4run processes every event of "
            "samples.gamma.simulation; set the events there",
        ),
        (
            lambda s: s["reconstruction"].update(seed=1),
            "samples.gamma.reconstruction.seed: k4run takes no seed of k4opt's",
        ),
        (
            lambda s: s["reconstruction"].pop("options"),
            "samples.gamma.reconstruction: missing required key 'options'",
        ),
        (
            lambda s: s["reconstruction"].update({"GeoSvc.detectors": "/my.xml"}),
            "samples.gamma.reconstruction.GeoSvc.detectors: k4opt sets this argument itself: the "
            "geometry is the candidate's",
        ),
        (
            lambda s: s["reconstruction"].update(options=["a.py", "../b.py"]),
            "samples.gamma.reconstruction.options[1]: '../b.py': with directory:",
        ),
        (
            lambda s: s["reconstruction"].update(executor="marlin"),
            "samples.gamma.reconstruction.executor: must be one of ['k4run']",
        ),
        (
            lambda s: s.pop("simulation"),
            "samples.gamma.reconstruction: reconstruction reads simulation: add "
            "samples.gamma.simulation with executor: ddsim",
        ),
    ],
)
def test_reconstruction_errors(reco: dict[str, Any], edit: Any, message: str) -> None:
    edit(reco["samples"]["gamma"])
    reject(reco, message)


def test_a_reconstruction_no_evaluator_reads_is_unused(doc: dict[str, Any]) -> None:
    doc["samples"]["gamma"]["reconstruction"] = CLD
    reject(doc, "samples.gamma.reconstruction: is not used")


def test_a_parameter_sets_an_argument_of_the_job(reco: dict[str, Any]) -> None:
    del reco["samples"]["gamma"]["reconstruction"]["cms"]
    reco["parameters"]["cms"] = {
        "type": "categorical",
        "choices": [91, 240],
        "targets": {"reconstruction": "cms"},
    }
    problem = parse_problem(reco)
    assert problem.parameters[1].targets == (StageDestination("reconstruction", "cms", ("gamma",)),)
    assert problem.pending("gamma") == {"reconstruction": frozenset({"cms"})}
    values, _ = validate_candidate(problem, {"ecal_cell_size": 5.0, "cms": 240})
    options = configure(problem, values).samples["gamma"].stages["reconstruction"].options
    assert options["cms"] == 240  # the value itself: the job decides its unit


@pytest.mark.parametrize(
    ("target", "unit", "message"),
    [
        ("cms", "GeV", "option 'cms' takes a bare value, whose unit the job decides"),
        ("options", None, "k4run option 'options': a parameter cannot set it"),
        ("IOSvc.Input", None, "k4run option 'IOSvc.Input': k4opt sets this argument itself"),
    ],
)
def test_what_a_parameter_cannot_set_in_a_reconstruction(
    reco: dict[str, Any], target: str, unit: str | None, message: str
) -> None:
    del reco["samples"]["gamma"]["reconstruction"]["cms"]
    reco["parameters"]["x"] = {
        "type": "continuous",
        "bounds": [1.0, 2.0],
        "targets": {"reconstruction": target},
        **({"unit": unit} if unit else {}),
    }
    reject(reco, message)


def test_the_reconstruction_example_parses() -> None:
    path = EXAMPLES / "cld-ecal-pi0-reco.yaml"
    problem = parse_problem(yaml.safe_load(path.read_text()), source=path)
    stage = problem.samples["pi0_50gev"].stages["reconstruction"]
    assert stage.options["geometry"] == "compactFile"
    assert [e.stage for e in problem.evaluators] == ["reconstruction", None]
