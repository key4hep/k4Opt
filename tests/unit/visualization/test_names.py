"""What the figures call the things of a problem: titles, names, and more only where alike."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from k4opt.problem import parse_problem
from k4opt.visualization import names

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "problem"
RESIDUAL = "calo.photon/simulation/ecal_barrel.angular_residual"


def bindings() -> dict[str, Any]:
    """Two samples, an objective of one metric on each; without the example's titles, which each
    test gives as it needs them."""
    document: dict[str, Any] = yaml.safe_load((EXAMPLES / "cld-ecal-bindings.yaml").read_text())
    for item in (
        *document["parameters"].values(),
        *document["samples"].values(),
        *document["objectives"],
    ):
        item.pop("title", None)
    return document


def test_without_titles_things_are_called_by_their_names_and_alike_metrics_by_sample() -> None:
    called = names(parse_problem(bindings()))
    assert called.parameter("ecal_cell_size") == "ecal_cell_size"
    assert called.sample("gamma_10gev") == "gamma_10gev"
    assert (
        called.objective(f"gamma_10gev/{RESIDUAL}") == "ecal_barrel.angular_residual · gamma_10gev"
    )
    assert (
        called.objective(f"gamma_50gev/{RESIDUAL}") == "ecal_barrel.angular_residual · gamma_50gev"
    )


def test_titles_name_them_and_a_sample_is_said_by_its_own_title() -> None:
    document = bindings()
    document["parameters"]["ecal_cell_size"]["title"] = "ECAL cell size"
    document["samples"]["gamma_10gev"]["title"] = "10 GeV γ"
    for objective in document["objectives"]:
        objective["title"] = "Angular residual"
    called = names(parse_problem(document))
    assert called.parameter("ecal_cell_size") == "ECAL cell size"
    assert called.parameter("range_cut") == "range_cut"
    assert called.objective(f"gamma_10gev/{RESIDUAL}") == "Angular residual · 10 GeV γ"
    assert called.objective(f"gamma_50gev/{RESIDUAL}") == "Angular residual · gamma_50gev"


def test_a_metric_alone_of_its_title_is_called_by_its_title_alone() -> None:
    document = bindings()
    document["objectives"][0]["title"] = "Residual at 10 GeV"
    document["objectives"][1]["title"] = "Residual at 50 GeV"
    called = names(parse_problem(document))
    assert called.objective(f"gamma_10gev/{RESIDUAL}") == "Residual at 10 GeV"
    assert called.objective(f"gamma_50gev/{RESIDUAL}") == "Residual at 50 GeV"


def test_metrics_alike_on_one_sample_are_called_by_their_labels() -> None:
    document = bindings()
    fired = "calo.photon/simulation/ecal_barrel.fired_cells"
    document["objectives"].append(
        {"sample": "gamma_10gev", "stage": "simulation", "metric": "ecal_barrel.fired_cells"}
        | {"direction": "minimize"}
    )
    for objective in document["objectives"]:
        objective["title"] = "Resolution"
    called = names(parse_problem(document))
    assert called.objective(f"gamma_10gev/{RESIDUAL}") == f"gamma_10gev/{RESIDUAL}"
    assert called.objective(f"gamma_10gev/{fired}") == f"gamma_10gev/{fired}"
    assert called.objective(f"gamma_50gev/{RESIDUAL}") == "Resolution · gamma_50gev"


def test_parameters_titled_alike_are_each_called_by_the_title_and_their_name() -> None:
    document = bindings()
    document["parameters"]["ecal_cell_size"]["title"] = "Cut"
    document["parameters"]["range_cut"]["title"] = "Cut"
    document["parameters"]["cell_threshold"]["title"] = "Threshold"
    called = names(parse_problem(document))
    assert called.parameter("ecal_cell_size") == "Cut (ecal_cell_size)"
    assert called.parameter("range_cut") == "Cut (range_cut)"
    assert called.parameter("cell_threshold") == "Threshold"


def test_samples_titled_alike_are_each_called_by_the_title_and_their_name() -> None:
    document = bindings()
    for sample in document["samples"].values():
        sample["title"] = "γ"
    called = names(parse_problem(document))
    assert called.sample("gamma_10gev") == "γ (gamma_10gev)"
    assert called.sample("gamma_50gev") == "γ (gamma_50gev)"


def test_an_objective_and_a_constraint_on_one_metric_are_each_called_by_their_own_title() -> None:
    document = bindings()
    document["objectives"][0]["title"] = "Angular residual"
    document["constraints"] = [
        document["objectives"][0]
        | {"title": "Residual limit", "max": 1.0, "unit": "mrad", "direction": None}
    ]
    del document["constraints"][0]["direction"]
    called = names(parse_problem(document))
    assert called.objective(f"gamma_10gev/{RESIDUAL}") == "Angular residual"
    assert called.constraint(f"gamma_10gev/{RESIDUAL}") == "Residual limit"
