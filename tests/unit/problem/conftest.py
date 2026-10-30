"""Shared fixtures for the problem unit tests."""

from __future__ import annotations

import copy

import pytest

_BASE: dict[str, object] = {
    "schema_version": "k4opt.problem/v1alpha1",
    "name": "demo",
    "environment": {"key4hep": {"channel": "nightly", "version": "2026-10-20"}},
    "geometry": {"entry": "FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml"},
    "parameters": [
        {
            "geometry": {"constant": "ECal_cell_size"},
            "type": "continuous",
            "bounds": [4.0, 8.0],
            "unit": "mm",
        }
    ],
    "stages": {
        "generation": {
            "executor": "gun",
            "particle": "gamma",
            "energy": {"value": 10, "unit": "GeV"},
            "direction": {
                "distribution": "uniform",
                "theta": [80, 100],
                "phi": [0, 360],
                "unit": "deg",
            },
        },
        "simulation": {
            "executor": "ddsim",
            "events": 1000,
            "seed": 42,
            "physics.list": "FTFP_BERT",
        },
    },
    "evaluators": [
        {
            "name": "calo.single_particle_shower",
            "version": "1",
            "stage": "simulation",
            "settings": {"collection": "ECalBarrelCollection"},
        }
    ],
    "objectives": [{"stage": "simulation", "metric": "a.rms", "direction": "minimize"}],
    "constraints": [{"stage": "simulation", "metric": "a.cells", "max": 100}],
}


@pytest.fixture
def doc() -> dict[str, object]:
    """A fresh, valid problem document (plain dict) that a test may mutate."""
    return copy.deepcopy(_BASE)
