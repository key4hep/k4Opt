"""Evaluators loaded the way ``k4opt evaluate --evaluator`` loads them: the example and a
minimal one for the framework's tests."""

from __future__ import annotations

from pathlib import Path

from k4opt.evaluation import Evaluator, load_evaluator

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CALO_PHOTON = f"{EXAMPLES / 'evaluators' / 'calo_photon.py'}:CaloPhoton"
CELL_COUNT = f"{Path(__file__).with_name('cell_count.py')}:CellCount"


def calo_photon() -> Evaluator:
    """A new instance of the example evaluator."""
    return load_evaluator(CALO_PHOTON)


def cell_count() -> Evaluator:
    """A new instance of the test evaluator ``test.cell_count``."""
    return load_evaluator(CELL_COUNT)
