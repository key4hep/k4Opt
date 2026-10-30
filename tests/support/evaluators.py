"""Evaluators loaded the way ``k4opt evaluate --evaluator`` loads them: the example and minimal
ones for the framework's tests."""

from __future__ import annotations

from pathlib import Path

from k4opt.evaluation import Evaluator, load_evaluator

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CALO_PHOTON = f"{EXAMPLES / 'evaluators' / 'calo_photon.py'}:CaloPhoton"
CELL_COUNT = f"{Path(__file__).with_name('cell_count.py')}:CellCount"
HIT_ENERGY = f"{Path(__file__).with_name('hit_energy.py')}:HitEnergy"
PARAMETER_ECHO = f"{Path(__file__).with_name('parameter_echo.py')}:ParameterEcho"
TRADE_OFF = f"{Path(__file__).with_name('trade_off.py')}:TradeOff"
CANDIDATE_CHECK = f"{Path(__file__).with_name('candidate_check.py')}:CandidateCheck"


def calo_photon() -> Evaluator:
    """A new instance of the example evaluator."""
    return load_evaluator(CALO_PHOTON)


def cell_count() -> Evaluator:
    """A new instance of the test evaluator ``test.cell_count``."""
    return load_evaluator(CELL_COUNT)


def hit_energy() -> Evaluator:
    """A new instance of the test evaluator ``test.hit_energy``, a second simulation reader."""
    return load_evaluator(HIT_ENERGY)


def parameter_echo() -> Evaluator:
    """A new instance of the test evaluator ``test.parameter_echo``, which reads no stage."""
    return load_evaluator(PARAMETER_ECHO)


def trade_off() -> Evaluator:
    """A new instance of the test evaluator ``test.trade_off``, which reads no stage."""
    return load_evaluator(TRADE_OFF)


def candidate_check() -> Evaluator:
    """A new instance of ``test.candidate_check``, which records what it is asked."""
    return load_evaluator(CANDIDATE_CHECK)
