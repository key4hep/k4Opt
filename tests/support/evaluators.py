"""Evaluator classes loaded the way ``k4opt evaluate --evaluator`` loads them: the example and
minimal ones for the framework's tests. Each helper returns the class; k4opt creates the
instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from k4opt.evaluation import Evaluator, load_evaluator

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CALO_PHOTON = f"{EXAMPLES / 'evaluators' / 'calo_photon.py'}:CaloPhoton"
CELL_COUNT = f"{Path(__file__).with_name('cell_count.py')}:CellCount"
HIT_ENERGY = f"{Path(__file__).with_name('hit_energy.py')}:HitEnergy"
PARAMETER_ECHO = f"{Path(__file__).with_name('parameter_echo.py')}:ParameterEcho"
TRADE_OFF = f"{Path(__file__).with_name('trade_off.py')}:TradeOff"
CANDIDATE_CHECK = f"{Path(__file__).with_name('candidate_check.py')}:CandidateCheck"


def calo_photon() -> type[Evaluator]:
    """The class of the example evaluator."""
    return load_evaluator(CALO_PHOTON)


def cell_count() -> type[Evaluator]:
    """The class of the test evaluator ``test.cell_count``."""
    return load_evaluator(CELL_COUNT)


def hit_energy() -> type[Evaluator]:
    """The class of the test evaluator ``test.hit_energy``, a second simulation reader."""
    return load_evaluator(HIT_ENERGY)


def parameter_echo() -> type[Evaluator]:
    """The class of the test evaluator ``test.parameter_echo``, which reads no stage."""
    return load_evaluator(PARAMETER_ECHO)


def trade_off() -> type[Evaluator]:
    """The class of the test evaluator ``test.trade_off``, which reads no stage."""
    return load_evaluator(TRADE_OFF)


def candidate_check() -> type[Evaluator]:
    """The class of ``test.candidate_check``, which records what its instances are asked."""
    return load_evaluator(CANDIDATE_CHECK)


def implementations(*given: Any) -> list[type[Any]]:
    """What a test evaluates with: a class as it is, and an instance as a class each of whose new
    instances is that very instance, so that the test can watch what it is asked. k4opt itself
    creates a new instance for every candidate."""
    return [item if isinstance(item, type) else _watched(item) for item in given]


def _watched(instance: Any) -> type[Any]:
    cls = type(instance)
    # __new__ gives an object that is no instance of the subclass, so __init__ does not run again,
    # and the class k4opt sees, type(instance), is the same for every "new" instance
    return type(cls.__name__, (cls,), {"__new__": lambda subclass: instance})
