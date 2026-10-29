"""The example evaluator, loaded the way ``k4opt evaluate --evaluator`` loads it."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from k4opt.evaluation import Evaluator, load_evaluator

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CALO_HIT_MULTIPLICITY = (
    f"{EXAMPLES / 'evaluators' / 'calo_hit_multiplicity.py'}:CaloHitMultiplicity"
)


def calo_hit_multiplicity() -> Evaluator:
    """A new instance of the example evaluator."""
    return load_evaluator(CALO_HIT_MULTIPLICITY)


# The loader runs a file once, so tests and the command share this module object and
# monkeypatching its functions affects the evaluator that ``k4opt evaluate`` loaded.
calo_hits: ModuleType = sys.modules[type(calo_hit_multiplicity()).__module__]
