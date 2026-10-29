"""The example evaluator, loaded the way ``k4opt evaluate --evaluator`` loads it."""

from __future__ import annotations

from pathlib import Path

from k4opt.evaluation import Evaluator, load_evaluator

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CALO_HIT_MULTIPLICITY = (
    f"{EXAMPLES / 'evaluators' / 'calo_hit_multiplicity.py'}:CaloHitMultiplicity"
)


def calo_hit_multiplicity() -> Evaluator:
    """A new instance of the example evaluator."""
    return load_evaluator(CALO_HIT_MULTIPLICITY)
