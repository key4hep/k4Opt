"""Concrete evaluators that ship with k4opt, built on the :mod:`k4opt.evaluation` contract.

The contract does not depend on any of them: an evaluator is anything with ``name``,
``version`` and ``evaluate``, and users add their own the same way. Nothing here is looked up
by name yet; ``k4opt evaluate`` uses :class:`CaloHitMultiplicity` for a problem that declares
``calo.hit_multiplicity``. See ``docs/evaluators.md``.
"""

from k4opt.evaluators.calo_hits import CaloHitMultiplicity

__all__ = ["CaloHitMultiplicity"]
