"""Example evaluator ``calo.hit_multiplicity``: distinct cells hit in the simulated ECAL barrel.

A complete, working evaluator to copy from. It is not part of the k4opt package; name it on the
command line::

    k4opt evaluate examples/problem/cld-ecal-hit-count.yaml \\
        --evaluator examples/evaluators/calo_hit_multiplicity.py:CaloHitMultiplicity \\
        --param constants.ECal_cell_size=6.0 --output runs/cld-ecal-6mm

It only reads: ``requires = "simulation"`` makes k4opt run generation and ``ddsim`` first, check
that the EDM4hep file holds the requested events and hand it over. The one measurement is
``simulation`` / ``ecal_barrel.mean_hit_cells``, defined in ``docs/evaluators.md``. The metric is
a technical observable, not a detector-performance measure.
"""

from k4opt.evaluation import Evaluator


class CaloHitMultiplicity(Evaluator):
    name = "calo.hit_multiplicity"
    version = "1"
    requires = "simulation"
    settings = {"collection": str}
    metrics = {"ecal_barrel.mean_hit_cells": "1"}

    def evaluate(self, request):
        hits = request.read(
            "simulation", request.settings["collection"], "edm4hep::SimCalorimeterHit"
        )
        cells = [len({hit.getCellID() for hit in event}) for event in hits]
        if max(cells) == 0:
            self.invalid("no hits in any event")
        return {"ecal_barrel.mean_hit_cells": cells}
