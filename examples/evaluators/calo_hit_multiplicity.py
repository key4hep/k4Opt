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
    name = "calo.hit_multiplicity"  # must match evaluation.evaluator in the problem file
    version = "1"  # and evaluation.version
    requires = "simulation"  # k4opt runs generation and ddsim first and checks the event count
    settings = {"collection": str}  # evaluation.settings: the collection to read (required)
    metrics = {"ecal_barrel.mean_hit_cells": "1"}  # metric -> unit; its stage is `requires`

    def evaluate(self, request):
        # one list of hits per event, from the EDM4hep file of the simulation stage
        hits = self.read(request, request.settings["collection"], "SimCalorimeterHit")
        # distinct cells hit in each event (several hits can share a cell)
        cells = [len({hit.getCellID() for hit in event}) for event in hits]
        if max(cells) == 0:
            self.invalid("no hits in any event")  # ran, but the result is unusable
        # a list is averaged: its mean with the standard error becomes the measurement
        return {"ecal_barrel.mean_hit_cells": cells}
