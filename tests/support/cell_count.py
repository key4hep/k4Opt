"""``test.cell_count``: the distinct cells hit per event, a minimal evaluator for the
framework's tests. Loaded from its file, as ``k4opt evaluate --evaluator`` loads one."""

from k4opt.evaluation import Evaluator


class CellCount(Evaluator):
    name = "test.cell_count"
    version = "1"
    requires = "simulation"
    settings = {"collection": str}
    metrics = {"ecal_barrel.mean_hit_cells": "1"}

    def evaluate(self, request):
        hits = self.read(request, request.settings["collection"], "SimCalorimeterHit")
        cells = [len({hit.getCellID() for hit in event}) for event in hits]
        if max(cells) == 0:
            self.invalid("no hits in any event")
        return {"ecal_barrel.mean_hit_cells": cells}
