"""``test.cell_count``: the distinct cells hit per event, a minimal evaluator for the
framework's tests. Loaded from its file, as ``k4opt evaluate --evaluator`` loads one."""

from k4opt.evaluation import Collection, Evaluator


class CellCount(Evaluator):
    name = "test.cell_count"
    version = "1"
    stage = "simulation"
    collections = {"hits": Collection("SimCalorimeterHit")}
    metrics = {"ecal_barrel.mean_hit_cells": "1"}

    def evaluate(self, request):
        cells = [len({hit.getCellID() for hit in event.hits}) for event in self.events(request)]
        if max(cells) == 0:
            self.invalid("no hits in any event")
        return {"ecal_barrel.mean_hit_cells": cells}
