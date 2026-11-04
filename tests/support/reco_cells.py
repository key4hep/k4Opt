"""``test.reco_cells``: the distinct cells hit per event of a reconstruction, a minimal reader of
the reconstruction stage for the framework's tests. Loaded from its file, as ``k4opt evaluate
--evaluator`` loads one."""

from k4opt.evaluation import Collection, Evaluator


class RecoCells(Evaluator):
    name = "test.reco_cells"
    version = "1"
    stage = "reconstruction"
    collections = {"hits": Collection("SimCalorimeterHit")}
    metrics = {"ecal_barrel.mean_hit_cells": "1"}

    def evaluate(self, request):
        cells = [len({hit.getCellID() for hit in event.hits}) for event in self.events(request)]
        return {"ecal_barrel.mean_hit_cells": cells}
