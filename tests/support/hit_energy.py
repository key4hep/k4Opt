"""``test.hit_energy``: the energy deposited per event, a second reader of the simulation for the
framework's tests. Loaded from its file, as ``k4opt evaluate --evaluator`` loads one."""

from k4opt.evaluation import Evaluator


class HitEnergy(Evaluator):
    name = "test.hit_energy"
    version = "1"
    stage = "simulation"
    settings = {"collection": str}
    metrics = {"ecal_barrel.mean_hit_energy": "GeV"}

    def evaluate(self, request):
        hits = self.read(request, request.settings["collection"], "SimCalorimeterHit")
        return {
            "ecal_barrel.mean_hit_energy": [sum(h.getEnergy() for h in event) for event in hits]
        }
