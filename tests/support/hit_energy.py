"""``test.hit_energy``: the energy deposited per event, a second reader of the simulation for the
framework's tests. Loaded from its file, as ``k4opt evaluate --evaluator`` loads one."""

from k4opt.evaluation import Collection, Evaluator


class HitEnergy(Evaluator):
    name = "test.hit_energy"
    version = "1"
    stage = "simulation"
    collections = {"hits": Collection("SimCalorimeterHit")}
    metrics = {"ecal_barrel.mean_hit_energy": "GeV"}

    def evaluate(self, request):
        return {
            "ecal_barrel.mean_hit_energy": [
                sum(hit.getEnergy() for hit in event.hits) for event in self.events(request)
            ]
        }
