"""Evaluator ``pfo.energy``: the mean energy of the particle-flow objects per event, which every
calibration of the reconstruction changes (the acceptance test of a rerun)."""

from k4opt.evaluation import Collection, Evaluator


class PfoEnergy(Evaluator):
    name = "pfo.energy"
    version = "1"
    stage = "reconstruction"
    collections = {"pfos": Collection("ReconstructedParticle", default="PandoraPFOs")}
    metrics = {"pfo.energy": "GeV"}

    def evaluate(self, request):
        totals = [sum(p.getEnergy() for p in event.pfos) for event in self.events(request)]
        if not totals:
            self.invalid("no events")
        return {"pfo.energy": totals}
