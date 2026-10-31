"""``test.yields``: how many events of a sample pass a selection, as sums of weights (every event
weighs 1), and the energy each event deposited as the artifact ``histogram``; for the framework's
tests of combined evaluators. Loaded from its file, as ``k4opt evaluate --evaluator`` loads one."""

import json

from k4opt.evaluation import Collection, EvaluationResult, Evaluator, Measurement, Quantity

SELECTED = ("analysis", "selected.sum_weights")
GENERATED = ("analysis", "generated.sum_weights")


class Yields(Evaluator):
    name = "test.yields"
    version = "1"
    stage = "simulation"
    collections = {"hits": Collection("SimCalorimeterHit")}
    # an event is selected if its hits deposit at least min_energy
    settings = {"min_energy": Quantity("GeV", default=0.0)}
    metrics = {SELECTED: "1", GENERATED: "1"}
    artifacts = {"histogram": "the energy each event deposited, in GeV: JSON [energy, ...]"}

    def evaluate(self, request):
        energies = [sum(hit.getEnergy() for hit in event.hits) for event in self.events(request)]
        selected = sum(1 for energy in energies if energy >= request.settings["min_energy"])
        histogram = request.workdir / "histogram.json"
        histogram.write_text(json.dumps(energies))
        return EvaluationResult.valid(
            [
                Measurement(*SELECTED, selected, "1", selected**0.5),
                Measurement(*GENERATED, len(energies), "1"),
            ],
            artifacts={"histogram": histogram},
        )
