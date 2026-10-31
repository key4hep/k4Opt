"""Approximate cylindrical ECAL barrel channel cost, not a count from detector segmentation.

Area = 2 pi radius length layers; channels = area / cell_size**2. Dimensions and layer count
are explicit settings; gaps, endcaps and variations between layers are omitted.
"""

import math

from k4opt.evaluation import Evaluator, Quantity


class EcalChannels(Evaluator):
    name, version = "ecal.channels", "1"
    settings = {"layers": int, "radius": Quantity("mm"), "length": Quantity("mm")}
    metrics = {("geometry", "ecal_barrel.channels"): "1"}

    def check(self, request):
        if any(request.settings[k] <= 0 for k in ("layers", "radius", "length")):
            self.fail("layers, radius and length must be positive")
        if request.candidate.get("constants.ECal_cell_size", 0) <= 0:
            self.fail("needs a positive constants.ECal_cell_size candidate in mm")

    def evaluate(self, request):
        size = request.candidate["constants.ECal_cell_size"]
        area = 2 * math.pi * request.settings["radius"] * request.settings["length"]
        return {("geometry", "ecal_barrel.channels"): area * request.settings["layers"] / size**2}
