"""Approximate cylindrical ECAL barrel channel cost, not a count from detector segmentation.

Area = 2 pi radius length layers; channels = area / cell_size**2. Dimensions, layer count and
cell size are explicit settings: the problem sets ``cell_size`` from the parameter that sets the
geometry's cell size, so both always agree. Gaps, endcaps and variations between layers are
omitted.
"""

import math

from k4opt.evaluation import Evaluator, Quantity


class EcalChannels(Evaluator):
    name, version = "ecal.channels", "1"
    settings = {
        "layers": int,
        "radius": Quantity("mm"),
        "length": Quantity("mm"),
        "cell_size": Quantity("mm"),
    }
    metrics = {("geometry", "ecal_barrel.channels"): "1"}

    def check(self, request):
        if any(request.settings[k] <= 0 for k in ("layers", "radius", "length", "cell_size")):
            self.fail("layers, radius, length and cell_size must be positive")

    def evaluate(self, request):
        size = request.settings["cell_size"]
        area = 2 * math.pi * request.settings["radius"] * request.settings["length"]
        return {("geometry", "ecal_barrel.channels"): area * request.settings["layers"] / size**2}
