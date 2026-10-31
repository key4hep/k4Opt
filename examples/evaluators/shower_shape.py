"""Toy ECAL transverse shower widths for photon/pi0 gun samples, before reconstruction.

Each event contributes its energy-weighted RMS distance from the radial shower axis, in mm.
The histogram keeps overflow counts so a combined evaluator cannot silently discard tails.
"""

import json
from collections.abc import Mapping

import numpy as np

from k4opt.evaluation import Collection, EvaluationResult, Evaluator, Measurement, Quantity


class ShowerShape(Evaluator):
    name, version, stage = "calo.shower_shape", "1", "simulation"
    collections = {"hits": Collection("SimCalorimeterHit")}
    settings = {"threshold": Quantity("GeV"), "bins": dict}
    metrics = {"ecal_barrel.shower_width": "mm"}
    artifacts = {"histogram": "transverse widths in mm: JSON edges, counts, underflow, overflow"}

    def check(self, request):
        bins = request.settings["bins"]
        if not isinstance(bins, Mapping) or set(bins) != {"low", "high", "count"}:
            self.fail("bins needs low, high and count")
        low, high, count = (bins[k] for k in ("low", "high", "count"))
        if (
            any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in (low, high))
            or not 0 <= low < high
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            self.fail("bins needs 0 <= low < high and a positive integer count")
        if request.settings["threshold"] < 0:
            self.fail("threshold must be nonnegative")

    def evaluate(self, request):
        widths = []
        for event in self.events(request):
            hits = [h for h in event.hits if h.getEnergy() > request.settings["threshold"]]
            if not hits:
                self.invalid("an event has no energy above threshold; no shower width")
            weights = np.array([h.getEnergy() for h in hits])
            positions = np.array(
                [(h.getPosition().x, h.getPosition().y, h.getPosition().z) for h in hits]
            )
            center = np.average(positions, axis=0, weights=weights)
            radius = np.linalg.norm(center)
            if not radius:
                self.invalid("the shower center is at the origin; no radial axis")
            axis = center / radius
            offsets = positions - center
            transverse = offsets - np.outer(offsets @ axis, axis)
            widths.append(
                float(np.sqrt(np.average(np.sum(transverse**2, axis=1), weights=weights)))
            )
        if not widths:
            self.invalid("no showers to measure")
        bins = request.settings["bins"]
        edges = np.linspace(bins["low"], bins["high"], bins["count"] + 1)
        counts, _ = np.histogram(widths, bins=edges)
        histogram = {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
            "underflow": sum(w < float(edges[0]) for w in widths),
            "overflow": sum(w > float(edges[-1]) for w in widths),
        }
        path = request.workdir / "histogram.json"
        path.write_text(json.dumps(histogram), encoding="utf-8")
        return EvaluationResult.valid(
            [Measurement.from_values("simulation", "ecal_barrel.shower_width", widths, "mm")],
            artifacts={"histogram": path},
        )
