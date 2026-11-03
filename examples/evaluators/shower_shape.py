"""Toy ECAL halo fractions for photon/pi0 gun samples, before reconstruction.

Each event contributes the fraction of its energy deposited farther than ``radius`` from the
radial shower axis. A photon keeps most of its energy near its axis; the two photons of a pi0
put energy off the axis of their common center, where only cells finer than their separation
can place it. The histogram keeps overflow counts so a combined evaluator cannot silently
discard tails.
"""

import json
from collections.abc import Mapping

import numpy as np

from k4opt.evaluation import (
    Collection,
    EvaluationContractError,
    EvaluationResult,
    Evaluator,
    Measurement,
    Quantity,
)


class ShowerShape(Evaluator):
    name, version, stage = "calo.shower_shape", "1", "simulation"
    collections = {"hits": Collection("SimCalorimeterHit")}
    settings = {"threshold": Quantity("GeV"), "radius": Quantity("mm"), "bins": dict}
    metrics = {"ecal_barrel.halo_fraction": "1"}
    artifacts = {"histogram": "halo fractions: JSON edges, counts, underflow, overflow"}

    @classmethod
    def check(cls, request):
        bins = request.settings["bins"]
        if not isinstance(bins, Mapping) or set(bins) != {"low", "high", "count"}:
            raise EvaluationContractError("bins needs low, high and count")
        low, high, count = (bins[k] for k in ("low", "high", "count"))
        if (
            any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in (low, high))
            or not 0 <= low < high
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise EvaluationContractError("bins needs 0 <= low < high and a positive integer count")
        if request.settings["threshold"] < 0:
            raise EvaluationContractError("threshold must be nonnegative")
        if request.settings["radius"] <= 0:
            raise EvaluationContractError("radius must be positive")

    def evaluate(self, request):
        fractions = []
        for event in self.events(request):
            hits = [h for h in event.hits if h.getEnergy() > request.settings["threshold"]]
            if not hits:
                self.invalid("an event has no energy above threshold; no halo fraction")
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
            distances = np.linalg.norm(offsets - np.outer(offsets @ axis, axis), axis=1)
            halo = weights[distances > request.settings["radius"]].sum()
            fractions.append(float(halo / weights.sum()))
        if not fractions:
            self.invalid("no showers to measure")
        bins = request.settings["bins"]
        edges = np.linspace(bins["low"], bins["high"], bins["count"] + 1)
        counts, _ = np.histogram(fractions, bins=edges)
        histogram = {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
            "underflow": sum(f < float(edges[0]) for f in fractions),
            "overflow": sum(f > float(edges[-1]) for f in fractions),
        }
        path = request.workdir / "histogram.json"
        path.write_text(json.dumps(histogram), encoding="utf-8")
        return EvaluationResult.valid(
            [Measurement.from_values("simulation", "ecal_barrel.halo_fraction", fractions, "1")],
            artifacts={"histogram": path},
        )
