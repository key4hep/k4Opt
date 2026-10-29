"""Example evaluator ``calo.photon``: how well the ECAL measures a photon's direction, and how
many cells it fires doing so.

A complete, working evaluator to copy from. It is not part of the k4opt package; name it on the
command line::

    k4opt evaluate examples/problem/cld-ecal-granularity.yaml \\
        --evaluator examples/evaluators/calo_photon.py:CaloPhoton \\
        --param constants.ECal_cell_size=10.0 --output runs/cld-ecal-10mm

It only reads: ``requires = "simulation"`` makes k4opt run generation and ``ddsim`` first, check
that the EDM4hep file holds the requested events and hand it over. The two measurements,
``simulation`` / ``ecal_barrel.angular_residual`` and ``ecal_barrel.fired_cells``, are defined
in ``docs/evaluators.md``.
"""

import numpy as np

from k4opt.evaluation import Evaluator

# a cell fires above half the most probable energy a MIP leaves in 0.5 mm of silicon
THRESHOLD = 70e-6  # GeV, the energy unit of EDM4hep
PHOTON = 22  # PDG code


class CaloPhoton(Evaluator):
    # must match the name of an entry of evaluation.evaluators in the problem file
    name = "calo.photon"
    # must match the version of that entry
    version = "1"
    # k4opt runs generation and ddsim first and checks the event count
    requires = "simulation"
    # the settings of that entry: the calorimeter hits to read (required)
    settings = {"collection": str}
    # metric -> unit; its stage is `requires`
    metrics = {"ecal_barrel.angular_residual": "mrad", "ecal_barrel.fired_cells": "1"}

    def evaluate(self, request):
        residuals, fired_cells = [], []
        # per event: the calorimeter hits and the generated particles
        hits = self.read(request, request.settings["collection"], "SimCalorimeterHit")
        particles = self.read(request, "MCParticles", "MCParticle")
        for event_hits, event_particles in zip(hits, particles, strict=True):
            # the particle of the gun, one per event
            primaries = [p for p in event_particles if p.getGeneratorStatus() == 1]
            if len(primaries) != 1:
                self.fail(f"expected one generated particle per event, found {len(primaries)}")
            photon = primaries[0]
            # the metrics are defined for photons; any other particle would give values that
            # look valid but measure something else
            if photon.getPDG() != PHOTON:
                self.fail(f"expected a photon (PDG {PHOTON}), generated PDG {photon.getPDG()}")
            # a photon that converted in the tracker arrives as an electron and a positron
            if photon.isDecayedInTracker():
                continue
            fired = [hit for hit in event_hits if hit.getEnergy() >= THRESHOLD]
            fired_cells.append(len(fired))
            if not fired:  # nothing to take a direction from
                continue
            # measured direction: from the vertex to the energy-weighted center of the fired
            # cells (a hit sits at the center of its cell)
            energy = [hit.getEnergy() for hit in fired]
            centers = [(h.getPosition().x, h.getPosition().y, h.getPosition().z) for h in fired]
            vertex, momentum = photon.getVertex(), photon.getMomentum()
            measured = np.average(centers, axis=0, weights=energy) - (vertex.x, vertex.y, vertex.z)
            true = np.array((momentum.x, momentum.y, momentum.z))
            cosine = measured @ true / (np.linalg.norm(measured) * np.linalg.norm(true))
            residuals.append(1e3 * np.arccos(np.clip(cosine, -1.0, 1.0)))  # mrad
        # the evaluation ran, but there is nothing to measure
        if not residuals:
            self.invalid("no photon reached the calorimeter and fired a cell")
        # lists are averaged: their mean with the standard error becomes the measurement
        return {"ecal_barrel.angular_residual": residuals, "ecal_barrel.fired_cells": fired_cells}
