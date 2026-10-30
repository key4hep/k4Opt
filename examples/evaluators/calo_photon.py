"""Example evaluator ``calo.photon``: how well the ECAL measures a photon's direction, and how
many cells it fires doing so.

A complete, working evaluator to copy from. It is not part of the k4opt package; name it on the
command line::

    k4opt evaluate examples/problem/cld-ecal-granularity.yaml \\
        --evaluator examples/evaluators/calo_photon.py:CaloPhoton \\
        --param constants.ECal_cell_size=10.0 --output runs/cld-ecal-10mm

It only reads: ``stage = "simulation"`` makes k4opt run the problem's stages (the gun and
``ddsim``) first, check that the EDM4hep file holds the requested events and hand it over. The
two measurements, ``simulation`` / ``ecal_barrel.angular_residual`` and
``ecal_barrel.fired_cells``, are defined in ``docs/evaluators.md``.
"""

import numpy as np

from k4opt.evaluation import Collection, Evaluator, Quantity

PHOTON = 22  # PDG code


def xyz(vector):
    """An EDM4hep vector (a position, a momentum) as a numpy array."""
    return np.array((vector.x, vector.y, vector.z))


class CaloPhoton(Evaluator):
    # must match the name of an entry of evaluators in the problem file
    name = "calo.photon"
    # must match the version of that entry
    version = "1"
    # the stage it reads: k4opt runs the gun and ddsim first and checks the event count; the
    # problem's entry says the same, stage: simulation
    stage = "simulation"
    # the collections it reads from that stage's file, by role, with the type of their elements:
    # the problem names the calorimeter hits; the generated particles have a default name
    collections = {
        "hits": Collection("SimCalorimeterHit"),
        "particles": Collection("MCParticle", default="MCParticles"),
    }
    # the settings of that entry: the threshold above which a cell fires (required). The problem
    # writes it with any energy unit; the evaluator gets it in GeV, the energy unit of EDM4hep
    settings = {"threshold": Quantity("GeV")}
    # metric -> unit; the metrics are classified under `stage`
    metrics = {"ecal_barrel.angular_residual": "mrad", "ecal_barrel.fired_cells": "1"}

    def evaluate(self, request):
        threshold = request.settings["threshold"]
        residuals, fired_cells = [], []
        # one event at a time, its hits and particles read from that same event
        for event in self.events(request):
            # the particle of the gun, one per event
            primaries = [p for p in event.particles if p.getGeneratorStatus() == 1]
            if len(primaries) != 1:
                self.fail(f"expected one generated particle per event, found {len(primaries)}")
            photon = primaries[0]
            # the metrics are defined for photons; any other particle would give values that
            # look valid but measure something else
            if photon.getPDG() != PHOTON:
                self.fail(f"expected a photon (PDG {PHOTON}), generated PDG {photon.getPDG()}")
            # its true direction; checked first, so no skipped event hides a malformed gun
            true = xyz(photon.getMomentum())
            if not np.linalg.norm(true):
                self.fail("the generated photon has no momentum, so no direction to compare with")
            # a photon that converted in the tracker arrives as an electron and a positron
            if photon.isDecayedInTracker():
                continue
            fired = [hit for hit in event.hits if hit.getEnergy() >= threshold]
            fired_cells.append(len(fired))
            energies = [hit.getEnergy() for hit in fired]
            # nothing to take a direction from: no cell fired, or (threshold 0) none has energy
            if sum(energies) <= 0:
                continue
            # measured direction: from the vertex to the energy-weighted center of the fired
            # cells (a hit sits at the center of its cell)
            center = np.average([xyz(hit.getPosition()) for hit in fired], axis=0, weights=energies)
            measured = center - xyz(photon.getVertex())
            if not np.linalg.norm(measured):  # a center at the vertex points nowhere
                continue
            cosine = measured @ true / (np.linalg.norm(measured) * np.linalg.norm(true))
            residuals.append(1e3 * np.arccos(np.clip(cosine, -1.0, 1.0)))  # mrad
        # the evaluation ran, but there is nothing to measure
        if not residuals:
            self.invalid("no photon reached the calorimeter and fired a cell")
        # lists are averaged: their mean with the standard error becomes the measurement
        return {"ecal_barrel.angular_residual": residuals, "ecal_barrel.fired_cells": fired_cells}
