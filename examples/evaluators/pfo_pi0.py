"""Example evaluator ``pfo.pi0``: how often the particle flow resolves the two photons of a
neutral pion, and how many photons it finds.

A complete, working evaluator that reads a reconstruction. It is not part of the k4opt package;
name it on the command line::

    k4opt evaluate examples/problem/cld-ecal-pi0-reco.yaml \\
        --evaluator examples/evaluators/pfo_pi0.py:PfoPi0 \\
        --evaluator examples/evaluators/ecal_channels.py:EcalChannels \\
        --param ecal_cell_size=10.0 --output runs/cld-pi0-reco-10mm

``stage = "reconstruction"`` makes k4opt run the problem's stages (the gun, ``ddsim`` and the
reconstruction, ``k4run``) first, check that the reconstruction holds every simulated event and
hand its EDM4hep file over. The particle flow (Pandora, in CLD's reconstruction) makes the
reconstructed particles; the generated ones are kept in the same file. The measurements,
``reconstruction`` / ``pi0.resolved`` and ``pi0.photons``, are defined in
``docs/evaluators.md``.
"""

from k4opt.evaluation import Collection, Evaluator, Quantity

PHOTON, PI0 = 22, 111  # PDG codes


class PfoPi0(Evaluator):
    name = "pfo.pi0"
    version = "1"
    stage = "reconstruction"
    # the particle-flow objects the problem names, and the generated particles, which the
    # reconstruction keeps under their default name
    collections = {
        "pfos": Collection("ReconstructedParticle"),
        "particles": Collection("MCParticle", default="MCParticles"),
    }
    # a photon below this energy is noise or a fragment of a shower, not one of the pair
    settings = {"min_energy": Quantity("GeV")}
    metrics = {"pi0.resolved": "1", "pi0.photons": "1"}

    def evaluate(self, request):
        threshold = request.settings["min_energy"]
        resolved, photons_found = [], []
        for event in self.events(request):
            primaries = [p for p in event.particles if p.getGeneratorStatus() == 1]
            if len(primaries) != 1 or primaries[0].getPDG() != PI0:
                self.fail("expected one generated neutral pion per event")
            photons = [d for d in primaries[0].getDaughters() if d.getPDG() == PHOTON]
            # the Dalitz decay (1.2 %) has no second photon, and a photon that converted in the
            # tracker reaches the ECAL as an electron and a positron: neither is the question
            if len(photons) != 2 or any(p.isDecayedInTracker() for p in photons):
                continue
            found = [p for p in event.pfos if p.getPDG() == PHOTON and p.getEnergy() >= threshold]
            # two photons or more: the particle flow saw both (a split shower adds a third)
            resolved.append(1.0 if len(found) >= 2 else 0.0)
            photons_found.append(len(found))
        if not resolved:
            self.invalid("no neutral pion decayed into two photons that reached the calorimeter")
        # lists are averaged: their mean with the standard error becomes the measurement
        return {"pi0.resolved": resolved, "pi0.photons": photons_found}
