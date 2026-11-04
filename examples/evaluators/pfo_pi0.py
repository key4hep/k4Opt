"""Example evaluator ``pfo.pi0``: how often the particle flow reconstructs each of the two photons
of a neutral pion as a photon of its own, and how many photons it finds.

A complete, working evaluator that reads a reconstruction. It is not part of the k4opt package;
name it on the command line::

    k4opt evaluate examples/problem/cld-ecal-pi0-reco.yaml \\
        --evaluator examples/evaluators/pfo_pi0.py:PfoPi0 \\
        --evaluator examples/evaluators/ecal_channels.py:EcalChannels \\
        --param ecal_cell_size=10.0 --output runs/cld-pi0-reco-10mm

``stage = "reconstruction"`` makes k4opt run the problem's stages (the gun, ``ddsim`` and the
reconstruction, ``k4run``) first, check that the reconstruction holds every simulated event and
hand its EDM4hep file over. The particle flow (Pandora, in CLD's reconstruction) makes the
reconstructed particles; the reconstruction keeps the generated particles in the same file, and
links each reconstructed particle to the generated ones whose energy it holds. The measurements,
``reconstruction`` / ``pi0.resolved`` and ``pi0.photons``, are defined in ``docs/evaluators.md``.
"""

from k4opt.evaluation import Collection, Evaluator, Quantity

PHOTON, PI0 = 22, 111  # PDG codes


def key(particle):
    """What identifies a particle within its event: its collection and its index in it."""
    identity = particle.getObjectID()
    return identity.collectionID, identity.index


def calorimeter_share(weight):
    """The fraction of a reconstructed particle's calorimeter energy that a generated particle
    deposited, as CLD's RecoMCTruthLinker encodes it in the link's weight:
    10000 * int(1000 * calorimeter share) + int(1000 * track share)."""
    return (int(weight) // 10000) / 1000


def origin(particle, photons):
    """The photon of ``photons`` (by key) that ``particle`` descends from, ``None`` if none: the
    showers' electrons that the simulation kept descend from their photon."""
    seen = set()
    while key(particle) not in photons:
        if key(particle) in seen or not len(particle.getParents()):
            return None
        seen.add(key(particle))
        particle = particle.getParents()[0]
    return key(particle)


class PfoPi0(Evaluator):
    name = "pfo.pi0"
    version = "2"
    stage = "reconstruction"
    # the particle-flow objects the problem names; the generated particles and the links between
    # the two, which CLD's reconstruction writes under these names
    collections = {
        "pfos": Collection("ReconstructedParticle"),
        "particles": Collection("MCParticle", default="MCParticles"),
        "links": Collection("Link<ReconstructedParticle,MCParticle>", default="RecoMCTruthLink"),
    }
    # a photon below this energy, generated or reconstructed, is not one of the pair asked about
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
            # the Dalitz decay (1.2 %) has no second photon, a photon that converted in the tracker
            # reaches the ECAL as an electron and a positron, and a photon below the threshold
            # cannot be found above it: none of these is the question of separation
            if (
                len(photons) != 2
                or any(p.isDecayedInTracker() for p in photons)
                or any(p.getEnergy() < threshold for p in photons)
            ):
                continue
            wanted = {key(p) for p in photons}
            found = [p for p in event.pfos if p.getPDG() == PHOTON and p.getEnergy() >= threshold]
            # the generated photon that gave each reconstructed photon most of its energy
            shares = {}
            for link in event.links:
                source = origin(link.getTo(), wanted)
                if source is not None:
                    by = shares.setdefault(key(link.getFrom()), {})
                    by[source] = by.get(source, 0.0) + calorimeter_share(link.getWeight())
            main = {
                max(shares[key(p)].items(), key=lambda item: item[1])[0]
                for p in found
                if shares.get(key(p))
            }
            # resolved: each generated photon is the main origin of a reconstructed photon of its
            # own; two fragments of one photon do not count, however many there are
            resolved.append(1.0 if main == wanted else 0.0)
            photons_found.append(len(found))
        if not resolved:
            self.invalid("no neutral pion decayed into two photons that reached the calorimeter")
        # lists are averaged: their mean with the standard error becomes the measurement
        return {"pi0.resolved": resolved, "pi0.photons": photons_found}
