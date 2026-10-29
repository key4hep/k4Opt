"""Example evaluator ``calo.hit_multiplicity``: distinct cells hit in the simulated ECAL barrel.

A complete, working evaluator to copy from. It is not part of the k4opt package; name it on the
command line::

    k4opt evaluate examples/problem/cld-ecal-hit-count.yaml \\
        --evaluator examples/evaluators/calo_hit_multiplicity.py:CaloHitMultiplicity \\
        --param constants.ECal_cell_size=6.0 --output runs/cld-ecal-6mm

It only reads. ``requires = "simulation"`` asks k4opt to run generation and ``ddsim`` first, to
check that the EDM4hep file holds the requested events and to hand it over as
``request.inputs["simulation"]``. Whatever is wrong with the file is an
``EvaluationFailed`` raised by ``read_collection``; k4opt turns it into a ``failed`` result.
The one measurement is ``simulation`` / ``ecal_barrel.mean_hit_cells``; see
``docs/evaluators.md`` for the definition. The metric is a technical observable, not a
detector-performance measure.
"""

from __future__ import annotations

from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationInvalid,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
    read_collection,
)


class CaloHitMultiplicity:
    """Mean number of distinct ECAL barrel cells hit per simulated event.

    Reads ``request.inputs["simulation"]``, the EDM4hep file of the simulation stage it
    ``requires``; ``request.settings["collection"]`` names the collection.
    """

    name = "calo.hit_multiplicity"
    version = "1"
    requires = "simulation"
    #: the ``(stage, metric)`` pairs a valid result reports, with their units
    metrics = {("simulation", "ecal_barrel.mean_hit_cells"): "1"}

    def check(self, request: EvaluationRequest) -> None:
        """Called by k4opt before anything runs: reject settings that cannot be evaluated."""
        collection = request.settings.get("collection")
        if not isinstance(collection, str) or not collection.strip() or len(request.settings) != 1:
            raise EvaluationContractError("settings: expected exactly 'collection: <name>'")

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        collection = request.settings["collection"]
        events = read_collection(
            request.inputs["simulation"], collection, "edm4hep::SimCalorimeterHit"
        )
        cells_per_event = [len({hit.getCellID() for hit in hits}) for hits in events]
        if max(cells_per_event) == 0:
            raise EvaluationInvalid(f"no hits in {collection!r} in any event")
        return EvaluationResult.valid(
            [
                Measurement.from_samples(
                    "simulation", "ecal_barrel.mean_hit_cells", cells_per_event, "1"
                )
            ]
        )
