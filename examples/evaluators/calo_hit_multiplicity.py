"""Example evaluator ``calo.hit_multiplicity``: distinct cells hit in the simulated ECAL barrel.

A complete, working evaluator to copy from. It is not part of the k4opt package; name it on the
command line::

    k4opt evaluate examples/problem/cld-ecal-hit-count.yaml \\
        --evaluator examples/evaluators/calo_hit_multiplicity.py:CaloHitMultiplicity \\
        --param constants.ECal_cell_size=6.0 --output runs/cld-ecal-6mm

It only reads. ``requires = "simulation"`` asks k4opt to run generation and ``ddsim`` first and
to hand over the EDM4hep file as ``request.inputs["simulation"]``; the evaluator starts no tool
and handles none of their failures. The one measurement is ``simulation`` /
``ecal_barrel.mean_hit_cells``; see ``docs/evaluators.md`` for the definition. ``podio`` comes
from the Key4hep stack and is imported only when a file is read, so this module imports without
it. The metric is a technical observable, not a detector-performance measure.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from k4opt.evaluation import (
    EvaluationContractError,
    EvaluationRequest,
    EvaluationResult,
    Measurement,
)

HIT_TYPE = "edm4hep::SimCalorimeterHit"


class _ReadError(Exception):
    """The simulation output cannot be used; the message is shown as the failure reason."""


def mean_and_error(counts: list[int]) -> tuple[float, float | None]:
    """Sample mean of ``counts`` and its standard error, ``None`` for a single value.

    The standard error is ``s / sqrt(N)`` with the sample standard deviation ``s``
    (``ddof=1``).
    """
    mean = statistics.fmean(counts)
    if len(counts) < 2:
        return mean, None
    return mean, statistics.stdev(counts) / math.sqrt(len(counts))


def _open_events(path: Path) -> Iterator[Any]:
    """The events (podio frames) of an EDM4hep file; imports the Key4hep ``podio``."""
    try:
        from podio.reading import get_reader
    except ImportError as exc:
        raise _ReadError(
            f"podio is not available (is the Key4hep environment set up?): {exc}"
        ) from exc
    reader = get_reader(str(path))
    if "events" not in reader.categories:
        raise _ReadError(f"{path.name} has no 'events' category")
    yield from reader.get("events")


def _type_name(name: Any) -> str:
    """The text of a podio type name, which is a C++ ``string_view`` (not a ``str``) in PyROOT."""
    return name if isinstance(name, str) else str(name.data())[: int(name.size())]


def read_cell_counts(path: Path, collection: str) -> list[int]:
    """Number of distinct ``cellID`` values of ``collection`` in each event of ``path``.

    An existing but empty collection counts as 0. A collection that is missing in an event,
    has another element type than ``edm4hep::SimCalorimeterHit`` or lacks ``cellID`` raises
    :class:`_ReadError`; hits are never counted instead of cells.
    """
    counts: list[int] = []
    try:
        for index, event in enumerate(_open_events(path)):
            if collection not in event.getAvailableCollections():
                raise _ReadError(
                    f"collection {collection!r} is missing in event {index} of {path.name}"
                )
            hits = event.get(collection)
            element = _type_name(hits.getValueTypeName())
            if element != HIT_TYPE:
                raise _ReadError(
                    f"collection {collection!r} holds {element!r}, expected {HIT_TYPE!r}"
                )
            try:
                cells = {hit.getCellID() for hit in hits}
            except AttributeError as exc:
                raise _ReadError(f"hits of {collection!r} have no cellID: {exc}") from exc
            counts.append(len(cells))
    except _ReadError:
        raise
    except Exception as exc:
        # The Key4hep reader raises C++ exceptions (std::runtime_error, ROOT's RException)
        # that are not subclasses of Python's RuntimeError, so no narrower type is enough. They
        # mean an unusable file, not a bug here.
        raise _ReadError(f"cannot read {path.name}: {type(exc).__name__}: {exc}") from exc
    return counts


def _readout(settings: Mapping[str, Any]) -> str:
    unknown = [key for key in settings if key != "readouts"]
    if unknown:
        raise EvaluationContractError(
            f"settings: unknown key(s) {unknown}; supported: ['readouts']"
        )
    readouts = settings.get("readouts")
    if not isinstance(readouts, Mapping) or list(readouts) != ["ecal_barrel"]:
        raise EvaluationContractError(
            "settings.readouts: must map exactly the role 'ecal_barrel' to a collection name"
        )
    collection = readouts["ecal_barrel"]
    if not isinstance(collection, str) or not collection.strip():
        raise EvaluationContractError(
            f"settings.readouts.ecal_barrel: must be a collection name, got {collection!r}"
        )
    return collection


class CaloHitMultiplicity:
    """Mean number of distinct ECAL barrel cells hit per simulated event.

    Reads ``request.inputs["simulation"]``, the EDM4hep file of the simulation stage it
    ``requires``; ``request.settings["readouts"]`` names the collection. It runs no tool.
    """

    name = "calo.hit_multiplicity"
    version = "1"
    requires = "simulation"
    #: the ``(stage, metric)`` pairs a valid result reports, with their units
    metrics: Mapping[tuple[str, str], str] = MappingProxyType(
        {("simulation", "ecal_barrel.mean_hit_cells"): "1"}
    )

    def check(self, request: EvaluationRequest) -> None:
        """Raise :class:`EvaluationContractError` if the settings cannot be evaluated."""
        _readout(request.settings)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        collection = _readout(request.settings)
        events_file = request.inputs.get("simulation")
        if events_file is None or not events_file.is_file():
            raise EvaluationContractError(
                f"inputs['simulation'] must be an existing EDM4hep file, got {events_file!r}"
            )
        provenance: dict[str, Any] = {
            "collection": collection,
            "events_requested": request.events,
        }
        try:
            counts = read_cell_counts(events_file, collection)
        except _ReadError as exc:
            return EvaluationResult.failed(str(exc), provenance=provenance)
        provenance["events_read"] = len(counts)

        if not counts:
            return EvaluationResult.invalid(
                f"{events_file.name} contains no events", provenance=provenance
            )
        if request.events is not None and len(counts) != request.events:
            return EvaluationResult.failed(
                f"incomplete simulation: {len(counts)} of {request.events} events in "
                f"{events_file.name}",
                provenance=provenance,
            )
        provenance["cells_min"], provenance["cells_max"] = min(counts), max(counts)
        if max(counts) == 0:
            return EvaluationResult.invalid(
                f"no hits in {collection!r} in any of {len(counts)} events", provenance=provenance
            )
        mean, error = mean_and_error(counts)
        return EvaluationResult.valid(
            [Measurement("simulation", "ecal_barrel.mean_hit_cells", mean, "1", error)],
            provenance=provenance,
        )
