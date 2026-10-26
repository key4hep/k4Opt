"""The ``calo.hit_multiplicity`` evaluator: distinct cells hit in the simulated ECAL barrel.

Runs ``ddsim`` on a prepared geometry and reads the EDM4hep file it writes. The one
measurement is ``simulation`` / ``ecal_barrel.mean_hit_cells``; see ``docs/evaluation.md`` for
the definition. ``podio`` comes from the Key4hep stack and is imported only when a file is
read, so this module imports without it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from k4opt.ddsim import check_arguments, simulate
from k4opt.evaluation.errors import EvaluationContractError
from k4opt.evaluation.models import EvaluationRequest, EvaluationResult, Measurement
from k4opt.frozen import Json

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


def _readout(settings: Mapping[str, Json]) -> str:
    unknown = [key for key in settings if key not in ("readouts", "generator")]
    if unknown:
        raise EvaluationContractError(
            f"settings: unknown key(s) {unknown}; supported: ['readouts', 'generator']"
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

    ``request.inputs["geometry"]`` is the compact file to simulate, ``request.workdir`` the
    directory to write into (``simulation/`` is created in it), ``request.events`` the
    number of events (required), ``request.seed`` the random seed (optional), and
    ``request.settings`` the ``readouts`` and ``generator`` of the problem.
    """

    name = "calo.hit_multiplicity"
    version = "1"
    #: the ``(stage, metric)`` pairs a valid result reports, with their units
    metrics: Mapping[tuple[str, str], str] = MappingProxyType(
        {("simulation", "ecal_barrel.mean_hit_cells"): "1"}
    )

    def check(self, request: EvaluationRequest) -> None:
        """Raise if the request cannot be simulated (:class:`EvaluationContractError` or
        :class:`~k4opt.ddsim.DdsimError`).

        Needs no geometry and starts nothing, so a caller can run it before preparing files.
        """
        _readout(request.settings)
        if request.events is None:
            raise EvaluationContractError("events is required to simulate")
        unknown = [tool for tool in request.backend_args if tool != "ddsim"]
        if unknown:
            raise EvaluationContractError(f"backend: unsupported tool(s) {unknown}; only 'ddsim'")
        check_arguments(
            events=request.events,
            seed=request.seed,
            generator=request.settings.get("generator"),
            args=request.backend_args.get("ddsim", ()),
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.check(request)
        assert request.events is not None
        collection = _readout(request.settings)
        geometry = request.inputs.get("geometry")
        if geometry is None or not geometry.is_file():
            raise EvaluationContractError(
                f"inputs['geometry'] must be an existing compact file, got {geometry!r}"
            )
        if request.workdir is None:
            raise EvaluationContractError("workdir is required to simulate")

        run = simulate(
            compact=geometry,
            workdir=request.workdir / "simulation",
            events=request.events,
            seed=request.seed,
            generator=request.settings.get("generator"),
            args=request.backend_args.get("ddsim", ()),
        )
        artifacts = run.files
        provenance: dict[str, Json] = {
            "argv": run.argv,
            "cwd": str(run.cwd),
            "returncode": run.returncode,
            "collection": collection,
            "events_requested": request.events,
        }

        def failed(message: str) -> EvaluationResult:
            return EvaluationResult(
                "failed", artifacts=artifacts, message=message, provenance=provenance
            )

        if run.error is not None:
            return failed(run.error)
        events_file = run.output
        try:
            counts = read_cell_counts(events_file, collection)
        except _ReadError as exc:
            return failed(str(exc))
        provenance["events_read"] = len(counts)

        if not counts:
            return EvaluationResult(
                "invalid",
                artifacts=artifacts,
                provenance=provenance,
                message=f"{events_file.name} contains no events",
            )
        if len(counts) != request.events:
            return failed(
                f"incomplete simulation: {len(counts)} of {request.events} events in "
                f"{events_file.name}"
            )
        provenance["cells_min"], provenance["cells_max"] = min(counts), max(counts)
        if max(counts) == 0:
            return EvaluationResult(
                "invalid",
                artifacts=artifacts,
                provenance=provenance,
                message=f"no hits in {collection!r} in any of {len(counts)} events",
            )
        mean, error = mean_and_error(counts)
        return EvaluationResult(
            "valid",
            (Measurement("simulation", "ecal_barrel.mean_hit_cells", mean, "1", error),),
            artifacts,
            provenance=provenance,
        )
