"""Internal: the record of a candidate whose tasks all ended, checked and published.

A candidate's ``summary.json`` is written only if it can say what the candidate was evaluated
with and what its files hold: nothing the session verifies changed while it ran
(:meth:`~k4opt.execution._baseline.Baseline.verify`), no file a task made changed after it was
made (what the coordinator watched of them, :class:`~k4opt.execution._files.Observed`), and the
references only its geometry names still read what they read when it started, as for every
candidate recorded meanwhile. Then the candidate is assessed against the objectives and
constraints (:mod:`k4opt.record.assess`), its record assembled from its task results
(:mod:`k4opt.record.write`) and written. Candidates are published one after another.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from k4opt.execution._baseline import Baseline, candidate_reads, plain
from k4opt.execution._drive import _finish_plan, _Run
from k4opt.execution._files import changed_files
from k4opt.execution._work import candidate_externals
from k4opt.execution.errors import RunError
from k4opt.frozen import Json
from k4opt.jsonio import write_json
from k4opt.problem.plan import GEOMETRY, Kind
from k4opt.record.assess import Assessment, assess
from k4opt.record.results import CandidateRecord
from k4opt.record.write import SUMMARY_NAME, assemble


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What one run produced: the candidate's record, the written summary and where it is, the
    fingerprint of the geometry it was evaluated with (provenance: candidates that differ only
    in what they set elsewhere, a ddsim option say, share it) and the candidate's assessment
    against the objectives and constraints."""

    candidate: CandidateRecord
    summary: Mapping[str, object]
    run_dir: Path
    geometry_fingerprint: str
    assessment: Assessment

    @property
    def summary_path(self) -> Path:
        return self.run_dir / SUMMARY_NAME

    @property
    def events(self) -> dict[str, tuple[object, object]]:
        """``(found, requested)``: the events found in the simulation output of each sample whose
        output was inspected, and those its simulation stage asks for (``None`` for none)."""
        definition = self.summary["problem"]
        assert isinstance(definition, Mapping)
        samples = definition["definition"].get("samples", {})
        events: dict[str, tuple[object, object]] = {}
        for task in self.candidate.plan.of(Kind.STAGE):
            found = self.candidate.results[task.id].provenance.get("events_read")
            if task.id.name == "simulation" and found is not None:
                assert task.id.sample is not None
                requested = samples[task.id.sample]["simulation"].get("events")
                events[task.id.sample] = (found, requested)
        return events


def publish(run: _Run, baseline: Baseline, execution: Mapping[str, Json]) -> RunOutcome:
    """The record of the candidate ``run`` of a session that evaluates with ``baseline``,
    checked and written into its run directory; ``execution`` says how its tasks ran.

    Nothing is published, and nothing is remembered of the candidate, if what the session
    verifies changed while it ran, a file a task made is no longer what its record says, or a
    reference only its geometry names reads other files than for a candidate recorded meanwhile:
    that raises :class:`RunError`."""
    spec, source = baseline.spec, baseline.problem.source
    run_dir = run.workspace.root
    assert run_dir is not None
    record = _finish_plan(run)
    assert run.prepared is not None
    geometry = record.results[GEOMETRY]
    externals = candidate_reads(geometry)
    try:
        baseline.verify()  # the record names what the stages and evaluators read
        # in plan order, what is watched of the files each task that ran made
        watched = {t.id: run.observed[t.id] for t in record.plan.tasks if t.id in run.observed}
        changed = changed_files(watched)
        if changed:
            raise RunError(
                f"{', '.join(changed)} changed after it was made, so the record could not "
                "say what it holds"
            )
        now = candidate_externals(
            source.compact, run_dir / GEOMETRY.name, run.own_refs, baseline.externals
        )
        if plain(now) != plain(externals):
            refs = dict.fromkeys([*externals, *now])
            changed = [ref for ref in refs if now.get(ref) != externals.get(ref)]
            raise RunError(
                f"geometry: {', '.join(changed)} does not read the files it read when the "
                "candidate started"
            )
        # again, now: a candidate recorded while this one ran may have read other files
        baseline.check_read(externals)
    except RunError as exc:
        raise RunError(
            f"{exc} (while the candidate in {run_dir} ran, which has no {SUMMARY_NAME})"
        ) from None
    assessment = assess(spec, record)
    summary = assemble(
        run.workspace.id,
        baseline.path,
        spec,
        run.values,
        record,
        assessment,
        run_dir,
        baseline.taken,
        run.reference,
        baseline.process,
        run.prepared.configuration,
        execution,
    )
    try:
        write_json(run_dir / SUMMARY_NAME, summary)
    except OSError as exc:
        raise RunError(f"cannot write summary in {run_dir}: {exc}") from exc
    baseline.remember(externals)
    fingerprint = geometry.provenance["fingerprint"]
    assert isinstance(fingerprint, str)
    return RunOutcome(record, summary, run_dir, fingerprint, assessment)
