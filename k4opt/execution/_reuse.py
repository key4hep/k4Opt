"""Internal: stages that are not run again, because a stage with the same signature made its output
already.

Two stages with equal signatures (:mod:`k4opt.record.signature`) were made from the same things,
so the output of one is the output the other would make: reusing it changes nothing a candidate
is evaluated with. The candidates of a study that vary only the reconstruction all simulate the
same events, and so do two samples defined alike.

:class:`Reuse` keeps, by signature, the outputs that can stand in for a stage: of the session's
own stages that ended ``done``, and of the stages that the records it is given hold
(:meth:`Reuse.index`: a study's earlier trials on a resume, or runs named by the caller). Only an
output whose SHA-256 is known is kept, and it is checked before each use: a file that changed or
went is dropped, and the stage runs. A record is published only once every task of its candidate
ended, so an output listed in one was complete; an output of a stage of this session is kept once
the stage ended ``done``, pending until its candidate is published: it is dropped if the
candidate stops or is not published (:meth:`Reuse.withdraw`), and a candidate that reused it is
handed on only after that candidate ended, and published only if it was. While a stage runs,
another with its signature waits for it (:meth:`Reuse.wait`) and then reuses its output, or runs
itself if it did not succeed.

A reused stage's result names the file where it is, outside the candidate's directory, as a file
the caller supplies is named, and says where it comes from (``reused``: the record, the stage, its
path in that run, its size and SHA-256).
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from k4opt.errors import K4OptError
from k4opt.evaluation import Artifact
from k4opt.execution._files import Made, Observed, readable_identity, stat_of
from k4opt.frozen import FrozenMap, Json
from k4opt.problem.config import parse_problem
from k4opt.problem.plan import Task
from k4opt.problem.stages import stage_output
from k4opt.record.read import SUMMARY_SCHEMA, check_summary
from k4opt.record.results import DONE, TaskResult
from k4opt.record.signature import SIGNATURE, digest
from k4opt.record.write import SUMMARY_NAME

REUSED = "reused"
"""Where a reused stage's provenance says where its output comes from."""


@dataclass(frozen=True, slots=True)
class Source:
    """An output that stands in for any stage with its signature: the file ``path`` (absolute)
    with its ``bytes`` and ``sha256``, made by the stage ``task`` (``gamma/simulation``) of the
    candidate whose record is ``record``, in the run directory ``run_dir``, and the events found
    in it."""

    path: Path
    bytes: int
    sha256: str
    record: str
    task: str
    run_dir: Path
    events_read: Json = None

    def provenance(self) -> dict[str, Json]:
        """What a reused stage's provenance says of it."""
        try:
            where = self.path.relative_to(self.run_dir).as_posix()
        except ValueError:
            where = str(self.path)
        reused: dict[str, Json] = {
            "record": self.record,
            "task": self.task,
            "run_dir": str(self.run_dir),
            "path": where,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }
        provenance: dict[str, Json] = {REUSED: FrozenMap(reused)}
        if self.events_read is not None:
            provenance["events_read"] = self.events_read
        return provenance


@dataclass(slots=True)
class _Waiting:
    """A stage waiting for another with its signature, which runs: its candidate (by key) and
    task, with whatever the coordinator keeps of it (``item``)."""

    key: Hashable
    task: Task
    item: object


@dataclass(slots=True)
class Reuse:
    """The outputs that stand in for stages, by signature, and the stages that run or wait."""

    _sources: dict[str, Source] = field(default_factory=dict)
    _running: dict[str, Hashable] = field(default_factory=dict)  # signature: candidate key
    _waiting: dict[str, list[_Waiting]] = field(default_factory=dict)
    # the signatures of the sources of this session's candidates not yet published, by record
    _unpublished: dict[str, set[str]] = field(default_factory=dict)
    _published: set[str] = field(default_factory=set)  # this session's records published

    def find(self, signature: str) -> Source | None:
        """The output that stands in for a stage with ``signature``, checked now: ``None`` if
        there is none, or if it changed or went since it was recorded (then it is forgotten)."""
        source = self._sources.get(signature)
        if source is None:
            return None
        now = readable_identity(source.path)  # hashed again only if its stat changed
        if now != {"bytes": source.bytes, "sha256": source.sha256}:
            del self._sources[signature]
            return None
        return source

    def add(self, signature: str, source: Source) -> None:
        """Keep ``source`` for the stages with ``signature``; one kept already stays."""
        self._sources.setdefault(signature, source)

    def running(self, signature: str) -> bool:
        return signature in self._running

    def start(self, signature: str, key: Hashable) -> None:
        """A stage with ``signature`` of the candidate ``key`` was handed out to run."""
        self._running[signature] = key

    def wait(self, signature: str, key: Hashable, task: Task, item: object) -> None:
        """``task`` of the candidate ``key`` waits for the stage with ``signature`` that runs."""
        self._waiting.setdefault(signature, []).append(_Waiting(key, task, item))

    def ended(self, signature: str, source: Source | None) -> list[object]:
        """The stage with ``signature`` that ran ended, with ``source`` if it made an output that
        can stand in for others; the items of the stages that waited for it, which now reuse
        ``source`` or, without one, run themselves."""
        self._running.pop(signature, None)
        if source is not None:
            self.add(signature, source)
            if self._sources.get(signature) is source:  # pending until its record is published
                self._unpublished.setdefault(source.record, set()).add(signature)
        return [waiting.item for waiting in self._waiting.pop(signature, [])]

    def pending(self, record: str) -> bool:
        """Whether ``record`` is of a candidate of this session whose outputs stand in for
        stages, but which is not published yet (and may never be)."""
        return record in self._unpublished

    def published(self, record: str) -> None:
        """The candidate whose record is ``record`` was published: its outputs stand for good."""
        self._unpublished.pop(record, None)
        self._published.add(record)

    def is_published(self, record: str) -> bool:
        """Whether this session published the record ``record``."""
        return record in self._published

    def withdraw(self, record: str) -> None:
        """The candidate whose record is ``record`` will not be published: its outputs stand in
        for no stage any more, since no record would say where they come from."""
        for signature in self._unpublished.pop(record, set()):
            source = self._sources.get(signature)
            if source is not None and source.record == record:
                del self._sources[signature]

    def forget(self, key: Hashable) -> list[object]:
        """The candidate ``key`` stopped: its stages wait no more, its outputs stand in for none
        (it will have no record), and the items of the stages of other candidates that waited
        for one of its stages, which now run themselves."""
        if isinstance(key, str):
            self.withdraw(key)
        for signature, waiting in list(self._waiting.items()):
            left = [each for each in waiting if each.key != key]
            if left:
                self._waiting[signature] = left
            else:
                del self._waiting[signature]
        released: list[object] = []
        for signature, runner in list(self._running.items()):
            if runner == key:
                released += self.ended(signature, None)
        return released

    def index(self, run_dirs: Iterable[Path]) -> int:
        """Keep the outputs of the stages that the records in ``run_dirs`` hold, each a run
        directory with a ``summary.json``; how many were kept. A stage is kept if its record
        holds up as any record is read (against the problem it records: each signature is then
        what the record says its stage was made from, and its SHA-256 that of its document) and
        its output, hashed, is still there with that hash. Records that cannot be read, of
        another schema or that do not hold up, and stages without these are left out."""
        kept = 0
        for run_dir in run_dirs:
            for signature, source in _recorded(Path(run_dir)):
                if signature not in self._sources and self.find_file(source):
                    self.add(signature, source)
                    kept += 1
        return kept

    @staticmethod
    def find_file(source: Source) -> bool:
        """Whether ``source``'s file is there with its size and SHA-256."""
        return readable_identity(source.path) == {"bytes": source.bytes, "sha256": source.sha256}


def _recorded(run_dir: Path) -> list[tuple[str, Source]]:
    """The stages of the record in ``run_dir`` whose outputs can stand in for others, with their
    signatures."""
    try:
        summary = json.loads((run_dir / SUMMARY_NAME).read_bytes())
    except (OSError, ValueError):
        return []
    if not isinstance(summary, Mapping) or summary.get("schema") != SUMMARY_SCHEMA:
        return []
    # read as any record is read, against the problem it was made for: each stage's signature is
    # then what the record says it was made from, not only a document with its own hash
    try:
        definition = summary["problem"]["definition"]
        check_summary(parse_problem(definition), summary)
    except (K4OptError, LookupError, TypeError, ValueError):
        return []
    record, samples = summary.get("id"), summary.get("samples")
    if not isinstance(record, str) or not isinstance(samples, Mapping):
        return []
    found = []
    for sample, part in samples.items():
        stages = part.get("stages") if isinstance(part, Mapping) else None
        for name, stage in (stages if isinstance(stages, Mapping) else {}).items():
            source = _source(run_dir, record, f"{sample}/{name}", stage)
            if source is not None:
                found.append(source)
    return found


def _source(run_dir: Path, record: str, task: str, stage: object) -> tuple[str, Source] | None:
    """A recorded stage's signature and output, if it has both: it ran, is ``done`` and made
    its output in the run (a stage that itself reused one is no source: the one it reused is)."""
    if not isinstance(stage, Mapping) or stage.get("status") != DONE or REUSED in stage:
        return None
    signature = stage.get(SIGNATURE)
    output = (stage.get("outputs") or {}).get(stage_output(task.partition("/")[2]))
    if not isinstance(signature, Mapping) or not isinstance(output, Mapping):
        return None
    document, sha256 = signature.get("document"), signature.get("sha256")
    if not isinstance(document, Mapping) or sha256 != digest(document):
        return None
    path, size, content = output.get("path"), output.get("bytes"), output.get("sha256")
    if not isinstance(path, str) or not isinstance(size, int) or not isinstance(content, str):
        return None
    if Path(path).is_absolute() or ".." in Path(path).parts:
        return None
    source = Source(run_dir / path, size, content, record, task, run_dir, stage.get("events_read"))
    return str(sha256), source


def made(task: Task, result: TaskResult, record: str, run_dir: Path | None) -> Source | None:
    """The output of the stage ``task`` of this session, which ended with ``result``, as a
    source for others: ``None`` unless it is ``done`` and made its output, hashed, in the run
    directory ``run_dir`` of the candidate whose record is ``record``."""
    if result.status != DONE or run_dir is None or REUSED in result.provenance:
        return None
    output = result.made.get(stage_output(task.id.name))
    if output is None or output.sha256 is None:
        return None
    events = result.provenance.get("events_read")
    return Source(output.path, output.bytes, output.sha256, record, task.id.label, run_dir, events)


def reused(task: Task, source: Source) -> tuple[TaskResult, Observed]:
    """The result of the stage ``task`` whose output ``source`` stands in for it, and what the
    coordinator watches of that file until the candidate ends, as of a file the stage made."""
    name = stage_output(task.id.name)
    artifact = Artifact(f"{task.id.label}#{name}", source.path, source.bytes, source.sha256)
    stat = stat_of(source.path)
    watched = Observed(FrozenMap({} if stat is None else {name: Made(artifact, stat)}))
    result = TaskResult(
        task.id,
        DONE,
        None,
        source.path,
        provenance=FrozenMap(source.provenance()),
        elapsed_seconds=0.0,
    )
    return result, watched
