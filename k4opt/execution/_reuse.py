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
went is skipped (the next one with its signature is tried), and the stage runs if none holds
up. A record is published only once every task of its candidate
ended, so an output listed in one was complete; an output of a stage of this session is kept once
the stage ended ``done``, pending until its candidate is published: it is dropped if the
candidate stops or is not published (:meth:`Reuse.withdraw`), and a candidate that reused it is
handed on only after that candidate ended, and published only if it was. While a stage runs
(:meth:`Reuse.start`), another with its signature waits for it, parked by the scheduler with no
slot of its own (:mod:`k4opt.execution._drive`), and then reuses its output, or runs itself if it
did not succeed: :class:`Reuse` knows which signatures run, not who waits for them.

Where the records are is found apart from what is kept of them: :func:`run_directories` finds the
run directories below the directories it is given, at any depth (the trials of every level of a
nested study), and :meth:`Reuse.index` reads each record once. An output a record lists is hashed
only when a stage would reuse it, never before (:meth:`Reuse.find`).

A reused stage's result names the file where it is, outside the candidate's directory, as a file
the caller supplies is named, and says where it comes from (``reused``: the record, the stage, its
path in that run, its size and SHA-256).

An output recorded by an earlier session serves only while what made it would still find what it
found: its stage's footprint (:mod:`k4opt.software.observe`) must hold, each file it read the same,
each directory the same names, no path it failed to find there now. That is checked once per
session, the first time the output would serve; from then on its undeclared inputs are watched as
those of a stage of the session are, and a change to one stops the session. An output of another
study (:meth:`Reuse.index` with ``across``) needs more: a complete footprint, observed at the
``files`` level, and no network connection, since what is not known is taken to differ. With equal
signatures both ran on the same software, its local builds included. A refused output is skipped, as
one that changed is, and the stage that runs instead says why (:meth:`Reuse.refusal`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from k4opt.errors import K4OptError
from k4opt.evaluation import Artifact
from k4opt.execution._files import Made, Observed, snapshot
from k4opt.frozen import FrozenMap, Json
from k4opt.hashing import Stat, sha256_of
from k4opt.problem.config import parse_problem
from k4opt.problem.plan import Task
from k4opt.problem.stages import stage_output
from k4opt.record.read import SUMMARY_SCHEMA, check_summary
from k4opt.record.results import DONE, TaskResult
from k4opt.record.signature import SIGNATURE, digest
from k4opt.record.write import SUMMARY_NAME
from k4opt.software import SoftwareError
from k4opt.software.observe import load

REUSED = "reused"
"""Where a reused stage's provenance says where its output comes from."""


@dataclass(frozen=True, slots=True)
class Source:
    """An output that stands in for any stage with its signature: the file ``path`` (absolute)
    with its ``bytes`` and ``sha256``, made by the stage ``task`` (``gamma/simulation``) of the
    candidate whose record is ``record``, in the run directory ``run_dir``, and the events found
    in it. One that a record of an earlier session holds says how its stage was ``observed``
    (as the record says) and where its footprint is (``footprint``: the file and its SHA-256),
    and whether it comes from another study (``across``)."""

    path: Path
    bytes: int
    sha256: str
    record: str
    task: str
    run_dir: Path
    events_read: Json = None
    observed: Json = None
    footprint: tuple[Path, str] | None = None
    recorded: bool = False
    across: bool = False

    def refused(self) -> str | None:
        """Why this output may not stand in for a stage now, ``None`` if it may: an output of an
        earlier session whose footprint no longer holds, or, from another study, one without a
        complete footprint or that used the network."""
        if not self.recorded:
            return None
        observed = self.observed if isinstance(self.observed, Mapping) else {}
        if self.across:
            if observed.get("level") != "files" or observed.get("complete") is not True:
                return "its stage was not observed at the files level, so what it read is not known"
            if observed.get("network"):
                return "its stage used the network, which no footprint can vouch for"
        if self.footprint is None:
            return "its footprint is missing" if self.across else None
        path, sha256 = self.footprint
        try:
            if sha256_of(path) != sha256:
                return f"its footprint {path} is not the file its record names"
            changes = load(path).changes()
        except (OSError, SoftwareError) as exc:
            return f"its footprint cannot be read: {exc}"
        if changes:
            return f"what its stage read differs now: {'; '.join(changes)}"
        return None

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
            # from another study: admitted because its footprint held (Source.refused)
            "by_footprint": self.across,
        }
        undeclared = self.observed.get("undeclared") if isinstance(self.observed, Mapping) else None
        if undeclared:  # what the output depends on: watched as what a stage of the session read
            reused["undeclared"] = undeclared
        provenance: dict[str, Json] = {REUSED: FrozenMap(reused)}
        if self.events_read is not None:
            provenance["events_read"] = self.events_read
        return provenance


@dataclass(slots=True)
class Reuse:
    """The outputs that stand in for stages, by signature, and the signatures of the stages that
    run."""

    # every output recorded with a signature, by its file, in the order found: the first that
    # holds up serves
    _sources: dict[str, dict[Path, Source]] = field(default_factory=dict)
    _running: dict[str, Hashable] = field(default_factory=dict)  # signature: candidate key
    # the signatures of the sources of this session's candidates not yet published, by record
    _unpublished: dict[str, set[str]] = field(default_factory=dict)
    _published: set[str] = field(default_factory=set)  # this session's records published
    _read: set[Path] = field(default_factory=set)  # where the records read are, symlinks resolved
    # why each output of an earlier session may not serve, None if it may: checked once
    _verdicts: dict[Path, str | None] = field(default_factory=dict)
    _refusals: dict[str, str] = field(default_factory=dict)  # signature: why its outputs did not

    def find(self, signature: str) -> tuple[Source, Stat] | None:
        """The output that stands in for a stage with ``signature``, checked now, with the stat
        it was checked at: watched from that stat (:func:`reused`), a file replaced since the
        check is told apart. The first, in the order they were kept, that is there with the size
        and SHA-256 recorded, ``None`` if none is. One that is not is skipped, not forgotten: it
        is checked again at the next use (a stat, and a hash only if its stat changed), so a file
        that was out of reach for a while (a network mount) serves again once it is back."""
        for source in self._sources.get(signature, {}).values():
            try:
                stat, now = snapshot(source.path)  # hashed again only if its stat changed
            except OSError:
                continue
            if now != {"bytes": source.bytes, "sha256": source.sha256}:
                continue
            if source.path not in self._verdicts:
                self._verdicts[source.path] = source.refused()
            why = self._verdicts[source.path]
            if why is None:
                return source, stat
            self._refusals[signature] = f"{source.task} of {source.run_dir}: {why}"
        return None

    def refusal(self, signature: str) -> str | None:
        """Why an output recorded with ``signature`` did not stand in for a stage, ``None`` if
        none was refused (:meth:`Source.refused`)."""
        return self._refusals.get(signature)

    def add(self, signature: str, source: Source) -> bool:
        """Keep ``source`` for the stages with ``signature``, after those kept already; whether
        it was not kept yet (by its file)."""
        sources = self._sources.setdefault(signature, {})
        if source.path in sources:
            return False
        sources[source.path] = source
        return True

    def running(self, signature: str) -> bool:
        return signature in self._running

    def start(self, signature: str, key: Hashable) -> None:
        """A stage with ``signature`` of the candidate ``key`` was handed out to run."""
        self._running[signature] = key

    def ended(self, signature: str, source: Source | None) -> None:
        """The stage with ``signature`` that ran ended, with ``source`` if it made an output that
        can stand in for others, pending until its record is published."""
        self._running.pop(signature, None)
        if source is not None and self.add(signature, source):
            self._unpublished.setdefault(source.record, set()).add(signature)

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
            sources = self._sources.get(signature, {})
            kept = {path: each for path, each in sources.items() if each.record != record}
            if kept:
                self._sources[signature] = kept
            else:
                self._sources.pop(signature, None)

    def forget(self, key: Hashable) -> list[str]:
        """The candidate ``key`` stopped: its outputs stand in for none (it will have no record),
        and the stages of the signatures it ran run no more; those signatures, whose waiting
        stages now run themselves."""
        if isinstance(key, str):
            self.withdraw(key)
        stopped = [signature for signature, runner in self._running.items() if runner == key]
        for signature in stopped:
            del self._running[signature]
        return stopped

    def index(self, run_dirs: Iterable[Path], across: bool = True) -> int:
        """Keep the outputs of the stages that the records in ``run_dirs`` hold, each a run
        directory with a ``summary.json``; how many were kept, not yet checked. ``across`` says
        they are of another study, which an output serves only with what its footprint shows
        (:meth:`Source.refused`); not across, the records are of the study a session resumes,
        made with its software as the resume checked. A stage is kept
        if its record holds up as any record is read (against the problem it records: each
        signature is then what the record says its stage was made from, and its SHA-256 that of
        its document); its output is hashed only when a stage would reuse it, and serves only if
        it is still there with that hash (:meth:`find`). Records that cannot be read, of another
        schema or that do not hold up, and stages without these are left out. A record read
        once is not read again, whatever path reaches it (a symlink, or the directory it leads
        to): a published record never changes. Its outputs are kept where the first path that
        reached it puts them."""
        kept = 0
        for run_dir in run_dirs:
            path = Path(run_dir)
            real = Path(os.path.realpath(path))
            if real in self._read or not (path / SUMMARY_NAME).is_file():
                continue
            self._read.add(real)
            for signature, source in _recorded(path):
                kept += self.add(signature, replace(source, across=across))
        return kept


def run_directories(directory: Path) -> list[Path]:
    """The run directories in ``directory``: itself, or those below it at any depth (the trials of
    a study, and of each study of a level below it), in path order. A run directory holds the
    summary of a k4opt run (:func:`is_run`); nothing below one is looked at, but a file of that
    name that is no such summary (another program's) hides nothing below it. Symlinks are
    followed, and each directory is looked at once, wherever it is reached from."""
    if is_run(directory / SUMMARY_NAME):
        return [directory]
    found, seen = [], set()
    for parent, children, files in os.walk(directory, followlinks=True):
        real = os.path.realpath(parent)
        if real in seen:  # reached already, through another link (or a link to itself)
            children.clear()
            continue
        seen.add(real)
        if SUMMARY_NAME in files and is_run(Path(parent) / SUMMARY_NAME):
            found.append(Path(parent))
            children.clear()  # a run directory holds no other
        children.sort()
    return sorted(found)


def is_run(path: Path) -> bool:
    """Whether ``path`` is the summary of a k4opt run: a JSON object of its schema. A file of
    that name that is not one (unreadable, another program's) makes no run directory."""
    try:
        if not path.is_file():
            return False
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return False
    return isinstance(data, Mapping) and data.get("schema") == SUMMARY_SCHEMA


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
    if not _inside(path):
        return None
    observed = stage.get("observed")
    footprint = None
    if isinstance(observed, Mapping):
        file = (stage.get("logs") or {}).get("footprint")
        held = (observed.get("footprint") or {}).get("sha256")
        if isinstance(file, str) and _inside(file) and isinstance(held, str):
            footprint = (run_dir / file, held)
    source = Source(
        run_dir / path,
        size,
        content,
        record,
        task,
        run_dir,
        stage.get("events_read"),
        observed,
        footprint,
        recorded=True,
    )
    return str(sha256), source


def _inside(path: str) -> bool:
    """Whether ``path`` names a file inside the run directory it is relative to."""
    return not Path(path).is_absolute() and ".." not in Path(path).parts


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


def reused(task: Task, source: Source, stat: Stat) -> tuple[TaskResult, Observed]:
    """The result of the stage ``task`` whose output ``source`` stands in for it, and what the
    coordinator watches of that file until the candidate ends, as of a file the stage made: from
    ``stat``, at which :meth:`Reuse.find` checked its hash, never one taken later, which could be
    that of another version of the file."""
    name = stage_output(task.id.name)
    artifact = Artifact(f"{task.id.label}#{name}", source.path, source.bytes, source.sha256)
    watched = Observed(FrozenMap({name: Made(artifact, stat)}))
    result = TaskResult(
        task.id,
        DONE,
        None,
        source.path,
        provenance=FrozenMap(source.provenance()),
        elapsed_seconds=0.0,
    )
    return result, watched
