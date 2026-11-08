"""What a rerun of a study needs, what of it is available, and what that allows: its *status*.

:func:`requirements` reads what the study kept (:mod:`.keep`) and what its records say its stages
used, and lists each thing a rerun needs (:class:`Requirement`): the build of the stack (``base``),
each local build and directory of local code (``package``, ``code``), each environment of installed
distributions (``environment``), each reconstruction's job directory (``steering``), the geometry
(``geometry``), each input
the stack does not hold (``input``), k4opt's code (``k4opt``) and each evaluator (``evaluator``).
Each says what identifies it, which stages used it and how its availability was checked, with the
date (:class:`Check`): a check is true when it was made, never a promise for later.

For a local build two facts are kept apart, since a patch does not rebuild the same binary: its
source (``rebuildable``: the state of its checkout is kept against a known revision, and its link to
the binary is ``verified`` or ``likely``) and the binary that ran, of which only the fingerprint is
kept (``original: fingerprint-only``). So a study with a local build that ran can at best be
*rebuilt*: a rerun runs other bytes, and is compared by its results.

:attr:`Readiness.status` condenses them; it never says *reproduced*: that takes a rerun, which k4opt
does not record yet. The records it reads are every trial's of the study, at every level of a
nested optimization, and for a stage whose output was reused, the record of the stage that made it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from k4opt.frozen import Json, thaw
from k4opt.software.errors import SoftwareError
from k4opt.software.identity import Build
from k4opt.software.keep import (
    DIRECTORY,
    INPUTS,
    SCHEMA,
    USED,
    USED_SCHEMA,
    Store,
    candidate_geometry,
    commitment,
    declared_inputs,
    kept_inputs,
    known_salt,
)
from k4opt.software.observe import _LISTED, identity

type Kind = Literal[
    "base", "package", "code", "environment", "steering", "geometry", "input", "k4opt", "evaluator"
]
type CheckState = Literal["available", "gone", "altered", "not-checked"]

ALL_AVAILABLE = "Not rerun · all available"
REBUILD = "Not rerun · rebuild needed"
BLOCKED = "Not rerun · blocked"
INCOMPLETE = "Not rerun · record incomplete"
UNVERIFIED = "Not rerun · availability not established"


@dataclass(frozen=True, slots=True)
class Check:
    """Whether something was there as it was, when (``at``) and how it was checked."""

    at: str
    state: CheckState
    how: str

    def to_dict(self) -> dict[str, str]:
        return {"at": self.at, "state": self.state, "how": self.how}


@dataclass(frozen=True, slots=True)
class Requirement:
    """One thing a rerun needs.

    ``id`` names it within the study (``base``, ``package:ddmarlinpandora``,
    ``input:7c1e09a2b4f3``: an id that holds between checks), ``kind``
    is what it is and ``label`` what to call it; ``identified`` says what identifies it, for a
    person; ``used_by`` the stages that used it, as their records say, and ``trials`` the trials
    whose records say so (their run directories, relative to the study at the top). ``source`` is
    how its source
    is known (a revision, the changes against it, the link to its binary, whether it can be built
    again, ``rebuildable``) and ``ran`` what of it executed (``original``: ``on-cvmfs`` for the
    stack's, ``fingerprint-only`` for a local build's, whose binaries are not kept). ``checks`` are
    how its availability was checked; ``notes`` what else a reader must know (a local build that
    did not run, a source changed after its build). ``unused`` says that it is a local build every
    stage of every record was observed not to read: listed, and needed by no rerun of them.
    ``entry`` is the study's private record of it, for what publishes a part of it; never shown."""

    id: str
    kind: Kind
    label: str
    identified: str
    used_by: tuple[str, ...] = ()
    trials: tuple[str, ...] = ()
    source: Mapping[str, Json] | None = None
    ran: Mapping[str, Json] | None = None
    checks: tuple[Check, ...] = ()
    notes: tuple[str, ...] = ()
    unused: bool = False
    entry: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def available(self) -> bool:
        """Whether it was checked, and every check found it as it was: one not checked is not
        known to be available."""
        return bool(self.checks) and all(check.state == "available" for check in self.checks)

    @property
    def unproven(self) -> bool:
        """Whether its kept source is not known to be what its binaries were built from: changed
        after they were built or between sessions (``stale``), or with files not kept
        (``incomplete``)."""
        source = self.source or {}
        return bool(source.get("stale") or source.get("incomplete"))

    @property
    def missing(self) -> bool:
        """Whether a check found it gone or altered."""
        return any(check.state in ("gone", "altered") for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "identified": self.identified,
            "used_by": list(self.used_by),
            "trials": list(self.trials),
            "source": None if self.source is None else thaw(self.source),
            "ran": None if self.ran is None else thaw(self.ran),
            "checks": [check.to_dict() for check in self.checks],
            "notes": list(self.notes),
            "unused": self.unused,
        }


@dataclass(frozen=True, slots=True)
class Readiness:
    """What a rerun of a study needs (``requirements``), what its records leave out
    (``gaps``: stages not observed completely or that used the network, a study that kept nothing,
    trials whose evidence was not kept), and when that was checked (``at``)."""

    requirements: tuple[Requirement, ...]
    gaps: tuple[str, ...]
    at: str
    problems: tuple[str, ...] = field(default=())
    altered: tuple[str, ...] = field(default=())
    """The records of what the study kept that are gone or not what was written (a
    ``software-*.json`` a trial ran with, or whose content is not that of its name; an unreadable
    ``inputs.json`` or ``used.json``): nothing they say is used."""
    public_gaps: tuple[str, ...] = field(default=())
    """``gaps``, in the same order, as they can be published: without a path or a host."""
    store: Store | None = field(default=None, repr=False, compare=False)
    """Where the bytes the study kept are."""
    ran: Mapping[tuple[str, str], set[str]] = field(default_factory=dict, repr=False, compare=False)
    """What the observed stages read of local builds: by path and identity, the stages."""

    @property
    def status(self) -> str:
        """The headline (see the module), the first that holds: *blocked* if something is gone or
        altered, or a local build that ran cannot be built again; *record incomplete* if what it
        needs is not fully known (``gaps``: a stage not observed, or that used the network, a local
        build that ran whose kept source is not known to be what it was built from); *availability
        not established* if
        something could not be checked (an environment's distributions, an unrecognized stack, a
        k4opt commit no remote was seen to have); *rebuild needed* if a local build ran, whose
        original binary is not kept, both said if both hold; *all available* otherwise, with the
        date of the checks. Never *reproduced*. What no rerun needs (``unused``) changes none."""
        date = self.at[:10]
        needed = [r for r in self.requirements if not r.unused]
        if (
            self.altered
            or any(r.missing for r in needed)
            or any(
                r.kind == "package"
                and r.used_by
                and not (r.source or {}).get("rebuildable")
                and not r.unproven
                for r in needed
            )
        ):
            return BLOCKED
        if self.gaps:
            return INCOMPLETE
        rebuild = any(r.kind == "package" and r.used_by for r in needed)
        unverified = not all(r.available for r in needed)
        if rebuild and unverified:
            return f"{REBUILD} · {UNVERIFIED.removeprefix('Not rerun · ')}"
        if unverified:
            return UNVERIFIED
        if rebuild:
            return REBUILD
        return f"{ALL_AVAILABLE} on {date}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def kept_directory(study: Path) -> Path:
    """Where the study in ``study`` keeps its software: its own :data:`~.keep.DIRECTORY`, or that
    of the study at the top of the nested optimization it belongs to (``<top>/inner/trial-N``)."""
    top = study
    while top.parent.name == "inner" and (top.parent.parent / "study.json").is_file():
        top = top.parent.parent
    return top / DIRECTORY


def _documents(directory: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The states the study kept, by file name, in the order they were taken (each file is
    written once), and the names of those that are not what was written: a file is named by the
    SHA-256 of its content."""
    found: dict[str, dict[str, Any]] = {}
    altered: list[str] = []
    paths = sorted(directory.glob("software-*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    for path in paths:
        try:
            data = path.read_bytes()
            document = json.loads(data)
        except (OSError, ValueError):
            altered.append(path.name)
            continue
        named = path.name.removeprefix("software-").removesuffix(".json")
        if hashlib.sha256(data).hexdigest()[:16] != named or not isinstance(document, dict):
            altered.append(path.name)
        elif document.get("schema") == SCHEMA:
            found[path.name] = document
    return found, altered


def _used(directory: Path) -> dict[str, Any] | None:
    """The study's :data:`~.keep.USED`, ``None`` if it has none; raises :class:`SoftwareError` if
    it is not one."""
    path = directory / USED
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise SoftwareError(f"{path} cannot be read: {exc}") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema") != USED_SCHEMA
        or not all(isinstance(document.get(key), dict) for key in ("footprints", "trials"))
        or not all(
            isinstance(entry, dict) and isinstance(entry.get("files"), dict)
            for entry in document["footprints"].values()
        )
    ):
        raise SoftwareError(f"{path} is not of {USED_SCHEMA}")
    return document


def _stages(summary: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    samples = summary.get("samples")
    for sample, entry in (samples if isinstance(samples, Mapping) else {}).items():
        stages = entry.get("stages") if isinstance(entry, Mapping) else None
        for name, stage in (stages if isinstance(stages, Mapping) else {}).items():
            if isinstance(stage, Mapping):
                yield f"{sample}/{name}", stage


def _footprint(stage: Mapping[str, Any]) -> object:
    """The SHA-256 of the footprint ``stage`` depends on, as its record holds it: its own, or that
    of the stage whose output it reused; ``None`` if it has none (it was not observed)."""

    def within(*keys: str) -> Mapping[str, Any]:
        found: object = stage
        for key in keys:
            found = found.get(key) if isinstance(found, Mapping) else None
        return found if isinstance(found, Mapping) else {}

    if within("logs").get("footprint") is not None:
        return within("observed", "footprint").get("sha256")
    return within("reused", "footprint").get("sha256")


def _load(path: Path) -> Mapping[str, Any] | None:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return summary if isinstance(summary, Mapping) else None


def records(top: Path) -> list[tuple[str, Mapping[str, Any]]]:
    """The record of every trial of the study in ``top`` and of every study of a level below it
    (``inner/trial-N``), each by its run directory relative to ``top``."""
    found: list[tuple[str, Mapping[str, Any]]] = []

    def walk(study: Path) -> None:
        for run in sorted(study.glob("trial-*")):
            summary = _load(run / "summary.json")
            if summary is not None:
                found.append((run.relative_to(top).as_posix(), summary))
        for inner in sorted((study / "inner").glob("trial-*")):
            if (inner / "study.json").is_file():
                walk(inner)

    walk(top)
    return found


_HOPS = 8  # how many reused outputs a stage's evidence is followed through


def _evidence(
    label: str, stage: Mapping[str, Any], cache: dict[str, Mapping[str, Any] | None]
) -> tuple[Mapping[str, Any] | None, tuple[str, str] | None]:
    """What the stage ``label`` used (its ``observed``) and, if that is not fully known, why, as
    a gap (:class:`_Gaps`): a stage that ran its program says how it was observed; one whose output
    was reused is what the stage that made it says, in that stage's record, wherever it is."""
    for _ in range(_HOPS):
        reused = stage.get("reused")
        if not isinstance(reused, Mapping):
            break
        where = str(reused.get("run_dir"))
        if where not in cache:
            cache[where] = _load(Path(where) / "summary.json")
        source = cache[where]
        made = dict(_stages(source)).get(str(reused.get("task"))) if source else None
        if made is None:
            return None, (
                f"{label} reused an output of {where}, whose record does not say what its stage "
                "used",
                f"{label} reused an output of another run, whose record does not say what its "
                "stage used",
            )
        stage = made
    observed = stage.get("observed")
    if "argv" not in stage:  # it ran no program
        return (observed if isinstance(observed, Mapping) else None), None
    if not isinstance(observed, Mapping):
        said = f"{label} ran its program, and its record does not say what it used"
        return None, (said, said)
    if observed.get("level") != "files" or observed.get("complete") is not True:
        said = f"{label} was not observed completely"
        return observed, (said, said)
    return observed, None


class _Gaps:
    """What the records leave out: each said for the author, and as it can be published, without a
    path or a host (the two the same when it holds neither)."""

    def __init__(self) -> None:
        self.said: dict[str, str] = {}

    def add(self, text: str, public: str | None = None) -> None:
        self.said.setdefault(text, text if public is None else public)

    def __bool__(self) -> bool:
        return bool(self.said)


def requirements(
    study: Path,
    identity: Mapping[str, Any],
    summaries: Iterable[Mapping[str, Any]] | None = None,
) -> Readiness:
    """What a rerun of the study in ``study`` needs, checked now: from what it kept, what the
    records of its trials (:func:`records`: every level of a nested optimization; or
    ``summaries``, if given) say their stages used, and the study's ``identity``
    (``study.json``)."""
    at = _now()
    directory = kept_directory(study)
    store = Store(directory)
    kept, altered = _documents(directory)
    try:
        evidence = _used(directory)
    except SoftwareError:
        evidence = None
        altered.append(USED)
    used = evidence["trials"] if evidence else {}  # trial: the state its session kept
    footprints = evidence["footprints"] if evidence else {}  # digest: the files a stage read
    named = {str(name) for name in used.values()}
    altered += sorted(named - kept.keys() - set(altered))  # a state a trial ran with, gone
    # the states the trials ran with; all of them, where that is not recorded
    documents = [document for name, document in kept.items() if name in named] or list(
        kept.values()
    )
    if summaries is None:
        found_records = records(directory.parent)
    else:
        found_records = [(f"record {n}", summary) for n, summary in enumerate(summaries)]
    gaps = _Gaps()
    used_local: dict[str, set[tuple[str, str]]] = {}  # entry path: (trial, stage) that read it
    stages_seen: set[str] = set()
    trials_seen: set[str] = set()
    cache: dict[str, Mapping[str, Any] | None] = {}
    network: dict[str, set[str]] = {}  # stage: what it reached
    unidentified: dict[str, set[str]] = {}  # stage: what it read that could not be identified
    lost: set[str] = set()  # stages whose footprint the study should have noted, and did not
    expected: set[tuple[str, str]] = set()  # the inputs the records name, as they were read
    listed: dict[str, str] = {}  # each directory a stage listed: its names then
    reused_from: dict[str, str] = {}  # stage: the run directory it reused an output of
    for trial, summary in found_records:
        trials_seen.add(trial)
        expected.update((path, then) for path, then, _, _ in candidate_geometry(summary))
        for label, stage in _stages(summary):
            observed, gap = _evidence(label, stage, cache)
            if gap is not None:
                gaps.add(*gap)
            if isinstance(_footprint(stage), str) and _footprint(stage) not in footprints:
                lost.add(label)
            if observed is None:
                continue
            stages_seen.add(label)
            for path in cast(Sequence[str], observed.get("local") or ()):
                used_local.setdefault(path, set()).add((trial, label))
            if observed.get("network"):
                network.setdefault(label, set()).update(map(str, observed["network"]))
            read = observed.get("undeclared")
            for path, then in (read if isinstance(read, Mapping) else {}).items():
                if not isinstance(then, str):
                    unidentified.setdefault(label, set()).add(str(path))
                elif then.startswith(_LISTED):
                    listed.setdefault(str(path), then)
                else:
                    expected.add((str(path), then))
            reused = stage.get("reused")
            if isinstance(reused, Mapping):
                reused_from[label] = str(reused.get("run_dir"))
    if not found_records:
        gaps.add("no trial has a record yet: what its stages use is not known")
    if summaries is None and found_records and USED not in altered:
        if evidence is None:
            gaps.add(f"{USED} is gone: what the trials ran with and read is not known")
        else:
            untold = sorted(trials_seen - used.keys())
            if untold:
                gaps.add(f"nothing says what {', '.join(untold)} ran with")
    if evidence is not None:
        for label in sorted(lost):
            gaps.add(f"what {label} read of local builds was not noted")
    # a local build no stage read from did not run, if every stage of every record says so
    known = bool(found_records) and not gaps and USED not in altered
    for label, reached in sorted(network.items()):
        gaps.add(
            f"{label} reached the network ({', '.join(sorted(reached))}): what it received is not "
            "kept",
            f"{label} reached the network: what it received is not kept",
        )
    for label, paths in sorted(unidentified.items()):
        gaps.add(
            f"{label} read {', '.join(sorted(paths))}, which could not be identified",
            f"{label} read {len(paths)} files that could not be identified",
        )
    if not documents:
        gaps.add("the study kept nothing of what it ran with")
        return Readiness(
            (), tuple(gaps.said), at, altered=tuple(altered), public_gaps=tuple(gaps.said.values())
        )
    # one study, one software: its build and system are the last state's. Each place a local build
    # or code was in is its own origin, as the last state that has it says: a build may have been
    # copied elsewhere between sessions, and its first trials read the first copy. A source that
    # changed between the states of one place is stale
    document = documents[-1]
    latest: dict[str, Mapping[str, Any]] = {}
    trees: dict[str, set[object]] = {}
    for each in documents:
        for origin in each.get("origins") or []:
            git = origin.get("git") or {}
            latest[str(origin.get("prefix"))] = origin
            trees.setdefault(str(origin.get("prefix")), set()).add(git.get("tree"))
    moved = {prefix for prefix, seen in trees.items() if len(seen) > 1}
    # what no state describes was read through an output reused from another study, whose own
    # record of it stays there: said, and no build here can be taken as not having run
    described = {str(entry) for origin in latest.values() for entry in origin.get("entries") or ()}
    for path in sorted(set(used_local) - described):
        where = sorted({reused_from[s] for _, s in used_local[path] if s in reused_from})
        known = False
        gaps.add(
            f"{path} was read, and this study did not keep what it is"
            + (f": see the study of {', '.join(where)}, whose output it reused" if where else ""),
            f"{os.path.basename(path)} was read, and this study did not keep what it is",
        )
    ran = _ran(footprints)
    ours = {each.get("k4opt") for each in documents}
    found: list[Requirement] = []
    base = _base(document, at, sorted(stages_seen), sorted(trials_seen))
    found.append(base)
    names = Counter(_name(origin) for origin in latest.values())
    seen: Counter[str] = Counter()
    salted = known_salt(directory)
    for index, (prefix, origin) in enumerate(latest.items(), start=1):
        name = _name(origin)
        seen[name] += 1
        # its id holds whatever else is installed and says nothing of where it is: the study's
        # salt with its place; without a salt, its rank among those of its name
        token = (
            hashlib.sha256(f"{salted}\0{prefix}".encode()).hexdigest()[:8]
            if salted
            else str(seen[name])
        )
        key = f"{name}:{token}"
        if names[name] > 1:  # the author is told where each is; its id, which is public, is not
            name = f"{name} at {prefix}"
        found.append(
            _origin(origin, name, key, index, store, at, used_local, ran, ours, known, moved)
        )
    found += [_steering(entry, store, at, base) for entry in document.get("steering") or []]
    if isinstance(document.get("geometry"), Mapping):
        found.append(_geometry(document["geometry"], store, at, base))
    try:
        inputs = sorted(
            kept_inputs(directory), key=lambda e: (str(e.get("path")), str(e.get("identity")))
        )
    except SoftwareError:
        inputs = []
        altered.append(INPUTS)
    versions: dict[str, int] = {}
    for entry in inputs:
        versions[str(entry.get("path"))] = versions.get(str(entry.get("path")), 0) + 1
    for path, count in versions.items():
        if count > 1:
            gaps.add(
                f"{path} was read as {count} different files by the trials of the study",
                f"{os.path.basename(path)} was read as {count} different files by the trials of "
                "the study",
            )
    found += [_input(index, entry, store, at) for index, entry in enumerate(inputs, start=1)]
    found += _listings(listed, at)
    if INPUTS not in altered:
        have = {(str(entry.get("path")), str(entry.get("identity"))) for entry in inputs}
        unlisted = sorted(
            {path for path, _ in (expected | declared_inputs(identity).keys()) - have}
        )
        if unlisted:
            gaps.add(
                f"{INPUTS} does not list what the study read: {', '.join(unlisted)}",
                f"{INPUTS} does not list {len(unlisted)} files the study read",
            )
    found.append(_k4opt(identity, document, store, at))
    for name, entry in (document.get("evaluators") or {}).items():
        found.append(_evaluator(name, entry, identity, store, at))
    for r in found:
        if r.kind in ("package", "code") and r.used_by and r.unproven:
            gaps.add(f"{r.label}: {_unproven(r)}", f"{os.path.basename(r.label)}: {_unproven(r)}")
    problems = tuple(
        dict.fromkeys(str(p) for each in documents for p in each.get("problems") or ())
    )
    return Readiness(
        tuple(found),
        tuple(gaps.said),
        at,
        problems,
        tuple(altered),
        tuple(gaps.said.values()),
        store,
        ran,
    )


def _unproven(requirement: Requirement) -> str:
    source = requirement.source or {}
    if source.get("incomplete"):
        return (
            "untracked files of its checkout were too large to keep, so what is kept of its "
            "source is not all of it"
        )
    return (
        "its kept source changed after the binaries that ran were built, so it is not known to "
        "be what they were built from"
    )


def _ran(footprints: Mapping[str, Any]) -> dict[tuple[str, str], set[str]]:
    """Each file of a local entry that an observed stage read (``footprints``, of
    :data:`~.keep.USED`), by where it was read and its identity: the stages. The place counts as
    much as the bytes: two installations may hold the same library, of which one ran."""
    found: dict[tuple[str, str], set[str]] = {}
    for entry in footprints.values():
        if not isinstance(entry, Mapping):
            continue
        for path, then in (entry.get("files") or {}).items():
            if isinstance(then, str):
                found.setdefault((path, then), set()).add(str(entry.get("stage")))
    return found


def _executed(
    prefix: str, binary: Mapping[str, Any], ran: Mapping[tuple[str, str], set[str]]
) -> set[str]:
    """The stages that read ``binary`` of the build at ``prefix`` from there, with its bytes."""
    identity_ = f"sha256:{binary.get('sha256')}"
    roots = {prefix.rstrip("/"), os.path.realpath(prefix)}
    file = str(binary.get("file"))
    places = {os.path.join(root, file) for root in roots} if not os.path.isabs(file) else {file}
    return {
        stage
        for (path, then), stages in ran.items()
        if then == identity_
        and (path in places or any(path.startswith(f"{root}/") for root in roots))
        and os.path.basename(path).split(".so")[0] == os.path.basename(file).split(".so")[0]
        for stage in stages
    }


def _base(
    document: Mapping[str, Any], at: str, stages: list[str], trials: list[str]
) -> Requirement:
    software = document.get("software") or {}
    release = document.get("release")
    try:
        build = Build.from_dict(software.get("build"))
    except Exception:  # a kept file of another k4opt: say so, do not stop
        check = Check(at, "not-checked", "what the study kept of its build cannot be read")
        return Requirement("base", "base", "stack", "not readable", checks=(check,))
    if not isinstance(release, Mapping):
        check = Check(at, "not-checked", "an unrecognized stack: k4opt cannot say where it is")
        return Requirement(
            "base", "base", build.label, build.label, tuple(stages), tuple(trials), checks=(check,)
        )
    setup = str(release.get("setup"))
    now = Build.of(setup)
    if now == build:
        check = Check(at, "available", f"{setup} holds this build")
    elif now.kind == "unrecognized" and not now.digests.get("setup"):
        check = Check(at, "gone", f"{setup} cannot be read")
    else:
        check = Check(at, "gone", f"{setup} holds another build ({now.label})")
    notes: tuple[str, ...] = ()
    if "/nightlies/" in setup or "-nightlies." in setup:
        notes = ("a nightly: a later build overwrites its slot, and this one is then gone",)
    heads = len(release.get("packages") or {})  # the HEAD ones: the rest are in the manifests
    source = {"release": release.get("build", {}).get("release"), "head_packages": heads}
    return Requirement(
        "base",
        "base",
        f"Key4hep {build.label}",
        f"{build.platform}, {release.get('count')} packages, {heads} built from "
        "a repository's head, with their revisions",
        tuple(stages),
        tuple(trials),
        source=source,
        ran={"original": "on-cvmfs"},
        checks=(check,),
        notes=notes,
    )


def _git_line(git: Mapping[str, Any]) -> str:
    base = str(git.get("base") or "")[:7] or "no public commit"
    parts = [base]
    if git.get("ahead"):
        parts.append(f"{git['ahead']} local commits")
    uncommitted = git.get("uncommitted") or {}
    if uncommitted.get("files"):
        parts.append(f"{uncommitted['files']} uncommitted files")
    if git.get("untracked"):
        parts.append(f"{git['untracked']} untracked")
    return " + ".join(parts)


def _patch_check(git: Mapping[str, Any], store: Store, at: str) -> Check:
    if not git.get("changed"):
        but = ", but for untracked files too large to keep" if git.get("large") else ""
        return Check(at, "available", f"nothing differs from its base commit{but}")
    patch = git.get("patch")
    if not isinstance(patch, str):
        return Check(at, "gone", "its changes were not kept (too large)")
    state = store.state(patch)
    return Check(at, state, f"its changes, kept as {patch[:12]}")


def _name(origin: Mapping[str, Any]) -> str:
    return str(origin.get("package") or os.path.basename(str(origin.get("prefix")).rstrip("/")))


def _kept_files(files: Mapping[str, Any], store: Store, at: str) -> list[Check]:
    """Whether each of ``files`` (of a record: its path, its SHA-256, what it is kept as) is kept
    as it was."""
    checks = []
    for relative, entry in files.items():
        entry = entry if isinstance(entry, Mapping) else {}
        checks.append(_kept_check(store, at, str(relative), entry.get("kept"), entry.get("sha256")))
    return checks


def _origin(
    origin: Mapping[str, Any],
    name: str,
    key: str,
    index: int,
    store: Store,
    at: str,
    used_local: Mapping[str, set[tuple[str, str]]],
    ran: Mapping[tuple[str, str], set[str]],
    ours: Collection[object],
    known: bool,
    moved: Collection[str] = (),
) -> Requirement:
    kind = origin.get("kind")
    entries = [str(e) for e in origin.get("entries") or ()]
    reads = {read for entry in entries for read in used_local.get(entry, ())}
    used = sorted({label for _, label in reads})
    trials = tuple(sorted({trial for trial, _ in reads}))
    git = origin.get("git")
    notes: list[str] = []
    checks: list[Check] = []
    problem = origin.get("problem") if entries else None  # code that could be loaded, not kept
    if problem:
        checks.append(Check(at, "gone", f"its state was not kept: {problem}"))
    source: dict[str, Json] = {}
    if isinstance(git, Mapping):
        checks.append(_patch_check(git, store, at))
        if git.get("base") and not git.get("remote"):
            checks.append(Check(at, "not-checked", "no remote is known to fetch its base commit"))
        source = {
            "base": git.get("base"),
            "base_is": git.get("base_is"),
            "head": git.get("head"),
            "changed": len(git.get("changed") or ()),
            "patch": git.get("patch"),
            "incomplete": bool(git.get("large")),
        }
        if git.get("fetch"):
            notes.append(
                f"the nightly's revision {origin.get('revision')} is not in the checkout: "
                "diffing against it needs a fetch"
            )
        if git.get("large"):
            notes.append(
                f"untracked files too large to keep: {', '.join(git['large'])}; a rebuild from "
                "what is kept lacks them"
            )
        if git.get("base_is") == "none":
            notes.append("no remote has its history: its every file is in the kept changes")
    if kind == "build":
        binaries = origin.get("binaries") or []
        prefix = str(origin.get("prefix"))
        by = {str(b.get("file")): _executed(prefix, b, ran) for b in binaries}
        executed = [b for b in binaries if by[str(b.get("file"))]]
        used = sorted({*used, *(s for stages in by.values() for s in stages)})
        unused = bool(entries) and not used and known
        if unused:
            notes.append("changed locally — not used: no observed stage read from it")
        stale = bool(origin.get("stale")) or origin.get("prefix") in moved
        if origin.get("stale"):
            notes.append("its source changed after its binaries were built: not what they ran")
        elif stale:
            notes.append("its source changed between sessions of the study: not what they ran")
        rebuildable = bool(origin.get("rebuildable")) and not stale and not git_large(git)
        installed = origin.get("files") or {}
        checks += _kept_files(installed, store, at)
        configuration = origin.get("configuration")
        if isinstance(configuration, str):
            checks.append(_kept_check(store, at, "its CMake configuration", configuration, None))
        else:
            rebuildable = False
            notes.append("its CMake configuration was not kept: a rebuild could use other options")
        source |= {
            "link": origin.get("link"),
            "rebuildable": rebuildable,
            "stale": stale,
            "configuration": configuration,
            "installed": len(installed),
        }
        identified = (
            f"{_git_line(git) if isinstance(git, Mapping) else 'source unknown'} "
            f"({origin.get('link')} link)"
        )
        return Requirement(
            f"package:{key}",
            "package",
            name,
            identified,
            tuple(used),
            trials,
            source=source,
            ran={
                "original": "fingerprint-only",
                "files": tuple(
                    {k: b.get(k) for k in ("file", "sha256", "build_id")} for b in executed
                ),
            },
            checks=tuple(checks),
            notes=tuple(notes),
            unused=unused,
            entry=origin,
        )
    if kind == "environment":
        distributions = origin.get("distributions") or []
        unlisted = origin.get("unlisted") or []
        check = Check(at, "not-checked", "installed distributions, not checked against an index")
        said = f"{len(distributions)} distributions"
        notes_: tuple[str, ...] = ()
        if unlisted:
            said += f", {len(unlisted)} not from an index"
            notes_ = (
                f"distributions not installed from an index: {len(unlisted)}, from the author",
            )
        return Requirement(
            f"environment:{key}",
            "environment",
            str(origin.get("prefix")),
            said,
            tuple(used),
            trials,
            source={"distributions": tuple(distributions), "unlisted": tuple(unlisted)},
            checks=(check,),
            notes=notes_,
        )
    loose = origin.get("binaries") or []
    if loose and not isinstance(git, Mapping):  # binaries of no build and no checkout
        prefix = str(origin.get("prefix"))
        stages = sorted({*used, *(s for b in loose for s in _executed(prefix, b, ran))})
        return Requirement(
            f"package:{key}",
            "package",
            prefix,
            f"{len(loose)} binaries of no known build",
            tuple(stages),
            trials,
            source={"rebuildable": False, "link": "none"},
            ran={"original": "fingerprint-only", "files": tuple(thaw(loose))},  # type: ignore[arg-type]
            checks=(Check(at, "available", "identified by SHA-256 and build-id, not kept"),),
            notes=("binary only — source unknown: its binaries are identified, never kept",),
            unused=bool(entries) and not stages and known,
            entry=origin,
        )
    if isinstance(git, Mapping):
        identified = _git_line(git)
        ignored = origin.get("files") or {}
        checks += _kept_files(ignored, store, at)
        if ignored:
            identified += f" + {len(ignored)} ignored files, kept"
        if origin.get("binaries"):
            names = ", ".join(str(b.get("file")) for b in origin["binaries"])
            checks.append(Check(at, "not-checked", f"ignored binaries, never kept: {names}"))
        if git.get("checkout") in ours:
            notes.append("k4opt's own checkout")
    else:
        files = origin.get("files") or {}
        checks += _kept_files(files, store, at)
        identified = f"{len(files)} files in no checkout, kept"
        if not files and not problem:
            checks.append(Check(at, "available", "nothing in it can be loaded: nothing to keep"))
    return Requirement(
        f"code:{key}",
        "code",
        str(origin.get("prefix")),
        identified,
        tuple(used),
        trials,
        source=source or None,
        checks=tuple(checks),
        notes=tuple(notes),
        entry=origin,
    )


def _kept_check(store: Store, at: str, what: str, kept: object, sha256: object) -> Check:
    """Whether ``what``, kept as ``kept`` (``None`` if it was not), is there as it was, and is the
    file the record says (``sha256``, if it says one): a record pointing at other bytes is
    altered, whatever those bytes are."""
    if not isinstance(kept, str):
        return Check(at, "gone", f"{what} was not kept")
    if isinstance(sha256, str) and kept != sha256:
        return Check(at, "altered", f"{what}: the record names other bytes than it was read as")
    return Check(at, store.state(kept), f"{what}, kept")


def git_large(git: object) -> bool:
    """Whether a kept checkout state left out untracked files too large to keep."""
    return isinstance(git, Mapping) and bool(git.get("large"))


def _steering(entry: Mapping[str, Any], store: Store, at: str, base: Requirement) -> Requirement:
    stage = str(entry.get("stage"))
    package, revision = entry.get("package"), entry.get("revision")
    changed, added = entry.get("changed") or {}, entry.get("added") or {}
    removed = entry.get("removed") or []
    checks = []
    for name, file in {**changed, **added}.items():
        file = file if isinstance(file, Mapping) else {}
        checks.append(_kept_check(store, at, str(name), file.get("kept"), file.get("sha256")))
    if not checks:  # nothing of its own: the stack's files, as available as the build
        state = base.checks[0].state if base.checks else "not-checked"
        checks.append(Check(at, state, "the files of the stack's package, with the build"))
    if entry.get("original") == entry.get("directory"):
        identified = f"the stack's own {package}"
    elif package is None:
        identified = f"{len(added)} files of no package of the stack, kept"
    else:
        counts = [f"{len(changed)} changed", f"{len(added)} added", f"{len(removed)} removed"]
        identified = f"a copy of {package} {revision or ''}".rstrip() + f": {', '.join(counts)}"
    return Requirement(
        f"steering:{stage}",
        "steering",
        stage,
        identified,
        (stage,),
        source={"package": package, "revision": revision, "removed": tuple(removed)},
        checks=tuple(checks),
        entry=entry,
    )


def _geometry(entry: Mapping[str, Any], store: Store, at: str, base: Requirement) -> Requirement:
    files = entry.get("files")
    if files is None:
        checks = [Check(at, "gone", "its files could not be read when the session opened")]
    else:
        checks = _kept_files(files, store, at)
    identified = f"{len(files or ())} files, kept"
    if entry.get("stack"):  # as available as the build
        state = base.checks[0].state if base.checks else "not-checked"
        checks.append(Check(at, state, "the files of the stack, with the build"))
        identified = f"the stack's, and {identified}" if files else "the stack's"
    compact = str(entry.get("compact"))
    return Requirement(
        "geometry", "geometry", compact, identified, checks=tuple(checks), entry=entry
    )


def _listings(listed: Mapping[str, str], at: str) -> list[Requirement]:
    """Each directory a stage listed whose names changed since; only those, and never as gone: a
    listing is what any Python program makes of its own search path."""
    return [
        Requirement(
            f"listing:{path}",
            "input",
            os.path.basename(path),
            "a directory a stage listed",
            checks=(Check(at, "not-checked", f"the names in {path} changed since"),),
        )
        for path, then in sorted(listed.items())
        if identity(path) != then
    ]


def _stable(entry: Mapping[str, Any]) -> str:
    """What names an input between checks and publications, revealing nothing of it: its
    commitment, or the digest of its random salt."""
    said = entry.get("commitment")
    if not isinstance(said, str):
        said = hashlib.sha256(str(entry.get("salt")).encode()).hexdigest()
    return said[:12]


def _input(index: int, entry: Mapping[str, Any], store: Store, at: str) -> Requirement:
    path = str(entry.get("path"))
    kept = entry.get("kept")
    then = str(entry.get("identity"))
    sha256 = then.removeprefix("sha256:") if then.startswith("sha256:") else None
    salt, promised = entry.get("salt"), entry.get("commitment")
    if sha256 is not None and not _commits(salt, promised, sha256):
        check = Check(at, "altered", "its record's commitment is not that of the file it names")
    elif isinstance(kept, str):
        check = _kept_check(store, at, "the file", kept, sha256)
    elif sha256 is None and not then.startswith("cvmfs:"):  # a pipe still there vouches for nothing
        check = Check(
            at, "not-checked", f"not kept ({entry.get('why')}); what it gave is not known"
        )
    elif identity(path) == then:
        check = Check(at, "available", f"not kept ({entry.get('why')}); {path} is unchanged")
    else:
        check = Check(at, "gone", f"not kept ({entry.get('why')}); {path} changed or went")
    through = f", read through {entry['variable']}" if entry.get("variable") else ""
    size = f"{entry['bytes']} bytes" if isinstance(entry.get("bytes"), int) else "size unknown"
    return Requirement(
        f"input:{_stable(entry)}",
        "input",
        os.path.basename(path),
        f"{entry.get('origin')} input, {size}{through}",
        tuple(entry.get("stages") or ()),
        tuple(entry.get("trials") or ()),
        source={"path": path, "commitment": entry.get("commitment")},
        checks=(check,),
        entry=entry,
    )


def _commits(salt: object, promised: object, sha256: str) -> bool:
    """Whether ``promised`` is the commitment of the file ``sha256`` under ``salt``; not if they
    are no such thing."""
    try:
        return isinstance(salt, str) and promised == commitment(salt, sha256)
    except ValueError:
        return False


def _k4opt(
    identity: Mapping[str, Any], document: Mapping[str, Any], store: Store, at: str
) -> Requirement:
    k4opt = identity.get("k4opt") or {}
    git = k4opt.get("git") or {}
    commit = str(git.get("commit") or "")[:7]
    dirty = git.get("dirty")
    label = f"k4opt {k4opt.get('version')}"
    if not commit:
        check = Check(at, "not-checked", "no checkout tracks it: a released version")
        return Requirement("k4opt", "k4opt", label, "no git revision", checks=(check,))
    git_kept = next(
        (
            o["git"]
            for o in document.get("origins") or []
            if isinstance(o.get("git"), Mapping)
            and o["git"].get("checkout") == document.get("k4opt")
        ),
        None,
    )
    if git_kept is None:
        check = (
            Check(at, "gone", "its uncommitted changes were not kept")
            if dirty
            else Check(
                at, "not-checked", "its checkout was not kept: no remote was seen to have it"
            )
        )
    elif git_kept.get("base_is") == "public" and git_kept.get("remote"):
        check = _patch_check(git_kept, store, at)  # a public commit, and its changes as kept
        if check.state == "available":
            public = str(git_kept["base"])[:7]
            how = f"a remote branch had {public} when the session opened; {check.how}"
            check = Check(at, "available", how)
    else:
        check = Check(at, "not-checked", "no remote of its checkout was seen to have its commit")
    return Requirement(
        "k4opt",
        "k4opt",
        label,
        f"{commit}{' with uncommitted changes' if dirty else ''}",
        source={"commit": git.get("commit"), "sha256": k4opt.get("sha256")},
        checks=(check,),
        entry=git_kept,
    )


def _evaluator(
    name: str, entry: Mapping[str, Any], identity: Mapping[str, Any], store: Store, at: str
) -> Requirement:
    version = ((identity.get("implementations") or {}).get(name) or {}).get("version")
    files = entry.get("files") or {}
    checks = tuple(
        _kept_check(store, at, str(file), kept.get("kept"), kept.get("sha256"))
        if isinstance(kept, Mapping)
        else Check(at, "altered", f"{file}: its record is not one k4opt writes")
        for file, kept in files.items()
    )
    if not checks:
        checks = (Check(at, "gone", "its file was not kept"),)
    return Requirement(
        f"evaluator:{name}",
        "evaluator",
        f"{name} v{version}",
        f"{len(files)} files, kept",
        source={"sha256": entry.get("sha256")},
        checks=checks,
        entry=entry,
    )


__all__ = [
    "ALL_AVAILABLE",
    "BLOCKED",
    "INCOMPLETE",
    "REBUILD",
    "UNVERIFIED",
    "Check",
    "Readiness",
    "Requirement",
    "kept_directory",
    "records",
    "requirements",
]
