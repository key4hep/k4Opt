"""What a rerun of a study needs, what of it is available, and what that allows: its *status*.

:func:`requirements` reads what the study kept (:mod:`.keep`) and what its records say its stages
used, and lists each thing a rerun needs (:class:`Requirement`): the build of the stack (``base``),
each local build and directory of local code (``package``, ``code``), each environment of installed
distributions (``environment``), each reconstruction's job directory (``steering``), each input
the stack does not hold (``input``), k4opt's code (``k4opt``) and each evaluator (``evaluator``).
Each says what identifies it, which stages used it and how its availability was checked, with the
date (:class:`Check`): a check is true when it was made, never a promise for later.

For a local build two facts are kept apart, since a patch does not rebuild the same binary: its
source (``rebuildable``: the state of its checkout is kept against a known revision, and its link to
the binary is ``verified`` or ``likely``) and the binary that ran, of which only the fingerprint is
kept (``original: fingerprint-only``). So a study with a local build that ran can at best be
*rebuilt*: a rerun runs other bytes, and is compared by its results.

:func:`status` condenses them; it never says *reproduced*: that takes a rerun, which k4opt does not
record yet.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from k4opt.frozen import Json, thaw
from k4opt.software.identity import Build
from k4opt.software.keep import DIRECTORY, INPUTS, SCHEMA, USED, Store
from k4opt.software.observe import identity

type Kind = Literal[
    "base", "package", "code", "environment", "steering", "input", "k4opt", "evaluator"
]
type CheckState = Literal["available", "gone", "altered", "not-checked"]

ALL_AVAILABLE = "Not rerun · all available"
REBUILD = "Not rerun · rebuild needed"
BLOCKED = "Not rerun · blocked"
INCOMPLETE = "Not rerun · record incomplete"


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

    ``id`` names it within the study (``base``, ``package:ddmarlinpandora``, ``input:2``), ``kind``
    is what it is and ``label`` what to call it; ``identified`` says what identifies it, for a
    person; ``used_by`` the stages that used it, as their records say. ``source`` is how its source
    is known (a revision, the changes against it, the link to its binary, whether it can be built
    again, ``rebuildable``) and ``ran`` what of it executed (``original``: ``on-cvmfs`` for the
    stack's, ``fingerprint-only`` for a local build's, whose binaries are not kept). ``checks`` are
    how its availability was checked; ``notes`` what else a reader must know (a local build that
    did not run, a source changed after its build)."""

    id: str
    kind: Kind
    label: str
    identified: str
    used_by: tuple[str, ...] = ()
    source: Mapping[str, Json] | None = None
    ran: Mapping[str, Json] | None = None
    checks: tuple[Check, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Whether every check found it as it was."""
        return all(check.state == "available" for check in self.checks)

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
            "source": None if self.source is None else thaw(self.source),
            "ran": None if self.ran is None else thaw(self.ran),
            "checks": [check.to_dict() for check in self.checks],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class Readiness:
    """What a rerun of a study needs (``requirements``), what its records leave out
    (``gaps``: stages not observed completely, a study that kept nothing), and when that was
    checked (``at``)."""

    requirements: tuple[Requirement, ...]
    gaps: tuple[str, ...]
    at: str
    problems: tuple[str, ...] = field(default=())

    @property
    def status(self) -> str:
        """The headline (see the module): *blocked* if something is gone or altered, or a local
        build that ran cannot be built again; *record incomplete* if what it needs is not fully
        known; *rebuild needed* if a local build ran, whose original binary is not kept; *all
        available* otherwise, with the date of the checks. Never *reproduced*."""
        date = self.at[:10]
        if any(r.missing for r in self.requirements) or any(
            r.kind == "package" and r.used_by and not (r.source or {}).get("rebuildable")
            for r in self.requirements
        ):
            return BLOCKED
        if self.gaps:
            return INCOMPLETE
        if any(r.kind == "package" and r.used_by for r in self.requirements):
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


def _documents(directory: Path) -> list[dict[str, Any]]:
    found = []
    for path in sorted(directory.glob("software-*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and document.get("schema") == SCHEMA:
            found.append(document)
    return found


def _read(path: Path, key: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    value = document.get(key) if isinstance(document, dict) else None
    return value if isinstance(value, dict) else {}


def _stages(summary: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    samples = summary.get("samples")
    for sample, entry in (samples if isinstance(samples, Mapping) else {}).items():
        stages = entry.get("stages") if isinstance(entry, Mapping) else None
        for name, stage in (stages if isinstance(stages, Mapping) else {}).items():
            if isinstance(stage, Mapping):
                yield f"{sample}/{name}", stage


def requirements(
    study: Path, summaries: Iterable[Mapping[str, Any]], identity: Mapping[str, Any]
) -> Readiness:
    """What a rerun of the study in ``study`` needs, checked now: from what it kept, what its
    trials' records (``summaries``) say their stages used, and the study's ``identity``
    (``study.json``)."""
    at = _now()
    directory = kept_directory(study)
    store = Store(directory)
    documents = _documents(directory)
    gaps: list[str] = []
    used_local: dict[str, set[str]] = {}  # entry path: stages that read from it
    stages_seen: set[str] = set()
    records = 0
    for summary in summaries:
        records += 1
        for label, stage in _stages(summary):
            observed = stage.get("observed")
            if not isinstance(observed, Mapping) or "reused" in stage:
                continue
            stages_seen.add(label)
            if "argv" in stage and (
                observed.get("level") != "files" or observed.get("complete") is not True
            ):
                gap = f"{label} was not observed completely"
                if gap not in gaps:
                    gaps.append(gap)
            for path in cast(Sequence[str], observed.get("local") or ()):
                used_local.setdefault(path, set()).add(label)
    if not records:
        gaps.append("no trial has a record yet: what its stages use is not known")
    if not documents:
        gaps.append("the study kept nothing of what it ran with")
        return Readiness((), tuple(gaps), at)
    # a local build no stage read from did not run, if every stage of every record says so
    known = records > 0 and not gaps
    document = documents[-1]  # one study, one software: the states differ only in their sources
    ran = _ran(directory)
    found: list[Requirement] = []
    base = _base(document, at, sorted(stages_seen))
    if base is not None:
        found.append(base)
    for index, origin in enumerate(document.get("origins") or [], start=1):
        found.append(_origin(origin, index, store, at, used_local, ran, document, known))
    found += [_steering(entry, store, at) for entry in document.get("steering") or []]
    for index, (path, entry) in enumerate(sorted(_read(directory / INPUTS, "inputs").items()), 1):
        found.append(_input(index, path, entry, store, at))
    found.append(_k4opt(identity, document, at))
    for name, entry in (document.get("evaluators") or {}).items():
        found.append(_evaluator(name, entry, identity, store, at))
    problems = tuple(str(p) for p in document.get("problems") or ())
    return Readiness(tuple(found), tuple(gaps), at, problems)


def _ran(directory: Path) -> dict[str, set[str]]:
    """Each file a local build provided that an observed stage read, by its identity: the
    stages."""
    found: dict[str, set[str]] = {}
    for entry in _read(directory / USED, "footprints").values():
        if not isinstance(entry, Mapping):
            continue
        for then in (entry.get("files") or {}).values():
            if isinstance(then, str):
                found.setdefault(then, set()).add(str(entry.get("stage")))
    return found


def _base(document: Mapping[str, Any], at: str, stages: list[str]) -> Requirement | None:
    software = document.get("software") or {}
    release = document.get("release")
    try:
        build = Build.from_dict(software.get("build"))
    except Exception:  # a kept file of another k4opt: say so, do not stop
        return Requirement("base", "base", "stack", "not readable", checks=(), notes=())
    if not isinstance(release, Mapping):
        check = Check(at, "not-checked", "an unrecognized stack: k4opt cannot say where it is")
        return Requirement("base", "base", build.label, build.label, tuple(stages), checks=(check,))
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
        return Check(at, "available", "nothing differs from its base commit")
    patch = git.get("patch")
    if not isinstance(patch, str):
        return Check(at, "gone", "its changes were not kept (too large)")
    state = store.state(patch)
    return Check(at, state, f"its changes, kept as {patch[:12]}")


def _origin(
    origin: Mapping[str, Any],
    index: int,
    store: Store,
    at: str,
    used_local: Mapping[str, set[str]],
    ran: Mapping[str, set[str]],
    document: Mapping[str, Any],
    known: bool,
) -> Requirement:
    kind = origin.get("kind")
    entries = [str(e) for e in origin.get("entries") or ()]
    used = sorted({label for entry in entries for label in used_local.get(entry, ())})
    name = origin.get("package") or os.path.basename(str(origin.get("prefix")).rstrip("/"))
    git = origin.get("git")
    notes: list[str] = []
    checks: list[Check] = []
    source: dict[str, Json] = {}
    if isinstance(git, Mapping):
        checks.append(_patch_check(git, store, at))
        source = {
            "base": git.get("base"),
            "base_is": git.get("base_is"),
            "head": git.get("head"),
            "changed": len(git.get("changed") or ()),
            "patch": git.get("patch"),
        }
        if git.get("fetch"):
            notes.append(
                f"the nightly's revision {origin.get('revision')} is not in the checkout: "
                "diffing against it needs a fetch"
            )
        if git.get("large"):
            notes.append(f"untracked files too large to keep: {', '.join(git['large'])}")
        if git.get("base_is") == "none":
            notes.append("no remote has its history: its every file is in the kept changes")
    if kind == "build":
        binaries = origin.get("binaries") or []
        executed = [b for b in binaries if f"sha256:{b.get('sha256')}" in ran]
        stages = sorted({s for b in executed for s in ran[f"sha256:{b.get('sha256')}"]})
        used = sorted({*used, *stages})
        if entries and not used and known:
            notes.append("changed locally — not used: no observed stage read from it")
        if origin.get("stale"):
            notes.append("its source changed after its binaries were built")
        source |= {"link": origin.get("link"), "rebuildable": bool(origin.get("rebuildable"))}
        identified = (
            f"{_git_line(git) if isinstance(git, Mapping) else 'source unknown'} "
            f"({origin.get('link')} link)"
        )
        return Requirement(
            f"package:{name}",
            "package",
            str(name),
            identified,
            tuple(used),
            source=source,
            ran={
                "original": "fingerprint-only",
                "files": tuple(
                    {k: b.get(k) for k in ("file", "sha256", "build_id")} for b in executed
                ),
            },
            checks=tuple(checks),
            notes=tuple(notes),
        )
    if kind == "environment":
        distributions = origin.get("distributions") or []
        check = Check(at, "not-checked", "installed distributions, not checked against an index")
        return Requirement(
            f"environment:{index}",
            "environment",
            str(origin.get("prefix")),
            f"{len(distributions)} distributions",
            tuple(used),
            source={"distributions": tuple(distributions)},
            checks=(check,),
        )
    if isinstance(git, Mapping):
        identified = _git_line(git)
        if document.get("k4opt") == git.get("checkout"):
            notes.append("k4opt's own checkout")
    else:
        files = origin.get("files") or {}
        for relative, entry in files.items():
            kept = entry.get("kept") if isinstance(entry, Mapping) else None
            checks.append(
                Check(at, store.state(kept), f"{relative}, kept")
                if isinstance(kept, str)
                else Check(at, "gone", f"{relative} was not kept")
            )
        identified = f"{len(files)} files in no checkout, kept"
    return Requirement(
        f"code:{index}",
        "code",
        str(origin.get("prefix")),
        identified,
        tuple(used),
        source=source or None,
        checks=tuple(checks),
        notes=tuple(notes),
    )


def _steering(entry: Mapping[str, Any], store: Store, at: str) -> Requirement:
    stage = str(entry.get("stage"))
    package, revision = entry.get("package"), entry.get("revision")
    changed, added = entry.get("changed") or {}, entry.get("added") or {}
    removed = entry.get("removed") or []
    checks = []
    for name, file in {**changed, **added}.items():
        kept = file.get("kept") if isinstance(file, Mapping) else None
        checks.append(
            Check(at, store.state(kept), f"{name}, kept")
            if isinstance(kept, str)
            else Check(at, "gone", f"{name} was not kept")
        )
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
    )


def _input(index: int, path: str, entry: Mapping[str, Any], store: Store, at: str) -> Requirement:
    kept = entry.get("kept")
    then = str(entry.get("identity"))
    if isinstance(kept, str):
        check = Check(at, store.state(kept), "kept in the study")
    elif identity(path) == then:
        check = Check(at, "available", f"not kept ({entry.get('why')}); {path} is unchanged")
    else:
        check = Check(at, "gone", f"not kept ({entry.get('why')}); {path} changed or went")
    through = f", read through {entry['variable']}" if entry.get("variable") else ""
    size = f"{entry['bytes']} bytes" if isinstance(entry.get("bytes"), int) else "size unknown"
    return Requirement(
        f"input:{index}",
        "input",
        os.path.basename(path),
        f"{entry.get('origin')} input, {size}{through}",
        tuple(entry.get("stages") or ()),
        source={"path": path, "commitment": entry.get("commitment")},
        checks=(check,),
    )


def _k4opt(identity: Mapping[str, Any], document: Mapping[str, Any], at: str) -> Requirement:
    k4opt = identity.get("k4opt") or {}
    git = k4opt.get("git") or {}
    commit = str(git.get("commit") or "")[:7]
    dirty = git.get("dirty")
    label = f"k4opt {k4opt.get('version')}"
    if not commit:
        check = Check(at, "not-checked", "no checkout tracks it: a released version")
        return Requirement("k4opt", "k4opt", label, "no git revision", checks=(check,))
    kept = any(
        (o.get("git") or {}).get("checkout") == document.get("k4opt")
        for o in document.get("origins") or []
    )
    if dirty and not kept:
        check = Check(at, "gone", "its uncommitted changes were not kept")
    elif dirty:
        check = Check(at, "available", "its uncommitted changes are kept with its checkout")
    else:
        check = Check(at, "not-checked", "a commit; whether a remote has it is not checked here")
    return Requirement(
        "k4opt",
        "k4opt",
        label,
        f"{commit}{' with uncommitted changes' if dirty else ''}",
        source={"commit": git.get("commit"), "sha256": k4opt.get("sha256")},
        checks=(check,),
    )


def _evaluator(
    name: str, entry: Mapping[str, Any], identity: Mapping[str, Any], store: Store, at: str
) -> Requirement:
    version = ((identity.get("implementations") or {}).get(name) or {}).get("version")
    files = entry.get("files") or {}
    checks = tuple(Check(at, store.state(kept), f"{file}, kept") for file, kept in files.items())
    if not checks:
        checks = (Check(at, "gone", "its file was not kept"),)
    return Requirement(
        f"evaluator:{name}",
        "evaluator",
        f"{name} v{version}",
        f"{len(files)} files, kept",
        source={"sha256": entry.get("sha256")},
        checks=checks,
    )


__all__ = [
    "ALL_AVAILABLE",
    "BLOCKED",
    "INCOMPLETE",
    "REBUILD",
    "Check",
    "Readiness",
    "Requirement",
    "kept_directory",
    "requirements",
]
