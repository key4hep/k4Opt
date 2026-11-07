"""Internal: the files of a candidate, as its records identify them and its tasks watch them.

A file is identified by its size and SHA-256 (:func:`k4opt.hashing.file_identity`), hashed once
per version of it, and told apart from a rewritten one by its ``stat``
(:data:`k4opt.hashing.Stat`).

The files a task makes in its own directory are :class:`Made`: what the record says of each (an
:class:`~k4opt.evaluation.Artifact`, hashed once, when the task ends: an evaluator's artifact
and the output of a stage alike), and the file's ``stat`` when it was made. A file whose stat
changed is no longer what the record says (:meth:`Made.changed`), so the tasks that read it check it
before and after they run, and the candidate checks all of them at its end. The files of the
geometry, which the record identifies together by their fingerprint and every simulation of
every sample reads, are watched as one :class:`Tree`. :func:`artifacts` checks and hashes what
an evaluator returned.

A file a task made is named relative to its candidate's workspace (:func:`portable`), so that a
task description stays true wherever the workspace is; :func:`local` names it where the
workspace is on this machine.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from k4opt.evaluation import Artifact, EvaluationContractError, EvaluationResult
from k4opt.frozen import FrozenMap, Json
from k4opt.hashing import Stat, file_identity, stat_of

if TYPE_CHECKING:  # pragma: no cover - typing only
    from k4opt.problem.plan import TaskId


def snapshot(path: Path) -> tuple[Stat, dict[str, Json]]:
    """The stat of the file at ``path`` and its :func:`file_identity` at that stat: one version
    of it, so that a later equal stat vouches for its hash. Raises :class:`OSError` if it cannot
    be read, or if it changes while it is hashed (a size and a hash of other versions)."""
    stat = stat_of(path)
    if stat is None:
        raise OSError(f"{path} is no file")
    identity = file_identity(path)
    if stat_of(path) != stat:
        raise OSError(f"{path} changed while it was hashed")
    return stat, identity


@dataclass(frozen=True, slots=True)
class Made:
    """A file a task made in its own directory: its ``artifact`` as the record names it, and its
    ``stat`` when it was made."""

    artifact: Artifact
    stat: Stat

    @classmethod
    def file(cls, task: TaskId, name: str, path: Path, *, hashed: bool) -> Made:
        """The file ``path`` that ``task`` made as ``name``, with its SHA-256 if ``hashed``;
        raises :class:`OSError` if it cannot be read, or if it changes while it is hashed (its
        size and its hash would describe other versions of it)."""
        stat = stat_of(path)
        if stat is None or not path.is_file():
            raise OSError(f"{path} is no file")
        sha256: str | None = None
        if hashed:
            stat, identity = snapshot(path)
            digest = identity["sha256"]
            assert isinstance(digest, str)
            sha256 = digest
        return cls(Artifact(f"{task.label}#{name}", path, stat[0], sha256), stat)

    def changed(self) -> bool:
        """Whether the file is no longer what the record says: it is gone, or its stat changed
        and it is not hashed, or hashed again it has other bytes. Equal stats are trusted."""
        now = stat_of(self.artifact.path)
        if now == self.stat:
            return False
        if now is None or self.artifact.sha256 is None:
            return True
        try:
            return file_identity(self.artifact.path)["sha256"] != self.artifact.sha256
        except OSError:
            return True


@dataclass(frozen=True, slots=True)
class Tree:
    """Every file below ``root``, a directory a task made, by its path relative to ``root``, with
    its stat when the task was done: a tree the record identifies as a whole (the geometry, by
    its fingerprint), so any file in it that changes, goes or is added changes it."""

    root: Path
    stats: Mapping[str, Stat]

    @classmethod
    def of(cls, root: Path) -> Tree:
        stats = {
            path.relative_to(root).as_posix(): stat
            for path in sorted(root.rglob("*"))
            if not path.is_dir() and (stat := stat_of(path)) is not None
        }
        return cls(root, FrozenMap(stats))

    def changed(self) -> list[str]:
        """The files of the tree, by relative path, that are not as they were."""
        now = Tree.of(self.root).stats
        return [
            name
            for name in dict.fromkeys([*self.stats, *now])
            if now.get(name) != self.stats.get(name)
        ]


@dataclass(frozen=True, slots=True)
class Observed:
    """What the coordinator watches of the files one task made, until its candidate ends: each
    file with its stat where it was made (``made``, by name) and the geometry it made
    (``tree``). As machine-specific as the stats it holds, so never part of a record."""

    made: Mapping[str, Made] = field(default_factory=lambda: FrozenMap({}))
    tree: Tree | None = None


NOTHING = Observed()
"""What is watched of a task that made no file, or never ran."""


def changed_files(observed: Mapping[TaskId, Observed]) -> list[str]:
    """The files the tasks ``observed`` made that changed since: the reference of each, or for a
    tree its path below the task (``geometry/compact/ECalBarrel.xml``)."""
    found = []
    for task, watched in observed.items():
        if watched.tree is not None:
            found += [f"{task.label}/{name}" for name in watched.tree.changed()]
        found += [made.artifact.ref for made in watched.made.values() if made.changed()]
    return found


def portable(path: Path, root: Path | None) -> Path:
    """``path`` relative to the workspace ``root`` if it lies in it (lexically), else absolute."""
    absolute = Path(os.path.abspath(path))
    if root is None:
        return absolute
    try:
        return absolute.relative_to(Path(os.path.abspath(root)))
    except ValueError:
        return absolute


def local(path: Path, root: Path | None) -> Path:
    """The file ``path`` names in the workspace whose directory is ``root`` here."""
    if path.is_absolute():
        return path
    assert root is not None  # only a workspace's files are named relative to it
    return root / path


def portable_artifact(artifact: Artifact, root: Path | None) -> Artifact:
    return replace(artifact, path=portable(artifact.path, root))


def local_artifact(artifact: Artifact, root: Path | None) -> Artifact:
    return replace(artifact, path=local(artifact.path, root))


def artifacts(task: TaskId, result: EvaluationResult, workdir: Path | None) -> dict[str, Made]:
    """The artifacts of the result of the evaluation ``task``, each hashed once, now.

    Each must be a file in the evaluation's own directory ``workdir``, its links resolved (which
    names it may return is the problem's to say, :func:`~k4opt.problem.contract.check_result`).
    Raises :class:`EvaluationContractError`.
    """
    where = f"evaluator {task.label!r}"
    made: dict[str, Made] = {}
    for name, path in result.artifacts.items():
        if workdir is None:
            raise EvaluationContractError(
                f"{where}: artifact {name!r}: the evaluation has no directory of its own to "
                "write it in"
            )
        own, resolved = workdir.resolve(), path.resolve()
        if resolved == own or not resolved.is_relative_to(own):
            raise EvaluationContractError(
                f"{where}: artifact {name!r} ({path}) is not in its own directory {workdir}; "
                "write it in request.workdir"
            )
        try:
            made[name] = Made.file(task, name, resolved, hashed=True)
        except OSError as exc:
            raise EvaluationContractError(
                f"{where}: artifact {name!r} ({path}) is no file it can read: {exc}"
            ) from None
    return made
