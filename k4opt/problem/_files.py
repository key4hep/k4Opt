"""Internal: the files of a candidate, as its records identify them and its tasks watch them.

A file is identified by its size and SHA-256 (:func:`file_identity`), hashed once per version of
it, and told apart from a rewritten one, on the filesystem that took it, by its ``stat``: its
size, times and inode (:data:`Stat`). The stats are machine-specific and never recorded.

The files a task makes in its own directory are :class:`Made`: what the record says of each (an
:class:`~k4opt.evaluation.Artifact`, hashed for an evaluator's artifact, never for the possibly
huge output of a stage), and the file's ``stat`` when it was made. A file whose stat changed is
no longer what the record says (:meth:`Made.changed`), so the tasks that read it check it
before and after they run, and the candidate checks all of them at its end. The files of the
geometry, which the record identifies together by their fingerprint and every simulation of
every sample reads, are watched as one :class:`Tree`. :func:`artifacts` checks and hashes what
an evaluator returned.

A file a task made is named relative to its candidate's workspace (:func:`portable`), so that a
task description stays true wherever the workspace is; :func:`local` names it where the
workspace is on this machine.
"""

from __future__ import annotations

import functools
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from k4opt.evaluation import Artifact, EvaluationContractError, EvaluationResult, Status
from k4opt.frozen import FrozenMap, Json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from k4opt.problem._tasks import TaskId


type Stat = tuple[int, int, int, int]
"""What tells a file apart from a rewritten one: its size, mtime, ctime and inode."""


def stat_of(path: Path) -> Stat | None:
    try:
        status = path.stat()
    except OSError:
        return None
    return (status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino)


@functools.lru_cache(maxsize=32)
def _sha256(path: str, size: int, mtime_ns: int, ctime_ns: int, inode: int) -> str:
    """SHA-256 of a file. The other arguments only key the cache, so the same file is read once
    per run of candidates but a changed one is hashed again: writing to a file or replacing it
    changes its ctime or inode even if the content keeps its size and its mtime is restored.
    The kernel stamps ctime with a coarse clock (milliseconds), so a file rewritten within the
    same tick as it was hashed is not told apart; input files are assumed not to change while
    a campaign hashes them."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Json]:
    """``bytes`` and ``sha256`` of the file at ``path``; raises :class:`OSError`."""
    status = path.stat()
    return {
        "bytes": status.st_size,
        "sha256": _sha256(
            str(path), status.st_size, status.st_mtime_ns, status.st_ctime_ns, status.st_ino
        ),
    }


def readable_identity(path: Path) -> dict[str, Json] | None:
    """:func:`file_identity`, or ``None`` for a file that cannot be read: whatever reads it
    reports that."""
    try:
        return file_identity(path)
    except OSError:
        return None


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
        sha256 = file_identity(path)["sha256"] if hashed else None
        assert sha256 is None or isinstance(sha256, str)
        if hashed and stat_of(path) != stat:
            raise OSError(f"{path} changed while it was hashed")
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


def artifacts(
    task: TaskId,
    declared: Mapping[str, str] | None,
    result: EvaluationResult,
    workdir: Path | None,
) -> dict[str, Made]:
    """The artifacts of the result of the evaluation ``task``, each hashed once, now.

    Each must be a file in the evaluation's own directory ``workdir``, its links resolved. An
    evaluator that ``declared`` artifacts returns no other, and a ``valid`` result of it every
    one. Raises :class:`EvaluationContractError`.
    """
    where = f"evaluator {task.label!r}"
    if declared is not None:
        unknown = [name for name in result.artifacts if name not in declared]
        if unknown:
            raise EvaluationContractError(
                f"{where} returned the artifact(s) {', '.join(unknown)}, which it does not "
                f"declare; it declares {', '.join(declared) or 'none'}"
            )
        missing = [name for name in declared if name not in result.artifacts]
        if missing and result.status is Status.VALID:
            raise EvaluationContractError(
                f"{where} returned a valid result without its declared artifact(s) "
                f"{', '.join(missing)}"
            )
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
