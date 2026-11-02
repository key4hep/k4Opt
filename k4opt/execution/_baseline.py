"""Internal: what an evaluation session evaluates with, taken once when it opens, and the checks
that none of it changed since.

A record says which problem, files, geometry, implementations, environment and code its
candidate was evaluated with. A session takes all of that once, when it opens
(:meth:`Baseline.take`): the identity (size and SHA-256) of every file the stages read, what the
geometry's external references read, what the implementations declare and their sources, the
environment and k4opt's own code. Before a candidate is prepared and again before its record is
written, :meth:`Baseline.verify` refuses if any of it changed, so that no record names files or
code other than those its candidate ran with. What a reference only candidates name reads is
remembered as candidates are recorded (:meth:`Baseline.remember`), so that every candidate of
the session reads the same files through it (:meth:`Baseline.check_read`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from k4opt import __version__
from k4opt.execution._binding import _Binding, _implementations
from k4opt.execution._files import readable_identity
from k4opt.execution._work import content
from k4opt.execution.errors import RunError
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.problem.models import ProblemSpec
from k4opt.record.results import TaskResult
from k4opt.revision import code_identity, code_unchanged


def check_code() -> None:
    """Raise :class:`RunError` if k4opt's files are no longer those this process identified."""
    if not code_unchanged():
        raise RunError(
            "k4opt's code changed on disk after this process first identified it, so its "
            "records could not say which code ran; start a new process"
        )


class Baseline:
    """What a session evaluates with: the problem at ``path`` as loaded (``spec``), the files
    the stages read with their identities by where they are named (``files``), what the
    geometry's external references read (``externals``), what a record says of each
    implementation (``taken``) and of the process (``process``: k4opt's code and the
    environment)."""

    __slots__ = ("_read", "externals", "files", "path", "process", "spec", "taken")

    def __init__(
        self,
        path: Path,
        spec: ProblemSpec,
        files: Mapping[str, tuple[Path, Mapping[str, Json] | None]],
        externals: Mapping[str, Json],
        taken: Mapping[str, Mapping[str, Json]],
        process: Mapping[str, Json],
    ) -> None:
        self.path = path
        self.spec = spec
        self.files = FrozenMap(files)
        self.externals = externals
        self.taken = FrozenMap(taken)
        self.process = process
        # what the external references only candidates have read, by reference, without the path
        self._read: dict[str, Json] = {}

    @classmethod
    def take(
        cls, path: Path, spec: ProblemSpec, binding: _Binding, named: Mapping[str, Path]
    ) -> Baseline:
        """What a session of the problem ``spec`` loaded from ``path``, bound as ``binding``,
        evaluates with, taken now; ``named`` are the files its stages read, by where they are
        named. Raises :class:`RunError` if k4opt's code changed since this process identified
        it."""
        # an unreadable file has no identity: its stage reports it, and a record still names it
        files = {label: (file, readable_identity(file)) for label, file in named.items()}
        assert spec._source is not None
        externals = freeze(spec._source.external_files(spec._catalog), "geometry", RunError)
        assert isinstance(externals, Mapping)
        # before anything runs, so that an edit to the checkout during a long run is not
        # recorded as the code that ran, nor a later change of the environment as the session's
        environment = {name: os.environ.get(name) for name in ("KEY4HEP_STACK", "K4GEO")}
        process = freeze(
            {"k4opt": code_identity(), "environment": environment}, "process", RunError
        )
        check_code()
        assert isinstance(process, Mapping)
        return cls(path, spec, files, externals, _implementations(binding), process)

    def identity(self) -> dict[str, object]:
        """What the session evaluates with, as plain JSON-compatible data
        (:meth:`~k4opt.execution.EvaluationSession.identity`). A fresh copy on every call."""
        source, catalog = self.spec._source, self.spec._catalog
        assert source is not None
        assert catalog is not None
        process = thaw(self.process)
        assert isinstance(process, dict)
        return {
            "problem": {"path": str(self.path), "definition": self.spec.to_dict()},
            "inputs": {
                label: {"path": str(file), **(identity or {})}
                for label, (file, identity) in self.files.items()
            },
            "implementations": {name: thaw(entry) for name, entry in self.taken.items()},
            "geometry": {
                "compact": source.compact,
                "source_root": str(source.root),
                "fingerprint": source.fingerprint([], catalog=catalog),
                "externals": thaw(self.externals),
            },
            "environment": process["environment"],
            "k4opt": {"version": __version__, **process["k4opt"]},
        }

    def verify_inputs(self) -> None:
        """Raise :class:`RunError` if a file the stages read is not the file it was when the
        session opened: other bytes, or now readable or unreadable. A file is hashed again only
        when its size, times or inode changed."""
        for label, (file, opened) in self.files.items():
            now = readable_identity(file)
            if now != opened:
                raise RunError(
                    f"{label}: {file} changed after the session opened ({_described(opened)} "
                    f"then, {_described(now)} now); evaluate the candidates of a problem with the "
                    "same files"
                )

    def verify(self) -> None:
        """Raise :class:`RunError` if the environment, k4opt's code, a file the stages read or
        what the geometry's external references read changed since the session opened."""
        self._verify_environment()
        check_code()
        self.verify_inputs()
        self._verify_externals()

    def check_read(self, externals: Mapping[str, Mapping[str, Json]]) -> None:
        """Raise :class:`RunError` if a reference only candidates name reads other files than
        it did for a candidate recorded before."""
        for ref, entry in externals.items():
            if ref in self._read and self._read[ref] != content(entry):
                raise RunError(
                    f"geometry: {ref} does not read the files it read for an earlier candidate "
                    f"(now {entry.get('path') or 'no file'}); the candidates of a session start "
                    "from one geometry"
                )

    def remember(self, externals: Mapping[str, Mapping[str, Json]]) -> None:
        """What the references only a recorded candidate names read: later candidates must read
        the same through them."""
        self._read |= {ref: content(entry) for ref, entry in externals.items()}

    def _verify_externals(self) -> None:
        source = self.spec._source
        assert source is not None
        now = source.external_files(self.spec._catalog)
        for ref, opened in self.externals.items():
            if now.get(ref) != thaw(opened):
                path = (now.get(ref) or {}).get("path") or "no file"
                raise RunError(
                    f"geometry: {ref} does not read the files it read when the session opened "
                    f"(now {path}); the candidates of a session start from one geometry"
                )

    def _verify_environment(self) -> None:
        """``KEY4HEP_STACK`` and ``K4GEO`` are what they were when the session opened: the stages
        would run in another environment than the geometry was discovered in, and than the
        records say."""
        opened = self.process["environment"]
        assert isinstance(opened, Mapping)
        for name, then in opened.items():
            now = os.environ.get(name)
            if now != then:
                raise RunError(
                    f"{name} changed after the session opened ({then!r} then, {now!r} now); the "
                    "candidates of a session run in one environment"
                )


def candidate_reads(geometry: TaskResult) -> dict[str, dict[str, Json]]:
    """What the references only the candidate's geometry names read, as its geometry task
    recorded it."""
    externals = thaw(geometry.provenance.get("candidate_externals") or {})
    assert isinstance(externals, dict)
    return externals


def plain(value: object) -> object:
    """``value`` as plain JSON data, as a record holds it."""
    return thaw(freeze(value, "geometry", RunError))


def _described(identity: Mapping[str, Json] | None) -> str:
    if identity is None:
        return "unreadable"
    return f"{identity['bytes']} bytes, SHA-256 {str(identity['sha256'])[:12]}"
