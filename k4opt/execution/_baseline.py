"""Internal: what an evaluation session evaluates with, taken once when it opens, and the checks
that none of it changed since.

A record says which problem, files, geometry, implementations, software, programs and code its
candidate was evaluated with. A session takes all of that once, when it opens
(:meth:`Baseline.take`): the identity (size and SHA-256) of every file the stages read, what the
geometry's external references read, what the implementations declare and their sources, the
software it runs on (the build of the stack, the system and the variables:
:class:`k4opt.software.Software`), the programs of the stack its stages run (each executable the
stage contexts pinned, with its size and SHA-256) and k4opt's own code. Before a candidate is
prepared and again before its record is written, :meth:`Baseline.verify` refuses if any of it
changed, a nightly's slot republished as much as a variable set, so that no record names files,
programs, software or code other than those its candidate ran with. What
a reference only candidates name reads is remembered as candidates are recorded
(:meth:`Baseline.remember`), so that every candidate of the session reads the same files
through it (:meth:`Baseline.check_read`). So are the files a stage read that nothing else
identifies, its *undeclared inputs* (:mod:`k4opt.software.observe`: a ``~/.rootrc``, a calibration
file a helper reads through a variable): once a record names one, every later candidate is
evaluated with it as it was (:meth:`Baseline.check_observed`), and a change to it stops the session
(:meth:`Baseline.verify`). So does a path such a stage looked for in vain that a stage would find
now (:meth:`Baseline.check_looked`), unless a stage of the study made it: a cache, by the study's
own doing.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path

from k4opt import __version__
from k4opt.errors import K4OptError
from k4opt.evaluation.loader import changed_code, evaluator_source
from k4opt.execution._binding import _Binding, _implementations
from k4opt.execution._work import content
from k4opt.execution.errors import ProvenanceError, RunError
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.hashing import readable_identity, sha256_of
from k4opt.problem.candidate import Configuration
from k4opt.problem.models import ProblemSpec, ResolvedProblem
from k4opt.program import identify
from k4opt.record.results import TaskResult
from k4opt.revision import code_identity, code_unchanged
from k4opt.software import Build, Footprint, Software, SoftwareError, System, compared_variables
from k4opt.software.observe import changed_inputs, findable, load


def check_code() -> None:
    """Raise :class:`ProvenanceError` if k4opt's files are no longer those this process
    identified."""
    if not code_unchanged():
        raise ProvenanceError(
            "k4opt's code changed on disk after this process first identified it, so its "
            "records could not say which code ran; start a new process"
        )


class Baseline:
    """What a session evaluates with: the problem as loaded (``problem``), the files
    the stages read with their identities by where they are named (``files``), what the
    geometry's external references read (``externals``), what a record says of each
    implementation (``taken``) and of the process (``process``: k4opt's code and the
    environment), and the implementations by name (``classes``)."""

    __slots__ = (
        "_inputs",
        "_listed",
        "_looked",
        "_made",
        "_read",
        "_variables",
        "classes",
        "externals",
        "files",
        "problem",
        "process",
        "taken",
    )

    def __init__(
        self,
        problem: ResolvedProblem,
        files: Mapping[str, tuple[Path, Mapping[str, Json] | None]],
        externals: Mapping[str, Json],
        taken: Mapping[str, Mapping[str, Json]],
        process: Mapping[str, Json],
        classes: Mapping[str, type] | None = None,
        listed: Callable[[], Mapping[str, Path]] | None = None,
    ) -> None:
        self.problem = problem
        # the files the stages read as the directories the problem names hold them now
        self._listed = listed
        self.files = FrozenMap(files)
        self.externals = externals
        self.taken = FrozenMap(taken)
        self.process = process
        self.classes = FrozenMap(classes or {})
        # what the external references only candidates have read, by reference, without the path
        self._read: dict[str, Json] = {}
        # the compared variables as they were, which tell a change apart in microseconds
        self._variables = compared_variables()
        # what each undeclared input the records name was, by its path
        self._inputs: dict[str, Json] = {}
        # what their stages looked for in vain (path and how), and what the study's stages made
        self._looked: set[tuple[str, str]] = set()
        self._made: set[str] = set()

    @property
    def software(self) -> Software:
        """What the session runs on, as it took it when it opened."""
        return Software.from_dict(self.process["software"])

    @property
    def spec(self) -> ProblemSpec:
        return self.problem.spec

    @property
    def path(self) -> Path:
        return self.problem.path

    @classmethod
    def take(
        cls,
        problem: ResolvedProblem,
        binding: _Binding,
        named: Mapping[str, Path] | Callable[[], Mapping[str, Path]],
    ) -> Baseline:
        """What a session of ``problem``, bound as ``binding``, evaluates with, taken now;
        ``named`` are the files its stages read, by where they are named, or what lists them as
        the directories the problem names (a reconstruction's) hold them when it is called: then
        :meth:`verify_inputs` refuses a file that appeared in one or went. Raises
        :class:`RunError` if k4opt's code changed since this process identified it, or if a
        program a stage runs was found but cannot be identified (:func:`k4opt.program.identify`)."""
        listed = named if callable(named) else None
        named = named() if callable(named) else named
        # an unreadable file has no identity: its stage reports it, and a record still names it
        files = {label: (file, readable_identity(file)) for label, file in named.items()}
        externals = freeze(problem.source.external_files(problem.catalog), "geometry", RunError)
        assert isinstance(externals, Mapping)
        # before anything runs, so that an edit to the checkout during a long run is not
        # recorded as the code that ran, nor a later change of the environment as the session's
        environment = {name: os.environ.get(name) for name in ("KEY4HEP_STACK", "K4GEO")}
        executables = {
            name: path
            for context in binding.contexts.values()
            for name, path in context.programs.items()
        }
        programs = {
            name: None if path is None else identify(Path(path))
            for name, path in sorted(executables.items())
        }
        for name, path in executables.items():
            if path is not None and programs[name] is None:
                raise ProvenanceError(
                    f"{name}: {path} cannot be read, or the interpreter its #! line names cannot "
                    "be found or read, so no record could say which program ran"
                )
        process = freeze(
            {
                "k4opt": code_identity(),
                "environment": environment,
                "programs": programs,
                "software": Software.now().to_dict(),
            },
            "process",
            RunError,
        )
        check_code()
        assert isinstance(process, Mapping)
        classes = {b.declared.declaration.name: b.implementation for b in binding.evaluators}
        return cls(problem, files, externals, _implementations(binding), process, classes, listed)

    def identity(self) -> dict[str, object]:
        """What the session evaluates with, as plain JSON-compatible data
        (:meth:`~k4opt.execution.EvaluationSession.identity`). A fresh copy on every call."""
        source, catalog = self.problem.source, self.problem.catalog
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
                "targets": self.targets(),
            },
            "environment": process["environment"],
            "programs": process["programs"],
            "software": process["software"],
            "k4opt": {"version": __version__, **process["k4opt"]},
        }

    def targets(self) -> list[dict[str, object]]:
        """The attribute each geometry destination of the problem sets, as the session resolved
        it, with its parameter and its unmodified value, in the problem's order: what a
        candidate's geometry changes, wherever the geometry is."""
        destinations = self.spec.geometry_destinations
        return [
            {
                "parameter": parameter.name,
                "target": target.target,
                "file": target.file,
                "locator": target.locator,
                "attribute": target.attribute,
                "value": target.value,
            }
            for (parameter, _), target in zip(destinations, self.problem.targets, strict=True)
        ]

    def check_changes(self, geometry: TaskResult, configuration: Configuration) -> None:
        """Raise :class:`ProvenanceError` unless the geometry task's ``changes`` are those of the
        attributes the session resolved, each by its parameter, from its unmodified value to the
        text ``configuration`` gives it, in the problem's order."""
        expected = [
            {**{k: v for k, v in target.items() if k != "value"}, "old_value": target["value"]}
            | {"new_value": text}
            for target, text in zip(self.targets(), configuration.geometry, strict=True)
        ]
        recorded = thaw(geometry.provenance.get("changes"))
        if recorded != expected:
            raise ProvenanceError(
                f"geometry: it changed {recorded!r}, but the session resolved the candidate's "
                f"changes as {expected!r}"
            )

    def verify_inputs(self) -> None:
        """Raise :class:`ProvenanceError` if a file the stages read is not the file it was when the
        session opened: other bytes, or now readable or unreadable, or if the directories the
        problem names (a reconstruction's job and data) no longer hold the files they held, a
        file added as much as one removed. A file is hashed again only when its size, times or
        inode changed."""
        if self._listed is not None:
            try:
                now_listed = self._listed()
            except (K4OptError, OSError) as exc:  # a directory gone, a link loop made since
                raise ProvenanceError(f"the files the stages read cannot be listed: {exc}") from exc
            added = sorted(now_listed.keys() - self.files.keys())
            gone = sorted(self.files.keys() - now_listed.keys())
            if added or gone:
                said = [
                    *(f"{label} appeared" for label in added),
                    *(f"{label} went" for label in gone),
                ]
                raise ProvenanceError(
                    f"{'; '.join(said)} after the session opened; evaluate the candidates of a "
                    "problem with the same files"
                )
        for label, (file, opened) in self.files.items():
            now = readable_identity(file)
            if now != opened:
                raise ProvenanceError(
                    f"{label}: {file} changed after the session opened ({_described(opened)} "
                    f"then, {_described(now)} now); evaluate the candidates of a problem with the "
                    "same files"
                )

    def verify(self, local: bool = True, made: Collection[str] = ()) -> None:
        """Raise :class:`ProvenanceError` if the environment, k4opt's code, a file the stages read,
        what the geometry's external references read, the identity of an implementation or a
        module it imports changed since the session opened. The files of a local build are looked at
        only with ``local`` (tens of milliseconds for a virtual environment): before a record is
        written, which a change since the session opened refuses. ``made`` are the paths the
        stages of the candidate to be recorded made, as those of the study
        (:meth:`check_looked`)."""
        self._verify_environment(local)
        check_code()
        self._verify_programs()
        self.verify_inputs()
        self._verify_externals()
        self._verify_implementations()
        self._verify_observed(made)

    def _verify_observed(self, made: Collection[str]) -> None:
        """Each undeclared input a record names is what it was (:meth:`remember_observed`), and
        no path its stages looked for in vain can be found (:meth:`remember_looked`)."""
        changed = changed_inputs(self._inputs)
        if changed:
            raise ProvenanceError(
                f"{', '.join(changed)}, read by stages of this session, changed after a record "
                "named it; the candidates of a session read the same files"
            )
        self.check_looked(self._looked, made)

    def check_looked(self, looked: Iterable[tuple[str, str]], made: Collection[str] = ()) -> None:
        """Raise :class:`ProvenanceError` if a stage would now find one of the paths ``looked`` for
        in vain (path and how, :meth:`k4opt.software.Footprint.looked`) that neither a stage of the
        study (:meth:`remember_looked`) nor one in ``made`` made: a candidate evaluated now would
        not be evaluated as the one that looked for it."""
        appeared = [p for p in findable(looked) if p not in self._made and p not in made]
        if appeared:
            raise ProvenanceError(
                f"{', '.join(appeared)}, which a stage of this session looked for and did not "
                "find, can be found now; the candidates of a session read the same files"
            )

    def remember_looked(self, looked: Iterable[tuple[str, str]], made: Iterable[str]) -> None:
        """The paths ``looked`` for in vain by a recorded candidate's stages (path and how), and
        those they ``made``: no later candidate may find one of the first that the study did not
        make, however it failed to."""
        self._looked.update(looked)
        self._made.update(made)

    def check_observed(self, inputs: Mapping[str, Json]) -> None:
        """Raise :class:`ProvenanceError` if a stage read one of the undeclared ``inputs`` (path:
        identity) as another file than a stage of a candidate recorded before, or if one is no
        longer what the stage read."""
        other = sorted(path for path, now in inputs.items() if self._inputs.get(path, now) != now)
        changed = changed_inputs(inputs)
        if other or changed:
            raise ProvenanceError(
                f"{', '.join(other or changed)}, read by a stage, is not the file "
                f"{'earlier candidates read' if other else 'the stage read'}; the candidates of a "
                "session read the same files"
            )

    def remember_observed(self, inputs: Mapping[str, Json]) -> None:
        """The undeclared ``inputs`` a recorded candidate's stages read: later candidates must
        read the same, and none may change while the session runs."""
        self._inputs |= inputs

    def _verify_implementations(self) -> None:
        """Each implementation is still identified as when the session opened: loading another
        file that imports its class (and may change it) in this process identifies it by that
        file, and the class the session evaluates with may have changed. And the code each runs
        is what its identity says (:func:`~k4opt.evaluation.loader.changed_code`): the module its
        reference names, not reloaded since, and the modules it imports from beside it, one it
        imports only when it evaluates (inside a function) too."""
        for name, implementation in self.classes.items():
            source = self.taken[name]["source"]
            assert isinstance(source, Mapping)
            now = evaluator_source(implementation)
            if any(now.get(key) != source.get(key) for key in ("path", "sha256", "helpers")):
                raise ProvenanceError(
                    f"evaluator {name!r}: its class was loaded again in this process, through "
                    f"{now.get('reference')}, which may have changed it; open the session in a new "
                    "process"
                )
            changed = changed_code(implementation)
            if changed:
                raise ProvenanceError(
                    f"evaluator {name!r}: {', '.join(changed)} changed after the session loaded "
                    "it; evaluate with the code it was loaded with, or open a new session in a new "
                    "process"
                )

    def check_read(self, externals: Mapping[str, Mapping[str, Json]]) -> None:
        """Raise :class:`ProvenanceError` if a reference only candidates name reads other files than
        it did for a candidate recorded before."""
        for ref, entry in externals.items():
            if ref in self._read and self._read[ref] != content(entry):
                raise ProvenanceError(
                    f"geometry: {ref} does not read the files it read for an earlier candidate "
                    f"(now {entry.get('path') or 'no file'}); the candidates of a session start "
                    "from one geometry"
                )

    def remember(self, externals: Mapping[str, Mapping[str, Json]]) -> None:
        """What the references only a recorded candidate names read: later candidates must read
        the same through them."""
        self._read |= {ref: content(entry) for ref, entry in externals.items()}

    def _verify_externals(self) -> None:
        now = self.problem.source.external_files(self.problem.catalog)
        for ref, opened in self.externals.items():
            if now.get(ref) != thaw(opened):
                path = (now.get(ref) or {}).get("path") or "no file"
                raise ProvenanceError(
                    f"geometry: {ref} does not read the files it read when the session opened "
                    f"(now {path}); the candidates of a session start from one geometry"
                )

    def _verify_programs(self) -> None:
        """Each program the stages run is the executable the session identified: the same size
        and SHA-256 where it was found."""
        programs = self.process["programs"]
        assert isinstance(programs, Mapping)
        for name, opened in programs.items():
            if not isinstance(opened, Mapping):
                continue  # it was not found: its stages fail (StageContext.unfound)
            now = identify(Path(str(opened["path"])))
            if now != thaw(opened):
                raise ProvenanceError(
                    f"{name}: {opened['path']} changed after the session opened; the candidates "
                    "of a session run the same programs"
                )

    def _verify_environment(self, local: bool) -> None:
        """``KEY4HEP_STACK`` and ``K4GEO`` are what they were when the session opened: the stages
        would run in another environment than the geometry was discovered in, and than the
        records say. So is the software: every program inherits this process's variables, and
        the records and the stages' signatures say what it was then. A nightly's slot republished
        since is another build, wherever ``KEY4HEP_STACK`` still points."""
        opened = self.process["environment"]
        assert isinstance(opened, Mapping)
        for name, then in opened.items():
            now = os.environ.get(name)
            if now != then:
                raise ProvenanceError(
                    f"{name} changed after the session opened ({then!r} then, {now!r} now); the "
                    "candidates of a session run in one environment"
                )
        software = self.software
        if not local and compared_variables() == self._variables:
            build = Build.of(os.environ.get("KEY4HEP_STACK"))
            if build == software.build and System.now() == software.system:
                return
        changed = software.differences(Software.now())
        if changed:
            raise ProvenanceError(
                f"the software changed after the session opened: {'; '.join(changed)}; the "
                "candidates of a session run on one software"
            )


def recorded_footprint(path: Path | None, sha256: object, stage: str, advice: str) -> Footprint:
    """The footprint a record names for ``stage``, the file it is at with the SHA-256 the record
    holds. Raises :class:`ProvenanceError` if it is not there, cannot be read or is another file:
    what the stage looked for could not be checked, which ``advice`` says what to do of."""
    why = "its record names none where one may be"
    if path is not None:
        try:
            if sha256_of(path) != sha256:
                why = f"{path} is not the file its record names"
            else:
                return load(path)
        except OSError as exc:
            why = f"{path} cannot be read ({exc.strerror or exc})"
        except SoftwareError as exc:
            why = str(exc)
    raise ProvenanceError(
        f"{stage}: what it looked for cannot be checked, since its footprint is not there as "
        f"recorded: {why}; {advice}"
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
