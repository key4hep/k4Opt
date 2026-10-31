"""Internal: the directory of a study and what makes resuming it safe; nothing of Optuna.

A study directory holds ``study.json``, written once with what the study is made with, the
optimizer's storage, a ``.lock`` while a process works on the study, and a ``trial-NNNNNN``
directory for each trial that got as far as its evaluation, laid out as ``k4opt evaluate``
lays out its output.

Whether a study may be resumed, and whether the record of a trial belongs to its study, are decided
by one comparison (:func:`compared`) of what must be equal: the problem's definition (but not its
name), the content of every file it names, each evaluator implementation's name, version, source
hash, declared metrics and declared artifacts, the unmodified geometry's fingerprint and the
content of the files its external references read, ``KEY4HEP_STACK`` and the hash of k4opt's
code. Locations are recorded and
never compared, so a study whose files moved resumes. A file that only candidates name (a ``ref``
parameter) is not known before they do: each record must read it as every earlier record of the
study did (:func:`check_candidate_files`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from k4opt.jsonio import write_json
from k4opt.optimization.errors import OptimizationError

STUDY_NAME = "study.json"
STUDY_SCHEMA = "k4opt.study/1"
LOCK_NAME = ".lock"
JOURNAL_NAME = "study.journal"
CONSTRAINT_VALUES = "physical margins"  # how a limit's margin becomes the optimizer's value

_TRIAL = re.compile(r"trial-[0-9]{6,}")


def trial_directory(number: int) -> str:
    """The directory of trial ``number``, relative to the study: ``trial-000007``."""
    return f"trial-{number:06d}"


def trial_directories(directory: Path) -> list[str]:
    """The trial directories in a study directory, by name."""
    return sorted(path.name for path in directory.iterdir() if _TRIAL.fullmatch(path.name))


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def session_seed(study_seed: int, trials: int) -> int:
    """The sampler's seed for a session that starts when the study has ``trials`` trials.

    Derived, so that a resumed session does not replay the proposals the first session started
    with; the same history and seed give the same proposals again.
    """
    digest = hashlib.sha256(f"{study_seed}:{trials}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# --- the lock --------------------------------------------------------------------------------


class Lock:
    """``.lock`` in a study directory while a process works on the study.

    It is created exclusively, holds the host, process and start time of its owner and is
    removed when the owner ends. A lock that a killed process left behind is never taken over:
    the error says whose it is and which file to remove once nothing works on the study.
    """

    def __init__(self, directory: Path) -> None:
        self.path = directory / LOCK_NAME
        self._held = False

    def acquire(self) -> None:
        owner = {"host": socket.gethostname(), "pid": os.getpid(), "started": now()}
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            raise OptimizationError(_in_use(self.path)) from None
        except OSError as exc:
            raise OptimizationError(f"cannot create {self.path}: {exc}") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(owner, handle)
                handle.write("\n")
        except BaseException as exc:
            # a lock without its owner would refuse every later start, as an unknown process's
            self.path.unlink(missing_ok=True)
            if isinstance(exc, OSError):
                raise OptimizationError(f"cannot write {self.path}: {exc}") from exc
            raise
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> Lock:
        self.acquire()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it runs, as another user
        return True
    return True


def _in_use(path: Path) -> str:
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        host, pid, started = owner["host"], owner["pid"], owner["started"]
    except (OSError, ValueError, KeyError, TypeError):
        who, state = "an unknown process (the file cannot be read)", ""
    else:
        who, state = f"process {pid} on {host}, since {started}", ""
        if host == socket.gethostname() and isinstance(pid, int) and not isinstance(pid, bool):
            state = ", which is still running" if _running(pid) else ", which no longer runs"
    return (
        f"the study is in use: {path} names {who}{state}. Once nothing works on the study any "
        f"more, remove {path} and start again"
    )


# --- study.json ------------------------------------------------------------------------------


def write_study(directory: Path, document: Mapping[str, object]) -> None:
    try:
        write_json(directory / STUDY_NAME, document)
    except OSError as exc:
        raise OptimizationError(f"cannot write {directory / STUDY_NAME}: {exc}") from exc


def read_study(directory: Path) -> dict[str, Any]:
    """The ``study.json`` of ``directory``; raises :class:`OptimizationError` if there is none or
    it is not one this k4opt reads."""
    path = directory / STUDY_NAME
    if not path.is_file():
        raise OptimizationError(
            f"{directory} holds no {STUDY_NAME}, so there is no study to resume; start one "
            "without --resume"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OptimizationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != STUDY_SCHEMA:
        schema = document.get("schema") if isinstance(document, dict) else None
        raise OptimizationError(f"{path}: schema {schema!r} is not {STUDY_SCHEMA!r}")
    compared(document)  # every part a resume compares is there
    return document


def _contents(externals: object) -> object:
    """What the external references of a geometry read, without where the files are."""
    if not isinstance(externals, Mapping):
        return externals
    return {
        ref: {k: v for k, v in entry.items() if k != "path"} if isinstance(entry, Mapping) else None
        for ref, entry in externals.items()
    }


def compared(identity: Mapping[str, Any]) -> dict[str, Any]:
    """What must be equal for two runs to belong to one study, from an identity
    (:meth:`k4opt.problem.EvaluationSession.identity`, or ``study.json`` which holds one)."""
    try:
        definition = dict(identity["problem"]["definition"])
        definition.pop("name", None)  # a label, the file name by default
        return {
            "problem": definition,
            "inputs": {
                label: {"bytes": entry.get("bytes"), "sha256": entry.get("sha256")}
                for label, entry in identity["inputs"].items()
            },
            "implementations": {
                name: {
                    "version": entry["version"],
                    "sha256": entry["source"].get("sha256"),
                    # the units its values are in: an instance can declare them otherwise
                    "metrics": entry.get("metrics"),
                    "artifacts": entry.get("artifacts"),
                    "roles": entry.get("roles"),
                }
                for name, entry in identity["implementations"].items()
            },
            "geometry": {
                "fingerprint": identity["geometry"]["fingerprint"],
                "externals": _contents(identity["geometry"]["externals"]),
            },
            "stack": identity["environment"]["KEY4HEP_STACK"],
            "k4opt": identity["k4opt"]["sha256"],
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise OptimizationError(f"a study identity lacks {exc}") from None


def digest(identity: Mapping[str, Any]) -> str:
    """SHA-256 of the :func:`compared` parts of ``identity``: equal digests, one study."""
    text = json.dumps(compared(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def unhashed(identity: Mapping[str, Any]) -> list[str]:
    """Why an identity cannot identify a study: evaluators without a source hash, or a file the
    problem names that could not be read."""
    reasons = [
        f"evaluator {name!r} has no source hash: its module was imported before "
        "load_evaluator could hash it, so load it with load_evaluator in a new process"
        for name, entry in identity["implementations"].items()
        if not entry["source"].get("sha256")
    ]
    reasons += [
        f"{label}: {entry['path']} cannot be read"
        for label, entry in identity["inputs"].items()
        if not entry.get("sha256")
    ]
    return reasons


def differences(study: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """What differs between the study's identity and the current one, each said for a person."""
    then, now_ = compared(study), compared(current)
    found = []
    if then["problem"] != now_["problem"]:
        keys = dict.fromkeys([*then["problem"], *now_["problem"]])
        changed = [k for k in keys if then["problem"].get(k) != now_["problem"].get(k)]
        found.append(f"the problem's {', '.join(changed)} changed")
    for label in dict.fromkeys([*then["inputs"], *now_["inputs"]]):
        if label not in now_["inputs"]:
            found.append(f"the problem no longer names {label}")
        elif label not in then["inputs"]:
            found.append(f"the problem names {label}, which the study did not read")
        elif then["inputs"][label] != now_["inputs"][label]:
            path = current["inputs"][label]["path"]
            found.append(f"{label}: {path} is not the file the study read (other bytes)")
    old_implementations, new_implementations = then["implementations"], now_["implementations"]
    named = [(name, e["version"]) for name, e in old_implementations.items()]
    if named != [(name, e["version"]) for name, e in new_implementations.items()]:
        found.append(f"the evaluators are not {named}")
    else:
        for name, old in old_implementations.items():
            new, entry = new_implementations[name], current["implementations"][name]
            if old["sha256"] != new["sha256"]:
                found.append(f"evaluator {name!r}: {entry['source'].get('path')} changed")
            elif old["metrics"] != new["metrics"]:
                found.append(
                    f"evaluator {name!r} declares other metrics or units: "
                    f"{new['metrics']}, not {old['metrics']}"
                )
            elif old["artifacts"] != new["artifacts"]:
                found.append(
                    f"evaluator {name!r} declares other artifacts: "
                    f"{new['artifacts']}, not {old['artifacts']}"
                )
            elif old["roles"] != new["roles"]:
                found.append(
                    f"evaluator {name!r} declares other roles: {new['roles']}, not {old['roles']}"
                )
    if then["geometry"]["fingerprint"] != now_["geometry"]["fingerprint"]:
        found.append("the geometry the candidates start from changed (another fingerprint)")
    old, new = then["geometry"]["externals"], now_["geometry"]["externals"]
    for ref in dict.fromkeys([*old, *new]):
        if old.get(ref) != new.get(ref):
            path = (current["geometry"]["externals"].get(ref) or {}).get("path")
            found.append(
                f"the geometry's reference {ref} does not read the files the study read (it "
                f"reads {path or 'no file'} now)"
            )
    if then["stack"] != now_["stack"]:
        found.append(f"KEY4HEP_STACK is {now_['stack']!r}, not {then['stack']!r}")
    if then["k4opt"] != now_["k4opt"]:
        found.append(
            "k4opt's code changed; run the study from the code it was made with (a checkout of "
            "its own for a long campaign)"
        )
    return found


def dirty_checkouts(implementations: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The implementations whose git checkout has uncommitted changes: helper modules beside
    their file are then identified by neither its hash nor a commit."""
    return [
        f"evaluator {name!r}: the checkout of {entry['source'].get('path')} has "
        "uncommitted changes, so the helper modules beside it are not identified"
        for name, entry in implementations.items()
        if (entry["source"].get("revision") or {}).get("dirty")
    ]


def revision_warnings(study: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """The implementations whose git checkout is at another commit, or clean or dirty otherwise,
    than when the study was made: their own file is unchanged, but helpers beside it may not
    be. ``study`` and ``current`` hold the same implementations, by name."""
    warnings = []
    for name, old in study.items():
        new = current[name]
        then, now_ = old["source"].get("revision"), new["source"].get("revision")
        if then != now_:
            warnings.append(
                f"evaluator {name!r}: the checkout of {new['source'].get('path')} is at "
                f"{now_} now, at {then} when the study was made; its own file is unchanged, "
                "helper modules beside it may not be"
            )
    return warnings


# --- the record of a trial -------------------------------------------------------------------


def _get(data: object, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
    return data


def _recorded_input(summary: Mapping[str, Any], label: str) -> Mapping[str, Any] | None:
    """The size and SHA-256 a record holds for the file ``label`` names
    (``samples.<sample>.<stage>.file`` or ``.<option>``), ``None`` if its stage did not get as
    far as reading it."""
    parts = label.split(".", 3)
    if len(parts) != 4 or parts[0] != "samples":
        return None  # a supplied file: a study supplies none
    _, sample, stage, rest = parts
    if rest == "file":
        entry = _get(summary, "samples", sample, "stages", stage)
    else:
        entry = _get(summary, "samples", sample, "stages", stage, "files", rest)
    if not isinstance(entry, Mapping) or "sha256" not in entry:
        return None
    return entry


def check_record(
    study: Mapping[str, Any], summary: object, number: int, params: Mapping[str, Any]
) -> list[str]:
    """Refuse a record that is not the one trial ``number`` of ``study`` asked for with
    ``params``, or was made with something the study does not compare equal; return the
    warnings of :func:`revision_warnings`. Its assessment is checked apart, by
    :meth:`k4opt.problem.Assessment.from_summary`."""
    where = f"trial {number}: {trial_directory(number)}/summary.json"
    if not isinstance(summary, Mapping):
        raise OptimizationError(f"{where} is not a run record")
    if summary.get("run") != {"study": study["id"], "trial": number}:
        raise OptimizationError(f"{where} was made for {summary.get('run')!r}, not this trial")
    candidate = summary.get("candidate")
    # as given, type and value: 1, 1.0 and True are different candidates
    given = list(candidate.items()) if isinstance(candidate, Mapping) else None
    if given is None or [(k, type(v), v) for k, v in given] != [
        (k, type(v), v) for k, v in params.items()
    ]:
        raise OptimizationError(f"{where} holds candidate {candidate!r}, but the trial is {params}")
    if summary.get("represents_candidate") is not True or summary.get("supplied") != []:
        raise OptimizationError(f"{where} was not made from its candidate alone (a supplied stage)")
    implementations = summary.get("implementations")
    if not isinstance(implementations, Mapping) or list(implementations) != list(
        study["implementations"]
    ):
        raise OptimizationError(f"{where} does not hold the study's evaluators")
    for name, old in study["implementations"].items():
        new = implementations[name]
        if _get(new, "source", "sha256") != old["source"].get("sha256"):
            raise OptimizationError(f"{where}: evaluator {name!r} ran other code than the study's")
        if _get(new, "metrics") != old.get("metrics"):
            raise OptimizationError(
                f"{where}: evaluator {name!r} declared other metrics or units than in the study"
            )
        if _get(new, "artifacts") != old.get("artifacts"):
            raise OptimizationError(
                f"{where}: evaluator {name!r} declared other artifacts than in the study"
            )
        if _get(new, "roles") != old.get("roles"):
            raise OptimizationError(
                f"{where}: evaluator {name!r} declared other roles than in the study"
            )
    observed = _get(summary, "environment", "observed")
    if _get(observed, "KEY4HEP_STACK") != study["environment"]["KEY4HEP_STACK"]:
        raise OptimizationError(f"{where} ran in another KEY4HEP_STACK than the study")
    if _get(observed, "k4opt_source", "sha256") != study["k4opt"]["sha256"]:
        raise OptimizationError(f"{where} was made by other k4opt code than the study")
    externals = _get(summary, "geometry", "externals")
    if externals is None or _contents(externals) != _contents(study["geometry"]["externals"]):
        raise OptimizationError(
            f"{where} read other files through the geometry's external references than the study"
        )
    for label, entry in study["inputs"].items():
        recorded = _recorded_input(summary, label)
        if recorded is not None and (recorded.get("bytes"), recorded.get("sha256")) != (
            entry.get("bytes"),
            entry.get("sha256"),
        ):
            raise OptimizationError(f"{where} read another {label} than the study")
    return revision_warnings(study["implementations"], implementations)


def check_candidate_files(
    read: dict[str, tuple[int, object]], summary: object, number: int
) -> None:
    """Refuse the record of trial ``number`` if an external reference that only candidates name
    (``geometry.candidate_externals``: a ``ref`` parameter choosing ``${CAD}/other.stl``) reads
    other files than for an earlier trial of the study; then add what it read to ``read``, by
    reference: the first trial that read it and the files' content."""
    where = f"trial {number}: {trial_directory(number)}/summary.json"
    contents = _contents(_get(summary, "geometry", "candidate_externals"))
    if not isinstance(contents, dict):
        raise OptimizationError(f"{where} does not say what its candidate's own references read")
    for ref, content in contents.items():
        if ref in read and read[ref][1] != content:
            raise OptimizationError(
                f"{where}: {ref} reads other files than for trial {read[ref][0]}, so the two "
                "trials did not evaluate one geometry"
            )
    for ref, content in contents.items():
        read.setdefault(ref, (number, content))


# --- reports ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialReport:
    """One trial of a study, as its record and the optimizer know it.

    ``status`` is the candidate's (``valid``, ``invalid``, ``failed``), or ``interrupted`` for a
    trial that ended without a record, or ``error`` for one that stopped the study (its
    ``message`` says why). ``feasible`` and ``objectives`` (by ``evaluator/stage/metric``, in the
    problem's order) are those of a ``valid`` trial; ``record`` is the ``id`` of its
    ``summary.json``, and ``diagnostics`` the number of its diagnostics that did not succeed,
    which its status leaves out. ``recovered`` marks a trial an interrupted session left running
    and a resume finished.
    """

    number: int
    directory: Path
    candidate: Mapping[str, Any]
    status: str
    feasible: bool | None = None
    objectives: Mapping[str, float] | None = None
    record: str | None = None
    message: str | None = None
    recovered: bool = False
    diagnostics: int = 0


@dataclass(frozen=True, slots=True)
class StudyReport:
    """A study after a call of :func:`k4opt.optimization.optimize`: all its trials, in order.

    ``pareto`` holds the numbers of the trials on the Pareto front of the feasible ``valid``
    ones; it is empty while there is none.
    """

    name: str
    directory: Path
    id: str
    trials: tuple[TrialReport, ...]
    pareto: tuple[int, ...]
    objectives: tuple[str, ...]
    constraints: tuple[str, ...]

    @property
    def study_path(self) -> Path:
        return self.directory / STUDY_NAME
