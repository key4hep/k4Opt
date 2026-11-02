"""Internal: what makes two runs belong to one study, compared between a study and a session.

Whether a study may be resumed, and whether the record of a trial belongs to its study, are
decided by one comparison (:func:`compared`) of what must be equal: the problem's definition (but
not its name), the content of every file it names, each evaluator implementation's name, version,
source hash, declared metrics and declared artifacts, the unmodified geometry's fingerprint, the
attribute each geometry destination of the parameters resolves to and the content of the files
its external references read, ``KEY4HEP_STACK`` and the hash of k4opt's code. Locations are
recorded and never compared, so a study whose files moved resumes. :func:`digest` condenses the
comparison, :func:`differences` says what differs, and the revision warnings say what a hash
leaves out (an evaluator checkout with uncommitted changes, or at another commit).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from k4opt.optimization.errors import OptimizationError
from k4opt.problem.models import ordered


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
    (:meth:`k4opt.execution.EvaluationSession.identity`, or ``study.json`` which holds one)."""
    try:
        definition = ordered(identity["problem"]["definition"])
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
                    "helpers": entry["source"].get("helpers"),
                    # the units its values are in: an instance can declare them otherwise
                    "metrics": entry.get("metrics"),
                    "artifacts": entry.get("artifacts"),
                    "roles": entry.get("roles"),
                }
                for name, entry in identity["implementations"].items()
            },
            "geometry": {
                "fingerprint": identity["geometry"]["fingerprint"],
                # paths relative to the geometry's root: content, not where it is
                "targets": identity["geometry"]["targets"],
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
    """Why an identity cannot identify a study: evaluators without a source hash, or whose hash
    is not known to be that of the code that runs (``executed``), or a file the problem names
    that could not be read."""
    reasons = []
    for name, entry in identity["implementations"].items():
        source = entry["source"]
        if not source.get("sha256"):
            reasons.append(
                f"evaluator {name!r} has no source hash: its module was imported before "
                "load_evaluator could hash it, or is no source file on disk (in a ZIP archive, "
                "say), so load it with load_evaluator in a new process, from a source file"
            )
        elif source.get("executed") is not True:
            reasons.append(
                f"evaluator {name!r}: the code it runs is not known to be the source its hash "
                "names (an import hook imported it, or a module it imports, perhaps from a cached "
                ".pyc), so load it from source files Python's path finder imports, or as "
                "FILE.py:Class"
            )
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
            elif old["helpers"] != new["helpers"]:
                then_, now_helpers = old["helpers"] or {}, new["helpers"] or {}
                changed = [
                    module
                    for module in dict.fromkeys([*then_, *now_helpers])
                    if then_.get(module) != now_helpers.get(module)
                ]
                found.append(
                    f"evaluator {name!r}: the modules {entry['source'].get('path')} imports "
                    f"changed: {', '.join(changed)}"
                )
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
    if then["geometry"]["targets"] != now_["geometry"]["targets"]:
        found.append("the parameters set other attributes of the geometry")
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
    """The implementations whose git checkout has uncommitted changes: what the hashes of their
    file and the modules it imports leave out (data files, modules imported by name at run time)
    is then identified by no commit either."""
    return [
        f"evaluator {name!r}: the checkout of {entry['source'].get('path')} has "
        "uncommitted changes, so files beside it that it does not import (data, modules imported "
        "at run time) are not identified"
        for name, entry in implementations.items()
        if (entry["source"].get("revision") or {}).get("dirty")
    ]


def revision_warnings(study: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """The implementations whose git checkout is at another commit, or clean or dirty otherwise,
    than when the study was made: their file and the modules it imports are unchanged, but
    other files beside them may not be. ``study`` and ``current`` hold the same
    implementations, by name."""
    warnings = []
    for name, old in study.items():
        new = current[name]
        then, now_ = old["source"].get("revision"), new["source"].get("revision")
        if then != now_:
            warnings.append(
                f"evaluator {name!r}: the checkout of {new['source'].get('path')} is at "
                f"{now_} now, at {then} when the study was made; its file and the modules it "
                "imports are unchanged, other files beside them may not be"
            )
    return warnings
