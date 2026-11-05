"""What a pipeline stage of a candidate is made from, as one document, and its SHA-256: the
stage's *signature*.

Two stages with equal signatures were made from the same things: the same stage type, executor
and definition (events, seed, options, as the candidate configures them), the same inputs (the
content of the geometry it reads, and the signature or file of each stage it reads), the same
files the problem names for it (by content), the same program of the stack (its executable, by
content, and the interpreter it runs in), the same ``KEY4HEP_STACK`` and search variables (but
``PATH``, of which only that interpreter matters), and the same k4opt code. The sample a
stage belongs to, its time limit, how many workers ran it and where its files and its program
are play no part (the record says where): a stage of another sample or another candidate with
the same signature made the same events (as far as the program is deterministic for a seed).

The writer of a record (:mod:`k4opt.record.write`, through the execution) and its reader
(:mod:`k4opt.record.read`) build the document with the same functions here, so that a record's
signature is checked against what the record itself says.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

from k4opt.frozen import Json, thaw
from k4opt.problem.models import Stage

SIGNATURE = "signature"
"""Where a stage's part of a record holds its signature: ``{"sha256": …, "document": …}``."""


def digest(document: Mapping[str, object]) -> str:
    """SHA-256 of ``document`` written as canonical JSON (sorted keys, no spaces)."""
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _content(entry: object) -> object:
    """What an external reference reads, without where its files are."""
    if not isinstance(entry, Mapping):
        return None
    return {key: value for key, value in entry.items() if key != "path"}


def geometry_identity(geometry: Mapping[str, object]) -> dict[str, object] | None:
    """What a geometry is made of, from what its task recorded (``fingerprint``, ``externals``,
    ``candidate_externals``): its fingerprint and the content of what its external references
    read, without paths; ``None`` for a geometry without a fingerprint (one given, not made)."""
    fingerprint = geometry.get("fingerprint")
    if not isinstance(fingerprint, str):
        return None
    references = {}
    for key in ("externals", "candidate_externals"):
        entries = geometry.get(key)
        entries = entries if isinstance(entries, Mapping) else {}
        references[key] = {ref: _plain(_content(entry)) for ref, entry in entries.items()}
    return {"fingerprint": fingerprint, **references}


def given_identity(provenance: Mapping[str, object]) -> dict[str, object] | None:
    """What a stage given as a file (by the problem or the caller) is: the size and SHA-256 its
    task recorded of the file; ``None`` if it could not be read."""
    size, sha256 = provenance.get("bytes"), provenance.get("sha256")
    if not isinstance(size, int) or not isinstance(sha256, str):
        return None
    return {"bytes": size, "sha256": sha256}


def stage_document(
    name: str,
    stage: Stage,
    *,
    reads: Mapping[str, object],
    files: Mapping[str, object],
    program: object,
    environment: Mapping[str, object],
    k4opt: object,
) -> dict[str, object]:
    """The signature document of the stage ``name``, run as ``stage`` (as its candidate
    configures it): what it ``reads`` by name (``geometry``: :func:`geometry_identity`; a stage:
    its signature's SHA-256, or the :func:`given_identity` of its file), the ``files`` the problem
    names for it by key (size and SHA-256), the size and SHA-256 of the ``program`` it runs, the
    ``environment`` (``KEY4HEP_STACK`` and the search variables) and the SHA-256 of k4opt's
    code."""
    definition = _plain(stage.to_dict())
    assert isinstance(definition, dict)
    definition.pop("timeout", None)  # how long it may take, not what it makes
    return {
        "stage": name,
        "definition": definition,
        "reads": _plain(reads),
        "files": {key: _identity(value) for key, value in files.items()},
        "program": _program(program),
        "environment": _signed_environment(environment),
        "k4opt": k4opt,
    }


def _plain(value: object) -> Any:
    """``value`` as plain JSON data (dicts and lists), as a record read back holds it."""
    return thaw(cast(Json, value))


def _program(entry: object) -> dict[str, object] | None:
    """A program by its size and SHA-256, not where it is, and the interpreter it runs in, with
    its path (a virtual environment's interpreter finds its packages from where it is);
    ``None`` for a stage that runs none, or one that could not be read."""
    identity = _identity(entry)
    if identity is None:
        return None
    assert isinstance(entry, Mapping)
    return {**identity, "interpreter": _plain(entry.get("interpreter"))}


def _signed_environment(environment: Mapping[str, object]) -> dict[str, object]:
    """``environment`` without ``PATH`` among the search variables: of ``PATH``, a program's
    interpreter is what decides what it runs, and that is signed with the program. The rest of
    ``PATH`` differs between shells that run the same things (an editor's terminal, a batch
    job), which would keep stages from being reused for nothing."""
    plain = _plain(environment)
    assert isinstance(plain, dict)
    search = plain.get("search")
    if isinstance(search, dict):
        plain["search"] = {name: value for name, value in search.items() if name != "PATH"}
    return plain


def _identity(entry: object) -> dict[str, object] | None:
    """A file's size and SHA-256 without its path; ``None`` for one that could not be read."""
    if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
        return None
    return {"bytes": entry.get("bytes"), "sha256": entry.get("sha256")}


def signed(document: Mapping[str, object]) -> dict[str, object]:
    """A stage's signature, as its record holds it: the ``document`` and its SHA-256."""
    plain = _plain(document)
    return {"sha256": digest(plain), "document": plain}
