"""What a publication would expose that its author may not mean to: secrets and local paths.

:func:`secrets_in` finds what looks like a secret in some bytes, best effort: enough to stop an
accident, no substitute for looking. :func:`k4opt.publication.scan` looks through the files offered
beside a snapshot with it, :func:`public_snapshot` through the snapshot, and :func:`exposed`
through any JSON, such as a software sheet. In JSON, a value under a key that names a credential
is one as well (``db_password: <the password>``, written in YAML), which neither text says alone.

:func:`public_snapshot` is a study's snapshot as it is published: its JSON form, but that each
message of a trial (its own, and those of its diagnostics, at every level of a nested study) that
looks like it holds a secret is withheld, replaced by a note that says what it looked like. A
message is what a program printed or an evaluator wrote, which the study's author cannot change,
and nothing reads it back. What looks like a secret anywhere else is a finding, which blocks
publishing: the problem's definition and ``study.json`` are the study itself, whose identity
digest a reader checks, so they are published as they are or not at all. Each is checked as text,
as a reader of the snapshot reads it, not as JSON escapes it (``password=\\"...\\"``).

:func:`local_paths` names each absolute path outside ``/cvmfs`` that the snapshot carries where a
person writes text.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

from k4opt.optimization import StudyResults

PUBLIC_PREFIXES = ("/cvmfs/",)
"""Absolute paths that are public by nature, which :func:`local_paths` does not report."""

# an absolute path in text: a / that starts the text or follows a space, a quote, = or (, then
# what a path holds; "1/mm", "GeV/c" and "${K4GEO}/x" are no absolute paths
_ABSOLUTE = re.compile(r"(?:^|(?<=[\s\"'=(]))/[\w.+-][^\s\"',;)]*")


@dataclass(frozen=True, slots=True)
class Finding:
    """A secret a scan saw: ``what`` it looks like, in which file or field (``label``), at
    ``line`` of it; ``snapshot`` if it is in the snapshot, the study itself, not in a file offered
    beside it."""

    label: str
    line: int
    what: str
    snapshot: bool = False


# what names a credential: a name or a key that ends in one of these, in any case (DB_PASSWORD,
# githubToken, private_key, JWT_SECRET_KEY, client-secret, credentials), alike in text and in a
# mapping; not tokenizer, candidate_key or public_key, whose last words name no secret
_CREDENTIAL = (
    r"(?:password|passwd|passphrase|pwd|secret|token|credentials?"
    r"|(?:api|access|private|secret|signing|encryption)[_-]?key)"
)

# what is looked for, best effort: enough to stop an accident, no substitute for looking
_SECRETS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("a private key", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")),
    ("an AWS access key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{22,})\b")),
    ("a GitLab token", re.compile(rb"\bglpat-[\w-]{20,}\b")),
    ("a Slack token", re.compile(rb"\bxox[abprs]-[\w-]{10,}\b")),
    ("a Google API key", re.compile(rb"\bAIza[\w-]{35}\b")),
    (
        "a URL with a password",
        re.compile(rb"\b[a-z][a-z0-9+.-]{0,31}://[^/\s:@'\"${}]{1,256}:[^/\s@'\"${}]{1,256}@"),
    ),
    (
        "a secret assigned in the open",
        re.compile(
            rb"(?i)"
            + _CREDENTIAL.encode()
            # the name ends there, and may be quoted, as JSON and a dict write it; a quoted value
            # that starts like a path or a variable (${API_KEY}) refers to one
            + rb"[\"']?\s*[:=]\s*(?:[\"'](?![/$~])[^\"'\s]{8,256}[\"']"
            # unquoted: what a token looks like, not a name, a call or a path
            rb"|(?=[\w+=-]{0,255}\d)[\w+=-]{8,256}(?![\w+=(.\[/-]))"
        ),
    ),
)
# files named as what holds credentials: an environment file (.env, .env.local, .env.production;
# not a template meant to be shared, .env.example, whose content is scanned as any file's), direnv's
# .envrc, a key or a certificate store
_CREDENTIAL_NAMES = re.compile(
    r"(?:^|/)(?:\.netrc|\.pgpass|\.htpasswd|\.envrc|id_(?:rsa|dsa|ecdsa|ed25519)|credentials"
    r"|\.env(?:\.(?!(?:example|sample|template|dist)$)[^/]+)?)$"
    r"|\.(?:pem|p12|pfx|key)$"
)

# a key that names a credential, and what such a key's value is when it is one: a word of 8 to
# 256 characters, not a path or a variable
_CREDENTIAL_KEY = re.compile(_CREDENTIAL + "$", re.IGNORECASE)
_CREDENTIAL_VALUE = re.compile(r"(?![/$~])\S{8,256}")
KEYED = "a secret, by the key it is under"
"""What a value under a key that names a credential is found as (:func:`exposed`)."""

WITHHELD = "[withheld by k4opt publish: it looks like it holds {what}]"
"""What a message that looks like it holds a secret is published as."""


def said_of_path(where: str, path: str) -> str:
    """What a publisher is told of a local path the snapshot carries (:func:`local_paths`)."""
    return f"{where} holds the local path {path}; it is published"


def said_of_withheld(withheld: Finding) -> str:
    """What a publisher is told of a message withheld (:func:`public_snapshot`)."""
    return f"{withheld.label} looks like it holds {withheld.what}: published as a note saying so"


def secrets_in(data: bytes) -> list[tuple[str, int, bytes]]:
    """What of ``data`` looks like a secret: the first match of each kind, as what it looks like,
    its offset and the matched bytes."""
    found = []
    for what, pattern in _SECRETS:
        match = pattern.search(data)
        if match is not None:
            found.append((what, match.start(), match[0]))
    return found


def credentials_named(name: str) -> bool:
    """Whether a file's ``name`` (a path relative to what it is in) says it holds credentials."""
    return _CREDENTIAL_NAMES.search(name) is not None


def _texts(value: object, path: str = "") -> Iterator[tuple[str, str]]:
    """Each text of the JSON-like ``value``, keys included, with where it is in it (``a.b[0]``)."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield path, str(key)
            yield from _texts(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _texts(item, f"{path}[{index}]")


def _entries(value: object, path: str = "") -> Iterator[tuple[str, str, object]]:
    """Each entry of every mapping in the JSON-like ``value``: where its value is, its key and its
    value."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            at = f"{path}.{key}" if path else str(key)
            yield at, str(key), item
            yield from _entries(item, at)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _entries(item, f"{path}[{index}]")


def _keyed(key: str, item: object) -> bool:
    """Whether ``item`` is a credential by the ``key`` it is under: a key that names one, as an
    assignment in text does (:data:`_CREDENTIAL`), and a value that is one
    (:data:`_CREDENTIAL_VALUE`)."""
    if isinstance(item, bool) or not isinstance(item, (str, int)):
        return False
    return bool(_CREDENTIAL_KEY.search(key)) and bool(_CREDENTIAL_VALUE.fullmatch(str(item)))


def _line(text: bytes, offset: int) -> int:
    return text.count(b"\n", 0, offset) + 1


def _found(value: object, owner: str, snapshot: bool) -> Iterator[tuple[Finding, bytes]]:
    """What of ``value`` looks like a secret, labeled by ``owner`` and the field
    (``the study's problem.definition.x``), with what matched: in each text, and each value whose
    key names a credential, which no text of it says alone."""
    for path, text in _texts(value):
        data = text.encode()
        for what, offset, matched in secrets_in(data):
            label = f"{owner} {path}" if path else owner
            yield Finding(label, _line(data, offset), what, snapshot), matched
    for path, key, item in _entries(value):
        if _keyed(key, item):
            yield Finding(f"{owner} {path}", 1, KEYED, snapshot), str(item).encode()


def exposed(value: object, owner: str) -> list[Finding]:
    """What of the JSON-like ``value`` looks like a secret, each text read as text, labeled by
    ``owner`` and the field it is in (``the software sheet's requirements[2].label``)."""
    return [finding for finding, _ in _found(value, owner, False)]


@dataclass(frozen=True, slots=True)
class PublicSnapshot:
    """A study's snapshot as published (:func:`public_snapshot`): its ``text``; the ``findings``
    that block publishing it; the messages ``withheld`` (where, and what each looked like it
    holds); and the local ``paths`` it carries (:func:`local_paths`)."""

    text: str
    findings: tuple[Finding, ...]
    withheld: tuple[Finding, ...]
    paths: tuple[tuple[str, str], ...]

    @property
    def data(self) -> bytes:
        return self.text.encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def public_snapshot(results: StudyResults) -> PublicSnapshot:
    """The snapshot of ``results`` as it is published (see the module): its JSON form with each
    message that looks like it holds a secret withheld, what looks like one anywhere else, the
    first place of each, and the local paths it carries."""
    withheld: list[Finding] = []
    shown = _withheld(results, "", withheld)
    seen: set[tuple[str, bytes]] = set()
    findings = []
    for finding, matched in _snapshot_found(shown.to_dict(), "the study's", ""):
        if (finding.what, matched) not in seen:  # a nested study holds the problem at each level
            seen.add((finding.what, matched))
            findings.append(finding)
    return PublicSnapshot(
        shown.to_json(), tuple(findings), tuple(withheld), tuple(local_paths(shown))
    )


def _withheld_text(text: str | None, where: str, withheld: list[Finding]) -> str | None:
    if text is None:
        return None
    data = text.encode()
    found = secrets_in(data)
    if not found:
        return text
    withheld.extend(Finding(where, _line(data, offset), what, True) for what, offset, _ in found)
    return WITHHELD.format(what=" and ".join(what for what, _, _ in found))


def _withheld(results: StudyResults, trials: str, withheld: list[Finding]) -> StudyResults:
    """``results`` with each message of its trials, at every level, that looks like it holds a
    secret replaced by :data:`WITHHELD`; each noted in ``withheld``."""
    kept = []
    for trial in results.trials:
        own = f"{trials}trial {trial.number}"
        message = _withheld_text(trial.message, f"{own}'s message", withheld)
        diagnostics = tuple(
            {
                **entry,
                "message": _withheld_text(
                    entry.get("message"), f"{own}'s diagnostic {entry.get('task')}", withheld
                ),
            }
            for entry in trial.diagnostics
        )
        inner = None if trial.inner is None else _withheld(trial.inner, f"{own}'s inner ", withheld)
        kept.append(replace(trial, message=message, diagnostics=diagnostics, inner=inner))
    return replace(results, trials=tuple(kept))


def _snapshot_found(
    data: Mapping[str, object], study: str, trials: str
) -> Iterator[tuple[Finding, bytes]]:
    """What of the JSON form ``data`` looks like a secret: in its ``study.json`` (labeled by
    ``study``) and in each trial, at every level (``trials`` names the trial a level is of)."""
    yield from _found(data["study"], study, True)
    listed = data["trials"]
    assert isinstance(listed, list)
    for trial in listed:
        own = f"{trials}trial {trial['number']}"
        rest = {key: value for key, value in trial.items() if key != "inner"}
        yield from _found(rest, f"{own}'s", True)
        if trial["inner"] is not None:
            yield from _snapshot_found(trial["inner"], f"{own}'s inner study's", f"{own}'s inner ")


def local_paths(results: StudyResults) -> list[tuple[str, str]]:
    """Each absolute path, outside :data:`PUBLIC_PREFIXES`, that the JSON form of ``results``
    carries, with where it is: in the problem's definition as written (a stage's ``file``, an
    evaluator setting), anywhere else in ``study.json`` (an external reference of the geometry
    written as an absolute path, ``the study's geometry.externals``) and in the trials, their
    messages and diagnostics, at every level of a nested study (``trial 3's inner trial 12's
    message``). A path that ``study.json`` repeats at a level below is named once; the locator of
    a geometry target, an XPath (``/lccdd/define/constant``), is no path."""
    found: list[tuple[str, str]] = []
    named: set[str] = set()  # of study.json, at any level

    def scan(value: object, where: str, join: str = " ", once: bool = False) -> None:
        for path, text in _texts(value):
            for local in _ABSOLUTE.findall(text):
                if local.startswith(PUBLIC_PREFIXES) or (once and local in named):
                    continue
                if once:
                    named.add(local)
                found.append((f"{where}{join}{path}" if path else where, local))

    def level(
        data: Mapping[str, object], study: str, trials: str, definition: object = None
    ) -> None:
        """The paths of one level: its ``study.json`` (labeled by ``study``), its problem's
        definition as its author wrote it, and its trials (``trials`` names what they are of)."""
        document = data.get("study")
        document = document if isinstance(document, Mapping) else {}
        problem = document.get("problem")
        problem = problem if isinstance(problem, Mapping) else {}
        top = study == "the study's"
        written = problem.get("definition", definition)
        where = "the problem's definition" if top else f"{study} problem.definition"
        scan(written, where, ".", once=True)
        rest = {key: value for key, value in document.items() if key != "problem"}
        rest["problem"] = {key: value for key, value in problem.items() if key != "definition"}
        geometry = document.get("geometry")
        if isinstance(geometry, Mapping) and isinstance(geometry.get("targets"), list):
            rest["geometry"] = {
                **geometry,
                "targets": [
                    {key: value for key, value in target.items() if key != "locator"}
                    if isinstance(target, Mapping)
                    else target
                    for target in geometry["targets"]
                ],
            }
        scan(rest, study, once=True)
        listed = data.get("trials")
        for trial in listed if isinstance(listed, list) else ():
            own = f"{trials}trial {trial['number']}"
            scan(trial.get("message"), f"{own}'s message")
            for diagnostic in trial.get("diagnostics") or ():
                scan(diagnostic.get("message"), f"{own}'s diagnostic {diagnostic.get('task')}")
            kept = ("message", "diagnostics", "inner")
            scan({key: value for key, value in trial.items() if key not in kept}, f"{own}'s")
            if isinstance(trial.get("inner"), Mapping):
                level(trial["inner"], f"{own}'s inner study's", f"{own}'s inner ")

    level(results.to_dict(), "the study's", "", results.problem.to_dict())
    return found
