"""What looks like a credential: in bytes, in JSON-like data, by a file's or a variable's name.

Best effort: enough to stop an accident, no substitute for looking. :func:`secrets_in` finds what
looks like a secret in some bytes; :func:`suspected` in any JSON-like value, where a value under a
key that names a credential is one as well (``db_password: <the password>``, written in YAML),
which neither text says alone; :func:`credentials_named` says whether a file's name says it holds
credentials, :func:`names_credential` whether a key or an environment variable names one
(``DB_PASSWORD``, ``githubToken``, ``ANTHROPIC_API_KEY``). Shared by every component that must
not record, publish or compare a credential: what a publication would expose
(:mod:`k4opt.publication`), the variables of a study's software identity (:mod:`k4opt.software`)
and what a strategy of one's own gives a study to record (:mod:`k4opt.optimization`).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping

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
"""What a value under a key that names a credential is found as (:func:`suspected`)."""


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


def names_credential(name: str) -> bool:
    """Whether a key or a variable ``name`` names a credential: it ends in what names one, in any
    case (``ANTHROPIC_API_KEY``, ``HF_TOKEN``, ``db_password``)."""
    return _CREDENTIAL_KEY.search(name) is not None


def line_of(data: bytes, offset: int) -> int:
    """The line of ``data`` that ``offset`` is on, from 1."""
    return data.count(b"\n", 0, offset) + 1


def texts(value: object, path: str = "") -> Iterator[tuple[str, str]]:
    """Each text of the JSON-like ``value``, keys included, with where it is in it (``a.b[0]``)."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield path, str(key)
            yield from texts(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from texts(item, f"{path}[{index}]")


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
    return names_credential(key) and bool(_CREDENTIAL_VALUE.fullmatch(str(item)))


def suspected(value: object) -> Iterator[tuple[str, int, str, bytes]]:
    """What of the JSON-like ``value`` looks like a secret: where it is (``a.b[0]``, empty for
    ``value`` itself), the line of that text, what it looks like and what matched. In each text,
    read as text, not as JSON escapes it; then each value whose key names a credential
    (:data:`KEYED`), which no text of it says alone."""
    for path, text in texts(value):
        data = text.encode()
        for what, offset, matched in secrets_in(data):
            yield path, line_of(data, offset), what, matched
    for path, key, item in _entries(value):
        if _keyed(key, item):
            yield path, 1, KEYED, str(item).encode()
