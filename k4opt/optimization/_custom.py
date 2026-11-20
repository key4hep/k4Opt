"""Internal: a strategy of one's own, a class of the user's code that proposes a study's candidates.

It is named as ``FILE.py:Class`` or ``package.module:Class`` and loaded as any class of the user's
code is (:func:`load_strategy`, :mod:`k4opt.usercode`): it must have ``propose``, ``proposed`` and
``tell`` (:class:`~k4opt.optimization._proposal.Strategy`), and is made for a study as
``Class(context, **settings)`` (:class:`~k4opt.optimization._proposal.StudyContext`).

What it is given has three roles (:class:`Custom`): its *settings*, plain JSON passed as keyword
arguments and recorded as they are; the *files* some of them name, those the class lists in its
``files`` attribute (a prompt, say), identified by their bytes and SHA-256 and given to it as
absolute paths; and *credentials*, which it reads where it runs and which a study records nowhere:
settings that look like they hold one are refused (:mod:`k4opt.credentials`). ``study.json``
records the strategy (:meth:`Custom.record`) by the name of its class, its source as an
evaluator's is identified (without where it is), its settings but files and its files by content;
a resume is given the same class and settings again, and refused if anything of them differs
(:func:`differences`). A session refuses to keep a proposal once its code or a file changed
(:meth:`Custom.changed`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from k4opt.credentials import suspected
from k4opt.errors import CodeLoadError
from k4opt.frozen import FrozenMap, Json, freeze, thaw
from k4opt.hashing import file_identity
from k4opt.optimization._proposal import Strategy, StudyContext
from k4opt.optimization.errors import OptimizationError
from k4opt.usercode import changed_code, load_class, source_of

NAME = "custom"
"""What ``study.json`` records as the name of a strategy of one's own."""

METHODS = ("propose", "proposed", "tell")
"""What a strategy of one's own must have."""

_SOURCE = ("bytes", "sha256", "helpers", "executed", "revision")
"""What ``study.json`` records of the strategy's source: as an evaluator's, without where it is."""


def load_strategy(reference: str) -> type:
    """The strategy class that ``reference`` names, ``FILE.py:Class`` or
    ``package.module:Class``, loaded and identified as :func:`k4opt.usercode.load_class` loads
    any class of the user's code; :class:`OptimizationError` if it cannot be, or has no
    ``propose``, ``proposed`` or ``tell``."""
    try:
        return load_class(reference, "strategy", METHODS)
    except CodeLoadError as exc:
        raise OptimizationError(str(exc)) from exc


def declared_files(strategy: type) -> tuple[str, ...]:
    """The settings ``strategy`` says name files (its ``files`` attribute)."""
    files = getattr(strategy, "files", ())
    if isinstance(files, str) or not isinstance(files, Sequence):
        raise OptimizationError(f"{strategy.__name__}.files is {files!r}, not a tuple of settings")
    if not all(isinstance(name, str) and name.isidentifier() for name in files):
        raise OptimizationError(f"{strategy.__name__}.files names no settings: {files!r}")
    return tuple(files)


def resolved(
    strategy: type, settings: Mapping[str, Any], base: str | os.PathLike[str]
) -> dict[str, Any]:
    """``settings`` with each setting ``strategy`` says names a file (:func:`declared_files`) as
    an absolute path, a relative one being relative to ``base``."""
    given = dict(settings)
    for name in declared_files(strategy):
        value = given.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise OptimizationError(f"strategy setting {name}: {value!r} names no file")
        given[name] = os.path.abspath(Path(base) / Path(value).expanduser())
    return given


@dataclass(frozen=True, slots=True)
class Custom:
    """A strategy of one's own as a session runs it: its class (``strategy``), the settings it is
    made with but files (``settings``), the files the others name, by setting, as absolute paths
    (``files``), and their bytes and SHA-256 when it was made (``identities``). Made by
    :meth:`of`, which checks them."""

    strategy: type
    settings: Mapping[str, Json]
    files: Mapping[str, Path]
    identities: Mapping[str, Mapping[str, Json]]

    @classmethod
    def of(
        cls,
        strategy: type,
        settings: Mapping[str, Any] | None,
        base: str | os.PathLike[str] = ".",
    ) -> Custom:
        """``strategy``, which :func:`load_strategy` loaded, with ``settings`` (a JSON object; a
        relative path of a file setting is relative to ``base``). Raises
        :class:`OptimizationError` for a class whose source is not known to be the code that
        runs, settings that are no JSON object or look like they hold a credential, and a file
        setting that names no readable file."""
        source = source_of(strategy)
        name = _name(strategy)
        if not source.get("reference") or not source.get("sha256"):
            raise OptimizationError(
                f"strategy {name}: its source is not identified; load it with load_strategy, "
                "in a new process, from a source file"
            )
        if source.get("executed") is not True:
            raise OptimizationError(
                f"strategy {name}: the code it runs is not known to be the source its hash names "
                "(an import hook imported it, or a module it imports); load it as FILE.py:Class"
            )
        plain = freeze(
            dict(resolved(strategy, settings or {}, base)), "settings", OptimizationError
        )
        if not isinstance(plain, Mapping):
            raise OptimizationError(f"strategy {name}: its settings are no JSON object")
        for where, _, what, _ in suspected(plain):
            raise OptimizationError(
                f"strategy {name}: its setting {where} looks like it holds {what}, which a study "
                "records nowhere; let the strategy read it where it runs (an environment "
                "variable, a file of its own)"
            )
        names = declared_files(strategy)
        files = {
            key: Path(value)
            for key, value in plain.items()
            if key in names and isinstance(value, str)
        }
        for key, path in files.items():
            if not path.is_file():
                raise OptimizationError(f"strategy setting {key}: {path} is no file")
        identities = FrozenMap(
            {key: FrozenMap(_identity(key, path)) for key, path in files.items()}
        )
        kept = FrozenMap({key: value for key, value in plain.items() if key not in files})
        return cls(strategy, kept, FrozenMap(files), identities)

    @property
    def name(self) -> str:
        return _name(self.strategy)

    def record(self) -> dict[str, Any]:
        """What ``study.json`` records of it: no location, so that a study moved resumes and its
        JSON form keeps it whole."""
        source = source_of(self.strategy)
        return {
            "name": NAME,
            "class": self.name,
            "source": {key: thaw(source.get(key)) for key in _SOURCE},
            "settings": thaw(self.settings),
            "files": thaw(self.identities),
        }

    def kept(self) -> dict[str, Any]:
        """What a study keeps privately of it (:func:`k4opt.software.keep.take`): its name, its
        source, where it is, and the files its settings name, where they are."""
        return {
            "name": self.name,
            "source": source_of(self.strategy),
            "files": {key: str(path) for key, path in self.files.items()},
        }

    def make(self, context: StudyContext) -> Strategy:
        """The strategy of the study of ``context``: ``Class(context, **settings)``, each file
        setting an absolute path. Raises :class:`OptimizationError` if it cannot be made."""
        given: dict[str, Any] = {**thaw(self.settings), **self.files}  # type: ignore[dict-item]
        try:
            made = self.strategy(context, **given)
        except TypeError as exc:
            raise OptimizationError(
                f"strategy {self.name} cannot be made with the settings "
                f"{sorted(given) or 'none'}: {exc}"
            ) from exc
        except Exception as exc:
            raise OptimizationError(
                f"strategy {self.name} could not be made for the study: {type(exc).__name__}: {exc}"
            ) from exc
        return made  # type: ignore[no-any-return]  # a strategy: it has the methods of one

    def changed(self) -> list[str]:
        """What it runs that is no longer what the study records: its code
        (:func:`~k4opt.usercode.changed_code`) and the files its settings name."""
        found = [f"its code ({name})" for name in changed_code(self.strategy)]
        for key, path in self.files.items():
            try:
                now = file_identity(path)
            except OSError:
                now = None
            if now != thaw(self.identities[key]):
                found.append(f"its {key} {path}")
        return found

    def dirty(self) -> list[str]:
        """What its source's identity leaves out: a checkout with uncommitted changes."""
        revision = source_of(self.strategy).get("revision")
        if isinstance(revision, Mapping) and revision.get("dirty"):
            return [
                f"strategy {self.name}: the checkout of its file has uncommitted changes, so files "
                "beside it that it does not import are not identified"
            ]
        return []


def _name(strategy: type) -> str:
    """The name a strategy class was loaded by: the attribute its reference names."""
    reference = source_of(strategy).get("reference")
    return str(reference).rpartition(":")[2] if reference else strategy.__name__


def _identity(key: str, path: Path) -> dict[str, Json]:
    try:
        return file_identity(path)
    except OSError as exc:
        raise OptimizationError(f"strategy setting {key}: {path} cannot be read: {exc}") from exc


def differences(recorded: Mapping[str, Any], custom: Custom) -> list[str]:
    """Why the strategy of one's own a study records is not ``custom``: another class, another
    source or modules it imports, other settings or files. Each said for a person; none if it is
    the same."""
    now = custom.record()
    found = []
    if recorded.get("class") != now["class"]:
        found.append(f"it was made with the strategy {recorded.get('class')}, not {now['class']}")
        return found
    then_source = recorded.get("source")
    then_source = then_source if isinstance(then_source, Mapping) else {}
    if then_source.get("sha256") != now["source"]["sha256"]:
        found.append(f"the file of the strategy {now['class']} changed")
    elif thaw(then_source.get("helpers")) != now["source"]["helpers"]:
        then_helpers: dict[str, Any] = dict(then_source.get("helpers") or {})
        helpers: dict[str, Any] = dict(now["source"]["helpers"] or {})
        changed = [
            module
            for module in dict.fromkeys([*then_helpers, *helpers])
            if then_helpers.get(module) != helpers.get(module)
        ]
        found.append(f"the modules the strategy's file imports changed: {', '.join(changed)}")
    if thaw(recorded.get("settings")) != now["settings"]:
        found.append(
            f"its strategy's settings are {thaw(recorded.get('settings'))!r}, not "
            f"{now['settings']!r} (give the same --strategy-settings)"
        )
    then_files = thaw(recorded.get("files"))
    then_files = then_files if isinstance(then_files, Mapping) else {}
    for key in dict.fromkeys([*then_files, *now["files"]]):
        if then_files.get(key) != now["files"].get(key):
            path = custom.files.get(key)
            found.append(
                f"its strategy's {key} is not the file the study read"
                + (f" ({path} holds other bytes)" if path is not None else " (none given)")
            )
    return found


def revision_warnings(recorded: Mapping[str, Any], custom: Custom) -> list[str]:
    """What a resume with ``custom`` leaves out: its checkout at another commit, or clean or dirty
    otherwise, than when the study was made."""
    source = recorded.get("source")
    then = source.get("revision") if isinstance(source, Mapping) else None
    now = source_of(custom.strategy).get("revision")
    if thaw(then) == thaw(now):
        return []
    return [
        f"strategy {custom.name}: the checkout of its file is at {now} now, at {thaw(then)} when "
        "the study was made; its file and the modules it imports are unchanged, other files "
        "beside them may not be"
    ]
