"""Internal: the strategies a study can be run with, by name, and what ``study.json`` records of
each under ``strategy``.

``optuna`` (the default) is Optuna's TPE sampler (:mod:`._optuna`), imported only when a study
runs with it; ``grid`` (:mod:`._grid`) needs nothing besides k4opt. The recorded strategy is the
study's: a resume runs with it, and refuses one this k4opt would not run alike
(:func:`differences`); each study of the level below gets its own from it (:func:`child`).
Reading a study back needs no strategy.
"""

from __future__ import annotations

import contextlib
import secrets
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from k4opt.optimization import _grid
from k4opt.optimization._proposal import Past, Strategy
from k4opt.optimization._scope import StudyScope
from k4opt.optimization.errors import OptimizationError

STRATEGIES = ("optuna", "grid")
"""The names of the strategies a study can be run with, the default first."""


@contextlib.contextmanager
def needs_optuna(what: str) -> Iterator[None]:
    """Optuna missing, while its strategy is imported, becomes an error that says how to install
    it; any other missing module is not dressed up."""
    try:
        yield
    except ModuleNotFoundError as exc:
        if exc.name != "optuna" and not (exc.name or "").startswith("optuna."):
            raise
        raise OptimizationError(
            f"{what} needs Optuna, which is not installed: pip install 'optuna>=5.0,<6' "
            "(or k4opt[optimize]); --strategy grid needs no Optuna"
        ) from exc


def new_strategy(
    name: str,
    scope: StudyScope,
    *,
    seed: int | None = None,
    grid: Mapping[str, Sequence[object]] | None = None,
) -> dict[str, Any]:
    """What ``study.json`` records of the strategy ``name`` of a new study of ``scope``: Optuna's
    sampler with ``seed`` (drawn without it), or the grid of the values ``grid`` gives."""
    if name not in STRATEGIES:
        raise OptimizationError(f"strategy {name!r} is none of {', '.join(STRATEGIES)}")
    if name == _grid.NAME:
        if seed is not None:
            raise OptimizationError("the grid draws nothing at random; leave out --seed")
        return _grid.config(scope, _grid.grid_values(scope, grid))
    if grid:
        raise OptimizationError("--grid gives the values of the grid strategy: add --strategy grid")
    with needs_optuna("the strategy optuna"):
        from k4opt.optimization import _optuna
    return _optuna.config(scope, secrets.randbelow(2**32) if seed is None else seed)


def _name(recorded: object) -> str:
    name = recorded.get("name") if isinstance(recorded, Mapping) else None
    if name not in STRATEGIES:
        raise OptimizationError(f"study.json records no strategy k4opt runs: {recorded!r}")
    assert isinstance(name, str)
    return name


def differences(recorded: object, scope: StudyScope) -> list[str]:
    """Why the recorded strategy cannot go on with the study of ``scope``: it is not the one this
    k4opt would run it with. Each said for a person; none if it can."""
    try:
        name = _name(recorded)
    except OptimizationError as exc:
        return [str(exc)]
    assert isinstance(recorded, Mapping)
    if name == _grid.NAME:
        return _grid.differences(recorded, scope)
    with needs_optuna("resuming a study of the strategy optuna"):
        from k4opt.optimization import _optuna
    return _optuna.differences(recorded, scope)


def child(recorded: Mapping[str, Any], scope: StudyScope, number: int) -> dict[str, Any]:
    """The strategy of the study of ``scope`` of the level below, started for trial ``number`` of
    the study whose strategy is ``recorded``."""
    if _name(recorded) == _grid.NAME:
        return _grid.child(recorded, scope)
    from k4opt.optimization import _optuna  # imported to run the study the child belongs to

    return _optuna.child(recorded, scope, number)


def open_strategy(
    recorded: Mapping[str, Any], scope: StudyScope, history: Sequence[Past]
) -> Strategy:
    """The recorded strategy of the study of ``scope``, made from its trials so far."""
    if _name(recorded) == _grid.NAME:
        return _grid.Grid(scope, recorded, history)
    with needs_optuna("the strategy optuna"):
        from k4opt.optimization import _optuna
    return _optuna.TPE(scope, recorded, history)


def describe(recorded: object) -> str:
    """The recorded strategy, for a person: ``optuna 5.0.0, TPESampler, seed 42``, ``grid of 12
    points``."""
    if not isinstance(recorded, Mapping):
        return "not recorded"
    if recorded.get("name") == _grid.NAME:
        return f"grid of {recorded.get('points')} points"
    sampler = recorded.get("sampler")
    sampler = sampler if isinstance(sampler, Mapping) else {}
    return (
        f"{recorded.get('name')} {recorded.get('version')}, {sampler.get('name')}, seed "
        f"{sampler.get('seed')}"
    )
