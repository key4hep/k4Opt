"""What the optimizer is asked to search and what it is told: distributions, objective values and
constraint values, and the declarations it refuses before a study starts."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("optuna")

from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from k4opt.evaluation import Measurement
from k4opt.optimization import OptimizationError, optimize
from k4opt.optimization._optuna import (
    constraint_values,
    limit_names,
    objective_values,
    search_space,
)
from k4opt.problem import Assessment, Limit, parse_problem
from tests.support.evaluators import candidate_check, trade_off
from tests.support.geometry import MINI_GEO
from tests.unit.optimization.conftest import Log, problem_document, summary

CHECK = [{"name": "test.candidate_check", "version": "1"}]
SEEN = [("analysis", "seen", "minimize")]


def parameter(kind: str, **declared: Any) -> dict[str, Any]:
    return {"geometry": {"constant": "InnerTracker_Barrel_radius_1"}, "type": kind, **declared}


@pytest.mark.parametrize(
    ("declared", "distribution"),
    [
        (parameter("continuous", bounds=[100, 600.5], unit="mm"), FloatDistribution(100.0, 600.5)),
        (parameter("integer", bounds=[-3, 7]), IntDistribution(-3, 7)),
        (parameter("categorical", choices=["a", "b"]), CategoricalDistribution(["a", "b"])),
        (parameter("categorical", choices=[True, False]), CategoricalDistribution([True, False])),
        (
            parameter("categorical", choices=[1, 2.5, 2**60]),
            CategoricalDistribution([1, 2.5, 2**60]),
        ),
    ],
)
def test_each_parameter_is_searched_as_declared(
    declared: dict[str, Any], distribution: Any
) -> None:
    spec = parse_problem(problem_document(parameter=declared))
    assert search_space(spec) == {"constants.InnerTracker_Barrel_radius_1": distribution}


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        (parameter("continuous", bounds=[0, 10**400]), "the bound 100000000000... (401 digits)"),
        (parameter("continuous", bounds=[0, 2**53 + 1]), "is no float"),
        (parameter("continuous", bounds=[-1e308, 1e308]), "too wide to sample"),
        (parameter("integer", bounds=[0, 2**53 + 1]), "beyond 2**53"),
    ],
)
def test_bounds_the_sampler_cannot_draw_exactly_are_refused(
    declared: dict[str, Any], message: str
) -> None:
    with pytest.raises(OptimizationError, match=re.escape(message)):
        search_space(parse_problem(problem_document(parameter=declared)))


@pytest.mark.parametrize(
    "declared",
    [
        parameter("continuous", bounds=[100, 600], unit="mm"),
        parameter("integer", bounds=[100, 105]),
        parameter("categorical", choices=[150, 250.5, 400]),
    ],
)
def test_proposed_values_are_in_their_domain_with_their_type_and_recorded_as_proposed(
    tmp_path: Path, write_problem: Callable[..., Path], declared: dict[str, Any]
) -> None:
    problem = write_problem(parameter=declared, evaluators=CHECK, objectives=SEEN)
    log = Log()
    report = optimize(
        problem, [candidate_check()], tmp_path / "s", trials=6, seed=3, **log.kwargs()
    )
    kinds = {"continuous": (float,), "integer": (int,), "categorical": (int, float)}[
        declared["type"]
    ]
    for trial in report.trials:
        (value,) = trial.candidate.values()
        assert type(value) in kinds and not isinstance(value, bool)
        if "bounds" in declared:
            low, high = declared["bounds"]
            assert low <= value <= high
        else:
            assert value in declared["choices"]
            assert type(value) is type(declared["choices"][declared["choices"].index(value)])
        # the record holds the candidate as the optimizer proposed it, type and all
        recorded = summary(tmp_path / "s", trial.number)["candidate"]
        assert recorded == dict(trial.candidate)
        assert [type(v) for v in recorded.values()] == [type(v) for v in trial.candidate.values()]


def assessment(*limits: Limit, objectives: tuple[Measurement, ...] = ()) -> Assessment:
    return Assessment(objectives, limits, all(limit.satisfied for limit in limits))


def limit(bound: str, margin: float | None, satisfied: bool) -> Limit:
    return Limit("simulation", "x.rms", bound, 1.0, "mm", 2.0, 0.1, margin, satisfied)  # type: ignore[arg-type]


def test_the_physical_margins_are_the_constraint_values() -> None:
    told = constraint_values(assessment(limit("min", -1.0, True), limit("max", 0.25, False)))
    assert told == {"simulation/x.rms:min": -1.0, "simulation/x.rms:max": 0.25}
    assert constraint_values(assessment()) == {}


def test_a_margin_that_fits_no_float_cannot_be_told() -> None:
    with pytest.raises(OptimizationError, match=r"simulation/x\.rms:max: the margin .* float"):
        constraint_values(assessment(limit("max", None, True)))


def test_objective_values_are_floats_in_the_problems_order() -> None:
    measured = (Measurement("analysis", "b", 3), Measurement("analysis", "a", 0.5))
    assert objective_values(assessment(objectives=measured)) == [3.0, 0.5]
    for value in (10**400, 2**53 + 1):  # too large for a float, or told as another value
        with pytest.raises(OptimizationError, match=r"no float holds .* exactly"):
            objective_values(assessment(objectives=(Measurement("analysis", "a", value),)))
    assert objective_values(assessment(objectives=(Measurement("analysis", "a", 2**53),))) == [
        2.0**53
    ]


def test_a_storage_url_is_recorded_and_shown_without_its_secrets(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from k4opt.optimization import _optuna

    url = "postgresql://k4opt:s3cr%2Bt@db.cern.ch:6600/studies?sslmode=require&password=hush#x"
    assert _optuna._redacted(url) == (
        "postgresql://k4opt:***@db.cern.ch:6600/studies?sslmode=***&password=***"
    )
    for plain in ("sqlite:///studies.db", "sqlite:////abs/studies.db", "postgresql://h/db"):
        assert _optuna._redacted(plain) == plain
    assert _optuna._redacted("sqlite:///s.db?timeout=20") == "sqlite:///s.db?timeout=***"

    def refusing(given: str) -> None:  # what a database driver may say
        raise ValueError(f"could not connect to {given} (password 'hush', secret s3cr+t)")

    monkeypatch.setattr(_optuna, "RDBStorage", refusing)
    with pytest.raises(OptimizationError) as info:
        optimize(write_problem(), [trade_off()], tmp_path / "s", trials=1, storage=url)
    message = str(info.value)
    assert "hush" not in message and "s3cr" not in message and "sslmode=require" not in message
    assert "postgresql://k4opt:***@db.cern.ch:6600/studies" in message
    assert not (tmp_path / "s").exists()

    # an error of the database later on, once the storage was made
    import optuna

    def failing(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"lost the connection {url} (password=hush)")

    monkeypatch.setattr(_optuna, "RDBStorage", lambda given: optuna.storages.InMemoryStorage())
    monkeypatch.setattr(optuna, "create_study", failing)
    with pytest.raises(OptimizationError) as info:
        optimize(write_problem(), [trade_off()], tmp_path / "t", trials=1, storage=url)
    message = str(info.value)
    assert "hush" not in message and "s3cr" not in message and "lost the connection" in message
    assert info.value.__cause__ is None and info.value.__suppress_context__  # nor in a traceback

    # a secret only in what an error was raised from, which a traceback shows too
    monkeypatch.undo()
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    made: list[str] = []

    def database(given: str) -> Any:
        made.append(given)
        return optuna.storages.InMemoryStorage()

    def asking(self: Any, *args: Any, **kwargs: Any) -> None:
        try:
            raise RuntimeError(f"driver: {url}")
        except RuntimeError as cause:
            raise optuna.exceptions.StorageInternalError("lost the database") from cause

    monkeypatch.setattr(_optuna, "RDBStorage", database)
    monkeypatch.setattr(optuna.Study, "ask", asking)
    with pytest.raises(
        OptimizationError, match="the study cannot go on: StorageInternalError"
    ) as info:
        optimize(write_problem(), [trade_off()], tmp_path / "u", trials=1, storage=url)
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert len(made) == 1  # the database is set up once, before the directory is made


def test_each_bound_of_a_constraint_is_a_limit_of_its_own() -> None:
    constraints = [
        {"stage": "analysis", "metric": "a", "min": 150, "max": 500, "unit": "mm"},
        {"stage": "analysis", "metric": "b", "max": 0.01},
    ]
    spec = parse_problem(problem_document(constraints=constraints))
    assert limit_names(spec) == ["analysis/a:min", "analysis/a:max", "analysis/b:max"]
