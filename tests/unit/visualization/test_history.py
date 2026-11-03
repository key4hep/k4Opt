"""``history_figure``: an objective in the order the trials were proposed, each trial as the
study judged it, the best feasible value so far for a study of one objective."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

from k4opt.visualization import VisualizationError, history_figure
from tests.unit.visualization.test_trials import notes, numbers, study_of, traces


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_one_objective_has_its_best_feasible_value_so_far(direction: str) -> None:
    sign = 1 if direction == "minimize" else -1
    values = [5, 7, 3, 1, 4]  # trial 3 is infeasible: the best so far stays 3
    outcomes = ["feasible", "feasible", "feasible", "infeasible", "feasible"]
    trials = [(o, [sign * v]) for o, v in zip(outcomes, values, strict=True)]
    study = study_of(
        (direction,), [*trials, ("failed", None)], constraint=sign * 2 if sign == 1 else None
    )
    figure = history_figure(study)
    shown = traces(figure)
    best = shown["best feasible so far"]
    assert figure.data[0] is best and best.line.shape == "hv"
    assert list(best.x) == [0, 1, 2, 3, 4] and list(best.y) == [sign * v for v in (5, 5, 3, 3, 3)]
    assert numbers(shown["best"]) == list(study.pareto) == [2]
    assert figure.layout.xaxis.title.text == "trial"
    assert "Not shown, without a value: 1 failed" in notes(figure)


def test_several_objectives_set_the_front_apart_without_a_best_line() -> None:
    study = study_of(
        ("minimize", "minimize"), [("feasible", [1, 9]), ("feasible", [9, 1]), ("feasible", [9, 9])]
    )
    figure = history_figure(study, "objective:test.trade_off/analysis/m1")
    shown = traces(figure)
    assert "best feasible so far" not in shown
    assert numbers(shown["on the Pareto front"]) == [0, 1] and list(shown["feasible"].y) == [9]
    assert figure.layout.yaxis.title.text.startswith("m1 [mm], minimize")


def test_a_parameter_has_no_history() -> None:
    study = study_of(("minimize",), [("feasible", [1.0])])
    with pytest.raises(VisualizationError, match="is a parameter"):
        history_figure(study, "parameter:radius")
