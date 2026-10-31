"""Resuming a study: what it continues with, what makes it refuse, and the trials an interrupted
session left behind."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("optuna")

import optuna
from optuna.trial import TrialState

from k4opt.evaluation import load_evaluator
from k4opt.optimization import OptimizationError, optimize
from k4opt.optimization._optuna import run as adapter
from k4opt.optimization._optuna.space import search_space
from k4opt.problem import load_problem
from tests.support import simulation
from tests.support.common import git, git_checkout, requires_git, tree_digest
from tests.support.evaluators import TRADE_OFF, cell_count, toy_significance, trade_off, yields
from tests.support.geometry import MINI_GEO
from tests.unit.optimization.conftest import (
    RADIUS,
    TRADE,
    Log,
    evaluator_from,
    left_running,
    optuna_study,
    summary,
)


def run(problem: Path, output: Path, trials: int, **kwargs: Any) -> Any:
    evaluators = kwargs.pop("evaluators", None) or [trade_off()]
    return optimize(problem, evaluators, output, trials=trials, **kwargs)


def test_a_resumed_study_continues_with_new_proposals_and_keeps_its_trials(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    first = run(problem, output, 3, seed=11)
    before = {p.name: tree_digest(p) for p in output.glob("trial-*")}
    resumed = run(problem, output, 2, resume=True)
    assert [t.number for t in resumed.trials] == [0, 1, 2, 3, 4]
    assert {p.name: tree_digest(p) for p in output.glob("trial-*") if p.name in before} == before
    # the resumed session does not replay the first one's proposals
    assert resumed.trials[3].candidate != first.trials[0].candidate
    seeds = [t.user_attrs["k4opt.sampler_seed"] for t in optuna_study(output).trials]
    assert seeds[0] == seeds[1] == seeds[2] != seeds[3] == seeds[4]
    assert resumed.id == first.id and resumed.name == first.name


def _edit(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data))


PROBLEM_CHANGES: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "bounds": (lambda d: d["parameters"][0].update(bounds=[100.0, 500.0]), "parameters changed"),
    "direction": (lambda d: d["objectives"][0].update(direction="maximize"), "objectives changed"),
    "limit": (
        lambda d: d.update(
            constraints=[{"stage": "analysis", "metric": "a", "max": 9, "unit": "mm"}]
        ),
        "constraints changed",
    ),
    "settings": (
        lambda d: d["evaluators"][0].update(settings={"invalid_below": 1}),
        "evaluators changed",
    ),
}


@pytest.mark.parametrize("change", sorted(PROBLEM_CHANGES))
def test_a_study_does_not_resume_with_another_problem(
    tmp_path: Path, write_problem: Callable[..., Path], change: str
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 2, seed=1)
    before = tree_digest(output)
    edit, message = PROBLEM_CHANGES[change]
    _edit(problem, edit)
    with pytest.raises(OptimizationError, match=f"cannot resume trade-.*{message}"):
        run(problem, output, 1, resume=True)
    assert tree_digest(output) == before  # nothing changed, no lock left


def two_bindings(write_problem: Callable[..., Path]) -> Path:
    """``test.trade_off`` bound twice: as itself, and as ``strict``, which is invalid below 250
    mm, with an objective on the ``a`` of each."""
    strict = TRADE | {"as": "strict", "settings": {"invalid_below": 250.0}}
    return write_problem(
        evaluators=[TRADE, strict],
        objectives=[
            ("test.trade_off", "analysis", "a", "minimize"),
            ("strict", "analysis", "a", "maximize"),
        ],
    )


def _renamed(data: dict[str, Any]) -> None:
    """The second binding, and the objective on it, under another name."""
    data["evaluators"][1]["as"] = "lenient"
    data["objectives"][1]["evaluator"] = "lenient"


BINDING_CHANGES: dict[str, Callable[[dict[str, Any]], None]] = {
    "settings": lambda d: d["evaluators"][1]["settings"].update(invalid_below=300.0),
    "name": _renamed,
}


def test_a_study_of_two_bindings_of_one_implementation_resumes(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = two_bindings(write_problem), tmp_path / "s"
    run(problem, output, 2, seed=3)
    resumed = run(problem, output, 1, resume=True)
    assert len(resumed.trials) == 3
    assert resumed.objectives == ("test.trade_off/analysis/a", "strict/analysis/a")
    study = json.loads((output / "study.json").read_text())
    assert list(study["implementations"]) == ["test.trade_off"]  # one code, bound twice
    record = summary(output, 0)
    assert list(record["evaluations"]) == ["test.trade_off", "strict"]
    assert record["evaluations"]["strict"]["settings"]["used"]["invalid_below"] == 250.0


@pytest.mark.parametrize("change", sorted(BINDING_CHANGES))
def test_a_study_does_not_resume_when_one_binding_changes(
    tmp_path: Path, write_problem: Callable[..., Path], change: str
) -> None:
    problem, output = two_bindings(write_problem), tmp_path / "s"
    run(problem, output, 1, seed=3)
    before = tree_digest(output)
    _edit(problem, BINDING_CHANGES[change])
    with pytest.raises(OptimizationError, match=r"cannot resume trade-.*evaluators.* changed"):
        run(problem, output, 1, resume=True)
    assert tree_digest(output) == before


def test_a_problem_that_only_has_another_name_resumes(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 1, seed=1)
    _edit(problem, lambda d: d.update(name="renamed"))
    assert len(run(problem, output, 1, resume=True).trials) == 2


def test_a_study_resumes_only_with_the_files_its_geometry_reads_outside_its_root(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # the mini geometry includes ${DD4hepINSTALL}/DDDetectors/compact/detector_types.xml
    types = tmp_path / "dd4hep" / "DDDetectors" / "compact" / "detector_types.xml"
    types.parent.mkdir(parents=True)
    types.write_text("<define/>")
    monkeypatch.setenv("DD4hepINSTALL", str(tmp_path / "dd4hep"))
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 1, seed=1)
    moved = shutil.copytree(tmp_path / "dd4hep", tmp_path / "elsewhere")
    monkeypatch.setenv("DD4hepINSTALL", str(moved))  # the same file elsewhere
    assert len(run(problem, output, 1, resume=True).trials) == 2
    (moved / types.relative_to(tmp_path / "dd4hep")).write_text("<define>edited</define>")
    before = tree_digest(output)
    message = (
        r"cannot resume trade-.*the geometry's reference \$\{DD4hepINSTALL\}/DDDetectors/compact/"
        r"detector_types\.xml does not read the files the study read"
    )
    with pytest.raises(OptimizationError, match=message):
        run(problem, output, 1, resume=True)
    assert tree_digest(output) == before


def test_a_file_only_candidates_name_reads_alike_in_every_trial_of_a_study(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "cad" / "b.stl"
    model.parent.mkdir()
    model.write_text("solid b")
    monkeypatch.setenv("CAD", str(model.parent))
    problem, output = write_problem(), tmp_path / "s"
    support = {
        "geometry": {"target": "InnerTrackerSupportCAD.shape.ref"},  # cad/support.stl
        "type": "categorical",
        "choices": ["${CAD}/b.stl"],
    }
    _edit(problem, lambda d: d["parameters"].append(support))
    run(problem, output, 2, seed=1)
    read = {"path": str(model), "sha256": hashlib.sha256(b"solid b").hexdigest()}
    for number in (0, 1):
        geometry = summary(output, number)["geometry"]
        assert geometry["candidate_externals"] == {"${CAD}/b.stl": read}
        assert "${CAD}/b.stl" not in geometry["externals"]  # not in the unmodified geometry
    model.write_text("solid other")  # between two sessions, which the study identity cannot see
    message = r"trial 2 stops the study.*\$\{CAD\}/b\.stl reads other files than for trial 0"
    with pytest.raises(OptimizationError, match=message):
        run(problem, output, 1, resume=True)
    assert optuna_study(output).trials[2].state == TrialState.FAIL
    model.write_text("solid b")  # the file the study read is back
    assert len(run(problem, output, 1, resume=True).trials) == 4


def _copy_evaluator(directory: Path, comment: str = "") -> Any:
    source = Path(TRADE_OFF.partition(":")[0])
    directory.mkdir(parents=True)
    copy = directory / "trade_off.py"
    copy.write_text(source.read_text() + comment)
    return load_evaluator(f"{copy}:TradeOff")


def test_a_study_does_not_resume_with_other_evaluator_code(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 1, seed=1, evaluators=[_copy_evaluator(tmp_path / "a")])
    edited = _copy_evaluator(tmp_path / "b", "# a helper changed\n")
    with pytest.raises(
        OptimizationError, match=r"evaluator 'test\.trade_off': .*trade_off\.py changed"
    ):
        run(problem, output, 1, resume=True, evaluators=[edited])


UNITS = """
    import os
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"

        def __init__(self):  # the same code, configured to report in another unit
            unit = os.environ.get("K4OPT_TEST_UNIT", "mm")
            self.metrics = {("analysis", "a"): unit, ("analysis", "b"): "1/mm"}

        def evaluate(self, request):
            x, *_ = request.candidate.values()
            return {("analysis", "a"): x, ("analysis", "b"): 1 / x}
"""


def test_a_study_does_not_resume_with_metrics_in_other_units(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 1, seed=1, evaluators=[evaluator_from(tmp_path / "e", "units", UNITS)])
    monkeypatch.setenv("K4OPT_TEST_UNIT", "cm")
    other = evaluator_from(tmp_path / "e", "units", UNITS)  # the same bytes, another instance
    with pytest.raises(OptimizationError, match=r"declares other metrics or units: .*'cm'"):
        run(problem, output, 1, resume=True, evaluators=[other])


ARTIFACTS = """
    import os
    from k4opt.evaluation import EvaluationResult, Evaluator, Measurement

    class Probe(Evaluator):
        name, version = "test.trade_off", "1"
        metrics = {("analysis", "a"): "mm", ("analysis", "b"): "1/mm"}

        def __init__(self):  # the same code, configured to name its file otherwise
            self.artifacts = {os.environ.get("K4OPT_TEST_ARTIFACT", "plot"): "a and b"}

        def evaluate(self, request):
            x, *_ = request.candidate.values()
            (name,) = self.artifacts
            path = request.workdir / "plot.txt"
            path.write_text(f"{x} {1 / x}")
            a = Measurement("analysis", "a", x, "mm")
            b = Measurement("analysis", "b", 1 / x, "1/mm")
            return EvaluationResult.valid([a, b], artifacts={name: path})
"""


def test_a_study_does_not_resume_with_other_artifacts_declared(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    source = ARTIFACTS
    run(problem, output, 1, seed=1, evaluators=[evaluator_from(tmp_path / "e", "named", source)])
    assert summary(output, 0)["implementations"]["test.trade_off"]["artifacts"] == ["plot"]
    monkeypatch.setenv("K4OPT_TEST_ARTIFACT", "figure")
    other = evaluator_from(tmp_path / "e", "named", source)  # the same bytes, another instance
    with pytest.raises(OptimizationError, match=r"declares other artifacts: \['figure'\]"):
        run(problem, output, 1, resume=True, evaluators=[other])


def test_location_is_not_identity(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    geometry = tmp_path / "k4geo-a"
    shutil.copytree(MINI_GEO, geometry)
    monkeypatch.setenv("K4GEO", str(geometry))
    run(problem, output, 1, seed=1, evaluators=[_copy_evaluator(tmp_path / "a")])
    # the evaluator file, the problem, the k4geo tree and the study itself, all moved
    moved_geometry = tmp_path / "k4geo-b"
    geometry.rename(moved_geometry)
    monkeypatch.setenv("K4GEO", str(moved_geometry))
    (tmp_path / "elsewhere").mkdir()
    moved_problem = problem.rename(tmp_path / "elsewhere" / problem.name)
    moved_output = output.rename(tmp_path / "moved-study")
    evaluator = _copy_evaluator(tmp_path / "b")  # the same bytes
    log = Log()
    report = run(
        moved_problem, moved_output, 1, resume=True, evaluators=[evaluator], **log.kwargs()
    )
    assert [t.status for t in report.trials] == ["valid", "valid"]
    assert log.warnings == []
    # where the evaluator was loaded from is recorded, as what loads it again, but is no identity
    study = json.loads((moved_output / "study.json").read_text())
    reference = study["implementations"]["test.trade_off"]["source"]["reference"]
    assert reference == f"{tmp_path / 'a' / 'trade_off.py'}:TradeOff"
    assert summary(moved_output, 1)["implementations"]["test.trade_off"]["source"]["reference"] == (
        f"{tmp_path / 'b' / 'trade_off.py'}:TradeOff"
    )


def test_another_baseline_geometry_or_stack_refuses(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    geometry = tmp_path / "k4geo"
    shutil.copytree(MINI_GEO, geometry)
    monkeypatch.setenv("K4GEO", str(geometry))
    run(problem, output, 1, seed=1)
    monkeypatch.setenv("KEY4HEP_STACK", "/stack/2026-11-01")
    with pytest.raises(OptimizationError, match="KEY4HEP_STACK is '/stack/2026-11-01'"):
        run(problem, output, 1, resume=True)
    monkeypatch.setenv("KEY4HEP_STACK", "/stack/2026-10-20")
    tracker = geometry / "compact" / "det_v1" / "tracker.xml"
    tracker.write_text(tracker.read_text().replace("</lccdd>", "<!-- edited --></lccdd>"))
    with pytest.raises(OptimizationError, match="the geometry the candidates start from changed"):
        run(problem, output, 1, resume=True)


def _edit_study(output: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = output / "study.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda d: d["optimizer"].update(version="4.9.0"),
            r"made with Optuna 4\.9\.0, this is Optuna 5\.0\.0: install optuna==4\.9\.0",
        ),
        (lambda d: d["k4opt"].update(sha256="0" * 64), "k4opt's code changed"),
        (lambda d: d.update(identity="0" * 64), "does not hold together"),
        (
            lambda d: d["optimizer"]["sampler"].update(n_startup_trials=5),
            "its recorded optimizer sampler is",
        ),
        (lambda d: d["optimizer"]["sampler"].update(seed="7"), "its sampler seed '7' is no seed"),
        (
            lambda d: d["optimizer"].update(constraint_values="scaled"),
            "its recorded optimizer constraint_values is 'scaled'",
        ),
        (
            lambda d: d["optimizer"]["parameters"][RADIUS]["attributes"].update(low=50.0),
            "its recorded optimizer parameters is",
        ),
    ],
)
def test_other_software_than_the_study_was_made_with_refuses(
    tmp_path: Path, write_problem: Callable[..., Path], change: Any, message: str
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 1, seed=1)
    _edit_study(output, change)
    with pytest.raises(OptimizationError, match=message):
        run(problem, output, 1, resume=True)


def test_resume_needs_a_study_and_a_study_needs_resume(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    with pytest.raises(
        OptimizationError, match=r"holds no study\.json, so there is no study to resume"
    ):
        run(problem, output, 1, resume=True)
    run(problem, output, 1, seed=1)
    with pytest.raises(OptimizationError, match="is in use; resume the study in it with --resume"):
        run(problem, output, 1, seed=1)
    with pytest.raises(OptimizationError, match=r"give the same --storage"):
        run(problem, output, 1, resume=True, storage="sqlite:///elsewhere.db")


@requires_git
def test_another_revision_of_an_evaluators_checkout_only_warns(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    source = Path(TRADE_OFF.partition(":")[0]).read_text()
    checkout = tmp_path / "evaluators"
    git_checkout(checkout, {"trade_off.py": source, "helper.py": "x = 1\n"})
    problem, output = write_problem(), tmp_path / "s"
    run(
        problem, output, 1, seed=1, evaluators=[load_evaluator(f"{checkout}/trade_off.py:TradeOff")]
    )
    (checkout / "helper.py").write_text("x = 2\n")
    git(checkout, "commit", "--quiet", "--all", "--message", "helper")
    # a new process would load it anew; here, from a second checkout at the new commit
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "--quiet", str(checkout), str(clone))
    log = Log()
    evaluator = load_evaluator(f"{clone}/trade_off.py:TradeOff")
    report = run(problem, output, 1, resume=True, evaluators=[evaluator], **log.kwargs())
    assert len(report.trials) == 2
    assert any("helper modules beside it may not be" in w for w in log.warnings)


# --- recovery --------------------------------------------------------------------------------


@pytest.mark.parametrize("directory", [False, True])
def test_a_trial_left_running_without_a_record_is_closed_as_failed(
    tmp_path: Path, write_problem: Callable[..., Path], directory: bool
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 2, seed=1)
    number = left_running(output, problem, directory)
    left = tree_digest(output / f"trial-{number:06d}") if directory else None
    log = Log()
    report = run(problem, output, 1, resume=True, **log.kwargs())
    assert len(report.trials) == 4  # the recovered trial does not count among the new ones
    reported = [(t.number, t.status, t.recovered) for t in log.trials]
    assert reported == [(number, "interrupted", True), (3, "valid", False)]  # recovered first
    recovered = report.trials[number]
    assert (recovered.status, recovered.recovered) == ("interrupted", True)
    assert optuna_study(output).trials[number].state == TrialState.FAIL
    if directory:  # left as it is, and never reused
        assert tree_digest(output / f"trial-{number:06d}") == left
    assert report.trials[3].status == "valid"


class Crash(BaseException):
    """A process killed between publishing a record and telling the optimizer."""


def test_a_trial_left_running_with_its_record_is_told_that_record(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    constraints = [{"stage": "analysis", "metric": "a", "max": 350, "unit": "mm"}]
    problem, output = write_problem(constraints=constraints), tmp_path / "s"
    tell = optuna.Study.tell

    def crashing(self: optuna.Study, *args: Any, **kwargs: Any) -> Any:
        raise Crash

    monkeypatch.setattr(optuna.Study, "tell", crashing)
    with pytest.raises(Crash):
        run(problem, output, 1, seed=1)
    monkeypatch.setattr(optuna.Study, "tell", tell)
    (left,) = optuna_study(output).trials
    assert left.state == TrialState.RUNNING and left.constraints  # set just before the crash
    log = Log()
    report = run(problem, output, 1, resume=True, **log.kwargs())
    reported = [(t.number, t.status, t.recovered) for t in log.trials]
    assert reported == [(0, "valid", True), (1, "valid", False)]
    assert log.trials[0] == report.trials[0]  # reported as the study reports it
    recovered = optuna_study(output).trials[0]
    record = summary(output, 0)
    metrics = {m["metric"]: m["value"] for m in record["metrics"]}
    assert recovered.state == TrialState.COMPLETE
    assert recovered.values == [metrics["a"], metrics["b"]]  # what a live trial is told
    assert recovered.constraints == {"test.trade_off/analysis/a:max": metrics["a"] - 350}
    assert recovered.user_attrs["k4opt.record"]["id"] == record["id"]
    assert report.trials[0].recovered and report.trials[0].status == "valid"


def test_a_left_record_that_does_not_hold_up_refuses_the_resume(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    monkeypatch.setattr(adapter, "_report", lambda *args, **kwargs: (_ for _ in ()).throw(Crash))
    with pytest.raises(Crash):
        run(problem, output, 1, seed=1)
    monkeypatch.undo()
    monkeypatch.setenv("K4GEO", str(MINI_GEO))
    monkeypatch.setenv("KEY4HEP_STACK", "/stack/2026-10-20")
    path = output / "trial-000000" / "summary.json"
    data = json.loads(path.read_text())
    data["assessment"]["feasible"] = False  # what its measurements do not give
    path.write_text(json.dumps(data))
    with pytest.raises(OptimizationError, match="its assessment is not the one its measurements"):
        run(problem, output, 1, resume=True)
    assert optuna_study(output).trials[0].state == TrialState.RUNNING  # left for a person


@pytest.mark.parametrize(
    ("change", "state"),
    [("remove", "is gone"), ("edit", "is not the record the optimizer was told")],
)
def test_a_finished_trial_keeps_the_record_the_optimizer_was_told(
    tmp_path: Path, write_problem: Callable[..., Path], change: str, state: str
) -> None:
    settings = {"failed_below": 300}  # failed trials keep their records too
    problem = write_problem(
        evaluators=[{"name": "test.trade_off", "version": "1", "settings": settings}]
    )
    output = tmp_path / "s"
    run(problem, output, 3, seed=1)
    record = output / "trial-000001" / "summary.json"
    if change == "remove":
        record.unlink()
    else:
        record.write_text(record.read_text() + " ")  # the same content to JSON, other bytes
    before = tree_digest(output)
    with pytest.raises(OptimizationError, match=rf"trial 1: .*summary\.json {state}"):
        run(problem, output, 1, resume=True)
    assert tree_digest(output) == before


def test_a_study_whose_setup_was_cut_short_resumes(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    set_user_attr, written = optuna.Study.set_user_attr, []

    def killed_after_two(self: optuna.Study, key: str, value: Any) -> None:
        if len(written) == 2:  # the study is made, two of its attributes are written
            raise Crash
        written.append(key)
        set_user_attr(self, key, value)

    monkeypatch.setattr(optuna.Study, "set_user_attr", killed_after_two)
    with pytest.raises(Crash):
        run(problem, output, 1, seed=1)
    monkeypatch.setattr(optuna.Study, "set_user_attr", set_user_attr)
    assert optuna_study(output).trials == []
    report = run(problem, output, 1, resume=True)
    assert [t.status for t in report.trials] == ["valid"]
    assert set(optuna_study(output).user_attrs) >= {"k4opt.study", "k4opt.constraints"}


def test_a_trial_another_client_added_to_the_storage_refuses_the_resume(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 2, seed=1)
    study = optuna_study(output)  # values without any evaluation behind them
    trial = optuna.trial.create_trial(
        params={RADIUS: 300.0}, distributions=search_space(load_problem(problem)), values=[1, 2]
    )
    study.add_trial(trial)
    before = tree_digest(output)
    with pytest.raises(OptimizationError, match="trial 2 is COMPLETE without the record k4opt"):
        run(problem, output, 1, resume=True)
    assert tree_digest(output) == before


def test_a_failed_trial_another_client_added_refuses_the_resume(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 2, seed=1)
    study = optuna_study(output)  # no values, but one more trial: another seed for the next session
    trial = optuna.trial.create_trial(
        params={RADIUS: 300.0},
        distributions=search_space(load_problem(problem)),
        state=TrialState.FAIL,
    )
    study.add_trial(trial)
    with pytest.raises(OptimizationError, match=r"trial 2 is FAIL without .* any mark of k4opt"):
        run(problem, output, 1, resume=True)


def test_a_trial_directory_of_no_trial_refuses_the_resume(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    run(problem, output, 1, seed=1)
    (output / "trial-000099").mkdir()
    with pytest.raises(OptimizationError, match=r"trial-000099 in .* belong.* to no trial"):
        run(problem, output, 1, resume=True)
    assert len(optuna_study(output).trials) == 1


# --- files the problem names -----------------------------------------------------------------

FILE_SIZE = """
    from k4opt.evaluation import Evaluator

    class Probe(Evaluator):
        name, version = "test.file_size", "1"
        stage = "generation"
        metrics = {"generation.bytes": "B"}

        def evaluate(self, request):
            (x,) = request.candidate.values()
            return {"generation.bytes": request.inputs["generation"].stat().st_size + x / 1000}
"""


@pytest.fixture
def file_problem(tmp_path: Path, write_problem: Callable[..., Path]) -> Path:
    problem = write_problem(
        evaluators=[{"name": "test.file_size", "version": "1", "stage": "generation"}],
        objectives=[("generation", "generation.bytes", "minimize")],
    )
    (tmp_path / "zh.hepmc").write_text("HepMC::Version 3\n")
    _edit(problem, lambda d: d.update(samples={"gamma": {"generation": {"file": "zh.hepmc"}}}))
    return problem


def test_a_file_the_problem_names_that_changes_refuses_the_resume(
    tmp_path: Path, file_problem: Path
) -> None:
    sizes = evaluator_from(tmp_path / "evaluators", "file_size", FILE_SIZE)
    run(file_problem, tmp_path / "s", 1, seed=1, evaluators=[sizes])
    (tmp_path / "zh.hepmc").write_text("HepMC::Version 3\nE 0\n")  # the same path
    with pytest.raises(
        OptimizationError, match=r"samples\.gamma\.generation\.file: .*zh\.hepmc is not the"
    ):
        run(file_problem, tmp_path / "s", 1, resume=True, evaluators=[sizes])


def test_a_file_the_problem_names_that_changes_during_a_study_stops_it(
    tmp_path: Path, file_problem: Path
) -> None:
    sizes = evaluator_from(tmp_path / "evaluators", "file_size", FILE_SIZE)

    def change(trial: Any) -> None:  # after the first trial, while the study runs
        (tmp_path / "zh.hepmc").write_text("HepMC::Version 3\nE 1\n")

    with pytest.raises(
        OptimizationError, match=r"trial 1 stops the study.*changed after the session"
    ):
        run(file_problem, tmp_path / "s", 3, seed=1, evaluators=[sizes], progress=change)
    first, stopped = optuna_study(tmp_path / "s").trials
    assert first.state == TrialState.COMPLETE and stopped.state == TrialState.FAIL
    assert not (tmp_path / "s" / "trial-000001").exists()


SAMPLE_CHANGES: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "seed": (lambda d: d["samples"]["pion"]["simulation"].update(seed=3), "samples changed"),
    "events": (lambda d: d["samples"]["pion"]["simulation"].update(events=3), "samples changed"),
    "particle": (lambda d: d["samples"]["pion"]["generation"].update(particle="pi+"), "samples"),
    "measured": (lambda d: d["evaluators"][0].update(samples=["pion", "gamma"]), "evaluators"),
    "renamed": (
        lambda d: d.update(
            samples={"pions" if k == "pion" else k: v for k, v in d["samples"].items()},
            evaluators=[d["evaluators"][0] | {"samples": ["gamma", "pions"]}],
            objectives=[d["objectives"][0], d["objectives"][1] | {"sample": "pions"}],
        ),
        "samples, evaluators, objectives changed",
    ),
}


def test_a_study_of_two_samples_resumes_and_refuses_another_sample(
    tmp_path: Path, sampled_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    output = tmp_path / "s"
    run(sampled_problem, output, 1, seed=1, evaluators=[cell_count()])
    resumed = run(sampled_problem, output, 1, resume=True, evaluators=[cell_count()])
    assert [t.status for t in resumed.trials] == ["valid", "valid"]
    assert list(summary(output, 1)["samples"]) == ["gamma", "pion"]
    before = tree_digest(output)
    for change, (edit, message) in SAMPLE_CHANGES.items():
        original = sampled_problem.read_text()
        _edit(sampled_problem, edit)
        with pytest.raises(OptimizationError, match=f"cannot resume trade-.*{message}"):
            run(sampled_problem, output, 1, resume=True, evaluators=[cell_count()])
        sampled_problem.write_text(original)
        assert tree_digest(output) == before, change


def test_a_study_whose_sample_reads_a_moved_file_resumes(
    tmp_path: Path, sampled_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a file is compared by content: where it is does not identify the study
    simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    (tmp_path / "zh.hepmc").write_text("HepMC::Version 3\n")
    _edit(sampled_problem, lambda d: d["samples"]["pion"].update(generation={"file": "zh.hepmc"}))
    output = tmp_path / "s"
    run(sampled_problem, output, 1, seed=1, evaluators=[cell_count()])
    moved = tmp_path / "moved"
    moved.mkdir()
    for name in ("problem.yaml", "zh.hepmc"):  # the problem and the file it names, beside it
        (tmp_path / name).rename(moved / name)
    resumed = run(moved / "problem.yaml", output, 1, resume=True, evaluators=[cell_count()])
    assert [t.status for t in resumed.trials] == ["valid", "valid"]
    read = summary(output, 1)["samples"]["pion"]["stages"]["generation"]
    assert read["file"] == str(moved / "zh.hepmc")


COMBINED_CHANGES: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "property": (
        lambda d: d["samples"]["bkg"]["properties"]["cross_section"].update(value=31),
        "samples changed",
    ),
    "role": (
        lambda d: d["evaluators"][1]["roles"].update(
            backgrounds={"samples": ["sig", "bkg"], "evaluator": "test.yields"}
        ),
        "evaluators changed",
    ),
}

ROLES = """
    import os
    from k4opt.evaluation import Evaluator, Role

    class Probe(Evaluator):
        name, version = "test.toy_significance", "1"
        settings = {"luminosity_invab": float}
        metrics = {("analysis", "expected_significance"): "1"}

        def __init__(self):  # the same code, configured to read another metric
            metric = os.environ.get("K4OPT_TEST_METRIC", "selected.sum_weights")
            role = Role(metrics=[("analysis", metric)])
            self.roles = {"signal": role, "backgrounds": Role(many=True)}

        def evaluate(self, request):
            return {("analysis", "expected_significance"): 1.0}
"""


def test_a_study_of_a_combined_evaluator_resumes_and_refuses_other_roles_or_properties(
    tmp_path: Path, combined_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    output = tmp_path / "s"
    evaluators = [toy_significance(), yields()]
    run(combined_problem, output, 1, seed=1, evaluators=evaluators)
    resumed = run(combined_problem, output, 1, resume=True, evaluators=evaluators)
    assert [t.status for t in resumed.trials] == ["valid", "valid"]
    before = tree_digest(output)
    for change, (edit, message) in COMBINED_CHANGES.items():
        original = combined_problem.read_text()
        _edit(combined_problem, edit)
        with pytest.raises(OptimizationError, match=f"cannot resume trade-.*{message}"):
            run(combined_problem, output, 1, resume=True, evaluators=evaluators)
        combined_problem.write_text(original)
        assert tree_digest(output) == before, change


def test_a_study_does_not_resume_with_other_roles_declared(
    tmp_path: Path, combined_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation.install(monkeypatch, simulation.events_with([1, 2], [3]))
    output = tmp_path / "s"
    probe = evaluator_from(tmp_path / "e", "roles", ROLES)
    run(combined_problem, output, 1, seed=1, evaluators=[probe, yields()])
    declared = summary(output, 0)["implementations"]["test.toy_significance"]["roles"]
    assert declared["signal"]["metrics"] == [["analysis", "selected.sum_weights"]]
    monkeypatch.setenv("K4OPT_TEST_METRIC", "generated.sum_weights")
    other = evaluator_from(tmp_path / "e", "roles", ROLES)  # the same bytes, other roles
    with pytest.raises(OptimizationError, match=r"'test\.toy_significance' declares other roles"):
        run(combined_problem, output, 1, resume=True, evaluators=[other, yields()])
