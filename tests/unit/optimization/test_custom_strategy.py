"""A strategy of one's own: a class of the user's code, loaded as ``FILE.py:Class`` and made for a
study with its context and settings, proposes candidates and learns from what they measured; made
again from the study's ledger after an interruption, it is told every call it got before, in the
same order, and goes on where it was. Everything of it a study records is compared on a resume,
nothing that looks like a credential is recorded, and nothing of it is needed to read a study back.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from k4opt.cli import main
from k4opt.optimization import OptimizationError, load_strategy, load_study, optimize
from k4opt.optimization._custom import Custom
from tests.support.evaluators import TRADE_OFF, candidate_check, cell_count, trade_off
from tests.support.stack import alive, eventually
from tests.unit.optimization.conftest import GUN, PROBE, RADIUS, endings, evaluator_from

REPOSITORY = Path(__file__).resolve().parents[3]
COMPASS = REPOSITORY / "examples" / "strategies" / "compass.py"
SETTINGS = REPOSITORY / "examples" / "strategies" / "compass.yaml"
INVALID_BELOW = {"name": "test.trade_off", "version": "1", "settings": {"invalid_below": 250.0}}

RECORDER = """
    import json
    import os

    from k4opt.optimization import EXHAUSTED, WAITING, Proposal

    CALLS = []  # every call of every instance made in this process, as the log holds it


    class Recorder:
        \"\"\"Proposes the radii it is given, at most ``batch`` under way, and writes each call it
        gets to the file PYTEST_STRATEGY_LOG names, if any.\"\"\"

        def __init__(self, study, *, radii, batch=2):
            self.radii, self.batch, self.next, self.open = list(radii), batch, 0, set()
            self.log("made", None, [p.name for p in study.parameters])

        def propose(self, trial):
            if self.next >= len(self.radii):
                return WAITING if self.open else EXHAUSTED
            if len(self.open) >= self.batch:
                return WAITING
            return Proposal({"radius": self.radii[self.next]}, {"index": self.next})

        def proposed(self, trial, proposal):
            self.next = proposal.note["index"] + 1
            self.open.add(trial)
            self.log("proposed", trial, dict(proposal.values))

        def tell(self, trial, feedback):
            self.open.discard(trial)
            assessment = feedback.result.assessment
            self.log(
                "told",
                trial,
                {
                    "status": feedback.status,
                    "values": feedback.values,
                    "feasible": feedback.feasible,
                    "assessed": assessment is not None,
                    "objectives": None if assessment is None or assessment.objectives is None
                    else [o.value for o in assessment.objectives],
                },
            )

        def log(self, kind, trial, data):
            CALLS.append([kind, trial, data])
            if "PYTEST_STRATEGY_LOG" in os.environ:
                with open(os.environ["PYTEST_STRATEGY_LOG"], "a", encoding="utf-8") as log:
                    log.write(json.dumps([kind, trial, data]) + "\\n")
"""

PROMPTED = """
    from pathlib import Path

    from k4opt.optimization import EXHAUSTED, Proposal

    import STEPS as steps


    class Prompted:
        \"\"\"Proposes a radius as long as its prompt is, times ``scale``; when ``edit`` says so, it
        rewrites its prompt or its helper module as a trial ends.\"\"\"

        files = ("prompt",)

        def __init__(self, study, *, prompt, scale=10.0, edit=None):
            self.prompt, self.scale, self.edit = Path(prompt), scale, edit
            self.length = len(self.prompt.read_text())
            self.count = 0

        def propose(self, trial):
            if self.count >= 3:
                return EXHAUSTED
            return Proposal({"radius": steps.STEP + self.length * self.scale + self.count})

        def proposed(self, trial, proposal):
            self.count += 1

        def tell(self, trial, feedback):
            if self.edit == "prompt":
                self.prompt.write_text(self.prompt.read_text() + " and more")
            elif self.edit == "helper":
                helper = Path(__file__).with_name("STEPS.py")
                helper.write_text("STEP = 1.0\\n")
"""


def strategy_file(directory: Path, source: str, name: str = "mine") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def prompted_file(directory: Path, helper: str) -> Path:
    """``PROMPTED`` in ``directory``, its helper module beside it named ``helper`` (a process
    imports one module of a name: each test gives its own)."""
    code = strategy_file(directory, PROMPTED.replace("STEPS", helper), name=f"prompted_{helper}")
    (directory / f"{helper}.py").write_text("STEP = 100.0\n")
    return code


def calls_of(strategy: type) -> list[Any]:
    """Every call the instances of ``strategy`` made in this process got, in order."""
    return sys.modules[strategy.__module__].CALLS  # type: ignore[no-any-return]


def cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


# --- the example: compass search, as documented -------------------------------------------------


def test_the_compass_search_searches_by_what_it_measured_and_ends_by_itself(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(evaluators=[INVALID_BELOW]), tmp_path / "s"
    strategy = f"{COMPASS}:CompassSearch"
    argv = ("--evaluator", TRADE_OFF, "--strategy", strategy, "--strategy-settings", str(SETTINGS))
    code, _, err = cli("optimize", str(problem), *argv, "--trials", "30", "--output", str(output))
    assert code == 0, err
    study = load_study(output)
    assert study.exhausted and len(study.trials) < 30  # it ended by itself
    radii = [t.candidate[RADIUS] for t in study.trials]
    assert len(set(radii)) == len(radii)  # never the same point twice
    # below 250 nothing is measured: never compared, never a center, so it stays above
    assert all(t.status == "invalid" for t in study.trials if t.candidate[RADIUS] < 250)
    best = min(float(t.candidate[RADIUS]) for t in study.trials if t.status == "valid")
    assert 250 <= best < 260
    # what study.json records: the class, its source, its settings; no location
    recorded = json.loads((output / "study.json").read_text())["strategy"]
    assert {k: recorded[k] for k in ("name", "class", "settings", "files")} == {
        "name": "custom",
        "class": "CompassSearch",
        "settings": {"step": 0.25, "min_step": 0.02},
        "files": {},
    }
    assert recorded["source"]["sha256"] and recorded["source"]["executed"] is True
    assert str(REPOSITORY) not in json.dumps(recorded)
    assert study.to_dict()["study"]["strategy"] == recorded  # the JSON form keeps it whole
    code, shown, _ = cli("study", "show", str(output))
    assert code == 0 and "CompassSearch, a strategy of its own (source" in shown
    # kept privately, for a rerun: its file
    (kept,) = (output / "software").glob("software-*.json")
    files = json.loads(kept.read_text())["strategy"]["files"]
    assert files["compass.py"]["kept"] == recorded["source"]["sha256"]


# --- what it is told, and when ----------------------------------------------------------------


def test_it_is_told_what_was_measured_and_never_a_value_no_evaluation_produced(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem = write_problem(
        evaluators=[
            {"name": "test.trade_off", "version": "1", "settings": {"invalid_below": 200.0}},
            {
                "name": "test.candidate_check",
                "version": "1",
                "settings": {"reject": 300},
                "required": True,  # its check rejects a candidate
            },
        ],
        objectives=[("test.trade_off", "analysis", "a", "minimize")],
        constraints=[
            {
                "stage": "analysis",
                "metric": "a",
                "max": 400,
                "unit": "mm",
                "evaluator": "test.trade_off",
            }
        ],
    )
    recorder = load_strategy(f"{strategy_file(tmp_path / 'code', RECORDER)}:Recorder")
    radii = [150.0, 300.0, 350.0, 500.0, 700.0]  # invalid, rejected, feasible, not, malformed
    optimize(
        problem,
        [trade_off(), candidate_check()],
        tmp_path / "s",
        trials=5,
        strategy=recorder,
        strategy_settings={"radii": radii, "batch": 1},
    )
    told = {trial: data for kind, trial, data in calls_of(recorder) if kind == "told"}
    assert {n: told[n]["status"] for n in told} == {
        0: "invalid",
        1: "rejected",
        2: "valid",
        3: "valid",
        4: "malformed",
    }
    for data in told.values():
        if data["status"] != "valid":
            assert (data["values"], data["feasible"]) == (None, None)
    # an invalid trial has a record, assessed without objectives; the others none
    assert (told[0]["assessed"], told[0]["objectives"]) == (True, None)
    assert told[1]["assessed"] is told[4]["assessed"] is False
    assert tuple(told[2]["values"]) == (350.0,) and told[2]["objectives"] == [350.0]
    assert (told[2]["feasible"], told[3]["feasible"]) == (True, False)
    # the malformed proposal was told kept, as given, before it was told malformed
    assert ["proposed", 4, {RADIUS: 700.0}] in calls_of(recorder)


def test_a_reopened_strategy_is_told_every_call_again_in_order_after_the_study_was_killed(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    marks, log = tmp_path / "marks", tmp_path / "calls.jsonl"
    marks.mkdir()
    problem, output = write_problem(), tmp_path / "s"
    probe = evaluator_from(tmp_path / "evaluators", "probe", PROBE)
    never = str(marks / "never")
    told = {
        "marks": str(marks),
        # trial 1 ends before trial 0, which waits for trial 2 to start; 2 and 3 never end
        "await": {
            "trial-000000": str(marks / "trial-000002.started"),
            "trial-000002": never,
            "trial-000003": never,
        },
    }
    monkeypatch.setenv("PYTEST_K4OPT_PROBE", json.dumps(told))
    monkeypatch.setenv("PYTEST_STRATEGY_LOG", str(log))
    code = strategy_file(tmp_path / "code", RECORDER)
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump({"radii": [150.0, 200.0, 250.0, 300.0, 350.0, 400.0]}))
    command = [
        *(sys.executable, "-m", "k4opt", "optimize", str(problem)),
        *("--evaluator", f"{tmp_path / 'evaluators' / 'probe.py'}:Probe", "--output", str(output)),
        *("--strategy", f"{code}:Recorder", "--strategy-settings", str(settings)),
        *("--trials", "6", "--workers", "2"),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert eventually(lambda: (marks / "trial-000003.started").exists(), timeout=120)
        process.send_signal(signal.SIGKILL)
        process.communicate(timeout=60)
    finally:
        process.kill()
    workers = [int(path.read_text()) for path in marks.glob("*.started")]
    assert all(eventually(lambda pid=pid: not alive(pid)) for pid in workers)
    (output / ".lock").unlink()  # nothing works on the study any more
    live = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(kind, trial) for kind, trial, _ in live] == [
        ("made", None),
        ("proposed", 0),
        ("proposed", 1),
        ("told", 1),  # it ended first
        ("proposed", 2),
        ("told", 0),
        ("proposed", 3),
    ]
    monkeypatch.setenv("PYTEST_K4OPT_PROBE", json.dumps({"marks": str(marks)}))
    report = optimize(
        problem,
        [probe],
        output,
        trials=2,
        resume=True,
        workers=2,
        strategy=load_strategy(f"{code}:Recorder"),
        strategy_settings=yaml.safe_load(settings.read_text()),
    )
    after = [json.loads(line) for line in log.read_text().splitlines()][len(live) :]
    reopened = after[: after.index(["proposed", 4, {RADIUS: 350.0}])]
    interrupted = {
        "status": "interrupted",
        "values": None,
        "feasible": None,
        "assessed": False,
        "objectives": None,
    }
    # made again from the ledger: every call it got live, in the same order, then the trials
    # the killed session left running, ended as the resume found them
    assert reopened == [
        *live,
        ["told", 2, interrupted],
        ["told", 3, interrupted],
    ]
    assert [(t.status, t.recovered) for t in report.trials] == [
        ("valid", False),
        ("valid", False),
        ("interrupted", True),
        ("interrupted", True),
        ("valid", False),
        ("valid", False),
    ]
    assert endings(output) == ["recorded"] * 2 + ["interrupted"] * 2 + ["recorded"] * 2


# --- settings, files and credentials ----------------------------------------------------------


def test_a_file_a_setting_names_is_identified_by_its_content_and_watched(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    code = prompted_file(tmp_path / "code", "steps_identified")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "agent.md").write_text("propose wisely")  # 14 characters
    prompted = load_strategy(f"{code}:Prompted")
    problem, output = write_problem(), tmp_path / "s"
    settings = {"prompt": "prompts/agent.md", "scale": 10.0}  # relative to where it runs
    report = optimize(
        problem, [trade_off()], output, trials=3, strategy=prompted, strategy_settings=settings
    )
    assert [t.candidate[RADIUS] for t in report.trials] == [240.0, 241.0, 242.0]
    recorded = json.loads((output / "study.json").read_text())["strategy"]
    assert recorded["settings"] == {"scale": 10.0}  # the file by its content, not where it is
    assert recorded["files"] == {
        "prompt": {"bytes": 14, "sha256": recorded["files"]["prompt"]["sha256"]}
    }
    assert "prompts" not in json.dumps(recorded)
    # another prompt is another study
    (tmp_path / "prompts" / "agent.md").write_text("propose boldly")
    with pytest.raises(OptimizationError, match="its strategy's prompt is not the file the study"):
        optimize(
            problem,
            [trade_off()],
            output,
            trials=1,
            resume=True,
            strategy=prompted,
            strategy_settings=settings,
        )


@pytest.mark.parametrize(
    ("edit", "changed"), [("prompt", "its prompt"), ("helper", "steps_helper")]
)
def test_a_strategy_whose_code_or_file_changes_during_a_session_stops_before_its_next_proposal(
    tmp_path: Path, write_problem: Callable[..., Path], edit: str, changed: str
) -> None:
    code = prompted_file(tmp_path / "code", f"steps_{edit}")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("propose wisely")
    prompted = load_strategy(f"{code}:Prompted")
    output = tmp_path / "s"
    with pytest.raises(OptimizationError, match=rf"is not kept: .*{changed}.* changed since"):
        optimize(
            write_problem(),
            [trade_off()],
            output,
            trials=3,
            strategy=prompted,
            strategy_settings={"prompt": str(prompt), "edit": edit},
        )
    assert endings(output) == ["recorded"]  # what it proposed before is kept, nothing after


DEFAULTED = """
    from pathlib import Path

    from k4opt.optimization import EXHAUSTED, Proposal


    class Defaulted:
        files = ("prompt",)

        def __init__(self, study, *, prompt=PROMPT):
            self.text = "" if prompt is None else Path(prompt).read_text()
            self.done = False

        def propose(self, trial):
            return EXHAUSTED if self.done else Proposal({"radius": 300.0})

        def proposed(self, trial, proposal):
            self.done = True

        def tell(self, trial, feedback): ...
"""


def test_a_file_a_strategy_reads_by_default_must_be_given_to_be_identified(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompt.md").write_text("propose wisely")
    source = DEFAULTED.replace("PROMPT", repr("prompt.md"))
    defaulted = load_strategy(f"{strategy_file(tmp_path / 'code', source)}:Defaulted")
    problem, output = write_problem(), tmp_path / "s"
    message = r"its file setting prompt is not given, so the file it reads by default \('prompt"
    with pytest.raises(OptimizationError, match=message):
        optimize(problem, [trade_off()], output, trials=1, strategy=defaulted)
    assert not output.exists()
    # given, it is identified; given as null, or none by default, there is no file to identify
    report = optimize(
        problem,
        [trade_off()],
        output,
        trials=1,
        strategy=defaulted,
        strategy_settings={"prompt": "prompt.md"},
    )
    assert report.trials[0].status == "valid"
    assert list(json.loads((output / "study.json").read_text())["strategy"]["files"]) == ["prompt"]
    optional = strategy_file(tmp_path / "optional", DEFAULTED.replace("PROMPT", "None"), "opt")
    loaded = load_strategy(f"{optional}:Defaulted")
    assert Custom.of(loaded, None).files == {}
    assert Custom.of(defaulted, {"prompt": None}).files == {}


def test_settings_that_look_like_a_credential_are_refused_before_anything_is_created(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    recorder = load_strategy(f"{strategy_file(tmp_path / 'code', RECORDER)}:Recorder")
    settings = {"radii": [300.0], "api_key": "sk-ant-api03-0123456789abcdef"}
    with pytest.raises(
        OptimizationError, match=r"its setting api_key looks like it holds a secret"
    ):
        optimize(
            write_problem(),
            [trade_off()],
            tmp_path / "s",
            trials=1,
            strategy=recorder,
            strategy_settings=settings,
        )
    assert not (tmp_path / "s").exists()


def test_a_key_rotated_between_two_sessions_does_not_refuse_the_resume(
    tmp_path: Path, write_problem: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = load_strategy(f"{strategy_file(tmp_path / 'code', RECORDER)}:Recorder")
    problem, output, settings = write_problem(), tmp_path / "s", {"radii": [200.0, 300.0]}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-first-key-0123456789")
    optimize(
        problem, [trade_off()], output, trials=1, strategy=recorder, strategy_settings=settings
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-second-key-9876543210")
    report = optimize(
        problem,
        [trade_off()],
        output,
        trials=1,
        resume=True,
        strategy=recorder,
        strategy_settings=settings,
    )
    assert [t.status for t in report.trials] == ["valid", "valid"]
    assert "sk-ant" not in (output / "study.json").read_text()


# --- what is refused ---------------------------------------------------------------------------

BROKEN = """
    from k4opt.optimization import Proposal

    NOT_A_CLASS = 3

    class Half:
        def propose(self, trial):
            return Proposal({"radius": 300.0})

    class Refusing:
        def __init__(self, study):
            raise ValueError("this study is not for me")

        def propose(self, trial): ...

        def proposed(self, trial, proposal): ...

        def tell(self, trial, feedback): ...
"""


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("missing.py:Recorder", r"strategy file .*missing\.py does not exist"),
        ("broken.py:Nowhere", r"has no 'Nowhere'"),
        ("broken.py:NOT_A_CLASS", "NOT_A_CLASS is not a class: name the class of a strategy"),
        ("broken.py:Half", r"Half is not a strategy: it has no proposed\(\) method"),
        ("Recorder", r"strategy 'Recorder': expected FILE\.py:Class"),
    ],
)
def test_a_class_that_is_no_strategy_is_refused_as_it_is_loaded(
    tmp_path: Path, reference: str, message: str
) -> None:
    strategy_file(tmp_path, BROKEN, name="broken")
    if ":" in reference:
        reference = f"{tmp_path / reference.partition(':')[0]}:{reference.partition(':')[2]}"
    with pytest.raises(OptimizationError, match=message):
        load_strategy(reference)


def test_what_cannot_run_a_strategy_of_ones_own_is_refused_before_anything_is_created(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    code = strategy_file(tmp_path / "code", RECORDER)
    strategy_file(tmp_path / "code", BROKEN, name="broken")
    recorder = load_strategy(f"{code}:Recorder")
    refusing = load_strategy(f"{tmp_path / 'code' / 'broken.py'}:Refusing")
    problem, output = write_problem(), tmp_path / "s"

    class Unloaded:
        def propose(self, trial: int) -> Any: ...

    cases: list[tuple[dict[str, Any], str]] = [
        (
            {"strategy": recorder, "strategy_settings": {"radii": [1], "nope": 2}},
            r"cannot be made with the settings \['nope', 'radii'\].*unexpected keyword",
        ),
        ({"strategy": recorder}, r"cannot be made with the settings none.*radii"),
        ({"strategy": refusing}, "could not be made for the study: ValueError: this study is not"),
        (
            {"strategy": recorder, "strategy_settings": {"radii": object()}},
            "no plain JSON|unsupported",
        ),
        ({"strategy": recorder, "strategy_settings": [("radii", [1])]}, "a JSON object by name"),
        ({"strategy": recorder, "strategy_settings": False}, "a JSON object by name"),
        (
            {"strategy": recorder, "strategy_settings": {"radii": [1]}, "seed": 3},
            "--seed seeds Optuna's sampler; a strategy of one's own takes a seed in its settings",
        ),
        ({"strategy": recorder, "grid": {RADIUS: [300.0]}}, "--grid gives the values of the grid"),
        ({"strategy": "grid", "strategy_settings": {}}, "--strategy-settings gives the settings"),
        ({"strategy": "annealing"}, "'annealing' is none of optuna, grid, nor FILE.py:Class"),
        ({"strategy": Unloaded}, "its source is not identified; load it with load_strategy"),
    ]
    for kwargs, message in cases:
        with pytest.raises(OptimizationError, match=message):
            optimize(problem, [trade_off()], output, trials=1, **kwargs)
        assert not output.exists(), kwargs


def test_a_problem_with_inner_levels_is_refused_for_a_strategy_of_ones_own(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    cells = ("test.cell_count", "simulation", "ecal_barrel.mean_hit_cells", "minimize")
    path = write_problem(
        evaluators=[
            {
                "name": "test.cell_count",
                "version": "1",
                "stage": "simulation",
                "collections": {"hits": "ECalBarrelCollection"},
            },
            {"name": "test.trade_off", "version": "1"},
        ],
        objectives=[cells],
    )
    document = yaml.safe_load(path.read_text())
    document["parameters"][RADIUS]["targets"] = {"geometry": "InnerTracker_Barrel_radius_1"}
    document["parameters"]["x"] = {
        "type": "continuous",
        "bounds": [100.0, 600.0],
        "targets": {"evaluators": {"test.trade_off": "x"}},
    }
    simulation = {"executor": "ddsim", "events": 2, "seed": 1}
    gamma = {"generation": GUN | {"particle": "gamma"}, "simulation": simulation}
    document["samples"] = {"gamma": gamma}
    a = {"evaluator": "test.trade_off", "stage": "analysis", "metric": "a", "direction": "minimize"}
    document["optimization"] = {
        "parameters": [RADIUS],
        "inner": {"parameters": ["x"], "trials": 2, "objective": a},
    }
    path.write_text(yaml.safe_dump(document))
    recorder = load_strategy(f"{strategy_file(tmp_path / 'code', RECORDER)}:Recorder")
    with pytest.raises(OptimizationError, match="runs a problem without inner levels, for now"):
        optimize(
            path,
            [cell_count(), trade_off()],
            tmp_path / "s",
            trials=1,
            strategy=recorder,
            strategy_settings={"radii": [300.0]},
        )
    assert not (tmp_path / "s").exists()


def test_a_resume_is_given_the_same_strategy_and_settings_or_refused(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    code = strategy_file(tmp_path / "code", RECORDER)
    recorder = load_strategy(f"{code}:Recorder")
    other = strategy_file(tmp_path / "other", RECORDER.replace("CALLS = []", "CALLS = [] # x"))
    changed = load_strategy(f"{other}:Recorder")
    renamed = strategy_file(tmp_path / "renamed", RECORDER.replace("class Recorder", "class Other"))
    problem, output, settings = write_problem(), tmp_path / "s", {"radii": [200.0, 300.0, 400.0]}
    optimize(
        problem, [trade_off()], output, trials=1, strategy=recorder, strategy_settings=settings
    )

    def resume(**kwargs: Any) -> Any:
        return optimize(problem, [trade_off()], output, trials=1, resume=True, **kwargs)

    cases: list[tuple[dict[str, Any], str]] = [
        ({}, "it was made with a strategy of its own, Recorder: give it again"),
        ({"strategy": "optuna"}, "leave out --strategy, --grid and --seed"),
        (
            {"strategy": recorder, "strategy_settings": {"radii": [500.0]}},
            r"its strategy's settings are \{'radii': \[200\.0, 300\.0, 400\.0\]\}, not",
        ),
        (
            {"strategy": changed, "strategy_settings": settings},
            "the file of the strategy Recorder changed",
        ),
        (
            {"strategy": load_strategy(f"{renamed}:Other"), "strategy_settings": settings},
            "it was made with the strategy Recorder, not Other",
        ),
    ]
    for kwargs, message in cases:
        with pytest.raises(OptimizationError, match=message):
            resume(**kwargs)
    assert endings(output) == ["recorded"]  # nothing changed
    report = resume(strategy=recorder, strategy_settings=settings)
    assert [t.candidate[RADIUS] for t in report.trials] == [200.0, 300.0]


@pytest.mark.parametrize(
    "changed",
    [
        {"radii": [200.0, 300.0], "batch": 1.0},  # a float where an integer was
        {"radii": [200.0, 300.0], "batch": True},  # a boolean, which Python takes for 1
        {"radii": [200, 300], "batch": 1},  # integers deep in where floats were
    ],
)
def test_a_resume_with_settings_of_other_json_types_is_refused(
    tmp_path: Path, write_problem: Callable[..., Path], changed: dict[str, Any]
) -> None:
    recorder = load_strategy(f"{strategy_file(tmp_path / 'code', RECORDER)}:Recorder")
    problem, output = write_problem(), tmp_path / "s"
    settings = {"radii": [200.0, 300.0], "batch": 1}
    optimize(
        problem, [trade_off()], output, trials=1, strategy=recorder, strategy_settings=settings
    )
    assert changed == settings  # equal to Python, other JSON: another strategy's settings
    with pytest.raises(OptimizationError, match="its strategy's settings are"):
        optimize(
            problem,
            [trade_off()],
            output,
            trials=1,
            resume=True,
            strategy=recorder,
            strategy_settings=changed,
        )
    assert endings(output) == ["recorded"]


def test_a_study_of_the_grid_is_not_resumed_with_a_strategy_of_ones_own(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    problem, output = write_problem(), tmp_path / "s"
    optimize(
        problem, [trade_off()], output, trials=1, strategy="grid", grid={RADIUS: [300.0, 400.0]}
    )
    recorder = load_strategy(f"{strategy_file(tmp_path / 'code', RECORDER)}:Recorder")
    with pytest.raises(OptimizationError, match="made with the strategy grid, not one of its own"):
        optimize(
            problem,
            [trade_off()],
            output,
            trials=1,
            resume=True,
            strategy=recorder,
            strategy_settings={"radii": [500.0]},
        )


FAILING = """
    from k4opt.optimization import Proposal


    class Failing:
        def __init__(self, study, *, error="ValueError"):
            self.error = error

        def propose(self, trial):
            return Proposal({"radius": 300.0})

        def proposed(self, trial, proposal): ...

        def tell(self, trial, feedback): ...

        @property
        def exhausted(self):
            raise {"ValueError": ValueError, "AttributeError": AttributeError}[self.error](
                "invalid optimizer state"
            )
"""


@pytest.mark.parametrize("error", ["ValueError", "AttributeError"])
def test_a_strategy_that_cannot_say_whether_it_is_exhausted_stops_the_study(
    tmp_path: Path, write_problem: Callable[..., Path], error: str
) -> None:
    failing = load_strategy(f"{strategy_file(tmp_path / 'code', FAILING)}:Failing")
    message = rf"could not say whether it is exhausted: {error}: invalid optimizer state"
    with pytest.raises(OptimizationError, match=message):  # an error, not "nothing said"
        optimize(
            write_problem(),
            [trade_off()],
            tmp_path / "s",
            trials=2,
            strategy=failing,
            strategy_settings={"error": error},
        )


# --- reading back ------------------------------------------------------------------------------


def test_a_study_reads_back_without_its_strategy(
    tmp_path: Path, write_problem: Callable[..., Path]
) -> None:
    code = strategy_file(tmp_path / "code", RECORDER, name="vanishing")
    recorder = load_strategy(f"{code}:Recorder")
    output = tmp_path / "s"
    optimize(
        write_problem(),
        [trade_off()],
        output,
        trials=2,
        strategy=recorder,
        strategy_settings={"radii": [200.0, 300.0]},
    )
    code.rename(tmp_path / "gone.py")  # the strategy is nowhere any more
    script = textwrap.dedent(
        f"""
        import json, sys
        from k4opt.optimization import load_study
        study = load_study({str(output)!r})
        print(json.dumps([len(study.trials), study.to_dict()["study"]["strategy"]["class"]]))
        assert not [name for name in sys.modules if name.startswith("k4opt_user_")]
        """
    )
    found = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert found.returncode == 0, found.stderr
    assert json.loads(found.stdout) == [2, "Recorder"]
    code, out, _ = cli("study", "export", str(output), "--format", "json")
    assert code == 0 and json.loads(out)["study"]["strategy"]["name"] == "custom"


def test_custom_strategy_construction_is_checked_for_its_kind(tmp_path: Path) -> None:
    recorder = load_strategy(f"{strategy_file(tmp_path / 'code', RECORDER)}:Recorder")
    custom = Custom.of(recorder, {"radii": [1]})
    assert custom.name == "Recorder" and dict(custom.settings) == {"radii": (1,)}
    assert custom.changed() == [] and custom.files == {}
