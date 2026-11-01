"""``test.worker_probe``: whatever a test of worker processes needs an evaluator to do there,
which no monkeypatch reaches. Reads the simulation stage (none of its collections) and measures
``analysis/checked``: 1 if ``check()`` ran on this very instance before ``evaluate()``.

What else it does is said by the environment variable ``WORKER_PROBE``, a JSON object keyed by
the binding, and by ``<run>.<binding>`` for one run directory alone (a worker keeps the
environment it started with): ``mark`` writes a file first, ``await`` waits until each file of
a list exists, ``print`` prints that text to stdout and stderr, ``raise`` raises a
``ValueError`` with that text, ``contract`` an ``EvaluationContractError``, and ``kill`` sends
its own process SIGKILL."""

import json
import os
import signal
import sys
import time
from pathlib import Path

from k4opt.evaluation import EvaluationContractError, Evaluator


class WorkerProbe(Evaluator):
    name = "test.worker_probe"
    version = "1"
    stage = "simulation"
    settings = {"binding": ""}
    metrics = {("analysis", "checked"): "1"}

    def __init__(self):
        self.checked = False

    def check(self, request):
        super().check(request)
        self.checked = True

    def evaluate(self, request):
        binding = request.settings["binding"]
        run = request.workdir.parents[3].name  # <run>/samples/<sample>/evaluators/<binding>
        given = json.loads(os.environ.get("WORKER_PROBE", "{}"))
        told = {**given.get(binding, {}), **given.get(f"{run}.{binding}", {})}
        if "mark" in told:
            Path(told["mark"]).write_text(str(os.getpid()))
        deadline = time.monotonic() + 60
        while not all(Path(name).exists() for name in told.get("await", [])):
            if time.monotonic() > deadline:
                raise TimeoutError(f"still waiting for {told['await']}")
            time.sleep(0.01)
        if "print" in told:
            print(told["print"], flush=True)
            print(told["print"], file=sys.stderr, flush=True)
        if "raise" in told:
            raise ValueError(told["raise"])
        if "contract" in told:
            raise EvaluationContractError(told["contract"])
        if "kill" in told:
            os.kill(os.getpid(), signal.SIGKILL)
        return {("analysis", "checked"): 1 if self.checked else 0}
