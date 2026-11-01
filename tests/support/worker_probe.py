"""``test.worker_probe``: whatever a test of worker processes needs an evaluator to do there,
which no monkeypatch reaches. Reads the simulation stage (none of its collections) and measures
``analysis/checked``: 1 if ``check()`` ran on this very instance before ``evaluate()``.

What else it does is said by the environment variable ``WORKER_PROBE``, a JSON object keyed by
the binding, and by ``<run>.<binding>`` for one run directory alone (a worker keeps the
environment it started with): ``mark`` writes a file first, ``await`` waits until each file of
a list exists, ``allocate`` holds that many MiB for a moment, ``print`` prints that text to
stdout and stderr, ``raise`` raises a
``ValueError`` with that text, ``contract`` an ``EvaluationContractError``, ``kill`` sends
its own process SIGKILL, and ``stall`` writes the start of an answer to the coordinator and
never the rest."""

import json
import os
import signal
import struct
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
        if "allocate" in told:
            held = bytearray(told["allocate"] << 20)
            held[:: 1 << 12] = b"x" * len(held[:: 1 << 12])  # every page resident
            del held
        if "print" in told:
            print(told["print"], flush=True)
            print(told["print"], file=sys.stderr, flush=True)
        if "raise" in told:
            raise ValueError(told["raise"])
        if "contract" in told:
            raise EvaluationContractError(told["contract"])
        if "kill" in told:
            os.kill(os.getpid(), signal.SIGKILL)
        if "stall" in told:  # the header of an answer whose rest never comes, then nothing
            answers = int(sys.argv[2])  # the worker's end of its socket (``_worker.main``)
            os.write(answers, struct.pack("!i", 1000))
            time.sleep(600)
        return {("analysis", "checked"): 1 if self.checked else 0}
