"""``test.worker_probe``: whatever a test of worker processes needs an evaluator to do there,
which no monkeypatch reaches. Reads the simulation stage (none of its collections) and measures
``analysis/checked``: 1.

What else it does is said by the environment variable ``WORKER_PROBE``, a JSON object keyed by
the binding, by ``<sample>/<binding>`` for one sample alone, and by ``<run>.<binding>`` for one
run directory alone (a worker keeps the environment it started with): ``mark`` writes a file
first, ``await`` waits until each file of a list exists, ``allocate`` holds that many MiB for a
moment, ``print`` prints that text to stdout and stderr, ``raise`` raises a
``ValueError`` with that text, ``contract`` an ``EvaluationContractError``, ``kill`` sends its
own process SIGKILL, ``stall`` writes the start of an answer to the coordinator and never
the rest, ``sleep`` sleeps that many seconds, ``ignore_term`` ignores SIGTERM first (so that only
SIGKILL ends it), ``child`` starts a ``sleep`` program that would outlive it, writing its pid to
that file. Under the key ``init``, ``sleep``, ``print`` and ``allocate`` do the same while any
instance is created. In a worker process that loaded it, ``{"materialize": {"sleep": S}}`` makes
every candidate's geometry take S seconds more to make: k4opt's own work hanging there."""

import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

from k4opt.evaluation import EvaluationContractError, Evaluator
from k4opt.geometry import GeometrySource


def _materialize_as_told(make):
    def materialize(self, *args, **kwargs):
        told = json.loads(os.environ.get("WORKER_PROBE", "{}")).get("materialize", {})
        time.sleep(told.get("sleep", 0))
        return make(self, *args, **kwargs)

    return materialize


if "k4opt.execution._worker" in sys.modules and sys.argv[0] == "-c":  # a worker process alone
    GeometrySource.materialize = _materialize_as_told(GeometrySource.materialize)


class WorkerProbe(Evaluator):
    name = "test.worker_probe"
    version = "1"
    stage = "simulation"
    settings = {"binding": ""}
    metrics = {("analysis", "checked"): "1"}

    def __init__(self):
        told = json.loads(os.environ.get("WORKER_PROBE", "{}")).get("init", {})
        if "print" in told:
            print(told["print"], flush=True)
        if "allocate" in told:
            held = bytearray(told["allocate"] << 20)
            held[:: 1 << 12] = b"x" * len(held[:: 1 << 12])  # every page resident
            del held
        time.sleep(told.get("sleep", 0))

    def evaluate(self, request):
        binding = request.settings["binding"]
        # <run>/samples/<sample>/evaluators/<binding>
        run, sample = request.workdir.parents[3].name, request.workdir.parents[1].name
        given = json.loads(os.environ.get("WORKER_PROBE", "{}"))
        told = {
            **given.get(binding, {}),
            **given.get(f"{sample}/{binding}", {}),
            **given.get(f"{run}.{binding}", {}),
        }
        if "mark" in told:
            Path(told["mark"]).write_text(str(os.getpid()))
        if "ignore_term" in told:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if "child" in told:
            child = subprocess.Popen(["sleep", "600"])  # in the worker's process group
            Path(told["child"]).write_text(str(child.pid))
        time.sleep(told.get("sleep", 0))
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
        return {("analysis", "checked"): 1}
