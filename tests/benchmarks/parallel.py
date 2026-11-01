"""Benchmark of ``--jobs``: one fixed set of candidates evaluated with 1, 2, 4 and 8 jobs.

Not a test (pytest does not collect it) and no thresholds: it prints, and writes to
``OUTPUT/benchmark.json``, for each number of jobs

- the wall time, the speedup over one job and the efficiency (speedup / jobs);
- the processor time used, by k4opt and by everything it started and waited for (workers,
  ddsim), how many processors that kept busy on average, and the share of the jobs it used;
- the occupancy of the job slots: the time the tasks took, each times the slots it held, over
  jobs x wall time (what the scheduling leaves idle). A task's time is the ``elapsed_seconds``
  its summary records: a stage's or the geometry's whole task, an evaluation's ``evaluate()``
  call alone (the record's time; the task's whole time, with making and checking its instance
  and hashing its artifacts, is not written to the summary), so the occupancy leaves out what an
  evaluation spends around that call;
- the peak memory the records hold: what a task needed at most (k4opt's process and its
  program added, an estimate), ddsim's and k4opt's own, and the session's process;
- whether every record says what the one-job run says of its candidate (all but how it ran).

The candidates are a fixed grid, not proposals, so that every run evaluates the same ones. With
``--stack real`` (needs the Key4hep stack and ``$K4GEO``) it is the pi0-rejection example on
CLD at a few events, two samples per candidate; with ``--stack fake`` the mini geometry and the
fake ddsim of the tests, which keeps a processor busy ``--burn`` seconds per simulation::

    python tests/benchmarks/parallel.py --stack real --events 10 --candidates 8 \\
        --jobs 1 2 4 8 --output /tmp/bench
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from k4opt.evaluation import load_evaluator  # noqa: E402
from k4opt.problem import EvaluationSession, RunRequest, peak_memory  # noqa: E402
from tests.support.common import scientific  # noqa: E402

EXAMPLES = REPOSITORY / "examples"
CELL = "constants.ECal_cell_size"
RADIUS = "constants.InnerTracker_Barrel_radius_1"
GRIDS = {  # the candidates of each stack, the first --candidates of them evaluated
    "real": [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0],
    "fake": [150.0 + 50 * n for n in range(8)],
}


def _real(directory: Path, events: int, threads: int) -> tuple[Path, list[Any], str]:
    document = yaml.safe_load((EXAMPLES / "problem" / "cld-ecal-pi0-rejection.yaml").read_text())
    simulation = document["stages"]["simulation"]
    simulation["events"] = events
    if threads > 1:
        simulation["numberOfThreads"] = threads
    path = directory / "problem.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    evaluators = [
        load_evaluator(f"{EXAMPLES / 'evaluators' / file}:{name}")
        for file, name in (
            ("shower_shape.py", "ShowerShape"),
            ("separation.py", "Separation"),
            ("ecal_channels.py", "EcalChannels"),
        )
    ]
    return path, evaluators, CELL


def _fake(directory: Path, events: int, burn: float) -> tuple[Path, list[Any], str]:
    from tests.support import stack
    from tests.support.evaluators import cell_count
    from tests.unit.problem.conftest import _BASE
    from tests.unit.problem.test_parallel import write_problem

    os.environ["PATH"] = f"{stack.FAKE_STACK / 'bin'}{os.pathsep}{os.environ['PATH']}"
    sys.path.insert(0, str(stack.FAKE_STACK / "python"))  # the workers get this path too
    os.environ["FAKE_DDSIM"] = json.dumps({"*": {"burn": burn}})
    from tests.support.geometry import MINI_GEO

    os.environ["K4GEO"] = str(MINI_GEO)
    document = copy.deepcopy(_BASE)
    path = write_problem(directory / "problem.yaml", document)
    raw = yaml.safe_load(path.read_text())
    for sample in raw["samples"].values():
        sample["simulation"]["events"] = events
    path.write_text(yaml.safe_dump(raw))
    return path, [cell_count()], RADIUS


def _cpu() -> float:
    """Processor seconds of this process and of what it waited for, so far."""
    own, waited = (
        resource.getrusage(resource.RUSAGE_SELF),
        resource.getrusage(resource.RUSAGE_CHILDREN),
    )
    return own.ru_utime + own.ru_stime + waited.ru_utime + waited.ru_stime


def _tasks(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = [summary["geometry"], *summary["evaluations"].values()]
    for sample in summary["samples"].values():
        entries += [*sample["stages"].values(), *sample["evaluations"].values()]
    return entries


def held(entry: Mapping[str, Any], jobs: int) -> float:
    """The slot seconds a task held: its time, times the slots it held, as the local backend
    counts them (``ProcessBackend.cost``): a simulation one per thread, at most every slot."""
    seconds = float(entry.get("elapsed_seconds") or 0.0)
    threads = [
        int(arg.split("=", 1)[1])
        for arg in entry.get("argv") or ()
        if arg.startswith("--numberOfThreads=")
    ]
    return seconds * min(max(threads or [1]), jobs)


def run(
    jobs: int,
    path: Path,
    evaluators: list[Any],
    parameter: str,
    values: list[float],
    directory: Path,
) -> dict[str, Any]:
    requests = [RunRequest({parameter: value}, directory / f"c{value:g}") for value in values]
    clear = Path("/proc/self/clear_refs")
    if clear.exists():
        clear.write_text("5")  # the session's own peak, from now
    cpu, began = _cpu(), time.perf_counter()
    summaries = []
    with (
        EvaluationSession.open(path, evaluators, jobs=jobs) as session,
        session.evaluate_each(requests) as evaluations,
    ):
        for _, outcome in evaluations:
            if isinstance(outcome, Exception):
                raise outcome
            summaries.append(dict(outcome.summary))
    wall = time.perf_counter() - began
    used = _cpu() - cpu
    used_slots = sum(held(entry, jobs) for s in summaries for entry in _tasks(s))
    memories = [m for s in summaries for m in peak_memory(s).values()]
    per_task = max((m.per_task or 0 for m in memories), default=0)
    programs = max((m.program or 0 for m in memories), default=0)
    processes = max((m.process or 0 for m in memories), default=0)
    session = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "jobs": jobs,
        "wall_seconds": wall,
        "cpu_seconds": used,
        "processors_busy": used / wall,
        "utilization": used / wall / jobs,
        "occupancy": used_slots / (jobs * wall),
        "peak_memory_bytes": {
            "per_task": per_task,
            "ddsim": programs,
            "k4opt_task_process": processes,
            "session_process": session if jobs > 1 else None,  # one job: it ran the tasks
        },
        "candidates": {s["candidate"][parameter]: scientific(s) for s in summaries},
    }


def _gib(count: int | None) -> str:
    return "-" if count is None else f"{count / 2**30:.2f}"


def _check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse what would run something else than asked, or report what was not measured."""
    if args.jobs[0] != 1:  # speedup and efficiency are over one job, measured
        parser.error("the first of --jobs must be 1: speedup and efficiency are over one job")
    if min(args.jobs) < 1 or len(set(args.jobs)) != len(args.jobs):
        parser.error(f"--jobs must be distinct positive numbers, got {args.jobs}")
    grid = GRIDS[args.stack]
    if not 1 <= args.candidates <= len(grid):
        parser.error(f"--candidates must be 1 to {len(grid)} (the {args.stack} grid)")
    if args.events < 1:
        parser.error(f"--events must be positive, got {args.events}")
    if args.threads < 1:
        parser.error(f"--threads must be positive, got {args.threads}")
    if args.stack == "fake" and args.threads != 1:
        parser.error("--threads is for --stack real: the fake ddsim runs no threads")
    if args.burn < 0:
        parser.error(f"--burn must not be negative, got {args.burn}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stack", choices=("real", "fake"), default="fake")
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--events", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1, help="numberOfThreads (real stack)")
    parser.add_argument("--burn", type=float, default=5.0, help="processor s per fake ddsim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _check(parser, args)  # before anything is created
    args.output.mkdir(parents=True)
    if args.stack == "real":
        path, evaluators, parameter = _real(args.output, args.events, args.threads)
    else:
        path, evaluators, parameter = _fake(args.output, args.events, args.burn)
    values = GRIDS[args.stack][: args.candidates]
    results = []
    for jobs in args.jobs:
        directory = args.output / f"jobs{jobs}"
        results.append(run(jobs, path, evaluators, parameter, values, directory))
        print(f"jobs {jobs}: {results[-1]['wall_seconds']:.1f} s", file=sys.stderr, flush=True)
    base = results[0]  # one job
    print(
        "| jobs | wall s | speedup | efficiency | CPU s | processors busy | utilization | "
        "occupancy | GiB per task | ddsim GiB | k4opt task GiB | session GiB | records |"
    )
    print("|" + " --- |" * 13)
    for result in results:
        speedup = base["wall_seconds"] / result["wall_seconds"]
        same = result["candidates"] == base["candidates"]
        memory = result["peak_memory_bytes"]
        result |= {"speedup": speedup, "efficiency": speedup / result["jobs"], "same": same}
        print(
            f"| {result['jobs']} | {result['wall_seconds']:.1f} | {speedup:.2f} | "
            f"{speedup / result['jobs']:.0%} | {result['cpu_seconds']:.1f} | "
            f"{result['processors_busy']:.2f} | {result['utilization']:.0%} | "
            f"{result['occupancy']:.0%} | {_gib(memory['per_task'])} | {_gib(memory['ddsim'])} | "
            f"{_gib(memory['k4opt_task_process'])} | {_gib(memory['session_process'])} | "
            f"{'same' if same else 'DIFFER'} |"
        )
    environment = {
        "stack": args.stack,
        "events": args.events,
        "candidates": values,
        "threads": args.threads,
        "processors": len(os.sched_getaffinity(0)),
        "KEY4HEP_STACK": os.environ.get("KEY4HEP_STACK"),
    }
    for result in results:
        result.pop("candidates")
    (args.output / "benchmark.json").write_text(
        json.dumps({"environment": environment, "runs": results}, indent=2) + "\n"
    )
    return 0 if all(result["same"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
