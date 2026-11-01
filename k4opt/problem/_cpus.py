"""Internal: how many processors this process may use, for ``--jobs auto``.

The processors it may run on (its affinity: a Slurm allocation, ``taskset``) and, where the
system has cgroups v2, the processor time its cgroup and every cgroup above it may take
(``cpu.max``: a container, a systemd unit with ``CPUQuota``), rounded up. Nothing else: not the
other users of a shared machine, nor memory, nor a batch system's request that is not enforced
as one of these (HTCondor without affinity), for which ``--jobs`` is given as a number.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

AUTO = "auto"

_PROC = Path("/proc/self/cgroup")
_CGROUPS = Path("/sys/fs/cgroup")


def usable_cpus(proc: Path = _PROC, cgroups: Path = _CGROUPS) -> int:
    """The processors this process may use: its affinity, at most its cgroup's quota; at
    least 1."""
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:  # pragma: no cover - not Linux
        count = os.cpu_count() or 1
    quota = cgroup_quota(proc, cgroups)
    return max(1, count if quota is None else min(count, quota))


def cgroup_quota(proc: Path = _PROC, cgroups: Path = _CGROUPS) -> int | None:
    """The processors the cgroup v2 of this process and those above it may keep busy, rounded
    up: the smallest ``cpu.max`` quota among them; ``None`` without one, or without cgroups
    v2."""
    try:
        lines = proc.read_text().splitlines()
    except OSError:
        return None
    unified = [line[3:] for line in lines if line.startswith("0::")]
    if not unified:
        return None
    directory = cgroups / unified[0].strip().lstrip("/")
    quotas = []
    while True:
        quota = _quota(directory / "cpu.max")
        if quota is not None:
            quotas.append(quota)
        if directory == cgroups or cgroups not in directory.parents:
            break
        directory = directory.parent
    return min(quotas) if quotas else None


def _quota(path: Path) -> int | None:
    """``cpu.max`` (``"200000 100000"``: 2 processors; ``"max 100000"``: no limit)."""
    try:
        quota, period = path.read_text().split()[:2]
        if quota == "max":
            return None
        return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        return None
