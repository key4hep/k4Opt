"""Internal: how many processors this process may use, for ``--jobs auto``.

The processors it may run on (its affinity: a Slurm allocation, ``taskset``) and, where the
system has cgroups v2, the processor time its cgroup and every cgroup above it that it can see
may take (``cpu.max``: a container, a systemd unit with ``CPUQuota``), rounded up. Nothing else:
not the other users of a shared machine, nor memory, nor a batch system's request that is not
enforced as one of these (HTCondor without affinity), for which ``--jobs`` is given as a number.
"""

from __future__ import annotations

import math
import os
from pathlib import Path, PurePosixPath

AUTO = "auto"

_PROC = Path("/proc/self/cgroup")
_MOUNTINFO = Path("/proc/self/mountinfo")


def usable_cpus(proc: Path = _PROC, mountinfo: Path = _MOUNTINFO) -> int:
    """The processors this process may use: its affinity, at most its cgroup's quota; at
    least 1."""
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:  # pragma: no cover - not Linux
        count = os.cpu_count() or 1
    quota = cgroup_quota(proc, mountinfo)
    return max(1, count if quota is None else min(count, quota))


def cgroup_quota(proc: Path = _PROC, mountinfo: Path = _MOUNTINFO) -> int | None:
    """The processors the cgroup v2 of this process and those above it may keep busy, rounded
    up: the smallest ``cpu.max`` quota among them; ``None`` without one, or without cgroups
    v2.

    The cgroup's path (``/proc/self/cgroup``) is taken in the hierarchy, and that hierarchy may
    be mounted from a subtree of its own (a container mounts its cgroup alone): the mount's
    root and mount point (``/proc/self/mountinfo``) say where the cgroup's directory is. Only
    the cgroups below the mount point can be read; those above it are not seen.
    """
    try:
        lines = proc.read_text().splitlines()
    except OSError:
        return None
    unified = [line[3:].strip() for line in lines if line.startswith("0::")]
    mount = _cgroup2_mount(mountinfo)
    if not unified or mount is None:
        return None
    root, point = mount
    cgroup = PurePosixPath(unified[0])
    try:
        below = cgroup.relative_to(root)
    except ValueError:  # not below what is mounted: only the mount's own limit is seen
        below = PurePosixPath()
    directory = point / below
    quotas = []
    while True:
        quota = _quota(directory / "cpu.max")
        if quota is not None:
            quotas.append(quota)
        if directory == point or point not in directory.parents:
            break
        directory = directory.parent
    return min(quotas) if quotas else None


def _cgroup2_mount(mountinfo: Path) -> tuple[PurePosixPath, Path] | None:
    """``(root, mount point)`` of the first cgroup2 mount: the cgroup the mount shows at its
    point (``/`` for the whole hierarchy). ``None`` without one."""
    try:
        lines = mountinfo.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        fields, _, after = line.partition(" - ")
        parts = fields.split()
        if after.split()[:1] == ["cgroup2"] and len(parts) >= 5:
            return PurePosixPath(_unescaped(parts[3])), Path(_unescaped(parts[4]))
    return None


def _unescaped(field: str) -> str:
    """A path of mountinfo, whose spaces, tabs, newlines and backslashes are octal escapes."""
    for code, char in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        field = field.replace(code, char)
    return field


def _quota(path: Path) -> int | None:
    """``cpu.max`` (``"200000 100000"``: 2 processors; ``"max 100000"``: no limit)."""
    try:
        quota, period = path.read_text().split()[:2]
        if quota == "max":
            return None
        return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        return None
