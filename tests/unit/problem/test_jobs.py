"""``--jobs auto``: the processors this process may use, from its affinity and its cgroup v2
quota, and how a session takes it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from k4opt.cli import main
from k4opt.problem import EvaluationSession, RunError, _cpus, _run
from k4opt.problem._cpus import cgroup_quota, usable_cpus
from tests.support.evaluators import cell_count
from tests.support.geometry import MINI_GEO
from tests.unit.problem.test_parallel import write_problem


@pytest.fixture(autouse=True)
def mini_k4geo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K4GEO", str(MINI_GEO))


@pytest.fixture
def problem_file(tmp_path: Path, doc: dict[str, Any]) -> Path:
    return write_problem(tmp_path / "problem.yaml", doc)


def cgroups(tmp_path: Path, own: str, limits: dict[str, str], root: str = "/") -> tuple[Path, Path]:
    """``/proc/self/cgroup`` naming ``own`` and ``/proc/self/mountinfo`` with the cgroup2
    hierarchy mounted from ``root`` at ``tmp_path/sys/fs/cgroup``, whose ``cpu.max`` ``limits``
    are by directory relative to that mount point (``""`` for the mount point itself)."""
    point = tmp_path / "sys" / "fs" / "cgroup"
    below = own[len(root) :] if own.startswith(root) else ""
    (point / below.lstrip("/")).mkdir(parents=True, exist_ok=True)
    for directory, limit in limits.items():
        (point / directory / "cpu.max").write_text(limit + "\n")
    proc = tmp_path / "cgroup"
    proc.write_text(f"0::{own}\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "24 1 253:0 / / rw,relatime shared:1 - xfs /dev/vda1 rw\n"
        f"30 24 0:27 {root} {point} rw,nosuid,nodev shared:4 - cgroup2 cgroup2 rw,nsdelegate\n"
    )
    return proc, mountinfo


@pytest.mark.parametrize(
    ("limits", "quota"),
    [
        ({}, None),
        ({"user.slice": "max 100000"}, None),
        ({"user.slice/job": "200000 100000"}, 2),
        ({"user.slice/job": "150000 100000"}, 2),  # 1.5 processors: rounded up
        ({"user.slice/job": "50000 100000"}, 1),
        ({"user.slice/job": "400000 100000", "user.slice": "300000 100000"}, 3),  # the parent's
        ({"user.slice/job": "garbled"}, None),
    ],
)
def test_the_quota_is_the_smallest_cpu_max_of_the_cgroup_and_those_above_it(
    tmp_path: Path, limits: dict[str, str], quota: int | None
) -> None:
    proc, mountinfo = cgroups(tmp_path, "/user.slice/job", limits)
    assert cgroup_quota(proc, mountinfo) == quota


def test_a_container_that_mounts_its_own_cgroup_alone_finds_its_quota(tmp_path: Path) -> None:
    # the hierarchy's /kubepods/pod7/ctr is what /sys/fs/cgroup shows: its cpu.max is there
    proc, mountinfo = cgroups(
        tmp_path, "/kubepods/pod7/ctr", {"": "200000 100000"}, root="/kubepods/pod7/ctr"
    )
    assert cgroup_quota(proc, mountinfo) == 2
    # a cgroup below it, and the mounted one above it
    proc, mountinfo = cgroups(
        tmp_path / "deeper",
        "/kubepods/pod7/ctr/init",
        {"": "400000 100000", "init": "max 100000"},
        root="/kubepods/pod7/ctr",
    )
    assert cgroup_quota(proc, mountinfo) == 4


def test_a_cgroup_outside_what_is_mounted_sees_only_the_mount_points_limit(tmp_path: Path) -> None:
    proc, mountinfo = cgroups(tmp_path, "/elsewhere/job", {"": "300000 100000"}, root="/kubepods")
    assert cgroup_quota(proc, mountinfo) == 3


def test_without_cgroups_v2_there_is_no_quota(tmp_path: Path) -> None:
    proc, mountinfo = cgroups(tmp_path, "/job", {"job": "200000 100000"})
    proc.write_text("12:cpu,cpuacct:/user.slice\n")  # cgroups v1 only
    assert cgroup_quota(proc, mountinfo) is None
    proc.write_text("0::/job\n")
    mountinfo.write_text("24 1 253:0 / / rw - xfs /dev/vda1 rw\n")  # no cgroup2 mounted
    assert cgroup_quota(proc, mountinfo) is None
    assert cgroup_quota(tmp_path / "missing", mountinfo) is None
    assert cgroup_quota(proc, tmp_path / "missing") is None


def test_usable_processors_are_the_affinity_at_most_the_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_cpus.os, "sched_getaffinity", lambda pid: set(range(8)))
    proc, mountinfo = cgroups(tmp_path, "/job", {"job": "300000 100000"})
    assert usable_cpus(proc, mountinfo) == 3
    proc, mountinfo = cgroups(tmp_path / "wide", "/job", {"job": "3200000 100000"})
    assert usable_cpus(proc, mountinfo) == 8
    assert usable_cpus(tmp_path / "missing", mountinfo) == 8


def test_this_machine_resolves_to_a_positive_number_of_processors() -> None:
    assert usable_cpus() >= 1


def test_auto_takes_as_many_jobs_as_usable_processors(
    problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_run, "usable_cpus", lambda: 3)
    with EvaluationSession.open(problem_file, [cell_count()], jobs="auto") as session:
        assert session._execution["jobs"] == 3
        assert not [warning for warning in session.warnings if "processors" in warning]


def test_more_jobs_than_processors_are_warned_about(
    problem_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_run, "usable_cpus", lambda: 1)
    with EvaluationSession.open(problem_file, [cell_count()], jobs=2) as session:
        assert [w for w in session.warnings if "processors" in w] == [
            "2 jobs, more than the 1 processors this process may use: tasks will share processors"
        ]


@pytest.mark.parametrize("jobs", [0, -2, True, 1.5, "2", "all"])
def test_jobs_must_be_a_positive_integer_or_auto(problem_file: Path, jobs: object) -> None:
    with pytest.raises(RunError, match="jobs must be a positive integer or 'auto'"):
        EvaluationSession.open(problem_file, [cell_count()], jobs=jobs)  # type: ignore[arg-type]


@pytest.mark.parametrize("command", ["evaluate", "optimize"])
def test_the_command_lines_take_a_number_or_auto(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        main([command, "problem.yaml", "--jobs", "many"])
    assert info.value.code == 2  # a usage error, before anything is read
    assert "expected a number of jobs or auto, got 'many'" in capsys.readouterr().err
