"""What a stage used, from the traces of the program it ran: what a footprint holds, what it leaves
out, and when it no longer holds."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from k4opt import program
from k4opt.software import Footprint, Local, SoftwareError
from k4opt.software.observe import (
    FOOTPRINT,
    changed_inputs,
    covered,
    footprint,
    identity,
    load,
    observe,
)

LOGS = {"stdout": "out.log", "stderr": "err.log", "command": "command.json"}

strace = pytest.mark.skipif(
    program.unobservable("files") is not None, reason="strace cannot trace here"
)


STARTED = 'execve("/usr/bin/true", ["true"], 0x7ffc /* 180 vars */) = 0'


def traces(directory: Path, *lines: str, name: str = "t.1") -> Path:
    """A directory of raw traces, one file with ``lines``, as strace writes them."""
    directory.mkdir(exist_ok=True)
    (directory / name).write_text("".join(f"{line}\n" for line in lines))
    return directory


def opened(path: Path | str, flags: str = "O_RDONLY|O_CLOEXEC", cwd: str = "/work") -> str:
    return f'openat(AT_FDCWD<{cwd}>, "{path}", {flags}) = 3<{path}>'


def failed(path: Path | str, error: str = "ENOENT (No such file or directory)") -> str:
    return f'openat(AT_FDCWD</work>, "{path}", O_RDONLY) = -1 {error}'


def test_what_a_run_read_found_missing_and_started_is_its_footprint(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("1.05\n")
    listed = tmp_path / "plugins"
    listed.mkdir()
    (listed / "one.so").touch()
    found = footprint(
        traces(
            tmp_path / "raw",
            STARTED,
            opened(calibration),
            opened(listed, "O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY"),
            failed(tmp_path / "absent"),
            failed(tmp_path / "secret", "EACCES (Permission denied)"),
            failed("/etc/shadow-rw", "EROFS (Read-only file system)"),
        ),
        "files",
        [],
    )
    assert found.complete
    digest = hashlib.sha256(b"1.05\n").hexdigest()
    assert found.read[str(calibration)] == f"sha256:{digest}"
    assert str(found.read[str(listed)]).startswith("names:")
    assert found.missing == (str(tmp_path / "absent"),)
    assert found.denied == (str(tmp_path / "secret"),)
    assert found.ran == ("/usr/bin/true",)
    assert "/usr/bin/true" in found.read  # what a program is, as any file it read


@pytest.mark.parametrize(
    ("flags", "read"),
    [
        ("O_RDWR", True),  # a database opened to be queried
        ("O_RDONLY|O_CREAT, 0666", True),  # an existing file, opened as it is
        ("O_RDWR|O_APPEND", True),
        ("O_WRONLY|O_CREAT|O_TRUNC, 0666", False),  # written, never read
        ("O_WRONLY|O_APPEND", False),
        ("O_RDWR|O_TRUNC", False),  # emptied first: what it held was never read
        ("O_RDWR|O_CREAT|O_EXCL, 0600", False),  # made anew
    ],
)
def test_a_file_opened_to_be_written_too_is_still_an_input(
    tmp_path: Path, flags: str, read: bool
) -> None:
    database = tmp_path / "calibration.db"
    database.write_text("rows")
    found = footprint(traces(tmp_path / "raw", STARTED, opened(database, flags)), "files", [])
    assert (str(database) in found.read) is read


def test_a_file_opened_read_write_and_changed_by_the_run_vouches_for_nothing(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cache.db"
    database.write_text("before")
    began = time.time_ns()
    time.sleep(0.02)
    database.write_text("after the run wrote it")
    line = opened(database, "O_RDWR")
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [], began)
    assert found.read[str(database)] is None


def test_a_program_started_through_a_link_retargeted_while_it_ran_vouches_for_nothing(
    tmp_path: Path,
) -> None:
    for name in ("helper-v1", "helper-v2"):
        (tmp_path / name).write_text("#!/bin/sh\n")
    current = tmp_path / "current"
    current.symlink_to("helper-v1")
    began = time.time_ns()
    time.sleep(0.02)
    current.unlink()
    current.symlink_to("helper-v2")
    line = f'execve("{current}", ["current"], 0x7ffc /* 1 var */) = 0'
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [], began)
    assert found.read[str(current)] is None
    assert (
        footprint(traces(tmp_path / "again", STARTED, line), "files", []).read[str(current)]
        is not None
    )  # unchanged since it began: identified


def test_the_lines_that_could_not_be_read_are_kept(tmp_path: Path) -> None:
    found = footprint(traces(tmp_path / "raw", STARTED, "garbage"), "files", [])
    assert found.unreadable == ("garbage",)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert footprint(empty, "files", []).unreadable == ("no program started",)


@pytest.mark.parametrize(
    "line",
    [
        "socket(AF_UNIX, SOCK_STREAM, 0)         = 3<socket:[59859]>",
        'mkdir("/tmp/x", 0777)                   = -1 EEXIST (File exists)',
        'open("/etc/hosts", O_RDONLY)           = 3</etc/hosts>',
    ],
)
def test_a_short_call_strace_pads_to_its_column_is_read(tmp_path: Path, line: str) -> None:
    # strace starts what a call returned at column 40, padding a short call with spaces
    assert footprint(traces(tmp_path / "raw", STARTED, line), "files", []).complete


def test_a_relative_lookup_is_where_its_process_was(tmp_path: Path) -> None:
    line = 'openat(AT_FDCWD</some/job>, "rootlogon.C", O_RDONLY) = -1 ENOENT (No such file)'
    assert footprint(traces(tmp_path / "raw", line), "files", []).missing == (
        "/some/job/rootlogon.C",
    )


def test_a_lookup_through_a_link_and_up_is_where_the_system_looked(tmp_path: Path) -> None:
    # current -> data/release/bin: current/.. is data/release, not where current is
    (tmp_path / "data" / "release" / "bin").mkdir(parents=True)
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "current").symlink_to(tmp_path / "data" / "release" / "bin")
    looked = f"{tmp_path}/work/current/../calibration.dat"
    found = footprint(traces(tmp_path / "raw", STARTED, failed(looked)), "files", [])
    assert found.missing == (looked,) and found.changes() == []
    (tmp_path / "work" / "calibration.dat").touch()  # where a lexical .. would look: no matter
    assert found.changes() == []
    (tmp_path / "data" / "release" / "calibration.dat").touch()  # where the system looks
    assert found.changes() == [f"{looked} appeared"]


def test_a_relative_lookup_up_from_a_link_is_where_the_system_looked(tmp_path: Path) -> None:
    (tmp_path / "data" / "job").mkdir(parents=True)
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "job").symlink_to(tmp_path / "data" / "job")
    line = f'openat(AT_FDCWD<{tmp_path}/work/job>, "../override.json", O_RDONLY) = -1 ENOENT (x)'
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [])
    # whichever directory strace names, the path is kept as the system walks it from there
    assert found.missing == (f"{tmp_path}/work/job/../override.json",)


def test_what_the_stage_made_was_given_or_wrote_is_no_input(tmp_path: Path) -> None:
    own, given, scratch = tmp_path / "stage", tmp_path / "events.root", tmp_path / "cling.o"
    own.mkdir()
    (own / "job.py").touch()
    given.touch()
    scratch.touch()
    found = footprint(
        traces(
            tmp_path / "raw",
            opened(own / "job.py"),
            opened(given),
            opened(scratch, "O_RDWR|O_CREAT|O_EXCL, 0600"),
            opened(scratch),
            failed(own / "rootlogon.C"),
            opened("/proc/self/maps"),
            opened("/dev/urandom"),
            opened("/var/lib/sss/mc/passwd"),
        ),
        "files",
        [own, given],
    )
    assert found.read == {} and found.missing == ()


def test_a_file_removed_while_it_was_open_is_the_file_it_was(tmp_path: Path) -> None:
    gone = tmp_path / "scratch"
    line = (
        f'openat(AT_FDCWD</work>, "{gone}", O_RDWR|O_CREAT|O_EXCL, 0600) = 3<{gone} (deleted)>\n'
        f'openat(AT_FDCWD</work>, "{gone}", O_RDONLY) = 4<{gone} (deleted)>'
    )
    assert footprint(traces(tmp_path / "raw", line), "files", []).read == {}


def test_paths_are_read_as_strace_escapes_them(tmp_path: Path) -> None:
    odd = tmp_path / 'with "quotes" and\ta tab'
    odd.write_text("x")
    written = str(odd).replace('"', r"\"").replace("\t", r"\t")
    line = f'openat(AT_FDCWD</work>, "{written}", O_RDONLY) = 3<{written}>'
    assert str(odd) in footprint(traces(tmp_path / "raw", line), "files", []).read


def test_a_path_that_is_no_text_is_kept_and_comes_back(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    name = os.fsencode(tmp_path) + b"/data\xe9"
    with open(name, "wb") as handle:
        handle.write(b"x")
    escaped = name.replace(b"\xe9", b"\\351")
    line = b'openat(AT_FDCWD</work>, "' + escaped + b'", O_RDONLY) = 3<' + escaped + b">\n"
    (raw / "t.1").write_bytes(STARTED.encode() + b"\n" + line)
    summary, path = observe(raw, "files", directory=tmp_path, own=[])
    assert path is not None and summary["complete"] is True
    assert os.fsdecode(name) in load(path).read


def test_a_connection_to_the_network_is_kept_and_a_local_socket_is_not(tmp_path: Path) -> None:
    found = footprint(
        traces(
            tmp_path / "raw",
            'connect(3<socket:[1]>, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110)'
            " = -1 ENOENT (No such file or directory)",
            'connect(4<socket:[2]>, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("'
            '10.0.0.1")}, 16) = -1 EINPROGRESS (Operation now in progress)',
            "connect(5<socket:[3]>, {sa_family=AF_INET6, sin6_port=htons(443), sin6_flowinfo="
            'htonl(0), inet_pton(AF_INET6, "::1", &sin6_addr), sin6_scope_id=0}, 28) = 0',
        ),
        "files",
        [],
    )
    assert found.connected == ("10.0.0.1:53", "::1:443")


def test_a_datagram_to_a_local_service_is_an_input_and_to_the_system_s_own_is_not(
    tmp_path: Path,
) -> None:
    service = tmp_path / "service.sock"
    found = footprint(
        traces(
            tmp_path / "raw",
            f'sendto(3<socket:[1]>, "request", 7, 0, {{sa_family=AF_UNIX, sun_path="{service}"}},'
            " 110) = 7",
            'sendto(4<socket:[2]>, "<13>log", 7, MSG_NOSIGNAL, {sa_family=AF_UNIX, sun_path='
            '"/dev/log"}, 110) = 7',
            "sendto(5<socket:[3]>, [{nlmsg_len=20, nlmsg_type=RTM_GETLINK}, {}], 20, 0, "
            "{sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, 12) = 20",
            f'sendmsg(6<socket:[4]>, {{msg_name={{sa_family=AF_UNIX, sun_path="{service}2"}}, '
            "msg_namelen=110, msg_iov=[], msg_iovlen=0, msg_controllen=0, msg_flags=0}, 0) = 0",
            'sendto(7<socket:[5]>, "data", 4, 0, NULL, 0) = 4',  # on a connected socket
        ),
        "files",
        [],
    )
    assert found.connected == (f"unix:{service}", f"unix:{service}2")


def test_a_network_socket_is_network_use_without_a_connection(tmp_path: Path) -> None:
    found = footprint(
        traces(
            tmp_path / "raw",
            "socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC, IPPROTO_IP) = 3<socket:[1]>",
            "socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP) = -1 EAFNOSUPPORT (Not supported)",
            "socket(AF_UNIX, SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK, 0) = 4<socket:[2]>",
            "socket(AF_NETLINK, SOCK_RAW|SOCK_CLOEXEC, NETLINK_ROUTE) = 5<socket:[3]>",
        ),
        "files",
        [],
    )
    assert found.connected == ("AF_INET socket",)


def test_a_local_service_may_answer_with_anything_and_the_system_s_own_does_not(
    tmp_path: Path,
) -> None:
    service, absent = tmp_path / "service.sock", tmp_path / "gone.sock"
    found = footprint(
        traces(
            tmp_path / "raw",
            f'connect(3<socket:[1]>, {{sa_family=AF_UNIX, sun_path="{service}"}}, 110) = 0',
            f'connect(4<socket:[2]>, {{sa_family=AF_UNIX, sun_path="{absent}"}}, 110)'
            " = -1 ENOENT (No such file or directory)",
            'connect(5<socket:[3]>, {sa_family=AF_UNIX, sun_path="/var/lib/sss/pipes/nss"}, 110)'
            " = 0",
            'connect(6<socket:[4]>, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110)'
            " = -1 ENOENT (No such file or directory)",
            'connect(7<socket:[5]>, {sa_family=AF_UNIX, sun_path="/run/user/1/agent"}, 110)'
            " = -1 EINPROGRESS (Operation now in progress)",
        ),
        "files",
        [],
    )
    assert found.connected == ("unix:/run/user/1/agent", f"unix:{service}")
    assert found.missing == (str(absent),)


@pytest.mark.parametrize(
    "line",
    [
        "something strace never writes",
        'openat(AT_FDCWD</work>, "/a"',  # cut off
        'openat(AT_FDCWD, "relative", O_RDONLY) = -1 ENOENT (No such file or directory)',
        'execve("./job", ["job"], 0x7ffc /* 1 var */) = 0',
        'openat(AT_FDCWD</work>, "/a) = b", O_RDONLY) = 3</a) = b>',
    ],
)
def test_a_line_that_cannot_be_read_leaves_the_footprint_incomplete(
    tmp_path: Path, line: str
) -> None:
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [])
    assert found.unread == 1 and not found.complete


def test_a_footprint_holds_until_what_it_found_is_otherwise(tmp_path: Path) -> None:
    calibration, listed = tmp_path / "calibration.txt", tmp_path / "plugins"
    calibration.write_text("1.05\n")
    listed.mkdir()
    absent = tmp_path / "rootmap"
    found = footprint(
        traces(tmp_path / "raw", opened(calibration), opened(listed), failed(absent)), "files", []
    )
    assert found.changes() == []

    calibration.write_text("1.10\n")
    assert found.changes() == [f"{calibration} changed"]
    calibration.write_text("1.05\n")
    assert found.changes() == []

    (listed / "new.so").touch()  # a name a scan of it would find
    assert found.changes() == [f"{listed} changed"]
    (listed / "new.so").unlink()

    absent.touch()
    assert found.changes() == [f"{absent} appeared"]
    absent.unlink()

    calibration.unlink()
    assert found.changes() == [f"{calibration} went"]


def test_a_link_retargeted_to_another_file_no_longer_holds(tmp_path: Path) -> None:
    # the old file stays as it was: only the link says what a new run would read
    old, new, link = tmp_path / "v1", tmp_path / "v2", tmp_path / "current"
    old.write_text("one")
    new.write_text("two")
    link.symlink_to(old)
    line = f'openat(AT_FDCWD</work>, "{link}", O_RDONLY) = 3<{old}>'
    found = footprint(traces(tmp_path / "raw", line), "files", [])
    assert {str(link), str(old)} <= found.read.keys()
    assert found.changes() == []
    link.unlink()
    link.symlink_to(new)
    assert found.changes() == [f"{link} changed"]


def test_a_directory_on_the_way_retargeted_no_longer_holds(tmp_path: Path) -> None:
    for name, text in (("v1", "one"), ("v2", "two")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "file").write_text(text)
    link = tmp_path / "current"
    link.symlink_to("v1")
    line = f'openat(AT_FDCWD</work>, "{link}/file", O_RDONLY) = 3<{tmp_path}/v1/file>'
    found = footprint(traces(tmp_path / "raw", line), "files", [])
    link.unlink()
    link.symlink_to("v2")
    assert found.changes() == [f"{link}/file changed"]


def test_a_path_through_a_link_and_up_is_kept_as_the_system_walks_it(tmp_path: Path) -> None:
    (tmp_path / "real" / "bin").mkdir(parents=True)
    (tmp_path / "real" / "lib").mkdir()
    target = tmp_path / "real" / "lib" / "file"
    target.write_text("x")
    (tmp_path / "bin").symlink_to("real/bin")
    named = f"{tmp_path}/bin/../lib/file"  # no such file, were the .. taken before the link
    line = f'openat(AT_FDCWD</work>, "{named}", O_RDONLY) = 3<{target}>'
    found = footprint(traces(tmp_path / "raw", line), "files", [])
    assert found.read[named] == found.read[str(target)] is not None


def test_a_link_into_the_kernel_s_files_is_no_input(tmp_path: Path) -> None:
    line = 'openat(AT_FDCWD</work>, "/etc/mtab", O_RDONLY) = 3</proc/1/mounts>'
    assert footprint(traces(tmp_path / "raw", line), "files", []).read == {}


def test_a_file_written_through_a_link_is_no_input_by_either_name(tmp_path: Path) -> None:
    made, link = tmp_path / "made", tmp_path / "link"
    made.touch()
    link.symlink_to(made)
    lines = (
        f'openat(AT_FDCWD</work>, "{link}", O_WRONLY|O_CREAT, 0600) = 3<{made}>',
        f'openat(AT_FDCWD</work>, "{link}", O_RDONLY) = 4<{made}>',
    )
    assert footprint(traces(tmp_path / "raw", *lines), "files", []).read == {}


def test_a_file_that_changed_while_the_run_ran_never_holds(tmp_path: Path) -> None:
    file = tmp_path / "input"
    file.write_text("read")
    began = time.time_ns()
    time.sleep(0.02)  # past a tick of the clock files are stamped with
    file.write_text("changed after it was read")
    found = footprint(traces(tmp_path / "raw", opened(file)), "files", [], began)
    assert found.read == {str(file): None}
    assert found.changes() == [f"{file} could not be identified"]


@pytest.mark.parametrize(
    "making",
    [
        'rename("{tmp}/x.tmp", "{path}") = 0',
        'renameat2(AT_FDCWD<{tmp}>, "x.tmp", AT_FDCWD<{tmp}>, "made", RENAME_NOREPLACE) = 0',
        'linkat(AT_FDCWD<{tmp}>, "x.tmp", AT_FDCWD<{tmp}>, "made", 0) = 0',
        'symlink("elsewhere", "{path}") = 0',
        'symlinkat("elsewhere", AT_FDCWD<{tmp}>, "made") = 0',
        'mkdir("{path}", 0755) = 0',
        'mkdirat(AT_FDCWD<{tmp}>, "made", 0777) = 0',
    ],
)
def test_a_path_looked_for_and_then_made_by_the_run_is_its_own(tmp_path: Path, making: str) -> None:
    made = tmp_path / "made"
    made.touch()
    line = making.format(tmp=tmp_path, path=made)
    found = footprint(traces(tmp_path / "raw", STARTED, failed(made), line), "files", [])
    assert found.missing == () and found.complete


def test_a_path_another_process_made_while_the_run_ran_stays_missing(tmp_path: Path) -> None:
    # looked for, not found, defaults taken; then made by someone else before the run ended
    override = tmp_path / "override.json"
    override.touch()
    found = footprint(traces(tmp_path / "raw", STARTED, failed(override)), "files", [])
    assert found.missing == (str(override),)
    assert found.changes() == [f"{override} appeared"]


def test_a_path_found_there_by_an_exclusive_create_is_an_input(tmp_path: Path) -> None:
    # a marker probed with O_CREAT|O_EXCL: it exists, so the cached path is taken
    marker = tmp_path / "initialized"
    marker.touch()
    line = (
        f'openat(AT_FDCWD<{tmp_path}>, "initialized", O_WRONLY|O_CREAT|O_EXCL, 0600)'
        " = -1 EEXIST (File exists)"
    )
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [])
    assert found.read[str(marker)] is not None and found.changes() == []
    marker.unlink()
    assert found.changes() == [f"{marker} went"]


def test_a_path_a_failed_call_would_have_made_stays_missing(tmp_path: Path) -> None:
    made = tmp_path / "made"
    line = f'mkdir("{made}", 0755) = -1 EROFS (Read-only file system)'
    found = footprint(traces(tmp_path / "raw", STARTED, failed(made), line), "files", [])
    assert found.missing == (str(made),)


def test_a_link_retargeted_while_the_run_ran_vouches_for_nothing(tmp_path: Path) -> None:
    # read through current -> v1; while it ran, current -> v2; both files older than the run
    old, new, link = tmp_path / "v1", tmp_path / "v2", tmp_path / "current"
    old.write_text("one")
    new.write_text("two")
    link.symlink_to(old)
    began = time.time_ns()
    time.sleep(0.02)
    link.unlink()
    link.symlink_to(new)
    line = f'openat(AT_FDCWD</work>, "{link}", O_RDONLY) = 3<{old}>'
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [], began)
    assert found.read[str(link)] is None and found.read[str(old)] is not None
    assert f"{link} could not be identified" in found.changes()


def test_a_link_retargeted_and_back_while_the_run_ran_vouches_for_nothing(tmp_path: Path) -> None:
    old, link = tmp_path / "v1", tmp_path / "current"
    old.write_text("one")
    link.symlink_to(old)
    began = time.time_ns()
    time.sleep(0.02)
    link.unlink()
    link.symlink_to(old)  # where it was, made anew
    line = f'openat(AT_FDCWD</work>, "{link}", O_RDONLY) = 3<{old}>'
    assert (
        footprint(traces(tmp_path / "raw", STARTED, line), "files", [], began).read[str(link)]
        is None
    )


def test_a_relative_open_of_what_the_stage_made_says_nothing_it_needs(tmp_path: Path) -> None:
    own = tmp_path / "stage"
    line = f'open("made.root", O_RDONLY) = 3<{own}/made.root>'
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [own])
    assert found.unread == 0 and found.complete


def test_a_relative_open_from_a_directory_the_trace_does_not_say_is_unread(tmp_path: Path) -> None:
    line = f'open("current.root", O_RDONLY) = 3<{tmp_path}/v1.root>'
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [])
    assert found.unread == 1 and not found.complete
    assert found.unreadable == (
        f"opened by a path relative to an unknown directory: {tmp_path}/v1.root",
    )


def test_python_s_own_cache_is_no_name_of_a_directory(tmp_path: Path) -> None:
    package = tmp_path / "helpers"
    package.mkdir()
    (package / "scale.py").touch()
    before = identity(str(package))
    (package / "__pycache__").mkdir()  # another stage imported it
    assert identity(str(package)) == before


def test_a_compiled_module_is_never_missing_whoever_makes_it(tmp_path: Path) -> None:
    cached = tmp_path / "__pycache__" / "helper.cpython-314.pyc"
    assert footprint(traces(tmp_path / "raw", failed(cached)), "files", []).missing == ()


def test_what_could_not_be_identified_never_holds(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    found = footprint(traces(tmp_path / "raw", opened(gone)), "files", [])
    assert found.read == {str(gone): None}
    assert found.changes() == [f"{gone} could not be identified"]


def test_a_refused_file_that_can_be_read_now_no_longer_holds(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.touch()
    assert Footprint("files", denied=(str(file),)).changes() == [f"{file} can be read now"]


def test_a_file_of_a_local_build_that_nothing_else_identifies_is_watched_as_any(
    tmp_path: Path,
) -> None:
    # the software identifies what can be loaded from a library directory: not a data file in it
    data = tmp_path / "install" / "lib" / "calibration.dat"
    data.parent.mkdir(parents=True)
    data.write_text("1\n")
    found = footprint(traces(tmp_path / "raw", opened(data)), "files", [])
    data.write_text("2\n")
    assert found.changes() == [f"{data} changed"]


def test_the_summary_says_what_was_used_and_what_nothing_else_identifies(tmp_path: Path) -> None:
    release = "/cvmfs/sft.cern.ch/lcg/releases"
    nightly = "/cvmfs/sft-nightlies.cern.ch/lcg/nightlies/devkey-head/Sat"
    platform = "x86_64-el9-gcc16-opt"
    local = Local("LD_LIBRARY_PATH", str(tmp_path / "install/lib"), "0" * 64, 1, 1)
    unread = Local("PYTHONPATH", str(tmp_path / "helpers"), "0" * 64, 1, 1)
    found = Footprint(
        "files",
        {
            f"{release}/Geant4/11.4.3-abcde/{platform}/lib/libG4run.so": "cvmfs:a",
            f"{nightly}/MCGenerators/sherpa/3.0.5/{platform}/lib/libSherpa.so": "cvmfs:b",
            f"{nightly}/k4fwcore/HEAD/{platform}/lib/libk4FWCore.rootmap": "cvmfs:c",
            f"{nightly}/marlin/HEAD/{platform}/lib/libMarlin.components": "cvmfs:d",
            "/cvmfs/geant4.cern.ch/share/data/G4EMLOW/file": "cvmfs:e",
            "/usr/lib64/libc.so.6": "sha256:f",
            "/usr/local/lib/libbyhand.so": "sha256:g",
            "/home/someone/.rootrc": "sha256:h",
            "/cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh": "cvmfs:j",
            "/cvmfs/experiment.cern.ch/data/calibration.root": "cvmfs:k",  # data like any other
            str(tmp_path / "install/lib/libDDMarlinPandora.so"): "sha256:i",
        },
        connected=("10.0.0.1:53",),
    )
    assert found.summary([local, unread]) == {
        "level": "files",
        "complete": True,
        "packages": {"Geant4": "11.4.3-abcde", "sherpa": "3.0.5"},  # no package by a registry
        "local": [local.path],
        "undeclared": {
            "/cvmfs/experiment.cern.ch/data/calibration.root": "cvmfs:k",
            "/home/someone/.rootrc": "sha256:h",
            "/usr/local/lib/libbyhand.so": "sha256:g",
        },
        "network": ["10.0.0.1:53"],
    }


def test_what_the_software_does_not_identify_is_undeclared_wherever_it_is(tmp_path: Path) -> None:
    library = tmp_path / "install" / "lib"
    library.mkdir(parents=True)
    for name in ("libX.so", "calibration.dat"):
        (library / name).touch()
    helpers = tmp_path / "helpers"
    (helpers / "package").mkdir(parents=True)
    for name in ("scale.py", "package/table.json"):
        (helpers / name).touch()
    local = [
        Local("LD_LIBRARY_PATH", str(library), "0" * 64, 2, 0),
        Local("PYTHONPATH", str(helpers), "0" * 64, 2, 0),
    ]
    paths = [
        str(library / "libX.so"),  # loaded through the entry: the software holds it
        str(library / "calibration.dat"),  # beside it: nothing does
        str(helpers / "scale.py"),
        str(helpers / "package" / "table.json"),
        "/opt/fcc/calibration.dat",  # no package manager's
        "/var/data/table.root",
        "/usr/share/zoneinfo/UTC",  # the system's
    ]
    found = Footprint("files", dict.fromkeys(paths, "sha256:a"))
    assert sorted(found.summary(local)["undeclared"]) == sorted(
        [paths[1], paths[3], paths[4], paths[5]]
    )
    assert covered(paths[0], local) and covered(paths[2], local)


def test_a_local_entry_reached_through_a_link_is_named_as_the_variable_names_it(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    entry = Local("PYTHONPATH", str(link), "0" * 64, 1, 1)
    found = Footprint("files", {str(real / "helper.py"): "sha256:a"})
    summary = found.summary([entry])
    assert summary["local"] == [str(link)] and summary["undeclared"] == {}


def test_a_local_entry_only_listed_was_not_used(tmp_path: Path) -> None:
    # Python lists every directory of its path, whatever it imports
    entry = Local("PYTHONPATH", str(tmp_path / "helpers"), "0" * 64, 1, 1)
    found = Footprint("files", {entry.path: "names:a"})
    assert found.summary([entry])["local"] == []


def test_the_libraries_level_says_what_was_loaded_and_no_more(tmp_path: Path) -> None:
    library = tmp_path / "libphysics.so"
    library.write_bytes(b"ELF")
    raw = traces(
        tmp_path / "raw",
        "   123:\tfile=libphysics.so [0];  generating link map",
        f"   123:\tcalling init: {library}",
        name="ld.123",
    )
    found = footprint(raw, "libraries", [])
    assert list(found.read) == [str(library)]
    assert not found.complete  # what else it read is not known
    assert found.summary([])["undeclared"] == {}


def test_observe_keeps_the_footprint_and_removes_the_traces(tmp_path: Path) -> None:
    file = tmp_path / "input"
    file.write_text("x")
    raw = traces(tmp_path / "raw", opened(file))
    summary, path = observe(raw, "files", directory=tmp_path, own=[])
    assert path == tmp_path / FOOTPRINT and not raw.exists()
    data = path.read_bytes()
    assert summary["footprint"] == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert load(path).read == {str(file): identity(str(file))}


def test_the_same_run_makes_the_same_footprint_file(tmp_path: Path) -> None:
    file = tmp_path / "input"
    file.touch()
    made = []
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        summary, _ = observe(
            traces(directory / "raw", opened(file)), "files", directory=directory, own=[]
        )
        made.append(summary["footprint"])
    assert made[0] == made[1]


def test_a_run_not_observed_says_so_and_keeps_nothing(tmp_path: Path) -> None:
    assert observe(None, "none", directory=tmp_path, own=[]) == (
        {"level": "none", "complete": False},
        None,
    )


def test_a_footprint_that_cannot_be_kept_says_why_and_the_stage_goes_on(tmp_path: Path) -> None:
    raw = traces(tmp_path / "raw", opened("/usr/bin/true"))
    summary, path = observe(raw, "files", directory=tmp_path / "absent", own=[])
    assert path is None and summary["complete"] is False
    assert "footprint not kept" in str(summary["problem"])


@pytest.mark.parametrize(
    "data",
    [
        b"not gzip",
        gzip.compress(b"not json"),
        gzip.compress(b"[]"),
        gzip.compress(json.dumps({"schema": "other"}).encode()),
    ],
)
def test_what_is_no_footprint_is_refused(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / FOOTPRINT
    path.write_bytes(data)
    with pytest.raises(SoftwareError):
        load(path)


def test_a_missing_footprint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SoftwareError, match="no footprint"):
        load(tmp_path / FOOTPRINT)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("level", "none"),
        ("read", []),
        ("read", {"/a": 1}),
        ("missing", "/a"),
        ("ran", [1]),
        ("unread", -1),
        ("unread", True),
    ],
)
def test_a_footprint_that_is_not_what_one_holds_is_refused(key: str, value: object) -> None:
    data = Footprint("files").to_dict() | {key: value}
    with pytest.raises(SoftwareError):
        Footprint.from_dict(data)


def test_a_footprint_comes_back_as_it_was_written() -> None:
    found = Footprint(
        "files", {"/a": "sha256:a", "/b": None}, ("/c",), ("/d",), ("/e",), ("f:1",), 2
    )
    assert Footprint.from_dict(found.to_dict()) == found


def test_a_pipe_never_vouches_for_what_was_read_from_it(tmp_path: Path) -> None:
    pipe = tmp_path / "calibration"
    os.mkfifo(pipe)
    assert identity(str(pipe)) == "special"
    found = footprint(traces(tmp_path / "raw", STARTED, opened(pipe)), "files", [])
    assert found.changes() == [f"{pipe} could not be identified"]


def test_a_local_socket_no_one_was_at_is_missing_wherever_it_is(tmp_path: Path) -> None:
    line = (
        'connect(3<socket:[1]>, {sa_family=AF_UNIX, sun_path="/run/my-calibration.sock"}, 110)'
        " = -1 ENOENT (No such file or directory)"
    )
    found = footprint(traces(tmp_path / "raw", STARTED, line), "files", [])
    assert found.missing == ("/run/my-calibration.sock",)


def test_a_relative_or_abstract_local_socket_is_never_taken_for_a_file(tmp_path: Path) -> None:
    relative = (
        'connect(3<socket:[1]>, {sa_family=AF_UNIX, sun_path="calibration.sock"}, 110)'
        " = -1 ENOENT (No such file or directory)"
    )
    found = footprint(traces(tmp_path / "raw", STARTED, relative), "files", [])
    assert found.missing == () and not found.complete  # where it looked is not known
    abstract = (
        'connect(4<socket:[2]>, {sa_family=AF_UNIX, sun_path=@"/tmp/.X11-unix/X0"}, 20)'
        " = -1 ECONNREFUSED (Connection refused)"
    )
    found = footprint(traces(tmp_path / "again", STARTED, abstract), "files", [])
    assert found.connected == ("unix:@/tmp/.X11-unix/X0",) and found.missing == ()


def test_what_a_run_made_outside_its_own_directory_is_kept(tmp_path: Path) -> None:
    own, cache = tmp_path / "stage", tmp_path / "cache" / "fonts"
    lines = (
        STARTED,
        opened(cache, "O_WRONLY|O_CREAT|O_TRUNC, 0666"),
        opened(own / "events.root", "O_WRONLY|O_CREAT|O_TRUNC, 0666"),
        f'mkdir("{tmp_path}/cache", 0755) = 0',
    )
    found = footprint(traces(tmp_path / "raw", *lines), "files", [own])
    assert found.made == (str(tmp_path / "cache"), str(cache))


def test_what_appeared_of_what_a_run_looked_for_is_what_a_resume_refuses(tmp_path: Path) -> None:
    override, cache = tmp_path / "override.json", tmp_path / "cache"
    found = Footprint(
        "files",
        missing=(str(override), str(cache), "/usr/share/absent", "/cvmfs/sft.cern.ch/absent"),
    )
    assert found.appeared([]) == []
    override.touch()
    cache.touch()
    assert found.appeared([], made={str(cache)}) == [str(override)]  # the study made the cache


def test_inputs_that_changed_are_named(tmp_path: Path) -> None:
    same, other = tmp_path / "same", tmp_path / "other"
    same.write_text("a")
    other.write_text("a")
    held = {str(same): identity(str(same)), str(other): identity(str(other))}
    other.write_text("b")
    assert changed_inputs(held) == [str(other)]


@strace
def test_a_real_run_is_observed_with_its_children(tmp_path: Path) -> None:
    file = tmp_path / "input"
    file.write_text("x")
    work = tmp_path / "work"
    work.mkdir()
    script = f"cat {file} > made; cat made; cat absent 2>/dev/null; exit 0"
    run = program.run("sh", ["-c", script], workdir=work, logs=LOGS, observe="files")
    assert run.error is None and run.observed == "files" and run.trace == work / program.TRACE
    found = footprint(run.trace, "files", [work])
    assert found.complete, found.unreadable  # the lines it could not read, if any
    assert identity(str(file)) == found.read[str(file)]
    assert str(work / "made") not in found.read  # its own
    assert any(path.endswith("/cat") for path in found.ran)  # a child's program
    assert run.stdout.read_text() == "x"  # the program's own output, as ever


@strace
def test_a_datagram_sent_without_a_connection_is_seen(tmp_path: Path) -> None:
    code = (
        "import socket\n"
        "socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'', ('127.0.0.1', 9))\n"
    )
    run = program.run("python3", ["-c", code], workdir=tmp_path, logs=LOGS, observe="files")
    assert run.trace is not None
    assert "AF_INET socket" in footprint(run.trace, "files", [tmp_path]).connected


@strace
def test_a_datagram_sent_to_a_local_socket_without_a_connection_is_seen(tmp_path: Path) -> None:
    import socket

    address = tmp_path / "s"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as service:
        service.bind(str(address))
        code = (
            "import socket\n"
            f"socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM).sendto(b'?', {str(address)!r})\n"
        )
        run = program.run("python3", ["-c", code], workdir=tmp_path, logs=LOGS, observe="files")
    assert run.error is None and run.trace is not None
    assert f"unix:{address}" in footprint(run.trace, "files", [tmp_path]).connected


@strace
def test_a_retargeted_link_is_seen_in_a_real_run(tmp_path: Path) -> None:
    (tmp_path / "v1").write_text("one")
    (tmp_path / "v2").write_text("two")
    link = tmp_path / "current"
    link.symlink_to("v1")
    work = tmp_path / "work"
    work.mkdir()
    run = program.run("cat", [str(link)], workdir=work, logs=LOGS, observe="files")
    assert run.trace is not None
    found = footprint(run.trace, "files", [work])
    link.unlink()
    link.symlink_to("v2")
    assert found.changes() == [f"{link} changed"]


@strace
def test_a_relative_file_a_process_failed_to_open_is_found_where_it_ran(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "job").mkdir(parents=True)
    run = program.run(
        "sh",
        ["-c", "cat absent; exit 0"],
        workdir=work,
        cwd=work / "job",
        logs=LOGS,
        observe="files",
    )
    assert run.trace is not None
    assert str(work / "job" / "absent") in footprint(run.trace, "files", []).missing


def test_traces_in_which_no_program_started_are_no_complete_footprint(tmp_path: Path) -> None:
    # what a tracer that died at once, or a full disk, leaves: nothing that says nothing was read
    raw = tmp_path / "raw"
    raw.mkdir()
    summary, _ = observe(raw, "files", directory=tmp_path, own=[])
    assert summary["complete"] is False
