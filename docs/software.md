# Software

`k4opt.software` says what candidates run on, so that two runs are compared by what they ran on
and not by where it was. A study records it, its stages are signed with it, and a study resumes only
on the same software.

`KEY4HEP_STACK` names a place. The default Key4hep nightly is an LCG view in a weekday slot that a
later build overwrites, so the same path holds other software from one week to the next. What
identifies a run is what is there:

| Part | What it is |
| --- | --- |
| build | the build of the stack, from what the stack says of itself |
| system | the operating system with its major version, the glibc version, the architecture |
| variables | the SHA-256 of every environment variable not known to be irrelevant, a local build on a search path by its content |
| site | what the interpreter's site directories hold (a virtual environment, the user's `pip install --user`), apart from the variables, so that no variable can stand for it |

Their digest is the software's `id`: two sessions with one `id` run on the same software. The build
is identified wherever the stack is, and so is a [local build](#local-builds): by what it holds, not
where. Other text is compared as it is: a variable that holds a path into the stack differs between
two prefixes of the stack, and `HOME` between two users. `latest` and its weekday are one, since the
view writes the paths it was built at. See what this shell runs on with

```text
$ k4opt software show
build:     LCG devkey-head, built Sat Nov 7 04:17:32 2026
platform:  x86_64-el9-gcc16-opt
system:    rhel 9, glibc 2.34, x86_64
variables: 181 compared
id:        58e34b81d98af6bbf117d9b8b8ff931267aa58ce87b95cfec432a9e5c1fd01c3
```

Nothing in a problem file says which software to use: a problem describes the physics, and the
shell it runs in says the rest.

## The build

| Stack | Recognized by | Identified by |
| --- | --- | --- |
| LCG view | `…/views/<release>/<slot>/<platform>/setup.sh` with a `# Generated:` line | that time, and the SHA-256 of `setup.sh` and of `LCG_externals_<platform>.txt` and `LCG_contrib_<platform>.txt` in `…/nightlies/<release>/<slot>/` |
| anything else | | *unrecognized*: the SHA-256 of the file `KEY4HEP_STACK` names, or of that text if there is no such file |

Links are followed first, so `latest` and the weekday it points to are one build. No location enters
a build. A manifest that cannot be read counts as `null`, so a build that loses one is no longer the
same build.

A session reads the build again before each candidate: a slot republished while a study runs stops
it, naming the build it found. k4Opt identifies a build and never keeps a copy of it, so a nightly
that was overwritten cannot be run again.

## The variables

A value leaves no trace in any file, and a program inherits every variable, so one rule decides:
**a variable is compared unless it is explicitly known to be irrelevant**, whether k4Opt has heard
of it or not. `LD_LIBRARY_PATH`, `PYTHONPATH`, `MARLIN_DLL`, `GAUDI_PLUGIN_PATH`, `LD_PRELOAD`,
`G4LEDATA`, `LC_ALL` and a user's own `MY_SCALE` are all compared. Values are kept as digests, so a
difference names a variable and cannot show its old value.

These are not compared (`IGNORED` in `k4opt/software/identity.py`; `*` stands for any ending):

| Why | Names |
| --- | --- |
| The state of a shell or terminal: nothing a job reads but for the width of what it prints | `_`, `PWD`, `OLDPWD`, `SHLVL`, `SHELL`, `TERM`, `COLUMNS`, `LINES`, `DISPLAY`, `MOTD_SHOWN` |
| The login session and its credentials, which differ between two logins of one person | `USER`, `HOSTNAME`, `TMPDIR`, `KRB5CCNAME`, `SSH_*`, `XDG_RUNTIME_DIR`, `XDG_SESSION_*`, `XDG_SEAT`, `XDG_VTNR`, `DBUS_*`, `SELINUX_*` |
| New on every run of a service: without these, no study run by a systemd unit could resume | `INVOCATION_ID`, `JOURNAL_STREAM`, `MANAGERPID`, `SYSTEMD_EXEC_PID` |
| Bookkeeping of other tools, or what only a person at a terminal sees | `MODULEPATH`, `CONDA_SHLVL`, `CONDA_BACKUP_*`, `VSCODE_*`, `GIT_PAGER`, `GIT_EDITOR`, `GIT_ASKPASS`, `PYTEST_*` |
| Set by ROOT's Python in whatever process imports it, from the build: an evaluator that reads a file through `podio` sets them in the middle of a session | `CPPYY_API_PATH`, `CPPYY_NO_ROOT_FILTER` |
| Identified by what they lead to: the geometry by its fingerprint, the stack by its build | `K4GEO`, `KEY4HEP_STACK` |

`PATH`, `BASH_FUNC_*`, `HOME`, `LANG`, `LC_*`, `TZ`, `CONDA_PREFIX`, `XDG_CONFIG_HOME` and `GIT_DIR`
stay compared: ROOT runs `g++` through `PATH` and an evaluator may run any program, an exported
shell function is code every `bash` it starts can run, ROOT reads the home
directory, cling was seen consulting a conda environment, and matplotlib reads its settings from
`XDG_CONFIG_HOME`. Within a session `K4GEO` and `KEY4HEP_STACK` must keep
their values too, since the geometry was found through them.

A wrongly compared variable costs a refusal that names it; a wrongly ignored one could cost
correctness. So the list stays short, and a name joins it only once it is shown to be irrelevant.

## Local builds

A local build (`k4_local_repo`, a helper directory, a virtual environment) is put in front of the
stack through the variables that tell programs where to load from. Their text says where it is, not
what it is: a library rebuilt in place keeps the text, and the same build elsewhere changes it. So
each entry of such a variable that is a local file or directory is identified by the SHA-256 of what
can be loaded through it, as the variable uses it. Entries under `/cvmfs`, `/usr`, `/lib`, `/lib64`,
`/bin`, `/sbin` and `/etc` stay as text, which the build and the system identify; `/usr/local`, where
software is installed by hand or by `pip`, is local.

A relative or empty entry (`.`, or the empty one in `a::b`) is the directory k4opt runs in, which
its own process and its workers import from, as Python resolves it when it starts; so is a file
named with a `/` (`LD_PRELOAD=./libpatch.so`). A bare library name (`LD_PRELOAD=libpatch.so`) is
looked up on the library path, whose local directories are identified already. A program a stage
runs starts in the stage's own directory, which k4opt makes. One exception: an empty entry
of `ROOT_INCLUDE_PATH` (the one `k4_local_repo` leaves at its end) stays as text. ROOT skips it: a
header in the working directory is found by `#include "..."` with the variable unset, empty or
`.`, and by `#include <...>` only with `.` (verified), so the empty entry adds nothing, and
identifying every file below the working directory would take in the study being written there,
which then could never run. `.` is identified as any other entry.

| Variable | What can be loaded through a local entry |
| --- | --- |
| `LD_LIBRARY_PATH`, `ROOT_LIBRARY_PATH`, `GAUDI_PLUGIN_PATH` | the libraries directly in the directory (`*.so`, `*.so.*`), which the loader looks up by name, and the registries and dictionaries ROOT, DD4hep and Gaudi read (`*.rootmap`, `*.components`, `*.pcm`); not build files, nor a log written there |
| `ROOT_INCLUDE_PATH` | every file below the directory, which an `#include` may name relative to it, with any extension or none |
| `PYTHONPATH`, and the interpreter's site directories with what their `.pth` files add | what Python can import: modules (`.py`, `.pyc`, `.so`) and `.pth` files, every file of a regular package (one with an `__init__`), the modules of a namespace package (a directory whose name is an identifier), package metadata (`*.dist-info`, `*.egg-info`) and the native libraries a wheel bundles (`*.libs`, which its extension modules load); never `__pycache__` |
| `MARLIN_DLL`, `LD_PRELOAD` | each file named |

So a library rebuilt in place, a module edited in a helper directory, a package installed with
`pip install --user` and an edited editable install are other software, and the same build at
another prefix is the same software. A refusal names what changed: *the files LD_LIBRARY_PATH leads
to in /home/me/install/lib changed*. The record keeps each local entry with where it is, its digest,
its number of files and bytes; a snapshot keeps none of it.

Each file is read once per version, as the files a problem names are (its size, times and inode
tell a rewritten file apart); before each candidate a session only looks at those. A worker takes
what its session read, and after each task looks at the variables alone, since a task changes
variables, not installed files. Links are followed, as a lookup through them is; a link back up
the tree is not. An entry of more than 50,000 files or 4 GiB is identified by its very files instead of
their content, which would be read on every session: each file's size, times and inode, so that any
write to it, or another file in its place, is other software, and a copy elsewhere is too; the
record says which it is (`content`).

A local build moved together with an environment that names its prefix in other variables
(`VIRTUAL_ENV`, `CONDA_PREFIX`, `PATH`) is still other software through those: they are compared as
text.

What lies outside: `PATH` is compared as text, since the directories on it hold tools that update
themselves (a program k4opt runs is identified by its content, and by its interpreter); a file a
module reads that Python does not import (data next to a module of a namespace package), a source
directory an editable install maps without a `.pth` path, and a value that names a file in any
other variable (`MY_SCALE_FILE`) are identified by their text. The files each stage really opened
are what [a stage's footprint](#what-each-stage-used) says.

## What each stage used

The identity says what a stage could reach; its *footprint* says what it did reach. While a program
of the stack runs, k4opt watches it, at the level `--observe` names (`k4opt evaluate`, `k4opt
optimize`; `observe=` in Python):

| Level | How | Says | Another study may reuse its output |
| --- | --- | --- | --- |
| `files` (default) | `strace`, limited to opens, program starts, connections, sockets, datagrams sent and the calls that make a path | every file read, listed or looked for in vain, by the program and all it started | yes |
| `libraries` | the loader (`LD_DEBUG`) | the libraries loaded | no |
| `none` | nothing | nothing | no |

A session tries `strace` on `true` when it opens. Where it is missing or may not trace (some
containers and batch nodes), the session takes the next level that works and warns once. The level
is no part of any identity and changes no output: a program's exit status, signals and peak memory
are its own. On a CLD job `strace` costs 3 to 4% of the wall time, mostly at start-up, and reading
the traces about 0.2 s for a reconstruction and 0.4 s for a simulation. One real cost: under
`strace`, ROOT cannot print the backtrace of a crash, so a stage that died by a signal says to
observe it at the `libraries` level to see one.

The raw traces are removed once they are read. What stays is `footprint.json.gz` in the stage's
directory (65 to 200 kB, private, with every path the stage touched, and never published), and in
the stage's record, under `observed`:

| Key | What it is |
| --- | --- |
| `level`, `complete` | how it was observed, and whether its traces could be read to the end: every file it opened is in the footprint, which is not every dependency it has (see below) |
| `packages` | the LCG packages it read a file of, by name and version; not one it only listed, or whose `.rootmap` or `.components` alone it read, as every package on the library path is |
| `local` | the [local entries](#local-builds) it read a file from |
| `undeclared` | the files and directories it read that neither the stack (on CVMFS: the LCG, Key4hep and Geant4 repositories), the system's package manager (`/usr`, `/lib*`, `/bin`, `/sbin`, `/etc`), the software nor the problem identify, each with its identity: a `~/.rootrc`, a calibration a helper reads through a variable, a file under `/opt`, an experiment's data on another CVMFS repository, a data file beside the libraries of a local build |
| `network` | what it reached outside this machine's own services: the addresses it tried to connect to, a network socket it made (`AF_INET socket`: a datagram needs no connection), any attempt at a local socket, by a connection or a datagram, other than the user database's, the journal's or the system bus's (`unix:<path>`, whose answers may come from anywhere). CLD's jobs have none |
| `footprint` | the size and SHA-256 of the footprint file |

`k4opt study trial` prints this for each stage, with the local entries no stage read a file from: a
local build that did not run, as a Marlin processor built with `k4_local_repo` on an LCG view, where
`MARLIN_DLL` still names the nightly's library. That is known only if every stage ran its program
itself, observed at the `files` level.

A footprint identifies what the stage read as it was when the stage ended: a file on CVMFS by the
content key CVMFS keeps of it, without reading it, any other by its SHA-256, a directory by the names
in it (but `__pycache__`, which Python makes as it imports). A file reached through a link is kept
under both names, so a link retargeted to another file is seen though the old file stays; one whose
link changed while the stage ran has no identity. It leaves out what the stage made, what it was
given, what the problem names, what it wrote or made (a file opened write-only, emptied or made anew,
a compiler's scratch file, a file renamed into place; a file opened read-write is read, and one the
stage changed has no identity), Python's `__pycache__`, whose files Python checks against their source, and `/proc`, `/sys`,
`/dev`, `/run` and the user database's caches. A path looked for in vain stays so though it exists
when the stage ends, unless the stage made it: another process may have, while it ran. It *holds* while
each file is the same, each directory holds the same names and no path looked for in vain exists. A program started through a link that changed while the stage ran has no identity. An
empty trace, a line that cannot be read, a file that could not be identified or that changed while
the stage ran, a pipe or a socket read as a file (it gives what it is sent), a relative path opened
from a directory the trace does not say, or a program started
by a relative path never vouch for anything: what is not known is taken to differ.

**What it cannot show.** A decision from a file's size or time, or from whether it exists, without
opening it; an import that failed in Python, which answers from a listing it keeps; a file the stage
both read and wrote, which counts as its own; whatever an evaluator reads, in k4opt's own process.
So a complete footprint is a complete list of what was opened, not proof of every dependency. This
is why [another study](execution.md#stages-made-once-reuse) reuses an output only on identical
software: the footprint then guards what lies outside the identity, and is never the proof that a
change in the software did not matter. Rebuilding a reconstruction package therefore simulates again
across studies, which costs time and never correctness.

## What a rerun needs

A study keeps privately, in its `software/` directory, what a rerun would need that is not public,
and `k4opt study show` says what that is and whether it is there (the study at the top of a nested
optimization keeps it for every level). Nothing of it leaves the machine.

| Kept | When | What |
| --- | --- | --- |
| the release | a session opens | the LCG view's manifests (34 kB gzipped), which name each package's version, build hash and place, and for each package built from a repository's head the commit it was built from and lcgcmake's revision (`.buildinfo_<package>.txt`), which they do not. No file of the build |
| local builds | a session opens | each local entry of the software grouped into its *origin*: a CMake installation (`<checkout>/install` from `k4_local_repo`), a directory of local code, or an environment of installed distributions. A build is linked to its checkout and the package of the stack it stands in for, its binaries identified by SHA-256 and GNU build-id, and its CMake cache kept (its options, compiler and flags). What else a build installed that can be loaded (Python modules, headers, registries) is kept file by file: it may differ from what its source installs. A binary is never kept, whether of a build or of a directory of no build |
| sources | a session opens | the complete state of each checkout (`HEAD`, staged, unstaged and untracked files; ignored files, its installation and its build trees left out, but for what can be loaded from a local entry, kept file by file or, if compiled, identified), as a patch against the revision the nightly built that package from, the newest commit a remote has, or, if no remote has any of its history, as every file. Untracked files over 10 MB are listed, not kept. Local code in no checkout is kept file by file |
| steering | a session opens | each reconstruction's job directory against the package of the stack it is a copy of (by its name, `CLDConfig`): its changed and added files are kept, with whether each is executable and where a link led, the removed ones named |
| geometry | a session opens; a trial's record is accepted | every file reachable from the compact file, with the other names of those files, unless the stack holds it, and what its external references read; then what a candidate's `ref` parameters reach beyond those, as inputs |
| evaluators | a session opens | each evaluator's file and every local module it imports, as its identity lists them; one that cannot be found is recorded as not kept |
| inputs | a trial's record is accepted | each file the stages read that the stack, the system and the software do not identify (from the problem or the footprints), up to 100 MB (a compiled file is identified, never kept), with the variable that names it, the trials that read it and a salted commitment `sha256(salt ‖ sha256(file))` that a publication can show without letting anyone test a guess. A path read as another file is another entry, never merged with the first |
| what ran | a trial's record is accepted | which files of local entries each observed stage read, where, and which kept state of the software each trial ran with |

All of it within 100 MB per study (a study without large inputs keeps a few hundred kB: the manifests,
the patches of its checkouts, a few small files), content-addressed and gzipped, written whole and
checked on every read: bytes altered on disk are reported, never used, and so are records that no
longer match what they name (a `software-*.json` is named by the SHA-256 of its content; a kept file
must be the bytes its record says it was read as). What cannot be kept never stops a study: it is said
once, recorded, and `study show` lists it. Nothing is written into a checkout: its state is a git tree
made with a temporary index and object store, reading the checkout's objects through git's
alternates. A remote is recorded without a password, and over HTTP(S) without any user part (a token
is often written there); an SSH user is kept.

**Only its owner can read it.** `software/` and every directory below it are `0700` (made so again
each time a session opens), and every file is `0600` from the moment it exists. On AFS the
directory's access list decides instead (`fs listacl`): a study there says so when it starts.

**Source and binary are two facts.** A patch does not rebuild the same binary: the same source built
in another directory has another build-id and other bytes. So a local build that ran is *rebuildable
from source* when its state is kept whole against a known revision (or whole, if no remote has its
history), its CMake configuration is kept, and its link to the binary is `verified` (a CMake cache
installs into its prefix, and the build tree holds every installed binary, with its build-id) or
`likely` (the cache does, but the binaries cannot be compared or differ); `adjacent` (in a checkout,
no build tree says so) and `none` are not enough. Nor is a source whose files changed after its newest
binary was made, or that differs between the kept states the study's trials ran with, or that lacks
untracked files too large to keep: what is kept is then not known to be what ran. Its original binary
is a *fingerprint only*: identified, never available. A local binary of no build and no checkout is
*binary only — source unknown*. A local build no stage read from (all of them observed completely) is
*changed locally — not used*: on LCG views `k4_local_repo` leaves `MARLIN_DLL` on the nightly's
library, so a rebuilt Marlin processor does not run unless it is set by hand. Which build ran is told
by where its files were read, not only by their bytes: two installations may hold the same library.

```text
$ k4opt study show runs/cld-ecal-study
...
rerun:      Not rerun · rebuild needed · availability not established (checked 2026-11-08T09:12:44+00:00)
needs:      base        Key4hep LCG devkey-head, built Sat Nov 7 04:17:32 2026: x86_64-el9-gcc16-opt, 989 packages, 66 built from a repository's head, with their revisions; available; used by pi0/reconstruction, pi0/simulation
                        a nightly: a later build overwrites its slot, and this one is then gone
            package     ddmarlinpandora: 821384f + 1 uncommitted files + 1 untracked (verified link); rebuildable from source; original binary: fingerprint only; available; used by pi0/reconstruction
            environment /home/me/k4Opt/py-venv/lib/python3.14/site-packages: 60 distributions; not checked: installed distributions, not checked against an index
            steering    pi0/reconstruction: a copy of cldconfig 8879e4e: 1 changed, 0 added, 0 removed; available; used by pi0/reconstruction
            input       ecal_scale.json: observed input, 16 bytes, read through MY_ECAL_SCALE_FILE; available; used by pi0/reconstruction
            ...
```

What a study used is read from the records of all its trials, at every level of a nested
optimization; a stage whose output was reused is what the stage that made it used, as that stage's
record says, in this study or another. The status never says a study was reproduced: that takes a
rerun, which k4opt does not record yet.

| Status | When |
| --- | --- |
| Not rerun · blocked | something a rerun needs is gone or altered (a nightly overwritten, a kept patch changed on disk, an input not kept that changed, a record of what was kept that no longer matches or cannot be read, a kept state a trial ran with), or a local build that ran cannot be built again (no kept source, a binary of no known build). A local build every stage was observed not to read blocks nothing |
| Not rerun · record incomplete | what it needs is not fully known: a stage that ran its program was not observed completely, or its record says nothing of what it used, or a reused output's record cannot be read; a stage used the network, whose answers are not kept, or read a file that could not be identified; what a trial ran with or read was not noted (`used.json`, `inputs.json`); a stage reused an output of another study made with local code this study did not keep; no trial has a record yet; the study kept nothing; a path was read as different files by its trials; or a local build that ran has a kept source not known to be what it was built from (see above) |
| Not rerun · availability not established | something could not be checked: the distributions of an environment, an unrecognized stack, a k4opt commit no remote was seen to have, a checkout's base commit no known remote can give, an input that was a device, a pipe or a socket, a directory a stage listed whose names changed |
| Not rerun · rebuild needed | a local build ran: it can be built again from its kept source and configuration, and a rerun runs other bytes, compared by its results |
| Not rerun · all available on DATE | every requirement was checked and found as it was, and no local build ran |

*Rebuild needed* and *availability not established* are said together when both hold. Each check is
dated, and true when it was made: a nightly is overwritten by a later build within about a week,
after which its studies cannot be rerun. `k4opt software show` lists the origins this shell would
record.

## What depends on it

| Where | What happens |
| --- | --- |
| A child process | Workers and the programs of the stack get exactly `os.environ`, never what a library set beneath Python: ROOT's interpreter sets `CLING_PREBUILT_MODULE_PATH` with C's `setenv` once it is used, and a child would otherwise inherit what no identity holds. |
| A session | The software is taken when the session opens. Before each candidate, and before its record is written, any difference stops the session: every program inherits this process's variables. An evaluator that sets a variable while it runs inline does the same, unless the variable is on the list above. So does an undeclared input once a record names it, whether its stage ran or was reused: every candidate of a session reads it as it was, and two stages of one candidate that read it as different files refuse its record. What the stack and the system's package manager hold is not watched within a session; it is checked when an output is reused from an earlier one. A path looked for in vain is checked when an output is reused, not during a session, since a cache another stage makes would look like one that appeared and stop a study. |
| A worker | It runs with the session's `sys.path`, compares the build, the system and each variable with the session's when it starts, and refuses by name; and compares the variables again after every task. A task that set a variable there (an evaluator writing `os.environ`) fails its candidate with a `ProvenanceError`, and the worker ends: the next task runs in a new one, started from the session's environment. |
| A stage's signature | It holds the `id`: a stage of another build, of a rebuilt local library, or run with another variable, is not reused, whatever its path. A reconstruction's `directory` is signed by its files, not by its text: the job runs from a copy, so the same job elsewhere makes the same signature. Any other path (an options file run where it is, `data`, a file a ddsim option names) is given to the program as written, which may read it, so its text is signed too. |
| A study | The `id`, the build and the system are part of its identity. A resume on other software is refused, naming what differs, and each trial's record must have run on the study's software. So is a resume after an undeclared input its trials read changed, or after a path one of them looked for in vain (outside the stack and the software) appeared, unless a stage of the study made it. |
| A record | `environment.observed.software` holds all of it, the variables as one digest, since every record of a session holds the same: `study.json` keeps them by name, for a refused resume to name them. A record is refused if its `id` is not that of its parts. |
| A snapshot | It holds the `id`, the build and the system, never the variables, whose names can be private, nor `KEY4HEP_STACK`, a location the build identifies, nor anything of a footprint. A catalog and the dashboard name the build that ran. |

**Start a study and resume it the same way.** There is no override and no warning that lets a resume
through: one study holds one software. From a login shell to a bare `systemd-run` unit about 60
variables differ, conda and modules among them, and the refusal names them. Two runs of the same
kind leave nothing to compare. Long studies are best started from a plain shell or a unit, not from
a tool whose shell carries variables of its own session, which could never be set again.

## From Python

```python
from k4opt.software import Software

opened = Software.now()
print(opened.build.label, opened.system.label, opened.id)
print(opened.differences(Software.now()))  # what differs, for a person; [] if nothing does

record = opened.to_dict()  # {"id", "build", "system", "variables", "local", "site"}
assert Software.from_dict(record) == opened  # SoftwareError if it is not what its id says
opened.to_dict(variables=False)  # as a record of a candidate holds it: one digest of the variables
```

`Build.of(path)` reads the build `KEY4HEP_STACK` would name, and `System.now()` this machine.
