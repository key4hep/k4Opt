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
| variables | the SHA-256 of every environment variable not known to be irrelevant |

Their digest is the software's `id`: two sessions with one `id` run on the same software. The build
is identified wherever the stack is; the variables are compared as they are, so a variable that
holds the stack's path (`LD_LIBRARY_PATH`) makes the same build at another prefix another `id`.
`latest` and its weekday are one, since the view writes the paths it was built at. See what this
shell runs on with

```text
$ k4opt software show
build:     LCG devkey-head, built Sat Nov 7 04:17:32 2026
platform:  x86_64-el9-gcc16-opt
system:    rhel 9, glibc 2.34, x86_64
variables: 181 compared
id:        58e34b81d98af6bbf117d9b8b8ff931267aa58ce87b95cfec432a9e5c1fd01c3
```

Nothing in a problem file says which software to use: a problem describes the physics, and the
shell it runs in says the rest. Local builds are for now seen only through the variables that point
at them, not by their content.

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
| The state of a shell or terminal: nothing a job reads but for the width of what it prints | `_`, `PWD`, `OLDPWD`, `SHLVL`, `SHELL`, `TERM`, `COLUMNS`, `LINES`, `DISPLAY`, `MOTD_SHOWN`, `BASH_FUNC_*` |
| The login session and its credentials, which differ between two logins of one person | `USER`, `HOSTNAME`, `TMPDIR`, `KRB5CCNAME`, `SSH_*`, `XDG_RUNTIME_DIR`, `XDG_SESSION_*`, `XDG_SEAT`, `XDG_VTNR`, `DBUS_*`, `SELINUX_*` |
| New on every run of a service: without these, no study run by a systemd unit could resume | `INVOCATION_ID`, `JOURNAL_STREAM`, `MANAGERPID`, `SYSTEMD_EXEC_PID` |
| Bookkeeping of other tools, or what only a person at a terminal sees | `MODULEPATH`, `CONDA_SHLVL`, `CONDA_BACKUP_*`, `VSCODE_*`, `GIT_PAGER`, `GIT_EDITOR`, `GIT_ASKPASS`, `PYTEST_*` |
| Set by ROOT's Python in whatever process imports it, from the build: an evaluator that reads a file through `podio` sets them in the middle of a session | `CPPYY_API_PATH`, `CPPYY_NO_ROOT_FILTER` |
| Identified by what they lead to: the geometry by its fingerprint, the stack by its build | `K4GEO`, `KEY4HEP_STACK` |

`PATH`, `HOME`, `LANG`, `LC_*`, `TZ`, `CONDA_PREFIX`, `XDG_CONFIG_HOME` and `GIT_DIR` stay
compared: ROOT runs `g++` through `PATH` and an evaluator may run any program, ROOT reads the home
directory, cling was seen consulting a conda environment, and matplotlib reads its settings from
`XDG_CONFIG_HOME`. Within a session `K4GEO` and `KEY4HEP_STACK` must keep
their values too, since the geometry was found through them.

A wrongly compared variable costs a refusal that names it; a wrongly ignored one could cost
correctness. So the list stays short, and a name joins it only once it is shown to be irrelevant.

## What depends on it

| Where | What happens |
| --- | --- |
| A child process | Workers and the programs of the stack get exactly `os.environ`, never what a library set beneath Python: ROOT's interpreter sets `CLING_PREBUILT_MODULE_PATH` with C's `setenv` once it is used, and a child would otherwise inherit what no identity holds. |
| A session | The software is taken when the session opens. Before each candidate, and before its record is written, any difference stops the session: every program inherits this process's variables. An evaluator that sets a variable while it runs inline does the same, unless the variable is on the list above. |
| A worker | It compares the build, the system and each variable with the session's when it starts, and refuses by name; and again after every task. A task that set a variable there (an evaluator writing `os.environ`) fails its candidate with a `ProvenanceError`, and the worker ends: the next task runs in a new one, started from the session's environment. |
| A stage's signature | It holds the `id`: a stage of another build, or run with another variable, is not reused, whatever its path. |
| A study | The `id`, the build and the system are part of its identity. A resume on other software is refused, naming what differs, and each trial's record must have run on the study's software. |
| A record | `environment.observed.software` holds all of it, and is refused if its `id` is not that of its parts. |
| A snapshot | It holds the `id`, the build and the system, never the variables, whose names can be private. A catalog and the dashboard name the build that ran. |

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

record = opened.to_dict()  # {"id", "build", "system", "variables"}
assert Software.from_dict(record) == opened  # SoftwareError if it is not what its id says
```

`Build.of(path)` reads the build `KEY4HEP_STACK` would name, and `System.now()` this machine.
