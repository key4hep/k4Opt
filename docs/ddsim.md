# ddsim

`k4opt.ddsim` runs Key4hep's `ddsim` once on a prepared geometry with a particle gun or an input
event file: it builds the command from the stages of a sample of the problem (the gun of its
`generation`, or its `file`, and `events`, `seed` and the ddsim options of its `simulation`), runs
the `ddsim` of the active Key4hep environment and keeps the command, exit code and logs. It does
not read the output; an evaluator does (see [evaluators.md](evaluators.md)). `k4opt evaluate`
runs it as the `simulation` [stage](evaluation.md#stages) of each sample once per candidate, so
evaluators never call it.
Checked against the `ddsim` of the nightly 2026-10-20 (its `--help` and its sources).

```python
from k4opt.ddsim import simulate

run = simulate(
    compact=compact,
    workdir=new_dir,
    events=100,
    seed=42,
    particle_gun=gun,  # the keys of a sample's generation; or input_file=Path("events.hepmc")
    options={"physics.list": "FTFP_BERT"},
    timeout=3600,  # seconds; no limit without it
)
print(run.error or run.output)
```

`simulate` raises `DdsimError` for unsupported settings, options it cannot pass on faithfully,
a `timeout` that is not a positive number of seconds or an existing `workdir`, before anything
is run. A `ddsim` that cannot run or fails is reported in
`SimulationRun.error` (not found, not started, nonzero exit with the tail of its standard
error, no output file, killed when it ran longer than `timeout` seconds) with `returncode`
(`None` when it was killed), `argv` and the files. `ddsim` runs in a process group of its own,
which is killed as a whole when its time runs out or k4opt is interrupted, so the processes a
wrapper script started (one without `exec`, a container) do not outlive it. `started`, a
function, is told that group's id as soon as ddsim started; `die_with_parent=True` has the kernel
kill ddsim when the thread that started it ends, even by SIGKILL (Linux). ddsim is then started
by a small launcher that asks for that signal and becomes ddsim (`exec`), so nothing runs
between fork and exec in a process whose libraries (ROOT) may have started threads; if the
system refuses the signal, the run fails and says so instead of starting ddsim unprotected.
k4opt's worker processes pass both, so that ddsim ends with a worker the coordinator stops or
the kernel's OOM killer ends.
`check_arguments` does the first
part alone, `check_particle_gun` and `check_options` check a gun or options on their own (the
problem does when it is loaded). `workdir` receives `events.edm4hep.root`, `ddsim.stdout.log`,
`ddsim.stderr.log` and `ddsim.json` (argument list, directory, exit code, time limit).

An unwritable simulation directory or command record is also reported in `SimulationRun.error`.
If the process already ran, its actual exit code and existing output and logs remain available.

## Particle gun (`generation`, `executor: gun`)

The keys of a generation stage run by the `gun` describe the particle gun. Unknown keys are
errors. The gun is fired inside `ddsim`, so it has no `events` or `seed` of its own: those of
the sample's `simulation` apply.

| Key | Meaning |
| --- | --- |
| `particle` | `--gun.particle` |
| `energy` | `value` and `unit` (`MeV`, `GeV`, `TeV`): the *total* energy, `--gun.energy` |
| `vertex` | optional `value` `[x, y, z]` and `unit` (`mm`, `cm`, `m`); default origin, always passed explicitly |
| `direction` | `distribution` (`cos(theta)` or `uniform`, passed to `--gun.distribution`), `theta` and `phi` as `[low, high]` and `unit` (`deg`, `rad`); theta within 0 to a half turn, phi at most a full turn |

One primary per event (`--gun.multiplicity=1`). `cos(theta)` is flat in cos(theta) (isotropic
in solid angle) and phi, whereas `uniform` is flat in theta. Checked on the nightly: of 4000
events with theta in 10-170 degrees, 50.2 % fell in |cos(theta)| < 0.5 (flat in cos: 50.8 %)
against 37 % for `uniform`. Another distribution is an error, never a fallback.

## Event file (`generation`, `file:`)

Instead of a gun, a generation stage given as a file simulates its events:

```yaml
samples:
  zh:
    generation:
      file: events/zh.hepmc
    simulation:
      executor: ddsim
      events: 500
```

A relative path is read from the directory of the problem file. `ddsim` receives it as
`--inputFiles`, chooses the reader from the file extension and validates the content, so k4opt
accepts every format `ddsim` does. The file must exist before anything runs; its path, size and
SHA-256 are recorded in the provenance, so a changed file is a different measurement.
The sample's `simulation.events` is the number of events to simulate; a file with fewer events gives
an incomplete simulation, which the simulation stage reports as `failed`.

## Options (`simulation`)

Every key of the simulation stage other than `executor`, `events` and `seed` is a ddsim option,
named as `ddsim --help` names it without the leading `--`, and is passed on in the order of the
problem. ddsim's command line reads options in four ways, and k4opt writes each value the way
ddsim reads it:

| ddsim option | YAML | Command line |
| --- | --- | --- |
| a flag: an option of a group named `enable*` or `force*` (`geometry.enableDebugVolumes`), and `enableDetailedShowerMode`, `disableSignalHandler` | `true` | `--geometry.enableDebugVolumes` |
| a flag | `false` | error: ddsim cannot switch a flag off on its command line, so leave it out |
| most options | a text, a number or a boolean | `--physics.list=FTFP_BERT`, `--physics.rangecut=0.7`, `--part.keepAllParticles=True` (ddsim reads `True`/`False`) |
| options that repeat: `action.run`, `action.event`, `action.track`, `action.step`, `action.stack` | a list | `--action.run=A --action.run=B` |
| options with several values: `meta.eventParameters`, `meta.runParameters` | a list | `--meta.eventParameters a/F=1 b/I=2` |

A list for any other option, a nested mapping (write `physics.list: FTFP_BERT`, not
`physics: {list: FTFP_BERT}`), an empty value or a prefix of a known option's name is an error,
never a guess. An integration test (`tests/integration/ddsim`) checks with
`ddsim --dumpParameter` that ddsim parses every kind as intended. Option names and values are
otherwise validated by ddsim when it starts, seconds before the first event.

Metadata values (`meta.eventParameters`, `meta.runParameters`) must not start with `-`, since
ddsim would interpret such a value as another option. This rule also protects the options
managed by k4opt when they appear inside a metadata list.

What an evaluation defines cannot be changed by an option: `compactFile`, `outputFile`,
`outputConfig.forceLCIO` and `outputConfig.forceDD4HEP` (k4opt reads the output as EDM4hep),
`numberOfEvents` (set `events`), `runType`, `inputFiles`, `macroFile`, `enableGun`,
`enableG4GPS`, `enableG4Gun`, any `gun.*` and `random.*` (set `seed`), `vertexSigma`,
`vertexOffset` and `crossingAngleBoost` are rejected with a pointer to where they come from,
also as unambiguous abbreviations (`ddsim` accepts them). These are refused too:
`steeringFile` (ddsim executes it as Python, which can change anything k4opt sets),
`skipNEvents` (it changes which events are simulated), `dumpParameter` and `dumpSteeringFile`
(ddsim exits without simulating) and `gdb`. A file named by `physics.pdgfile` is read from the
problem's directory if relative, passed to ddsim as an absolute path and recorded with its size
and SHA-256.

## Command

`--compactFile`, `--outputFile`, `--numberOfEvents`, the gun options or `--inputFiles` and (if
given) `--random.seed` are written by k4opt, followed by the options. Every value is written as
`--option=value`, so values may start with `-`. The argument list is run without a shell
(`shell=False`), so `$VARS` are not expanded, from `workdir`.

## Geometry

`ddsim` is given the compact file it is passed, normally a materialized snapshot. For
CLD_o2_v09 every file it loads comes from the snapshot (checked in the ddsim log). Other
geometries may reference files outside the tree; `k4opt geometry inspect` lists them as
external includes.
