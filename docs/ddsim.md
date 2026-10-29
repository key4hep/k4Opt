# ddsim

`k4opt.ddsim` runs Key4hep's `ddsim` once on a prepared geometry with a particle gun: it builds
the command from the problem's `evaluation.simulation` and `evaluation.backend.ddsim`, runs the `ddsim` of the active Key4hep environment
and keeps the command, exit code and logs. It does not read the output; an evaluator does
(see [evaluators.md](evaluators.md)). `k4opt evaluate` runs it as the `simulation`
[stage](evaluation.md#stages) once per candidate, so evaluators never call it. Checked against the `ddsim` of the nightly 2026-10-20.

```python
from k4opt.ddsim import simulate

run = simulate(
    compact=compact,
    workdir=new_dir,
    events=100,
    seed=42,
    particle_gun=generation,  # evaluation.generation; or sample=Path("events.hepmc") instead
    args=("--physics.list", "FTFP_BERT"),
)
print(run.error or run.output)
```

`simulate` raises `DdsimError` for unsupported settings, conflicting arguments or an existing
`workdir`, before anything is run. A `ddsim` that cannot run or fails is reported in
`SimulationRun.error` (not found, not started, nonzero exit with the tail of its standard
error, no output file) with `returncode`, `argv` and the files. `check_arguments` does the first
part alone. `workdir` receives `events.edm4hep.root`, `ddsim.stdout.log`, `ddsim.stderr.log` and
`ddsim.json` (argument list, directory, exit code).

## Particle gun (`evaluation.generation`)

Without a `sample`, the keys of `evaluation.generation` describe a particle gun. Unknown keys are
errors.

| Key | Meaning |
| --- | --- |
| `particle` | `--gun.particle` |
| `energy` | `value` and `unit` (`MeV`, `GeV`, `TeV`): the *total* energy, `--gun.energy` |
| `vertex` | optional `value` `[x, y, z]` and `unit` (`mm`, `cm`, `m`); default origin, always passed explicitly |
| `direction` | `distribution` (`cos(theta)` or `uniform`, passed to `--gun.distribution`), `theta` and `phi` as `[low, high]` and `unit` (`deg`, `rad`); theta within 0 to a half turn, phi at most a full turn |

One primary per event (`--gun.multiplicity=1`). `cos(theta)` is flat in cos(theta) (isotropic
in solid angle) and phi, whereas `uniform` is flat in theta. Checked on the nightly: of 4000 events with theta in 10-170 degrees,
50.2 % fell in |cos(theta)| < 0.5 (flat in cos: 50.8 %) against 37 % for `uniform`. Another
distribution is an error, never a fallback.

## Input sample (`evaluation.generation.sample`)

Instead of a gun, `generation: {sample: events.hepmc}` simulates the events of a file. A relative
path is read from the directory of the problem file. `ddsim` receives it as `--inputFiles`,
chooses the reader from the file extension and validates the content, so k4opt accepts every
format `ddsim` does. The `generation` stage checks that the file exists and records its path,
size and SHA-256 in the provenance, so a changed sample is a different measurement. A gun and a
sample together, or neither, are errors. `events` is the number of events to process; a sample
with fewer events gives an incomplete simulation, which the evaluator reports as `failed`.

## Command

`--compactFile`, `--outputFile`, `--numberOfEvents`, the gun options and (if given)
`--random.seed` are written by k4opt (as `--option=value`, so values may start with `-`),
followed by `backend.ddsim.args` unchanged, in order, repeats included. The argument list is
run without a shell (`shell=False`), so `$VARS` are not expanded, from `workdir`. An argument
that sets an option k4opt manages is rejected: `--compactFile`, `--outputFile`,
`--numberOfEvents`, `--runType`, `--inputFiles`, `--macroFile`, `--enableGun`, `--enableG4GPS`,
`--enableG4Gun`, any `--gun.*` and `--random.*`, `--vertexSigma`, `--vertexOffset`,
`--crossingAngleBoost`, the short forms `-N -O -G -I -M` and unambiguous abbreviations
(`ddsim` accepts them). Everything else is validated by `ddsim`. A `--steeringFile` is passed on
but not hashed, and it can change what is simulated.

## Geometry

`ddsim` is given the compact file it is passed, normally a materialized snapshot. For
CLD_o2_v09 every file it loads comes from the snapshot (checked in the ddsim log). Other
geometries may reference files outside the tree; `k4opt geometry inspect` lists them as
external includes.
