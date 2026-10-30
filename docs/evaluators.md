# Evaluators and `k4opt evaluate`

k4opt ships no evaluators: an evaluator is an ordinary class, usually a subclass of `Evaluator`
([how to write one](evaluation.md#writing-an-evaluator)), written by the physicist. An evaluator
only reads: the framework makes the pipeline [stage](evaluation.md#stages) it reads (its `stage`)
available, as the problem's `stages` say.
[`examples/evaluators/calo_photon.py`](../examples/evaluators/calo_photon.py)
is a complete, working one to copy from. There is no registry: `k4opt evaluate` is told which
evaluators to use, and their names and versions must be those the problem declares.

## `k4opt evaluate`

```bash
k4opt evaluate examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --param constants.ECal_cell_size=10.0 \
  --output runs/cld-ecal-10mm
```

`--evaluator FILE.py:Class` (or `package.module:Class`) names an evaluator; give it once for each
entry of `evaluators`, in any order. The file is executed as ordinary Python
([`load_evaluator`](evaluation.md#loading-an-evaluator)), so name only code you trust.

`--input STAGE=FILE` (repeatable) uses an existing file as the output of a stage instead of
running it, to try an evaluator on a simulation made earlier, say:
`--input simulation=runs/cld-ecal-10mm/simulation/events.edm4hep.root`. The stages only it would
need do not run either. Its path, size and SHA-256 are recorded, and a supplied stage that the
parameters change (the simulation, when they vary the geometry) was not made with this
candidate's geometry: the command says so, and `summary.json` records
`represents_candidate: false`, so such a run never counts as an evaluation of the candidate.

`--timeout SECONDS` kills `ddsim` if it runs longer than that, for a geometry on which Geant4
stalls: the simulation stage fails with the tail of ddsim's standard error, its logs stay in
`simulation/`, and the candidate is `failed`. The limit is recorded with the stage
(`stages.simulation.timeout`, `null` without one). It is not part of the problem: how long a
simulation takes depends on the machine as much as on the candidate.

One problem, one candidate, in an active Key4hep environment (`ddsim`, `podio`, `$K4GEO`); from
Python, an [`EvaluationSession`](evaluation.md#evaluating-candidates-evaluationsession)
evaluates many candidates of a problem the same way.
`--param REFERENCE=VALUE` uses the canonical reference of the problem file and is repeated for
each declared parameter. `--output` must not exist. Before anything is created, the candidate
(keys, types, bounds), the supplied inputs, the evaluators (each declared one given once, with
its version and the stage it reads, and no other), their settings and collections and their
metrics (every objective and constraint measured by exactly one evaluator, see
[problem](problem.md#evaluators)) are validated; an error exits with status 1 and creates
nothing. A collection that ddsim is not expected to write for the geometry is printed as a
warning (`k4opt: warning: ...`, with the closest expected name) before anything runs, and the
run goes on ([why only a warning](evaluation.md#collections)). Then, in order:

1. the candidate becomes `Override`s (`6.0` with unit `mm` is written `6.0*mm`; units are limited
   to `um mm cm m mrad rad deg MeV GeV TeV`, expressions are not interpreted) and the geometry
   is materialized into `<output>/geometry` from the catalog that `load_problem()` discovered;
2. the stages the evaluators read are made available once each: a generation `file` is checked
   and hashed, and the `simulation` stage runs `ddsim` on that snapshot in
   `<output>/simulation`. Then each evaluator, in the order of the problem, reads the files of
   the stages it needs and may write in `<output>/evaluators/<name>`. If a stage fails or its
   output lacks events, the evaluators that read it are `failed` and not called; the others
   still are. So is an evaluator whose stage's file lacks a collection it declares, or holds it
   with another type: the message lists what the file holds;
3. the candidate is [assessed](evaluation.md#assessment-objectives-limits-and-feasibility)
   against the objectives and constraints, and `<output>/summary.json` (`k4opt.run/1`) is
   written:
   - `id`, the record's own identity (a UUID), and `run`, what it was made for: `null` for
     `k4opt evaluate`, a study and trial when an optimizer asks for it;
   - `problem`: the original path and name plus `definition`, the normalized problem loaded
     for this run, including parameter domains, objective directions and constraint limits;
   - `status` and `message` of the candidate ([how they follow from the evaluators'](evaluation.md#internal-bridge-not-the-physicist-facing-workflow)),
     the candidate and its key, whether the run `represents_candidate`, which stages were
     `supplied` with `--input`, and the `warnings` printed before the run;
   - `metrics`: every objective and constraint metric, the evaluator that owns it, and its
     value, unit and uncertainty, which are `null` unless that evaluator's result is `valid`;
   - `assessment`: the objective values in the problem's order, each bound of each constraint
     (`limits`, with the value, its margin to the limit and whether it is `satisfied`), and
     whether the candidate is `feasible` (`true`, `false`, or `null` when it is not assessed:
     not `valid`, or not made from the candidate);
   - `evaluators`, in problem order: name, version, the stage it reads, its source identity
     (`source`: path, size and SHA-256 of the file defining its class, taken at import,
     whether they are the bytes `executed`, and the `revision` of the git checkout that tracks
     it; [details](evaluation.md#loading-an-evaluator)), the metrics it declares with their
     units (`metrics`, `null` if it declares none), status, message,
     settings (`declared` as the problem wrote them, `used` as the evaluator received them, with
     defaults and quantities in the declared unit), the collection of each role, the stages it
     read, its measurements with uncertainty, the artifacts and
     provenance it returned, its directory and the time of its `evaluate()` call (`null` if it
     was not called);
   - `stages`, each once and in pipeline order: the particle gun, or the size and SHA-256 of a
     given or supplied file; the exact `ddsim` command, directory and exit code, the files its
     options name, with their hashes; the events found, the error if there was one, and the time
     spent on it;
   - the geometry fingerprint and manifest, the path and SHA-256 of each file its external
     references read (`externals`) and of each file only this candidate's values name
     (`candidate_externals`: a `ref` parameter choosing `${CAD}/other.stl`), requested and
     observed events, the time of the whole
     evaluation, the files of the stages under `paths`, and the environment (`declared`: the
     problem's pin; `observed`: the k4opt version, `k4opt_source` with the SHA-256 of k4opt's
     own Python files, taken before anything runs, and the commit and dirty flag of its git
     checkout (`null` when it is installed rather than checked out), `KEY4HEP_STACK` and `K4GEO`
     (those the session opened with, checked before each candidate) and the `ddsim` that was
     started; the pin and the stack are not compared).

The command prints the candidate's status, each evaluator's status and metrics with their
uncertainty, each limit with its verdict and margin, whether the candidate is feasible, the
events, the fingerprint and the paths. For the 10 mm candidate of the
[example study](#the-example-study), with the
[1 mrad constraint](problem.md#constraints) added to the problem (abridged, values rounded):

```text
status:      valid
evaluator:   calo.photon 1: valid
metric:      simulation/ecal_barrel.angular_residual = 0.946 +- 0.034 [mrad]
limit:       simulation/ecal_barrel.angular_residual <= 1.0: satisfied, margin -0.054 [mrad]
feasible:    yes
events:      1000 of 1000
```

The exit status is 0 for a `valid` candidate, feasible or not; otherwise the message (each
evaluator that is not `valid`, with its status and reason) goes to stderr and the exit status
is 1.

The summary is published only after it has been completely serialized and closed. A failed
write leaves no partial `summary.json`, and an existing file is never overwritten. Filesystem
failures while running a simulation or preparing an evaluator are recorded as failed outcomes;
an unwritable summary is reported as a run error with the run directory for diagnosis.

Several evaluators measure one candidate in the same run and share its simulation. For example,
next to the example evaluator, an evaluator of your own that needs no simulation, say a readout
cost computed from the cell size (`my_cost.py` and `my.readout_cost` are illustrative):

```yaml
# stages as in the example
evaluators:
  - name: calo.photon
    version: "1"
    stage: simulation
    collections:
      hits: ECalBarrelCollection
    settings:
      threshold:
        value: 70
        unit: keV
  - name: my.readout_cost  # reads no stage
    version: "1"

objectives:
  - stage: simulation
    metric: ecal_barrel.angular_residual
    direction: minimize
  - stage: analysis
    metric: ecal.readout_cost
    direction: minimize
```

```bash
k4opt evaluate problem.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --evaluator my_cost.py:ReadoutCost \
  --param constants.ECal_cell_size=10.0 --output runs/cost-10mm
```

All evaluators of a problem read the same `stages`. A study that needs another sample or event
count for some evaluator is a separate problem for now.

## The example evaluator `calo.photon` version 1

`stage = "simulation"`: it reads `request.inputs["simulation"]`, the EDM4hep file the
framework's simulation stage ([ddsim](ddsim.md)) wrote, and runs no tool. The stage uses its
`events` (required, and checked against the file), `seed` (optional, passed to ddsim when given)
and ddsim options, and the generation stage (a particle gun or an event file); the evaluator
itself uses only its `collections` and `settings`. It reads each event with `self.events()`, so
the hits and the generated photon of an event always come from that one event.

**Selection.** In each event the photon is the one `MCParticles` entry with generator status 1;
it must be a photon (PDG 22), since both metrics are defined for photons. A photon that
converted in the tracker (`isDecayedInTracker()`, about 9 % of the 10 GeV photons in CLD)
reaches the calorimeter as an electron and a positron; its event is left out of both metrics.

**Metrics**, both in the `simulation` stage:

- `ecal_barrel.fired_cells`, unit `1`: the hits of the collection `hits` (type
  `edm4hep::SimCalorimeterHit`, one hit per cell) with at least `settings.threshold`; the
  example uses 70 keV, half the most probable deposit of a minimum-ionizing particle in the
  0.5 mm silicon of the CLD ECAL. An event without such a hit counts 0.
- `ecal_barrel.angular_residual`, unit `mrad`: the angle between the photon's generated momentum
  and the line from its generated vertex to the energy-weighted center of the fired cells. A
  hit's position is the center of its cell, which is how the cell size enters. An event without
  fired cells has no residual.

Each value is the mean over the selected events with the standard error of the mean
([`Measurement.from_samples`](evaluation.md#result): `s / sqrt(N)`, `s` the sample standard
deviation with `ddof=1`); a single event has no uncertainty. The error is statistical only (one
seed, one sample). There is no electronics noise, digitization or reconstruction, which would
change both numbers. The residual measures a direction, not whether two nearby showers can be
told apart, the main reason for fine cells in particle flow; that needs reconstruction.

**Status.** The result is `failed` for an unreadable file, a collection (`hits` or `particles`)
that is missing in any event or has another element type (`self.events` raises
`EvaluationFailed`; one missing from the file altogether fails it before it is called), an event
without exactly one generated particle or whose generated
particle is not a photon, or an event count different from `events` (the stage checks it, see
[the flow](evaluation.md#problem-evaluator-record)), and `invalid` if no photon reached the
calorimeter and fired a cell (`EvaluationInvalid`). A `ddsim` that is not found, not started,
exits nonzero or writes no output is a `failed` result of the stage. Misuse (unsupported
settings, conflicting ddsim options, a missing generation stage) raises before anything runs.

**Collections**: `hits`, the calorimeter hits to measure (`SimCalorimeterHit`, named by the
problem: `ECalBarrelCollection` in the example), and `particles`, the generated particles
(`MCParticle`, `MCParticles` unless the problem names another).

**Settings** (unknown keys are errors): `threshold`, the energy above which a cell fires,
required and written with its unit (`value: 70`, `unit: keV`; the evaluator receives GeV, the
energy unit of EDM4hep). It has no default because it is part of what `fired_cells` and the
centroid mean: two runs with different thresholds measure different things, and the summary
records the threshold as written and as used. The primary particles are the generation stage (a
particle gun or an event file), which [ddsim](ddsim.md) defines, together with the run, its
command and the geometry it simulates.

## The example study

Seven candidates of [`cld-ecal-granularity.yaml`](../examples/problem/cld-ecal-granularity.yaml),
run with `k4opt evaluate` on the nightly 2026-10-20 (1000 photons each, about 5 minutes per
candidate on one core: half a minute of `ddsim` start-up and 0.3 s per photon):

| `ECal_cell_size` [mm] | `angular_residual` [mrad] | `fired_cells` |
| --- | --- | --- |
| 2.5 | 0.894 | 489 |
| 3.5 | 0.896 | 409 |
| 5.1 (CLD) | 0.898 | 333 |
| 7.0 | 0.909 | 279 |
| 10.0 | 0.946 | 229 |
| 14.0 | 1.037 | 190 |
| 20.0 | 1.381 | 155 |

Each residual is ± 0.034 mrad, each cell count ± 0.4 to 0.8. Coarser cells always save cells
and always cost direction, so every candidate is on the Pareto front, but the front has a
knee: below about 7 mm the direction hardly improves, because the fluctuations of the shower,
not the cell size, limit it, while the cells per photon keep growing. The cell size changes only
the readout grid, not the material, so with the same seed every candidate simulates the same
showers, and the differences between candidates are known far better than each value: compared
photon by photon, 5.1 mm cells are 0.004 ± 0.002 mrad worse than 2.5 mm cells and 10 mm cells
0.052 ± 0.006 mrad. The direction alone would not justify CLD's 5.1 mm cells; shower
separation in particle flow does, and measuring it needs reconstruction.

## Not done

Reconstruction, other metrics, evaluator lookup by name, several samples or event counts in one
problem, branches of the stages, aggregation over seeds, retries, a persistent store of runs and
an environment check against the pin.
