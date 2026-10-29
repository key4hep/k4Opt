# Evaluators and `k4opt evaluate`

k4opt ships no evaluators: an evaluator is an ordinary class, usually a subclass of `Evaluator`
([how to write one](evaluation.md#writing-an-evaluator)), written by the physicist. An evaluator only reads: the
framework runs the pipeline [stage](evaluation.md#stages) it `requires`.
[`examples/evaluators/calo_photon.py`](../examples/evaluators/calo_photon.py)
is a complete, working one to copy from. There is no registry: `k4opt evaluate` is told which
evaluator to use, and its name and version must be those the problem declares.

## `k4opt evaluate`

```bash
k4opt evaluate examples/problem/cld-ecal-granularity.yaml \
  --evaluator examples/evaluators/calo_photon.py:CaloPhoton \
  --param constants.ECal_cell_size=10.0 \
  --output runs/cld-ecal-10mm
```

`--evaluator FILE.py:Class` (or `package.module:Class`) names the evaluator; the file is
executed as ordinary Python ([`load_evaluator`](evaluation.md#loading-an-evaluator)), so name
only code you trust.

One problem, one candidate, in an active Key4hep environment (`ddsim`, `podio`, `$K4GEO`).
`--param REFERENCE=VALUE` uses the canonical reference of the problem file and is repeated for
each declared parameter. `--output` must not exist. Before anything is created, the candidate
(keys, types, bounds), the evaluator identity and the evaluator settings are validated; an
error exits with status 1 and creates nothing. Then, in order:

1. the candidate becomes `Override`s (`6.0` with unit `mm` is written `6.0*mm`; units are limited
   to `um mm cm m mrad rad deg MeV GeV TeV`, expressions are not interpreted) and the geometry
   is materialized into `<output>/geometry` from the catalog that `load_problem()` discovered;
2. the `generation` stage checks the sample (if any) and the `simulation` stage runs `ddsim` on
   that snapshot in `<output>/simulation`; the evaluator reads the metric from the file it wrote.
   If a stage fails or its output lacks events, the result is `failed` and the evaluator is not
   called;
3. `<output>/summary.json` is written: candidate and key, status, measurements with
   uncertainty, evaluator with its `settings` and own `provenance`, geometry fingerprint and
   manifest, requested and observed events, elapsed seconds, the exact `ddsim` command and exit
   code, artifact paths, and the environment (`declared`: the problem's pin; `observed`: `KEY4HEP_STACK`, `K4GEO` and the
   `ddsim` that was started; the two are not compared).

The command prints status, metric with uncertainty, events, fingerprint and the paths. Exit
status is 0 only for `valid`; `invalid` and `failed` print the reason on stderr and exit 1.

## The example evaluator `calo.photon` version 1

`requires = "simulation"`: it reads `request.inputs["simulation"]`, the EDM4hep file the
framework's simulation stage ([ddsim](ddsim.md)) wrote, and runs no tool. The stage uses `events`
(required, and checked against the file), `seed` (optional, passed to ddsim when given),
`evaluation.generation` (a particle gun or a sample file) and `backend.ddsim.args`; the
evaluator itself uses only `settings`.

**Selection.** In each event the photon is the one `MCParticles` entry with generator status 1.
A photon that converted in the tracker (`isDecayedInTracker()`, about 9 % of the 10 GeV photons
in CLD) reaches the calorimeter as an electron and a positron; its event is left out of both
metrics.

**Metrics**, both in the `simulation` stage:

- `ecal_barrel.fired_cells`, unit `1`: the hits of `settings.collection` (type
  `edm4hep::SimCalorimeterHit`, one hit per cell) with at least 70 keV, half the most probable
  deposit of a minimum-ionizing particle in the 0.5 mm silicon of the CLD ECAL. An event without
  such a hit counts 0.
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

**Status.** The result is `failed` for an unreadable file, a collection (`settings.collection` or
`MCParticles`) that is missing in any event or has another element type (`read_collection`
raises `EvaluationFailed`), an event without exactly one generated particle, or an event count
different from `events` (the stage checks it, see [the flow](evaluation.md#problem-evaluator-record)),
and `invalid` if no photon reached the calorimeter and fired a cell (`EvaluationInvalid`). A
`ddsim` that is not found, not started, exits nonzero or writes no output is a `failed` result of
the stage. Misuse (unsupported settings, conflicting arguments, a missing `generation`) raises
before anything runs.

**Settings** (unknown keys are errors): `collection` is the collection to read (required, the
only key). The threshold is part of the metric definition, so another threshold is another
evaluator version. The primary particles are `evaluation.generation` (a particle gun or a sample
file), which [ddsim](ddsim.md) defines, together with the run, its command and the geometry it
simulates.

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

Reconstruction, other metrics, evaluator lookup by name, aggregation over seeds, retries,
a persistent store of runs and an environment check against the pin.
