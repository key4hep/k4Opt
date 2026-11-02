# Example evaluators

k4opt ships no evaluators: an evaluator is an ordinary class, usually a subclass of `Evaluator`
([how to write one](evaluation.md#writing-an-evaluator)), written by the physicist. An evaluator
only reads: the framework makes the pipeline [stage](evaluation.md#stages) it reads (its `stage`)
of each sample it measures available, as the problem's `samples` say.
[`examples/evaluators/calo_photon.py`](../examples/evaluators/calo_photon.py)
is a complete, working one to copy from. There is no registry:
[`k4opt evaluate`](execution.md#k4opt-evaluate) is told which evaluators to use, and their names
and versions must be those the problem declares.

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
([`Measurement.from_values`](evaluation.md#result): `s / sqrt(N)`, `s` the sample standard
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

## Photon/pi0 example

The [combined example](../examples/problem/cld-ecal-pi0-rejection.yaml) simulates 20 GeV photons
and neutral pions on one candidate geometry, then compares their shower-width histograms:

```bash
k4opt evaluate examples/problem/cld-ecal-pi0-rejection.yaml \
    --evaluator examples/evaluators/shower_shape.py:ShowerShape \
    --evaluator examples/evaluators/separation.py:Separation \
    --evaluator examples/evaluators/ecal_channels.py:EcalChannels \
    --param ecal_cell_size=8.0 --output runs/cld-pi0-8mm
```

Use the same three `--evaluator` arguments with `k4opt optimize`, replacing `--param` with
`--trials 30 --seed 42`, to search cell sizes. The pinned Key4hep stack is required.

`calo.shower_shape` computes each event's energy-weighted RMS hit distance transverse to the
radial axis through the shower's energy-weighted center, in mm. It uses simulated ECAL barrel
hits above the threshold, without particle reconstruction. An event with no energy above
threshold makes the result invalid. Its `histogram` artifact contains `edges`, `counts`,
`underflow` and `overflow`; a width exactly at the last edge is included in the last bin.

`calo.separation` reads both histograms through roles. It selects the first whole bin reaching
the requested photon efficiency and accepts all bins through that bin. The achieved efficiency
may exceed the request. `background_rejection` is **1 minus the accepted background fraction**,
from zero to one; it remains finite when no background passes. The reported `width_cut` is the
selected bin's upper edge. Histogram underflow or a cut requiring signal overflow is invalid;
background overflow remains in the denominator. Empty histograms are invalid. Bin edges must
match. No uncertainty is assigned: finite statistics, binning, using the signal sample to set
the cut, and missing reconstruction/systematics limit this demonstration.

`ecal.channels` is a separate candidate-level cost model: `2*pi*radius*length*layers/cell_size²`.
Its dimensions and cell size are explicit settings in mm; the parameter `ecal_cell_size` sets
both the geometry's constant and its `cell_size` (one value, two targets), so the cost is
always that of the geometry simulated. It approximates a cylindrical barrel and omits
gaps, endcaps and layer differences; it does not count the detector's actual channels.

The two simulations each run once, both histogram artifacts are hashed, and the combined
record names the producer of each role. This is a runnable demonstration of the data flow,
not a validated particle-flow separation or physics-sensitivity study.

## Not done

Reconstruction, evaluator lookup by name, chained combined evaluators, stage configuration
shared by samples, branches of the stages, aggregation over seeds, retries, a persistent store of runs and
an environment check against the pin.
