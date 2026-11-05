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
`events` (required, and checked against the file), `seed` (required, passed to ddsim)
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

How fine must the ECAL cells be to tell a photon from a neutral pion? A 50 GeV pi0 decays into
two photons whose opening angle is at least 2 m/E, about 5.4 mrad, so they reach the CLD ECAL
barrel (radius 2150 mm) at least about 12 mm apart. Cells finer than that see two showers side
by side; coarser cells see one shower, as wide as a photon's. The
[example](../examples/problem/cld-ecal-pi0-rejection.yaml) simulates 50 GeV photons and pi0s on
each candidate geometry, compares a shower shape of the two and weighs the rejection it gives
against the readout channels the cells cost:

```bash
k4opt evaluate examples/problem/cld-ecal-pi0-rejection.yaml \
    --evaluator examples/evaluators/shower_shape.py:ShowerShape \
    --evaluator examples/evaluators/separation.py:Separation \
    --evaluator examples/evaluators/ecal_channels.py:EcalChannels \
    --param ecal_cell_size=8.0 --output runs/cld-pi0-8mm
```

Use the same three `--evaluator` arguments with `k4opt optimize`, replacing `--param` with
`--trials 30 --seed 42 --workers 8`, to search cell sizes. The pinned Key4hep stack is
required; each sample takes about 11 minutes on one core.

`calo.shower_shape` computes each event's **halo fraction**: the fraction of its energy, in
simulated ECAL barrel hits above the threshold, deposited farther than `radius` (10 mm, about
the Moliere radius of tungsten) from the radial axis through the shower's energy-weighted
center. It uses no particle reconstruction. A single photon keeps most of its energy near its
axis; the two photons of a pi0 put theirs off the axis of their common center, and only cells
finer than their separation can place it there. An event with no energy above threshold makes
the result invalid. Its `histogram` artifact contains `edges`, `counts`, `underflow` and
`overflow`; a fraction exactly at the last edge is included in the last bin.

`calo.separation` reads both histograms through roles, the signal's values being the low ones.
It selects the first whole bin reaching the requested photon efficiency and accepts all bins
through that bin. The achieved efficiency may exceed the request. `background_rejection` is
**1 minus the accepted background fraction**, from zero to one; it remains finite when no
background passes. The reported `cut` is the selected bin's upper edge. Histogram underflow or
a cut requiring signal overflow is invalid; background overflow remains in the denominator.
Empty histograms are invalid. Bin edges must match.

The two objectives are the rejection at 90 % photon efficiency (maximized) and the channel
count of `ecal.channels` (minimized). With 500 events per sample:

| cell size (mm) | pi0 rejection | halo cut | channels |
| ---: | ---: | ---: | ---: |
| 2.5 | 0.960 | 0.267 | 3.8e8 |
| 5.0 | 0.960 | 0.273 | 9.6e7 |
| 7.5 | 0.944 | 0.288 | 4.2e7 |
| 10.0 | 0.912 | 0.317 | 2.4e7 |
| 12.5 | 0.844 | 0.356 | 1.5e7 |
| 15.0 | 0.570 | 0.442 | 1.1e7 |
| 17.5 | 0.384 | 0.528 | 7.8e6 |
| 20.0 | 0.278 | 0.643 | 6.0e6 |

Each rejection is ± 0.01 to 0.02 from the 500 pi0s alone. Below 5 mm the rejection no longer
improves: the fluctuations of the showers, not the cells, limit it, while the channels keep
growing as 1/cell size². Above about 12 mm, where the cells grow past the separation of the two
photons, it collapses. The knee lies between: 10 mm cells keep 0.91 of the rejection at a
quarter of the channels of 5 mm cells. As in the photon example, the cell size changes only the
readout grid, so with the same seeds every candidate simulates the same showers, and the
differences between candidates are known better than each value. No uncertainty is assigned:
finite statistics, binning, using the signal sample to set the cut, the fixed energy and the
missing reconstruction and systematics limit this demonstration.

`ecal.channels` is a separate candidate-level cost model: `2*pi*radius*length*layers/cell_size²`.
Its dimensions and cell size are explicit settings in mm; the parameter `ecal_cell_size` sets
both the geometry's constant and its `cell_size` (one value, two targets), so the cost is
always that of the geometry simulated. It approximates a cylindrical barrel and omits
gaps, endcaps and layer differences; it does not count the detector's actual channels. Its
`check` shows both ways a classmethod `check` ends ([evaluation](evaluation.md#writing-an-evaluator)):
dimensions that are not positive are the problem's mistake (`EvaluationContractError`), a cell
size that is not positive is the candidate's (`cls.reject`), which rejects that candidate alone.

The two simulations each run once, both histogram artifacts are hashed, and the combined
record names the producer of each role. This shows how the cell size limits the photon/pi0
separation of the showers themselves; it is not a validated pi0 identification or particle-flow
study.

## Reconstructed pi0 example

The photon/pi0 example asks the showers; this one asks the reconstruction: does the particle flow
see the two photons of a neutral pion as two? The
[example](../examples/problem/cld-ecal-pi0-reco.yaml) simulates 200 neutral pions of 50 GeV in
the barrel on each candidate geometry and reconstructs them with CLD's reconstruction
(CLDConfig's `CLDReconstruction.py`, run by k4run, [k4run](k4run.md)). The example evaluator
`pfo.pi0` version 2 ([`pfo_pi0.py`](../examples/evaluators/pfo_pi0.py)) reads the
reconstruction's particle-flow objects (`PandoraPFOs`), its generated particles and the links
between the two (`RecoMCTruthLink`, a `Link<ReconstructedParticle,MCParticle>`), and keeps the
events whose pion decayed into two photons, neither converted in the tracker and each of at least
`min_energy` (160 of the 200):

| Metric | Unit | Definition |
| --- | --- | --- |
| `reconstruction` / `pi0.resolved` | 1 | the fraction of these events in which each generated photon is the main origin of a reconstructed photon (PDG 22) of at least `min_energy` of its own, with its standard error. The main origin of a reconstructed photon is the generated photon that deposited most of its calorimeter energy, as the link's weight says (CLD's `RecoMCTruthLinker`: 10000 × int(1000 × calorimeter share) + int(1000 × track share)), a shower's electron counted for the photon it descends from. Two fragments of one photon do not resolve the pion, however many there are |
| `reconstruction` / `pi0.photons` | 1 | the mean number of reconstructed photons of at least `min_energy` per event, with its standard error |

`ecal.channels` weighs it against the readout cost, as in the photon/pi0 example. On the nightly
2026-10-20, seed 42, `min_energy` 1 GeV:

| `ECal_cell_size` [mm] | 2.5 | 4.0 | 5.1 | 7.0 | 10 | 14 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pi0.resolved` | 0.16 ± 0.03 | 0.69 ± 0.04 | 0.49 ± 0.04 | 0.19 ± 0.03 | 0 | 0 | 0 |
| `pi0.photons` | 1.16 | 1.68 | 1.48 | 1.18 | 0.99 | 0.99 | 0.99 |

In these events, every one with two reconstructed photons had them from the two generated
photons: counting reconstructed photons alone would have given the same fractions, but an
optimizer could be rewarded for split showers by such a count, which the truth matching rules
out. From 4 mm up, coarser cells merge the photons, as their separation (at least about 12 mm)
says, and from 10 mm on the particle flow always sees one photon. Below 4 mm the reconstruction
finds fewer photons again: its digitization and Pandora settings were tuned for CLD's 5.1 mm
cells (an energy threshold per cell, for one, keeps less of a shower spread over smaller cells).
The metric is therefore that of the reconstruction as configured, not of the cells alone; a
parameter can set an argument of the job as well ([problem](problem.md#parameters)), and a
reconstruction tuned for each candidate is not done. It is a demonstration of a study that needs
reconstruction, not a validated pi0 identification.

## Not done

Evaluator lookup by name, chained combined evaluators, branches of the stages, aggregation over seeds, retries, a persistent store of runs and
an environment check against the pin.
