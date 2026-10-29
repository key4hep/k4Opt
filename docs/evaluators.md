# Evaluators and `k4opt evaluate`

k4opt ships no evaluators: an evaluator is an ordinary class implementing the
[evaluation contract](evaluation.md), written by the physicist. An evaluator only reads: the
framework runs the pipeline [stage](evaluation.md#stages) it `requires`.
[`examples/evaluators/calo_hit_multiplicity.py`](../examples/evaluators/calo_hit_multiplicity.py)
is a complete, working one to copy from. There is no registry: `k4opt evaluate` is told which
evaluator to use, and its name and version must be those the problem declares.

## `k4opt evaluate`

```bash
k4opt evaluate examples/problem/cld-ecal-hit-count.yaml \
  --evaluator examples/evaluators/calo_hit_multiplicity.py:CaloHitMultiplicity \
  --param constants.ECal_cell_size=6.0 \
  --output runs/cld-ecal-6mm
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
   If a stage fails, the result is `failed` and the evaluator is not called;
3. `<output>/summary.json` is written: candidate and key, status, measurements with
   uncertainty, evaluator, geometry fingerprint and manifest, requested and observed events,
   elapsed seconds, the exact `ddsim` command and exit code, artifact paths, and the
   environment (`declared`: the problem's pin; `observed`: `KEY4HEP_STACK`, `K4GEO` and the
   `ddsim` that was started; the two are not compared).

The command prints status, metric with uncertainty, events, fingerprint and the paths. Exit
status is 0 only for `valid`; `invalid` and `failed` print the reason on stderr and exit 1.

## The example evaluator `calo.hit_multiplicity` version 1

`requires = "simulation"`: it reads `request.inputs["simulation"]`, the EDM4hep file the
framework's simulation stage ([ddsim](ddsim.md)) wrote, and runs no tool. The stage uses `events`
(required), `seed` (optional, passed to ddsim when given), `evaluation.generation` (a particle
gun or a sample file) and `backend.ddsim.args`; the evaluator itself uses `settings` and `events` (to check that all
events are present).

**Metric** `simulation` / `ecal_barrel.mean_hit_cells`, unit `1`. For each simulated event, the
number of distinct `cellID` values in the collection `settings.readouts.ecal_barrel` (type
`edm4hep::SimCalorimeterHit`); an existing empty collection counts 0. The value is the sample
mean over all events and the uncertainty the standard error of the mean, `s / sqrt(N)` with
`s` the sample standard deviation (`ddof=1`); for one event the uncertainty is absent. There is
no event selection beyond the generation settings. The error is statistical only (one seed, one
sample); it says nothing about detector or systematic effects, and the metric is a technical
observable, not a performance measure such as shower-position resolution.

**Status.** The evaluator reports `failed` for an unreadable file, a collection that is missing in
any event or has another element type or no `cellID`, or an event count different from `events`,
and `invalid` for no events in the file or no hits in any event. A `ddsim` that is not found, not
started, exits nonzero or writes no output is a `failed` result of the stage. Misuse
(unsupported settings, conflicting arguments, a missing `generation`) raises before anything runs.

**Settings** (unknown keys are errors): `readouts.ecal_barrel` is the collection to read
(required, the only role). The primary particles are `evaluation.generation` (a particle gun or
a sample file), which [ddsim](ddsim.md) defines, together with the run, its command and the
geometry it simulates.

## Not done

Reconstruction, other metrics, evaluator lookup by name, aggregation over seeds, retries,
a persistent store of runs and an environment check against the pin.
