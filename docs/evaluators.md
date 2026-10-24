# Evaluators and `k4opt evaluate`

`k4opt.evaluators` holds the evaluators that ship with k4opt. They are ordinary implementations of
the [evaluation contract](evaluation.md); the contract does not depend on them and users add
their own the same way. There is no registry: `k4opt evaluate` uses `calo.hit_multiplicity`
for a problem that declares it and rejects any other evaluator name.

## `k4opt evaluate`

```bash
k4opt evaluate examples/problem/cld-ecal-hit-count.yaml \
  --param constants.ECal_cell_size=6.0 \
  --output runs/cld-ecal-6mm
```

One problem, one candidate, in an active Key4hep environment (`ddsim`, `podio`, `$K4GEO`).
`--param REFERENCE=VALUE` uses the canonical reference of the problem file and is repeated for
each declared parameter. `--output` must not exist. Before anything is created, the candidate
(keys, types, bounds), the evaluator identity and the evaluator settings are validated; an
error exits with status 1 and creates nothing. Then, in order:

1. the candidate becomes `Override`s (`6.0` with unit `mm` is written `6.0*mm`; units are limited
   to `um mm cm m mrad rad deg MeV GeV TeV`, expressions are not interpreted) and the geometry
   is materialized into `<output>/geometry` from the catalog that `load_problem()` discovered;
2. `ddsim` runs on that snapshot in `<output>/simulation` and the metric is read from its file;
3. `<output>/summary.json` is written: candidate and key, status, measurements with
   uncertainty, evaluator, geometry fingerprint and manifest, requested and observed events,
   elapsed seconds, the exact `ddsim` command and exit code, artifact paths, and the
   environment (`declared`: the problem's pin; `observed`: `KEY4HEP_STACK`, `K4GEO` and the
   `ddsim` that was started; the two are not compared).

The command prints status, metric with uncertainty, events, fingerprint and the paths. Exit
status is 0 only for `valid`; `invalid` and `failed` print the reason on stderr and exit 1.

## `calo.hit_multiplicity` version 1

Runs `k4opt.ddsim.simulate` and reads the file it wrote. Inputs: `request.inputs["geometry"]` (compact file), `request.workdir`, `events` (required),
`seed` (optional, passed to ddsim when given), `settings` and `backend_args["ddsim"]`.

**Metric** `simulation` / `ecal_barrel.mean_hit_cells`, unit `1`. For each simulated event, the
number of distinct `cellID` values in the collection `settings.readouts.ecal_barrel` (type
`edm4hep::SimCalorimeterHit`); an existing empty collection counts 0. The value is the sample
mean over all events and the uncertainty the standard error of the mean, `s / sqrt(N)` with
`s` the sample standard deviation (`ddof=1`); for one event the uncertainty is absent. There is
no event selection beyond the generator settings. The error is statistical only (one seed, one
sample); it says nothing about detector or systematic effects, and the metric is a technical
observable, not a performance measure such as shower-position resolution.

**Status.** `failed`: ddsim not found or not started, nonzero exit, no output, unreadable
file, a collection that is missing in any event or has another element type or no `cellID`, or
an event count different from `events`. `invalid`: no events in the file, or no hits in any event.
Programming errors and misuse (unsupported settings, conflicting arguments) raise.

**Settings** (unknown keys are errors): `readouts.ecal_barrel` is the collection to read
(required, the only role); `generator` is passed to [ddsim](ddsim.md), which defines its keys.
The `ddsim` run, its command and the geometry it simulates are described there.

## Not done

Reconstruction, other metrics, evaluator lookup by name, aggregation over seeds, retries,
a persistent store of runs and an environment check against the pin.
