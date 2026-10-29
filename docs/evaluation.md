# Evaluation contract

`k4opt.evaluation` defines how anything that measures a candidate (a simulation, a
reconstruction, an analysis, a cost model) talks to k4opt: **one evaluator, one request, one
result**. This is the contract only: it does not run anything, look an evaluator up by the name in
`problem.yaml` or persist results. An example evaluator and `k4opt evaluate` are described in
[evaluators.md](evaluators.md); k4opt ships no evaluators of its own. An evaluator can be
written and tested without Key4hep and without a problem file.

## Writing an evaluator

```python
from k4opt.evaluation import Evaluator


class CellCount(Evaluator):
    name = "demo.cell_count"  # counterpart of evaluation.evaluator
    version = "1"  # counterpart of evaluation.version
    requires = "simulation"  # the pipeline stage to read (optional)
    settings = {"collection": str, "min_hits": 1}  # evaluation.settings (optional)
    metrics = {"ecal.cells": "1"}  # metric -> unit; the stage is `requires` (optional)

    def evaluate(self, request):
        hits = self.read(request, request.settings["collection"], "SimCalorimeterHit")
        cells = [len({hit.getCellID() for hit in event}) for event in hits]
        if max(cells) < request.settings["min_hits"]:
            self.invalid("too few hits")  # also: self.fail("...")
        return {"ecal.cells": cells}  # a list: its mean with the standard error; a number: itself
```

Everything but `name`, `version` and `evaluate` is optional, and `Evaluator` is only a
convenience: any class with those three members is an evaluator (the structural contract is
`EvaluatorProtocol`). There is no decorator or registry, and no `SimulationEvaluator` or
`AnalysisEvaluator`: those are scientific domains, not different calling conventions. `name`
and `version` identify the evaluator's measurement contract, as `evaluation.evaluator` and
`evaluation.version` do in the problem file; they are labels, not a lookup mechanism. The
*documentation* of an evaluator version defines what each of its metrics means: selection,
estimator, units, uncertainty convention.

What the optional parts do:

- `requires = "simulation"`: the framework runs that stage from the problem and passes the file
  it produced as `request.inputs["simulation"]`; the evaluator never builds a command, starts a
  tool or handles its failures (see [Stages](#stages)). `self.read(request, collection, type)`
  reads a collection of that file (see [Reading EDM4hep files](#reading-edm4hep-files)).
- `settings`: a mapping of key to a type (`str`, `int`, `float`, `bool`, `list`, `dict`) for a
  required setting, or to a default value for an optional one (`"min_hits": 1` is an optional
  `int`). Unknown keys, missing required keys, wrong types and a blank required `str` are an
  `EvaluationContractError`, raised before anything is created or run (a `bool` is not an `int`,
  an `int` is a `float`). The request an evaluator gets carries the defaults, so
  `request.settings` (and the record) show the settings that were used; a `dict` arrives as a
  read-only mapping and a `list` as a tuple. Override `check(request)` for rules beyond that and
  call `super().check(request)`; plain evaluators can define `check` alone. `check` may run more
  than once for one candidate, so keep it cheap and free of side effects.
- `metrics`: lets `k4opt evaluate` reject an unsupported objective before anything runs, and
  holds every returned measurement of those metrics to the declared unit, so one metric identity
  never carries two units. A metric of another stage is written `("analysis", "rms")`;
  declaring one metric twice (`"rms"` and `("simulation", "rms")`) is an error.
- the return value: a mapping of metric to a number or to a list of per-event numbers
  (`Measurement.from_samples`) becomes a `valid` result, with the unit from `metrics`. Return an
  `EvaluationResult` for artifacts, provenance or diagnostics.

The long form needs no base class and no shorthand:

```python
from k4opt.evaluation import EvaluationRequest, EvaluationResult, Measurement


class CellCount:
    name = "demo.cell_count"
    version = "1"

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        size = request.candidate["constants.ECal_cell_size"]
        if not isinstance(size, float):
            return EvaluationResult.failed("ECal_cell_size must be a float")
        return EvaluationResult.valid(
            [Measurement("analysis", "ecal.cells", 1000.0 / size**2, unit="1", uncertainty=0.5)]
        )
```

## Request

`EvaluationRequest` is immutable and every field is optional:

| Field | Meaning |
| --- | --- |
| `candidate` | parameter values for this evaluation, keyed by the canonical parameter identity (a geometry parameter: its canonical target such as `constants.ECal_cell_size`); booleans, integers, finite floats or strings. Not checked against bounds |
| `settings` | the evaluator-defined `evaluation.settings`, nested data, copied and read-only |
| `inputs` | role name to local `Path` of an already prepared artifact (a compact XML snapshot, an EDM4hep file). The evaluator documents the roles it accepts |
| `workdir` | where the evaluator may write. Never created or checked by the request |
| `events`, `seed` | the optional controls of `evaluation`; `events` is positive |
| `backend_args` | tool name to raw arguments, kept exactly (order, repeats, empty strings), never parsed |
| `generation` | the problem's `evaluation.generation`, a particle gun or `{"sample": path}` exactly as written (a relative path is not resolved), copied and read-only. The conditions of the measurement, for an evaluator that runs its own workflow; one that `requires` the `generation` stage gets the resolved, hashed file as `inputs["generation"]` instead |

The request holds no `ProblemSpec`, geometry or backend object; the later runner builds it
from those.

## Result

`EvaluationResult` is what `evaluate` returns:

| Field | Meaning |
| --- | --- |
| `status` | a `Status`, a `StrEnum` whose values are `"valid"`, `"invalid"` and `"failed"`. `valid`: ran, result scientifically usable. `invalid`: ran, but the result is not usable (the evaluator's own checks failed). `failed`: the execution failed |
| `measurements` | tuple of `Measurement`, mixing stages if needed. Independent of the status: an `invalid` result may keep diagnostic values (they are never objectives), and a `valid` one may have none if it only produced artifacts; a problem that asks for a metric still requires it from a `valid` result |
| `artifacts` | role name to local output `Path`, possibly empty; a later evaluator can take them as `inputs`. Existence is not checked at construction |
| `message` | explanation; required for `invalid` and `failed` |
| `provenance` | optional JSON-compatible metadata the evaluator chose to supply |

A `Measurement` is `stage`, `metric`, a finite `value` (booleans, NaN and infinity are
rejected), an optional descriptive `unit` (never converted) and an optional symmetric,
nonnegative `uncertainty`. An omitted uncertainty means *not provided*, not zero. Each
`(stage, metric)` occurs once per result. A missing measurement is absent, never `0`, `inf` or
a fake success.

`Measurement.from_samples(stage, metric, samples, unit)` averages per-event values: the value
is the sample mean and the uncertainty the standard error of the mean, `s / sqrt(N)` with `s`
the sample standard deviation (`ddof=1`); a single sample has no uncertainty. Statistical
uncertainty only: it says nothing about systematic effects.

Build a result with the constructor that matches the outcome:

```python
EvaluationResult.valid([Measurement(...)], artifacts=..., provenance=...)
EvaluationResult.invalid("no hits in any event", [Measurement(...)], provenance=...)
EvaluationResult.failed("ddsim exited with status 1", artifacts=...)
```

`invalid` and `failed` require a message, so an unusable result always says why.
`EvaluationResult(status, ...)` stays available and checks the same thing.

An evaluator can also stop at the first problem by raising `EvaluationInvalid("...")` or
`EvaluationFailed("...")`. `k4opt` turns the exception into the matching result with the text as
its message; anything else an evaluator raises propagates. The exceptions carry no
measurements: build the result by hand to keep diagnostics.

### Reading EDM4hep files

`self.read(request, collection, value_type)` is `read_collection` on the file of the stage in
`requires`, and `request.read(stage, collection, value_type)` on the file of any stage.
`read_collection(path, collection, value_type)` yields the collection of each event of an
EDM4hep file; `count_events(path)` counts the events. `podio` comes from the Key4hep stack and
is imported only when a file is opened, so `k4opt.evaluation` imports without it. A file that
cannot be opened, a missing `events` category or collection, an element type other than
`value_type` (`"SimCalorimeterHit"` means `"edm4hep::SimCalorimeterHit"`) and every exception of the C++ reader are an `EvaluationFailed`: a collection is
never silently read as empty. An existing but empty collection is yielded like any other.

Violations raise `EvaluationContractError` (a `K4OptError`) when the object is built.

## Stage classifies, it does not prescribe

`stage` is a free descriptive string next to `metric`, as in the problem file. It is not a
workflow instruction, and the set of stages is not enumerated. One evaluator may report
`simulation` and `analysis` metrics together.

## Stages

The stages of the pipeline are run by the framework, not by an evaluator. A stage turns the
problem's controls (and the earlier stages' files) into a file once per candidate, and an
evaluator that `requires` a stage reads that file from `request.inputs`. Several evaluators can
therefore share one expensive run, and an evaluator can be tested with a small fixture file and
no Key4hep stack.

| Stage | Needs | Configured by | Provides |
| --- | --- | --- | --- |
| `generation` | | `evaluation.generation`: a particle gun, or `sample` (an event file) | `inputs["generation"]` for a sample (checked and hashed); nothing for a gun, which `ddsim` generates |
| `simulation` | `generation` | `evaluation.backend.ddsim.args`, the shared `events` and `seed` | `inputs["simulation"]`, the EDM4hep file of `ddsim` |

An evaluator names only the last stage it reads; the stages it needs run first. Reconstruction
and analysis will follow the same pattern. A stage asks only for what it uses: `simulation` needs
`inputs["geometry"]` and a `workdir`, `generation` neither, so an evaluator that `requires` only
`generation` runs without a geometry. A stage that fails (sample unreadable, tool missing,
nonzero exit, no output) becomes a `failed` result with its logs as artifacts, and the evaluator
is not called.

An `inputs[stage]` supplied by the caller replaces running that stage and the ones only it
needs, and it is authoritative: a supplied `generation` file is simulated even if the problem
describes a particle gun. The record identifies it (`provenance["stages"][stage]` holds the
path, size and SHA-256 under `provided`; a file that could not be hashed carries only its path,
so a verified input is distinguishable from one identified by its name). A sample is hashed once
per file version, not once per candidate: a file is told apart by size, mtime, ctime and inode,
so one rewritten within a few milliseconds of being hashed is not, and samples are assumed
stable while a campaign runs.

The files of a stage are added to the result's `artifacts` as `<stage>.<role>`
(`simulation.stdout`, `simulation.edm4hep`, ...) so stages cannot overwrite each other, and the
stages' record goes to `provenance["stages"]`. An evaluator that returns an artifact role equal
to one of these gets an `EvaluationContractError`, and so does one that returns a `stages`
provenance key at all, whether or not a stage ran, because the record reads that key as what the
framework observed. Stages are internal for now (`k4opt.problem._stages`).

## Loading an evaluator

`load_evaluator("path/to/file.py:Class")` (or `"package.module:Class"`) imports the file or
module, creates the class without arguments and checks that it has `name`, `version` and
`evaluate`; `EvaluatorLoadError` otherwise. This is what `k4opt evaluate --evaluator` uses. The
file is executed as ordinary Python: name only evaluators you trust. Like `python file.py`, its
directory is put on `sys.path` (and stays there), so an evaluator can import helper modules next
to it, at the top of the file or inside a method. Python caches modules by name for the whole
process, so two evaluator files that each import a helper called `metrics` share whichever was
imported first. For one study this is fine. When several studies run in one process, give the
helpers unique names or, better, install the study as a package and use
`package.module:Class`. It is not a registry; the
problem's `evaluation.evaluator` stays a label that must equal the loaded evaluator's `name`.

## Problem, evaluator, record

- **Problem** (`k4opt.problem`): the declared parameter domains and the requested objective and
  constraint metrics.
- **Evaluator**: user-written code that accepts an `EvaluationRequest` and returns an
  `EvaluationResult`; the documentation of its version defines what its metrics mean.
- **`EvaluationRecord`**: the request and result of **one attempted evaluation**, plus what the
  framework observed.

| `EvaluationRecord` field | Source |
| --- | --- |
| `request`, `result` | the actual objects exchanged with the evaluator, unchanged |
| `evaluator_name`, `evaluator_version` | the evaluator object that ran, checked against the problem before the call |
| `candidate_key` | framework-computed, see below |
| `elapsed_seconds` | wall-clock time around the stage and the single `evaluate()` call (`time.perf_counter()`); not CPU time, memory or an estimate |

`result.provenance` is **evaluator-supplied**; the record's identity and elapsed time are
**framework-observed**. Neither proves which software environment ran: the problem's
`environment.key4hep` is a declared pin, and no executed command, input hash or version is
recorded yet.

`candidate_key` is a tuple with one `(reference, kind, tag, value)` entry per parameter, in the
problem's parameter order. It is problem-aware: `True`, `1` and `1.0` are not interchangeable
by Python equality here, so a categorical boolean, a categorical string `"1"` and a numeric `1`
stay distinct, while continuous `1`, `1.0` and `-0.0`/`0.0` (or numeric choices `1` and `1.0`)
agree. Numbers are written as exact decimal text, so huge integers are never forced through
`float`. The key is scoped to one problem definition; it is not a cross-problem hash, a
geometry fingerprint or a complete evaluation-cache key.

### Internal bridge (not the physicist-facing workflow)

`k4opt.problem._evaluate._evaluate_candidate(spec, candidate, evaluator, *, inputs=None,
workdir=None)` connects a `ProblemSpec` to an explicitly supplied evaluator. It is private,
meant for tests and a future runner; the user-facing entry point will take only the problem
YAML.

```python
record = _evaluate_candidate(spec, {"constants.ECal_cell_size": 6.0}, CellCount())
```

In order, it:

1. validates the candidate (`CandidateError` otherwise): exactly one value per declared
   parameter, keyed by `parameter.geometry.canonical_target`; continuous `int`/finite `float`
   within inclusive bounds; integer a real `int` (no `bool`, no rounding) within bounds;
   categorical one of the choices by kind (strings need a string, booleans a boolean, numbers
   compare numerically). Units are never interpreted or converted;
2. requires `evaluator.name` and `evaluator.version` to equal `evaluation.evaluator` and
   `evaluation.version` (`EvaluationContractError` otherwise), before anything runs;
3. if the evaluator has a `requires` stage that `inputs` does not already provide, checks the
   stage and runs it on `inputs["geometry"]` in `workdir` (both needed). A failed stage gives a
   `failed` result and the evaluator is not called. The output of the `simulation` stage, run
   or supplied, must hold exactly `events` events (at least one if `events` is not set), otherwise
   the result is `failed` ("incomplete simulation") and the evaluator is not called; the number
   found is `provenance["stages"]["simulation"]["events_read"]`;
4. builds the `EvaluationRequest` from the validated candidate plus `settings`, `events`,
   `seed`, the exact ordered `backend` arguments and the stage's file in `inputs`, and calls
   `evaluate()` exactly once. `EvaluationFailed` and `EvaluationInvalid` become a `failed` or
   `invalid` result; other exceptions of the evaluator propagate. The stage's files and
   process record are merged into the result's `artifacts` and into
   `provenance["stages"][stage]`;
5. for a `valid` result, requires every `(stage, metric)` of any objective or constraint to be
   measured (extra metrics are fine) and, where a constraint declares a `unit`, the
   measurement's `unit` to be equal. A violation is an `EvaluationContractError`: an
   inconsistent evaluator, not a violated constraint. Whether a value satisfies `min`/`max` is
   not judged. `invalid` and `failed` results are kept as returned.
6. if the evaluator declares `metrics`, every measurement of a declared metric, whatever the
   status, must carry the declared unit (`EvaluationContractError` otherwise); metrics it does
   not declare are not checked.

## Deliberately deferred

An evaluator registry or lookup by name (the evaluator is named on the command line),
validating metric names, `k4run`, a public execution API (`k4opt evaluate` is the one internal vertical slice),
chaining artifacts between evaluators (callers do it by hand, as the tests show), execution
cost, retries and scheduling, fingerprints and caching, persisting or serializing the `EvaluationRecord`,
and aggregation, significance or constraint judgment.
