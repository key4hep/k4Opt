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
    name = "demo.cell_count"  # counterpart of a name in evaluation.evaluators
    version = "1"  # counterpart of that entry's version
    requires = "simulation"  # the pipeline stage to read (optional)
    settings = {"collection": str, "min_hits": 1}  # that entry's settings (optional)
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
and `version` identify the evaluator's measurement contract, as the `name` and `version` of an
entry of `evaluation.evaluators` do in the problem file; they are labels, not a lookup mechanism. The
*documentation* of an evaluator version defines what each of its metrics means: selection,
estimator, units, uncertainty convention.

What the optional parts do:

- `requires = "simulation"`: the framework runs that stage from the problem and passes the file
  it produced as `request.inputs["simulation"]`, along with the files of the stages before it;
  the evaluator never builds a command, starts a tool or handles its failures, and a stage that
  several evaluators require runs once (see [Stages](#stages)). `self.read(request, collection, type)`
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
  never carries two units. When a problem has several evaluators, every one must declare its
  `metrics`: they say which evaluator measures which objective, and two evaluators may not
  declare the same metric. A metric of another stage is written `("analysis", "rms")`;
  declaring one metric twice (`"rms"` and `("simulation", "rms")`) is an error, and so is a unit
  that is not a nonblank string (`"1"` for a dimensionless metric). Both are checked before
  anything runs.
- the return value: a mapping of metric to a number or to per-event numbers (a list, a numpy
  array, ...; `Measurement.from_samples`) becomes a `valid` result, with the unit from `metrics`.
  A metric is written `"name"` (its stage is `requires`) or as a pair `(stage, name)`. Return an
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
| `settings` | the evaluator-defined `settings` of its entry in `evaluation.evaluators`, nested data, copied and read-only |
| `inputs` | role name to local `Path` of an already prepared artifact (a compact XML snapshot, an EDM4hep file). The evaluator documents the roles it accepts |
| `workdir` | where the evaluator may write. `k4opt evaluate` gives each evaluator its own, `<output>/evaluators/<name>`, created for the call; the request never creates or checks it |
| `events`, `seed` | the optional controls of the shared context, the same for every evaluator of the problem; `events` is positive |
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
nonnegative `uncertainty`. A numpy number is stored as the Python `int` or `float` it stands
for, here and in `provenance`. An omitted uncertainty means *not provided*, not zero. Each
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
problem's controls (and the earlier stages' files) into a file, and an evaluator that `requires`
a stage reads that file from `request.inputs`. Each stage runs at most once per candidate,
however many evaluators read it, so several evaluators share one expensive simulation, and an
evaluator can be tested with a small fixture file and no Key4hep stack.

| Stage | Needs | Configured by | Provides |
| --- | --- | --- | --- |
| `generation` | | `evaluation.generation`: a particle gun, or `sample` (an event file) | `inputs["generation"]` for a sample (checked and hashed); nothing for a gun, which `ddsim` generates |
| `simulation` | `generation` | `evaluation.backend.ddsim.args`, the shared `events` and `seed` | `inputs["simulation"]`, the EDM4hep file of `ddsim` |

An evaluator names only the last stage it reads; the stages it needs run first, and it gets the
files of all of them (an evaluator that requires a later reconstruction stage will read the
simulation file too). It does not get the files of stages that only other evaluators need, so
its request does not depend on which other evaluators the problem has. Reconstruction will
follow the same pattern. Pipeline stages are steps that run; the `stage` of a metric only
classifies it, so an evaluator may report `analysis` metrics without an analysis stage.

A stage asks only for what it uses: `simulation` needs `inputs["geometry"]` and a `workdir`,
`generation` neither, so an evaluator that `requires` only `generation` runs without a geometry,
and one that requires nothing runs no stage at all. A stage that fails (sample unreadable, tool
missing, nonzero exit, no output, an output without the requested events) makes every evaluator
that reads it `failed` with the stage's message, without calling them, and the stages after it do
not run. Evaluators that do not read it are still called and keep their results.

An `inputs[stage]` supplied by the caller replaces running that stage and the ones only it
needs, and it is authoritative: a supplied `generation` file is simulated even if the problem
describes a particle gun. The record identifies it (the path, size and SHA-256 under
`provided`; a file that could not be hashed carries only its path, so a verified input is
distinguishable from one identified by its name). A sample is hashed once per file version, not
once per candidate: a file is told apart by size, mtime, ctime and inode, so one rewritten within
a few milliseconds of being hashed is not, and samples are assumed stable while a campaign runs.

The stages are recorded once per candidate, apart from the evaluators' results: what ran (the
command and exit code), what was supplied, the events found, the error if any, the stage's files
and the time spent on it. An evaluator's result holds only what it returned, so its artifact and
provenance names cannot clash with the stages'. The stages take their conditions (events, seed,
generation, backend arguments) from the one evaluation context of the problem, never from an
evaluator; a later problem with several named contexts (one per physics sample, say) can run the
same stages once per context. Stages are internal for now (`k4opt.problem._stages`).

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
`package.module:Class`; the same holds for the several evaluators of one problem. It is not a
registry; the `name` of each entry of `evaluation.evaluators` stays a label that must equal the
`name` of an evaluator that is loaded.

## Problem, evaluator, record

- **Problem** (`k4opt.problem`): the declared parameter domains and the requested objective and
  constraint metrics.
- **Evaluator**: user-written code that accepts an `EvaluationRequest` and returns an
  `EvaluationResult`; the documentation of its version defines what its metrics mean.
- **`EvaluationRecord`**: the request and result of **one evaluator's attempt** at one
  candidate, plus what the framework observed. It holds only that evaluator's own work; the
  stages it read are recorded once for the candidate.

| `EvaluationRecord` field | Source |
| --- | --- |
| `request` | the object the evaluator received |
| `result` | what the evaluator returned, or the `failed` result that stands for it when a stage it reads failed |
| `evaluator_name`, `evaluator_version` | the evaluator object that ran, checked against the problem before the call |
| `candidate_key` | framework-computed, see below |
| `elapsed_seconds` | wall-clock time of the single `evaluate()` call alone (`time.perf_counter()`), `None` if it was not called; not CPU time, memory or an estimate. The stages it read are timed once, with the stages |

`result.provenance` is **evaluator-supplied**; the record's identity and elapsed time, and the
record of the stages, are **framework-observed** (the executed `ddsim` command and exit code,
the hash of a sample, the events found). None of it proves which software environment
ran: the problem's `environment.key4hep` is a declared pin, and the versions of the stack are
not recorded yet.

`candidate_key` is a tuple with one `(reference, kind, tag, value)` entry per parameter, in the
problem's parameter order. It is problem-aware: `True`, `1` and `1.0` are not interchangeable
by Python equality here, so a categorical boolean, a categorical string `"1"` and a numeric `1`
stay distinct, while continuous `1`, `1.0` and `-0.0`/`0.0` (or numeric choices `1` and `1.0`)
agree. Numbers are written as exact decimal text, so huge integers are never forced through
`float`. The key is scoped to one problem definition; it is not a cross-problem hash, a
geometry fingerprint or a complete evaluation-cache key.

### Internal bridge (not the physicist-facing workflow)

`k4opt.problem._evaluate._evaluate_candidate(spec, candidate, evaluators, *, inputs=None,
workdir=None)` connects a `ProblemSpec` to explicitly supplied evaluators and returns a
`CandidateRecord`. It is private, meant for tests and a future runner; the user-facing entry
point will take only the problem YAML.

```python
candidate = _evaluate_candidate(spec, {"constants.ECal_cell_size": 6.0}, [CellCount()])
```

In order, it:

1. validates the candidate (`CandidateError` otherwise): exactly one value per declared
   parameter, keyed by `parameter.geometry.canonical_target`; continuous `int`/finite `float`
   within inclusive bounds; integer a real `int` (no `bool`, no rounding) within bounds;
   categorical one of the choices by kind (strings need a string, booleans a boolean, numbers
   compare numerically). Units are never interpreted or converted;
2. matches the evaluators to `evaluation.evaluators` by `name`, in any order: each declared
   evaluator must be given exactly once, with the declared `version`, and no other
   (`EvaluationContractError` otherwise);
3. before anything runs, checks the stages the evaluators require, calls each evaluator's
   `check()` and decides which evaluator owns each objective and constraint metric (see
   [problem](problem.md#evaluation); `EvaluationContractError` otherwise);
4. runs every stage the evaluators require once, unless `inputs` supplies its file, on
   `inputs["geometry"]` in `workdir`. The output of the `simulation` stage, run or supplied,
   must hold exactly `events` events (at least one if `events` is not set), otherwise the stage
   fails ("incomplete simulation"); the number found is recorded as `events_read`;
5. calls each evaluator's `evaluate()` once, in problem order, with a request built from the
   validated candidate, its own `settings`, the context's `events`, `seed`, `generation` and
   exact ordered `backend` arguments, and the files of the stages it reads, in its own
   `workdir/evaluators/<name>`. An evaluator whose stage failed gets that stage's message as a
   `failed` result instead. `EvaluationFailed` and `EvaluationInvalid` become a `failed` or
   `invalid` result; other exceptions of an evaluator propagate;
6. checks each result; a violation is an `EvaluationContractError`, an inconsistent evaluator
   rather than a violated constraint. A result must not report a metric another evaluator
   declares; a `valid` result must report every metric its evaluator owns (extra metrics are
   fine) in the `unit` a constraint declares; and if the evaluator declares `metrics`, every
   measurement of one, whatever the status, must carry the declared unit. Whether a value
   satisfies `min`/`max` is not judged.

The `CandidateRecord` holds one `EvaluationRecord` per evaluator in problem order, the stages
(each recorded once), which evaluator owns each objective and constraint metric, and the time of
the whole evaluation. Its `status` is the candidate's:

| Evaluators | Candidate | Meaning |
| --- | --- | --- |
| all `valid` | `valid` | complete and usable; every objective and constraint has a value |
| at least one `failed` | `failed` | technically incomplete: an evaluator did not run to the end |
| no `failed`, at least one `invalid` | `invalid` | complete, and at least one evaluator found the result unusable |

`failed` outranks `invalid`, so a candidate is `invalid` only after a complete evaluation, and an
incomplete evaluation is never reported as a scientific verdict. Its `message` names every
evaluator that is not `valid`, with its status, so a mixed outcome shows both. Every evaluator
counts, also one that owns no objective (a validity check, say). `value(stage, metric)` gives an
objective or constraint metric only from the `valid` result of the evaluator that owns it: a
missing value is never made up, and the diagnostics of an `invalid` result never stand in for it.
The measurements of the other evaluators are kept whatever happened to one of them.

Every second is counted once: a stage's time (running it and checking its output) is recorded
with the stage, an `EvaluationRecord`'s `elapsed_seconds` is its own `evaluate()` call, and the
`CandidateRecord`'s is the whole evaluation, stages, evaluators and checks.

## Deliberately deferred

An evaluator registry or lookup by name (evaluators are named on the command line), several
evaluation contexts per problem and evaluators that read more than one,
validating metric names, `k4run`, a public execution API (`k4opt evaluate` is the one internal vertical slice),
chaining artifacts between evaluators (callers do it by hand, as the tests show), execution
cost, retries and scheduling, fingerprints and caching, persisting or serializing the `EvaluationRecord`,
and aggregation, significance or constraint judgment.
