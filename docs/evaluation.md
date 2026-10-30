# Evaluation contract

`k4opt.evaluation` defines how anything that measures a candidate (a simulation, a
reconstruction, an analysis, a cost model) talks to k4opt: **one evaluator, one request, one
result**. This is the contract only: it does not run anything, look an evaluator up by the name in
`problem.yaml` or persist results. An example evaluator and `k4opt evaluate` are described in
[evaluators.md](evaluators.md); k4opt ships no evaluators of its own. An evaluator can be
written and tested without Key4hep and without a problem file.

## Writing an evaluator

```python
from k4opt.evaluation import Collection, Evaluator, Quantity


class CellCount(Evaluator):
    name = "demo.cell_count"  # counterpart of a name in the problem's evaluators
    version = "1"  # counterpart of that entry's version
    stage = "simulation"  # the pipeline stage it reads, as that entry says too (optional)
    collections = {"hits": Collection("SimCalorimeterHit")}  # read from that stage (optional)
    settings = {"threshold": Quantity("GeV"), "min_cells": 1}  # that entry's settings (optional)
    metrics = {"ecal.cells": "1"}  # metric -> unit; classified under `stage` (optional)

    def evaluate(self, request):
        threshold = request.settings["threshold"]  # a float in GeV
        cells = [
            len({hit.getCellID() for hit in event.hits if hit.getEnergy() >= threshold})
            for event in self.events(request)
        ]
        if max(cells) < request.settings["min_cells"]:
            self.invalid("too few cells")  # also: self.fail("...")
        return {"ecal.cells": cells}  # a list: its mean with the standard error; a number: itself
```

Its entry in the problem:

```yaml
evaluators:
  - name: demo.cell_count
    version: "1"
    stage: simulation
    collections:
      hits: ECalBarrelCollection
    settings:
      threshold:
        value: 70
        unit: keV
```

Everything but `name`, `version` and `evaluate` is optional, and `Evaluator` is only a
convenience: any class with those three members is an evaluator (the structural contract is
`EvaluatorProtocol`). There is no decorator or registry, and no `SimulationEvaluator` or
`AnalysisEvaluator`: those are scientific domains, not different calling conventions. `name`
and `version` identify the evaluator's measurement contract, as the `name` and `version` of an
entry of `evaluators` do in the problem file; they are labels, not a lookup mechanism. The
*documentation* of an evaluator version defines what each of its metrics means: selection,
estimator, units, uncertainty convention.

What the optional parts do:

- `stage = "simulation"`: the framework makes that stage available as the problem's `stages` say
  (runs it, or takes its file) and passes its file as `request.inputs["simulation"]`, along with
  the files of the stages before it; the evaluator never builds a command, starts a tool or
  handles its failures, and a stage that several evaluators read runs once (see
  [Stages](#stages)). The evaluator's entry in the problem states the same `stage`, so the
  problem file shows what each evaluator reads; a mismatch is an error before anything runs.
  Without `stage` the evaluator reads no stage and needs none: a cost model, say, works from the
  candidate's parameters alone.
- `collections`: the EDM4hep collections it reads from the file of its stage, by role, each a
  `Collection` with the type of its elements (`"SimCalorimeterHit"`, the `edm4hep::` prefix is
  implied). The problem's entry names the collection of each role (`hits: ECalBarrelCollection`);
  a role with a `default` (`Collection("MCParticle", default="MCParticles")`) needs no name,
  though the problem may give another. A role is a Python identifier. `self.events(request)`
  yields each event of the file with its collections as attributes (`event.hits`), all read from
  that same event, so they belong together (see [Reading EDM4hep files](#reading-edm4hep-files)).
  An unknown or missing role is an `EvaluationContractError` before anything runs; how a missing
  collection is reported is described under [Collections](#collections).
- `settings`: a mapping of key to a type (`str`, `int`, `float`, `bool`, `list`, `dict`) for a
  required setting, to a default value for an optional one (`"min_cells": 1` is an optional
  `int`), or to a `Quantity(unit)` for a number with a unit. The problem writes a quantity as
  `value` and `unit`, in any unit of the same dimension (energy: `eV`, `keV`, `MeV`, `GeV`, `TeV`;
  length: `um`, `mm`, `cm`, `m`), and the evaluator receives a float in the declared unit,
  converted exactly by a power of ten (`70` `keV` is the float `70e-6` in `GeV`). A bare number
  is refused, so the unit is always written; `Quantity("GeV", default=70e-6)` is optional, its
  default in the declared unit. Unknown keys, missing required keys, wrong types, a blank
  required `str` and a quantity without a unit of its dimension are an
  `EvaluationContractError`, raised before anything is created or run (a `bool` is not an `int`,
  an `int` is a `float`). The request an evaluator gets carries the defaults and the converted
  quantities, so `request.settings` (and the record) show the settings that were used, next to
  those the problem wrote; a `dict` arrives as a read-only mapping and a `list` as a tuple.
  Override `check(request)` for rules beyond that and call `super().check(request)`; plain
  evaluators can define `check` alone. Settings and collection resolution always run, even
  without `check`. The check runs once per candidate, with a request made for that candidate,
  before materialization, and that request is reused during its execution. `check` is for the
  settings and the contract: what it raises stops the evaluation before anything is created,
  as a misuse. A candidate the evaluator finds scientifically unusable is not refused there:
  `evaluate()` says so with `self.invalid("...")`, and the candidate is recorded as `invalid`.
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
  A metric is written `"name"` (classified under `stage`) or as a pair `(stage, name)`. Return an
  `EvaluationResult` for artifacts, provenance or diagnostics. An empty list is an
  `EvaluationContractError`, since it may be a bug: an evaluator that found nothing to measure
  says so with `self.invalid("...")`, which is a scientific verdict.

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
| `settings` | the evaluator-defined `settings` of its entry in `evaluators`, with the declared defaults and each quantity as a float in its declared unit; nested data, copied and read-only |
| `collections` | role to the name of the collection it reads: the one the problem's entry names, or the declared default |
| `inputs` | role name to local `Path` of an already prepared artifact: the main output of each stage it reads, by stage (`inputs["simulation"]`, an EDM4hep file), and the compact XML snapshot as `inputs["geometry"]`. The evaluator documents the roles it accepts |
| `workdir` | where the evaluator may write. `k4opt evaluate` gives each evaluator its own, `<output>/evaluators/<name>`, created for the call; the request never creates or checks it |
| `events`, `seed` | those of the stage the evaluator reads, where that stage has them (a ddsim simulation); `None` otherwise. `events` is positive |

The request holds no `ProblemSpec`, geometry or stage object; the framework builds it from
those.

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

`self.events(request)` is `read_events` on the file of the stage in `stage`, with the declared
`collections` under the names the request gives them. `read_events(path, {role: (collection,
value_type)})` yields an `Event` per event (podio frame) of an EDM4hep file, in file order, whose
attributes are the roles: each the list of that collection's elements in this event. All the
collections of an `Event` come from the same frame, so they belong to the same physics event.
Nothing is matched up across files or samples: `request.read(stage, collection, value_type)`
reads one collection of any stage's file, and two collections read that way are not known to
be aligned event by event. `read_collection(path, collection, value_type)` is the one-collection
form, `available_collections(path)` lists the collections of a file without reading any,
`collection_types(path, names)` gives the element types of the named ones (podio unpacks a
collection when it is read, so only those are), and `count_events(path)` counts the events.

`podio` comes from the Key4hep stack and is imported only when a file is opened, so
`k4opt.evaluation` imports without it. A file that cannot be opened, a missing `events` category
or collection, an element type other than `value_type` (`"SimCalorimeterHit"` means
`"edm4hep::SimCalorimeterHit"`) and every exception of the C++ reader are an `EvaluationFailed`:
a collection is never silently read as empty. An existing but empty collection is yielded like
any other.

Violations raise `EvaluationContractError` (a `K4OptError`) when the object is built.

### Collections

What an evaluator declares in `collections` is checked in three ways, and only what is
demonstrably wrong blocks a run:

| Case | When | Outcome |
| --- | --- | --- |
| A declaration that cannot be read: a role that is no Python identifier, a type that is no type name, collections without a `stage`; a role the problem's entry names but the evaluator does not declare, or one without a name; collections from a stage without an EDM4hep file: a generation run by the gun, which ddsim fires, or given (or supplied) as a file podio cannot read, such as HepMC; podio reads `.root` and `.sio` | before anything runs; no Key4hep stack needed | `EvaluationContractError` |
| A name ddsim is not expected to write | before anything runs | a warning with the closest expected name ("did you mean 'ECalBarrelCollection'?"); the run goes on |
| A collection missing from the stage's actual file, or holding another type | after the stage, before the evaluator is called | that evaluator is `failed`, with the collections the file holds; it is not called. Only the collections it declares are read, so one it does not read (of a datamodel without a dictionary, say) cannot fail it |

The expectation comes from the unmodified geometry: one collection per `<readout>` (and
`<hits_collection>`) named in its XML, a `<readout>Contributions` collection next to each, and
`MCParticles` and `EventHeader`. For CLD_o2_v09 it is exactly what ddsim writes. It is not a
guarantee: the type of a sensitive detector is set in C++ detector code, and DDG4 actions may
define collections of their own, so it only warns. The file is authoritative. `invalid` stays a
scientific verdict that only the evaluator gives: an evaluator that treats an absent collection
as a scientific outcome does not declare it, reads it with `request.read` and calls
`self.invalid`.

## Stage classifies, it does not prescribe

`stage` is a free descriptive string next to `metric`, as in the problem file. It is not a
workflow instruction, and the set of stages is not enumerated. One evaluator may report
`simulation` and `analysis` metrics together.

## Stages

The stages of the pipeline are made available by the framework, not by an evaluator. The
problem's `stages` say how: each is run by an `executor` or given as a `file` (see
[problem](problem.md#stages)). A stage turns the stages before it (and the candidate's geometry)
into a file, and an evaluator whose `stage` names it reads that file from `request.inputs`. Each
stage runs at most once per candidate, however many evaluators read it, so several evaluators
share one expensive simulation, and an evaluator can be tested with a small fixture file and no
Key4hep stack.

| Stage | Reads | Executor | Given as `file:` | Provides |
| --- | --- | --- | --- | --- |
| `generation` | | `gun`: a particle gun, fired by `ddsim`, so no file of its own | an event file (HepMC, ...), checked and hashed | `inputs["generation"]` for a file; nothing for a gun |
| `simulation` | `generation`, the candidate geometry | `ddsim`, with the stage's `events`, `seed` and ddsim options | not with geometry parameters: that file was not simulated with the candidate's geometry | `inputs["simulation"]`, an EDM4hep file |

An evaluator names only the last stage it reads; the stages that stage reads are made available
first, and it gets the files of all of them (an evaluator that reads a later reconstruction stage
will read the simulation file too). It does not get the files of stages that only other
evaluators need, so its request does not depend on which other evaluators the problem has.
Reconstruction and analysis will follow the same pattern. Pipeline stages are steps; the `stage`
of a metric only classifies it, so an evaluator may report `analysis` metrics without an analysis
stage. A stage has named outputs; `inputs[stage]` is its main one (`events`), and its other files
(logs, the command) are its artifacts.

A stage asks only for what it uses: `simulation` needs `inputs["geometry"]` and a `workdir`,
`generation` neither, so an evaluator that reads only `generation` runs without a geometry, and
one that reads none runs no stage at all. The stages a candidate needs form a graph, each with
the stages it reads, and run in that order. A stage that fails (a file it cannot read, the tool
missing, a nonzero exit, no output, an output without the requested events) makes every evaluator
that reads it `failed` with the stage's message, without calling them, and the stages that read it
do not run ("not run because the 'generation' stage failed"). Stages and evaluators that do not
depend on it still run and keep their results.

An `inputs[stage]` supplied by the caller (`k4opt evaluate --input simulation=FILE`) replaces
running that stage and the ones only it needs, and it is authoritative: a supplied `generation`
file is simulated even if the problem describes a particle gun. The record identifies it (the
path, size and SHA-256 under `provided`; a file that could not be hashed carries only its path,
so a verified input is distinguishable from one identified by its name) and lists it as
`supplied`. A supplied stage that the parameters change (a simulation, when they vary the
geometry) was not made from the candidate: the candidate record says so,
`represents_candidate = False`, and must never count as an evaluation of that candidate,
whatever its status. A file the problem gives (`file:`) or the caller supplies is hashed once
per file version, not once per candidate: a file is told apart by size, mtime, ctime and inode,
so one rewritten within a few milliseconds of being hashed is not, and files are assumed stable
while a campaign runs.

The stages are recorded once per candidate, apart from the evaluators' results: what ran (the
command and exit code), what was given or supplied, the files a ddsim option names (with their
hashes), the events found, the error if any, the stage's files and the time spent on it. An
evaluator's result holds only what it returned, so its artifact and provenance names cannot
clash with the stages'. The stages take their conditions from the problem's `stages`, never from
an evaluator. Stages are internal for now (`k4opt.problem._stages`).

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
registry; the `name` of each entry of `evaluators` stays a label that must equal the `name` of
an evaluator that is loaded.

`evaluator_source(evaluator)` reports the path, size and SHA-256 of the file that defines the
evaluator's class, taken when `load_evaluator` imports it, and whether those are the bytes
Python `executed`. For `FILE.py:Class` they are (`executed: true`): the bytes hashed are the
bytes compiled. For `package.module:Class` the file is read right after the import
(`executed: false`); Python may have run a cached `.pyc` of it, which only an edit that keeps
size and timestamp leaves stale. Cached instances keep that identity if the file is later
edited or removed. A
class whose module was imported before `load_evaluator`, or reloaded since
(`importlib.reload`), carries only its path: hashing
the current file would not establish which code Python executed. Only the defining file is
identified, not imported helpers. To label those, `revision` holds the commit and dirty flag of
the git checkout that tracks the file, taken at the same time: an edited helper module of that
checkout shows as `dirty: true`. It is `null` for a file that no checkout tracks (outside one,
or new or ignored in it), and helpers outside a checkout are not tracked at all.

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
the hashes of given files, the events found). `k4opt evaluate` also records the k4opt version
with the SHA-256 of k4opt's own Python files and its git checkout, the active stack
(`KEY4HEP_STACK`, `K4GEO`, the `ddsim` that ran) and each evaluator's source identity as
described above. That identifies the evaluator's own code, not the modules it imports;
`name` and `version` remain labels. The problem's `environment.key4hep` is a declared pin and is
not compared with the stack that ran.

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
`CandidateRecord`. It is private and meant for tests; the public entry point is
[`EvaluationSession`](#evaluating-candidates-evaluationsession), which takes the problem file.

```python
candidate = _evaluate_candidate(spec, {"constants.ECal_cell_size": 6.0}, [CellCount()])
```

In order, it:

1. validates the candidate (`CandidateError` otherwise): exactly one value per declared
   parameter, keyed by `parameter.geometry.canonical_target`; continuous `int`/finite `float`
   within inclusive bounds; integer a real `int` (no `bool`, no rounding) within bounds;
   categorical one of the choices by kind (strings need a string, booleans a boolean, numbers
   compare numerically). Units are never interpreted or converted;
2. matches the evaluators to `evaluators` by `name`, in any order: each declared evaluator must
   be given exactly once, with the declared `version` and reading the declared `stage`, and no
   other (`EvaluationContractError` otherwise). Each entry becomes one evaluator instance,
   identified by its name; records, run directories and metric owners use that identity;
3. before anything runs, checks the stages the evaluators read, resolves each evaluator's
   settings and collections, calls its `check()` and decides which evaluator owns each
   objective and constraint metric (see [problem](problem.md#evaluators);
   `EvaluationContractError` otherwise). A collection ddsim is not expected to write is a
   warning, kept with the candidate;
4. makes every stage the evaluators read available once, unless `inputs` supplies its file:
   it runs it on `inputs["geometry"]` in `workdir`, or takes the problem's `file`. The output of
   the `simulation` stage, run or supplied, must hold exactly its `events` events (at least one
   when the problem gives the file), otherwise the stage fails ("incomplete simulation"); the
   number found is recorded as `events_read`;
5. calls each evaluator's `evaluate()` once, in problem order, with a request built from the
   validated candidate, its own `settings` (defaults filled in, quantities converted) and
   `collections`, the `events` and `seed` of the stage it reads and the files of the stages it
   reads, in its own `workdir/evaluators/<name>`. An evaluator whose stage failed gets that
   stage's message as a `failed` result instead, and so does one whose stage's file lacks a
   collection it declares (see [Collections](#collections)). `EvaluationFailed` and `EvaluationInvalid` become a `failed` or
   `invalid` result; other exceptions of an evaluator propagate;
6. checks each result; a violation is an `EvaluationContractError`, an inconsistent evaluator
   rather than a violated constraint. A result must not report a metric another evaluator
   declares; a `valid` result must report every metric its evaluator owns (extra metrics are
   fine) in the `unit` a constraint declares; and if the evaluator declares `metrics`, every
   measurement of one, whatever the status, must carry the declared unit. Whether a value
   satisfies `min`/`max` is not judged here but by the session's
   [assessment](#assessment-objectives-limits-and-feasibility).

The `CandidateRecord` holds one `EvaluationRecord` per evaluator in problem order, the stages
(each recorded once), which evaluator owns each objective and constraint metric, and the time of
the whole evaluation. Like the records it holds, it cannot be changed once made. Its `status` is
the candidate's:

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

## Evaluating candidates: `EvaluationSession`

`k4opt.problem.EvaluationSession` evaluates candidates of one problem file, one after another,
each into a run directory of its own with a `summary.json`, exactly as `k4opt evaluate` does
for one ([evaluators](evaluators.md#k4opt-evaluate)). It knows no optimizer: a scan by hand, a
script and an optimization library call it the same way, with candidates given as values.

```python
from k4opt.evaluation import load_evaluator
from k4opt.problem import EvaluationSession

session = EvaluationSession.open(
    "examples/problem/cld-ecal-granularity.yaml",
    [load_evaluator("examples/evaluators/calo_photon.py:CaloPhoton")],
)
for size in (5.0, 7.0, 10.0):
    outcome = session.evaluate({"constants.ECal_cell_size": size}, f"runs/scan/{size}mm")
    print(size, outcome.candidate.status, outcome.assessment.feasible)
```

| Once, when the session opens | For every candidate, in `check()` and `evaluate()` |
| --- | --- |
| load the problem file (later edits to it do not reach the session) and discover the geometry | check the candidate against the parameter domains |
| bind the evaluators, check the stages they read, decide who measures each metric | build each evaluator's request for this candidate and call its `check()` |
| collect the warnings about collections ddsim is not expected to write (`session.warnings`) | check that every file the problem names is still the one it was when the session opened |
| record the size and SHA-256 of every file the stages read (a stage `file`, a file a ddsim option names, a file supplied with `inputs`), and each evaluator's source identity and declared metrics | `evaluate()` only: materialize the geometry, run the stages, call the evaluators, assess, write the summary |

No request outlives the candidate it was made for, so an evaluator's `check()` and `evaluate()`
always see that candidate's values. `check(candidate)` creates and runs nothing: it raises for
what `evaluate()` would refuse before creating anything. A file the stages read (named by the
problem or supplied) that changes during a session (other bytes under the same path, or removed)
stops the next candidate with a `RunError` before anything is created; `verify_inputs()` checks
it on its own. So does a change of `KEY4HEP_STACK` or `K4GEO`: the candidates of a session run
in the environment it opened in, which every summary of the session records.
`evaluate(candidate, output, run=...)` records `run`, any JSON-compatible mapping, as the
summary's `run` (a study and trial, say; `null` for `k4opt evaluate`). `identity()` returns what
the session evaluates with: the problem's path and definition, the identities of the files the
stages read, the evaluators' names, versions, sources and declared metrics, the compact file,
source root and fingerprint of the unmodified geometry, `KEY4HEP_STACK` and `K4GEO`, and k4opt's
version, code SHA-256 and checkout. The environment and k4opt's identity are taken when the
session opens, before anything runs, and every summary of the session records the same k4opt
identity, so an edit to the k4opt checkout during a long run is not recorded as the code that
ran (a hash cannot prove which code Python executed, only narrow the gap). k4opt's code is
hashed once per process, since Python runs the modules it imported; a session does not open once
k4opt's files have changed since then, because no hash would then say which code runs. A session
evaluates one candidate at a time and is not thread- safe.

### Assessment: objectives, limits and feasibility

Each outcome carries an `Assessment` of the candidate, also written to the summary under
`assessment`:

| Evaluation | `objectives` | `limits` | `feasible` |
| --- | --- | --- | --- |
| `valid`, no constraints | the measurement of each objective, in the problem's order | none | `true` |
| `valid`, every limit satisfied | as above | each with `value`, `margin`, `satisfied` | `true` |
| `valid`, a limit violated | as above | as above | `false` |
| `invalid` or `failed` | `null`, never a partial vector | each listed, `value`, `margin` and `satisfied` `null` | `null`: not assessed |
| a supplied stage not made from the candidate (`--input`) | as measured | as measured | `null` |

A limit is one bound of a constraint (`simulation/ecal_barrel.angular_residual:max`), judged on
the central value of the measurement: `max` is satisfied when the value is at most the limit,
`min` when it is at least the limit, and equality satisfies both. The uncertainty is recorded
next to the value but does not decide. `margin` is `value - max` or `min - value` in the
metric's unit, zero or below when the limit is satisfied; it is computed exactly and rounded
once, so its sign is always that of the comparison, and it is `null` if it does not fit in a
float (the verdict stays exact). An infeasible candidate is a complete evaluation with real
objective values; it is only outside the limits.

`Assessment.from_summary(spec, summary)` recomputes the assessment of a `summary.json` as read
back: the candidate's status from its evaluators', each objective and constraint value from the
own measurements of the evaluator that `metrics` names as its owner. It checks again what a live
evaluation enforces: the candidate is in the problem's domains and `candidate_key` is its key,
an `invalid` or `failed` evaluator gives a message, each measurement of a metric its evaluator
declares carries the declared unit, each of several evaluators declares its metrics, no metric
is declared by two of them or reported by one that does not declare it, an owner declares the
metrics it owns, and a constraint that declares a unit is in the unit of its metric. It raises
`RecordError` if the record is malformed or breaks one of those rules, was made for another
problem (its definition differs, other than in `name`), or if its `status`, `message`, `metrics`
or `assessment` disagree with that recomputation, so any reader can check a record before
trusting it.

## Deliberately deferred

An evaluator registry or lookup by name (evaluators are named on the command line), several
samples per problem and evaluators that read more than one, branches of the stages,
validating metric names, `k4run`, chaining artifacts between evaluators (callers do it by hand,
as the tests show), execution cost, retries and scheduling, fingerprints and caching,
persisting or serializing the `EvaluationRecord` apart from `summary.json`, and aggregation,
significance or constraints judged with their uncertainty.
